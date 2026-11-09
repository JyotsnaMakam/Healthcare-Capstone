"""Compute the Part 2 no-show classification results for the capstone."""

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUT_DIR = ROOT / "outputs"
RESULTS_PATH = OUTPUT_DIR / "part2_results.txt"
THRESHOLDS_PATH = OUTPUT_DIR / "part2_thresholds.csv"
KNN_CV_PATH = OUTPUT_DIR / "part2_knn_cv.csv"

FEATURE_COLUMNS = (
    "months_registered",
    "missed_visit_count_12m",
    "copay_burden_ratio",
    "avg_bill_inr_thousands",
    "adherence_score",
    "chronic_condition_flag",
    "insured_flag",
)
TARGET_COLUMN = "no_show"
SEED = 42
TRAIN_FRACTION = 0.70
LOGISTIC_UPDATES = 1000
LOGISTIC_LEARNING_RATE = 0.1
FALSE_ALARM_COST_INR = 50.0
MISSED_NO_SHOW_COST_INR = 400.0


def load_records(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the model predictors and binary target."""
    required_columns = (*FEATURE_COLUMNS, TARGET_COLUMN)
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")

        missing_columns = [name for name in required_columns if name not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing_columns)}"
            )

        feature_rows: list[list[float]] = []
        targets: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                feature_rows.append([float(row[name]) for name in FEATURE_COLUMNS])
                targets.append(float(row[TARGET_COLUMN]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in a required column, row {row_number}."
                ) from error

    if not targets:
        raise ValueError(f"{path} contains no data rows.")

    x = np.asarray(feature_rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{path} contains non-finite numeric values.")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1.")
    if x.shape[0] != 800:
        raise ValueError(f"Expected 800 records for a 560/240 split; found {x.shape[0]}.")
    return x, y


def stratified_split(
    y: np.ndarray, train_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a shuffled, stratified split with the requested total test size."""
    rng = np.random.default_rng(seed)
    test_size = int(round(y.size * (1.0 - train_fraction)))
    classes, class_counts = np.unique(y, return_counts=True)
    exact_test_counts = class_counts * (test_size / y.size)
    test_counts = np.floor(exact_test_counts).astype(int)
    remainder = test_size - int(np.sum(test_counts))
    if remainder:
        order = np.argsort(-(exact_test_counts - test_counts), kind="stable")
        test_counts[order[:remainder]] += 1

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label, test_count in zip(classes, test_counts):
        class_indices = rng.permutation(np.flatnonzero(y == label))
        test_parts.append(class_indices[:test_count])
        train_parts.append(class_indices[test_count:])

    train_indices = rng.permutation(np.concatenate(train_parts))
    test_indices = rng.permutation(np.concatenate(test_parts))
    return train_indices, test_indices


def stratified_folds(y: np.ndarray, fold_count: int, seed: int) -> list[np.ndarray]:
    """Partition row indices into folds while preserving each class's proportions."""
    rng = np.random.default_rng(seed)
    folds: list[list[np.ndarray]] = [[] for _ in range(fold_count)]
    for label in np.unique(y):
        class_indices = rng.permutation(np.flatnonzero(y == label))
        for fold_index, part in enumerate(np.array_split(class_indices, fold_count)):
            folds[fold_index].append(part)
    return [rng.permutation(np.concatenate(parts)) for parts in folds]


def standardize(
    x_train: np.ndarray, x_other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale features using only the supplied training partition's statistics."""
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales[scales == 0.0] = 1.0
    return (x_train - means) / scales, (x_other - means) / scales, means, scales


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable logistic sigmoid."""
    clipped_logits = np.clip(logits, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped_logits))


def binary_cross_entropy(y: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute mean binary cross-entropy without taking log of probabilities."""
    clipped_probabilities = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    return float(
        -np.mean(
            y * np.log(clipped_probabilities)
            + (1.0 - y) * np.log(1.0 - clipped_probabilities)
        )
    )


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    updates: int,
    checkpoints: tuple[int, ...] = (),
) -> tuple[np.ndarray, float, dict[int, float]]:
    """Fit batch logistic regression with gradient descent from zero parameters."""
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    recorded_losses: dict[int, float] = {}
    if 0 in checkpoints:
        recorded_losses[0] = binary_cross_entropy(y, sigmoid(x @ weights + bias))

    for iteration in range(1, updates + 1):
        probabilities = sigmoid(x @ weights + bias)
        errors = probabilities - y
        weights -= learning_rate * (x.T @ errors) / y.size
        bias -= learning_rate * float(np.mean(errors))
        if iteration in checkpoints:
            recorded_losses[iteration] = binary_cross_entropy(
                y, sigmoid(x @ weights + bias)
            )

    return weights, bias, recorded_losses


def classification_metrics(
    y: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Calculate confusion-matrix counts and common binary classification metrics."""
    predictions = probabilities >= threshold
    actual = y == 1.0
    true_positive = int(np.sum(predictions & actual))
    true_negative = int(np.sum(~predictions & ~actual))
    false_positive = int(np.sum(predictions & ~actual))
    false_negative = int(np.sum(~predictions & actual))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    specificity = ratio(true_negative, true_negative + false_positive)
    f1 = ratio(2 * precision * recall, precision + recall)
    accuracy = ratio(true_positive + true_negative, y.size)
    cost = (
        FALSE_ALARM_COST_INR * false_positive
        + MISSED_NO_SHOW_COST_INR * false_negative
    )
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "financial_cost_inr": float(cost),
    }


def write_threshold_sweep(
    path: Path, y: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, float | int]]:
    """Write held-out classification metrics for thresholds from 0 to 1."""
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = [classification_metrics(y, probabilities, float(tau)) for tau in thresholds]
    columns = (
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "financial_cost_inr",
    )
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def knn_cross_validation(
    x: np.ndarray, y: np.ndarray, fold_count: int, seed: int
) -> list[dict[str, float | int]]:
    """Evaluate K=1..21 using stratified folds and fold-local feature scaling."""
    fold_indices = stratified_folds(y, fold_count, seed)
    fold_scores: dict[int, list[float]] = {k: [] for k in range(1, 22)}
    all_indices = np.arange(y.size)

    for validation_indices in fold_indices:
        is_validation = np.zeros(y.size, dtype=bool)
        is_validation[validation_indices] = True
        training_indices = all_indices[~is_validation]
        x_train, x_validation, _, _ = standardize(
            x[training_indices], x[validation_indices]
        )
        y_train = y[training_indices]
        y_validation = y[validation_indices]

        squared_distances = (
            np.sum(x_validation**2, axis=1, keepdims=True)
            + np.sum(x_train**2, axis=1)[None, :]
            - 2.0 * (x_validation @ x_train.T)
        )
        nearest = np.argsort(np.maximum(squared_distances, 0.0), axis=1, kind="stable")
        nearest_labels = y_train[nearest[:, :21]]
        cumulative_positive_votes = np.cumsum(nearest_labels, axis=1)
        for k in range(1, 22):
            predictions = cumulative_positive_votes[:, k - 1] * 2 > k
            fold_scores[k].append(float(np.mean(predictions == (y_validation == 1.0))))

    results: list[dict[str, float | int]] = []
    for k, scores in fold_scores.items():
        row: dict[str, float | int] = {
            "k": k,
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
        }
        row.update({f"fold_{index + 1}_accuracy": score for index, score in enumerate(scores)})
        results.append(row)
    return results


def write_knn_results(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write KNN cross-validation scores for every tested neighborhood size."""
    columns = (
        "k",
        "mean_accuracy",
        "std_accuracy",
        "fold_1_accuracy",
        "fold_2_accuracy",
        "fold_3_accuracy",
        "fold_4_accuracy",
        "fold_5_accuracy",
    )
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    x, y = load_records(DATA_PATH)
    train_indices, test_indices = stratified_split(y, TRAIN_FRACTION, SEED)
    x_train, x_test, feature_means, feature_scales = standardize(
        x[train_indices], x[test_indices]
    )
    y_train = y[train_indices]
    y_test = y[test_indices]

    checkpoints = (0, 1, 10, 100, LOGISTIC_UPDATES)
    weights, bias, training_losses = fit_logistic_regression(
        x_train,
        y_train,
        LOGISTIC_LEARNING_RATE,
        LOGISTIC_UPDATES,
        checkpoints,
    )
    test_probabilities = sigmoid(x_test @ weights + bias)
    test_bce = binary_cross_entropy(y_test, test_probabilities)

    learning_rate_losses: dict[float, float] = {}
    for learning_rate in (0.01, 0.1, 1.0):
        _, _, losses = fit_logistic_regression(
            x_train, y_train, learning_rate, 200, (200,)
        )
        learning_rate_losses[learning_rate] = losses[200]

    odds_ratios = np.exp(np.clip(weights, -700.0, 700.0))
    metrics_at_05 = classification_metrics(y_test, test_probabilities, 0.5)
    metrics_at_03 = classification_metrics(y_test, test_probabilities, 0.3)
    majority_label = float(np.mean(y_train) >= 0.5)
    majority_baseline_accuracy = float(np.mean(y_test == majority_label))

    threshold_rows = write_threshold_sweep(
        THRESHOLDS_PATH, y_test, test_probabilities
    )
    decision_threshold = FALSE_ALARM_COST_INR / (
        FALSE_ALARM_COST_INR + MISSED_NO_SHOW_COST_INR
    )
    decision_metrics = classification_metrics(
        y_test, test_probabilities, decision_threshold
    )

    # Include every distinct test probability so every possible test-set decision
    # pattern is considered, in addition to the readable 0.01 threshold sweep.
    observed_thresholds = np.unique(
        np.concatenate(([0.0], test_probabilities, [1.0]))
    )
    observed_cost_rows = [
        classification_metrics(y_test, test_probabilities, float(tau))
        for tau in observed_thresholds
    ]
    minimum_test_cost = min(float(row["financial_cost_inr"]) for row in observed_cost_rows)
    empirical_optima = [
        row
        for row in observed_cost_rows
        if float(row["financial_cost_inr"]) == minimum_test_cost
    ]
    empirical_optimum = max(empirical_optima, key=lambda row: float(row["threshold"]))

    knn_rows = knn_cross_validation(x_train, y_train, fold_count=5, seed=SEED + 1)
    write_knn_results(KNN_CV_PATH, knn_rows)
    best_knn = max(knn_rows, key=lambda row: float(row["mean_accuracy"]))

    results: dict[str, int | float | str] = {
        "records": int(y.size),
        "train_size": int(train_indices.size),
        "test_size": int(test_indices.size),
        "split_seed": SEED,
        "positive_train": int(np.sum(y_train)),
        "positive_test": int(np.sum(y_test)),
        "features": ", ".join(FEATURE_COLUMNS),
        "train_feature_means": np.array2string(feature_means, precision=6),
        "train_feature_scales": np.array2string(feature_scales, precision=6),
        "logistic_updates": LOGISTIC_UPDATES,
        "logistic_learning_rate": LOGISTIC_LEARNING_RATE,
        "training_bce_initial": training_losses[0],
        "training_bce_after_1_update": training_losses[1],
        "training_bce_after_10_updates": training_losses[10],
        "training_bce_after_100_updates": training_losses[100],
        "training_bce_final": training_losses[LOGISTIC_UPDATES],
        "logistic_weights": np.array2string(weights, precision=6),
        "logistic_bias": float(bias),
        "test_bce": test_bce,
        "learning_rate_0.01_bce_after_200": learning_rate_losses[0.01],
        "learning_rate_0.1_bce_after_200": learning_rate_losses[0.1],
        "learning_rate_1.0_bce_after_200": learning_rate_losses[1.0],
        "odds_ratios": np.array2string(odds_ratios, precision=6),
        "threshold_0.5_accuracy": metrics_at_05["accuracy"],
        "threshold_0.5_precision": metrics_at_05["precision"],
        "threshold_0.5_recall": metrics_at_05["recall"],
        "threshold_0.5_specificity": metrics_at_05["specificity"],
        "threshold_0.5_f1": metrics_at_05["f1"],
        "threshold_0.5_true_positives": metrics_at_05["true_positive"],
        "threshold_0.5_true_negatives": metrics_at_05["true_negative"],
        "threshold_0.5_false_positives": metrics_at_05["false_positive"],
        "threshold_0.5_false_negatives": metrics_at_05["false_negative"],
        "threshold_0.5_financial_cost_inr": metrics_at_05["financial_cost_inr"],
        "blind_test_accuracy_at_0.5": metrics_at_05["accuracy"],
        "majority_class_blind_baseline_accuracy": majority_baseline_accuracy,
        "threshold_0.3_accuracy": metrics_at_03["accuracy"],
        "threshold_0.3_precision": metrics_at_03["precision"],
        "threshold_0.3_recall": metrics_at_03["recall"],
        "threshold_0.3_specificity": metrics_at_03["specificity"],
        "threshold_0.3_f1": metrics_at_03["f1"],
        "threshold_0.3_true_positives": metrics_at_03["true_positive"],
        "threshold_0.3_true_negatives": metrics_at_03["true_negative"],
        "threshold_0.3_false_positives": metrics_at_03["false_positive"],
        "threshold_0.3_false_negatives": metrics_at_03["false_negative"],
        "threshold_0.3_financial_cost_inr": metrics_at_03["financial_cost_inr"],
        "false_alarm_cost_inr": FALSE_ALARM_COST_INR,
        "missed_no_show_cost_inr": MISSED_NO_SHOW_COST_INR,
        "expected_cost_decision_threshold": decision_threshold,
        "test_cost_at_expected_cost_threshold_inr": decision_metrics[
            "financial_cost_inr"
        ],
        "empirical_test_cost_minimizing_threshold": empirical_optimum["threshold"],
        "empirical_minimum_test_cost_inr": minimum_test_cost,
        "empirical_optimum_test_accuracy": empirical_optimum["accuracy"],
        "empirical_optimum_test_precision": empirical_optimum["precision"],
        "empirical_optimum_test_recall": empirical_optimum["recall"],
        "empirical_optimum_test_true_positives": empirical_optimum["true_positive"],
        "empirical_optimum_test_true_negatives": empirical_optimum["true_negative"],
        "empirical_optimum_test_false_positives": empirical_optimum["false_positive"],
        "empirical_optimum_test_false_negatives": empirical_optimum["false_negative"],
        "knn_cv_fold_count": 5,
        "knn_cv_best_k": int(best_knn["k"]),
        "knn_cv_best_mean_accuracy": float(best_knn["mean_accuracy"]),
        "thresholds_csv": str(THRESHOLDS_PATH.relative_to(ROOT)),
        "knn_cv_csv": str(KNN_CV_PATH.relative_to(ROOT)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        for key, result in results.items():
            if isinstance(result, (int, np.integer)):
                formatted = str(int(result))
            elif isinstance(result, (float, np.floating)):
                formatted = f"{result:.6f}"
            else:
                formatted = result
            output_file.write(f"{key} = {formatted}\n")

    for key, result in results.items():
        print(f"{key} = {result}")
    print(f"Results saved to {RESULTS_PATH}")
    print(f"Threshold sweep saved to {THRESHOLDS_PATH}")
    print(f"KNN CV results saved to {KNN_CV_PATH}")


if __name__ == "__main__":
    main()
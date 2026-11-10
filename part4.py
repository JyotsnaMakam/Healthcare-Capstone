"""Compute Part 4 decision-model and neural-network sizing results."""

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUT_DIR = ROOT / "outputs"
RESULTS_PATH = OUTPUT_DIR / "part4_results.txt"
MEMO_PATH = OUTPUT_DIR / "executive_memo.txt"

FEATURE_COLUMNS = (
    "months_registered",
    "missed_visit_count_12m",
    "copay_burden_ratio",
    "avg_bill_inr_thousands",
    "adherence_score",
    "chronic_condition_flag",
    "insured_flag",
)
BINARY_FEATURES = ("chronic_condition_flag", "insured_flag")
TARGET_COLUMN = "no_show"
SEED = 42
FALSE_ALARM_COST_INR = 50
MISSED_NO_SHOW_COST_INR = 400
BOOSTING_ROUNDS = 3


def load_records(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate predictors, binary predictors, and no-show labels."""
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

    x = np.asarray(feature_rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    if not y.size:
        raise ValueError(f"{path} contains no data rows.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{path} contains non-finite numeric values.")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1.")
    if not np.all(np.isin(x[:, [FEATURE_COLUMNS.index(name) for name in BINARY_FEATURES]], (0.0, 1.0))):
        raise ValueError("Binary predictors must contain only 0 and 1.")
    return x, x[:, [FEATURE_COLUMNS.index(name) for name in BINARY_FEATURES]], y


def stratified_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create the same reproducible 70/30 stratified split used in Part 2."""
    rng = np.random.default_rng(seed)
    test_size = int(round(y.size * 0.30))
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
    return (
        rng.permutation(np.concatenate(train_parts)),
        rng.permutation(np.concatenate(test_parts)),
    )


def entropy(labels: np.ndarray) -> float:
    """Calculate binary entropy in bits."""
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / labels.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def gini(labels: np.ndarray) -> float:
    """Calculate binary Gini impurity."""
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / labels.size
    return float(1.0 - np.sum(probabilities**2))


def tree_split_metrics(
    x_binary: np.ndarray, y: np.ndarray
) -> list[dict[str, float | str]]:
    """Measure entropy and Gini before and after each binary-feature split."""
    rows: list[dict[str, float | str]] = []
    parent_entropy = entropy(y)
    parent_gini = gini(y)
    for column, feature_name in enumerate(BINARY_FEATURES):
        weighted_entropy = 0.0
        weighted_gini = 0.0
        for value in (0.0, 1.0):
            child = y[x_binary[:, column] == value]
            weight = child.size / y.size
            weighted_entropy += weight * entropy(child)
            weighted_gini += weight * gini(child)
        rows.append(
            {
                "feature": feature_name,
                "parent_entropy": parent_entropy,
                "weighted_child_entropy": weighted_entropy,
                "information_gain": parent_entropy - weighted_entropy,
                "parent_gini": parent_gini,
                "weighted_child_gini": weighted_gini,
                "gini_gain": parent_gini - weighted_gini,
            }
        )
    return rows


def fit_binary_stump(
    x: np.ndarray, y: np.ndarray
) -> tuple[int, np.ndarray, float]:
    """Fit the best one-feature, two-leaf classifier with training majority leaves."""
    best_feature = -1
    best_predictions = np.zeros(y.size, dtype=float)
    best_error = float("inf")
    for feature_index in range(x.shape[1]):
        predictions = np.zeros(y.size, dtype=float)
        for value in (0.0, 1.0):
            mask = x[:, feature_index] == value
            if np.any(mask):
                positives = int(np.sum(y[mask] == 1.0))
                negatives = int(np.sum(y[mask] == 0.0))
                predictions[mask] = 1.0 if positives > negatives else 0.0
        error = float(np.mean(predictions != y))
        if error < best_error:
            best_feature = feature_index
            best_predictions = predictions
            best_error = error
    return best_feature, best_predictions, best_error


def majority_vote_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Train one stump per binary attribute and aggregate by majority vote."""
    predictions: list[np.ndarray] = []
    selected_features: list[int] = []
    training_errors: list[float] = []
    for feature_index in range(x_train.shape[1]):
        feature_train = x_train[:, [feature_index]]
        feature_test = x_test[:, [feature_index]]
        _, train_prediction, error = fit_binary_stump(feature_train, y_train)
        test_prediction = np.zeros(x_test.shape[0], dtype=float)
        for value in (0.0, 1.0):
            mask = feature_test[:, 0] == value
            if np.any(mask):
                leaf_train = feature_train[:, 0] == value
                positives = np.sum(y_train[leaf_train] == 1.0)
                negatives = np.sum(y_train[leaf_train] == 0.0)
                test_prediction[mask] = 1.0 if positives > negatives else 0.0
        predictions.append(test_prediction)
        selected_features.append(feature_index)
        training_errors.append(error)

    vote_matrix = np.column_stack(predictions)
    positive_votes = np.sum(vote_matrix, axis=1)
    prior_positive = float(np.mean(y_train) >= 0.5)
    ensemble_prediction = (positive_votes > vote_matrix.shape[1] / 2.0).astype(float)
    ties = positive_votes == vote_matrix.shape[1] / 2.0
    ensemble_prediction[ties] = prior_positive
    return ensemble_prediction, selected_features, np.asarray(training_errors)


def candidate_stump_predictions(
    x_train: np.ndarray, x_test: np.ndarray
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Enumerate all mappings from a binary feature's two values to binary classes."""
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for feature_index in range(x_train.shape[1]):
        for prediction_at_zero in (0.0, 1.0):
            for prediction_at_one in (0.0, 1.0):
                train_predictions = np.where(
                    x_train[:, feature_index] == 0.0,
                    prediction_at_zero,
                    prediction_at_one,
                )
                test_predictions = np.where(
                    x_test[:, feature_index] == 0.0,
                    prediction_at_zero,
                    prediction_at_one,
                )
                candidates.append(
                    (feature_index, train_predictions, test_predictions)
                )
    return candidates


def adaboost(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Perform three manual AdaBoost rounds using binary decision stumps."""
    signed_y = np.where(y_train == 1.0, 1.0, -1.0)
    weights = np.full(y_train.size, 1.0 / y_train.size)
    candidates = candidate_stump_predictions(x_train, x_test)
    fitted_rounds: list[dict[str, float | int]] = []
    test_predictions: list[np.ndarray] = []
    epsilon = 1e-12

    for round_number in range(1, BOOSTING_ROUNDS + 1):
        best_error = float("inf")
        best_feature = -1
        best_prediction = np.zeros(y_train.size, dtype=float)
        best_test_prediction = np.zeros(x_test.shape[0], dtype=float)
        for feature_index, candidate, candidate_test in candidates:
            candidate_signed = np.where(candidate == 1.0, 1.0, -1.0)
            weighted_error = float(
                np.sum(weights[candidate_signed != signed_y])
            )
            if weighted_error < best_error:
                best_error = weighted_error
                best_feature = feature_index
                best_prediction = candidate
                best_test_prediction = candidate_test

        clipped_error = float(np.clip(best_error, epsilon, 1.0 - epsilon))
        alpha = 0.5 * float(np.log((1.0 - clipped_error) / clipped_error))
        best_signed = np.where(best_prediction == 1.0, 1.0, -1.0)
        weights *= np.exp(-alpha * signed_y * best_signed)
        weights /= np.sum(weights)
        fitted_rounds.append(
            {
                "round": round_number,
                "feature_index": best_feature,
                "weighted_error": best_error,
                "alpha": alpha,
            }
        )
        test_predictions.append(best_test_prediction)

    weighted_scores = np.zeros(x_test.shape[0], dtype=float)
    for fitted, prediction in zip(fitted_rounds, test_predictions):
        weighted_scores += float(fitted["alpha"]) * np.where(
            prediction == 1.0, 1.0, -1.0
        )
    final_prediction = (weighted_scores > 0.0).astype(float)
    ties = weighted_scores == 0.0
    final_prediction[ties] = float(np.mean(y_train) >= 0.5)
    return final_prediction, fitted_rounds


def classification_metrics(
    y_true: np.ndarray, y_predicted: np.ndarray
) -> dict[str, float | int]:
    """Return confusion counts, accuracy, sensitivity, and cost."""
    actual = y_true == 1.0
    predicted = y_predicted == 1.0
    tp = int(np.sum(actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    return {
        "accuracy": float((tp + tn) / y_true.size),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "financial_cost_inr": int(
            FALSE_ALARM_COST_INR * fp + MISSED_NO_SHOW_COST_INR * fn
        ),
    }


def logistic_test_probabilities(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    """Fit the Part 2 logistic baseline using training-only standardization."""
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales[scales == 0.0] = 1.0
    train_scaled = (x_train - means) / scales
    test_scaled = (x_test - means) / scales
    weights = np.zeros(x_train.shape[1], dtype=float)
    bias = 0.0
    for _ in range(1000):
        logits = np.clip(train_scaled @ weights + bias, -500.0, 500.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        errors = probabilities - y_train
        weights -= 0.1 * (train_scaled.T @ errors) / y_train.size
        bias -= 0.1 * float(np.mean(errors))
    logits = np.clip(test_scaled @ weights + bias, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-logits))


def neural_network_shapes() -> dict[str, int | str]:
    """Describe explicit CNN and LSTM architectures, shapes, and parameter counts."""
    cnn_shapes = {
        "cnn_input_shape": "28x28x1",
        "cnn_architecture": (
            "Conv2D(32,3x3,valid,relu)->MaxPool(2x2)->"
            "Conv2D(64,3x3,valid,relu)->MaxPool(2x2)->"
            "Flatten->Dense(128,relu)->Dense(10,softmax)"
        ),
        "cnn_conv1_kernel_shape": "3x3x1x32",
        "cnn_conv1_bias_shape": "32",
        "cnn_conv1_output_shape": "26x26x32",
        "cnn_pool1_output_shape": "13x13x32",
        "cnn_conv2_kernel_shape": "3x3x32x64",
        "cnn_conv2_bias_shape": "64",
        "cnn_conv2_output_shape": "11x11x64",
        "cnn_pool2_output_shape": "5x5x64",
        "cnn_flatten_shape": "1600",
        "cnn_dense_kernel_shape": "1600x128",
        "cnn_dense_bias_shape": "128",
        "cnn_output_kernel_shape": "128x10",
        "cnn_output_bias_shape": "10",
    }
    cnn_counts = {
        "cnn_conv1_parameters": 3 * 3 * 1 * 32 + 32,
        "cnn_conv2_parameters": 3 * 3 * 32 * 64 + 64,
        "cnn_dense_parameters": 1600 * 128 + 128,
        "cnn_output_parameters": 128 * 10 + 10,
    }
    cnn_counts["cnn_total_parameters"] = sum(cnn_counts.values())

    lstm_shapes = {
        "lstm_input_shape": "100 token IDs",
        "lstm_architecture": (
            "Embedding(vocab=10000,dim=128)->LSTM(64,one bias vector)->"
            "Dense(1,sigmoid)"
        ),
        "lstm_embedding_matrix_shape": "10000x128",
        "lstm_input_kernel_shape": "128x256 (4 gates)",
        "lstm_recurrent_kernel_shape": "64x256 (4 gates)",
        "lstm_bias_shape": "256 (one bias vector)",
        "lstm_output_kernel_shape": "64x1",
        "lstm_output_bias_shape": "1",
    }
    lstm_counts = {
        "lstm_embedding_parameters": 10000 * 128,
        "lstm_input_kernel_parameters": 128 * 256,
        "lstm_recurrent_kernel_parameters": 64 * 256,
        "lstm_bias_parameters": 256,
        "lstm_output_parameters": 64 * 1 + 1,
    }
    lstm_counts["lstm_total_parameters"] = sum(lstm_counts.values())
    return {**cnn_shapes, **cnn_counts, **lstm_shapes, **lstm_counts}


def build_memo(
    logistic_metrics: dict[str, float | int],
    ada_metrics: dict[str, float | int],
    ensemble_metrics: dict[str, float | int],
    cost_threshold_metrics: dict[str, float | int],
) -> str:
    """Build a concise, evidence-based executive memo."""
    paragraphs = [
        "To: Head of Scheduling Operations",
        "Subject: No-show outreach model and controlled pilot recommendation",
        "",
        (
            "I recommend piloting the transparent logistic-regression no-show "
            "model as a decision-support tool, with staff review before outreach. "
            "On the reproducible stratified 240-record holdout, it achieved "
            f"{float(logistic_metrics['accuracy']):.1%} accuracy at a 0.50 "
            f"cutoff. Given the stated INR 50 false-alarm cost and INR 400 "
            "missed-no-show cost, the cost-derived cutoff is 0.1111; at that "
            f"cutoff the holdout cost was INR {int(cost_threshold_metrics['financial_cost_inr']):,}. "
            f"For comparison, the logistic model cost INR "
            f"{int(logistic_metrics['financial_cost_inr']):,} at 0.50."
        ),
        (
            "The binary-feature majority-vote tree ensemble achieved "
            f"{float(ensemble_metrics['accuracy']):.1%} holdout accuracy and "
            f"INR {int(ensemble_metrics['financial_cost_inr']):,} cost. The "
            "three-round AdaBoost simulation achieved "
            f"{float(ada_metrics['accuracy']):.1%} accuracy and INR "
            f"{int(ada_metrics['financial_cost_inr']):,} cost. These simple "
            "ensembles are useful benchmarks, but do not establish superiority "
            "over the richer logistic model."
        ),
        (
            "These figures are offline estimates, not guaranteed savings. The "
            "dataset is synthetic and small, the holdout has only 51 no-shows, "
            "and threshold costs are sensitive to prevalence and the assumed "
            "INR cost ratio. Reusing the holdout to select a threshold would "
            "bias its reported performance. Binary-only trees also omit five "
            "continuous predictors; the neural-network architectures sized in "
            "this analysis are design calculations, not trained models or "
            "evidence that imaging or referral-note inputs are available."
        ),
        (
            "Before deployment, validate on a larger, time-separated sample; "
            "confirm false-alarm and missed-visit costs with operations; "
            "calibrate probabilities and choose a threshold using validation "
            "data; then run a monitored pilot. Track outreach volume, captured "
            "no-shows, staff workload, subgroup performance, and realized "
            "costs. Keep scheduling decisions reviewable by staff and refresh "
            "the model only after drift and data-quality checks. Use an "
            "operations-approved contact cap and escalation policy during the "
            "pilot, and compare contacted patients with a suitable control "
            "group where feasible. The estimated INR amounts are per 240-record "
            "holdout, not annual savings or a forecast of the full patient "
            "population."
        ),
    ]
    return "\n".join(paragraphs) + "\n"


def main() -> None:
    x, x_binary, y = load_records(DATA_PATH)
    train_indices, test_indices = stratified_split(y, SEED)
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    binary_train = x_binary[train_indices]
    binary_test = x_binary[test_indices]

    split_metrics = tree_split_metrics(binary_train, y_train)
    majority_predictions, ensemble_features, ensemble_training_errors = (
        majority_vote_ensemble(binary_train, y_train, binary_test)
    )
    ensemble_metrics = classification_metrics(y_test, majority_predictions)
    ada_predictions, ada_rounds = adaboost(binary_train, y_train, binary_test)
    ada_metrics = classification_metrics(y_test, ada_predictions)

    logistic_probabilities = logistic_test_probabilities(x_train, y_train, x_test)
    logistic_predictions = (logistic_probabilities >= 0.5).astype(float)
    logistic_metrics = classification_metrics(y_test, logistic_predictions)
    cost_threshold = FALSE_ALARM_COST_INR / (
        FALSE_ALARM_COST_INR + MISSED_NO_SHOW_COST_INR
    )
    cost_threshold_predictions = (
        logistic_probabilities >= cost_threshold
    ).astype(float)
    cost_threshold_metrics = classification_metrics(y_test, cost_threshold_predictions)

    nn_shapes = neural_network_shapes()
    memo = build_memo(
        logistic_metrics, ada_metrics, ensemble_metrics, cost_threshold_metrics
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MEMO_PATH.open("w", encoding="utf-8", newline="\n") as memo_file:
        memo_file.write(memo)

    results: dict[str, int | float | str] = {
        "records": int(y.size),
        "train_size": int(train_indices.size),
        "test_size": int(test_indices.size),
        "split_seed": SEED,
        "binary_tree_features": ", ".join(BINARY_FEATURES),
        "tree_parent_entropy": float(split_metrics[0]["parent_entropy"]),
        "tree_parent_gini": float(split_metrics[0]["parent_gini"]),
        "tree_chronic_condition_weighted_child_entropy": float(
            split_metrics[0]["weighted_child_entropy"]
        ),
        "tree_chronic_condition_information_gain": float(
            split_metrics[0]["information_gain"]
        ),
        "tree_chronic_condition_weighted_child_gini": float(
            split_metrics[0]["weighted_child_gini"]
        ),
        "tree_chronic_condition_gini_gain": float(
            split_metrics[0]["gini_gain"]
        ),
        "tree_insured_weighted_child_entropy": float(
            split_metrics[1]["weighted_child_entropy"]
        ),
        "tree_insured_information_gain": float(
            split_metrics[1]["information_gain"]
        ),
        "tree_insured_weighted_child_gini": float(
            split_metrics[1]["weighted_child_gini"]
        ),
        "tree_insured_gini_gain": float(split_metrics[1]["gini_gain"]),
        "majority_ensemble_classifiers": int(len(ensemble_features)),
        "majority_ensemble_features": ", ".join(
            BINARY_FEATURES[index] for index in ensemble_features
        ),
        "majority_ensemble_chronic_stump_train_error": float(
            ensemble_training_errors[0]
        ),
        "majority_ensemble_insured_stump_train_error": float(
            ensemble_training_errors[1]
        ),
        "majority_ensemble_test_accuracy": ensemble_metrics["accuracy"],
        "majority_ensemble_test_sensitivity": ensemble_metrics["sensitivity"],
        "majority_ensemble_test_specificity": ensemble_metrics["specificity"],
        "majority_ensemble_test_false_positives": ensemble_metrics["false_positive"],
        "majority_ensemble_test_false_negatives": ensemble_metrics["false_negative"],
        "majority_ensemble_test_cost_inr": ensemble_metrics["financial_cost_inr"],
        "adaboost_rounds": BOOSTING_ROUNDS,
        "adaboost_round_1_feature": BINARY_FEATURES[int(ada_rounds[0]["feature_index"])],
        "adaboost_round_1_weighted_error": float(ada_rounds[0]["weighted_error"]),
        "adaboost_round_1_alpha": float(ada_rounds[0]["alpha"]),
        "adaboost_round_2_feature": BINARY_FEATURES[int(ada_rounds[1]["feature_index"])],
        "adaboost_round_2_weighted_error": float(ada_rounds[1]["weighted_error"]),
        "adaboost_round_2_alpha": float(ada_rounds[1]["alpha"]),
        "adaboost_round_3_feature": BINARY_FEATURES[int(ada_rounds[2]["feature_index"])],
        "adaboost_round_3_weighted_error": float(ada_rounds[2]["weighted_error"]),
        "adaboost_round_3_alpha": float(ada_rounds[2]["alpha"]),
        "adaboost_test_accuracy": ada_metrics["accuracy"],
        "adaboost_test_sensitivity": ada_metrics["sensitivity"],
        "adaboost_test_specificity": ada_metrics["specificity"],
        "adaboost_test_false_positives": ada_metrics["false_positive"],
        "adaboost_test_false_negatives": ada_metrics["false_negative"],
        "adaboost_test_cost_inr": ada_metrics["financial_cost_inr"],
        "logistic_test_accuracy_at_0.5": logistic_metrics["accuracy"],
        "logistic_test_false_positives_at_0.5": logistic_metrics["false_positive"],
        "logistic_test_false_negatives_at_0.5": logistic_metrics["false_negative"],
        "logistic_test_cost_at_0.5_inr": logistic_metrics["financial_cost_inr"],
        "false_alarm_cost_inr": FALSE_ALARM_COST_INR,
        "missed_no_show_cost_inr": MISSED_NO_SHOW_COST_INR,
        "logistic_cost_based_threshold": cost_threshold,
        "logistic_test_cost_at_cost_based_threshold_inr": cost_threshold_metrics[
            "financial_cost_inr"
        ],
        "executive_memo_word_count": len(memo.split()),
        **nn_shapes,
        "executive_memo_path": str(MEMO_PATH.relative_to(ROOT)),
    }

    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as results_file:
        for key, result in results.items():
            if isinstance(result, (int, np.integer)):
                formatted = str(int(result))
            elif isinstance(result, (float, np.floating)):
                formatted = f"{result:.4f}"
            else:
                formatted = result
            results_file.write(f"{key} = {formatted}\n")

    print(f"Results saved to {RESULTS_PATH}")
    print(f"Executive memo saved to {MEMO_PATH}")


if __name__ == "__main__":
    main()
"""Fit and evaluate no-show classifiers with a fixed stratified split."""

import numpy as np

from capstone_io import load_data, write_csv, write_results
from classification import (
    binary_cross_entropy,
    divide,
    fit_classifiers,
    fit_logistic,
    metrics,
    reminder_cost,
    sigmoid,
    split_rows,
)


def main():
    x, outcome, _ = load_data()
    train, test = split_rows(outcome)
    train_y, test_y = outcome[train], outcome[test]
    train_x, test_x, weights, bias, losses, best_k, cv_scores, neighbors = (
        fit_classifiers(x[train], x[test], train_y)
    )
    test_logits = test_x @ weights + bias
    test_probability = sigmoid(test_logits)
    train_probability = sigmoid(train_x @ weights + bias)

    results = {
        "train_rows": train_y.size,
        "train_positives": train_y.sum(),
        "test_rows": test_y.size,
        "test_positives": test_y.sum(),
    }
    results.update({f"loss_epoch_{epoch}": loss for epoch, loss in losses.items()})
    results.update({f"weight_X{i}": weight for i, weight in enumerate(weights, 1)})
    results["bias"] = bias
    results["test_bce"] = binary_cross_entropy(test_logits, test_y)
    for rate in (0.01, 0.1, 1.0):
        _, _, trial_losses = fit_logistic(train_x, train_y, rate, 200, {200})
        results[f"loss_epoch_200_eta_{rate}"] = trial_losses[200]
    results.update({f"odds_ratio_X{i}": ratio for i, ratio in enumerate(np.exp(weights), 1)})

    for tau in (0.5, 0.3):
        evaluation = metrics(test_y, test_probability >= tau)
        results.update({f"tau_{tau}_{name}": value for name, value in evaluation.items()})
    blind = metrics(test_y, np.zeros(test_y.size, dtype=bool))
    results["blind_accuracy"] = blind["accuracy"]

    threshold_rows = []
    for i in range(1, 10):
        tau = i / 10
        evaluation = metrics(test_y, test_probability >= tau)
        tp, fp, tn, fn = (evaluation[key] for key in ("tp", "fp", "tn", "fn"))
        threshold_rows.append((tau, tp, fp, tn, fn, evaluation["recall"], divide(fp, fp + tn)))
    write_csv("part2_thresholds.csv", ("tau", "tp", "fp", "tn", "fn", "tpr", "fpr"), threshold_rows)

    write_csv("part2_knn_cv.csv", ("K", "mean_cv_f1"), cv_scores.items())
    results["knn_best_k"] = best_k
    knn = metrics(test_y, neighbors.sum(axis=1) >= (best_k + 1) // 2)
    results.update({f"knn_{key}": value for key, value in knn.items() if key != "specificity"})

    candidates = [
        (reminder_cost(metrics(train_y, train_probability >= tau)), tau)
        for tau in (i / 20 for i in range(1, 20))
    ]
    training_cost, tau = min(candidates)
    results.update(
        cost_tau=tau,
        cost_train_at_tau=training_cost,
        cost_test_at_tau=reminder_cost(metrics(test_y, test_probability >= tau)),
        **{"cost_test_at_0.5": reminder_cost(metrics(test_y, test_probability >= 0.5))},
        cost_test_flag_nobody=reminder_cost(blind),
    )
    write_results("part2_results.txt", results)


if __name__ == "__main__":
    main()

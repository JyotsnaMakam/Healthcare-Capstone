"""Shared training split, classifiers, metrics and reminder costs."""

import numpy as np


def split_rows(outcome):
    train = np.zeros(outcome.size, dtype=bool)
    for label in (0, 1):
        indices = np.flatnonzero(outcome == label)
        train[indices[: 7 * indices.size // 10]] = True
    return train, ~train


def standardize(train, test):
    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=1)
    return (train - mean) / sd, (test - mean) / sd


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-logits))


def binary_cross_entropy(logits, outcome):
    # This is the probability-based loss expressed in logits for stability.
    return np.mean(np.logaddexp(0, logits) - outcome * logits)


def fit_logistic(x, y, learning_rate, updates, checkpoints):
    weights = np.zeros(x.shape[1])
    bias = 0.0
    losses = {}
    for epoch in range(updates + 1):
        logits = x @ weights + bias
        if epoch in checkpoints:
            losses[epoch] = binary_cross_entropy(logits, y)
        if epoch == updates:
            break
        errors = sigmoid(logits) - y
        weights -= learning_rate * (x.T @ errors) / y.size
        bias -= learning_rate * errors.mean()
    return weights, bias, losses


def divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def metrics(outcome, predicted):
    positive = outcome == 1
    tp = np.count_nonzero(predicted & positive)
    fp = np.count_nonzero(predicted & ~positive)
    tn = np.count_nonzero(~predicted & ~positive)
    fn = np.count_nonzero(~predicted & positive)
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": divide(tp + tn, outcome.size),
        "precision": precision,
        "recall": recall,
        "f1": divide(2 * precision * recall, precision + recall),
        "specificity": divide(tn, tn + fp),
    }


def neighbor_labels(reference, outcome, queries, max_k):
    differences = queries[:, None, :] - reference[None, :, :]
    distances_squared = np.sum(differences**2, axis=2)
    nearest = np.argsort(distances_squared, axis=1, kind="stable")[:, :max_k]
    return outcome[nearest]


def select_k(train_x, train_y):
    choices = list(range(1, 22, 2))
    scores = {k: [] for k in choices}
    folds = np.arange(train_y.size) % 5
    for fold in range(5):
        held_out = folds == fold
        labels = neighbor_labels(
            train_x[~held_out], train_y[~held_out], train_x[held_out], choices[-1]
        )
        for k in choices:
            predicted = labels[:, :k].sum(axis=1) >= (k + 1) // 2
            scores[k].append(metrics(train_y[held_out], predicted)["f1"])
    means = {k: np.mean(scores[k]) for k in choices}
    return max(means, key=means.get), means


def reminder_cost(confusion):
    return 50 * confusion["fp"] + 400 * confusion["fn"]


def fit_classifiers(raw_train, raw_test, train_y):
    """Fit on the training rows and return the shared Part 2 model results."""
    train_x, test_x = standardize(raw_train, raw_test)
    weights, bias, losses = fit_logistic(train_x, train_y, 0.1, 1000, {0, 1, 10, 100, 1000})
    best_k, cv_scores = select_k(train_x, train_y)
    test_neighbors = neighbor_labels(train_x, train_y, test_x, best_k)
    return train_x, test_x, weights, bias, losses, best_k, cv_scores, test_neighbors

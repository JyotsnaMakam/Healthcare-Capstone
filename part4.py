"""Compare structured-data models and count parameters for future models."""

import math

import numpy as np

from capstone_io import load_data, write_csv, write_results
from classification import fit_classifiers, metrics, sigmoid, split_rows


def impurities(outcome):
    p = outcome.mean()
    probabilities = (p, 1 - p)
    entropy = -sum(q * math.log2(q) for q in probabilities if q > 0)
    gini = 1 - sum(q**2 for q in probabilities)
    return np.array([entropy, gini])


def evaluate_splits(train_x, train_y):
    conditions = {
        "S1": lambda x: x[:, 5] == 1,
        "S2": lambda x: x[:, 6] == 1,
        "S3": lambda x: x[:, 2] > 0.5,
        "S4": lambda x: x[:, 0] <= 24,
    }
    root = impurities(train_y)
    rows = {}
    for name, condition in conditions.items():
        yes = condition(train_x)
        left, right = train_y[yes], train_y[~yes]
        child = (left.size * impurities(left) + right.size * impurities(right)) / train_y.size
        info_gain, gini_gain = root - child
        rows[name] = (left.size, left.sum(), right.size, right.sum(), info_gain, gini_gain)

    best_info = max(rows, key=lambda name: rows[name][-2])
    best_gini = max(rows, key=lambda name: rows[name][-1])
    n_yes, p_yes, n_no, p_no, _, _ = rows[best_info]
    results = {
        "root_entropy": root[0],
        "root_gini": root[1],
        "best_split_by_info_gain": best_info,
        "best_split_by_gini_gain": best_gini,
        "stump_rate_yes": p_yes / n_yes,
        "stump_rate_no": p_no / n_no,
    }
    write_csv(
        "part4_splits.csv",
        ("split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"),
        [(name, *row) for name, row in rows.items()],
    )
    return conditions[best_info], results


def evaluate_voting(train_x, train_y, test_x, test_y, stump_condition, results):
    _, standard_test, weights, bias, _, k, _, neighbors = fit_classifiers(train_x, test_x, train_y)
    logistic = sigmoid(standard_test @ weights + bias)
    knn = neighbors.mean(axis=1)
    stump = np.where(stump_condition(test_x), results["stump_rate_yes"], results["stump_rate_no"])
    logistic_label = logistic >= 0.3
    knn_label = neighbors.sum(axis=1) >= (k + 1) // 2
    stump_label = stump >= 0.3
    hard = np.count_nonzero([logistic_label, knn_label, stump_label], axis=0) >= 2
    soft = (logistic + knn + stump) / 3 >= 0.3
    predictions = {
        "logistic": logistic_label,
        "knn": knn_label,
        "stump": stump_label,
        "hard_vote": hard,
        "soft_vote": soft,
    }
    fields = ("tp", "fp", "tn", "fn", "precision", "recall", "f1")
    rows = []
    for name, predicted in predictions.items():
        evaluation = metrics(test_y, predicted)
        rows.append((name, *(evaluation[key] for key in fields)))
    write_csv("part4_voting.csv", ("model", *fields), rows)


def adaboost():
    y = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    rules = (
        np.array([1, 1, -1, -1, -1, -1, -1, -1]),
        np.array([-1, 1, 1, -1, 1, -1, 1, -1]),
    )
    weights = np.full(y.size, 1 / y.size)
    score = np.zeros(y.size)
    rows = []
    results = {}
    for round_no, prediction in enumerate(rules, 1):
        error = weights[prediction != y].sum()
        alpha = 0.5 * math.log((1 - error) / error)
        unnormalized = weights * np.exp(-alpha * y * prediction)
        normalized = unnormalized / unnormalized.sum()
        for point, (before, raw, after) in enumerate(zip(weights, unnormalized, normalized), 1):
            rows.append((round_no, point, before, raw, after))
        weights = normalized
        score += alpha * prediction
        results.update({f"round{round_no}_error": error, f"round{round_no}_alpha": alpha})
    results["ensemble_accuracy"] = np.mean(np.sign(score) == y)
    write_csv(
        "part4_adaboost.csv",
        ("round", "point", "weight_before", "weight_unnormalised", "weight_after"),
        rows,
    )
    return results


def dense_parameters(inputs, outputs):
    return (inputs + 1) * outputs


def convolution_parameters(filter_size, channels, filters):
    return (filter_size**2 * channels + 1) * filters


def spatial_size(size, kernel, stride, padding=0):
    return (size - kernel + 2 * padding) // stride + 1


def architecture():
    conv1_side = spatial_size(28, 5, 1, 2)
    pool1_side = spatial_size(conv1_side, 2, 2)
    conv2_side = spatial_size(pool1_side, 5, 1)
    pool2_side = spatial_size(conv2_side, 2, 2)
    flattened = pool2_side**2 * 16

    mlp = {
        "mlp_hidden1": dense_parameters(28 * 28, 16),
        "mlp_hidden2": dense_parameters(16, 16),
        "mlp_output": dense_parameters(16, 10),
    }
    cnn = {
        "cnn_conv1": convolution_parameters(5, 1, 6),
        "cnn_conv2": convolution_parameters(5, 6, 16),
        "cnn_dense1": dense_parameters(flattened, 32),
        "cnn_dense2": dense_parameters(32, 10),
    }
    write_csv("part4_architecture.csv", ("layer", "parameters"), [*mlp.items(), *cnn.items()])
    return {
        "mlp_total": sum(mlp.values()),
        "cnn_total": sum(cnn.values()),
        "fc_layer_params": dense_parameters(128 * 128 * 3, 1024),
        "conv_layer_params": convolution_parameters(3, 3, 64),
        "rnn_total": dense_parameters(64 + 128, 128) + dense_parameters(128, 16),
    }


def main():
    x, outcome, _ = load_data()
    train, test = split_rows(outcome)
    train_x, train_y = x[train], outcome[train]
    stump_condition, results = evaluate_splits(train_x, train_y)
    evaluate_voting(train_x, train_y, x[test], outcome[test], stump_condition, results)
    results.update(adaboost())
    results["oob_fraction"] = (1 - 1 / train_y.size) ** train_y.size
    results.update(architecture())
    write_results("part4_results.txt", results)


if __name__ == "__main__":
    main()

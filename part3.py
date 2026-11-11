"""Segment the synthetic records and compute their principal components."""

from itertools import combinations

import numpy as np

from capstone_io import load_data, write_csv, write_results


def kmeans(x, k):
    centroids = x[np.arange(k) * (len(x) // k)].copy()
    previous = None
    while True:
        distances = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = distances.argmin(axis=1)
        if np.array_equal(labels, previous):
            return labels, np.sum((x - centroids[labels]) ** 2)
        for i in range(k):
            members = x[labels == i]
            if len(members):
                centroids[i] = members.mean(axis=0)
        previous = labels


def hierarchical_clustering():
    distances = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ])
    rows = []
    counts = {}
    for name, reduce in (("single", np.min), ("complete", np.max), ("average", np.mean)):
        clusters = [(i,) for i in range(len(distances))]
        heights = []
        for merge in range(1, len(distances)):
            height, left, right = min(
                (reduce(distances[np.ix_(a, b)]), a, b)
                for a, b in combinations(clusters, 2)
            )
            cluster_a = "+".join(f"E{i + 1}" for i in left)
            cluster_b = "+".join(f"E{i + 1}" for i in right)
            rows.append((name, merge, cluster_a, cluster_b, height))
            heights.append(height)
            clusters.remove(left)
            clusters.remove(right)
            clusters.append(tuple(sorted(left + right)))
            clusters.sort()
        merges_below_cut = int(np.argmax(np.diff(heights))) + 1
        counts[f"{name}_clusters_at_cut"] = len(distances) - merges_below_cut
    return rows, counts


def pca(x):
    mean = x.mean(axis=0)
    centered = x - mean
    covariance = centered.T @ centered / (len(x) - 1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    largest_entries = np.abs(vectors).argmax(axis=0)
    vectors *= np.sign(vectors[largest_entries, np.arange(vectors.shape[1])])
    loadings = vectors[:, :2]
    return eigenvalues, loadings, centered @ loadings


def main():
    x, outcome, value = load_data()
    continuous = x[:, :5]
    standardized = (continuous - continuous.mean(axis=0)) / continuous.std(axis=0, ddof=1)
    fits = {k: kmeans(standardized, k) for k in range(2, 7)}
    wcss = {k: fit[1] for k, fit in fits.items()}
    results = {f"wcss_k{k}": total for k, total in wcss.items()}
    results["elbow_k"] = next(
        k for k in range(3, 6) if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k])
    )

    labels = fits[3][0]
    ordered = sorted(range(3), key=lambda i: continuous[labels == i, 2].mean())
    segments = {name: labels == label for name, label in zip("ABC", ordered)}
    segment_rows = []
    for name, members in segments.items():
        size = np.count_nonzero(members)
        results[f"size_segment_{name}"] = size
        means = continuous[members].mean(axis=0)
        segment_rows.append((name, size, *means, outcome[members].mean(), value[members].mean()))
    write_csv(
        "part3_segments.csv",
        ("segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
         "outcome_rate", "mean_value"),
        segment_rows,
    )

    merges, cluster_counts = hierarchical_clustering()
    write_csv("part3_hierarchical.csv", ("linkage", "merge_no", "cluster_a", "cluster_b", "height"), merges)
    results.update(cluster_counts)

    eigenvalues, loadings, projections = pca(standardized)
    eigenvalue_sum = eigenvalues.sum()
    explained = eigenvalues / eigenvalue_sum
    cumulative = explained.cumsum()
    results["eigenvalue_sum"] = eigenvalue_sum
    for prefix, values in (
        ("eigenvalue", eigenvalues), ("explained_ratio", explained), ("cumulative_ratio", cumulative)
    ):
        results.update({f"{prefix}_{i}": value for i, value in enumerate(values, 1)})
    results["components_for_80pct"] = int(np.searchsorted(cumulative, 0.8)) + 1
    for component, vector in enumerate(loadings.T, 1):
        results.update({f"pc{component}_loading_X{i}": value for i, value in enumerate(vector, 1)})
    for row, (pc1, pc2) in enumerate(projections[:3], 1):
        results.update({f"row{row}_pc1": pc1, f"row{row}_pc2": pc2})
    for name, members in segments.items():
        results[f"mean_pc1_segment_{name}"] = projections[members, 0].mean()
    write_results("part3_results.txt", results)


if __name__ == "__main__":
    main()

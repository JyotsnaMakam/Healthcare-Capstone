"""Compute the Part 3 clustering and PCA results for the capstone."""

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUT_DIR = ROOT / "outputs"
RESULTS_PATH = OUTPUT_DIR / "part3_results.txt"
SEGMENTS_PATH = OUTPUT_DIR / "part3_segments.csv"
HIERARCHICAL_PATH = OUTPUT_DIR / "part3_hierarchical.csv"

CONTINUOUS_FEATURES = (
    "months_registered",
    "missed_visit_count_12m",
    "copay_burden_ratio",
    "avg_bill_inr_thousands",
    "adherence_score",
)
PROFILE_FEATURES = (
    *CONTINUOUS_FEATURES,
    "chronic_condition_flag",
    "insured_flag",
)
TARGET_COLUMN = "no_show"
VALUE_COLUMN = "care_plan_cost_inr_thousands"
COHORTS = ("E1", "E2", "E3", "E4", "E5", "E6")
COHORT_DISSIMILARITIES = np.asarray(
    (
        (0.0, 7.9, 7.4, 2.9, 2.9, 6.1),
        (7.9, 0.0, 1.4, 5.9, 5.3, 2.0),
        (7.4, 1.4, 0.0, 5.8, 4.6, 1.3),
        (2.9, 5.9, 5.8, 0.0, 2.8, 4.5),
        (2.9, 5.3, 4.6, 2.8, 0.0, 3.4),
        (6.1, 2.0, 1.3, 4.5, 3.4, 0.0),
    ),
    dtype=float,
)


def load_records(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the numeric data required for Part 3."""
    required_columns = (*PROFILE_FEATURES, TARGET_COLUMN, VALUE_COLUMN)
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")
        missing_columns = [name for name in required_columns if name not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing_columns)}"
            )

        columns: dict[str, list[float]] = {name: [] for name in required_columns}
        for row_number, row in enumerate(reader, start=2):
            for name in required_columns:
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid numeric value in column {name!r}, row {row_number}."
                    ) from error

    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    if not arrays[TARGET_COLUMN].size:
        raise ValueError(f"{path} contains no data rows.")
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError(f"{path} contains non-finite numeric values.")
    if arrays[TARGET_COLUMN].size != 800:
        raise ValueError(
            f"Expected 800 records for the pinned K-means starting indices; "
            f"found {arrays[TARGET_COLUMN].size}."
        )
    if not np.all(np.isin(arrays[TARGET_COLUMN], (0.0, 1.0))):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1.")
    return arrays


def standardize(
    x: np.ndarray, ddof: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized data, means, and standard deviations."""
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0, ddof=ddof)
    if np.any(scales == 0.0):
        raise ValueError("PCA and K-means features must have nonzero variance.")
    return (x - means) / scales, means, scales


def kmeans(
    x: np.ndarray, cluster_count: int
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run Lloyd's K-means from pinned row indices until assignments converge."""
    row_count = x.shape[0]
    starting_indices = np.asarray(
        [int(index * row_count / cluster_count) for index in range(cluster_count)],
        dtype=int,
    )
    centroids = x[starting_indices].copy()
    labels = np.full(row_count, -1, dtype=int)

    for iteration in range(1, 10001):
        squared_distances = np.sum(
            (x[:, None, :] - centroids[None, :, :]) ** 2, axis=2
        )
        new_labels = np.argmin(squared_distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        updated_centroids = centroids.copy()
        assigned_counts = np.bincount(labels, minlength=cluster_count)
        for cluster_id in range(cluster_count):
            if assigned_counts[cluster_id]:
                updated_centroids[cluster_id] = np.mean(
                    x[labels == cluster_id], axis=0
                )
            else:
                nearest_distances = squared_distances[
                    np.arange(row_count), labels
                ]
                candidates = np.argsort(-nearest_distances, kind="stable")
                for candidate in candidates:
                    if np.sum(np.all(x[candidate] == updated_centroids, axis=1)) == 0:
                        updated_centroids[cluster_id] = x[candidate]
                        break
        centroids = updated_centroids
    else:
        raise RuntimeError(
            f"K-means with K={cluster_count} did not converge within 10000 updates."
        )

    final_squared_distances = np.sum(
        (x - centroids[labels]) ** 2, axis=1
    )
    wcss = float(np.sum(final_squared_distances))
    return labels, centroids, wcss, iteration


def save_segments(
    path: Path,
    records: dict[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], dict[int, str]]:
    """Write K=3 segment profiles ordered by ascending original-unit X3 mean."""
    means_by_cluster = {
        cluster_id: float(
            np.mean(records["copay_burden_ratio"][labels == cluster_id])
        )
        for cluster_id in range(3)
    }
    ordered_clusters = sorted(means_by_cluster, key=means_by_cluster.__getitem__)
    segment_names = {
        cluster_id: chr(ord("A") + segment_index)
        for segment_index, cluster_id in enumerate(ordered_clusters)
    }
    rows: list[dict[str, float | int | str]] = []
    for cluster_id in ordered_clusters:
        mask = labels == cluster_id
        row: dict[str, float | int | str] = {
            "segment": segment_names[cluster_id],
            "size": int(np.sum(mask)),
        }
        for feature in PROFILE_FEATURES:
            row[f"mean_{feature}"] = float(np.mean(records[feature][mask]))
        row["no_show_rate"] = float(np.mean(records[TARGET_COLUMN][mask]))
        row["mean_care_plan_cost_inr_thousands"] = float(
            np.mean(records[VALUE_COLUMN][mask])
        )
        rows.append(row)

    columns = (
        "segment",
        "size",
        *(f"mean_{feature}" for feature in PROFILE_FEATURES),
        "no_show_rate",
        "mean_care_plan_cost_inr_thousands",
    )
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows, segment_names


def hierarchical_clustering(
    dissimilarities: np.ndarray, method: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Perform five deterministic agglomerative merges with the requested linkage."""
    if method not in ("single", "complete", "average"):
        raise ValueError(f"Unsupported linkage method: {method}")
    if dissimilarities.shape != (6, 6):
        raise ValueError("The cohort dissimilarity matrix must be 6 by 6.")
    if not np.allclose(dissimilarities, dissimilarities.T):
        raise ValueError("The cohort dissimilarity matrix must be symmetric.")
    if not np.allclose(np.diag(dissimilarities), 0.0):
        raise ValueError("The cohort dissimilarity matrix diagonal must be zero.")
    if np.any(dissimilarities < 0.0):
        raise ValueError("Dissimilarities must be nonnegative.")

    clusters: dict[int, tuple[int, ...]] = {
        index: (index,) for index in range(len(COHORTS))
    }
    merge_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    next_cluster_id = len(COHORTS)

    def cluster_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        values = dissimilarities[np.ix_(left, right)]
        if method == "single":
            return float(np.min(values))
        if method == "complete":
            return float(np.max(values))
        return float(np.mean(values))

    for merge_number in range(1, len(COHORTS)):
        cluster_ids = sorted(clusters)
        candidates = [
            (
                cluster_distance(clusters[left_id], clusters[right_id]),
                tuple(sorted(clusters[left_id])),
                tuple(sorted(clusters[right_id])),
                left_id,
                right_id,
            )
            for position, left_id in enumerate(cluster_ids)
            for right_id in cluster_ids[position + 1 :]
        ]
        height, _, _, left_id, right_id = min(candidates)
        left_members = clusters.pop(left_id)
        right_members = clusters.pop(right_id)
        merged_members = tuple(sorted((*left_members, *right_members)))
        merged_cluster_id = next_cluster_id
        next_cluster_id += 1
        clusters[merged_cluster_id] = merged_members
        merge_rows.append(
            {
                "method": method,
                "record_type": "merge",
                "merge_number": merge_number,
                "cluster_left": "+".join(COHORTS[index] for index in left_members),
                "cluster_right": "+".join(COHORTS[index] for index in right_members),
                "merged_cluster": "+".join(
                    COHORTS[index] for index in merged_members
                ),
                "height": height,
                "cut_height": "",
                "clusters_at_cut": "",
                "cluster_members_at_cut": "",
            }
        )
        cluster_rows.append(
            {
                "merge_number": merge_number,
                "height": height,
                "clusters_after_merge": len(clusters),
                "clusters": tuple(
                    tuple(COHORTS[index] for index in clusters[cluster_id])
                    for cluster_id in sorted(clusters)
                ),
            }
        )

    heights = np.asarray([float(row["height"]) for row in merge_rows])
    gaps = np.diff(heights)
    largest_gap_index = int(np.argmax(gaps))
    cut_height = float(
        (heights[largest_gap_index] + heights[largest_gap_index + 1]) / 2.0
    )
    merges_below_cut = largest_gap_index + 1
    clusters_at_cut = len(COHORTS) - merges_below_cut
    cut_state = cluster_rows[merges_below_cut - 1]
    cut_members = cut_state["clusters"]
    cluster_summary = "; ".join("+".join(members) for members in cut_members)
    cut_row: dict[str, object] = {
        "method": method,
        "record_type": "cut",
        "merge_number": "",
        "cluster_left": "",
        "cluster_right": "",
        "merged_cluster": "",
        "height": "",
        "cut_height": cut_height,
        "clusters_at_cut": clusters_at_cut,
        "cluster_members_at_cut": cluster_summary,
    }
    for row in merge_rows:
        row["cut_height"] = cut_height
        row["clusters_at_cut"] = clusters_at_cut
        row["cluster_members_at_cut"] = cluster_summary
    return merge_rows, [cut_row]


def run_pca(
    x_standardized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate sample-covariance PCA scores and sign-canonicalized loadings."""
    covariance = x_standardized.T @ x_standardized / (x_standardized.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    for component_index in range(eigenvectors.shape[1]):
        anchor_index = int(np.argmax(np.abs(eigenvectors[:, component_index])))
        if eigenvectors[anchor_index, component_index] < 0.0:
            eigenvectors[:, component_index] *= -1.0
    projections = x_standardized @ eigenvectors
    return eigenvalues, eigenvectors, projections


def format_array(values: np.ndarray) -> str:
    """Render numeric arrays with the required four decimal places."""
    return np.array2string(
        values,
        precision=4,
        floatmode="fixed",
        suppress_small=True,
        max_line_width=100000,
    )


def main() -> None:
    records = load_records(DATA_PATH)
    x_original = np.column_stack(
        [records[feature] for feature in CONTINUOUS_FEATURES]
    )
    kmeans_standardized, _, _ = standardize(x_original, ddof=0)
    pca_standardized, pca_means, pca_scales = standardize(x_original, ddof=1)

    kmeans_results: dict[int, tuple[np.ndarray, np.ndarray, float, int]] = {}
    for cluster_count in range(2, 7):
        kmeans_results[cluster_count] = kmeans(kmeans_standardized, cluster_count)
    wcss = {k: result[2] for k, result in kmeans_results.items()}

    elbow_k = next(
        (
            k
            for k in (3, 4, 5)
            if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k])
        ),
        None,
    )
    labels_k3 = kmeans_results[3][0]
    segment_rows, segment_names = save_segments(
        SEGMENTS_PATH, records, labels_k3
    )

    all_hierarchical_rows: list[dict[str, object]] = []
    cut_rows: list[dict[str, object]] = []
    for method in ("single", "complete", "average"):
        merge_rows, method_cut_rows = hierarchical_clustering(
            COHORT_DISSIMILARITIES, method
        )
        all_hierarchical_rows.extend(merge_rows)
        all_hierarchical_rows.extend(method_cut_rows)
        cut_rows.extend(method_cut_rows)
    hierarchical_columns = (
        "method",
        "record_type",
        "merge_number",
        "cluster_left",
        "cluster_right",
        "merged_cluster",
        "height",
        "cut_height",
        "clusters_at_cut",
        "cluster_members_at_cut",
    )
    with HIERARCHICAL_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=hierarchical_columns)
        writer.writeheader()
        writer.writerows(all_hierarchical_rows)

    eigenvalues, loadings, projections = run_pca(pca_standardized)
    explained_ratios = eigenvalues / np.sum(eigenvalues)
    mean_pc1_by_segment = {
        segment_names[cluster_id]: float(np.mean(projections[labels_k3 == cluster_id, 0]))
        for cluster_id in range(3)
    }

    results: dict[str, int | float | str] = {
        "records": int(x_original.shape[0]),
        "kmeans_features": ", ".join(CONTINUOUS_FEATURES),
        "kmeans_standardization": "population standard deviation (ddof=0)",
        "kmeans_converged_K2": kmeans_results[2][3],
        "kmeans_converged_K3": kmeans_results[3][3],
        "kmeans_converged_K4": kmeans_results[4][3],
        "kmeans_converged_K5": kmeans_results[5][3],
        "kmeans_converged_K6": kmeans_results[6][3],
        "wcss_K2": wcss[2],
        "wcss_K3": wcss[3],
        "wcss_K4": wcss[4],
        "wcss_K5": wcss[5],
        "wcss_K6": wcss[6],
        "elbow_condition": (
            "smallest K in {3,4,5} with WCSS(K)-WCSS(K+1) "
            "< 0.5*(WCSS(K-1)-WCSS(K))"
        ),
        "elbow_k": "none" if elbow_k is None else elbow_k,
        "subsequent_task_cluster_count": 3,
        "segment_A_size": segment_rows[0]["size"],
        "segment_B_size": segment_rows[1]["size"],
        "segment_C_size": segment_rows[2]["size"],
        "segment_A_mean_X3": segment_rows[0]["mean_copay_burden_ratio"],
        "segment_B_mean_X3": segment_rows[1]["mean_copay_burden_ratio"],
        "segment_C_mean_X3": segment_rows[2]["mean_copay_burden_ratio"],
        "single_clusters_at_cut": cut_rows[0]["clusters_at_cut"],
        "single_cut_height": cut_rows[0]["cut_height"],
        "single_cut_members": cut_rows[0]["cluster_members_at_cut"],
        "complete_clusters_at_cut": cut_rows[1]["clusters_at_cut"],
        "complete_cut_height": cut_rows[1]["cut_height"],
        "complete_cut_members": cut_rows[1]["cluster_members_at_cut"],
        "average_clusters_at_cut": cut_rows[2]["clusters_at_cut"],
        "average_cut_height": cut_rows[2]["cut_height"],
        "average_cut_members": cut_rows[2]["cluster_members_at_cut"],
        "pca_features": ", ".join(CONTINUOUS_FEATURES),
        "pca_standardization": "sample standard deviation (ddof=1)",
        "pca_means": format_array(pca_means),
        "pca_sample_standard_deviations": format_array(pca_scales),
        "pca_eigenvalues": format_array(eigenvalues),
        "pca_explained_variance_ratios": format_array(explained_ratios),
        "pca_cumulative_explained_variance_ratios": format_array(
            np.cumsum(explained_ratios)
        ),
        "pca_loadings_rows_features_columns_PC1_to_PC5": format_array(loadings),
        "pca_first_3_row_projections_rows_records_columns_PC1_to_PC5": (
            format_array(projections[:3])
        ),
        "segment_A_mean_PC1": mean_pc1_by_segment["A"],
        "segment_B_mean_PC1": mean_pc1_by_segment["B"],
        "segment_C_mean_PC1": mean_pc1_by_segment["C"],
        "segments_csv": str(SEGMENTS_PATH.relative_to(ROOT)),
        "hierarchical_csv": str(HIERARCHICAL_PATH.relative_to(ROOT)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        for key, result in results.items():
            if isinstance(result, (int, np.integer)):
                formatted = str(int(result))
            elif isinstance(result, (float, np.floating)):
                formatted = f"{result:.4f}"
            else:
                formatted = result
            output_file.write(f"{key} = {formatted}\n")

    print(f"Results saved to {RESULTS_PATH}")
    print(f"Segment profiles saved to {SEGMENTS_PATH}")
    print(f"Hierarchical clustering saved to {HIERARCHICAL_PATH}")


if __name__ == "__main__":
    main()
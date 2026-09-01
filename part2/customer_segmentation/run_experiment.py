"""End-to-end K-Means customer segmentation experiment.

Author: Weihao Fu
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42


def prepare_directories() -> None:
    for directory in (IMAGES_DIR, RESULTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_and_validate_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    required = {"CustomerID", "Genre", "Age", *FEATURES}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data.empty or data[list(required)].isna().any().any():
        raise ValueError("The dataset is empty or contains missing values.")
    return data.rename(columns={"Genre": "Gender"})


def create_eda_chart(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(data=data, x="Age", hue="Gender", multiple="stack", bins=12, ax=axes[0])
    axes[0].set_title("Customer Age Distribution")
    sns.histplot(data=data, x="Annual Income (k$)", bins=15, color="#377eb8", ax=axes[1])
    axes[1].set_title("Annual Income Distribution")
    sns.histplot(data=data, x="Spending Score (1-100)", bins=15, color="#ff7f00", ax=axes[2])
    axes[2].set_title("Spending Score Distribution")
    fig.suptitle("Mall Customer Dataset Overview", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "eda_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate_k_values(scaled_features) -> pd.DataFrame:
    rows = []
    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(scaled_features)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(scaled_features, labels),
                "davies_bouldin_score": davies_bouldin_score(scaled_features, labels),
                "calinski_harabasz_score": calinski_harabasz_score(scaled_features, labels),
            }
        )
    return pd.DataFrame(rows)


def create_model_selection_chart(scores: pd.DataFrame, best_k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.lineplot(data=scores, x="k", y="inertia", marker="o", ax=axes[0], color="#377eb8")
    axes[0].set_title("Elbow Method")
    axes[0].set_ylabel("Within-Cluster Sum of Squares")
    sns.lineplot(data=scores, x="k", y="silhouette_score", marker="o", ax=axes[1], color="#e41a1c")
    axes[1].axvline(best_k, color="gray", linestyle="--", label=f"Selected k = {best_k}")
    axes[1].set_title("Silhouette Analysis")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(scores["k"])
        axis.grid(alpha=0.25)
    fig.suptitle("Selecting the Number of Customer Segments", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "elbow_silhouette.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_segment_names(centers: pd.DataFrame) -> dict[int, str]:
    """Assign readable business names to the expected five-cluster solution."""
    if len(centers) != 5:
        return {int(cluster): f"Segment {int(cluster) + 1}" for cluster in centers.index}

    overall_center = centers.mean()
    main_cluster = ((centers - overall_center) ** 2).sum(axis=1).idxmin()
    remaining = centers.drop(index=main_cluster)
    low_income = remaining.nsmallest(2, FEATURES[0])
    high_income = remaining.nlargest(2, FEATURES[0])
    names = {int(main_cluster): "Mainstream Customers"}
    names[int(low_income[FEATURES[1]].idxmax())] = "Enthusiastic Shoppers"
    names[int(low_income[FEATURES[1]].idxmin())] = "Budget-Conscious Customers"
    names[int(high_income[FEATURES[1]].idxmax())] = "Premium Customers"
    names[int(high_income[FEATURES[1]].idxmin())] = "Cautious Wealthy Customers"
    return names


def create_segment_charts(data: pd.DataFrame, profiles: pd.DataFrame) -> None:
    palette = sns.color_palette("Set2", n_colors=data["Segment"].nunique())
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=data,
        x=FEATURES[0],
        y=FEATURES[1],
        hue="Segment",
        palette=palette,
        s=85,
        alpha=0.85,
        ax=ax,
    )
    sns.scatterplot(
        data=profiles,
        x=f"mean_{FEATURES[0]}",
        y=f"mean_{FEATURES[1]}",
        color="black",
        marker="X",
        s=220,
        label="Cluster centroids",
        ax=ax,
    )
    ax.set_title("K-Means Customer Segments", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "customer_segments.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    profile_plot = profiles.set_index("Segment")[[f"mean_{FEATURES[0]}", f"mean_{FEATURES[1]}"]]
    profile_plot.columns = ["Mean Annual Income (k$)", "Mean Spending Score"]
    normalized = profile_plot / profile_plot.max(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    normalized.plot(kind="bar", ax=ax, color=["#377eb8", "#ff7f00"])
    ax.set_title("Relative Customer Segment Profiles", fontsize=15, fontweight="bold")
    ax.set_ylabel("Relative Level (Column Maximum = 1.0)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "cluster_profiles.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prepare_directories()
    sns.set_theme(style="whitegrid", context="notebook")
    data = load_and_validate_data()
    create_eda_chart(data)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[FEATURES])
    scores = evaluate_k_values(scaled_features)
    best_k = int(scores.loc[scores["silhouette_score"].idxmax(), "k"])
    create_model_selection_chart(scores, best_k)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)),
        ]
    )
    data["Cluster"] = pipeline.fit_predict(data[FEATURES])
    raw_centers = pd.DataFrame(
        pipeline.named_steps["scaler"].inverse_transform(pipeline.named_steps["kmeans"].cluster_centers_),
        columns=FEATURES,
    )
    segment_names = build_segment_names(raw_centers)
    data["Segment"] = data["Cluster"].map(segment_names)

    profiles = (
        data.groupby(["Cluster", "Segment"], as_index=False)
        .agg(
            customer_count=("CustomerID", "count"),
            mean_age=("Age", "mean"),
            **{
                f"mean_{FEATURES[0]}": (FEATURES[0], "mean"),
                f"mean_{FEATURES[1]}": (FEATURES[1], "mean"),
            },
        )
        .sort_values("Cluster")
    )
    profiles["customer_share_percent"] = profiles["customer_count"] / len(data) * 100
    create_segment_charts(data, profiles)

    selected = scores.loc[scores["k"] == best_k].iloc[0]
    summary = {
        "author": "Weihao Fu",
        "dataset_rows": int(len(data)),
        "features": FEATURES,
        "selected_k": best_k,
        "silhouette_score": round(float(selected["silhouette_score"]), 4),
        "davies_bouldin_score": round(float(selected["davies_bouldin_score"]), 4),
        "calinski_harabasz_score": round(float(selected["calinski_harabasz_score"]), 2),
        "random_state": RANDOM_STATE,
    }
    data.to_csv(RESULTS_DIR / "customer_segments.csv", index=False)
    scores.round(4).to_csv(RESULTS_DIR / "k_evaluation.csv", index=False)
    profiles.round(2).to_csv(RESULTS_DIR / "cluster_profiles.csv", index=False)
    (RESULTS_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    joblib.dump(pipeline, MODELS_DIR / "kmeans_pipeline.joblib")

    print("Customer segmentation experiment completed successfully.")
    print(json.dumps(summary, indent=2))
    print("\nCluster profiles:")
    print(profiles.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

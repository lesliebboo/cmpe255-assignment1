from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "train.csv"
IMAGES = ROOT / "images"
RESULTS = ROOT / "results"

AGE_BINS = [0, 12, 17, 34, 59, np.inf]
AGE_LABELS = ["Child", "Teen", "Young Adult", "Adult", "Senior"]


def distribution_age_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Impute ages by Sex/Pclass age-stage distributions with a fixed seed."""
    output = df.copy()
    observed = df.dropna(subset=["Age"]).copy()
    observed["AgeGroup"] = pd.cut(
        observed["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
    )
    rng = np.random.default_rng(42)

    for index in output.index[output["Age"].isna()]:
        pool = observed[
            (observed["Sex"] == output.at[index, "Sex"])
            & (observed["Pclass"] == output.at[index, "Pclass"])
        ]
        if pool.empty:
            pool = observed

        probabilities = (
            pool["AgeGroup"]
            .value_counts(normalize=True, sort=False)
            .reindex(AGE_LABELS, fill_value=0)
        )
        selected_group = rng.choice(AGE_LABELS, p=probabilities.to_numpy())
        age_pool = pool.loc[pool["AgeGroup"] == selected_group, "Age"].to_numpy()
        output.at[index, "Age"] = float(rng.choice(age_pool))

    return output


def survival_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        df.groupby(column, observed=False)["Survived"]
        .agg(Passengers="count", Survivors="sum", SurvivalRate="mean")
        .reset_index()
    )
    summary["SurvivalRate"] = (summary["SurvivalRate"] * 100).round(2)
    return summary


def label_rates(axis) -> None:
    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(SOURCE)
    data = distribution_age_imputation(raw)
    data["Embarked"] = data["Embarked"].fillna(raw["Embarked"].mode()[0])
    data["AgeGroup"] = pd.cut(
        data["Age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
    )
    data["FareQuartile"] = pd.qcut(
        data["Fare"],
        q=4,
        labels=["Q1 Lowest", "Q2", "Q3", "Q4 Highest"],
    )

    summaries = {
        "Sex": survival_summary(data, "Sex"),
        "Pclass": survival_summary(data, "Pclass"),
        "AgeGroup": survival_summary(data, "AgeGroup"),
        "FareQuartile": survival_summary(data, "FareQuartile"),
        "Embarked": survival_summary(data, "Embarked"),
    }

    combined = []
    for feature, table in summaries.items():
        exported = table.copy()
        exported.insert(0, "Feature", feature)
        exported.rename(columns={feature: "Category"}, inplace=True)
        combined.append(exported)
    pd.concat(combined, ignore_index=True).to_csv(
        RESULTS / "eda_survival_summary.csv", index=False
    )

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    counts = data["Survived"].value_counts().sort_index()
    axes[0, 0].bar(["Did not survive", "Survived"], counts.values)
    axes[0, 0].set_title("Overall Survival Distribution")
    axes[0, 0].set_ylabel("Passenger count")
    for index, value in enumerate(counts.values):
        axes[0, 0].text(index, value + 8, str(value), ha="center")

    plot_specs = [
        ("Sex", axes[0, 1], "Survival Rate by Sex", "Sex"),
        ("Pclass", axes[0, 2], "Survival Rate by Passenger Class", "Passenger class"),
        ("AgeGroup", axes[1, 0], "Survival Rate by Age Group", "Age group"),
        ("FareQuartile", axes[1, 1], "Survival Rate by Fare Quartile", "Fare quartile"),
        ("Embarked", axes[1, 2], "Survival Rate by Embarkation Port", "Embarkation port"),
    ]

    for feature, axis, title, xlabel in plot_specs:
        table = summaries[feature]
        sns.barplot(data=table, x=feature, y="SurvivalRate", ax=axis)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Survival rate (%)")
        axis.set_ylim(0, 85)
        label_rates(axis)
        axis.tick_params(axis="x", rotation=15)

    fig.suptitle(
        "Titanic Exploratory Data Analysis",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(IMAGES / "eda_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=data,
        x="Age",
        hue="Survived",
        bins=30,
        kde=True,
        multiple="layer",
        alpha=0.45,
        ax=axis,
    )
    axis.set_title("Age Distribution by Survival Outcome")
    axis.set_xlabel("Age")
    axis.set_ylabel("Passenger count")
    fig.tight_layout()
    fig.savefig(IMAGES / "age_distribution_by_survival.png", dpi=300)
    plt.close(fig)

    print("Overall survival:", counts.to_dict())
    for feature, table in summaries.items():
        print(f"\n{feature}\n{table.to_string(index=False)}")
    print("\nSaved:", RESULTS / "eda_survival_summary.csv")
    print("Saved:", IMAGES / "eda_overview.png")
    print("Saved:", IMAGES / "age_distribution_by_survival.png")


if __name__ == "__main__":
    main()

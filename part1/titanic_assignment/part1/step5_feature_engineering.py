from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_SOURCE = ROOT / "data" / "train.csv"
TEST_SOURCE = ROOT / "data" / "test.csv"
RESULTS = ROOT / "results"

AGE_BINS = [0, 12, 17, 34, 59, np.inf]
AGE_LABELS = ["Child", "Teen", "Young Adult", "Adult", "Senior"]


def build_age_reference(train: pd.DataFrame) -> pd.DataFrame:
    reference = train.dropna(subset=["Age"]).copy()
    reference["AgeGroup"] = pd.cut(
        reference["Age"], AGE_BINS, labels=AGE_LABELS, include_lowest=True
    )
    return reference


def distribution_age_imputation(
    df: pd.DataFrame, reference: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Sample missing ages from training distributions conditional on Sex/Pclass."""
    output = df.copy()
    rng = np.random.default_rng(seed)

    for index in output.index[output["Age"].isna()]:
        pool = reference[
            (reference["Sex"] == output.at[index, "Sex"])
            & (reference["Pclass"] == output.at[index, "Pclass"])
        ]
        if pool.empty:
            pool = reference

        probabilities = (
            pool["AgeGroup"]
            .value_counts(normalize=True, sort=False)
            .reindex(AGE_LABELS, fill_value=0)
        )
        selected_group = rng.choice(AGE_LABELS, p=probabilities.to_numpy())
        age_pool = pool.loc[pool["AgeGroup"] == selected_group, "Age"].to_numpy()
        output.at[index, "Age"] = float(rng.choice(age_pool))

    return output


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^.]*)\.", name)
    return match.group(1).strip() if match else "Unknown"


def normalize_titles(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["Title"] = train["Name"].map(extract_title)
    test["Title"] = test["Name"].map(extract_title)

    title_replacements = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    common_titles = {"Mr", "Miss", "Mrs", "Master"}
    for frame in (train, test):
        frame["Title"] = frame["Title"].replace(title_replacements)
        frame.loc[~frame["Title"].isin(common_titles), "Title"] = "Rare"
    return train, test


def engineer_features(
    frame: pd.DataFrame, ticket_sizes: pd.Series
) -> pd.DataFrame:
    output = frame.copy()
    output["FamilySize"] = output["SibSp"] + output["Parch"] + 1
    output["IsAlone"] = (output["FamilySize"] == 1).astype(int)
    output["CabinKnown"] = output["Cabin"].notna().astype(int)
    output["TicketGroupSize"] = output["Ticket"].map(ticket_sizes).astype(int)
    output["FarePerPerson"] = output["Fare"] / output["TicketGroupSize"]
    output["AgeGroup"] = pd.cut(
        output["Age"], AGE_BINS, labels=AGE_LABELS, include_lowest=True
    )
    return output


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw_train = pd.read_csv(TRAIN_SOURCE)
    raw_test = pd.read_csv(TEST_SOURCE)

    age_reference = build_age_reference(raw_train)
    train = distribution_age_imputation(raw_train, age_reference, seed=42)
    test = distribution_age_imputation(raw_test, age_reference, seed=43)

    embarked_mode = raw_train["Embarked"].mode()[0]
    fare_median = raw_train["Fare"].median()
    train["Embarked"] = train["Embarked"].fillna(embarked_mode)
    test["Embarked"] = test["Embarked"].fillna(embarked_mode)
    train["Fare"] = train["Fare"].fillna(fare_median)
    test["Fare"] = test["Fare"].fillna(fare_median)

    train, test = normalize_titles(train, test)
    combined_tickets = pd.concat([train["Ticket"], test["Ticket"]])
    ticket_sizes = combined_tickets.value_counts()
    train = engineer_features(train, ticket_sizes)
    test = engineer_features(test, ticket_sizes)

    selected_features = [
        "PassengerId",
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "AgeGroup",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title",
        "CabinKnown",
        "TicketGroupSize",
        "FarePerPerson",
    ]
    test_features = [column for column in selected_features if column != "Survived"]

    train_output = train[selected_features]
    test_output = test[test_features]
    train_path = RESULTS / "engineered_train.csv"
    test_path = RESULTS / "engineered_test.csv"
    train_output.to_csv(train_path, index=False)
    test_output.to_csv(test_path, index=False)

    feature_summary = pd.DataFrame(
        [
            ["FamilySize", "SibSp + Parch + 1", "Total family members traveling together"],
            ["IsAlone", "1 when FamilySize = 1", "Whether the passenger traveled alone"],
            ["Title", "Extracted from Name", "Condensed social title: Mr, Mrs, Miss, Master, or Rare"],
            ["CabinKnown", "1 when Cabin is present", "Whether cabin information was recorded"],
            ["TicketGroupSize", "Count of shared Ticket", "Number of passengers sharing the ticket"],
            ["FarePerPerson", "Fare / TicketGroupSize", "Approximate fare paid per person"],
            ["AgeGroup", "Age placed into five stages", "Interpretable age category"],
        ],
        columns=["Feature", "Definition", "Purpose"],
    )
    summary_path = RESULTS / "feature_engineering_summary.csv"
    feature_summary.to_csv(summary_path, index=False)

    print(f"Training output: {train_output.shape}")
    print(f"Test output: {test_output.shape}")
    print("Train missing values:", int(train_output.isna().sum().sum()))
    print("Test missing values:", int(test_output.isna().sum().sum()))
    print("\nTitle counts (train):")
    print(train_output["Title"].value_counts().to_string())
    print("\nEngineered feature ranges:")
    print(train_output[["FamilySize", "TicketGroupSize", "FarePerPerson"]].describe().to_string())
    print(f"\nSaved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

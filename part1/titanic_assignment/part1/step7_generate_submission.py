from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

TEST_SOURCE = RESULTS / "engineered_test.csv"
SAMPLE_SOURCE = ROOT / "data" / "gender_submission.csv"
MODEL_SOURCE = MODELS / "best_titanic_model.joblib"

FEATURES = [
    "Pclass",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "CabinKnown",
    "TicketGroupSize",
    "FarePerPerson",
    "Sex",
    "AgeGroup",
    "Embarked",
    "Title",
]


def main() -> None:
    test = pd.read_csv(TEST_SOURCE)
    sample = pd.read_csv(SAMPLE_SOURCE)
    model = joblib.load(MODEL_SOURCE)

    predictions = model.predict(test[FEATURES]).astype(int)
    probabilities = model.predict_proba(test[FEATURES])[:, 1]

    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": predictions,
        }
    )
    submission_path = RESULTS / "submission.csv"
    submission.to_csv(submission_path, index=False)

    prediction_details = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "PredictedSurvived": predictions,
            "SurvivalProbability": probabilities,
        }
    )
    details_path = RESULTS / "test_prediction_details.csv"
    prediction_details.to_csv(details_path, index=False)

    checks = {
        "columns_match_sample": list(submission.columns) == list(sample.columns),
        "row_count_matches_sample": len(submission) == len(sample),
        "passenger_ids_match_sample": submission["PassengerId"].equals(
            sample["PassengerId"]
        ),
        "passenger_ids_unique": submission["PassengerId"].is_unique,
        "no_missing_values": not submission.isna().any().any(),
        "binary_predictions_only": set(submission["Survived"].unique()).issubset({0, 1}),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(f"Submission validation failed: {failed_checks}")

    counts = submission["Survived"].value_counts().sort_index()
    print("Submission validation:")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
    print(f"\nRows: {len(submission)}")
    print(f"Predicted did not survive: {int(counts.get(0, 0))}")
    print(f"Predicted survived: {int(counts.get(1, 0))}")
    print(f"Predicted survival rate: {predictions.mean():.2%}")
    print("\nPreview:")
    print(submission.head(10).to_string(index=False))
    print(f"\nSaved: {submission_path}")
    print(f"Saved: {details_path}")


if __name__ == "__main__":
    main()

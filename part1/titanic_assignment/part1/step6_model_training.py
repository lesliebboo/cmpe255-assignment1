from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
MODELS = ROOT / "models"
TRAIN_SOURCE = RESULTS / "engineered_train.csv"

TARGET = "Survived"
ID_COLUMN = "PassengerId"
NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "CabinKnown",
    "TicketGroupSize",
    "FarePerPerson",
]
CATEGORICAL_FEATURES = ["Sex", "AgeGroup", "Embarked", "Title"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_pipeline(model) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(TRAIN_SOURCE)
    x = data[ALL_FEATURES]
    y = data[TARGET]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=4, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=3,
            random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC_AUC": "roc_auc",
    }

    rows = []
    pipelines = {}
    for name, model in models.items():
        pipeline = make_pipeline(model)
        pipelines[name] = pipeline
        scores = cross_validate(pipeline, x, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"Model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"Mean_{metric}"] = values.mean()
            row[f"Std_{metric}"] = values.std()
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(
        ["Mean_Accuracy", "Mean_F1"], ascending=False
    )
    comparison_path = RESULTS / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    best_name = comparison.iloc[0]["Model"]
    best_pipeline = pipelines[best_name]
    out_of_fold_predictions = cross_val_predict(best_pipeline, x, y, cv=cv, n_jobs=-1)
    report = pd.DataFrame(
        classification_report(y, out_of_fold_predictions, output_dict=True)
    ).transpose()
    report_path = RESULTS / "best_model_classification_report.csv"
    report.to_csv(report_path)

    best_pipeline.fit(x, y)
    model_path = MODELS / "best_titanic_model.joblib"
    joblib.dump(best_pipeline, model_path)

    sns.set_theme(style="whitegrid", context="notebook")
    plot_data = comparison.melt(
        id_vars="Model",
        value_vars=["Mean_Accuracy", "Mean_F1", "Mean_ROC_AUC"],
        var_name="Metric",
        value_name="Score",
    )
    plot_data["Metric"] = plot_data["Metric"].str.replace("Mean_", "", regex=False)
    fig, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=plot_data, x="Model", y="Score", hue="Metric", ax=axis)
    axis.set_title("Five-Fold Cross-Validation Model Comparison")
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean validation score")
    axis.set_ylim(0.55, 0.90)
    axis.tick_params(axis="x", rotation=12)
    axis.legend(title="Metric", loc="lower right")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    comparison_image = IMAGES / "model_comparison.png"
    fig.savefig(comparison_image, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y,
        out_of_fold_predictions,
        display_labels=["Did not survive", "Survived"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"Out-of-Fold Confusion Matrix: {best_name}")
    fig.tight_layout()
    confusion_image = IMAGES / "best_model_confusion_matrix.png"
    fig.savefig(confusion_image, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected best model: {best_name}")
    print("\nOut-of-fold classification report:")
    print(report.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {comparison_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {model_path}")
    print(f"Saved: {comparison_image}")
    print(f"Saved: {confusion_image}")


if __name__ == "__main__":
    main()

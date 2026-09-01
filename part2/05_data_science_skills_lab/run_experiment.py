"""Compact end-to-end classification skills lab. Author: Weihao Fu."""
from pathlib import Path
import json, joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE=Path(__file__).resolve().parent; DATA=BASE/"data"; IMAGES=BASE/"images"; RESULTS=BASE/"results"; MODELS=BASE/"models"

def main():
    for d in (DATA,IMAGES,RESULTS,MODELS): d.mkdir(parents=True,exist_ok=True)
    bunch=load_breast_cancer(as_frame=True); df=bunch.frame.rename(columns={"target":"diagnosis"}); df["diagnosis_label"]=df.diagnosis.map({0:"malignant",1:"benign"}); df.to_csv(DATA/"breast_cancer_wisconsin.csv",index=False)
    X=df[bunch.feature_names]; y=df.diagnosis; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    models={"Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000,random_state=42))]),
            "Random Forest":RandomForestClassifier(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1),
            "Gradient Boosting":GradientBoostingClassifier(n_estimators=120,max_depth=2,random_state=42)}
    cv=StratifiedKFold(5,shuffle=True,random_state=42); rows=[]
    for name,model in models.items():
        s=cross_validate(model,Xtr,ytr,cv=cv,scoring=["accuracy","f1","roc_auc"])
        rows.append({"model":name,"cv_accuracy":s["test_accuracy"].mean(),"cv_f1":s["test_f1"].mean(),"cv_roc_auc":s["test_roc_auc"].mean()})
    scores=pd.DataFrame(rows).sort_values("cv_roc_auc",ascending=False); best_name=scores.iloc[0].model; best=models[best_name]; best.fit(Xtr,ytr)
    pred=best.predict(Xte); prob=best.predict_proba(Xte)[:,1]; cm=confusion_matrix(yte,pred)
    test={"accuracy":accuracy_score(yte,pred),"f1":f1_score(yte,pred),"roc_auc":roc_auc_score(yte,prob)}
    scores.round(4).to_csv(RESULTS/"cross_validation_results.csv",index=False)
    pd.DataFrame(cm,index=["actual_malignant","actual_benign"],columns=["predicted_malignant","predicted_benign"]).to_csv(RESULTS/"test_confusion_matrix.csv")
    summary={"author":"Weihao Fu","dataset_rows":len(df),"numeric_features":len(bunch.feature_names),"best_model":best_name,
             "test_accuracy":round(test["accuracy"],4),"test_f1":round(test["f1"],4),"test_roc_auc":round(test["roc_auc"],4)}
    (RESULTS/"experiment_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); joblib.dump(best,MODELS/"best_classification_model.joblib",compress=3)
    if best_name=="Logistic Regression": importance=abs(best.named_steps["model"].coef_[0])
    else: importance=best.feature_importances_
    imp=pd.DataFrame({"feature":bunch.feature_names,"importance":importance}).sort_values("importance",ascending=False); imp.to_csv(RESULTS/"feature_importance.csv",index=False)
    sns.set_theme(style="whitegrid"); fig,ax=plt.subplots(1,3,figsize=(16,4.8))
    sns.countplot(data=df,x="diagnosis_label",ax=ax[0],hue="diagnosis_label",palette="Set2",legend=False); ax[0].set_title("Target Distribution")
    scores.set_index("model")[["cv_accuracy","cv_f1","cv_roc_auc"]].plot(kind="bar",ax=ax[1],ylim=(.85,1.01)); ax[1].set_title("Five-Fold Model Comparison"); ax[1].tick_params(axis="x",rotation=20)
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax[2],xticklabels=["Malignant","Benign"],yticklabels=["Malignant","Benign"]); ax[2].set(title=f"Test Confusion Matrix — {best_name}",xlabel="Predicted",ylabel="Actual")
    fig.suptitle("Data Science Skills Lab",fontsize=15,fontweight="bold"); fig.tight_layout(); fig.savefig(IMAGES/"skills_lab_results.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); sns.barplot(data=imp.head(12),x="importance",y="feature",ax=ax,color="#4e79a7"); ax.set_title("Most Influential Model Features"); fig.tight_layout(); fig.savefig(IMAGES/"feature_importance.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    print("Data science skills lab completed."); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()

"""Predict NYC taxi fares with three regression models. Author: Weihao Fu."""
from pathlib import Path
import json, joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

BASE=Path(__file__).resolve().parent; DATA=BASE/"data"; IMAGES=BASE/"images"; RESULTS=BASE/"results"; MODELS=BASE/"models"
NUM=["distance","passengers","pickup_hour","pickup_dayofweek"]; CAT=["pickup_borough","dropoff_borough"]

def main():
    for d in (IMAGES,RESULTS,MODELS): d.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA/"taxis.csv",parse_dates=["pickup","dropoff"])
    df=df.dropna(subset=["fare","distance"]).query("fare > 0 and distance > 0 and fare < 100").copy()
    df["pickup_hour"]=df.pickup.dt.hour; df["pickup_dayofweek"]=df.pickup.dt.dayofweek
    X=df[NUM+CAT]; y=df.fare
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),NUM),
                            ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),CAT)])
    models={"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=42,n_jobs=-1),
            "Gradient Boosting":GradientBoostingRegressor(n_estimators=120,max_depth=2,random_state=42)}
    rows=[]; fitted={}
    for name,model in models.items():
        pipe=Pipeline([("preprocessor",prep),("model",model)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); fitted[name]=(pipe,pred)
        rows.append({"model":name,"mae":mean_absolute_error(yte,pred),"rmse":mean_squared_error(yte,pred)**.5,"r2":r2_score(yte,pred)})
    scores=pd.DataFrame(rows).sort_values("rmse"); best=scores.iloc[0].model; pipe,pred=fitted[best]
    scores.round(4).to_csv(RESULTS/"model_comparison.csv",index=False)
    pd.DataFrame({"actual_fare":yte,"predicted_fare":pred}).round(2).to_csv(RESULTS/"test_predictions.csv",index=False)
    summary={"author":"Weihao Fu","usable_rows":len(df),"best_model":best,"test_rows":len(yte),
             "mae":round(float(scores.iloc[0].mae),3),"rmse":round(float(scores.iloc[0].rmse),3),"r2":round(float(scores.iloc[0].r2),3)}
    (RESULTS/"experiment_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); joblib.dump(pipe,MODELS/"best_fare_model.joblib",compress=3)
    sns.set_theme(style="whitegrid"); fig,ax=plt.subplots(1,2,figsize=(12,4.8))
    sns.barplot(data=scores,x="rmse",y="model",ax=ax[0],color="#4e79a7"); ax[0].set_title("Test RMSE by Model")
    ax[1].scatter(yte,pred,alpha=.35,s=25,color="#f28e2b"); lo=min(yte.min(),pred.min()); hi=max(yte.max(),pred.max()); ax[1].plot([lo,hi],[lo,hi],"k--")
    ax[1].set(xlabel="Actual Fare ($)",ylabel="Predicted Fare ($)",title=f"Actual vs Predicted — {best}")
    fig.suptitle("NYC Taxi Fare Prediction",fontsize=15,fontweight="bold"); fig.tight_layout(); fig.savefig(IMAGES/"taxi_model_results.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    print("NYC taxi experiment completed."); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()

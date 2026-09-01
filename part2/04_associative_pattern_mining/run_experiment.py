"""Mine frequent grocery pairs and association rules without extra packages. Author: Weihao Fu."""
from pathlib import Path
from collections import Counter
from itertools import combinations
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE=Path(__file__).resolve().parent; DATA=BASE/"data"; IMAGES=BASE/"images"; RESULTS=BASE/"results"
MIN_SUPPORT=.001; MIN_CONFIDENCE=.05

def main():
    for d in (IMAGES,RESULTS): d.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA/"Groceries_dataset.csv"); df["transaction_id"]=df.Member_number.astype(str)+"_"+df.Date.astype(str)
    baskets=df.groupby("transaction_id").itemDescription.apply(lambda x: sorted(set(x))).tolist(); n=len(baskets)
    items=Counter(i for b in baskets for i in b); pairs=Counter(p for b in baskets for p in combinations(b,2))
    item_rows=[{"item":i,"count":c,"support":c/n} for i,c in items.items()]
    pair_rows=[]; rules=[]
    for (a,b),count in pairs.items():
        support=count/n
        if support < MIN_SUPPORT: continue
        pair_rows.append({"item_a":a,"item_b":b,"count":count,"support":support})
        for antecedent,consequent in [(a,b),(b,a)]:
            confidence=count/items[antecedent]; consequent_support=items[consequent]/n
            if confidence>=MIN_CONFIDENCE:
                rules.append({"antecedent":antecedent,"consequent":consequent,"support":support,"confidence":confidence,"lift":confidence/consequent_support,"count":count})
    item_df=pd.DataFrame(item_rows).sort_values("support",ascending=False); pair_df=pd.DataFrame(pair_rows).sort_values("support",ascending=False)
    rules_df=pd.DataFrame(rules).sort_values(["lift","confidence"],ascending=False)
    item_df.to_csv(RESULTS/"item_support.csv",index=False); pair_df.to_csv(RESULTS/"frequent_pairs.csv",index=False); rules_df.to_csv(RESULTS/"association_rules.csv",index=False)
    top=rules_df.iloc[0]; summary={"author":"Weihao Fu","transactions":n,"unique_items":len(items),"frequent_pairs":len(pair_df),"association_rules":len(rules_df),
      "minimum_support":MIN_SUPPORT,"minimum_confidence":MIN_CONFIDENCE,"highest_lift_rule":f"{top.antecedent} -> {top.consequent}","highest_lift":round(float(top.lift),3)}
    (RESULTS/"experiment_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    sns.set_theme(style="whitegrid"); fig,ax=plt.subplots(1,2,figsize=(14,5))
    sns.barplot(data=item_df.head(12),x="support",y="item",ax=ax[0],color="#4e79a7"); ax[0].set_title("Most Frequent Grocery Items")
    plot=rules_df.sort_values("lift",ascending=False).head(25); sns.scatterplot(data=plot,x="support",y="confidence",size="lift",hue="lift",palette="viridis",sizes=(50,300),ax=ax[1])
    ax[1].set_title("Top Association Rules"); fig.suptitle("Market Basket Pattern Mining",fontsize=15,fontweight="bold"); fig.tight_layout(); fig.savefig(IMAGES/"market_basket_results.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    print("Association pattern experiment completed."); print(json.dumps(summary,indent=2)); print("\nTop rules:\n",rules_df.head(8).round(3).to_string(index=False))
if __name__=="__main__": main()

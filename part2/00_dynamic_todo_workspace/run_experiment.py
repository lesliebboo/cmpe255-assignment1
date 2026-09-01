"""Create and analyze a reproducible task-management workspace. Author: Weihao Fu."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
DATA, IMAGES, RESULTS = BASE / "data", BASE / "images", BASE / "results"

def create_tasks():
    rng = np.random.default_rng(42)
    categories = ["Coursework", "Coding", "Research", "Career", "Personal"]
    priorities = ["Low", "Medium", "High"]
    statuses = ["To Do", "In Progress", "Completed"]
    rows = []
    start = pd.Timestamp("2026-08-01")
    for task_id in range(1, 61):
        created = start + pd.Timedelta(days=int(rng.integers(0, 30)))
        status = rng.choice(statuses, p=[0.28, 0.25, 0.47])
        duration = int(rng.integers(1, 15))
        rows.append({"task_id": task_id, "title": f"Sample Task {task_id:02d}",
                     "category": rng.choice(categories), "priority": rng.choice(priorities, p=[.25,.45,.30]),
                     "status": status, "created_date": created.date(),
                     "due_date": (created + pd.Timedelta(days=int(rng.integers(2, 18)))).date(),
                     "estimated_hours": round(float(rng.uniform(.5, 8)), 1),
                     "completion_days": duration if status == "Completed" else np.nan})
    return pd.DataFrame(rows)

def main():
    for d in (DATA, IMAGES, RESULTS): d.mkdir(parents=True, exist_ok=True)
    tasks = create_tasks(); tasks.to_csv(DATA / "tasks.csv", index=False)
    tasks["is_completed"] = tasks.status.eq("Completed")
    summary = {"author":"Weihao Fu", "total_tasks":len(tasks),
               "completed_tasks":int(tasks.is_completed.sum()),
               "completion_rate_percent":round(tasks.is_completed.mean()*100,1),
               "estimated_hours":round(tasks.estimated_hours.sum(),1),
               "average_completion_days":round(tasks.completion_days.mean(),1)}
    by_category = tasks.groupby("category").agg(tasks=("task_id","count"), completed=("is_completed","sum"),
                                                  estimated_hours=("estimated_hours","sum")).reset_index()
    by_category["completion_rate_percent"] = (by_category.completed/by_category.tasks*100).round(1)
    by_category.to_csv(RESULTS / "category_summary.csv", index=False)
    tasks.groupby(["priority","status"]).size().rename("task_count").reset_index().to_csv(RESULTS / "priority_status.csv", index=False)
    (RESULTS / "workspace_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    sns.set_theme(style="whitegrid"); fig, ax = plt.subplots(1,3,figsize=(15,4.5))
    sns.countplot(data=tasks,x="status",order=["To Do","In Progress","Completed"],ax=ax[0],palette="Blues",hue="status",legend=False)
    ax[0].set_title("Task Pipeline")
    sns.barplot(data=by_category,x="completion_rate_percent",y="category",ax=ax[1],color="#59a14f")
    ax[1].set_title("Completion Rate by Category"); ax[1].set_xlim(0,100)
    p = tasks.groupby(["priority","status"]).size().unstack(fill_value=0).reindex(["High","Medium","Low"])
    p.plot(kind="bar",stacked=True,ax=ax[2],color=["#4e79a7","#f28e2b","#e15759"])
    ax[2].set_title("Status Mix by Priority"); ax[2].tick_params(axis="x",rotation=0)
    fig.suptitle("Dynamic Todo Workspace Analytics",fontsize=15,fontweight="bold"); fig.tight_layout()
    fig.savefig(IMAGES / "todo_analytics_dashboard.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    print("Dynamic todo workspace experiment completed."); print(json.dumps(summary,indent=2))
if __name__ == "__main__": main()

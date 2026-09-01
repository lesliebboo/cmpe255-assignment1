# YouTube Walkthrough Outline

**Suggested length:** 4–6 minutes  
**Presenter:** Weihao Fu

## 1. Introduction (30 seconds)

"Hello, my name is Weihao Fu. This is CMPE 255 Assignment 1, Part 2. I used ChatGPT and Codex to reproduce the instructor's customer segmentation experiment as a simplified end-to-end K-Means clustering project."

Show the public GitHub repository and open `part2/customer_segmentation`.

## 2. Dataset and Goal (40 seconds)

Open `data/Mall_Customers.csv` and explain:

- The dataset has 200 customer records.
- It includes customer ID, gender, age, annual income, and spending score.
- The goal is to discover customer groups without a target label.
- Annual income and spending score are used because they provide an interpretable two-dimensional segmentation.

## 3. Code Walkthrough (90 seconds)

Open `run_experiment.py` and briefly show:

- Data loading and validation.
- `StandardScaler` for comparable feature scales.
- The loop that evaluates `k=2` through `k=10`.
- Silhouette, Davies–Bouldin, and Calinski–Harabasz metrics.
- The final `Pipeline` containing preprocessing and K-Means.
- Saving CSV results, charts, summary JSON, and the trained model.

## 4. Live Execution (45 seconds)

From the repository root, run:

```bash
python part2/customer_segmentation/run_experiment.py
```

Point out that the script selects five clusters and prints a silhouette score of 0.5547.

## 5. Results and Interpretation (90 seconds)

Open the charts in this order:

1. `images/eda_overview.png`
2. `images/elbow_silhouette.png`
3. `images/customer_segments.png`
4. `images/cluster_profiles.png`

Explain in your own words:

"Five clusters gave the highest silhouette score among the tested values. The Premium Customers have both high income and high spending, while the Cautious Wealthy Customers have high income but low spending. Enthusiastic Shoppers have lower income but high spending. Budget-Conscious Customers are low on both measures, and Mainstream Customers are near the center and form the largest group."

Mention that the segment names summarize average behavior and should not be treated as facts about every individual.

## 6. Conclusion (30 seconds)

"This experiment reproduced the instructor's main clustering ideas in a smaller, reproducible workflow. ChatGPT and Codex helped create and debug the implementation, but I verified the execution, metrics, charts, and interpretation. The code, data, generated results, and documentation are available in my public GitHub repository."

After uploading the video, replace the placeholder YouTube link in this experiment's `README.md` and in the repository-level `README.md`.


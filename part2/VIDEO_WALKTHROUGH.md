# Part 2 Video Walkthrough — Experiments 00–05

**Presenter:** Weihao Fu  
**Suggested length:** 12–15 minutes

## Introduction

"Hello, my name is Weihao Fu. For CMPE 255 Assignment 1 Part 2, I used ChatGPT and Codex to create simplified, reproducible versions of instructor experiments 00 through 05. Each experiment includes code, data, generated results, charts, and documentation."

## Experiment 00 — Dynamic Todo Workspace

Show the task CSV and dashboard. Explain that operational task records can be grouped by status, category, and priority. Mention the 60 deterministic sample tasks and the 43.3% sample completion rate.

## Experiment 01 — NYC Taxi Trip Prediction

Show the taxi data and regression chart. Explain the fare target, distance and time features, common preprocessing pipeline, and comparison of three regression models. Random Forest achieved the lowest test RMSE of 3.171 and an R² of 0.925.

## Experiment 02 — Nano LLM Transformer

Show the corpus, attention code, loss curve, and generated sample. Explain character tokenization, positional embeddings, causal masking, next-character prediction, and autoregressive generation. State clearly that only the output projection is trained in this CPU-friendly teaching model.

## Experiment 03 — Customer Segmentation

Show the elbow/silhouette chart and colored segment plot. Explain standardization, testing `k=2` through `k=10`, selection of five clusters, and the five interpreted customer personas. The silhouette score is 0.5547, not classification accuracy.

## Experiment 04 — Associative Pattern Mining

Show the grocery CSV and market-basket chart. Define a basket as items purchased by one member on one date. Explain support, confidence, and lift, and note that association does not prove causation.

## Experiment 05 — Data Science Skills Lab

Show the three-model comparison and confusion matrix. Explain stratified splitting, five-fold cross-validation, final held-out testing, and feature interpretation. State that the educational breast-cancer model is not a clinical diagnostic system.

## Conclusion

"Together, these projects demonstrate operational analytics, regression, language modeling, clustering, association mining, and classification. ChatGPT and Codex accelerated implementation and debugging, while I ran the code, checked the metrics, and reviewed the limitations and interpretations."


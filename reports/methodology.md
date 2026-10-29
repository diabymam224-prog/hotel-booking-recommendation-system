# Research Methodology — Initial Plan

## Working title

**Hotel Booking Recommendation System Using Machine Learning: A Comparative Study of Data Preprocessing Techniques and Recommendation Algorithms**

## Aim

To develop and evaluate a hotel recommendation prototype and study how data preprocessing choices affect recommendation performance.

## Research questions

1. How effectively can recommendation algorithms rank relevant hotels using the selected public dataset?
2. How does a documented set of preprocessing techniques affect recommendation quality?
3. Which candidate algorithm performs best under one consistent evaluation protocol?
4. Do hybrid recommendations improve results over appropriate component baselines?
5. How do the findings compare with published hotel-recommendation research, considering dataset and protocol differences?

## Candidate preprocessing techniques

1. Missing-value handling.
2. Exact and key-based duplicate checks.
3. Domain-aware outlier inspection (do not automatically remove legitimate luxury properties).
4. Categorical feature encoding.
5. Review-text normalization and TF-IDF.

A minimally processed baseline and a consistently processed variant will be compared. Data splitting must happen before fitting learned transformations. Imputation values, encoders, vectorizers, and scalers must be fitted on training data only.

## Candidate recommendation approaches

1. Popularity baseline, using a documented score such as a Bayesian weighted rating when ratings are available.
2. Content-based ranking using hotel features and/or review text.
3. User-based collaborative filtering, only if stable user IDs and interaction data exist.
4. Matrix factorization, only if a suitable user-item interaction matrix exists.
5. Hybrid content/collaborative ranking, only when both component scores can be computed meaningfully; tune the mixing weight on validation data.

Dataset schema may require adapting this list. Do not fabricate users or interactions. If the chosen dataset cannot support all five methods, select a compatible dataset or explicitly revise the study before reporting results.

## Experimental protocol

- Inspect the dataset license, schema, identifiers, size, missingness, duplicates, and rating scale.
- Define the relevance rule before evaluation (for example, a rating threshold appropriate to the observed scale, or held-out positive interactions).
- Use a fixed train/validation/test split for comparable models; prefer temporal splitting if reliable timestamps and task semantics support it.
- Evaluate ranking recommendations against held-out relevant items.
- Use NDCG@10 as a primary ranking metric, with Precision@10, Recall@10, and Hit Rate@10 as secondary metrics when appropriate.
- Consider catalog coverage and inference latency as additional system measures.
- Use RMSE/MAE only for explicit rating prediction, not as a replacement for ranking metrics.
- Record actual outputs and random seeds. Do not invent or pre-fill results.
- Compare published findings qualitatively unless dataset, split, candidate set, relevance definition, and metric are sufficiently comparable.

## Current limitation

This is a plan, not a completed experiment. Dataset choice, exact schema, relevance threshold, split, trained models, measured scores, and deployment configuration remain to be confirmed.

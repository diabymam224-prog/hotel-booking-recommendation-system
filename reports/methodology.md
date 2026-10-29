# Research Methodology — Initial Plan

## Working title

**Hotel Booking Recommendation System Using Machine Learning: A Comparative Study of Data Preprocessing Techniques and Recommendation Algorithms**

## Aim

To develop and evaluate a hotel recommendation prototype and study how data preprocessing choices affect recommendation performance.

## Selected dataset

The initial dataset is **ViHoRec — Vietnamese Hotel Recommendation Dataset**. It provides interaction records with pseudonymous user IDs, hotel IDs, ratings, dates, and source, plus a hotel catalogue with hotel names and locations. It is small enough for iterative development in Google Colab. Its scope is Vietnamese hotels, and its catalogue attributes are limited; both limitations must be stated in the final report. The released data are marked CC BY-NC 4.0, so review license terms before any non-academic or commercial use.

Source repository: https://github.com/MinhNguyenDS/ViHoRec  
Dataset descriptor: https://arxiv.org/abs/2607.12946

## Research questions

1. How effectively can recommendation algorithms rank relevant hotels using the selected public dataset?
2. How does a documented set of preprocessing techniques affect recommendation quality?
3. Which candidate algorithm performs best under one consistent evaluation protocol?
4. Does a hybrid approach improve results over appropriate component baselines?
5. How do the findings compare with published hotel-recommendation research, considering dataset and protocol differences?

## Candidate preprocessing techniques

1. Missing-value handling.
2. Exact and key-based duplicate checks.
3. Domain-aware outlier inspection (ratings should be validated against the dataset's observed scale; legitimate high or low ratings must not be removed automatically).
4. Categorical feature encoding.
5. Text normalization and TF-IDF over available hotel name/location text.

Compare a minimally processed baseline with documented preprocessing variants. Data splitting must happen before fitting learned transformations. Imputation values, encoders, vectorizers, and scalers must be fitted on training data only.

## Candidate recommendation approaches

1. Popularity baseline, using a documented score such as a Bayesian weighted rating.
2. Content-based ranking using available hotel names and locations; acknowledge that the catalogue has limited descriptive features.
3. User-based collaborative filtering from genuine user–hotel rating interactions.
4. Matrix factorization for the user–hotel rating/interaction matrix.
5. Hybrid ranking combining content and collaborative scores, with its mixing weight tuned on validation data.

## Experimental protocol

- Audit schema, source license, row counts, missingness, duplicates, rating scale, and user/item interaction counts.
- Define relevance before evaluation. A rating threshold such as 7/10 may be explored because the observed scale appears to be 1–10, but confirm the distribution and justify the threshold in the notebook.
- Prefer temporal holdout using the date field where feasible. Ensure held-out interactions are not used in training or feature fitting.
- Use the same candidate set, split, and relevance rule for comparable models.
- Use NDCG@10 as a primary ranking metric, with Precision@10, Recall@10, and Hit Rate@10 as secondary metrics where appropriate.
- Consider catalog coverage and inference latency as additional measures.
- Use RMSE/MAE only for explicit rating prediction, not as a replacement for ranking metrics.
- Record actual outputs and random seeds. Do not invent or pre-fill results.
- Compare published findings cautiously because datasets and evaluation protocols may differ.

## Limitations to report

- Hotel catalogue metadata are limited to hotel name and location in the basic released table.
- The dataset covers Vietnamese hotels and does not provide live availability, booking inventory, or current prices.
- Pseudonymous user IDs are dataset keys, not verified real-world identities.
- Results on this dataset do not automatically generalize to all destinations or booking platforms.

## Current status

Dataset selected; code scaffold created. Data audit, final split, relevance threshold, trained models, measured scores, and deployment configuration remain to be completed.

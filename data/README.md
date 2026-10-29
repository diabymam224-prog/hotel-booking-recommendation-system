# Selected dataset: ViHoRec

For the first implementation, use **ViHoRec — Vietnamese Hotel Recommendation Dataset**. It is a manageable public research dataset with actual user–hotel ratings, dates, and a hotel catalogue, so the planned collaborative-filtering experiments are feasible without inventing interactions.

- Dataset repository: https://github.com/MinhNguyenDS/ViHoRec
- Dataset descriptor: https://arxiv.org/abs/2607.12946
- License: CC BY-NC 4.0 for the released data (non-commercial use only; review the license before reuse or broader deployment).
- Released tables: `interactions.csv` (user_id, hotel_id, rating, date, source), `hotels.csv` (hotel_id, name, location), and `users.csv`.
- Dataset scope: hotels in Vietnam; recommendations and conclusions must be described with that scope in mind.

The Colab notebook downloads the released interaction and hotel CSVs from the dataset's public repository. The raw data are not copied into this project repository.

## Why this dataset

It provides stable pseudonymous user identifiers, hotel identifiers, explicit ratings, dates, and basic hotel catalogue metadata. This supports a popularity baseline, content-based ranking from available hotel text/location, user-based collaborative filtering, matrix factorization, and a hybrid model. The catalogue has limited descriptive attributes, so content-based recommendations are constrained by hotel names and locations; document this limitation.

## Local data layout

If you choose to download the data manually, place files under `data/raw/` (this directory is ignored by Git):

```text
data/raw/interactions.csv
data/raw/hotels.csv
```

## Data-handling and research rules

- Do not commit raw data unless permitted by the dataset license and project requirements.
- Treat user IDs as pseudonymous dataset keys, not real-world identities.
- Verify column names and rating range in the notebook before modeling.
- Do not fabricate interactions or user identities.
- Split data before fitting learned transformations; use a documented, reproducible evaluation protocol.
- This dataset is not a live hotel inventory. The app must not claim real-time room availability, current prices, or booking capability.

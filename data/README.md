# Dataset instructions

Do not upload raw datasets to this repository unless the dataset license explicitly permits redistribution and the files are small enough for GitHub.

## Dataset requirements

Before finalizing the dataset, inspect its license, schema, row count, and available identifiers. For the planned research, ideally the data should include:

- A stable hotel/item identifier
- Hotel attributes such as name, location, rating, price, or amenities
- Review text and/or hotel ratings
- A stable reviewer/user identifier and user–hotel interactions if collaborative filtering is to be evaluated

## Candidate datasets to investigate

1. **HotelRec** — hotel recommendation interactions/reviews; verify the authors' download instructions, schema, terms, and access before use.
2. **515K Hotel Reviews Data in Europe** — useful hotel metadata and review text, but the commonly distributed schema does not provide a stable reviewer ID. It may support content-based and rating-based experiments, but should not be assumed sufficient for user-based collaborative filtering.
3. **Expedia Hotel Recommendations** — contains user search/booking context and hotel-cluster targets, but its target is a hotel cluster rather than a directly named hotel catalogue. Confirm whether this fits the project before selecting it.

## Proposed local layout

```text
data/
  raw/          # downloaded source files; ignored by Git
  processed/    # derived files; ignored by Git
  README.md
```

## Important

Do not create synthetic user IDs or pretend that nationality is a user identifier. If a dataset lacks user-item interactions, either choose a suitable interaction dataset or revise the collaborative-filtering experiment transparently.

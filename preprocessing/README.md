## Naming the `label` component of a recipe

The `label`/name segment inside `preprocessing_name` is free-form (see
DESIGN.md and PROTOCOL.md section 5), but should always make explicit the
two properties that most affect whether downstream results are comparable:

- **Reference scheme** used (e.g. CAR, Bipolar, Laplacian)
- **Effective output sampling frequency** after any decim/resampling
  (e.g. 250Hz)

Example: `10ICA-CAR-250Hz` tells you the cleaning method, the
reference, and the rate without opening config.json.
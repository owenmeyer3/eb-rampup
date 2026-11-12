# Test Priorities: Loss Ratio Project

Nov 11, 2026 · @Owen

Rank tests by how bad the failure they catch is, and whether anything else would catch it. The most valuable guard against failures that are silent (the code runs, but the answers are wrong) and costly (bad prices, a bad model in production).

## Essential tests

Seven groups of tests guard against failures that would be both silent and costly. These stay no matter how the suite changes.

| Priority | Tests | Guards against |
| --- | --- | --- |
| 1 | `test_columns_match_contract`, `test_single_industry_batch_still_has_all_columns`, `test_feature_mismatch_is_caught`, `test_save_and_reload_gives_same_predictions` | A model receiving the wrong inputs and returning confident, wrong numbers with no error: the most dangerous failure in ML |
| 2 | Promotion rule tests | A worse model reaching production through an off-by-one at a boundary: a tie, just under the margin, or worse than the book rate |
| 3 | `test_registration_is_gated` | The pipeline registering a model without passing the champion gate |
| 4 | Data validation tests | Bad batches reaching training and every step after it |
| 5 | Business rule boundary tests | Wrong referral decisions at exactly 10 lives or exactly 1.5, where business disagreements happen |
| 6 | API input validation (the 422s) and the live smoke test | A broken contract with callers, and a bad deploy reaching production |
| 7 | Model quality tests vs. baselines | A training run that "works" but learned nothing useful |

## Lower-value tests

These are cheap to keep, but they'd be the first to go if the suite got slow or noisy.

- **Most of the data generator tests.** In production there's no generator: data comes from the warehouse. Several checks (ranges, nulls) are now duplicated by `validate_groups`, which tests the real thing.
- **`test_load_table_reads_requested_table`.** It checks a one-line pass-through, using a mock that mirrors the implementation. It would rarely catch anything real.
- **Tests on how code calls libraries,** like asserting the exact `DMatrix` passed to the booster. They break on refactors without catching real bugs.
- **The training smoke test.** It overlaps with the quality tests and `test_main_runs_end_to_end_locally`.

## Tests to add

The first three close the gap between "passes on clean test data" and "works on real data," which is where production ML most often breaks.

- [ ] **BigQuery-shaped data types.** Probably the most likely real bug in this project. Tests use DataFrames from `generate_groups`, with plain NumPy types, but data loaded from BigQuery arrives with different pandas types (nullable `Int64` for integers, for example). Run `build_features` on a frame converted to those types and check the features are identical.
- [ ] **API vs. batch parity.** Send the same group through the API (`Group` → one-row DataFrame) and through batch scoring (a BigQuery-shaped DataFrame), and assert the features and predictions match exactly. It's the training–serving skew guard, applied to the two serving paths.
- [ ] **A golden snapshot of features and predictions.** Save the exact features and predictions for `tiny_groups` from a fixed model, and assert they don't change. Any unintended change, from a refactor or a library upgrade (like the scikit-learn surprise), shows up immediately. Intended changes update the snapshot in a reviewed commit.
- [ ] **Behavioral sanity checks on the trained model.** Holding everything else equal: an older workforce shouldn't get a lower predicted loss ratio, a riskier industry shouldn't get a lower prediction, and a larger group's prediction should move closer to its own experience. These catch models that score well on MAE but learned something nonsensical. Run them in CI on a small model, and as another gate in the pipeline before registration.
- [ ] **Allowed features.** Which rating factors are permitted varies by state and product, and a group's gender mix is the kind of factor that gets regulatory attention. A test asserting `FEATURE_COLUMNS` contains only factors from an approved list makes CI enforce the compliance rule. Whether `pct_female` is acceptable is a question for actuarial and compliance, not something to assume.

## How many tests is right

There's no target count. The current suite of about 60–70 tests is reasonable for a project this size. Aim for these instead:

- **Every silent, costly failure has a test.** That's the essential list above.
- **The unit suite stays fast,** a few seconds, so people run it constantly.
- **Each test fails for one clear reason,** so a failure says what broke.

The best trade from here: drop some generator tests and add the data-type, parity, and golden tests.

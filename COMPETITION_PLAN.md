# NEXUS OptiML 2026 — data-grounded competition plan

Prepared from the seven local CSV files on 13 October 2026. The supplied Kaggle overview and schema were also reviewed. Kaggle's overview, data and rules pages could not be fetched through the browser, so the live deadline, submission allowance, external-data policy and final-selection rules remain unverified. The five-day schedule below is relative to the time remaining shown in the supplied overview.

**Recommended approach:** build a reproducible, order-aware temporal validation pipeline; start with a categorical gradient-boosting model; emphasize product, store, customer and basket information; explicitly challenge the fading promotion signal; then add strictly historical statistics and a small validated ensemble. A random split and a model dominated by historical promotion effects are particularly poor choices for this dataset.

This document is a plan backed by full-file profiling and simple temporal ranking diagnostics. No CatBoost/LightGBM model has been trained, no competitive AUC is claimed, and no submission has been uploaded.

## 1. What the files actually contain

| File | Rows | Columns | Verified interpretation |
|---|---:|---:|---|
| train.csv | 1,400,005 | 27 | Order lines, including binary `returned` |
| test.csv | 479,995 | 26 | Same input columns, no target |
| customers.csv | 204,930 | 9 | 201,901 distinct customers; 3,029 IDs have two records |
| products.csv | 48,677 | 12 | Unique product IDs |
| stores.csv | 454 | 8 | Unique store IDs |
| calendar.csv | 132 | 7 | Unique weekly dates |
| sample_submission.csv | 479,995 | 2 | `order_line_id,returned`; IDs already match test row order |

The training set contains **252,737 returned lines, or 18.0526%**. There are 444,024 training orders and 151,459 test orders. Order IDs and line IDs do not overlap between train and test. Every transaction customer, product and store ID matches its supplied dimension table. No duplicate line IDs or exact duplicate transaction rows were found.

### Important corrections to the description

The actual customer fields are `region`, `age_band`, `newsletter_opt_in`, `preferred_contact`, `record_updated_at` and `loyalty_tier`; there is no `gender` or `location_code`. Products have `supplier_code`, dimensions, color and packaging, but no `brand`. Stores use `format` and `size_sqm`, not the described `store_size`. Calendar uses `unemployment`, not `unemployment_rate`, and contains no `season` field. Features must be built from the real headers, with any season convention explicitly defined.

The transaction table has useful fields omitted from the abbreviated schema: `tz_hint`, `ship_mode`, `discount`, `freight_cost`, `line_no`, `gift_wrap`, `promo_channel_code`, `return_window_days`, `warehouse_zone`, `picker_shift`, `banner_id`, `session_clicks`, `device_type` and `coupon_code`.

### The most consequential discovery: conditional drift

The overall return rate is fairly steady, but the relationship between promotion and returns changes dramatically.

| Period | P0 return rate | P5 return rate |
|---|---:|---:|
| January 2024 | 8.97% | 44.40% |
| April 2024 | 9.47% | 41.40% |
| July 2024 | 12.73% | 33.45% |
| October 2024 | 16.22% | 24.90% |
| December 2024 | 18.01% | 19.35% |

These are observed associations, not causal promotion effects. They show why checking only the global target rate or train/test feature distributions is insufficient. Whether the conditional relationship continues to weaken in test cannot be established without test labels.

Using the fixed ordinal score P0 < P1 < ... < P5, promotion-only monthly AUC falls from **0.7024 in January 2024 to 0.5056 in December 2024**. The latter is close to random ranking. This score uses the code ordering directly; it is distinct from the historical category-rate lookup diagnostics below.

Simple historical category-rate lookups confirm that promotion's usefulness weakens near the deployment period:

| Feature used alone | Validation from July 2024 | Validation from October 2024 |
|---|---:|---:|
| Store ID | 0.5979 | 0.6094 |
| Product ID | 0.5963 | 0.5989 |
| Customer ID | 0.5684 | 0.5731 |
| Customer type | 0.5539 | 0.5556 |
| Promotion code | 0.5606 | 0.5296 |
| Banner ID | 0.4955 | 0.4939 |
| Coupon code | 0.5006 | 0.5003 |

Method: estimate each category's return rate using only rows before the cutoff, with 100 pseudo-observations at the earlier training mean; score all later labeled rows. These are preliminary single-feature AUCs, not boosted-model scores, feature-importance estimates or incremental gains. Cutoffs use raw recorded timestamps, and the later blocks include 14 boundary rows dated 1 January 2025. Customer-type diagnostics use the latest supplied profile; that field agrees across the duplicate profiles observed.

**Priority implication:** spend the early modeling budget on stable entity information and robust validation, while testing promotion inclusion explicitly. Do not assume the historically strongest-looking raw feature will remain useful.

## 2. Data risks and their concrete treatment

### Customer duplicates and snapshot timing

All 3,029 duplicate customer IDs have differing update dates. Raw region values differ for 2,991 IDs; loyalty tier differs for 2,079. The other customer fields agree across these duplicate records. Some raw region differences may disappear after normalization.

A normal join to the undeduplicated customer table will multiply transaction rows. Implement and compare two documented join policies:

1. **Supplied-snapshot policy:** choose one deterministic record per customer, normally the latest `record_updated_at`, and use it consistently for train and test. Treat this as use of the supplied reference snapshot, not as a historically accurate reconstruction.
2. **As-of policy:** for each order, use the most recent customer record whose update date is no later than the order time. When no eligible record exists, keep mutable fields unknown. Do not silently substitute a future record. Stable fields can be considered separately with their assumptions documented.

The latest record date is after the transaction on 1,288,365 training rows (92.03%) and 100,370 test rows (20.91%). Consequently, a strict as-of view will have very different coverage. Compare snapshot, as-of and mutable-field-exclusion variants on the same temporal blocks. Retain a coverage flag for diagnostics; do not automatically give snapshot-update metadata to the prediction model.

Use `many_to_one` validation for ordinary dimension joins. For an as-of implementation, verify one output row per input line and effective-date ordering. No unresolved duplicate key should be allowed into feature generation.

### Mixed timestamp meaning and boundary rows

Raw timestamp ranges are:

- Train: 31 December 2022 16:01 through 1 January 2025 02:47.
- Test: 31 December 2024 17:31 through 1 July 2025 23:31.
- There are 7,111 test rows on 1 July 2025, as well as 26 test rows on 31 December 2024.

Approximately 20% of `tz_hint` is missing. Nonmissing hints are strongly tied to channel: online/marketplace are marked UTC; catalog/store are marked local. Raw timestamps cluster at hours 00, 03 and 23. Those patterns warrant investigation; they do not establish a reliable timezone correction.

Preserve the original timestamp and hint. Maintain raw calendar features. Only create a converted local/UTC view when the timezone can be justified; broad regions such as US-West or Asia Pacific do not uniquely identify it. Keep missing/ambiguous cases explicit. The preliminary audit parsed all timestamps into a common container to inspect ranges, without proving that every original value was UTC.

Select one canonical order time for fold assignment, with a documented rule such as the earliest reliable time. Flag orders with inconsistent line timestamps: 1,727 training orders and 570 test orders have more than one raw timestamp. If an order straddles a cutoff, keep it together or remove it from that boundary comparison. Avoid fitting a timezone correction using labels.

Retain every test row, including July rows. Do not force dates into the advertised month range. Inspect the July cohort by order, customer, channel and basket size to determine whether the large count reflects a few large orders, timestamp issues or an extra boundary day.

### `order_value` behaves more like a line amount

`order_value` varies within 201,829 training orders and 68,311 test orders. The median ratio `order_value / (unit_price * quantity)` is approximately 1.0. About 62.77% of training lines have `order_value` within 0.5 of that product, compared with only 24.29% within 0.5 of the discount-adjusted product.

Working hypothesis: this field often represents a rounded line amount rather than the advertised order total. Validate residual distributions by channel and product before adopting that interpretation. Do not take its first value per order as basket value.

Keep the raw field, construct an explicitly named `line_gross_candidate = unit_price * quantity`, and evaluate candidate basket sums separately. The low tail of the amount ratio includes values near 0.01, and unit-price/catalog ratios have extreme outliers. Investigate possible scale corruption using several corroborating fields; do not automatically divide high prices by 100.

### Basket counts and large orders

The observed number of lines equals `items_in_order` for every order in both files. `line_no` never exceeds that count, and `(order_id,line_no)` is unique. Thus `items_in_order` is verified as the line count, not total units.

The median order has one line, but the largest has 704 training lines or 748 test lines. When measured over prediction rows, mean `items_in_order` rises from 41.43 in train to 52.73 in test. Report both order-level and line-level distributions; large orders can dominate a row-weighted metric and naive uncertainty estimates.

Do not collapse the problem to one row per order. Do not use any sibling line's return outcome as a feature. Basket input aggregates are appropriate if the basket is available at prediction time, which is a reasonable working assumption for this supplied order-line task.

### Dates in supplementary tables

301,893 training lines precede their product's listed launch date, while no test lines do. There are also 212 training lines and 13 test lines before their customer's signup date under the raw-date comparison. No store opening date follows an order in the audit.

Create diagnostic signed ages and inconsistency flags, but compare product-age exclusion and guarded-age variants. A pre-launch flag is absent in test and may encourage a historical shortcut. Clipping an impossible age to zero alone would hide the problem. Do not delete 21.6% of training data because of the launch-date inconsistency.

`review_count`, `seller_rating`, `loyalty_tier` and `manager_tenure_years` are also potentially changing snapshots. Their observation dates are not fully supplied. They are not proven leakage merely because they are snapshots, nor proven historically available merely because the CSV includes them. Record provenance assumptions and test their incremental value. Seller rating has 453 unique values over 454 stores and can effectively identify stores; apparent signal need not represent a generalizable rating effect.

### Categories and missingness

Product categories have 60 raw spellings, including `Home Decor`, `HOME_DECOR`, ` home_decor ` and `home_ecor`, consistent with roughly 12 intended categories. Customer regions have 101 raw spellings versus 22 store-region values.

Build explicit, reviewed normalization maps: trim spaces, normalize case/separators, and map clear misspellings. Preserve raw values and a normalization flag for ablation. Avoid unrestricted fuzzy matching that could merge US-East and US-West.

| Training field | Missing share | Treatment |
|---|---:|---|
| tz_hint | 20.02% | Explicit unknown; preserve ambiguity |
| payment_method | 3.01% | Missing categorical level |
| ship_mode | 4.03% | Missing categorical level |
| discount | 3.06% | Numeric missing plus flag; do not equate with zero |
| freight_cost | 4.05% | Numeric missing plus flag |
| session_clicks | 7.41% | Numeric missing plus flag |
| coupon_code | 88.12% | Coupon-present indicator plus optional raw code |

Missing session clicks have a 20.07% return rate versus 17.89% for observed values in the full training set. Validate this relationship over time. Store size is missing for 348 of 454 stores; inspect channel dependence before imputing a physical-store median into online storefronts. Products have 2,082 missing weights; customer age band is missing on 16,334 profile rows.

Use missing-aware tree inputs. For categorical fields use a consistent string sentinel. For numeric imputation or clipping, fit thresholds on the training fold only. Preserve outlier flags and compare raw/log versions before deciding to cap valid extreme orders.

## 3. Validation design

The prediction unit and evaluation weight are order lines. The deployment direction is forward in time. Existing customers and products may legitimately occur in both sides of a temporal split, but the same order must not.

| Validation view | Fit period | Evaluate period | Purpose |
|---|---|---|---|
| Earlier seasonal check | Through December 2023 | January–June 2024 | Match the nominal test half-year |
| Main six-month backtest | Through June 2024 | July–December 2024 | Match the forecast horizon |
| Recent-period check | Through September 2024 | October–December 2024 | Detect fading relationships |
| Monthly diagnostics | Past-only fit or fixed block fit | Each held-out month | Reveal average-score concealment |

Raw-cutoff counts are 1,048,082 fit rows and 351,923 later labeled rows at July 2024, and 1,220,975 fit rows and 179,030 later labeled rows at October 2024. Those later counts include the 14 January 2025 boundary rows. Finalize exact inclusion after timestamp/order rules. The raw cutoffs examined currently have zero overlapping orders.

The six-month and recent checks overlap; they are sensitivity views, not independent folds. Do not average their AUCs as if independent. The latest months have already been examined for data understanding, so they are not a pristine untouched test set. Limit repeated tuning on them and track every experiment.

Use ROC-AUC as the primary selection metric, with overall block AUC and monthly AUC both reported. Pooled AUC and average monthly AUC are different quantities. Also record log loss, probability quantiles and calibration summaries to identify extreme outputs, without optimizing them at the expense of the stated ranking objective.

Report AUC for channel, customer type, basket-size bands, seen/unseen customers, seen/unseen products and missing-data cohorts, alongside row counts and positive counts. Do not report AUC for a cohort containing only one target class.

For uncertainty, compare paired predictions on identical rows. Bootstrap whole orders, and use week-block sensitivity where practical, rather than treating all lines as independent. Preserve the official unweighted line-level AUC inside each sample. Do not switch the main metric to equal weight per customer or order.

Standard shuffled cross-validation does not reflect a future prediction task. Use explicit calendar boundaries because the data are unevenly distributed in time. This follows the temporal-validation principles described in the [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

### Label availability and return windows

An order's outcome may not be known when the order is placed. No return event timestamp is supplied. Distinguish the competition's assumption that historical labels are provided from a production claim that every past label was already observable at each cutoff.

Use two checks: the ordinary competition temporal split, and a maturity sensitivity that excludes fit rows whose `order_ts + return_window_days` reaches beyond the prediction cutoff. A 30/60-day conservative gap is another sensitivity option, not a known property of this dataset. Return windows may start at delivery, so this is an approximation until clarified.

Never update target-derived validation features using earlier validation labels during the six-month forecast. Those labels would also be unavailable for test. If a rolling-label experiment is run for a different operational scenario, label it separately and do not use its score as the competition estimate.

## 4. Feature engineering, in priority order

Start with a compact set and introduce one family at a time. Keep a feature manifest recording its inputs, units, join rule, availability assumption and fold-fitting requirements.

### A. Strong baseline inputs

- Clean transaction numerics and categories, including missing indicators.
- Product category, list price, dimensions, weight, packaging, supplier and color.
- Customer type, cleaned region, age band and signup-based tenure, subject to date checks.
- Store ID, format, channel, region and opening-based age.
- Calendar variables joined to a verified Monday week start.
- Explicit comparisons with/without promotion and questionable snapshot/age fields.

Keep order-line and order IDs for joins, folds and output alignment. Exclude them as raw predictors. Treat product, customer and store IDs as categorical candidates, with history/frequency features and cold-start fallbacks. Treat `banner_id` as a category rather than assuming that its integer magnitude is meaningful. Promo P0–P5 should start as categories; their historical ordinal relationship may disappear.

### B. Price, quantity and freight relationships

Create `unit_price * quantity`, `unit_price / list_price`, `1 - unit_price / list_price`, and the residual between that implied discount and the recorded discount. Keep the possibility that unit price already reflects discounts; do not label a double-discounted calculation as actual net revenue.

Other candidates: freight per unit, freight as a fraction of the candidate line amount, unit price per kilogram, quantity-weighted product weight, `order_value` versus candidate gross residual, log1p price/quantity/amount/freight, and bulk-quantity bands.

Guard zero or missing denominators explicitly. Ratio features should not silently become infinite or substitute zero for unknown. Investigate scale-anomaly flags separately from proposed corrected values. Fit learned thresholds only inside each fold.

### C. Basket composition

Build per-order line count, total units, distinct products/categories/suppliers/stores, category shares, category entropy, maximum category share, candidate basket value, average and maximum line value, price spread, mean discount, fraction discounted, total estimated weight/volume, and repeated-product indicators.

Build each line's relative price rank, fraction of candidate basket value, fraction of total units, normalized line position, and whether it is unusually expensive within its basket. Compare basket sums from `order_value` and reconstructed line gross after validating semantics. Use names that identify which candidate was used.

Customer and channel are consistent within every order observed. Store ID is inconsistent in 698 training and 367 test orders, so do not overwrite line-level stores with an arbitrary first store. Number of stores in an order may itself be informative.

### D. Product and fulfillment

Volume in cubic centimeters, largest dimension, dimension ratios, density, quantity-weighted volume/weight, price relative to category median, and interactions of physical properties with shipping mode, channel and packaging.

Use category-derived descriptive indicators only where supported by the catalog. For example, a bulky-product feature can be based on dimensions; an invented fragile-product label needs a stated heuristic and separate validation.

### E. Customer and store context

Signup tenure and guarded store age; customer versus store country/region match after normalization; customer type × quantity; customer type × basket size; customer type × category; store format × shipping mode; channel × return window.

Avoid inventing exact geographic distances from broad region labels. Treat cross-region mismatch as a proxy whose meaning may differ between local retail and marketplace sales.

### F. Time and calendar

Month, weekday, hour, weekend, day of month, cyclical month/hour representations, calendar holiday flag, CPI, fuel price, fuel surcharge, temperature and unemployment. Verify all weekly joins remain within the intended week, especially around boundary timestamps. The raw-date audit found full calendar coverage.

Separate seasonality from monotonic time. Compare models without absolute date/index features, because trees do not automatically extrapolate a fading categorical effect beyond the training range. Calendar macro variables may also act as time proxies. Observed weekly economic/weather fields have unclear publication timing; their use is a supplied-data assumption, not proof they were available at checkout.

### G. Non-target history

Past customer order/line counts, recency, spend and quantity summaries, category mix, channel preferences, recent versus lifetime activity, product sales counts, store transaction volume, and prior customer-product purchases.

Compute past order counts on an order table, not on repeated order lines. Exclude the current order from customer history and avoid arbitrary row ordering among simultaneous transactions. Compare 30/90/180-day windows where event density supports them.

The safest baseline freezes history at the forecast cutoff. A separate sequential-covariate variant may update counts with earlier validation/test transactions, without labels, if the intended information setting allows this. Mirror the same protocol in validation and inference. Do not casually use all future test transactions to compute features for earlier test rows.

## 5. Historical target statistics

These merit focused effort because simple product/store/customer lookups already carry forward-period ranking information.

For a group g, estimate:

`smoothed_rate(g) = (past_returned_lines(g) + alpha * past_global_rate) / (past_lines(g) + alpha)`

Keep the count alongside the rate so the model can distinguish strong evidence from a prior-driven estimate. Test modest smoothing grids such as 20, 100 and 500 on forward folds; these are starting choices, not fitted optima.

Priority groups: store, product, customer, supplier, cleaned category, customer type, product × channel, category × customer type and store × category. Add sparse interactions only when counts and validation gains justify them. Product × customer is likely sparse; use support thresholds and hierarchical fallbacks.

Use distinct measures for returned lines per historical line and orders with any return per historical order. Do not divide a returned-line numerator by an order-count denominator and call it a probability.

Training features must exclude the current target and all current-order targets. Build them with forward blocks or expanding historical calculations, applying the chosen label-maturity rule. Fit each block's global prior using only eligible earlier labels. Use a fixed prior or omit a burn-in block when no eligible history exists. A naive full-training `groupby(...).mean()` merged back into its own rows is invalid.

For a validation block, freeze the mapping at the fit cutoff. For final test inference, build mappings using the permitted labeled training history only. Never add test predictions as if they were observed return labels. Compare the train/forecast history-count distributions so early training rows with little history do not create an unnoticed feature-distribution problem.

Fallback hierarchy examples: product → category → global; customer → customer type/region → global; store → channel/format → global. Parent priors and fallback levels must also come only from the fitting data. Keep explicit unseen indicators.

12.26% of test rows have customer IDs never seen in training; only 0.182% have unseen products, and eight rows have unseen stores. That makes product/store history promising while customer cold-start behavior remains a material evaluation slice. All these entities still have supplied profile records, so a new customer is not necessarily missing demographic/context information.

## 6. Modeling sequence and starting configurations

### Baseline 0: pipeline integrity

Create a constant training-prevalence prediction to validate submission writing; its AUC is 0.5 when both classes occur. Keep the simple store/product-rate lookups as diagnostic benchmarks. Confirm row alignment, feature availability and fold boundaries before expensive training.

### Baseline 1: CatBoost

Recommended first full model because the data mix numerical and categorical inputs, including recurring entity IDs. Begin with `loss_function='Logloss'`, `eval_metric='AUC'`, depth 6–8, learning rate 0.03–0.08, 2,000–5,000 maximum iterations, L2 regularization around 3–10, fixed seed, and validation-based stopping with roughly 150–250 rounds of patience. These are proposed search ranges, not measured best settings.

Use explicit categorical columns and consistent missing sentinels. Start with modest categorical interaction complexity to contain memory. Compare standard internal ordering against chronological ordering where appropriate. CatBoost's `has_time` preserves input ordering during relevant internal operations; it does not replace external temporal splits, label-delay handling or order-safe historical features. See the [official common-parameter reference](https://catboost.ai/docs/en/references/training-parameters/common).

Train matched variants with promotion included/excluded, and with entity IDs included/excluded. The no-promotion candidate deserves early attention given the measured drift. Avoid imposing monotonic constraints on promotion, quantity or discounts without evidence of a stable relationship.

### Baseline 2: LightGBM

Use the same folds and a documented categorical encoding, plus the explicit historical features. Suggested starting ranges: 31–63 leaves, learning rate 0.03–0.07, minimum leaf data 100–500, feature fraction 0.7–0.95, row sampling 0.7–0.95 with positive bagging frequency when enabled, and L2 regularization 1–20. Use binary loss, AUC monitoring and early stopping.

Keep fit/validation/test category mappings consistent and handle unseen categories explicitly. High-cardinality IDs need regularization and ablation. Leaf count and minimum leaf data are central capacity controls in the [official LightGBM tuning guide](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html).

### Optional comparator

A regularized logistic model with transformed numeric fields and low-cardinality one-hot features can expose simple interactions or data errors. A second boosting family or additive model should earn its place by validation value or ensemble diversity. Deep neural networks, graph models and extensive AutoML searches are low priorities within five days unless simpler approaches plateau and compute is available.

The class balance is about 18:82. Start with ordinary unweighted binary loss. Do not default to SMOTE, majority downsampling or aggressive class weights. Test weights only as a measured ablation, since they can alter ranking and probability behavior.

The current default Python lacks pandas; a bundled runtime successfully supplied pandas/numpy for this audit, but its CatBoost, LightGBM, scikit-learn and pyarrow packages are absent. For modeling, create an isolated project environment or use an appropriate Kaggle runtime, pin dependencies, and record hardware/package versions. Avoid modifying a shared bundled runtime.

## 7. Experiments for temporal robustness

Run matched experiments rather than changing many things at once:

1. Full-history model without promotion versus the same model with promotion.
2. Full-history versus trailing 12-month, 6-month and, if support is sufficient, 3-month fit windows. A shorter window is chosen relative to each fold's fit cutoff.
3. Full-history training with exponential recency weights, for example half-lives of 90, 180 and 365 days. Compare with unweighted fits.
4. Promotion × time interaction model. Expect it to learn historical changes, but do not assume a tree will extrapolate those changes into unseen months.
5. Historical rates using lifetime versus recent eligible labels, with minimum support and shrinkage.
6. Exclusion of launch-age, profile-update metadata, ratings/reviews and mutable customer fields, individually or in interpretable groups.
7. Noise-feature ablation for banner, coupon ID, warehouse zone and picker shift; weak marginal AUC alone does not prove absence of useful interactions.

Inspect feature-response stability by month, channel and customer type. Changes in group composition can create apparent effect drift, so compare promotion's incremental predictive contribution after controlling for the other inputs. Do not blindly extrapolate P5 below P0 or manually reverse their ordering.

Use adversarial validation only as a covariate-shift diagnostic: classify earlier versus later unlabeled rows and inspect the distinguishing features. Exclude obvious time/ID shortcuts in a second version. It cannot detect all changes in `P(returned | features)`, and a strong time classifier is expected here. Test importance weighting only if it improves the actual forward return-prediction task.

## 8. Experiment tracking and ensemble selection

| Stage | Change | Decision it resolves |
|---|---|---|
| E00 | Constant and historical lookup predictions | Correct metric and output plumbing |
| E01 | Clean joined baseline with/without promotion | Baseline strength and drift exposure |
| E02 | Basket, price and physical-product features | Value of transaction relationships |
| E03 | Customer/store non-target histories | Value of repeated behavior |
| E04 | Forward-built return statistics | Value of recurring entity risk |
| E05 | Snapshot and product-age ablations | Reliance on questionable reference timing |
| E06 | Recency windows and weights | Adaptation to late-period behavior |
| E07 | Second boosting family | Independent predictive value |
| E08 | Limited seeds and blending | Stable, reproducible final improvement |

For every run store: data fingerprint, feature manifest/version, join policy, fit/validation dates, availability/maturity policy, model configuration, seed, runtime, peak memory if measurable, overall/monthly/cohort AUC, best iteration and prediction-file path. Save validation predictions keyed by line ID.

Keep changes that improve relevant temporal comparisons without an unacceptable late-period or cold-start regression. Require gains larger than plausible validation noise; use paired order/block uncertainty for close decisions. Do not impose an arbitrary promised AUC target before a real baseline exists.

Blend only genuinely complementary models evaluated on identical held-out rows. Start with a coarse convex weight grid, for example CatBoost weight 0, 0.25, 0.5, 0.75 or 1. Inspect prediction correlation and monthly benefit. Probability averaging is the default; a rank blend is an optional ROC-AUC experiment, but its output should not be described as calibrated probability. Avoid large weight searches on one small holdout.

Do not concatenate overlapping validation predictions and call them independent out-of-fold predictions. If stacking is justified, generate nonoverlapping forward predictions for the meta-model and retain a later assessment block. A simple blend is the preferable five-day default.

Final refit: use all permitted labeled data or the chosen trailing window, with the selected features and historical-statistics policy. Choose the final tree count from backtest stopping behavior because the test has no labels for early stopping. A small number of seeds is reasonable after feature/model choices stabilize. Do not use public leaderboard scores as a replacement validation objective.

## 9. Five-day execution schedule

Confirm the actual closing time and submission limits on Kaggle before allocating the final day. Do not derive a calendar deadline from the relative countdown pasted in the prompt.

| Day | Main work | Required output / exit condition |
|---|---|---|
| 1 | Finish schema, timestamp, customer-join and amount semantics; freeze temporal folds; fit basic boosted models with/without promotion | Reproducible joined table, audited folds, first scored baseline and valid submission file |
| 2 | Price/basket/product features; cold-start handling; stable categorical normalization | Feature-family ablations on the six-month and recent views |
| 3 | Strict historical statistics; recent-window/recency experiments; snapshot and launch-age sensitivity | Evidence for the selected history and drift policy |
| 4 | Focused tuning of the best two candidates; second model family; paired comparisons and small blend search | Shortlist of models with stored forward predictions |
| 5 | Refit selected model(s); reproduce inference; verify submission and any final selection requirements | Final CSV, frozen configuration, model files and concise experiment record |

Allocate effort approximately as 20% audit/validation, 35% feature/history work, 25% model/drift experiments, 10% blending and 10% final verification. These are planning proportions, not measured runtimes. Benchmark the first fit before setting trial counts. If resources are limited, prioritize one reliable CatBoost model and the promotion/history ablations over a broad tuning search.

For development sampling, select complete orders across months and important channels, then verify finalists on full data. Cache cleaned, target-independent joins where safe, but fit encoders, statistics, imputations and learned transformations inside each fold. Parquet caching is useful after the required engine is available. Use compact dtypes where precision permits, preserving string IDs and sufficient timestamp/financial precision.

## 10. Submission and reproducibility checks

The required file contains exactly two columns in this order: `order_line_id,returned`.

- Exactly 479,995 output rows, one per supplied test line.
- Unique IDs matching the sample submission exactly; use an explicit ID join when predictions have been reordered.
- No train/test IDs lost or duplicated through joins or sorting.
- All scores finite and in [0,1]; probabilities, not thresholded binary labels.
- No accidental dataframe index column.
- All July/boundary test rows included.
- Feature names, dtypes, categorical mappings and history cutoffs match the saved model configuration.
- Inspect score distributions by month/channel/new-customer cohort for missing predictions or suspiciously constant outputs. Distribution differences alone do not justify manual rescaling.
- Generate the CSV again from frozen artifacts and verify matching outputs within the model's documented numerical reproducibility limits.
- Preserve a working baseline submission before the final experiment.

With ROC-AUC, a strictly increasing transformation of a single score preserves ranking. Threshold tuning and cosmetic probability rescaling are therefore low priorities. Calibration can affect blending or decision use, but should be validated on held-out data if attempted.

Before any actual upload, inspect the live competition rules for submission frequency, external data, team eligibility and the number of final submissions that may be selected. No external dataset is needed for the initial plan. Retain the locally strongest temporally robust candidate even when another candidate has a slightly better public score; the supplied overview says public and private each use half of test, but does not specify the split mechanism.

## 11. Suggested implementation layout

```text
nexus-opti-ml-2026-operation-insight/   # original CSVs, preserved
analysis/                             # existing profiling scripts and results
configs/                              # folds, feature groups, models
src/
  audit.py                            # schema and relation checks
  clean.py                            # normalization and snapshot policy
  features.py                         # line, basket and temporal inputs
  history.py                          # cutoff-safe history and target statistics
  validation.py                       # order-aware date splits and metrics
  train.py                            # fit, stop, save model and predictions
  predict.py                          # frozen inference and submission checks
artifacts/                            # saved models, fold predictions, manifests
submissions/                          # validated CSVs
```

The existing `analysis/profile_data.py` and `analysis/targeted_checks.py` are reproducible audit utilities, not a production feature pipeline. In particular, the latest-profile joins in descriptive diagnostics must not be reused without choosing the snapshot policy.

## 12. Immediate implementation priorities

1. Establish isolated modeling dependencies and confirm available CPU/GPU/RAM.
2. Implement row-preserving customer joins, category maps and order-aware temporal splits.
3. Resolve candidate line/basket amount semantics and document timestamp uncertainty.
4. Train matched CatBoost models with and without promotion on the main six-month and recent-period views.
5. Add forward-built store/product/customer return statistics with support counts and cold-start fallbacks.
6. Choose further work from measured late-period improvements, then refit and produce the final submission artifact.

### Evidence files

- `analysis/data_profile.json`: full-file shapes, keys, missingness, distribution summaries, monthly counts and join/semantic checks.
- `analysis/targeted_checks.json`: amount diagnostics, timestamp patterns, category spellings, promotion drift and historical-lookup AUCs.
- Original task context: the supplied competition overview and attached dataset-description text. The live [competition page](https://www.kaggle.com/competitions/nexus-opti-ml-2026-operation-insight/overview) was inaccessible during this analysis.

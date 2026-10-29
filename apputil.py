import pandas as pd


class GroupEstimate(object):
    def __init__(self, estimate):
        if estimate not in ("mean", "median"):
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate

    def fit(self, X, y, default_category=None):
        X = pd.DataFrame(X).copy()

        if X.empty or X.shape[1] == 0:
            raise ValueError(
                "X must have at least one observation and category column"
            )

        y = pd.Series(y).reset_index(drop=True)

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        if y.isna().any():
            raise ValueError("y must not contain missing values")

        if default_category is not None and default_category not in X.columns:
            raise ValueError("default_category must be a column in X")

        self.columns_ = list(X.columns)
        self.default_category_ = default_category

        data = X.reset_index(drop=True)

        target = object()
        data[target] = y

        groups = data.groupby(
            self.columns_,
            observed=True,
            dropna=False
        )[target]

        if self.estimate == "mean":
            self.estimates_ = groups.mean()
        else:
            self.estimates_ = groups.median()

        self.default_estimates_ = None

        if default_category is not None:
            fallback_groups = data.groupby(
                default_category,
                observed=True,
                dropna=False
            )[target]

            if self.estimate == "mean":
                self.default_estimates_ = fallback_groups.mean()
            else:
                self.default_estimates_ = fallback_groups.median()

        return self

    def predict(self, X_):
        if not hasattr(self, "estimates_"):
            raise ValueError("Call fit before predict")

        if isinstance(X_, pd.DataFrame):
            incoming = X_.loc[:, self.columns_].copy()
        else:
            incoming = pd.DataFrame(
                X_,
                columns=self.columns_
            )

        results = []
        missing = 0

        for values in incoming.itertuples(index=False, name=None):
            key = values[0] if len(values) == 1 else values

            prediction = self.estimates_.get(
                key,
                float("nan")
            )

            if pd.isna(prediction) and self.default_estimates_ is not None:
                fallback_key = values[
                    self.columns_.index(self.default_category_)
                ]

                prediction = self.default_estimates_.get(
                    fallback_key,
                    float("nan")
                )

            if pd.isna(prediction):
                missing += 1

            results.append(prediction)

        if missing:
            print(f"{missing} observation(s) had missing groups.")

        return pd.Series(results, dtype=float).to_numpy()
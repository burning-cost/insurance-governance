"""SHAPExplainer — SHAP value computation for insurance pricing models.

SHAP (SHapley Additive exPlanations) is the standard for post-hoc model
explainability in UK insurance. The FCA's Algorithmic Accountability guidance
and the PRA's SS1/23 both expect firms to be able to articulate why a model
produced a given output. SHAP values are the most defensible way to do that.

This module wraps the ``shap`` library with insurance-specific conventions:
- Feature importances are signed (positive = increased prediction).
- Explanations are returned as plain dicts keyed by feature name.
- ``shap`` is an optional dependency. If it is not installed, a clear error
  message is raised pointing the user to the installation command.

The class supports tree-based models (CatBoost, XGBoost, LightGBM,
scikit-learn random forests) and linear models via the appropriate SHAP
explainer type. Pass ``model_type='tree'`` or ``model_type='linear'``.
For anything else, use ``model_type='kernel'`` (slow but universal).

Usage::

    from insurance_governance.audit import SHAPExplainer

    explainer = SHAPExplainer(
        model=fitted_catboost_model,
        model_type='tree',
        feature_names=['driver_age', 'vehicle_age', 'region', 'ncb_years'],
    )
    importances = explainer.explain(X_test)
    # returns list of dicts, one per row
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

_SHAP_IMPORT_ERROR = (
    "The 'shap' package is required for SHAPExplainer but is not installed.\n"
    "Install it with:\n"
    "    pip install shap\n"
    "or add the extra when installing insurance-governance:\n"
    "    pip install insurance-governance[shap]"
)

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


MODEL_TYPES = frozenset({"tree", "linear", "kernel", "deep"})


class SHAPExplainer:
    """Compute SHAP feature importances for a fitted insurance pricing model.

    The explainer is initialised once and can be called repeatedly for
    individual rows or batches. For tree-based models the computation is fast
    (microseconds per row with TreeExplainer). For kernel-based explanations,
    pass a background dataset to :meth:`set_background`.

    Parameters
    ----------
    model:
        A fitted model object. Must be compatible with the chosen
        ``model_type``.
    model_type:
        Which SHAP explainer to use. One of ``'tree'``, ``'linear'``,
        ``'kernel'``, or ``'deep'``.
    feature_names:
        Ordered list of feature names. Must match the column order of arrays
        passed to :meth:`explain` and :meth:`explain_single`.
    background:
        Background dataset for KernelExplainer. Required when
        ``model_type='kernel'``. Ignored for other types. Should be a
        representative sample (100–500 rows) not the full training set.

    Raises:
        ImportError: If ``shap`` is not installed.
        ValueError: If ``model_type`` is not a recognised value.
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        feature_names: list[str],
        background: Any = None,
    ) -> None:
        if not _SHAP_AVAILABLE:
            raise ImportError(_SHAP_IMPORT_ERROR)

        if model_type not in MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {sorted(MODEL_TYPES)}, got {model_type!r}"
            )

        self._model = model
        self._model_type = model_type
        self._feature_names = list(feature_names)
        self._background = background
        self._explainer = self._build_explainer()

    def _build_explainer(self) -> Any:
        """Construct the appropriate SHAP explainer for the model type."""
        if self._model_type == "tree":
            return _shap.TreeExplainer(self._model)
        elif self._model_type == "linear":
            return _shap.LinearExplainer(self._model, self._background)
        elif self._model_type == "deep":
            return _shap.DeepExplainer(self._model, self._background)
        else:  # kernel
            if self._background is None:
                raise ValueError(
                    "A background dataset is required for model_type='kernel'. "
                    "Pass a representative sample as the 'background' argument."
                )
            return _shap.KernelExplainer(self._model.predict, self._background)

    @property
    def feature_names(self) -> list[str]:
        """The ordered list of feature names this explainer was built with."""
        return list(self._feature_names)

    def _to_array(self, X: Any) -> np.ndarray:
        """Convert input to numpy array, handling DataFrames."""
        if _PANDAS_AVAILABLE:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return X.values
        if isinstance(X, np.ndarray):
            return X
        return np.asarray(X)

    def explain(self, X: Any) -> list[dict[str, float]]:
        """Compute SHAP values for a batch of observations.

        Args:
            X: Input features as a 2-D numpy array or pandas DataFrame.
                Rows are observations; columns must match ``feature_names``
                in order.

        Returns:
            List of dicts, one per row. Each dict maps feature name to its
            signed SHAP value. Positive values increased the prediction;
            negative values decreased it.

        Raises:
            ValueError: If the number of columns does not match the number
                of feature names.
        """
        arr = self._to_array(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        n_cols = arr.shape[1]
        n_features = len(self._feature_names)
        if n_cols != n_features:
            raise ValueError(
                f"Expected {n_features} features but got {n_cols} columns. "
                f"Feature names: {self._feature_names}"
            )

        shap_values = self._explainer.shap_values(arr)

        # Some explainers return a list (one array per class) for classifiers.
        # For insurance pricing we typically have a single output; take [1] for
        # binary classifiers (probability of claim) or the only element.
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        # Ensure 2-D: (n_samples, n_features)
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        results: list[dict[str, float]] = []
        for row_values in shap_values:
            results.append(
                {name: float(val) for name, val in zip(self._feature_names, row_values)}
            )
        return results

    def explain_single(self, x: Any) -> dict[str, float]:
        """Compute SHAP values for a single observation.

        Convenience wrapper around :meth:`explain` that accepts a 1-D array
        or a single-row DataFrame and returns a single dict.

        Args:
            x: A 1-D array, list, or single-row DataFrame representing one
                observation.

        Returns:
            Dict mapping feature name to signed SHAP value.
        """
        if _PANDAS_AVAILABLE:
            import pandas as pd
            if isinstance(x, pd.DataFrame):
                if len(x) != 1:
                    raise ValueError(
                        "explain_single expects a single-row DataFrame; "
                        f"got {len(x)} rows."
                    )
                return self.explain(x)[0]

        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self.explain(arr)[0]

    def expected_value(self) -> float:
        """Return the SHAP expected value (base rate / model intercept).

        This is the average model output over the training set — the baseline
        to which SHAP values are additive. Useful for building waterfall charts
        in presentations.

        Returns:
            The expected value as a float. For multi-output models, returns
            the last element (consistent with the convention in :meth:`explain`).

        Raises:
            AttributeError: If the underlying explainer does not expose an
                expected_value attribute.
        """
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[-1])
        return float(ev)

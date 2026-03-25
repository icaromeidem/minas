"""
tests/test_minas.py
====================
Test suite for the MINAS package.

Run with:
    pytest tests/test_minas.py -v
"""

import warnings
import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

FILTERS = [
    "uJAVA", "J0378", "J0395", "J0410", "J0430", "gSDSS",
    "J0515", "rSDSS", "J0660", "iSDSS", "J0861", "zSDSS",
]
ERRORS = [f"{f}_err" for f in FILTERS]

N = 200  # number of synthetic objects


@pytest.fixture
def catalog():
    """Synthetic photometric catalog with magnitudes, errors and stellar params."""
    rng = np.random.default_rng(42)
    data = {f: rng.uniform(14, 20, N) for f in FILTERS}
    data.update({e: rng.uniform(0.01, 0.1, N) for e in ERRORS})
    data.update({f"Ax_{f}": rng.uniform(0.0, 0.3, N) for f in FILTERS})
    data["Dist"] = rng.uniform(100, 5000, N)
    data["Teff"] = rng.uniform(4000, 8000, N)
    data["logg"] = rng.uniform(1.0, 5.0, N)
    data["feh"]  = rng.uniform(-2.5, 0.5, N)
    return pd.DataFrame(data)


@pytest.fixture
def work_df(catalog):
    """Feature DataFrame assembled from the synthetic catalog."""
    import minas as mg
    return mg.preprocess.assemble_work_df(
        df=catalog,
        filters=FILTERS,
        correction_pairs=None,
        add_colors=True,
        verbose=False,
    )


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:

    def test_correct_magnitudes_shape(self, catalog):
        from minas.preprocess import correct_magnitudes
        pairs = {f: f"Ax_{f}" for f in FILTERS}
        result = correct_magnitudes(catalog, pairs)
        assert result.shape == catalog.shape

    def test_correct_magnitudes_values(self, catalog):
        from minas.preprocess import correct_magnitudes
        pairs = {FILTERS[0]: f"Ax_{FILTERS[0]}"}
        result = correct_magnitudes(catalog, pairs)
        expected = catalog[FILTERS[0]] - catalog[f"Ax_{FILTERS[0]}"]
        pd.testing.assert_series_equal(result[FILTERS[0]], expected, check_names=False)

    def test_correct_magnitudes_does_not_mutate(self, catalog):
        from minas.preprocess import correct_magnitudes
        original = catalog[FILTERS[0]].copy()
        correct_magnitudes(catalog, {FILTERS[0]: f"Ax_{FILTERS[0]}"})
        pd.testing.assert_series_equal(catalog[FILTERS[0]], original)

    def test_create_colors_column_count(self, catalog):
        from minas.preprocess import create_colors
        n = len(FILTERS)
        expected_cols = n * (n - 1) // 2
        result = create_colors(catalog, FILTERS)
        assert result.shape[1] == expected_cols

    def test_create_colors_values(self, catalog):
        from minas.preprocess import create_colors
        result = create_colors(catalog, FILTERS[:2])
        col = f"({FILTERS[0]} - {FILTERS[1]})"
        expected = catalog[FILTERS[0]] - catalog[FILTERS[1]]
        pd.testing.assert_series_equal(result[col], expected, check_names=False)

    def test_assemble_work_df_only_filters(self, catalog):
        from minas.preprocess import assemble_work_df
        result = assemble_work_df(catalog, FILTERS, None, add_colors=False, verbose=False)
        assert list(result.columns) == FILTERS
        assert len(result) == N

    def test_assemble_work_df_with_colors(self, catalog):
        from minas.preprocess import assemble_work_df
        n = len(FILTERS)
        result = assemble_work_df(catalog, FILTERS, None, add_colors=True, verbose=False)
        expected_cols = n + n * (n - 1) // 2
        assert result.shape[1] == expected_cols

    def test_assemble_work_df_with_corrections(self, catalog):
        from minas.preprocess import assemble_work_df
        pairs = {f: f"Ax_{f}" for f in FILTERS}
        result = assemble_work_df(catalog, FILTERS, pairs, add_colors=False, verbose=False)
        assert result.shape == (N, len(FILTERS))

    def test_calculate_abs_mag_shape(self, catalog):
        from minas.preprocess import calculate_abs_mag
        result = calculate_abs_mag(catalog, FILTERS, "Dist")
        assert result.shape == catalog.shape

    def test_calculate_abs_mag_values(self, catalog):
        from minas.preprocess import calculate_abs_mag
        result = calculate_abs_mag(catalog, [FILTERS[0]], "Dist")
        mu = 5 * (np.log10(catalog["Dist"]) - 1)
        expected = catalog[FILTERS[0]] - mu
        np.testing.assert_allclose(result[FILTERS[0]].values, expected.values, rtol=1e-5)

    def test_calculate_abs_mag_does_not_mutate(self, catalog):
        from minas.preprocess import calculate_abs_mag
        original = catalog[FILTERS[0]].copy()
        calculate_abs_mag(catalog, FILTERS, "Dist")
        pd.testing.assert_series_equal(catalog[FILTERS[0]], original)


# ── models ────────────────────────────────────────────────────────────────────

class TestModels:

    def test_create_model_rf_reg(self):
        from minas.models import create_model
        model = create_model("RF-REG")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_model_rf_cla(self):
        from minas.models import create_model
        model = create_model("RF-CLA")
        assert hasattr(model, "fit")

    def test_create_model_xgb_reg(self):
        from minas.models import create_model
        model = create_model("XGB-REG")
        assert hasattr(model, "fit")

    def test_create_model_xgb_cla(self):
        from minas.models import create_model
        model = create_model("XGB-CLA")
        assert hasattr(model, "fit")

    def test_create_model_invalid(self):
        from minas.models import create_model
        with pytest.raises(ValueError):
            create_model("INVALID")

    def test_rf_reg_fits_and_predicts(self, work_df, catalog):
        from minas.models import create_model
        model = create_model("RF-REG")
        model.fit(work_df, catalog["Teff"])
        preds = model.predict(work_df)
        assert len(preds) == N
        assert np.all(np.isfinite(preds))

    def test_xgb_reg_fits_and_predicts(self, work_df, catalog):
        from minas.models import create_model
        model = create_model("XGB-REG")
        model.fit(work_df, catalog["Teff"])
        preds = model.predict(work_df)
        assert len(preds) == N
        assert np.all(np.isfinite(preds))

    def test_rf_reg_with_hyperparams(self, work_df, catalog):
        from minas.models import create_model
        # (n_features, n_trees, min_samples_leaf, bootstrap, max_features)
        hp = (5, 10, 1, True, "sqrt")
        model = create_model("RF-REG", hp_combination=hp)
        model.fit(work_df, catalog["Teff"])
        preds = model.predict(work_df)
        assert len(preds) == N

    def test_xgb_reg_with_hyperparams(self, work_df, catalog):
        from minas.models import create_model
        # (colsample_bytree, gamma, learning_rate, max_depth, n_estimators, subsample)
        hp = (0.8, 0.1, 0.1, 3, 10, 0.8)
        model = create_model("XGB-REG", hp_combination=hp)
        model.fit(work_df, catalog["Teff"])
        preds = model.predict(work_df)
        assert len(preds) == N


# ── tuning ────────────────────────────────────────────────────────────────────

class TestTuning:

    def test_hyperparameter_search_rf(self, work_df, catalog, tmp_path):
        from minas.tuning import hyperparameter_search
        param_dist = {
            "randomforestregressor__n_estimators"    : [10],
            "randomforestregressor__min_samples_leaf": [1],
            "selectkbest__k"                         : [5],
        }
        best, search = hyperparameter_search(
            X=work_df, Y=catalog["Teff"],
            model_type="RF",
            param_dist=param_dist,
            tuning_id="test_rf",
            n_iter=1, cv=2,
            save_dir=str(tmp_path),
        )
        assert hasattr(best, "predict")
        assert search.best_score_ is not None

    def test_hyperparameter_search_xgb(self, work_df, catalog, tmp_path):
        from minas.tuning import hyperparameter_search
        param_dist = {
            "xgbregressor__n_estimators": [10],
            "xgbregressor__max_depth"   : [3],
            "selectkbest__k"            : [5],
        }
        best, search = hyperparameter_search(
            X=work_df, Y=catalog["Teff"],
            model_type="XGB",
            param_dist=param_dist,
            tuning_id="test_xgb",
            n_iter=1, cv=2,
            save_dir=str(tmp_path),
        )
        assert hasattr(best, "predict")

    def test_hyperparameter_search_saves_pipeline(self, work_df, catalog, tmp_path):
        from minas.tuning import hyperparameter_search
        import os
        param_dist = {
            "randomforestregressor__n_estimators": [10],
            "selectkbest__k"                     : [5],
        }
        hyperparameter_search(
            X=work_df, Y=catalog["Teff"],
            model_type="RF",
            param_dist=param_dist,
            tuning_id="test_save",
            n_iter=1, cv=2,
            save_dir=str(tmp_path),
        )
        assert os.path.exists(tmp_path / "test_save.joblib")

    def test_hyperparameter_search_invalid_model(self, work_df, catalog, tmp_path):
        from minas.tuning import hyperparameter_search
        with pytest.raises(ValueError):
            hyperparameter_search(
                X=work_df, Y=catalog["Teff"],
                model_type="INVALID",
                param_dist={},
                tuning_id="test",
                save_dir=str(tmp_path),
            )


# ── evaluation ────────────────────────────────────────────────────────────────

class TestEvaluation:

    def test_mad_perfect_prediction(self):
        from minas.evaluation import mad
        y = np.array([1.0, 2.0, 3.0])
        assert mad(y, y) == 0.0

    def test_mad_known_value(self):
        from minas.evaluation import mad
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0])
        result = mad(y_true, y_pred)
        assert result == pytest.approx(1.0)

    def test_mad_array_like(self):
        from minas.evaluation import mad
        result = mad([1, 2, 3], [1, 2, 3])
        assert result == 0.0

    def test_r2_score_perfect(self):
        from minas.evaluation import r2_score
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_score_range(self):
        from minas.evaluation import r2_score
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, 100)
        y_pred = rng.normal(0, 1, 100)
        result = r2_score(y_true, y_pred)
        assert result < 1.0

    def test_calculate_mad_returns_dataframe(self, catalog):
        from minas.evaluation import calculate_mad
        result = calculate_mad(
            predictions=catalog["Teff"] + 50,
            true_values=catalog["Teff"],
            bins=[4000, 5000, 6000, 7000, 8000],
            param_unit="K",
        )
        assert isinstance(result, pd.DataFrame)
        assert "bin" in result.columns
        assert "mad" in result.columns
        assert "objects" in result.columns

    def test_calculate_mad_full_sample_row(self, catalog):
        from minas.evaluation import calculate_mad
        result = calculate_mad(
            predictions=catalog["Teff"],
            true_values=catalog["Teff"],
            bins=[4000, 6000, 8000],
            param_unit="K",
        )
        full_row = result[result["bin"] == "Full Sample"]
        assert len(full_row) == 1
        assert full_row["mad"].values[0] == pytest.approx(0.0, abs=1e-6)

    def test_calculate_mad_empty_bins(self, catalog):
        from minas.evaluation import calculate_mad
        result = calculate_mad(
            predictions=catalog["Teff"],
            true_values=catalog["Teff"],
            bins=[],
            param_unit="K",
        )
        assert "Full Sample" in result["bin"].values
        assert result[result["bin"] == "Full Sample"]["mad"].values[0] == pytest.approx(0.0, abs=1e-6)

# ── feature_selection ─────────────────────────────────────────────────────────

class TestFeatureSelection:

    def test_get_important_features_rf_returns_list(self, work_df, catalog):
        from minas.evaluation.feature_selection import get_important_features
        features, df_feat = get_important_features(
            work_df, catalog["Teff"], n_features_to_save=5, n_estimators=10
        )
        assert isinstance(features, list)
        assert len(features) == 5

    def test_get_important_features_rf_valid_names(self, work_df, catalog):
        from minas.evaluation.feature_selection import get_important_features
        features, _ = get_important_features(
            work_df, catalog["Teff"], n_features_to_save=5, n_estimators=10
        )
        for f in features:
            assert f in work_df.columns

    def test_get_important_features_rf_dataframe(self, work_df, catalog):
        from minas.evaluation.feature_selection import get_important_features
        _, df_feat = get_important_features(
            work_df, catalog["Teff"], n_features_to_save=5, n_estimators=10
        )
        assert "feature" in df_feat.columns
        assert "importance" in df_feat.columns
        assert "cumulative_importance" in df_feat.columns

    def test_get_important_features_xgb_returns_list(self, work_df, catalog):
        from minas.evaluation.feature_selection import get_important_features_xgb
        features, _ = get_important_features_xgb(
            work_df, catalog["Teff"], n_features_to_save=5, n_estimators=10
        )
        assert isinstance(features, list)
        assert len(features) == 5

    def test_get_important_features_xgb_valid_names(self, work_df, catalog):
        from minas.evaluation.feature_selection import get_important_features_xgb
        features, _ = get_important_features_xgb(
            work_df, catalog["Teff"], n_features_to_save=5, n_estimators=10
        )
        for f in features:
            assert f in work_df.columns

    def test_n_features_to_save_respected(self, work_df, catalog):
        from minas.evaluation.feature_selection import get_important_features
        for n in [3, 7, 10]:
            features, _ = get_important_features(
                work_df, catalog["Teff"], n_features_to_save=n, n_estimators=10
            )
            assert len(features) == n


# ── bolometric ────────────────────────────────────────────────────────────────

class TestBolometric:

    @pytest.fixture
    def bc_catalog(self):
        """Minimal catalog for bolometric correction tests."""
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "Teff": rng.uniform(4000, 7000, 50),
            "logg": rng.uniform(1.5, 4.5, 50),
            "MH"  : rng.uniform(-1.5, 0.3, 50),
        })

    def test_apply_bc_returns_dataframe(self, bc_catalog):
        from minas.bolometric.magbol import apply_bc
        result = apply_bc(
            data=bc_catalog,
            teff_col="Teff", logg_col="logg", feh_col="MH",
            model_type="XGB",
        )
        assert isinstance(result, pd.DataFrame)

    def test_apply_bc_adds_pred_column(self, bc_catalog):
        from minas.bolometric.magbol import apply_bc
        result = apply_bc(
            data=bc_catalog,
            teff_col="Teff", logg_col="logg", feh_col="MH",
            model_type="XGB",
        )
        assert "BC_pred" in result.columns

    def test_apply_bc_adds_err_column(self, bc_catalog):
        from minas.bolometric.magbol import apply_bc
        result = apply_bc(
            data=bc_catalog,
            teff_col="Teff", logg_col="logg", feh_col="MH",
            model_type="XGB",
        )
        assert "err_BC_pred" in result.columns

    def test_apply_bc_predictions_finite(self, bc_catalog):
        from minas.bolometric.magbol import apply_bc
        result = apply_bc(
            data=bc_catalog,
            teff_col="Teff", logg_col="logg", feh_col="MH",
            model_type="XGB",
        )
        assert np.all(np.isfinite(result["BC_pred"]))

    def test_apply_bc_rf_model(self, bc_catalog):
        from minas.bolometric.magbol import apply_bc
        bc_catalog_rf = bc_catalog.rename(columns={"MH": "[M/H]"})
        result = apply_bc(
            data=bc_catalog_rf,
            teff_col="Teff", logg_col="logg", feh_col="[M/H]",
            model_type="RF",
        )
        assert "BC_pred" in result.columns

    def test_apply_bc_custom_output_col(self, bc_catalog):
        from minas.bolometric.magbol import apply_bc
        result = apply_bc(
            data=bc_catalog,
            teff_col="Teff", logg_col="logg", feh_col="MH",
            model_type="XGB",
            output_col="my_bc",
        )
        assert "my_bc" in result.columns
        assert "err_my_bc" in result.columns

    def test_apply_bc_invalid_model_type(self, bc_catalog):
        from minas.bolometric.magbol import apply_bc
        with pytest.raises(ValueError):
            apply_bc(
                data=bc_catalog,
                teff_col="Teff", logg_col="logg", feh_col="MH",
                model_type="INVALID",
            )

    def test_apply_bc_file_not_found(self):
        from minas.bolometric.magbol import apply_bc
        with pytest.raises(FileNotFoundError):
            apply_bc(
                data="nonexistent_file.csv",
                teff_col="Teff", logg_col="logg", feh_col="MH",
            )

    def test_apply_bc_does_not_mutate_input(self, bc_catalog):
        from minas.bolometric.magbol import apply_bc
        original_cols = list(bc_catalog.columns)
        apply_bc(
            data=bc_catalog,
            teff_col="Teff", logg_col="logg", feh_col="MH",
            model_type="XGB",
        )
        assert list(bc_catalog.columns) == original_cols

    def test_apply_bc_sigma_multiplier(self, bc_catalog):
        from minas.bolometric.magbol import apply_bc
        r1 = apply_bc(bc_catalog, "Teff", "logg", "MH", sigma_multiplier=1.0)
        r3 = apply_bc(bc_catalog, "Teff", "logg", "MH", sigma_multiplier=3.0)
        assert r3["err_BC_pred"].iloc[0] == pytest.approx(3 * r1["err_BC_pred"].iloc[0])
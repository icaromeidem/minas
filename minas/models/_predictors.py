"""
minas/models/_predictors.py
============================
Model loading utilities and Monte Carlo predictor for photometric surveys.

The ``Predictor`` class applies trained ML pipelines to photometric catalogs
with Monte Carlo error propagation, estimating both predicted values and
their uncertainties from photometric errors.
"""

import gc
import numpy as np
import pandas as pd
import joblib

from minas.preprocess import calculate_abs_mag, assemble_work_df


def load_model_pipeline(model_type, path):
    """
    Load a trained model from disk.

    Parameters
    ----------
    model_type : str
        Type of model to load. Either ``'XGB'`` (XGBoost, ``.json``) or
        ``'RF'`` (Random Forest, ``.sav``).
    path : str
        Path to the saved model file.

    Returns
    -------
    model
        Loaded model object ready for prediction.

    Raises
    ------
    ImportError
        If XGBoost is requested but not installed.
    ValueError
        If an unsupported model_type is provided.

    Examples
    --------
    >>> model = load_model_pipeline('XGB', 'models/teff_xgb.json')
    >>> model = load_model_pipeline('RF',  'models/logg_rf.sav')
    """
    if model_type == 'XGB':
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError('xgboost is not installed. Run: pip install xgboost')
        model = XGBRegressor()
        model.load_model(path)
        return model
    elif model_type == 'RF':
        return joblib.load(path)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported. Use 'XGB' or 'RF'.")


class Predictor:
    """
    Monte Carlo predictor for photometric stellar parameter estimation.

    Applies one or more trained ML pipelines to a photometric catalog and
    estimates prediction uncertainties by propagating photometric errors
    via Monte Carlo simulations. Memory usage is controlled by processing
    MC realisations in mini-batches using Welford's online algorithm for
    variance estimation.

    Parameters
    ----------
    id_col : str
        Name of the object identifier column.
    mag_cols : list of str
        Names of the magnitude (feature) columns.
    err_cols : list of str
        Names of the photometric error columns, in the same order as
        ``mag_cols``.
    dist_col : str or None
        Name of the distance column (parsecs). If provided and present in
        the data, apparent magnitudes are converted to absolute magnitudes
        before prediction.
    correction_pairs : dict or None
        Extinction correction mapping passed to ``assemble_work_df``.
    models : dict
        Dictionary mapping model names to trained pipeline objects.
        Example: ``{'Teff': pipeline_teff, 'logg': pipeline_logg}``
    mc_reps : int
        Total number of Monte Carlo realisations.
    batch_partitions : int
        Number of MC realisations to process per batch. Lower values
        reduce peak memory usage.

    Examples
    --------
    >>> predictor = Predictor(
    ...     id_col='ID',
    ...     mag_cols=mg.FILTERS['JPLUS'],
    ...     err_cols=mg.ERRORS['JPLUS'],
    ...     dist_col='r_est',
    ...     correction_pairs=None,
    ...     models={'Teff': pipeline_teff, 'logg': pipeline_logg},
    ...     mc_reps=100,
    ...     batch_partitions=10,
    ... )
    >>> predictor.predict_parameters((catalog, 'output.csv', [], 'w', True))
    """

    def __init__(
        self,
        id_col,
        mag_cols,
        err_cols,
        dist_col,
        correction_pairs,
        models,
        mc_reps,
        batch_partitions,
    ):
        self.id_col           = id_col
        self.mag_cols         = mag_cols
        self.err_cols         = err_cols
        self.dist_col         = dist_col
        self.correction_pairs = correction_pairs
        self.models           = models
        self.mc_reps          = mc_reps
        self.batch_partitions = batch_partitions

    def predict_parameters(self, args):
        """
        Run predictions with Monte Carlo uncertainty estimation.

        Parameters
        ----------
        args : tuple
            ``(input_data, output_path, keep_cols, save_mode, header)``

            - **input_data** : str or pd.DataFrame — catalog to predict on.
            - **output_path** : str — path for the output CSV file.
            - **keep_cols** : list of str — extra columns to carry over to output.
            - **save_mode** : str — pandas ``to_csv`` mode (``'w'`` or ``'a'``).
            - **header** : bool — whether to write the CSV header.

        Returns
        -------
        bool
            True on success.
        """
        input_data, output_path, keep_cols, save_mode, header = args

        if isinstance(input_data, str):
            input_data = pd.read_csv(input_data, dtype='float32')

        if self.dist_col and self.dist_col in input_data.columns:
            input_data = calculate_abs_mag(input_data, self.mag_cols, self.dist_col)

        work_df = assemble_work_df(
            df=input_data,
            filters=self.mag_cols,
            correction_pairs=self.correction_pairs,
            add_colors=True,
            verbose=False,
        )

        final_df = pd.DataFrame(
            index=input_data.index,
            columns=list(self.models.keys()) + [f"{x}-ERR" for x in self.models.keys()],
            dtype='float32',
        )

        for model_name in self.models:
            pipeline = self.models[model_name]
            true_predictions = None
            n = None
            mean = None
            m2 = None
            std_dev = None

            # Detect pipeline vs direct estimator
            model_obj = pipeline[-1] if hasattr(pipeline, '__getitem__') and not isinstance(pipeline, str) else pipeline

            if "Classifier" in str(type(model_obj)):
                true_predictions = [x[1] for x in pipeline.predict_proba(work_df)]
            elif "Regressor" in str(type(model_obj)):
                true_predictions = pipeline.predict(work_df)

            # Monte Carlo loop — process in mini-batches to control memory
            mc_batch_size = self.batch_partitions
            for batch_start in range(0, self.mc_reps, mc_batch_size):
                batch_end  = min(batch_start + mc_batch_size, self.mc_reps)
                batch_size = batch_end - batch_start
                batch_predictions = np.empty((batch_size, len(work_df)), dtype='float32')

                for i in range(batch_size):
                    norm_dist = np.random.normal(
                        size=(len(input_data), len(self.err_cols))
                    ).astype('float32')

                    mc_input_data = (
                        input_data[self.mag_cols].astype('float32')
                        + (input_data[self.err_cols].astype('float32') * norm_dist).values
                    )

                    if self.correction_pairs:
                        correction_cols = list(self.correction_pairs.values())
                        mc_input_data[correction_cols] = input_data[correction_cols]

                    mc_work_df = assemble_work_df(
                        df=mc_input_data,
                        filters=self.mag_cols,
                        correction_pairs=self.correction_pairs,
                        add_colors=True,
                        verbose=False,
                    )

                    model_obj = pipeline[-1] if hasattr(pipeline, '__getitem__') and not isinstance(pipeline, str) else pipeline

                    if "Classifier" in str(type(model_obj)):
                        batch_predictions[i] = [x[1] for x in pipeline.predict_proba(mc_work_df)]
                    elif "Regressor" in str(type(model_obj)):
                        batch_predictions[i] = pipeline.predict(mc_work_df)

                    del mc_work_df, norm_dist

                # Online variance — Welford / Chan's parallel algorithm
                if batch_start == 0:
                    n    = batch_size
                    mean = batch_predictions.mean(axis=0)
                    m2   = ((batch_predictions - mean) ** 2).sum(axis=0)
                else:
                    batch_mean = batch_predictions.mean(axis=0)
                    batch_var  = batch_predictions.var(axis=0, ddof=0)
                    new_n      = n + batch_size
                    delta      = batch_mean - mean
                    mean       = (n * mean + batch_size * batch_mean) / new_n
                    m2         = m2 + batch_size * batch_var + (n * batch_size * delta ** 2) / new_n
                    n          = new_n

                del batch_predictions
                gc.collect()

            # Final standard deviation
            if n is not None and n > 1:
                std_dev = np.sqrt(m2 / (n - 1))
            elif mean is not None:
                std_dev = np.zeros_like(mean)
            else:
                std_dev = np.zeros(len(work_df), dtype='float32')

            if true_predictions is None:
                true_predictions = np.zeros(len(work_df), dtype='float32')

            final_df[model_name]          = true_predictions
            final_df[f"{model_name}-ERR"] = std_dev

        final_df[["RA", "DEC"]] = input_data[["RA", "DEC"]]
        final_df[keep_cols]     = input_data[keep_cols]
        final_df = final_df[
            ["RA", "DEC"]
            + [f"{x}{y}" for x in self.models.keys() for y in ["", "-ERR"]]
            + keep_cols
        ]

        final_df.to_csv(output_path, mode=save_mode, header=header)

        del final_df, work_df, input_data
        gc.collect()

        return True
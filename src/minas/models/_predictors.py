def load_model_pipeline(model_type, path):
    """
    Loads a trained model of type XGB (.json) or RF (.sav) according to the specified type.
    model_type: 'XGB' or 'RF'
    path: model file path
    """
    if model_type == 'XGB':
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError('xgboost não está instalado.')
        model = XGBRegressor()
        model.load_model(path)
        return model
    elif model_type == 'RF':
        import joblib
        return joblib.load(open(path, 'rb'))
    else:
        raise ValueError('Model type not supported: use "XGB" or "RF"')
import gc
import numpy as np
import pandas as pd

from minas.preprocess import calculate_abs_mag, assemble_work_df

class Predictor:
    def __init__(self, id_col, mag_cols, err_cols, dist_col, correction_pairs, models, mc_reps, batch_partitions):
        self.id_col = id_col
        self.mag_cols = mag_cols
        self.err_cols = err_cols
        self.dist_col = dist_col
        self.correction_pairs = correction_pairs
        self.models = models
        self.mc_reps = mc_reps
        self.batch_partitions = batch_partitions

    def predict_parameters(self, args):
        input_data, output_path, keep_cols, save_mode, header = args

        if isinstance(input_data, str):
            input_data = pd.read_csv(input_data, dtype='float32')

        if self.dist_col and self.dist_col in input_data.columns:
            input_data = calculate_abs_mag(input_data, self.mag_cols, self.dist_col)

        # input_data.set_index(self.id_col, drop=True, inplace=True)

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
            dtype='float32'
        )



        for model_name in self.models:
            pipeline = self.models[model_name]

            true_predictions = None
            n = None
            mean = None
            m2 = None
            std_dev = None

            # Detect if it's a pipeline (list, tuple, sklearn Pipeline) or direct estimator
            if hasattr(pipeline, '__getitem__') and not isinstance(pipeline, str):
                model_obj = pipeline[-1]
            else:
                model_obj = pipeline

            if "Classifier" in str(type(model_obj)):
                true_predictions = pipeline.predict_proba(work_df)
                true_predictions = [x[1] for x in true_predictions]
            elif "Regressor" in str(type(model_obj)):
                true_predictions = pipeline.predict(work_df)

            # Process MC in mini-batches to save memory
            mc_batch_size = self.batch_partitions  # Defined by user when creating Predictor
            for batch_start in range(0, self.mc_reps, mc_batch_size):
                batch_end = min(batch_start + mc_batch_size, self.mc_reps)
                batch_size = batch_end - batch_start

                batch_predictions = np.empty((batch_size, len(work_df)), dtype='float32')

                for i in range(batch_size):
                    norm_dist = np.random.normal(size=(len(input_data), len(self.err_cols))).astype('float32')

                    mc_input_data = (
                        input_data[self.mag_cols].astype('float32')
                        + (input_data[self.err_cols].astype('float32') * norm_dist).values
                    )

                    if self.correction_pairs:
                        correction_cols = [x for x in self.correction_pairs.values()]
                        mc_input_data[correction_cols] = input_data[correction_cols]

                    mc_work_df = assemble_work_df(
                        df=mc_input_data,
                        filters=self.mag_cols,
                        correction_pairs=self.correction_pairs,
                        add_colors=True,
                        verbose=False,
                    )

                    # Detect if it's a pipeline (list, tuple, sklearn Pipeline) or direct estimator
                    if hasattr(pipeline, '__getitem__') and not isinstance(pipeline, str):
                        model_obj = pipeline[-1]
                    else:
                        model_obj = pipeline

                    if "Classifier" in str(type(model_obj)):
                        batch_predictions[i] = [x[1] for x in pipeline.predict_proba(mc_work_df)]
                    elif "Regressor" in str(type(model_obj)):
                        batch_predictions[i] = pipeline.predict(mc_work_df)

                    # Clear intermediate memory
                    del mc_work_df, norm_dist

                # Calculate partial variance (Welford's online algorithm)
                if batch_start == 0:
                    n = batch_size
                    mean = batch_predictions.mean(axis=0)
                    m2 = ((batch_predictions - mean) ** 2).sum(axis=0)
                else:
                    if n is None or mean is None or m2 is None:
                        # fallback: if for some reason it didn't initialize, initialize now
                        n = batch_size
                        mean = batch_predictions.mean(axis=0)
                        m2 = ((batch_predictions - mean) ** 2).sum(axis=0)
                    else:
                        batch_mean = batch_predictions.mean(axis=0)
                        batch_var = batch_predictions.var(axis=0, ddof=0)

                        # Combine statistics (Chan's algorithm)
                        new_n = n + batch_size
                        delta = batch_mean - mean
                        mean = (n * mean + batch_size * batch_mean) / new_n
                        m2 = m2 + batch_size * batch_var + (n * batch_size * delta ** 2) / new_n
                        n = new_n

                # Limpar batch
                del batch_predictions
                gc.collect()

            # Calculate final standard deviation
            if n is not None and mean is not None and m2 is not None and n > 1:
                std_dev = np.sqrt(m2 / (n - 1))
            elif mean is not None:
                std_dev = np.zeros_like(mean)
            else:
                std_dev = np.zeros(len(work_df), dtype='float32')

            if true_predictions is None:
                true_predictions = np.zeros(len(work_df), dtype='float32')

            final_df[model_name] = true_predictions
            final_df[f"{model_name}-ERR"] = std_dev

        final_df[["RA", "DEC"]] = input_data[["RA", "DEC"]]
        final_df[keep_cols] = input_data[keep_cols]
        final_df = final_df[
            ["RA", "DEC"]
            + [f"{x}{y}" for x in self.models.keys() for y in ["", "-ERR"]]
            + keep_cols
        ]

        final_df.to_csv(output_path, mode=save_mode, header=header)

        # clean the memory
        del final_df, work_df, input_data
        gc.collect()

        return True
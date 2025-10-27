"""
Bolometric Correction Module for MINAS
Module for calculating bolometric correction using pre-trained models

Usage:
    import minas as mg
    
    # Apply BC to a table
    df = mg.bolometric.apply_bc(
        data='my_table.csv',
        teff_col='jplusL_02_teff',
        logg_col='jplusL_02_logg', 
        feh_col='jplusL_02_feh',
        model_type='XGB'
    )
"""

import os
import pickle
import pandas as pd
import xgboost as xgb
from pathlib import Path

# Path to pre-trained models
_MODEL_DIR = Path(__file__).parent / 'models_bc'

# Path to validation data from trained models
_VALIDATION_DIR = Path(__file__).parent.parent.parent.parent.parent / 'apply_models' / 'bolometric_correction'


def _load_model_stats(model_type):
    """
    Load model statistics from validation files.
    
    Args:
        model_type: 'XGB' or 'RF'
    
    Returns:
        dict with 'std', 'mad', 'r2', 'n_samples'
    """
    import numpy as np
    import glob
    
    # Search for the most recent validation file
    if model_type == 'XGB':
        pattern = str(_VALIDATION_DIR / 'table_XG' / '*_pred_XG.csv')
    else:
        pattern = str(_VALIDATION_DIR / 'table_RF' / '*_pred_RF.csv')
    
    files = glob.glob(pattern)
    
    if not files:
        # If no files found, use default values
        if model_type == 'XGB':
            return {'std': 0.0430, 'mad': 0.0062, 'r2': 0.9983, 'n_samples': None}
        else:
            return {'std': 0.0573, 'mad': 0.0065, 'r2': 0.9970, 'n_samples': None}
    
    # Get the most recent file
    latest_file = max(files, key=lambda x: Path(x).stem)
    
    # Load data
    df = pd.read_csv(latest_file)
    
    # Calculate statistics
    residuals = df['BC_pred'] - df['BC_real']
    std = np.std(residuals)
    mad = np.median(np.abs(residuals))
    
    # Extract R² from filename (format: YYYYMMDDHHMMSS_bc_R2_pred_MODEL.csv)
    try:
        r2 = float(Path(latest_file).stem.split('_')[2])
    except:
        r2 = None
    
    return {
        'std': std,
        'mad': mad,
        'r2': r2,
        'n_samples': len(df),
        'validation_file': Path(latest_file).name
    }


class BolometricCorrection:
    """
    Class to load and apply bolometric correction models.
    
    Available models:
    - 'XGB': XGBoost
    - 'RF': Random Forest
    """
    
    def __init__(self, model_type='XGB'):
        """
        Args:
            model_type (str): 'XGB' or 'RF'
        """
        self.model_type = model_type.upper()
        if self.model_type not in ['XGB', 'RF']:
            raise ValueError("model_type must be 'XGB' or 'RF'")
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model."""
        if self.model_type == 'XGB':
            model_path = _MODEL_DIR / 'model_bc_XGB.json'
            if not model_path.exists():
                raise FileNotFoundError(f"XGBoost model not found at {model_path}")
            
            self.model = xgb.XGBRegressor()
            self.model.load_model(str(model_path))
            
        elif self.model_type == 'RF':
            model_path = _MODEL_DIR / 'model_bc_RF.sav'
            if not model_path.exists():
                raise FileNotFoundError(f"Random Forest model not found at {model_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)


def apply_bc(data, teff_col, logg_col, feh_col, model_type='XGB', 
             output_file=None, output_col='BC_pred', sigma_multiplier=3.0):
    """
    Apply bolometric correction to a table.
    
    Follows the application notebook pattern:
    1. Load table (CSV or DataFrame)
    2. Rename columns to model format
    3. Apply BC model
    4. Calculate uncertainties based on model's standard deviation
    5. Revert to original column names
    6. Add BC_pred and err_BC columns
    7. Show model statistics
    8. Save result (optional)
    
    Args:
        data: DataFrame or path to CSV file
        teff_col: Name of Teff column
        logg_col: Name of logg column
        feh_col: Name of [M/H] column
        model_type: 'XGB' or 'RF'
        output_file: Path to save CSV (optional)
        output_col: Name of output column (default: 'BC_pred')
        sigma_multiplier: Standard deviation multiplier to calculate uncertainty (default: 3.0)
    
    Returns:
        DataFrame with BC_pred and err_BC columns added
    
    Examples:
        >>> import minas as mg
        >>> 
        >>> # With CSV file and 3-sigma uncertainty
        >>> df = mg.bolometric.apply_bc(
        ...     data='20241029124657_pred_mc_02.csv',
        ...     teff_col='jplusL_02_teff',
        ...     logg_col='jplusL_02_logg',
        ...     feh_col='jplusL_02_feh',
        ...     model_type='XGB',
        ...     sigma_multiplier=3.0,
        ...     output_file='table/pred_mc_bc_xg_02.csv'
        ... )
        >>> 
        >>> # With 2-sigma
        >>> df = mg.bolometric.apply_bc(
        ...     data=df,
        ...     teff_col='Teff',
        ...     logg_col='logg',
        ...     feh_col='[M/H]',
        ...     model_type='RF',
        ...     sigma_multiplier=2.0
        ... )
    """
    # Load data
    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File not found: {data}")
        df = pd.read_csv(data)
        print(f"Table loaded: {data}")
    else:
        df = data.copy()
    
    # Validate model_type
    model_type = model_type.upper()
    if model_type not in ['XGB', 'RF']:
        raise ValueError("model_type must be 'XGB' or 'RF'")
    
    # Get model information (calculated dynamically from validation files)
    model_info = _load_model_stats(model_type)
    std = model_info['std']
    mad = model_info['mad']
    r2 = model_info.get('r2', 'N/A')
    n_samples = model_info.get('n_samples', 'N/A')
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model_type}")
    print(f"{'='*60}")
    if 'validation_file' in model_info:
        print(f"Validation file: {model_info['validation_file']}")
    print(f"R² Score: {r2}")
    print(f"Validation samples: {n_samples}")
    print(f"Standard Deviation (STD): {std:.4f} mag")
    print(f"MAD: {mad:.4f} mag")
    print(f"Selected multiplier: {sigma_multiplier:.1f}x")
    
    # Calculate uncertainty
    uncertainty = sigma_multiplier * std
    print(f"Calculated uncertainty: {uncertainty:.4f} mag ({sigma_multiplier:.1f} × {std:.4f})")
    print(f"{'='*60}\n")
    
    # Save original names
    original_teff = teff_col
    original_logg = logg_col
    original_feh = feh_col
    
    # Rename to model format
    df.rename(columns={
        teff_col: 'Teff',
        logg_col: 'logg',
        feh_col: '[M/H]'
    }, inplace=True)
    
    # XGBoost uses 'MH' instead of '[M/H]'
    if model_type == 'XGB':
        df.rename(columns={'[M/H]': 'MH'}, inplace=True)
        feature_cols = ['Teff', 'logg', 'MH']
    else:
        feature_cols = ['Teff', 'logg', '[M/H]']
    
    # Load model and make prediction
    bc_calc = BolometricCorrection(model_type=model_type)
    X_new = df[feature_cols]
    bc_predictions = bc_calc.model.predict(X_new)
    
    # Add predictions and uncertainties
    df[output_col] = bc_predictions
    df['err_' + output_col] = uncertainty
    
    # Load validation file to calculate statistics
    try:
        import numpy as np
        import glob
        
        # Search for validation file
        if model_type == 'XGB':
            pattern = str(_VALIDATION_DIR / 'table_XG' / '*_pred_XG.csv')
        else:
            pattern = str(_VALIDATION_DIR / 'table_RF' / '*_pred_RF.csv')
        
        files = glob.glob(pattern)
        
        if files:
            latest_file = max(files, key=lambda x: Path(x).stem)
            df_val = pd.read_csv(latest_file)
            
            # Calculate individual absolute error on validation set
            abs_error_individual = abs(df_val['BC_pred'] - df_val['BC_real'])
            
            # Calculate statistics based on chosen multiplier
            threshold_user = sigma_multiplier * std
            threshold_5x = 5 * std
            
            percent_below_user = (abs_error_individual < threshold_user).mean() * 100
            percent_above_5x = (abs_error_individual >= threshold_5x).mean() * 100
            
            print(f"ERROR STATISTICS (based on validation set):")
            print(f"{'='*60}")
            print(f"Percentage of errors below {sigma_multiplier:.1f}× std deviation: {percent_below_user:.2f}%")
            print(f"Percentage of errors >= 5× std deviation: {percent_above_5x:.2f}%")
            print(f"{'='*60}\n")
    except Exception as e:
        print(f"Warning: Could not calculate validation statistics: {e}\n")
    
    # Revert column names
    if model_type == 'XGB':
        df.rename(columns={'MH': '[M/H]'}, inplace=True)
    
    df.rename(columns={
        'Teff': original_teff,
        'logg': original_logg,
        '[M/H]': original_feh
    }, inplace=True)
    
    # Save if requested
    if output_file is not None:
        df.to_csv(output_file, index=False)
        print(f'Table with BC predictions saved to: {output_file}\n')
    
    return df



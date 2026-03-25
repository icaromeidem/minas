"""
minas/preprocess/_functions.py
================================
Core preprocessing functions for photometric data.

Provides utilities for magnitude correction, color creation,
and work DataFrame assembly used throughout the MINAS pipeline.
"""

from itertools import combinations
import time
import pandas as pd
import numpy as np


def correct_magnitudes(df, correction_pairs):
    """
    Apply extinction corrections to a set of photometric filters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing uncorrected apparent magnitudes and
        their corresponding extinction columns.
    correction_pairs : dict
        Dictionary mapping each filter column to its extinction column.
        Example: ``{'gSDSS': 'Ax_gSDSS', 'rSDSS': 'Ax_rSDSS'}``

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with corrected magnitudes.

    Examples
    --------
    >>> corrections = {'gSDSS': 'Ax_gSDSS', 'rSDSS': 'Ax_rSDSS'}
    >>> df_corrected = correct_magnitudes(df, corrections)
    """
    df_copy = df.copy()

    for filt in correction_pairs:
        df_copy[filt] = df_copy[filt] - df_copy[correction_pairs[filt]]

    return df_copy


def create_colors(df, filters):
    """
    Create all pairwise filter combinations (colors) from a set of filters.

    For N filters, produces N*(N-1)/2 colors of the form ``(A - B)``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the filter magnitude columns.
    filters : list of str
        List of filter column names to combine.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per color, named ``(A - B)``.

    Examples
    --------
    >>> filters = ['gSDSS', 'rSDSS', 'iSDSS']
    >>> colors_df = create_colors(df, filters)
    >>> # columns: '(gSDSS - rSDSS)', '(gSDSS - iSDSS)', '(rSDSS - iSDSS)'
    """
    comb_list = list(combinations(filters, 2))
    colors_df = pd.DataFrame()

    for comb in comb_list:
        color_name = f"({comb[0]} - {comb[1]})"
        color_values = df[comb[0]] - df[comb[1]]
        colors_df = pd.concat([colors_df, color_values.rename(color_name)], axis=1)

    return colors_df


def create_combinations(df, filters):
    """
    Create all 4-filter color combinations of the form ``(A-B) - (C-D)``.

    For N filters, produces C(N,4) combinations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the filter magnitude columns.
    filters : list of str
        Set of filter column names to combine.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per combination,
        named ``(A - B) - (C - D)``.
    """
    comb_list = list(combinations(filters, 4))
    combinations_df = pd.DataFrame()

    for comb in comb_list:
        combination_name = f"({comb[0]} - {comb[1]}) - ({comb[2]} - {comb[3]})"
        combination_value = (df[comb[0]] - df[comb[1]]) - (df[comb[2]] - df[comb[3]])
        combinations_df = pd.concat(
            [combinations_df, combination_value.rename(combination_name)], axis=1
        )

    return combinations_df


def assemble_work_df(
    df,
    filters,
    correction_pairs,
    add_colors=False,
    add_combinations=False,
    verbose=True,
):
    """
    Assemble the feature DataFrame used as input to ML models.

    Optionally applies extinction corrections, computes pairwise colors,
    and computes 4-filter color combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing magnitudes and extinction columns.
    filters : list of str
        Filter columns to include as base features.
    correction_pairs : dict or None
        Dictionary mapping filter columns to extinction columns.
        Pass ``None`` or an empty dict to skip corrections.
    add_colors : bool, optional
        If True, all pairwise colors are appended. Default: False.
    add_combinations : bool, optional
        If True, all 4-filter combinations are appended. Default: False.
    verbose : bool, optional
        If True, prints progress and timing information. Default: True.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame ready for model input.

    Examples
    --------
    >>> work_df = assemble_work_df(
    ...     df=catalog,
    ...     filters=mg.FILTERS['JPLUS'],
    ...     correction_pairs=dict(zip(mg.FILTERS['JPLUS'], mg.CORRECTIONS['JPLUS'])),
    ...     add_colors=True,
    ... )
    """
    if verbose:
        print("Assembling work DataFrame:\n")

    if correction_pairs:
        if verbose:
            print("  - Applying extinction corrections...", end="")
        start_time = time.time()
        df = correct_magnitudes(df, correction_pairs)
        if verbose:
            print(f" {(time.time() - start_time):.2f} s")

    work_df = df[filters].copy()

    if add_colors:
        if verbose:
            print("  - Computing pairwise colors...", end="")
        start_time = time.time()
        colors_df = create_colors(work_df, filters)
        work_df = pd.concat([work_df, colors_df], axis=1)
        if verbose:
            print(f" {(time.time() - start_time):.2f} s")

    if add_combinations:
        if verbose:
            print("  - Computing 4-filter combinations...", end="")
        start_time = time.time()
        combinations_df = create_combinations(work_df, filters)
        work_df = pd.concat([work_df, combinations_df], axis=1)
        if verbose:
            print(f" {(time.time() - start_time):.2f} s")

    if verbose:
        print(f"\nDone. Output shape: {work_df.shape}")

    return work_df


def calculate_abs_mag(df, filters, distance):
    """
    Convert apparent magnitudes to absolute magnitudes using distance in parsecs.

    Applies the distance modulus: ``M = m - 5 * (log10(d) - 1)``

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing apparent magnitude columns and a distance column.
    filters : list of str
        Filter columns to convert.
    distance : str
        Name of the column containing distances in parsecs.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with absolute magnitudes in the
        specified filter columns.

    Examples
    --------
    >>> df_abs = calculate_abs_mag(df, filters=mg.FILTERS['JPLUS'], distance='r_est')
    """
    df_copy = df.copy()
    y = 5 * (df[distance].apply(np.log10) - 1)

    for filt in filters:
        df_copy[filt] = df_copy[filt] - y

    return df_copy
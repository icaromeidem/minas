from itertools import combinations
import time
import pandas as pd
import numpy as np


def correct_magnitudes(df, correction_pairs):
    """
    Correct the magnitudes of a set of filters inside a dataframe

    Keyword Arguments:
    df - Dataframe with uncorrected magnitudes
    correction_pairs - Dictionary with all the combinations of filters and corrections to be applied
    """

    df_copy = df.copy()

    for filt in correction_pairs:
        df_copy[filt] = df_copy[filt] - df_copy[correction_pairs[filt]]

    return df_copy


def create_colors(df, filters):
    """
    Create all the possible filter combinations (colors) for a set of filters inside a dataframe

    Keyword arguments:
    df - Dataframe with the magnitudes
    filters - List of filters to combine
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
    Create all the possible color combinations for a set of filters inside a dataframe

    Keyword arguments:
    df - Dataframe with the magnitudes
    filters - Set of filters to combine
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
    df, filters, correction_pairs, add_colors=False, add_combinations=False, verbose=True
):
    """
    Assemble a dataframe with a set of magnitudes and, when asked, colors and color combinations.

    Keyword arguments:
    df - Dataframe with the magnitudes

    filters - Set of filters to use
    correction_pairs - Dictionary with the pairs of filters and corrections to be applied

    add_colors - If True, all the possible colors will be added to the returned dataframe
    add_combinations - If True all the possible color combinations will be added to the returned dataframe
    """
    
    if verbose:
        print("Iniciando processo de criação do dataframe de trabalho:\n")

    # If a dictionary of corrections is passed, apply the corrections to the magnitudes
    if correction_pairs:
        if verbose:
            print("  - Aplicando correção de magnitudes...", end="")

        start_time = time.time()
        df = correct_magnitudes(df, correction_pairs)
        
        if verbose:
            print(f" Tempo: {(time.time() - start_time):.2f} s")

    # Filter the df to get only the filters passed
    work_df = df[filters].copy()

    # If asked for, create a dataframe with all the possible colors and add it to the work dataframe
    if add_colors is True:
        if verbose:
            print("  - Adicionando cores ao dataframe...", end="")

        start_time = time.time()
        colors_df = create_colors(work_df, filters)
        work_df = pd.concat([work_df, colors_df], axis=1)
        
        if verbose:
            print(f" Tempo: {(time.time() - start_time):.2f} s")

    # If asked for, create a dataframe with all the possible color combinations and add it to the work dataframe
    if add_combinations is True:
        if verbose:
            print("  - Adicionando combinações de cores ao dataframe...", end="")

        start_time = time.time()
        combinations_df = create_combinations(work_df, filters)
        work_df = pd.concat([work_df, combinations_df], axis=1)
        
        if verbose:
            print(f" Tempo: {(time.time() - start_time):.2f} s")

    if verbose:
        print(f"\nProcesso finalizado! Shape da Tabela Gerada: {work_df.shape}")

    # Return the resulting dataframe
    return work_df


def calculate_abs_mag(df, filters, distance):
    """
    Calculate the absolute magnitudes of a set of apparent magnitudes inside a dataframe.

    Keyword arguments:
    df - Dataframe with the apparent magnitudes and distances

    filters - Set of filters to use
    distance - Name of the distance column to use
    """

    df_copy = df.copy()

    y = 5 * (df[distance].apply(np.log10) - 1)

    for filt in filters:
        df_copy[filt] = df_copy[filt] - y

    return df_copy

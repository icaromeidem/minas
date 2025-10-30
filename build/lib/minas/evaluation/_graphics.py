"""
Graphics module for MINAS
Funções para visualização de resultados de modelos de regressão
"""

# ==================== Imports e Configurações Globais ====================
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import os
import json
from math import log10, floor

# Configurações de fonte
label_fontdict = {"family": "sans-serif", "weight": "normal", "size": 10}
legend_label_fontdict = {"family": "sans-serif", "weight": "normal", "size": 10}
legend_title_fontdict = {"family": "sans-serif", "weight": "bold", "size": 10}
tick_size = 10

# Mapeamento de parâmetros (centralizado para evitar redundância)
PARAM_MAP = {
    'teff': {'name': 'Teff', 'unit': 'K', 'cmap': 'Reds', 'ylabel': 'Teff Predicted (K)'},
    'logg': {'name': 'logg', 'unit': 'dex', 'cmap': 'Greens', 'ylabel': 'logg Predicted (dex)'},
    'feh': {'name': '[Fe/H]', 'unit': 'dex', 'cmap': 'Blues', 'ylabel': '[Fe/H] Predicted (dex)'}
}

SURVEY_MAP = {'A': 'APOGEE', 'L': 'LAMOST', 'G': 'GALAH', 'W': 'WISE'}

# ==================== Funções Auxiliares ====================
def _set_times_font():
    """Configura fonte Times para os gráficos (sem definir tamanho global)."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times', 'DejaVu Serif', 'Liberation Serif']

def _detect_param_key(param_name):
    """Detecta a chave do parâmetro a partir do nome."""
    if param_name is None:
        return 'teff'
    pname = param_name.strip().lower().replace('[', '').replace(']', '').replace('/', '').replace(' ', '')
    if 'teff' in pname:
        return 'teff'
    elif 'feh' in pname:
        return 'feh'
    elif 'logg' in pname:
        return 'logg'
    return 'teff'

def _create_custom_cmap(cmap_name):
    """Cria colormap customizado com cinza inicial."""
    base_cmap = cm.get_cmap(cmap_name)
    gray_rgb = mcolors.to_rgb('#cccccc')
    new_colors = [gray_rgb] + [base_cmap(i) for i in np.linspace(0.15, 1, 255)]
    return LinearSegmentedColormap.from_list(f"gray_{cmap_name}", new_colors)

def _calculate_metrics(y_true, y_pred):
    """Calcula métricas de regressão."""
    residuals = y_pred - y_true
    sigma = np.std(residuals)
    mad = np.median(np.abs(residuals))
    try:
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
    except ImportError:
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    return r2, mad, sigma, residuals

def _load_metrics_from_json(metrics_json_path, r2, mad, param_unit):
    """Carrega métricas de arquivo JSON se disponível."""
    if metrics_json_path is not None and os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, 'r') as f:
                metrics = json.load(f)
            return f"R² = {metrics.get('r2', r2):.4f} | MAD = {metrics.get('mad', mad):.2f} {param_unit}"
        except:
            return None
    return None

def _get_xlabel(pinfo, survey_name):
    """Gera o label do eixo X com nome do survey se fornecido."""
    survey_label = SURVEY_MAP.get(survey_name, survey_name) if survey_name else None
    if survey_label:
        return f"{pinfo['name']} {survey_label} ({pinfo['unit']})"
    return f"{pinfo['name']} True ({pinfo['unit']})"

# ==================== Funções Principais ====================

# Matrix of regression plots
def plot_regression_matrix(results_dict, bins_dict, param_order=None, titles=None, point_size=3, 
                          metrics_json_paths=None, training_id=None, survey_name=None):
    """
    Generates a 3x2 matrix of regression plots for teff, logg, and feh.
    
    Args:
        results_dict: dict with keys ('teff_Restricted', 'teff_Less Restricted', ...), values: (y_true, y_pred)
        bins_dict: dict with keys 'teff', 'logg', 'feh', values: bins
        param_order: order of parameters (default: ['teff', 'logg', 'feh'])
        titles: list of column titles (e.g., ['Restricted', 'Less Restricted'])
        metrics_json_paths: dict with paths to metrics (optional)
    """
    _set_times_font()
    
    if param_order is None:
        param_order = ['teff', 'logg', 'feh']
    if titles is None:
        titles = ['Restricted', 'Less Restricted']

    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.12, wspace=0.02, 
                          left=0.06, right=0.91, top=0.95, bottom=0.06)
    
    first_col_axes = {}
    
    for i, param in enumerate(param_order):
        for j, restr in enumerate(['Restricted', 'Less Restricted']):
            key = f'{param}_{restr}'
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i, j], 
                                                        height_ratios=[0.5, 3.5], hspace=0.05)
            
            if j == 0:
                ax_res = fig.add_subplot(inner_gs[0])
                ax_main = fig.add_subplot(inner_gs[1], sharex=ax_res)
                first_col_axes[i] = {'res': ax_res, 'main': ax_main}
            else:
                ax_res = fig.add_subplot(inner_gs[0], sharex=first_col_axes[i]['res'], 
                                        sharey=first_col_axes[i]['res'])
                ax_main = fig.add_subplot(inner_gs[1], sharex=first_col_axes[i]['main'], 
                                         sharey=first_col_axes[i]['main'])
            
            plt.setp(ax_res.get_xticklabels(), visible=False)
            
            if key not in results_dict or results_dict[key] is None:
                ax_res.axis('off')
                ax_main.axis('off')
                continue
                
            y_true, y_pred = results_dict[key]
            bins = bins_dict.get(param, None)
            metrics_path = metrics_json_paths.get(key, None) if metrics_json_paths else None
            
            _plot_regression_with_residuals_panels(ax_main, ax_res, y_true, y_pred, bins=bins, 
                                                   param_name=param, point_size=point_size, 
                                                   metrics_json_path=metrics_path, training_id=training_id, 
                                                   survey_name=survey_name)
            
            if j == 1:
                ax_res.set_ylabel('')
                ax_main.set_ylabel('')
                plt.setp(ax_res.get_yticklabels(), visible=False)
                plt.setp(ax_main.get_yticklabels(), visible=False)
            
            ax_main.set_xlabel('')
            
            if i == 0:
                ax_res.text(0.5, 1.45, titles[j], transform=ax_res.transAxes, 
                            fontsize=12, fontweight='bold', ha='center', va='bottom')
    
    # Labels centralizados para cada linha
    param_positions = {'teff': 0.670, 'logg': 0.360, 'feh': 0.055}
    for param in param_order:
        pinfo = PARAM_MAP.get(param, PARAM_MAP['teff'])
        xlabel = _get_xlabel(pinfo, survey_name)
        fig.text(0.485, param_positions.get(param, 0.5), xlabel, ha='center', va='top', fontsize=10)
        
    
    # Colorbars - posição vertical individual para cada parâmetro
    colorbar_positions = {'teff': 0.675, 'logg': 0.3678, 'feh': 0.06}
    for idx, param in enumerate(param_order):
        key_right = f'{param}_Less Restricted'
        if key_right in results_dict and results_dict[key_right] is not None:
            bins = bins_dict.get(param, None)
            if bins is not None:
                pinfo = PARAM_MAP.get(param, PARAM_MAP['teff'])
                custom_cmap = _create_custom_cmap(pinfo['cmap'])
                n_colors = getattr(custom_cmap, 'N', 256)
                norm = mcolors.BoundaryNorm(bins, n_colors)
                sm = cm.ScalarMappable(norm=norm, cmap=custom_cmap)
                sm.set_array([])
                
                row_position = colorbar_positions.get(param, 0.68 - 0.315*idx)
                cax = fig.add_axes([0.92, row_position, 0.011, 0.275])
                cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=8)
                
                if len(bins) > 1:
                    tick_locs = [(bins[k]+bins[k+1])/2 for k in range(len(bins)-1)]
                    cbar.set_ticks(tick_locs)
                    cbar.ax.set_yticklabels([f"[{bins[k]}, {bins[k+1]})" for k in range(len(bins)-1)], 
                                            fontsize=8)
    
    return fig

# Individual graphics functions

def _plot_regression_with_residuals_panels(ax_main, ax_res, y_true, y_pred, bins=None, param_name=None, 
                                          point_size=3, metrics_json_path=None, training_id=None, survey_name=None):
    """Plota painéis de regressão e resíduos em eixos fornecidos."""
    param_key = _detect_param_key(param_name)
    pinfo = PARAM_MAP[param_key]
    
    xlabel = _get_xlabel(pinfo, survey_name)
    ylabel = pinfo['ylabel']
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2, mad, sigma, residuals = _calculate_metrics(y_true, y_pred)
    metrics_str = _load_metrics_from_json(metrics_json_path, r2, mad, pinfo['unit'])
    
    custom_cmap = _create_custom_cmap(pinfo['cmap'])
    c = np.digitize(y_true, bins) if bins is not None else y_true
    
    # Painel de resíduos
    ax_res.scatter(y_true, residuals, c=c, cmap=custom_cmap, s=point_size, alpha=0.7)
    ax_res.set_facecolor("#ffffff")
    ax_res.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_res.axhline(3*sigma, color='k', linestyle='--', linewidth=1)
    ax_res.axhline(-3*sigma, color='k', linestyle='--', linewidth=1)
    ax_res.set_ylabel("Residuals", fontsize=10)
    ax_res.tick_params(axis='y', labelsize=9)
    ax_res.set_xticks([])
    
    # Painel principal
    ax_main.scatter(y_true, y_pred, c=c, cmap=custom_cmap, s=point_size, alpha=0.7)
    ax_main.set_facecolor('#ffffff')
    minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax_main.plot([minv, maxv], [minv, maxv], 'k-', lw=1, zorder=2)
    
    for coll in ax_main.collections:
        coll.set_zorder(1)
        
    ax_main.set_xlabel(xlabel, fontsize=10)
    ax_main.set_ylabel(ylabel, fontsize=10)
    ax_main.tick_params(axis='both', labelsize=9)
    
    if metrics_str is not None:
        ax_main.legend([metrics_str], loc="upper left", fontsize=10, frameon=True)
    else:
        ax_main.text(0.05, 0.95, f"R² Score: {r2:.4f}", transform=ax_main.transAxes, fontsize=9, va='top')
        ax_main.text(0.95, 0.05, f"MAD: {mad:.2f} {pinfo['unit']}", transform=ax_main.transAxes, fontsize=9, ha='right', va='bottom')

def _plot_regression_panel(ax, y_true, y_pred, bins=None, param_name=None, point_size=3, 
                          metrics_json_path=None, training_id=None, survey_name=None):
    """Função auxiliar para desenhar apenas o painel principal de regressão em um ax fornecido."""
    param_key = _detect_param_key(param_name)
    pinfo = PARAM_MAP[param_key]
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2, mad, sigma, residuals = _calculate_metrics(y_true, y_pred)
    metrics_str = _load_metrics_from_json(metrics_json_path, r2, mad, pinfo['unit'])
    
    custom_cmap = _create_custom_cmap(pinfo['cmap'])
    c = np.digitize(y_true, bins) if bins is not None else y_true
    ax.scatter(y_true, y_pred, c=c, cmap=custom_cmap, s=point_size, alpha=0.7)
    
    minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([minv, maxv], [minv, maxv], 'k-', lw=1, zorder=2)
    ax.set_xlabel(f"{pinfo['name']} True ({pinfo['unit']})")
    ax.set_ylabel(pinfo['ylabel'])
    
    if metrics_str is not None:
        ax.legend([metrics_str], loc="upper left", fontsize=9, frameon=True)
    else:
        ax.text(0.05, 0.95, f"R² Score: {r2:.4f}", transform=ax.transAxes, fontsize=9, va='top')
        ax.text(0.95, 0.05, f"MAD: {mad:.2f} {pinfo['unit']}", transform=ax.transAxes, fontsize=9, ha='right', va='bottom')

def plot_feature_importance(df_feat, param, figsize=(8, 6), n_top_features=10):
    """
    Plota gráfico horizontal de importância das features.
    
    Args:
        df_feat: DataFrame com colunas 'feature' e 'importance'
        param: nome do parâmetro
        figsize: tamanho da figura
        n_top_features: número de features mais importantes a exibir
    """
    df_plot = df_feat.sort_values('importance', ascending=False).head(n_top_features)
    features = df_plot['feature']
    importances = df_plot['importance']
    importances_pct = 100 * importances / importances.sum()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(features, importances_pct, color='black')
    ax.set_xlabel('Importance (%)', fontdict=label_fontdict)
    ax.set_ylabel('')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.linspace(0, 100, 11))
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    
    for bar, val in zip(ax.patches, importances_pct):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=tick_size-2)
    
    ax.text(0.98, 0.98, str(param), transform=ax.transAxes, ha='right', va='top',
            fontsize=label_fontdict['size']+2, fontweight=label_fontdict.get('weight', 'normal'))
    
    plt.tight_layout()
    plt.show()

def plot_test_graphs(predictions, true_values, bins, cmap_name, param_string, param_unit, n_ticks, legend):
    """Plota gráficos de teste para avaliação de modelo."""
    if not bins:
        bins = [min(true_values) - 1, max(true_values) + 1]

    df = pd.merge(left=true_values, left_index=True, right=predictions, right_index=True)
    df.columns = ["TRUE_VALUE", "PREDICTION"]
    df["ERROR"] = df["PREDICTION"] - df["TRUE_VALUE"]

    fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.6, 0.4], figsize=(15, 6))
    colors = mpl.colormaps[cmap_name](np.linspace(0.5, 1.00, len(bins) - 1))

    bins_intervals = []
    for bin_index in range(0, len(bins) - 1):
        bin_min, bin_max = bins[bin_index], bins[bin_index + 1]
        bins_intervals.append(f"[{bin_min} {param_unit}, {bin_max} {param_unit}]")
        df_bin = df[(df["TRUE_VALUE"] >= bin_min) & (df["TRUE_VALUE"] < bin_max)].copy()
        sns.scatterplot(data=df_bin, x="PREDICTION", y="TRUE_VALUE", ax=ax[0], s=9,
                       color=colors[bin_index], linewidth=0, zorder=2)
        sns.kdeplot(data=df_bin, x="ERROR", ax=ax[1], color=colors[bin_index])

    handles = [plot_handles(ax[0], "s", colors[i]) for i in range(len(bins_intervals))]
    min_lim_x = round_to_n(bins[0] - (bins[-1] - bins[0]) * 0.05, 1)
    max_lim_x = round_to_n(bins[-1] + (bins[-1] - bins[0]) * 0.05, 1)
    ax[0].plot([min_lim_x, max_lim_x], [min_lim_x, max_lim_x], ls="--", lw=1.5, color="k", zorder=3)
    ax[0] = beautify_graph(ax=ax[0], x_limits=[min_lim_x, max_lim_x], y_limits=[min_lim_x, max_lim_x],
                           x_n_ticks=n_ticks, y_n_ticks=n_ticks, x_label=f"Predicted {param_string}",
                           y_label=f"True {param_string}", grid=True)

    min_lim_x = round_to_n(-(df["ERROR"].abs().median() * 20), 1)
    max_lim_x = round_to_n((df["ERROR"].abs().median() * 20), 1)
    y_maxes = [max(line.get_data()[1]) for line in ax[1].lines]
    max_lim_y = max(y_maxes) * 1.1
    ax[1].plot([0, 0], [0, max_lim_y], ls="--", lw=1.5, color="k", zorder=3)
    ax[1] = beautify_graph(ax=ax[1], x_limits=[min_lim_x, max_lim_x], y_limits=[0, max_lim_y],
                           x_n_ticks=n_ticks, y_n_ticks=n_ticks, x_label="Error", y_label="Density", grid=True)
    ax[1].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    if legend:
        leg = fig.legend(handles, bins_intervals, title=f"{param_string}", 
                        title_fontproperties=legend_title_fontdict, ncols=len(bins_intervals),
                        loc="upper center", bbox_to_anchor=(0.5, -0.075), framealpha=1,
                        prop=legend_label_fontdict, markerscale=3, borderpad=1)
        leg._legend_box.sep = 20

    return fig

def plot_comparison_graph(results, metric, error, cmap_name, param_unit, n_ticks, legend):
    """Plota gráfico de comparação entre diferentes modelos/configurações."""
    fig = plt.figure(figsize=(3 * len(results[next(iter(results))]), 5))
    ax = fig.add_axes([0, 0, 1, 1])
    n_bars = len(results)
    bar_width = (1 - 1/n_bars)/n_bars
    hatches = [''] * int(n_bars/2) + ['/'] * int(n_bars/2) 
    paddings = np.arange(0, n_bars * bar_width, bar_width) - (bar_width/2 * (n_bars - 1))
    colors = mpl.colormaps[cmap_name](np.linspace(0.25, 0.75, n_bars))

    for index, key in enumerate(results):
        ax.bar(x=results[key].index + paddings[index], height=results[key][metric],
               yerr=results[key][error], width=bar_width, color=colors[index],
               hatch=hatches[index], edgecolor='k', linewidth=2.5, capsize=5,
               error_kw={'elinewidth': 3}, label=key, zorder=2)

    max_lim_y = max([(x[metric] + x[error]).max() for x in list(results.values())])
    ax = beautify_graph(ax=ax, x_limits=None, y_limits=[0, round_to_n(max_lim_y + (max_lim_y * 0.1), 2)],
                        x_n_ticks=None, y_n_ticks=n_ticks, x_label='Parameter Interval',
                        y_label=f'MAD ({param_unit})')
    ax.set_xticks(ticks=results[key].index, labels=results[key]['bin'])
    ax.grid(axis='y', zorder=0)

    if legend:
        leg = ax.legend(title='Features', title_fontproperties=legend_title_fontdict,
                       prop=legend_label_fontdict, framealpha=1, handlelength=3,
                       handleheight=1.5, borderpad=1, bbox_to_anchor=(1.01, 1))
        leg._legend_box.sep = 20

    return fig

def beautify_graph(ax, x_limits, y_limits, x_n_ticks, y_n_ticks, x_label, y_label, grid=None):
    """Aplica estilo padronizado aos gráficos."""
    if x_limits:
        ax.set_xlim(x_limits[0], x_limits[1])
        ax.set_xticks(np.linspace(x_limits[0], x_limits[1], x_n_ticks))
    if y_limits:
        ax.set_ylim(y_limits[0], y_limits[1])
        ax.set_yticks(np.linspace(y_limits[0], y_limits[1], y_n_ticks))
    ax.tick_params(labelsize=tick_size)
    ax.set_xlabel(x_label, fontdict=label_fontdict, labelpad=15)
    ax.set_ylabel(y_label, fontdict=label_fontdict, labelpad=15)
    if grid:
        ax.grid(zorder=0)
    return ax

def round_to_n(x, n):
    """Arredonda número para n dígitos significativos."""
    return round(x, -int(floor(log10(abs(x)))) + n - 1)

def plot_handles(ax, m, c):
    """Cria handle para legenda."""
    return ax.plot([], [], marker=m, color=c, ls="None")[0]

def plot_regression_with_residuals(y_true, y_pred, bins=None, param_name=None, param_unit=None, cmap=None, 
                                   point_size=3, metrics_json_path=None, training_id=None, survey_name=None):
    """
    Gera um gráfico padrão para avaliação de regressão com painel de resíduos.
    
    Args:
        y_true: valores verdadeiros
        y_pred: valores preditos
        bins: bins para coloração
        param_name: nome do parâmetro
        param_unit: unidade do parâmetro
        cmap: colormap
        point_size: tamanho dos pontos
        metrics_json_path: caminho para arquivo JSON com métricas
        training_id: ID do treinamento (opcional)
        survey_name: código do survey (A, L, G, W)
    
    Returns:
        fig: figura matplotlib
    """
    param_key = _detect_param_key(param_name)
    pinfo = PARAM_MAP[param_key]
    
    if param_unit is None:
        param_unit = pinfo['unit']
    if cmap is None:
        cmap = pinfo['cmap']
    
    xlabel = _get_xlabel(pinfo, survey_name)
    ylabel = pinfo['ylabel']
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2, mad, sigma, residuals = _calculate_metrics(y_true, y_pred)
    metrics_str = _load_metrics_from_json(metrics_json_path, r2, mad, pinfo['unit'])
    
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 3.5], hspace=0.05)
    ax_main = fig.add_subplot(gs[1])
    ax_res = fig.add_subplot(gs[0], sharex=ax_main)
    plt.setp(ax_res.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.98, bottom=0.08, hspace=0.05)
    
    custom_cmap = _create_custom_cmap(cmap)
    c = np.digitize(y_true, bins) if bins is not None else y_true
    
    # Painel de resíduos
    ax_res.scatter(y_true, residuals, c=c, cmap=custom_cmap, s=point_size, alpha=0.7)
    ax_res.set_facecolor("#ffffff")
    ax_res.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_res.axhline(3*sigma, color='k', linestyle='--', linewidth=1)
    ax_res.axhline(-3*sigma, color='k', linestyle='--', linewidth=1)
    ax_res.set_ylabel("Residuals", fontsize=8)
    ax_res.set_xticks([])
    
    # Painel principal
    ax_main.scatter(y_true, y_pred, c=c, cmap=custom_cmap, s=point_size, alpha=0.7)
    ax_main.set_facecolor('#ffffff')
    
    if bins is not None:
        n_colors = getattr(custom_cmap, 'N', 256)
        norm = mcolors.BoundaryNorm(bins, n_colors)
        sm = cm.ScalarMappable(norm=norm, cmap=custom_cmap)
        sm.set_array([])
        cax = fig.add_axes([0.98, 0.08, 0.013, 0.90])
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical', aspect=60)
        
        if len(bins) > 1:
            tick_locs = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            cbar.set_ticks(tick_locs)
            cbar.ax.set_yticklabels([f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)])
    
    minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax_main.plot([minv, maxv], [minv, maxv], 'k-', lw=1, zorder=2)
    
    for coll in ax_main.collections:
        coll.set_zorder(1)
        
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    
    if metrics_str is not None:
        ax_main.legend([metrics_str], loc="upper left", fontsize=9, frameon=True)
    else:
        ax_main.text(0.05, 0.95, f"R² Score: {r2:.4f}", transform=ax_main.transAxes, fontsize=9, va='top')
        ax_main.text(0.95, 0.05, f"MAD: {mad:.2f} {pinfo['unit']}", transform=ax_main.transAxes, fontsize=9, ha='right', va='bottom')
    
    return fig

def plot_bolometric_correction(y_true, y_pred, point_size=8, metrics_json_path=None, model_type='XGB'):
    """
    Gera gráfico de regressão para correção bolométrica com estilo específico.
    
    Args:
        y_true: valores verdadeiros da correção bolométrica
        y_pred: valores preditos da correção bolométrica
        point_size: tamanho dos pontos (default: 8)
        metrics_json_path: caminho para arquivo JSON com métricas (opcional)
        model_type: tipo de modelo ('XGB' ou 'RF') para definir cor (default: 'XGB')
    
    Returns:
        fig: figura matplotlib
    """
    _set_times_font()
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2, mad, sigma, residuals = _calculate_metrics(y_true, y_pred)
    
    # Carregar métricas de JSON se disponível
    metrics_str = None
    if metrics_json_path is not None and os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, 'r') as f:
                metrics = json.load(f)
            metrics_str = f"R² = {metrics.get('r2', r2):.4f} | MAD = {metrics.get('mad', mad):.3f} mag"
        except:
            pass
    
    # Criar figura com painel de resíduos
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 3.5], hspace=0.05)
    ax_main = fig.add_subplot(gs[1])
    ax_res = fig.add_subplot(gs[0], sharex=ax_main)
    plt.setp(ax_res.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.98, bottom=0.08, hspace=0.05)
    
    # Definir cor baseada no tipo de modelo
    color = 'MediumVioletRed' if model_type.upper() == 'XGB' else 'MediumOrchid'
    ax_res.scatter(y_true, residuals, c=color, s=point_size, alpha=0.7)
    ax_res.set_facecolor("#ffffff")
    ax_res.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_res.axhline(3*sigma, color='k', linestyle='--', linewidth=1)
    ax_res.axhline(-3*sigma, color='k', linestyle='--', linewidth=1)
    ax_res.set_ylabel("Residuals", fontsize=8)
    ax_res.tick_params(axis='y', labelsize=8)
    ax_res.set_xticks([])
    
    # Painel principal - cor roxa
    ax_main.scatter(y_true, y_pred, c=color, s=point_size, alpha=0.7)
    ax_main.set_facecolor('#ffffff')
    
    # Linha 1:1
    minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax_main.plot([minv, maxv], [minv, maxv], 'k-', lw=1, zorder=2)
    
    # Configurar z-order
    for coll in ax_main.collections:
        coll.set_zorder(1)
    
    ax_main.set_xlabel('BC Jordi et. al. (2010) (mag)', fontsize=10)  # Labels principais: 10
    ax_main.set_ylabel('BC Predicted (mag)', fontsize=10)  # Labels principais: 10
    ax_main.tick_params(axis='both', labelsize=9)
    
    # Adicionar métricas
    if metrics_str is not None:
        ax_main.legend([metrics_str], loc="upper left", fontsize=9, frameon=True)  # Legenda: 9
    else:
        metrics_text = f"R² = {r2:.4f} | MAD = {mad:.3f}"
        ax_main.legend([metrics_text], loc="upper left", fontsize=9, frameon=True)  # Legenda: 9

    return fig

def show(fig):
    """Exibe o gráfico gerado (figura matplotlib)."""
    plt.show(fig)

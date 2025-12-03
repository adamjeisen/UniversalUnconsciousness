import h5py
import itertools
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.transforms as transforms
import numpy as np
import os
import scipy
import scipy.stats as stats
from scipy.stats import wilcoxon
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statsmodels.sandbox.stats.multicomp import multipletests
from .data_utils import get_loc_roc, get_section_info
from .sensory_responses import get_responses_etdc, get_responses_acf

# Set up figure based on data class
monkey_titles = {
    'Mary': 'NHP 1',
    'MrJones': 'NHP 2',
    'SPOCK': 'NHP 3',
    'PEDRI': 'NHP 4'
}

def clear_axes(ax):
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Remove ticks on the top and right axes
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def load_font():
    if os.path.exists('/om2/user/eisenaj'):
        font_path = "/om2/user/eisenaj/miniforge3/envs/communication-transformer/fonts/arial.ttf"
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        # return prop
    elif os.path.exists('/home/eisenaj'):
        font_path = "/home/eisenaj/miniforge3/envs/communication-jacobians/fonts/arial.ttf"
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        # return prop

def plot_curve_with_se(curve, x_vals=None, label=None, linestyle=None, c=None, alpha=1, alpha_se=0.5, ax=None):
    if ax is None:
        ax = plt.gca()
    mean_vals = curve.mean(axis=0)
    se_vals = curve.std(axis=0)/np.sqrt(curve.shape[0])
    if x_vals is None:
        x_vals = np.arange(len(mean_vals))
    ln = ax.plot(x_vals, mean_vals, label=label, alpha=alpha, color=c, linestyle=linestyle)
    ax.fill_between(x_vals, mean_vals - se_vals, mean_vals + se_vals, alpha=alpha_se, color=c)

    return ln

def _process_stability(x, top_percent):
    """Helper function to process stability parameters into timescales."""
    # Keep only negative roots and convert to timescales
    timescales = -1/np.array([r for r in x if r < 0])
    if len(timescales) == 0:
        return np.nan
    # Take only top percentage of timescales
    n_keep = max(1, int(len(timescales) * top_percent))
    top_timescales = np.sort(timescales)[-n_keep:]
    if len(top_timescales) == 0:
        print(timescales)
    return np.exp(np.mean(np.log(top_timescales)))  # geometric mean

def _calculate_baseline(stability_params, time_vals, infusion_time, plot_range, top_percent):
    """Helper function to calculate baseline timescales."""
    pre_infusion_mask = (time_vals >= (infusion_time + plot_range[0]*60)) & (time_vals < infusion_time)
    baseline_timescales = []
    for params in stability_params[pre_infusion_mask]:
        timescales = -1/np.array([r for r in params if r < 0])
        if len(timescales) > 0:
            n_keep = max(1, int(len(timescales) * top_percent))
            top_timescales = np.sort(timescales)[-n_keep:]
            baseline_timescales.extend(top_timescales)
    if len(baseline_timescales) == 0:
        return np.nan
    return np.exp(np.mean(np.log(baseline_timescales)))

def _process_session_data(session, delase_results, session_file, infusion_time, common_times, top_percent, plot_range, interpolate=True, timescales=True, area='all', data_type='delase'):
    """Helper function to process individual session data."""
    time_vals = delase_results[session][area].window_start.values
    
    if data_type == 'delase':
        data_input = delase_results[session][area].stability_params
    elif data_type == 'var':
        data_input = delase_results[session][area].stability_params_A
    elif data_type == 'var_small':
        data_input = delase_results[session][area].stability_params_A_small
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    if timescales:
    # Calculate baseline and normalized timescales
        baseline = _calculate_baseline(
            data_input,
            time_vals, 
            infusion_time,
            plot_range,
            top_percent
        )

        if baseline == np.nan:
            return np.nan

        stability_vals = data_input.apply(
            lambda x: _process_stability(x, top_percent)
        ).values 
        normalized_stability_vals = stability_vals / baseline
    else:
        stability_vals = data_input.apply(lambda x: x[:int(top_percent*len(x))].mean())
        pre_infusion_mask = (time_vals >= (infusion_time + plot_range[0]*60)) & (time_vals < infusion_time)
        baseline = stability_vals[pre_infusion_mask].mean()
        normalized_stability_vals = stability_vals
    
    # Align to infusion start and interpolate
    aligned_times = (time_vals - infusion_time) / 60
    if interpolate:
        interpolated = np.interp(common_times, aligned_times, stability_vals)
        interpolated_normalized = np.interp(common_times, aligned_times, normalized_stability_vals)
    else:
        interpolated = stability_vals
        interpolated_normalized = normalized_stability_vals
    return interpolated, interpolated_normalized, baseline

def _plot_statistics(ax, aligned_data, common_times, curve_color, timescales=True, aligned_data_unnormalized=None, baseline=None, plot_stars=True, baseline_inverse_timescales=None, aligned_data_inverse_timescales=None, label=None, star_offset_idx=0, star_offset_step=0.025, star_height_base=1.01):
    """Helper function to plot geometric mean and standard error."""
    if timescales:
        log_data = np.log(aligned_data)
        mean_log = np.nanmean(log_data, axis=0)
        sem_log = np.nanstd(log_data, axis=0) / np.sqrt(np.sum(~np.isnan(log_data), axis=0))
        
        mean_stability = np.exp(mean_log)
        upper_bound = np.exp(mean_log + sem_log)
        lower_bound = np.exp(mean_log - sem_log)
        ax.plot(common_times, mean_stability, label=(label if label is not None else 'Geometric Mean'), color=curve_color)
        ax.fill_between(common_times, lower_bound, upper_bound, alpha=0.3, color=curve_color)
    else:
        mean_stability = np.nanmean(aligned_data, axis=0)
        sem_stability = np.nanstd(aligned_data, axis=0) / np.sqrt(np.sum(~np.isnan(aligned_data), axis=0))
        ax.plot(common_times, mean_stability, label=(label if label is not None else 'Mean'), color=curve_color)
        ax.fill_between(common_times, mean_stability - sem_stability, mean_stability + sem_stability, alpha=0.3, color=curve_color)
    
    if aligned_data_unnormalized is not None and baseline is not None and plot_stars:
        # For each time point, perform Wilcoxon paired rank test vs baseline
        p_vals_all = np.zeros(aligned_data_unnormalized.shape[1])
        for t in range(aligned_data_unnormalized.shape[1]):
            # Extract data for this time point, removing NaN values
            # time_data = aligned_data_unnormalized[:, t]
            time_data = aligned_data_inverse_timescales[:, t]
            valid_indices = ~np.isnan(time_data)
            if np.sum(valid_indices) < 2:
                continue  # Skip if not enough data points
                
            current_data = time_data[valid_indices]
            # baseline_values = np.ones_like(current_data) * baseline  # Compare with baseline
            baseline_values = np.ones_like(baseline_inverse_timescales) * baseline_inverse_timescales  # Compare with baseline
            try:
                # Perform Wilcoxon paired rank test
                stat, p_val = wilcoxon(current_data, baseline_values)
                p_vals_all[t] = p_val
                # If significant, add a star at this position
            except:
                # Skip if the test fails (e.g., due to identical values)
                continue
        
        # print("CORRECTING FOR MULTIPLE TESTING")
        reject, p_adjusted, _, _ = multipletests(
                                    p_vals_all,
                                    alpha=0.05,
                                    method='fdr_bh'
                                )
        # print(p_vals_all)
        # print(p_adjusted)

        # Place stars using a blended transform; y in axes-fraction with slight per-series offset.
        # Keep them just above the data region but below/away from axis titles.
        star_y_axes_frac = star_height_base + star_offset_idx * star_offset_step  # small vertical offset per area
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for t in range(len(p_vals_all)):
            if reject[t]:
                ax.text(common_times[t], star_y_axes_frac, '*',
                        transform=trans, ha='center', va='bottom',
                        color=curve_color, fontsize=7, clip_on=False)

def _add_roc_ropap_lines(ax, rocs, ropaps, is_propofol=False):
    """Helper function to add ROC and ROPAP lines to plot."""
    ylim = ax.get_ylim()
    roc_vals = np.array(rocs)/60
    
    mean_roc = np.mean(roc_vals)
    sem_roc = np.std(roc_vals) / np.sqrt(len(roc_vals))
    ax.axvline(mean_roc, c='g', ls='-', label=f"ROC mean: {mean_roc:.2f} ± {sem_roc:.2f}")
    ax.fill_betweenx(ylim, mean_roc - sem_roc, mean_roc + sem_roc, alpha=0.3, color='g')
    
    if not is_propofol:
        ropap_vals = np.array(ropaps)/60
        mean_ropap = np.mean(ropap_vals)
        sem_ropap = np.std(ropap_vals) / np.sqrt(len(ropap_vals))
        ax.axvline(mean_ropap, c='orange', ls='-', label=f"ROPAP mean: {mean_ropap:.2f} ± {sem_ropap:.2f}")
        ax.fill_betweenx(ylim, mean_ropap - sem_ropap, mean_ropap + sem_ropap, alpha=0.3, color='orange')

def _add_loc_roc_region(ax, locs, rocs, loc_roc_color):
    """Helper function to add shaded region between LOC and ROC."""
    ylim = ax.get_ylim()
    loc_vals = np.array(locs)/60
    roc_vals = np.array(rocs)/60
    
    mean_loc = np.mean(loc_vals)
    mean_roc = np.mean(roc_vals)
    
    ax.axvspan(mean_loc, mean_roc, color=loc_roc_color, alpha=0.1)

def plot_roc_vs_max_timescale(cfg, agent, session_lists, delase_results, locs, rocs, ropaps, plot_range=(-15, 85), top_percent=0.1,
                             curve_colors={'propofol': 'blue', 'dexmedetomidine': 'purple', 'ketamine': 'red'},
                             figsize=None, dose=None, save_path=None, verbose=False):
    """Plot ROC time vs maximum timescale ratio for each session.
    
    Args:
        cfg: Config object
        agent: String indicating anesthetic agent ('propofol', 'dexmedetomidine', or 'ketamine')
        session_lists: Dict of sessions by monkey and dose
        delase_results: Dict of DeLASE results by session
        locs, rocs, ropaps: Lists of LOC, ROC and ROPAP times
        top_percent: Float indicating percent of eigenvalues to use for stability calculation
        curve_colors: Dict mapping agents to plot colors
        figsize: Tuple of figure dimensions
        dose: String indicating dose to plot ('low' or 'high'), if None plots both
        save_path: Path to save figure
        verbose: Bool to print debug info
    """
    is_lever = 'ket' in agent.lower() or 'dex' in agent.lower()
    curve_color = curve_colors[agent]

    if figsize is None:
        figsize = (3, 2)
    
    fig, axs = plt.subplots(1,1, figsize=figsize)
    if is_lever:
        if dose is None:
            doses = ['low', 'high']
        else:
            doses = [dose]
        monkeys = ['SPOCK', 'PEDRI']
    else:
        if dose is None:
            doses = ['high']
        else:
            doses = [dose]
        monkeys = ['Mary', 'MrJones']

    # Create common time grid
    # common_times = np.arange(plot_range[0], plot_range[1], 1/60) # in minutes
    common_times = np.arange(plot_range[0], plot_range[1], 1/2) # in minutes

    ax = axs
    max_ratios = []
    roc_times = []
    # Process data for each subplot
    for i, monkey in enumerate(monkeys):
        # ax = axs[i]
        # ax.set_title(f"{monkey_titles[monkey]}")
        for j, dose_level in enumerate(doses):
            
            # Get max timescale ratio and ROC time for each session
            for session_ind, session in enumerate(session_lists[monkey][dose_level]):
                if is_lever:
                    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))
                    infusion_time = session_file['sessionInfo']['infusionStart'][0, 0]
                else:
                    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f"{session}.mat"))
                    infusion_time = session_file['sessionInfo']['drugStart'][0]
                stability_vals = _process_session_data(session, delase_results, session_file, infusion_time, 
                                                    common_times, top_percent, plot_range, interpolate=False, area=area)
                if np.isnan(stability_vals).all():
                    continue
                max_ratios.append(np.max(stability_vals))
                roc_times.append(rocs[monkey][dose_level][session_ind]/60)  # Convert to minutes
            
    # Plot scatter of ROC time vs max ratio
    ax.scatter(max_ratios, roc_times, c=curve_color, alpha=0.6)
            
    # # Add trend line
    # z = np.polyfit(max_ratios, roc_times, 1)
    # p = np.poly1d(z)
    # x_trend = np.linspace(min(max_ratios), max(max_ratios), 100)
    # ax.plot(x_trend, p(x_trend), '--', c=curve_color, alpha=0.3)
            
    ax.set_xlabel('Max Timescale Ratio')
    ax.set_ylabel('ROC Time (min)')

    r, p = scipy.stats.pearsonr(max_ratios, roc_times)
    fig.suptitle(f'{agent.capitalize()}\nr={r:.2f}, p={p:.2e}', c=curve_color, y=0.9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()


def plot_session_stability_grouped(cfg, agent, session_lists, delase_results, locs, rocs, ropaps, plot_range=(-15, 85), top_percent=0.1,
                                  timescales=True,
                                  curve_colors={'propofol': 'blue', 'dexmedetomidine': 'purple', 'ketamine': 'red'},
                                  loc_roc_colors={'propofol': 'midnightblue', 'dexmedetomidine': 'darkviolet', 'ketamine': 'darkred'},
                                  figsize=None,
                                  dose=None,
                                  save_path=None,
                                  verbose=False,
                                  area='all',
                                  return_data=False,
                                  data_type='delase',
                                  axs_override=None,
                                  fig_override=None,
                                  suppress_suptitle=False,
                                  skip_tight_layout=False,
                                  add_legend=True,
                                  legend_outside=False,
                                  legend_fontsize=8,
                                  star_height_base=1.01):
    is_lever = 'ket' in agent.lower() or 'dex' in agent.lower()

    curve_color = curve_colors[agent]
    loc_roc_color = loc_roc_colors[agent]

    # Figure/axes setup (support external grid composition)
    if axs_override is None:
        if figsize is None:
            figsize = (3, 2)
        if is_lever:
            if dose is None:
                fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
                doses = ['low', 'high']
            else:
                fig, axs = plt.subplots(1, 2, figsize=figsize)
                doses = [dose]
            
            # fig.suptitle('Average DeLASE Results by Monkey and Dose (Aligned to Infusion)', fontsize=14)
            monkeys = ['SPOCK', 'PEDRI']
        else:
            if dose is None:
                fig, axs = plt.subplots(1, 2, figsize=figsize)
                doses = ['high']
            else:
                fig, axs = plt.subplots(1, 2, figsize=figsize)
                doses = [dose]
            # fig.suptitle('Average DeLASE Results by Monkey (Aligned to Infusion)', fontsize=14)
            monkeys = ['Mary', 'MrJones']
            # axs = axs.reshape(1, -1)  # Make 2D for consistent indexing
        # capitalize the first letter of agent in the title
        if not suppress_suptitle:
            fig.suptitle(f'{agent.capitalize()}', c=curve_color, y=0.9)
    else:
        # Use provided axes for a 1x2 layout
        fig = fig_override if fig_override is not None else plt.gcf()
        axs = np.array(axs_override)
        if is_lever:
            doses = [dose] if dose is not None else ['high']
            monkeys = ['SPOCK', 'PEDRI']
        else:
            doses = [dose] if dose is not None else ['high']
            monkeys = ['Mary', 'MrJones']

    # Create common time grid
    # common_times = np.arange(plot_range[0], plot_range[1], 1/60) # in minutes
    common_times = np.arange(plot_range[0], plot_range[1], 1/4) # in minutes
    
    # Normalize area to a list and prepare colors if multiple areas
    if isinstance(area, (list, tuple, np.ndarray)):
        areas_list = list(area)
    else:
        areas_list = [area]
    multiple_areas = len(areas_list) > 1
    if multiple_areas:
        cmap = plt.cm.get_cmap('tab10', len(areas_list))
        area_color_map = {a: cmap(i) for i, a in enumerate(areas_list)}
    else:
        area_color_map = {areas_list[0]: curve_color}

    if return_data:
        plot_return_data = {
            'agent': agent,
            'timescales': timescales,
            'plot_range': plot_range,
            'top_percent': top_percent,
            'common_times': common_times,
            'groups': {}
        }
        if multiple_areas:
            plot_return_data['areas'] = areas_list
        else:
            plot_return_data['area'] = areas_list[0]
    
    # Process data for each subplot
    for i, monkey in enumerate(monkeys):
        for j, dose in enumerate(doses):
            ax = axs[i, j] if len(doses) > 1 else axs[i]
            if len(doses) == 1:
                ax.set_title(f"{monkey_titles[monkey]}", pad=8, fontsize=10)
            else:
                ax.set_title(f"{monkey_titles[monkey]} {dose} dose", pad=8, fontsize=10)
            
            # Accumulate per-area data
            per_area_aligned_data = {a: [] for a in areas_list}
            per_area_aligned_data_unnormalized = {a: [] for a in areas_list}
            per_area_baselines = {a: [] for a in areas_list}
            per_area_baselines_inverse_timescales = {a: [] for a in areas_list}
            per_area_aligned_data_inverse_timescales = {a: [] for a in areas_list}
            if dose not in session_lists[monkey]:
                continue
            for session in session_lists[monkey][dose]:
                if is_lever:
                    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))
                    infusion_time = session_file['sessionInfo']['infusionStart'][0, 0]
                else:
                    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f"{session}.mat"))
                    infusion_time = session_file['sessionInfo']['drugStart'][0]
                
                for area_name in areas_list:
                    interpolated, interpolated_normalized, baseline = _process_session_data(
                        session, delase_results, session_file, infusion_time,
                        common_times, top_percent, plot_range, timescales=timescales, area=area_name, data_type=data_type
                    )
                    interpolated_inverse_timescales, _, baseline_inverse_timescales = _process_session_data(
                        session, delase_results, session_file, infusion_time,
                        common_times, top_percent, plot_range, timescales=False, area=area_name, data_type=data_type
                    )

                    per_area_baselines[area_name].append(baseline)
                    per_area_baselines_inverse_timescales[area_name].append(baseline_inverse_timescales)
                    per_area_aligned_data[area_name].append(interpolated_normalized)
                    per_area_aligned_data_unnormalized[area_name].append(interpolated)
                    per_area_aligned_data_inverse_timescales[area_name].append(interpolated_inverse_timescales)

            # Plot per-area statistics
            per_area_group_data = {}
            for area_idx, area_name in enumerate(areas_list):
                aligned_data_arr = np.array(per_area_aligned_data[area_name])
                aligned_data_unnorm_arr = np.array(per_area_aligned_data_unnormalized[area_name])
                if timescales:
                    plot_stars = True
                    baseline_arr = np.array(per_area_baselines[area_name])
                    baseline_inverse_arr = np.array(per_area_baselines_inverse_timescales[area_name])
                    aligned_inverse_arr = np.array(per_area_aligned_data_inverse_timescales[area_name])
                else:
                    plot_stars = False
                    baseline_arr = None
                    baseline_inverse_arr = np.array(per_area_baselines_inverse_timescales[area_name])
                    aligned_inverse_arr = np.array(per_area_aligned_data_inverse_timescales[area_name])

                color_for_area = area_color_map[area_name]
                # Map area names for legend display
                area_label_map = {'7b': 'PPC', 'CPB': 'STG'}
                legend_label = area_label_map.get(area_name, area_name) if multiple_areas else None
                _plot_statistics(
                    ax,
                    aligned_data_arr,
                    common_times,
                    color_for_area,
                    timescales=timescales,
                    aligned_data_unnormalized=aligned_data_unnorm_arr,
                    baseline=baseline_arr,
                    plot_stars=plot_stars,
                    baseline_inverse_timescales=baseline_inverse_arr,
                    aligned_data_inverse_timescales=aligned_inverse_arr,
                    label=legend_label,
                    star_offset_idx=area_idx,
                    star_height_base=star_height_base
                )

                if return_data:
                    sessions_list = list(session_lists[monkey][dose])
                    group_data = {
                        'sessions': sessions_list,
                        'aligned_data_normalized': aligned_data_arr,
                        'aligned_data_unnormalized': aligned_data_unnorm_arr,
                        'aligned_data_inverse_timescales': aligned_inverse_arr if timescales else np.array(per_area_aligned_data_inverse_timescales[area_name]),
                        'baselines_timescales': np.array(per_area_baselines[area_name]) if timescales else None,
                        'baselines_inverse_timescales': baseline_inverse_arr if timescales else np.array(per_area_baselines_inverse_timescales[area_name]),
                    }
                    # Map each session to its aligned curves for convenience
                    per_session = {}
                    for idx_sess, sess_name in enumerate(sessions_list):
                        per_session[sess_name] = {
                            'aligned_normalized': aligned_data_arr[idx_sess] if aligned_data_arr.ndim == 2 and idx_sess < aligned_data_arr.shape[0] else np.array([]),
                            'aligned_unnormalized': aligned_data_unnorm_arr[idx_sess] if aligned_data_unnorm_arr.ndim == 2 and idx_sess < aligned_data_unnorm_arr.shape[0] else np.array([]),
                            'aligned_inverse_timescales': aligned_inverse_arr[idx_sess] if aligned_inverse_arr is not None and aligned_inverse_arr.ndim == 2 and idx_sess < aligned_inverse_arr.shape[0] else np.array([]),
                        }
                    group_data['per_session'] = per_session
                    if timescales:
                        log_data = np.log(aligned_data_arr)
                        valid_counts = np.sum(~np.isnan(log_data), axis=0)
                        mean_log = np.nanmean(log_data, axis=0)
                        sem_log = np.nanstd(log_data, axis=0) / np.sqrt(np.maximum(valid_counts, 1))
                        group_data.update({
                            'mean_curve': np.exp(mean_log),
                            'lower_bound': np.exp(mean_log - sem_log),
                            'upper_bound': np.exp(mean_log + sem_log),
                            'mean_log': mean_log,
                            'sem_log': sem_log,
                        })
                    else:
                        valid_counts = np.sum(~np.isnan(aligned_data_arr), axis=0)
                        mean_curve = np.nanmean(aligned_data_arr, axis=0)
                        sem_curve = np.nanstd(aligned_data_arr, axis=0) / np.sqrt(np.maximum(valid_counts, 1))
                        group_data.update({
                            'mean_curve': mean_curve,
                            'sem_curve': sem_curve,
                        })
                    per_area_group_data[area_name] = group_data
            # _add_roc_ropap_lines(ax, rocs[monkey][dose], ropaps[monkey][dose])
            _add_loc_roc_region(ax, locs[monkey][dose], rocs[monkey][dose], loc_roc_color)
            # Legend will be added after infusion start and baseline lines

            # Collect return data for this group if requested
            if return_data:
                if monkey not in plot_return_data['groups']:
                    plot_return_data['groups'][monkey] = {}
                if multiple_areas:
                    plot_return_data['groups'][monkey][dose] = {
                        'per_area': per_area_group_data,
                        'sessions': list(session_lists[monkey][dose]),
                        'locs': locs[monkey].get(dose, None),
                        'rocs': rocs[monkey].get(dose, None),
                        'ropaps': ropaps[monkey].get(dose, None) if isinstance(ropaps, dict) and monkey in ropaps else None,
                    }
                else:
                    only_area = areas_list[0]
                    gd = per_area_group_data[only_area]
                    merged = {
                        'sessions': gd['sessions'],
                        'aligned_data_normalized': gd['aligned_data_normalized'],
                        'aligned_data_unnormalized': gd['aligned_data_unnormalized'],
                        'aligned_data_inverse_timescales': gd['aligned_data_inverse_timescales'],
                        'baselines_timescales': gd['baselines_timescales'],
                        'baselines_inverse_timescales': gd['baselines_inverse_timescales'],
                        'per_session': gd.get('per_session', {}),
                        'locs': locs[monkey].get(dose, None),
                        'rocs': rocs[monkey].get(dose, None),
                        'ropaps': ropaps[monkey].get(dose, None) if isinstance(ropaps, dict) and monkey in ropaps else None,
                    }
                    # Include summary curves if present
                    for k in ['mean_curve', 'lower_bound', 'upper_bound', 'mean_log', 'sem_log', 'sem_curve']:
                        if k in gd:
                            merged[k] = gd[k]
                    plot_return_data['groups'][monkey][dose] = merged
                
    
    # Add common elements to all subplots
    for ax in np.array(axs).flat:
        ax.axvline(0, c='k', ls='--', label='Infusion Start')
        if timescales:
            ax.axhline(1, c='k', ls=':', label='Baseline')
        ax.set_xlabel('Time Relative to Infusion Start (min)')
        if timescales:
            ax.set_ylabel('Mean Characteristic Timescale\nRatio to Awake Baseline')
        else:
            ax.set_ylabel('Mean Instability ($s^{-1}$)')
        # Add legend on the right side if multiple areas and legend is requested
        if add_legend and multiple_areas:
            handles, labels = ax.get_legend_handles_labels()
            # Filter out duplicate labels (infusion start and baseline might be added multiple times)
            seen = set()
            unique_handles = []
            unique_labels = []
            for handle, label in zip(handles, labels):
                if label not in seen:
                    seen.add(label)
                    unique_handles.append(handle)
                    unique_labels.append(label)
            # Show legend with all elements (areas, infusion start, baseline)
            # Position it to the right of the plot using bbox_to_anchor
            if len(unique_labels) > 0:
                ax.legend(handles=unique_handles, labels=unique_labels, 
                         loc='center left', bbox_to_anchor=(1.02, 0.5), 
                         frameon=False, fontsize=legend_fontsize)
    
    if not skip_tight_layout:
        plt.tight_layout()
    if save_path is not None and axs_override is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    elif axs_override is None:
        plt.show()
    if return_data:
        return plot_return_data


def plot_session_stability_grouped_grid(cfg, agent_data,
                                        data_class=None,
                                        data_class_by_agent=None,
                                        top_percent=0.1,
                                        timescales=True,
                                        curve_colors={'propofol': 'blue', 'dexmedetomidine': 'purple', 'ketamine': 'red'},
                                        loc_roc_colors={'propofol': 'midnightblue', 'dexmedetomidine': 'darkviolet', 'ketamine': 'darkred'},
                                        figsize=None,
                                        dose=None,
                                        save_path=None,
                                        verbose=False,
                                        area='all',
                                        area_by_agent=None,
                                        plot_range_by_agent=None,
                                        return_data=False,
                                        data_type='delase',
                                        star_height_base=1.01,
                                        layout='agent_rows',
                                        add_legend=False,
                                        legend_fontsize=7):
    """
    Compose a grid of session stability grouped plots with configurable layout:
      - layout='agent_rows' (default): 3-row x 2-column grid
        - Rows: agents (Propofol, Ketamine, Dexmedetomidine)
        - Columns: NHPs (NHP 1/2 for Propofol; NHP 3/4 for Ketamine and Dexmedetomidine)
      - layout='nhp_rows': 2-row x 3-column grid
        - Columns: agents (Propofol, Ketamine, Dexmedetomidine)
        - Rows: NHPs (NHP 1/2 for Propofol; NHP 3/4 for Ketamine and Dexmedetomidine)

    Args:
        cfg: Config object
        agent_data: dict keyed by (data_class, agent) with entries containing:
            'session_lists', 'delase_results', 'locs', 'rocs', 'ropaps'
        data_class: (optional) single data_class to use for all agents
        data_class_by_agent: (optional) dict mapping agent -> data_class; overrides single.
        area: default area or list of areas to use for all agents if area_by_agent is None
        area_by_agent: (optional) dict mapping agent -> area (string or list)
        plot_range_by_agent: (optional) dict mapping agent -> (min, max) minutes window
        return_data: whether to return per-agent data from the underlying function
        star_height_base: base height for significance stars (in axes-fraction units, default 1.01)
        layout: 'agent_rows' for 3x2 grid (agents as rows) or 'nhp_rows' for 2x3 grid (NHPs as rows)
        figsize: tuple for figure size. If None, defaults to (9.0, 4.5) for agent_rows or (9.0, 3.2) for nhp_rows
    """
    agents_in_order = ['propofol', 'ketamine', 'dexmedetomidine']

    # Build data_class mapping, with auto-detection from keys
    dc_map = {}
    if isinstance(data_class_by_agent, dict):
        for a in agents_in_order:
            if a in data_class_by_agent:
                dc_map[a] = data_class_by_agent[a]
    if data_class is not None:
        for a in agents_in_order:
            dc_map.setdefault(a, data_class)
    if isinstance(agent_data, dict):
        for (dc, ag) in agent_data.keys():
            if ag in agents_in_order and ag not in dc_map:
                dc_map[ag] = dc

    # Validate layout parameter
    if layout not in ['nhp_rows', 'agent_rows']:
        raise ValueError(f"layout must be 'nhp_rows' or 'agent_rows', got '{layout}'")
    
    # Set default figsize based on layout
    if figsize is None:
        if layout == 'nhp_rows':
            figsize = (9.0, 3.2)
        else:  # agent_rows
            figsize = (9.0, 4.5)
    
    # Preserve original cfg value
    original_dc = getattr(cfg.params, 'data_class', None)

    # Create subplot grid based on layout
    if layout == 'nhp_rows':
        # 2 rows (NHPs) x 3 columns (agents)
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=False)
    else:  # agent_rows
        # 3 rows (agents) x 2 columns (NHPs)
        fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=False)
    
    per_agent_returns = {}

    if layout == 'nhp_rows':
        # NHPs as rows, agents as columns
        for col_idx, agent in enumerate(agents_in_order):
            agent_dc = dc_map.get(agent, None)
            if agent_dc is None:
                continue
            key = (agent_dc, agent)
            if key not in agent_data:
                continue

            # Set cfg for this agent
            cfg.params.data_class = agent_dc

            session_lists = agent_data[key]['session_lists']
            delase_results = agent_data[key]['delase_results']
            locs = agent_data[key]['locs']
            rocs = agent_data[key]['rocs']
            ropaps = agent_data[key]['ropaps']

            # Determine area and plot_range per agent (if provided)
            agent_area = area_by_agent.get(agent, area) if isinstance(area_by_agent, dict) else area
            agent_plot_range = plot_range_by_agent.get(agent, (-15, 85)) if isinstance(plot_range_by_agent, dict) else (-15, 85)

            # Determine which monkeys to plot
            if agent == 'propofol':
                monkeys = ['Mary', 'MrJones']
            else:
                monkeys = ['SPOCK', 'PEDRI']

            # Get axes for this agent (top row = first NHP, bottom row = second NHP)
            axs_for_agent = np.array([axes[0, col_idx], axes[1, col_idx]])
            returned = plot_session_stability_grouped(
                cfg,
                agent,
                session_lists,
                delase_results,
                locs,
                rocs,
                ropaps,
                plot_range=agent_plot_range,
                top_percent=top_percent,
                timescales=timescales,
                curve_colors=curve_colors,
                loc_roc_colors=loc_roc_colors,
                figsize=None,
                dose=dose,
                save_path=None,
                verbose=verbose,
                area=agent_area,
                return_data=return_data,
                data_type=data_type,
                axs_override=axs_for_agent,
                fig_override=fig,
                suppress_suptitle=True,
                skip_tight_layout=True,
                add_legend=False,  # Will add column-specific legends later
                star_height_base=0.92
            )
            if return_data:
                per_agent_returns[agent] = returned

            # Set titles with agent name and NHP name
            for row_idx, monkey in enumerate(monkeys):
                current_title = axes[row_idx, col_idx].get_title()
                axes[row_idx, col_idx].set_title(
                    f'{agent.capitalize()} ({monkey_titles[monkey]})',
                    color=curve_colors[agent],
                    pad=12,
                    fontsize=8
                )
            
            # Add legend for this column (agent) on the top subplot of this column
            if add_legend:
                # Check if this agent has multiple areas
                if isinstance(agent_area, (list, tuple, np.ndarray)):
                    areas_list = list(agent_area)
                else:
                    areas_list = [agent_area]
                multiple_areas = len(areas_list) > 1
                
                if multiple_areas:
                    # Get handles and labels from the top subplot of this column
                    top_ax = axes[0, col_idx]  # Top row, this column
                    handles, labels = top_ax.get_legend_handles_labels()
                    # Filter out duplicate labels
                    seen = set()
                    unique_handles = []
                    unique_labels = []
                    for handle, label in zip(handles, labels):
                        if label not in seen:
                            seen.add(label)
                            unique_handles.append(handle)
                            unique_labels.append(label)
                    # Add legend to the right of the top subplot
                    if len(unique_labels) > 0:
                        top_ax.legend(handles=unique_handles, labels=unique_labels,
                                    loc='center left', bbox_to_anchor=(1.02, 0.5),
                                    frameon=False, fontsize=legend_fontsize)
    else:  # agent_rows
        # Agents as rows, NHPs as columns (original behavior)
        for row_idx, agent in enumerate(agents_in_order):
            agent_dc = dc_map.get(agent, None)
            if agent_dc is None:
                continue
            key = (agent_dc, agent)
            if key not in agent_data:
                continue

            # Set cfg for this agent
            cfg.params.data_class = agent_dc

            session_lists = agent_data[key]['session_lists']
            delase_results = agent_data[key]['delase_results']
            locs = agent_data[key]['locs']
            rocs = agent_data[key]['rocs']
            ropaps = agent_data[key]['ropaps']

            # Determine area and plot_range per agent (if provided)
            agent_area = area_by_agent.get(agent, area) if isinstance(area_by_agent, dict) else area
            agent_plot_range = plot_range_by_agent.get(agent, (-15, 85)) if isinstance(plot_range_by_agent, dict) else (-15, 85)

            # Determine which monkeys to plot
            if agent == 'propofol':
                monkeys = ['Mary', 'MrJones']
            else:
                monkeys = ['SPOCK', 'PEDRI']

            # Get axes for this row (one for each NHP)
            axs_for_agent = np.array([axes[row_idx, 0], axes[row_idx, 1]])
            returned = plot_session_stability_grouped(
                cfg,
                agent,
                session_lists,
                delase_results,
                locs,
                rocs,
                ropaps,
                plot_range=agent_plot_range,
                top_percent=top_percent,
                timescales=timescales,
                curve_colors=curve_colors,
                loc_roc_colors=loc_roc_colors,
                figsize=None,
                dose=dose,
                save_path=None,
                verbose=verbose,
                area=agent_area,
                return_data=return_data,
                data_type=data_type,
                axs_override=axs_for_agent,
                fig_override=fig,
                suppress_suptitle=True,
                skip_tight_layout=True,
                add_legend=False,  # Will add row-specific legends later (agents are rows in this layout)
                star_height_base=0.92
            )
            if return_data:
                per_agent_returns[agent] = returned

            # Set titles with agent name and NHP name
            for col_idx, monkey in enumerate(monkeys):
                current_title = axes[row_idx, col_idx].get_title()
                axes[row_idx, col_idx].set_title(
                    f'{agent.capitalize()} ({monkey_titles[monkey]})',
                    color=curve_colors[agent],
                    pad=12,
                    fontsize=8
                )
            
            # Add legend for this row (agent) on the rightmost subplot of this row
            if add_legend:
                # Check if this agent has multiple areas
                if isinstance(agent_area, (list, tuple, np.ndarray)):
                    areas_list = list(agent_area)
                else:
                    areas_list = [agent_area]
                multiple_areas = len(areas_list) > 1
                
                if multiple_areas:
                    # Get handles and labels from the rightmost subplot of this row
                    rightmost_ax = axes[row_idx, 1]  # This row, rightmost column
                    handles, labels = rightmost_ax.get_legend_handles_labels()
                    # Filter out duplicate labels
                    seen = set()
                    unique_handles = []
                    unique_labels = []
                    for handle, label in zip(handles, labels):
                        if label not in seen:
                            seen.add(label)
                            unique_handles.append(handle)
                            unique_labels.append(label)
                    # Add legend to the right of the rightmost subplot
                    if len(unique_labels) > 0:
                        rightmost_ax.legend(handles=unique_handles, labels=unique_labels,
                                          loc='center left', bbox_to_anchor=(1.02, 0.5),
                                          frameon=False, fontsize=legend_fontsize)

    # Remove individual axis labels (x-axis shared, y-axis shared)
    for ax in axes.flat:
        ax.set_ylabel('')
        ax.set_xlabel('')

    # Apply tight_layout first, then adjust for labels
    # Leave extra space on the right for legends (0.88 instead of 0.95)
    plt.tight_layout(rect=[0.05, 0.05, 0.88, 0.95])

    # Shared x-axis label (only show on bottom row) - matching ACF grid positioning
    fig.text(0.5, 0.04, 'Time Relative to Infusion Start (min)', ha='center', fontsize=8)

    # Shared y-axis label (only on left column) - matching ACF grid positioning
    if timescales:
        fig.text(0.03, 0.5, 'Mean Characteristic Timescale\nRatio to Awake Baseline', va='center', ha='center', rotation='vertical', fontsize=8)
    else:
        fig.text(0.03, 0.5, 'Mean Instability ($s^{-1}$)', va='center', rotation='vertical', fontsize=8)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()

    if original_dc is not None:
        cfg.params.data_class = original_dc
    return per_agent_returns

def plot_section_stability_boxes(cfg, agent, session_lists, delase_results, top_percent=0.1,
                                  curve_colors={'propofol': 'blue', 'dexmedetomidine': 'purple', 'ketamine': 'red'},
                                  figsize=None,
                                  dose=None,
                                  save_path=None,
                                  section_info_type='plot',
                                  verbose=False,
                                  area='all',
                                  data_type='delase',
                                  axs_override=None,
                                  fig_override=None,
                                  suppress_suptitle=False,
                                  skip_tight_layout=False,
                                  set_xticks=True):
    section_means = {}

    is_lever = 'ket' in agent.lower() or 'dex' in agent.lower()

    if dose is None:
        doses = session_lists[monkey].keys()
    else:
        doses = [dose]

    for monkey in session_lists.keys():
        section_means[monkey] = {}
        for _dose in doses:
            section_means[monkey][_dose] = {}
            if _dose not in session_lists[monkey]:
                continue
            for session in session_lists[monkey][_dose]:
                section_info, section_info_extended, section_colors, infusion_start = get_section_info(session, cfg.params.all_data_dir, cfg.params.data_class, section_info_type=section_info_type)
                session_delase_results = delase_results[session][area]
                
                # Convert times to minutes relative to infusion
                time_vals = (session_delase_results.window_start - infusion_start) / 60
                
                section_means[monkey][_dose][session] = {}
                for section_name, (start_time, end_time) in section_info:
                    # Get indices for times in this section
                    section_mask = (time_vals >= start_time) & (time_vals < end_time)
                    
                    # Get stability params for times in this section
                    if data_type == 'delase':
                        section_stability = session_delase_results.stability_params[section_mask].apply(lambda x: x[:int(len(x)*top_percent)]).values
                    elif data_type == 'var':
                        section_stability = session_delase_results.stability_params_A[section_mask].apply(lambda x: x[:int(len(x)*top_percent)]).values
                    elif data_type == 'var_small':
                        section_stability = session_delase_results.stability_params_A_small[section_mask].apply(lambda x: x[:int(len(x)*top_percent)]).values
                    else:
                        raise ValueError(f"Invalid data type: {data_type}")
                    if len(section_stability) > 0:
                        section_stability = np.hstack(section_stability)
                        section_means[monkey][_dose][session][section_name] = np.mean(section_stability.mean())

    # Set up axes (either new figure/axes or use overrides for grid composition)
    if axs_override is None:
        if figsize is None:
            figsize = (4.2, 2)
        # Create figure based on number of doses
        if len(doses) > 1:
            fig, axs = plt.subplots(2, 2, figsize=figsize)
        else:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        # Add agent name as title (unless suppressed)
        if not suppress_suptitle:
            fig.suptitle(f'{agent.capitalize()}', c=curve_colors[agent], y=0.9)
    else:
        fig = fig_override if fig_override is not None else plt.gcf()
        axs = np.array(axs_override)

    # For each monkey
    for i, monkey in enumerate(session_lists.keys()):
        for j, dose in enumerate(doses):
            if dose not in session_lists[monkey]:
                continue
            # Get subplot
            ax = axs[i, j] if (len(doses) > 1 and axs.ndim == 2) else axs[i]
            
            # Collect data for boxplot
            box_data = []
            colors = []
            labels = []
            
            for section_name, _ in section_info:
                section_values = [section_means[monkey][dose][session][section_name] 
                                for session in session_lists[monkey][dose] if section_name in section_means[monkey][dose][session]]
                box_data.append(section_values)
                colors.append(section_colors[section_name])
                labels.append(section_name)

            # Create boxplot with normal whiskers, no outliers
            bp = ax.boxplot(box_data, patch_artist=True, 
                        showmeans=False, meanline=False, showfliers=False)

            # Assuming bp = ax.boxplot(...)

            # Iterate over each upper cap (in a list of boxes, upper caps are at odd indices)
            upper_caps = bp["caps"][1::2]
            lower_caps = bp["caps"][0::2]
            box_data_max = []
            box_data_min = []
            for i, cap in enumerate(upper_caps):
                # The get_data() method returns (x_data, y_data)
                x_data, y_data = cap.get_data()
                # Here, y_data contains the y-coordinate for the upper cap of box i
                box_data_max.append(y_data[0])
            for i, cap in enumerate(lower_caps):
                # The get_data() method returns (x_data, y_data)
                x_data, y_data = cap.get_data()
                # Here, y_data contains the y-coordinate for the upper cap of box i
                box_data_min.append(y_data[0])

            # Color the boxes and set median line to black
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            for median in bp['medians']:
                median.set_color('black')

            # Customize subplot
            # ax.set_xlabel('Section')
            ax.set_ylabel('Mean Instability ($s^{-1}$)')
            ax.set_title(f'{monkey_titles[monkey]}')

            # Set x-ticks
            # Set ticks explicitly to avoid issues with shared axes
            if set_xticks:
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels([label.replace(' ', '\n') for label in labels], rotation=45, fontsize=5)

            if section_info_type == 'plot':
                both_present = [session for session in session_lists[monkey][dose] if 'Awake' in section_means[monkey][dose][session] and 'Anesthesia' in section_means[monkey][dose][session]]
                awake_vals = [section_means[monkey][dose][session]['Awake'] for session in both_present]
                unconscious_vals = [section_means[monkey][dose][session]['Anesthesia'] for session in both_present]
            elif section_info_type == 'regular':
                if is_lever:
                    # Get awake and unconscious oddball values
                    awake_vals = [section_means[monkey][dose][session]['awake oddball'] 
                                  for session in session_lists[monkey][dose]]
                    if agent == 'dexmedetomidine':
                        unconscious_vals = [section_means[monkey][dose][session]['early unconscious']
                                        for session in session_lists[monkey][dose]]
                    else:
                        unconscious_vals = [section_means[monkey][dose][session]['unconscious oddball']
                                            for session in session_lists[monkey][dose]]
                else:
                    # Get awake and loading dose values
                    awake_vals = [section_means[monkey][dose][session]['awake'] 
                                  for session in session_lists[monkey][dose]]
                    unconscious_vals = [section_means[monkey][dose][session]['loading dose']
                                        for session in session_lists[monkey][dose]]
            
            # Perform wilcoxon test
            stat, p_val = wilcoxon(unconscious_vals, awake_vals, alternative='greater')
            mean_diff = np.mean(unconscious_vals) - np.mean(awake_vals)
            
            print(f"{agent} {dose} unconscious-awake: {p_val}")

            # Determine significance stars based on the p-value
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**' 
            elif p_val < 0.05:
                stars = '*'
            else:
                stars = 'ns'
            
            if section_info_type == 'plot':
                awake_idx = labels.index('Awake')
                unconscious_idx = labels.index('Anesthesia')
            elif section_info_type == 'regular':
                # Determine the x-positions from the boxplot labels (1-indexed)
                if is_lever:
                    awake_idx = labels.index('awake oddball')
                    if agent == 'dexmedetomidine':
                        unconscious_idx = labels.index('early unconscious')
                    else:
                        unconscious_idx = labels.index('unconscious oddball')
                else:
                    awake_idx = labels.index('awake')
                    unconscious_idx = labels.index('loading dose')
            
            x1 = awake_idx + 1
            x2 = unconscious_idx + 1

            y_bar_max = np.max(box_data_max[awake_idx:unconscious_idx + 1])
            y_bar_min = np.min(box_data_min[awake_idx:unconscious_idx + 1])
            axis_range = y_bar_max - y_bar_min
            offset = axis_range * 0.05   # For example, 5% of the axis range
            tick_length = offset * 0.3
            if mean_diff > 0:
                y_bar = y_bar_max + offset
                # Draw horizontal significance line
                ax.plot([x1, x2], [y_bar, y_bar], 'k-')
                # Draw vertical ticks at the ends
                ax.plot([x1, x1], [y_bar, y_bar - tick_length], 'k-')
                ax.plot([x2, x2], [y_bar, y_bar - tick_length], 'k-')
                # Place the significance stars just above the bar
                text_obj = ax.text((x1 + x2) / 2, y_bar + tick_length, stars, ha='center', va='bottom')

                # Ensure the figure is drawn so that the text's bounding box is available.
                plt.draw()

                # Get the renderer from the figure's canvas
                renderer = text_obj.figure.canvas.get_renderer()

                # Obtain the bounding box in display (pixel) coordinates
                bbox = text_obj.get_window_extent(renderer=renderer)

                # Convert the top of the bounding box (y1) into data coordinates
                top_x, top_y = ax.transData.inverted().transform((bbox.x1, bbox.y1))
                # if we've exceeded the ylim, set the ylim to the y_bar_min including the offset and text
                if top_y > ax.get_ylim()[1]:
                    ax.set_ylim(ax.get_ylim()[0], top_y + offset)
            
            elif mean_diff < 0:
                y_bar = y_bar_min - offset
                # Draw horizontal significance line
                ax.plot([x1, x2], [y_bar, y_bar], 'k-')
                ax.plot([x1, x1], [y_bar, y_bar + tick_length], 'k-')
                ax.plot([x2, x2], [y_bar, y_bar + tick_length], 'k-')
                # Place the significance stars just below the bar
                text_obj =ax.text((x1 + x2) / 2, y_bar - tick_length, stars, ha='center', va='top')

               # Ensure the figure is drawn so that the text's bounding box is available.
                plt.draw()

                # Get the renderer from the figure's canvas
                renderer = text_obj.figure.canvas.get_renderer()

                # Obtain the bounding box in display (pixel) coordinates
                bbox = text_obj.get_window_extent(renderer=renderer)

                # Convert the bottom of the bounding box (y0) into data coordinates
                bottom_x, bottom_y = ax.transData.inverted().transform((bbox.x0, bbox.y0))
                # if we've exceeded the ylim, set the ylim to the y_bar_min including the offset and text
                if bottom_y < ax.get_ylim()[0]:
                    ax.set_ylim(bottom_y - offset, ax.get_ylim()[1])
            
            if agent == 'ketamine':
                # section_to_compare = 'Late Anesthesia'
                section_to_compare = 'Emergence'
                if section_info_type == 'plot':
                    awake_idx = labels.index('Awake')
                    recovery_idx = labels.index(section_to_compare)
                elif section_info_type == 'regular':
                    awake_idx = labels.index('awake oddball')
                    recovery_idx = labels.index('recovery oddball')
                
                x1 = awake_idx + 1
                x2 = recovery_idx + 1
                y_bar_max = np.max(box_data_max[awake_idx:recovery_idx + 1])
                y_bar_min = np.min(box_data_min[awake_idx:recovery_idx + 1])
                axis_range = y_bar_max - y_bar_min
                offset = axis_range * 0.05   # For example, 5% of the axis range
                tick_length = offset * 0.3


                if section_info_type == 'plot':
                    both_present = [session for session in session_lists[monkey][dose] if 'Awake' in section_means[monkey][dose][session] and section_to_compare in section_means[monkey][dose][session]]
                    awake_vals = [section_means[monkey][dose][session]['Awake'] for session in both_present]
                    comparison_vals = [section_means[monkey][dose][session][section_to_compare] for session in both_present]
                elif section_info_type == 'regular':
                    both_present = [session for session in session_lists[monkey][dose] if 'awake oddball' in section_means[monkey][dose][session] and 'recovery oddball' in section_means[monkey][dose][session]]
                    awake_vals = [section_means[monkey][dose][session]['awake oddball'] for session in both_present]
                    comparison_vals = [section_means[monkey][dose][session]['recovery oddball'] for session in both_present]
                
                stat, p_val = wilcoxon(comparison_vals, awake_vals, alternative='less')
                mean_diff = np.mean(comparison_vals) - np.mean(awake_vals)
                print(f"{agent} {dose} recovery-awake: {p_val}")

                # Determine significance stars based on the p-value
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**' 
                elif p_val < 0.05:
                    stars = '*'
                else:
                    stars = 'ns'

                if mean_diff > 0:
                    y_bar = y_bar_max + offset
                    ax.plot([x1, x2], [y_bar, y_bar], 'k-')
                    ax.plot([x1, x1], [y_bar, y_bar - tick_length], 'k-')
                    ax.plot([x2, x2], [y_bar, y_bar - tick_length], 'k-')
                    text_obj = ax.text((x1 + x2) / 2, y_bar + tick_length, stars, ha='center', va='bottom')

                    # Ensure the figure is drawn so that the text's bounding box is available.
                    plt.draw()

                    # Get the renderer from the figure's canvas
                    renderer = text_obj.figure.canvas.get_renderer()

                    # Obtain the bounding box in display (pixel) coordinates
                    bbox = text_obj.get_window_extent(renderer=renderer)

                    # Convert the top of the bounding box (y1) into data coordinates
                    top_x, top_y = ax.transData.inverted().transform((bbox.x1, bbox.y1))
                    # if we've exceeded the ylim, set the ylim to the y_bar_min including the offset and text
                    if top_y > ax.get_ylim()[1]:
                        ax.set_ylim(ax.get_ylim()[0], top_y + offset)
                else:
                    y_bar = y_bar_min - offset
                    ax.plot([x1, x2], [y_bar, y_bar], 'k-')
                    ax.plot([x1, x1], [y_bar, y_bar + tick_length], 'k-')
                    ax.plot([x2, x2], [y_bar, y_bar + tick_length], 'k-')
                    text_obj = ax.text((x1 + x2) / 2, y_bar - tick_length, stars, ha='center', va='top')

                    # Ensure the figure is drawn so that the text's bounding box is available.
                    plt.draw()

                    # Get the renderer from the figure's canvas
                    renderer = text_obj.figure.canvas.get_renderer()

                    # Obtain the bounding box in display (pixel) coordinates
                    bbox = text_obj.get_window_extent(renderer=renderer)

                    # Convert the bottom of the bounding box (y0) into data coordinates
                    bottom_x, bottom_y = ax.transData.inverted().transform((bbox.x0, bbox.y0))
                    # if we've exceeded the ylim, set the ylim to the y_bar_min including the offset and text
                    if bottom_y < ax.get_ylim()[0]:
                        ax.set_ylim(bottom_y - offset, ax.get_ylim()[1])
                    

    if not skip_tight_layout:
        plt.tight_layout()
    if save_path is not None and axs_override is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    elif axs_override is None:
        plt.show()
    
    return section_means


def plot_section_stability_boxes_grid(cfg, agent_data, results_by_agent,
                                      data_class=None,
                                      data_class_by_agent=None,
                                      top_percent=0.1,
                                      curve_colors={'propofol': 'blue', 'dexmedetomidine': 'purple', 'ketamine': 'red'},
                                      figsize=None,
                                      dose=None,
                                      save_path=None,
                                      section_info_type='plot',
                                      verbose=False,
                                      area='all',
                                      data_type='delase',
                                      layout='nhp_rows',
                                      sharex=False):
    """
    Create a grid of section stability boxplots with configurable layout:
      - layout='nhp_rows' (default): 2-row x 3-column grid
        - Columns: agents (Propofol, Ketamine, Dexmedetomidine)
        - Rows: NHPs (NHP 1/2 for Propofol; NHP 3/4 for Ketamine and Dexmedetomidine)
      - layout='agent_rows': 3-row x 2-column grid
        - Rows: agents (Propofol, Ketamine, Dexmedetomidine)
        - Columns: NHPs (NHP 1/2 for Propofol; NHP 3/4 for Ketamine and Dexmedetomidine)
    
    Args:
        cfg: Config object (will use cfg.params.data_class)
        agent_data: dict keyed by (data_class, agent) with entry containing 'session_lists'
        results_by_agent: dict keyed by (data_class, agent) with DeLASE/VAR results
        data_class: (optional) single data_class to use for all agents
        data_class_by_agent: (optional) dict mapping agent -> data_class. If not
            provided, we auto-detect per agent from keys present in agent_data.
        layout: 'nhp_rows' for 2x3 grid (NHPs as rows) or 'agent_rows' for 3x2 grid (agents as rows)
        figsize: tuple for figure size. If None, defaults to (9.0, 3.2) for nhp_rows or (6.0, 9.0) for agent_rows
        sharex: bool, whether to share x-axis across subplots (default: False)
        Other plotting params mirror plot_section_stability_boxes
    
    Returns:
        A dict mapping agent -> section_means (same structure as plot_section_stability_boxes return).
    """
    agents_in_order = ['propofol', 'ketamine','dexmedetomidine']
    # Build a data_class mapping per agent
    dc_map = {}
    # If explicit mapping provided, use it
    if isinstance(data_class_by_agent, dict):
        for a in agents_in_order:
            if a in data_class_by_agent:
                dc_map[a] = data_class_by_agent[a]
    # If a single data_class is provided, use it as fallback
    if data_class is not None:
        for a in agents_in_order:
            dc_map.setdefault(a, data_class)
    # Auto-detect for any remaining agents from keys in agent_data
    if isinstance(agent_data, dict):
        for (dc, ag) in agent_data.keys():
            if ag in agents_in_order and ag not in dc_map:
                dc_map[ag] = dc
    # Validate layout parameter
    if layout not in ['nhp_rows', 'agent_rows']:
        raise ValueError(f"layout must be 'nhp_rows' or 'agent_rows', got '{layout}'")
    
    # Set default figsize based on layout
    if figsize is None:
        if layout == 'nhp_rows':
            figsize = (9.0, 3.2)
        else:  # agent_rows
            figsize = (6.0, 9.0)
    
    # Preserve original cfg value to restore later
    original_dc = getattr(cfg.params, 'data_class', None)
    
    # Create subplot grid based on layout
    if layout == 'nhp_rows':
        # 2 rows (NHPs) x 3 columns (agents)
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=sharex)
    else:  # agent_rows
        # 3 rows (agents) x 2 columns (NHPs)
        fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=sharex)
    
    all_section_means = {}
    
    if layout == 'nhp_rows':
        # Original layout: NHPs as rows, agents as columns
        for col_idx, agent in enumerate(agents_in_order):
            # Determine data_class to use for this agent
            agent_dc = dc_map.get(agent, None)
            if agent_dc is None:
                continue
            key = (agent_dc, agent)
            if key not in agent_data or key not in results_by_agent:
                continue
            # Set cfg for this agent's data_class
            cfg.params.data_class = agent_dc
            session_lists = agent_data[key]['session_lists']
            delase_results = results_by_agent[key]
            # Compose the two axes for this agent (top row = first NHP, bottom row = second NHP)
            axs_for_agent = np.array([axes[0, col_idx], axes[1, col_idx]])
            # When sharex=True, only set ticks on bottom row (row 1)
            set_ticks_top = not sharex
            set_ticks_bottom = True  # Always set ticks on bottom row
            # Delegate actual box rendering to the base function, but into provided axes
            # We need to call it twice - once for each axis with different set_xticks values
            # Actually, we can't easily do that since the function loops over monkeys
            # Instead, we'll set ticks after calling the function, but only on bottom row when sharex=True
            section_means = plot_section_stability_boxes(
                cfg,
                agent,
                session_lists,
                delase_results,
                top_percent=top_percent,
                curve_colors=curve_colors,
                figsize=None,
                dose=dose,
                save_path=None,
                section_info_type=section_info_type,
                verbose=verbose,
                area=area,
                data_type=data_type,
                axs_override=axs_for_agent,
                fig_override=fig,
                suppress_suptitle=True,
                skip_tight_layout=True,
                set_xticks=not sharex  # Only set ticks if not sharing x-axis
            )
            all_section_means[agent] = section_means
            # Set each subplot title to "Agent (NHP #)" in agent color
            top_title = axes[0, col_idx].get_title()
            bottom_title = axes[1, col_idx].get_title()
            axes[0, col_idx].set_title(f'{agent.capitalize()} ({top_title})', color=curve_colors[agent], fontsize=10)
            axes[1, col_idx].set_title(f'{agent.capitalize()} ({bottom_title})', color=curve_colors[agent], fontsize=10)
            
            # If sharex=True, set ticks only on bottom row after all plots are done
            # We'll do this after the loop
    else:  # agent_rows
        # New layout: agents as rows, NHPs as columns
        for row_idx, agent in enumerate(agents_in_order):
            # Determine data_class to use for this agent
            agent_dc = dc_map.get(agent, None)
            if agent_dc is None:
                continue
            key = (agent_dc, agent)
            if key not in agent_data or key not in results_by_agent:
                continue
            # Set cfg for this agent's data_class
            cfg.params.data_class = agent_dc
            session_lists = agent_data[key]['session_lists']
            delase_results = results_by_agent[key]
            # Compose the two axes for this agent (left column = first NHP, right column = second NHP)
            axs_for_agent = np.array([axes[row_idx, 0], axes[row_idx, 1]])
            # When sharex=True, only set ticks on bottom row (last row)
            section_means = plot_section_stability_boxes(
                cfg,
                agent,
                session_lists,
                delase_results,
                top_percent=top_percent,
                curve_colors=curve_colors,
                figsize=None,
                dose=dose,
                save_path=None,
                section_info_type=section_info_type,
                verbose=verbose,
                area=area,
                data_type=data_type,
                axs_override=axs_for_agent,
                fig_override=fig,
                suppress_suptitle=True,
                skip_tight_layout=True,
                set_xticks=not sharex or row_idx == len(agents_in_order) - 1  # Only set ticks if not sharing or on bottom row
            )
            all_section_means[agent] = section_means
            # Set each subplot title to "Agent (NHP #)" in agent color
            left_title = axes[row_idx, 0].get_title()
            right_title = axes[row_idx, 1].get_title()
            axes[row_idx, 0].set_title(f'{agent.capitalize()} ({left_title})', color=curve_colors[agent], fontsize=10)
            axes[row_idx, 1].set_title(f'{agent.capitalize()} ({right_title})', color=curve_colors[agent], fontsize=10)
    
    # If sharex=True, set ticks on bottom row only
    if sharex:
        if layout == 'nhp_rows':
            # Set ticks on bottom row for each column
            for col_idx in range(len(agents_in_order)):
                agent = agents_in_order[col_idx]
                agent_dc = dc_map.get(agent, None)
                if agent_dc is None:
                    continue
                key = (agent_dc, agent)
                if key not in agent_data:
                    continue
                # Get section info to determine labels
                session_lists = agent_data[key]['session_lists']
                # Get first session to determine section labels
                labels = None
                for monkey in session_lists.keys():
                    if dose in session_lists[monkey] and len(session_lists[monkey][dose]) > 0:
                        first_session = session_lists[monkey][dose][0]
                        section_info, _, section_colors, _ = get_section_info(
                            first_session, cfg.params.all_data_dir, agent_dc, section_info_type=section_info_type
                        )
                        labels = [section_name for section_name, _ in section_info]
                        break
                if labels:
                    # Set ticks on bottom row axis for this column
                    axes[1, col_idx].set_xticks(range(1, len(labels) + 1))
                    axes[1, col_idx].set_xticklabels(
                        [label.replace(' ', '\n') for label in labels], rotation=45, fontsize=5
                    )
        else:  # agent_rows layout
            # Set ticks on bottom row (last row) for each column
            bottom_row_idx = len(agents_in_order) - 1
            for col_idx in range(2):
                # Get section info from the bottom row agent
                agent = agents_in_order[bottom_row_idx]
                agent_dc = dc_map.get(agent, None)
                if agent_dc is None:
                    continue
                key = (agent_dc, agent)
                if key not in agent_data:
                    continue
                # Get section info to determine labels
                session_lists = agent_data[key]['session_lists']
                # Get first session to determine section labels
                labels = None
                for monkey in session_lists.keys():
                    if dose in session_lists[monkey] and len(session_lists[monkey][dose]) > 0:
                        first_session = session_lists[monkey][dose][0]
                        section_info, _, section_colors, _ = get_section_info(
                            first_session, cfg.params.all_data_dir, agent_dc, section_info_type=section_info_type
                        )
                        labels = [section_name for section_name, _ in section_info]
                        break
                if labels:
                    # Set ticks on bottom row axis for this column
                    axes[bottom_row_idx, col_idx].set_xticks(range(1, len(labels) + 1))
                    axes[bottom_row_idx, col_idx].set_xticklabels(
                        [label.replace(' ', '\n') for label in labels], rotation=45, fontsize=8
                    )
    # Finalize layout once, save/show once
    # Remove individual y-axis labels and add a single shared y-label on the left
    for ax in axes.flat:
        ax.set_ylabel('')
    fig.text(-0.02, 0.5, 'Mean Instability ($s^{-1}$)', va='center', rotation='vertical', fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    # Restore original cfg data_class
    if original_dc is not None:
        cfg.params.data_class = original_dc
    return all_section_means


def plot_combined_stability_grid(cfg, agent_data, results_by_agent,
                                 data_class=None,
                                 data_class_by_agent=None,
                                 top_percent=0.1,
                                 timescales=True,
                                 curve_colors={'propofol': 'blue', 'dexmedetomidine': 'purple', 'ketamine': 'red'},
                                 loc_roc_colors={'propofol': 'midnightblue', 'dexmedetomidine': 'darkviolet', 'ketamine': 'darkred'},
                                 figsize=None,
                                 dose=None,
                                 save_path=None,
                                 section_info_type='plot',
                                 verbose=False,
                                 area='all',
                                 area_by_agent=None,
                                 plot_range_by_agent=None,
                                 return_data=False,
                                 data_type='delase',
                                 star_height_base=1.01):
    """
    Create a combined 3-row x 4-column grid combining:
      - Columns 0-1: Session stability grouped plots (NHP 1/2 for Propofol; NHP 3/4 for Ketamine/Dexmedetomidine)
      - Columns 2-3: Section stability boxplots (same NHPs)
      - Rows: agents (Propofol, Ketamine, Dexmedetomidine)
    
    Args:
        cfg: Config object
        agent_data: dict keyed by (data_class, agent) with entries containing:
            'session_lists', 'delase_results', 'locs', 'rocs', 'ropaps'
        results_by_agent: dict keyed by (data_class, agent) with DeLASE/VAR results for section plots
        data_class: (optional) single data_class to use for all agents
        data_class_by_agent: (optional) dict mapping agent -> data_class; overrides single.
        area: default area or list of areas to use for all agents if area_by_agent is None
        area_by_agent: (optional) dict mapping agent -> area (string or list)
        plot_range_by_agent: (optional) dict mapping agent -> (min, max) minutes window
        return_data: whether to return per-agent data from the underlying functions
        star_height_base: base height for significance stars (in axes-fraction units, default 1.01)
        Other params mirror plot_session_stability_grouped_grid and plot_section_stability_boxes_grid
    
    Returns:
        A dict with keys 'grouped' and 'section', each containing per-agent return data if return_data=True
    """
    agents_in_order = ['propofol', 'ketamine', 'dexmedetomidine']
    
    # Build data_class mapping, with auto-detection from keys
    dc_map = {}
    if isinstance(data_class_by_agent, dict):
        for a in agents_in_order:
            if a in data_class_by_agent:
                dc_map[a] = data_class_by_agent[a]
    if data_class is not None:
        for a in agents_in_order:
            dc_map.setdefault(a, data_class)
    if isinstance(agent_data, dict):
        for (dc, ag) in agent_data.keys():
            if ag in agents_in_order and ag not in dc_map:
                dc_map[ag] = dc
    
    # Preserve original cfg value
    original_dc = getattr(cfg.params, 'data_class', None)
    
    # Set default figsize
    if figsize is None:
        figsize = (12.0, 9.0)  # Wider to accommodate 4 columns
    
    # Create figure with two subfigures side by side
    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(1, 2, wspace=0.00, width_ratios=[1, 1])
    
    # Left subfigure: Grouped stability (3x2 grid)
    axes_grouped = subfigs[0].subplots(3, 2, sharex=False)
    
    # Right subfigure: Section stability (3x2 grid)
    axes_section = subfigs[1].subplots(3, 2, sharex=False)
    
    per_agent_returns_grouped = {}
    per_agent_returns_section = {}
    
    for row_idx, agent in enumerate(agents_in_order):
        agent_dc = dc_map.get(agent, None)
        if agent_dc is None:
            continue
        key = (agent_dc, agent)
        if key not in agent_data or key not in results_by_agent:
            continue
        
        # Set cfg for this agent
        cfg.params.data_class = agent_dc
        
        session_lists = agent_data[key]['session_lists']
        delase_results = agent_data[key]['delase_results']
        locs = agent_data[key]['locs']
        rocs = agent_data[key]['rocs']
        ropaps = agent_data[key]['ropaps']
        section_delase_results = results_by_agent[key]
        
        # Determine area and plot_range per agent (if provided)
        agent_area = area_by_agent.get(agent, area) if isinstance(area_by_agent, dict) else area
        agent_plot_range = plot_range_by_agent.get(agent, (-15, 85)) if isinstance(plot_range_by_agent, dict) else (-15, 85)
        
        # Determine which monkeys to plot
        if agent == 'propofol':
            monkeys = ['Mary', 'MrJones']
        else:
            monkeys = ['SPOCK', 'PEDRI']
        
        # Left subfigure: Grouped stability plots
        axs_grouped = np.array([axes_grouped[row_idx, 0], axes_grouped[row_idx, 1]])
        returned_grouped = plot_session_stability_grouped(
            cfg,
            agent,
            session_lists,
            delase_results,
            locs,
            rocs,
            ropaps,
            plot_range=agent_plot_range,
            top_percent=top_percent,
            timescales=timescales,
            curve_colors=curve_colors,
            loc_roc_colors=loc_roc_colors,
            figsize=None,
            dose=dose,
            save_path=None,
            verbose=verbose,
            area=agent_area,
            return_data=return_data,
            data_type=data_type,
            axs_override=axs_grouped,
            fig_override=subfigs[0],
            suppress_suptitle=True,
            skip_tight_layout=True,
            add_legend=False,
            star_height_base=star_height_base
        )
        if return_data:
            per_agent_returns_grouped[agent] = returned_grouped
        
        # Set titles for grouped stability plots
        for col_idx, monkey in enumerate(monkeys):
            axes_grouped[row_idx, col_idx].set_title(
                f'{agent.capitalize()} ({monkey_titles[monkey]})',
                color=curve_colors[agent],
                pad=8,
                fontsize=8
            )
        
        # Right subfigure: Section stability plots
        axs_section = np.array([axes_section[row_idx, 0], axes_section[row_idx, 1]])
        # Don't set x-ticks here, we'll set them with colors for all rows afterwards
        returned_section = plot_section_stability_boxes(
            cfg,
            agent,
            session_lists,
            section_delase_results,
            top_percent=top_percent,
            curve_colors=curve_colors,
            figsize=None,
            dose=dose,
            save_path=None,
            section_info_type=section_info_type,
            verbose=verbose,
            area=agent_area,
            data_type=data_type,
            axs_override=axs_section,
            fig_override=subfigs[1],
            suppress_suptitle=True,
            skip_tight_layout=True,
            set_xticks=False
        )
        if return_data:
            per_agent_returns_section[agent] = returned_section
        
        # Set titles for section stability plots
        for col_idx, monkey in enumerate(monkeys):
            axes_section[row_idx, col_idx].set_title(
                f'{agent.capitalize()} ({monkey_titles[monkey]})',
                color=curve_colors[agent],
                pad=8,
                fontsize=8
            )
    
    # Remove individual axis labels
    for ax in axes_grouped.flat:
        ax.set_ylabel('')
        ax.set_xlabel('')
    for ax in axes_section.flat:
        ax.set_ylabel('')
        ax.set_xlabel('')
    
    # Apply tight_layout first
    plt.tight_layout()
    
    # Adjust layout to leave space for y-axis labels (after tight_layout)
    subfigs[0].subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    subfigs[1].subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    
    # Add shared labels for left subfigure (grouped stability)
    subfigs[0].text(0.5, 0.02, 'Time Relative to Infusion Start (min)', ha='center', fontsize=8)
    if timescales:
        subfigs[0].text(-0.02, 0.5, 'Mean Characteristic Timescale\nRatio to Awake Baseline', va='center', ha='center', rotation='vertical', fontsize=8)
    else:
        subfigs[0].text(-0.02, 0.5, 'Mean Instability ($s^{-1}$)', va='center', ha='center', rotation='vertical', fontsize=8)
    
    # Add shared labels for right subfigure (section stability)
    # subfigs[1].text(0.5, 0.02, 'Epoch', ha='center', fontsize=8)
    subfigs[1].text(-0.02, 0.5, 'Mean Instability ($s^{-1}$)', va='center', ha='center', rotation='vertical', fontsize=8)
    
    # Set x-ticks for section stability plots for all rows with colored labels
    for row_idx, agent in enumerate(agents_in_order):
        agent_dc = dc_map.get(agent, None)
        if agent_dc is None:
            continue
        key = (agent_dc, agent)
        if key not in agent_data:
            continue
        
        # Temporarily set cfg for this agent to get section info
        cfg.params.data_class = agent_dc
        session_lists = agent_data[key]['session_lists']
        labels = None
        section_colors = None
        for monkey in session_lists.keys():
            if dose in session_lists[monkey] and len(session_lists[monkey][dose]) > 0:
                first_session = session_lists[monkey][dose][0]
                section_info, _, section_colors, _ = get_section_info(
                    first_session, cfg.params.all_data_dir, agent_dc, section_info_type=section_info_type
                )
                labels = [section_name for section_name, _ in section_info]
                break
        
        if labels and section_colors:
            for col_idx in [0, 1]:
                ax = axes_section[row_idx, col_idx]
                ax.set_xticks(range(1, len(labels) + 1))
                tick_labels = [label.replace(' ', '\n') for label in labels]
                ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
                
                # Color each tick label according to section_colors
                for tick_idx, (tick_label, section_name) in enumerate(zip(ax.get_xticklabels(), labels)):
                    color = section_colors.get(section_name, 'black')
                    tick_label.set_color(color)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    
    if original_dc is not None:
        cfg.params.data_class = original_dc
    
    if return_data:
        return {
            'grouped': per_agent_returns_grouped,
            'section': per_agent_returns_section
        }
    else:
        return None


def plot_sensory_responses_etdc(agent, curve_colors, epoch_colors, sensory_responses, leadup, response, dt=0.001, n_delays=1, delay_interval=1, plot_legend=False, save_path=None, dims=1, use_mean=False, min_time=None, max_time=None, figsize=None, axs_override=None, fig_override=None, dose=None, skip_tight_layout=False, suppress_suptitle=False):
    if use_mean and dims > 1:
        raise ValueError('use_mean is not supported for dims > 1')
    responses_etdc = get_responses_etdc(sensory_responses, n_delays, delay_interval, use_mean)
    # Utilities for creating zoomed insets when needed
    # add_inset = agent.lower() in ['ketamine', 'dexmedetomidine']
    add_inset = False
    time_vals = np.arange(-leadup + (n_delays - 1)*delay_interval, response)*dt
    # raise ValueError(time_vals)

    if min_time is None:
        min_time = time_vals[0]/dt
    if max_time is None:
        max_time = time_vals[-1]/dt
    plot_indices = np.where((time_vals/dt >= min_time) & (time_vals/dt <= max_time))[0]

    # Figure/axes setup (support external grid composition)
    if axs_override is None:
        if figsize is None:
            if agent == 'propofol':
                figsize = (4.2, 1.5)
            else:
                figsize = (4.2, 1.5*len(responses_etdc[list(responses_etdc.keys())[0]].keys()))

        if agent == 'propofol':
            n_plots = 1
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        else:
            n_plots = len(responses_etdc[list(responses_etdc.keys())[0]].keys())
            fig, axs = plt.subplots(n_plots, 2, figsize=figsize)
    else:
        # Use provided axes for grid composition
        fig = fig_override if fig_override is not None else plt.gcf()
        axs = np.array(axs_override)
        n_plots = 1  # For grid, we always plot one dose per subplot


    plot_return_data = {}

    for monkey in responses_etdc.keys():
        plot_return_data[monkey] = {}
        # Filter doses if specified
        doses_to_plot = [dose] if dose is not None else responses_etdc[monkey].keys()
        for dose_level in doses_to_plot:
            if dose_level not in responses_etdc[monkey]:
                continue
            plot_return_data[monkey][dose_level] = {}
            if agent == 'propofol':
                if monkey == 'Mary':
                    ax_idx = 0
                else:
                    ax_idx = 1
            else:
                if n_plots == 1:
                    ax_idx = 0 if monkey == 'SPOCK' else 1
                else:
                    if monkey == 'SPOCK':
                        ax_idx = (0, 0) if dose_level == 'low' else (1, 0)
                    else:  # PEDRI
                        ax_idx = (0, 1) if dose_level == 'low' else (1, 1)
                
            for section in responses_etdc[monkey][dose_level].keys():
                if responses_etdc[monkey][dose_level][section] is None:
                    continue
                if dims == 1:
                    mean_trajectory = responses_etdc[monkey][dose_level][section].mean(axis=0)[:, 0]
                    sem_trajectory = responses_etdc[monkey][dose_level][section].std(axis=0)[:, 0] / np.sqrt(responses_etdc[monkey][dose_level][section].shape[0])
                elif dims == 2:
                    mean_trajectory = responses_etdc[monkey][dose_level][section].mean(axis=0)[:, :dims]
                    sem_trajectory = responses_etdc[monkey][dose_level][section].std(axis=0)[:, :dims] / np.sqrt(responses_etdc[monkey][dose_level][section].shape[0])
                
                plot_return_data[monkey][dose_level][section] = {
                    'mean_trajectory': mean_trajectory,
                    'sem_trajectory': sem_trajectory,
                }
                
                # color = 'green' if 'awake' in section else 'orange' if 'recovery' in section else 'purple'
                if 'awake' in section:
                    color = epoch_colors['awake']
                elif 'recovery' in section:
                    color = epoch_colors['emergence']
                else:
                    color = epoch_colors['anesthesia']

                # Plot individual trajectories
                for i in range(responses_etdc[monkey][dose_level][section].shape[0]):
                    if dims == 1:
                        axs[ax_idx].plot(time_vals[plot_indices], responses_etdc[monkey][dose_level][section][i, plot_indices, 0], 
                                    color=color, alpha=0.1)
                    # if dims == 2:
                    #     axs[ax_idx].plot(responses_etdc[monkey][dose_level][section][i, :, 0], 
                    #                 responses_etdc[monkey][dose_level][section][i, :, 1], 
                    #                 color=color, alpha=0.1)
                    #             # color=color, alpha=0.6)
                
                # Plot mean and standard error
                if dims == 1:
                    axs[ax_idx].plot(time_vals[plot_indices], mean_trajectory[plot_indices], color=color)
                    axs[ax_idx].fill_between(time_vals[plot_indices], 
                                    mean_trajectory[plot_indices] - sem_trajectory[plot_indices],
                                    mean_trajectory[plot_indices] + sem_trajectory[plot_indices],
                                    color=color, alpha=0.2)
                elif dims == 2:
                    axs[ax_idx].plot(mean_trajectory[plot_indices, 0], mean_trajectory[plot_indices, 1], color=color)
                    axs[ax_idx].fill_between(mean_trajectory[plot_indices, 0], 
                                    mean_trajectory[plot_indices, 1] - sem_trajectory[plot_indices, 1],
                                    mean_trajectory[plot_indices, 1] + sem_trajectory[plot_indices, 1],
                                    color=color, alpha=0.2)
            
            
            axs[ax_idx].set_title(f'{monkey_titles[monkey]}' + ('\n' + dose_level + ' dose' if n_plots > 1 else ''))
            axs[ax_idx].set_xlabel('Time Relative to Tone Start (s)')
            if n_delays > 1:
                axs[ax_idx].set_ylabel('Eigen-Time-Delay Coordinate 1')
            elif not use_mean:
                axs[ax_idx].set_ylabel('PC 1')
            else:
                axs[ax_idx].set_ylabel('Mean LFP (mV)')
            # axs[ax_idx].legend(['Awake', 'Unconscious'], loc='upper right')
            axs[ax_idx].grid(True, alpha=0.3)
            # --------------------------------------------------
            # Optional zoomed inset: show first 250 ms in detail
            # --------------------------------------------------
            if add_inset:
                ax_main = axs[ax_idx]
                # Create a small inset in the upper-right corner of the main axis
                inset_ax = inset_axes(ax_main, width="40%", height="40%", loc='upper right')
                inset_ax.set_xlim(0, 0.25)
                # We'll set the y-limits after plotting so they reflect just the zoom-window values
                local_min = np.inf
                local_max = -np.inf
 
                if dims == 1:
                    zoom_inds = np.where((time_vals >= 0) & (time_vals <= 0.25))[0]
                    for sec in responses_etdc[monkey][dose].keys():
                        print(sec)
                        col = 'green' if 'awake' in sec else 'orange' if 'recovery' in sec else 'purple'
                        mt = plot_return_data[monkey][dose][sec]['mean_trajectory']
                        se = plot_return_data[monkey][dose][sec]['sem_trajectory']
                        inset_ax.plot(time_vals[zoom_inds], mt[zoom_inds], color=col)
                        inset_ax.fill_between(time_vals[zoom_inds],
                                             mt[zoom_inds] - se[zoom_inds],
                                             mt[zoom_inds] + se[zoom_inds],
                                             color=col, alpha=0.2)
                        # Update local y-range based on mean ± SEM
                        cur_min = np.min(mt[zoom_inds] - se[zoom_inds])
                        cur_max = np.max(mt[zoom_inds] + se[zoom_inds])
                        local_min = min(local_min, cur_min)
                        local_max = max(local_max, cur_max)

                # Apply the data-based y-limits with a small margin
                if np.isfinite(local_min) and np.isfinite(local_max):
                    rng = local_max - local_min
                    margin = rng * 0.05 if rng > 0 else 0.01
                    inset_ax.set_ylim(local_min - margin, local_max + margin)

                # Aesthetics for inset
                inset_ax.tick_params(axis='both', which='major', labelsize=6)
                inset_ax.grid(True, alpha=0.2)
    # if plot_legend is True, add one green line (awake) and one purple line (unconscious)
    if plot_legend:
        # Create empty lines for legend
        line1, = plt.plot([], [], color=epoch_colors['awake'], label='Awake', visible=True)
        line2, = plt.plot([], [], color=epoch_colors['anesthesia'], label='Anesthesia', visible=True)
        line3, = plt.plot([], [], color=epoch_colors['emergence'], label='Emergence', visible=True)
        
        # Add legend centered below all subplots
        fig.legend(handles=[line1, line2, line3], loc='center', bbox_to_anchor=(0.5, 0), ncol=3)
    
    # Add agent name as title (unless suppressed)
    if not suppress_suptitle:
        fig.suptitle(f'{agent.capitalize()}', c=curve_colors[agent], y=0.9)
    if not skip_tight_layout:
        plt.tight_layout()
    if save_path is not None and axs_override is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    elif axs_override is None:
        plt.show()
    return plot_return_data

def plot_sensory_responses_etdc_grid(cfg, agent_data, curve_colors, epoch_colors,
                                     data_class=None,
                                     data_class_by_agent=None,
                                     n_delays=1,
                                     delay_interval=1,
                                     min_time=None,
                                     max_time=None,
                                     figsize=(9.0, 4.5),
                                     save_path=None,
                                     use_mean=True,
                                     dims=1,
                                     dose='high',
                                     plot_legend_by_agent=None):
    """
    Compose a 3-row x 2-column grid of sensory response ETDC plots:
      - Rows: agents (Propofol, Ketamine, Dexmedetomidine)
      - Columns: NHPs (NHP 1/2 for Propofol; NHP 3/4 for Ketamine and Dexmedetomidine)
    
    Args:
        cfg: Config object
        agent_data: dict keyed by (data_class, agent) with entries containing:
            'sensory_responses', 'leadup', 'response', 'dt'
        curve_colors: dict mapping agent -> color
        epoch_colors: dict mapping epoch -> color
        data_class: (optional) single data_class to use for all agents
        data_class_by_agent: (optional) dict mapping agent -> data_class; overrides single.
        n_delays, delay_interval: ETDC parameters
        min_time, max_time: time window for plotting (in samples relative to dt)
        figsize: figure size tuple
        save_path: path to save figure
        use_mean: whether to use mean LFP
        dims: number of dimensions to plot
        dose: which dose to plot ('high' or 'low')
        plot_legend_by_agent: dict mapping agent -> bool for whether to show legend
    
    Returns:
        dict mapping (data_class, agent) -> plot_return_data
    """
    agents_in_order = ['propofol', 'ketamine', 'dexmedetomidine']
    
    # Build data_class mapping, with auto-detection from keys
    dc_map = {}
    if isinstance(data_class_by_agent, dict):
        for a in agents_in_order:
            if a in data_class_by_agent:
                dc_map[a] = data_class_by_agent[a]
    if data_class is not None:
        for a in agents_in_order:
            dc_map.setdefault(a, data_class)
    if isinstance(agent_data, dict):
        for (dc, ag) in agent_data.keys():
            if ag in agents_in_order and ag not in dc_map:
                dc_map[ag] = dc
    
    # Preserve original cfg value
    original_dc = getattr(cfg.params, 'data_class', None)
    
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    per_agent_returns = {}
    
    if plot_legend_by_agent is None:
        plot_legend_by_agent = {}
    
    for row_idx, agent in enumerate(agents_in_order):
        agent_dc = dc_map.get(agent, None)
        if agent_dc is None:
            continue
        key = (agent_dc, agent)
        if key not in agent_data:
            continue
        
        # Set cfg for this agent
        cfg.params.data_class = agent_dc
        
        sensory_responses = agent_data[key]['sensory_responses']
        leadup = agent_data[key]['leadup']
        response = agent_data[key]['response']
        dt = agent_data[key]['dt']
        
        # Determine which monkeys to plot
        if agent == 'propofol':
            monkeys = ['Mary', 'MrJones']
        else:
            monkeys = ['SPOCK', 'PEDRI']
        
        # Create filtered sensory_responses dict for this agent with only the specified dose
        # Need to preserve the full nested structure: monkey -> dose -> session -> section
        filtered_sensory_responses = {}
        for monkey in monkeys:
            if monkey not in sensory_responses:
                continue
            if dose not in sensory_responses[monkey]:
                continue
            filtered_sensory_responses[monkey] = {dose: sensory_responses[monkey][dose]}
        
        # Get axes for this row (one for each NHP)
        axs_for_agent = np.array([axes[row_idx, 0], axes[row_idx, 1]])
        
        # Call the plotting function with overrides (don't create legend here, we'll do it at the end)
        returned = plot_sensory_responses_etdc(
            agent,
            curve_colors,
            epoch_colors,
            filtered_sensory_responses,
            leadup,
            response,
            dt=dt,
            n_delays=n_delays,
            delay_interval=delay_interval,
            plot_legend=False,  # We'll create a single legend at the end
            save_path=None,
            dims=dims,
            use_mean=use_mean,
            min_time=min_time,
            max_time=max_time,
            axs_override=axs_for_agent,
            fig_override=fig,
            dose=dose,
            skip_tight_layout=True,
            suppress_suptitle=True
        )
        per_agent_returns[key] = returned
        
        # Set titles with agent name and NHP name
        for col_idx, monkey in enumerate(monkeys):
            current_title = axes[row_idx, col_idx].get_title()
            axes[row_idx, col_idx].set_title(
                f'{agent.capitalize()} ({monkey_titles[monkey]})',
                color=curve_colors[agent],
                pad=8
            )
    
    # Remove individual axis labels (x-axis shared, y-axis shared)
    for ax in axes.flat:
        ax.set_ylabel('')
        ax.set_xlabel('')
    
    # Apply tight_layout first, then adjust for labels
    # Increase bottom margin significantly to accommodate x-axis label and legend
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    # Shared x-axis label (only show on bottom row) - positioned above legend
    fig.text(0.5, 0.04, 'Time Relative to Tone Start (s)', ha='center', fontsize=8)
    
    # Create a single legend for the entire figure if any agent requested it
    if any(plot_legend_by_agent.get(agent, False) for agent in agents_in_order):
        # Create empty lines for legend
        line1, = plt.plot([], [], color=epoch_colors['awake'], label='Awake', visible=True)
        line2, = plt.plot([], [], color=epoch_colors['anesthesia'], label='Anesthesia', visible=True)
        line3, = plt.plot([], [], color=epoch_colors['emergence'], label='Emergence', visible=True)
        
        # Add legend centered below x-axis label with proper spacing
        fig.legend(handles=[line1, line2, line3], loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=7, frameon=False)
    
    # Shared y-axis label (only on left column) - moderate font size
    if n_delays > 1:
        ylabel = 'Eigen-Time-Delay Coordinate 1'
    elif not use_mean:
        ylabel = 'PC 1'
    else:
        ylabel = 'Mean LFP (mV)'
    fig.text(0.03, 0.5, ylabel, va='center', rotation='vertical', fontsize=8)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    
    if original_dc is not None:
        cfg.params.data_class = original_dc
    
    return per_agent_returns

def plot_sensory_responses_acf(agent, curve_colors, epoch_colors, sensory_responses, leadup, response, dt=0.001, n_delays=1, delay_interval=1, plot_legend=False, save_path=None, dims=1, method='individual', use_mean=False, n_lags=50, n_ac_pts=None, verbose=False, data_save_dir=None, axs_override=None, fig_override=None, dose=None, skip_tight_layout=False, suppress_suptitle=False):
    responses_acf = get_responses_acf(sensory_responses, agent, response, n_delays, delay_interval, method, use_mean, n_lags, n_ac_pts, verbose, data_save_dir)
    time_vals = np.arange(0, n_lags*dt + dt/2, dt)*1000
    
    # Figure/axes setup (support external grid composition)
    if axs_override is None:
        if agent == 'propofol':
            n_plots = 1
            fig, axs = plt.subplots(1, 2, figsize=(4.2, 1.5))
        else:
            n_plots = len(responses_acf[list(responses_acf.keys())[0]].keys())
            fig, axs = plt.subplots(n_plots, 2, figsize=(4.2, 1.5*n_plots))
    else:
        # Use provided axes for grid composition
        fig = fig_override if fig_override is not None else plt.gcf()
        axs = np.array(axs_override)
        n_plots = 1  # For grid, we always plot one dose per subplot
    
    plot_return_data = {}

    for monkey in responses_acf.keys():
        plot_return_data[monkey] = {}
        # Filter doses if specified
        doses_to_plot = [dose] if dose is not None else responses_acf[monkey].keys()
        for dose_level in doses_to_plot:
            if dose_level not in responses_acf[monkey]:
                continue
            plot_return_data[monkey][dose_level] = {}
            
            if agent == 'propofol':
                if monkey == 'Mary':
                    ax_idx = 0
                else:
                    ax_idx = 1
            else:
                if n_plots == 1:
                    ax_idx = 0 if monkey == 'SPOCK' else 1
                else:
                    if monkey == 'SPOCK':
                        ax_idx = (0, 0) if dose_level == 'low' else (1, 0)
                    else:  # PEDRI
                        ax_idx = (0, 1) if dose_level == 'low' else (1, 1)

            for section in responses_acf[monkey][dose_level].keys():
                if responses_acf[monkey][dose_level][section] is None:
                    continue
                mean_trajectory = responses_acf[monkey][dose_level][section].mean(axis=0)
                sem_trajectory = responses_acf[monkey][dose_level][section].std(axis=0) / np.sqrt(responses_acf[monkey][dose_level][section].shape[0])
                plot_return_data[monkey][dose_level][section] = {
                    'mean_trajectory': mean_trajectory,
                    'sem_trajectory': sem_trajectory,
                }
                if 'awake' in section:
                    color = epoch_colors['awake']
                elif 'recovery' in section:
                    color = epoch_colors['emergence']
                else:
                    color = epoch_colors['anesthesia']
                
            
                axs[ax_idx].plot(time_vals, mean_trajectory, color=color)
                axs[ax_idx].fill_between(time_vals, 
                                mean_trajectory - sem_trajectory,
                                mean_trajectory + sem_trajectory,
                                color=color, alpha=0.2)
        
            # Statistical Tests: Compare at each time point
            # --------------------------------------------------
            # We'll run Wilcoxon tests if both data sets exist:
            awake_data = responses_acf[monkey][dose_level].get('awake', None)
            unconscious_data = responses_acf[monkey][dose_level].get('unconscious', None)
            recovery_data = responses_acf[monkey][dose_level].get('recovery', None)

            # Calculate y-coordinates for plotting significance stars at the bottom
            y_min, y_max = axs[ax_idx].get_ylim()
            purple_star_y = y_min - 0.01 * (y_max - y_min)   # Close to the bottom
            orange_star_y = y_min - 0.06 * (y_max - y_min)  # Offset a bit above purple

            # 1) Unconscious vs. Awake
            if (awake_data is not None) and (unconscious_data is not None):
                n_time = min(awake_data.shape[1], unconscious_data.shape[1])
                for t in range(1, n_time):
                    # Perform Wilcoxon rank-sum (or rank test). You can replace with ranksums if you prefer
                    _, pval = wilcoxon(awake_data[:, t], unconscious_data[:, t])
                    if pval < 0.05:
                        # Place a purple star at the bottom
                        axs[ax_idx].plot(time_vals[t], purple_star_y, marker='*', color=epoch_colors['anesthesia'], markersize=2, alpha=0.5)

            # 2) Recovery vs. Awake
            if (awake_data is not None) and (recovery_data is not None):
                n_time = min(awake_data.shape[1], recovery_data.shape[1])
                for t in range(1, n_time):
                    _, pval = wilcoxon(awake_data[:, t], recovery_data[:, t])
                    if pval < 0.05:
                        # Place an orange star at the bottom (slightly above purple star)
                        axs[ax_idx].plot(time_vals[t], orange_star_y, marker='*', color=epoch_colors['emergence'], markersize=2, alpha=0.5)
            # --------------------------------------------------

            axs[ax_idx].set_title(f'{monkey_titles[monkey]}' + ('\n' + dose_level + ' dose' if n_plots > 1 else ''))
            axs[ax_idx].set_xlabel('Time Lag (ms)')
            axs[ax_idx].set_ylabel('Autocorrelation')
            axs[ax_idx].grid(True, alpha=0.3)
    
    # Legend and title handling
    if plot_legend:
        # Create empty lines for legend
        line1, = plt.plot([], [], color=epoch_colors['awake'], label='Awake', visible=True)
        line2, = plt.plot([], [], color=epoch_colors['anesthesia'], label='Anesthesia', visible=True)
        line3, = plt.plot([], [], color=epoch_colors['emergence'], label='Emergence', visible=True)
        
        # Add legend centered below all subplots
        fig.legend(handles=[line1, line2, line3], loc='center', bbox_to_anchor=(0.5, 0), ncol=3)
    
    # Add agent name as title (unless suppressed)
    if not suppress_suptitle:
        fig.suptitle(f'{agent.capitalize()}', c=curve_colors[agent], y=0.9)
    if not skip_tight_layout:
        plt.tight_layout()
    if save_path is not None and axs_override is None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    elif axs_override is None:
        plt.show()
    return plot_return_data

def plot_sensory_responses_acf_grid(cfg, agent_data, curve_colors, epoch_colors,
                                    data_class=None,
                                    data_class_by_agent=None,
                                    n_delays=1,
                                    delay_interval=1,
                                    figsize=(9.0, 4.5),
                                    save_path=None,
                                    method='individual',
                                    use_mean=False,
                                    n_lags=50,
                                    n_ac_pts=None,
                                    verbose=False,
                                    data_save_dir=None,
                                    dose='high',
                                    plot_legend_by_agent=None):
    """
    Compose a 3-row x 2-column grid of sensory response ACF plots:
      - Rows: agents (Propofol, Ketamine, Dexmedetomidine)
      - Columns: NHPs (NHP 1/2 for Propofol; NHP 3/4 for Ketamine and Dexmedetomidine)
    
    Args:
        cfg: Config object
        agent_data: dict keyed by (data_class, agent) with entries containing:
            'sensory_responses', 'leadup', 'response', 'dt'
        curve_colors: dict mapping agent -> color
        epoch_colors: dict mapping epoch -> color
        data_class: (optional) single data_class to use for all agents
        data_class_by_agent: (optional) dict mapping agent -> data_class; overrides single.
        n_delays, delay_interval: ETDC parameters
        figsize: figure size tuple
        save_path: path to save figure
        method, use_mean, n_lags, n_ac_pts, verbose, data_save_dir: ACF parameters
        dose: which dose to plot ('high' or 'low')
        plot_legend_by_agent: dict mapping agent -> bool for whether to show legend
    
    Returns:
        dict mapping (data_class, agent) -> plot_return_data
    """
    agents_in_order = ['propofol', 'ketamine', 'dexmedetomidine']
    
    # Build data_class mapping, with auto-detection from keys
    dc_map = {}
    if isinstance(data_class_by_agent, dict):
        for a in agents_in_order:
            if a in data_class_by_agent:
                dc_map[a] = data_class_by_agent[a]
    if data_class is not None:
        for a in agents_in_order:
            dc_map.setdefault(a, data_class)
    if isinstance(agent_data, dict):
        for (dc, ag) in agent_data.keys():
            if ag in agents_in_order and ag not in dc_map:
                dc_map[ag] = dc
    
    # Preserve original cfg value
    original_dc = getattr(cfg.params, 'data_class', None)
    
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    per_agent_returns = {}
    
    if plot_legend_by_agent is None:
        plot_legend_by_agent = {}
    
    for row_idx, agent in enumerate(agents_in_order):
        agent_dc = dc_map.get(agent, None)
        if agent_dc is None:
            continue
        key = (agent_dc, agent)
        if key not in agent_data:
            continue
        
        # Set cfg for this agent
        cfg.params.data_class = agent_dc
        
        sensory_responses = agent_data[key]['sensory_responses']
        leadup = agent_data[key]['leadup']
        response = agent_data[key]['response']
        dt = agent_data[key]['dt']
        
        # Determine which monkeys to plot
        if agent == 'propofol':
            monkeys = ['Mary', 'MrJones']
        else:
            monkeys = ['SPOCK', 'PEDRI']
        
        # Create filtered sensory_responses dict for this agent with only the specified dose
        # Need to preserve the full nested structure: monkey -> dose -> session -> section
        filtered_sensory_responses = {}
        for monkey in monkeys:
            if monkey not in sensory_responses:
                continue
            if dose not in sensory_responses[monkey]:
                continue
            filtered_sensory_responses[monkey] = {dose: sensory_responses[monkey][dose]}
        
        # Get axes for this row (one for each NHP)
        axs_for_agent = np.array([axes[row_idx, 0], axes[row_idx, 1]])
        
        # Call the plotting function with overrides (don't create legend here, we'll do it at the end)
        returned = plot_sensory_responses_acf(
            agent,
            curve_colors,
            epoch_colors,
            filtered_sensory_responses,
            leadup,
            response,
            dt=dt,
            n_delays=n_delays,
            delay_interval=delay_interval,
            plot_legend=False,  # We'll create a single legend at the end
            save_path=None,
            dims=1,
            method=method,
            use_mean=use_mean,
            n_lags=n_lags,
            n_ac_pts=n_ac_pts,
            verbose=verbose,
            data_save_dir=data_save_dir,
            axs_override=axs_for_agent,
            fig_override=fig,
            dose=dose,
            skip_tight_layout=True,
            suppress_suptitle=True
        )
        per_agent_returns[key] = returned
        
        # Set titles with agent name and NHP name
        for col_idx, monkey in enumerate(monkeys):
            current_title = axes[row_idx, col_idx].get_title()
            axes[row_idx, col_idx].set_title(
                f'{agent.capitalize()} ({monkey_titles[monkey]})',
                color=curve_colors[agent],
                pad=8
            )
    
    # Remove individual axis labels (x-axis shared, y-axis shared)
    for ax in axes.flat:
        ax.set_ylabel('')
        ax.set_xlabel('')
    
    # Set explicit x-axis ticks to show 0, 10, 20, 30, 40, 50 ms
    x_ticks = [0, 10, 20, 30, 40, 50]
    for ax in axes.flat:
        ax.set_xticks(x_ticks)
    
    # Apply tight_layout first, then adjust for labels
    # Using same positioning as ETDC grid based on user's adjustments
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    # Create a single legend for the entire figure if any agent requested it
    if any(plot_legend_by_agent.get(agent, False) for agent in agents_in_order):
        # Create empty lines for legend
        line1, = plt.plot([], [], color=epoch_colors['awake'], label='Awake', visible=True)
        line2, = plt.plot([], [], color=epoch_colors['anesthesia'], label='Anesthesia', visible=True)
        line3, = plt.plot([], [], color=epoch_colors['emergence'], label='Emergence', visible=True)
        
        # Add legend centered below x-axis label with proper spacing (matching ETDC grid)
        fig.legend(handles=[line1, line2, line3], loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=7, frameon=False)
    
    # Shared x-axis label (only show on bottom row) - matching ETDC grid positioning
    fig.text(0.5, 0.04, 'Time Lag (ms)', ha='center', fontsize=8)
    
    # Shared y-axis label (only on left column) - matching ETDC grid positioning
    fig.text(0.03, 0.5, 'Autocorrelation', va='center', rotation='vertical', fontsize=8)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    
    if original_dc is not None:
        cfg.params.data_class = original_dc
    
    return per_agent_returns

def plot_power_analysis(plot_info, data_class, agent, curve_colors, save_path=None):
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2), sharey='row')
    axes = [ax1, ax2]
    monkey_titles = {
        'Mary': 'NHP 1',
        'MrJones': 'NHP 2',
        'SPOCK': 'NHP 3',
        'PEDRI': 'NHP 4'
    }
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    x = np.arange(len(bands))
    width = 0.35

    # Colors for each frequency band
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors = curve_colors[agent]

    data_dict = plot_info[(data_class, agent)]

    monkeys = list(data_dict.keys())
    doses = list(data_dict[monkeys[0]].keys())

    # Process and plot data for each monkey
    for idx, monkey in enumerate(monkeys):
        ax = axes[idx]
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        for dose_idx, dose in enumerate(doses):
            means = []
            sems = []
            
            # Calculate mean and SEM for each frequency band
            for band in bands:
                data = data_dict[monkey][dose][band]
                mean = np.mean(data)
                sem = np.std(data) / np.sqrt(len(data))
                means.append(mean)
                sems.append(sem)  # Fixed: Was using mean instead of sem for error bars
            
            # Create the bar plot with different styles for low/high doses
            # Offset the x positions for low vs high dose
            x_pos = x - width/2 if dose == 'low' else x + width/2
            if dose == 'low':
                bars = ax.bar(x_pos, means, width, yerr=sems, capsize=5, color=colors,
                            alpha=0.5, hatch='///', label='Low dose')
            else:
                bars = ax.bar(x_pos, means, width, yerr=sems, capsize=5, color=colors,
                            label='High dose')
            
        # Add significance stars for each bar
        for band_idx, band in enumerate(bands):
            for dose_idx, dose in enumerate(doses):
                data = data_dict[monkey][dose][band]
                # Perform one-sample t-test against 0
                # t_stat, p_val = stats.ttest_1samp(data, 0)
                stat, p_val = wilcoxon(data) # test against 0

                print(f"{agent} {monkey} {dose} {band}: p-val = {p_val}")
                
                # Determine number of stars based on p-value
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                elif p_val < 0.05:
                    stars = '*'
                else:
                    continue
                    
                # Position stars above or below bar based on mean value
                x_pos = band_idx - width/2 if dose == 'low' else band_idx + width/2
                y_pos = np.mean(data)
                y_pos = (y_pos + (np.std(data) / np.sqrt(len(data)))) if y_pos >= 0 else (y_pos - (np.std(data) / np.sqrt(len(data))))  # Add SEM to position stars above error bars
                offset = 0.02 if y_pos >= 0 else -0.12  # Offset from bar
                if agent == 'propofol':
                    if band in ['beta', 'gamma']:
                        if y_pos >= 0:
                            offset = 0.04
                        else:
                            offset = -0.13
                elif agent == 'dexmedetomidine':
                    if band in ['beta', 'gamma']:
                        if y_pos >= 0:
                            offset = 0.04
                        else:
                            offset = -0.16
                ax.text(x_pos, y_pos + offset, stars, ha='center', va='bottom')
            
        # Customize the plot
        ax.set_title(monkey_titles[monkey])
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45)
        if idx == 0:
            ax.set_ylabel('Correlation Between\nInstability and Power')
        ax.axhline(y=0, color='black', linewidth=0.5)
        # ax.set_ylim(0, 1)

    # Add single ylabel for all subplots
    # fig.text(0.02, 0.5, 'Correlation Between\nInstability and Power', 
    #          va='center', rotation='vertical')

    # Adjust layout and display
    fig.suptitle(f'{agent.capitalize()}', c=curve_colors[agent], y=0.9)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()

def plot_power_analysis_grid(plot_info, anesthetic_agent_list, curve_colors, figsize=None, save_path=None, layout='agent_rows'):
    """
    Create a grid of power analysis plots with configurable layout:
      - layout='agent_rows' (default): 3-row x 2-column grid
        - Rows: agents (Propofol, Ketamine, Dexmedetomidine)
        - Columns: NHPs (NHP 1/2 for Propofol; NHP 3/4 for Ketamine and Dexmedetomidine)
      - layout='nhp_rows': 2-row x 3-column grid
        - Columns: agents (Propofol, Ketamine, Dexmedetomidine)
        - Rows: NHPs (NHP 1/2 for Propofol; NHP 3/4 for Ketamine and Dexmedetomidine)
    
    Args:
        plot_info: dict keyed by (data_class, agent) containing plot data
        anesthetic_agent_list: list of (data_class, agent) tuples
        curve_colors: dict mapping agent -> color
        figsize: figure size tuple. If None, defaults to (9.0, 3.2) for nhp_rows or (6.0, 9.0) for agent_rows
        save_path: optional path to save figure
        layout: 'nhp_rows' for 2x3 grid (NHPs as rows) or 'agent_rows' for 3x2 grid (agents as rows)
    """
    agents_in_order = ['propofol', 'ketamine', 'dexmedetomidine']
    monkey_titles = {
        'Mary': 'NHP 1',
        'MrJones': 'NHP 2',
        'SPOCK': 'NHP 3',
        'PEDRI': 'NHP 4'
    }
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    x = np.arange(len(bands))
    width = 0.35
    
    # Validate layout parameter
    if layout not in ['nhp_rows', 'agent_rows']:
        raise ValueError(f"layout must be 'nhp_rows' or 'agent_rows', got '{layout}'")
    
    # Set default figsize based on layout
    if figsize is None:
        if layout == 'nhp_rows':
            figsize = (9.0, 3.2)
        else:  # agent_rows
            figsize = (6.0, 9.0)
    
    # Create subplot grid based on layout
    if layout == 'nhp_rows':
        # 2 rows (NHPs) x 3 columns (agents)
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharey='row')
    else:  # agent_rows
        # 3 rows (agents) x 2 columns (NHPs)
        fig, axes = plt.subplots(3, 2, figsize=figsize, sharey='row')
    
    # Helper function to plot a single subplot
    def plot_single_subplot(ax, agent, monkey, data_dict, doses, colors):
        """Plot power analysis for a single agent-monkey combination."""
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        for dose_idx, dose in enumerate(doses):
            means = []
            sems = []
            
            # Calculate mean and SEM for each frequency band
            for band in bands:
                data = data_dict[monkey][dose][band]
                mean = np.mean(data)
                sem = np.std(data) / np.sqrt(len(data))
                means.append(mean)
                sems.append(sem)
            
            # Create the bar plot with different styles for low/high doses
            x_pos = x - width/2 if dose == 'low' else x + width/2
            if dose == 'low':
                bars = ax.bar(x_pos, means, width, yerr=sems, capsize=5, color=colors,
                            alpha=0.5, hatch='///', label='Low dose')
            else:
                bars = ax.bar(x_pos, means, width, yerr=sems, capsize=5, color=colors,
                            label='High dose')
        
        # Add significance stars for each bar
        for band_idx, band in enumerate(bands):
            for dose_idx, dose in enumerate(doses):
                data = data_dict[monkey][dose][band]
                stat, p_val = wilcoxon(data)  # test against 0
                
                print(f"{agent} {monkey} {dose} {band}: p-val = {p_val}")
                
                # Determine number of stars based on p-value
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                elif p_val < 0.05:
                    stars = '*'
                else:
                    continue
                
                # Position stars above or below bar based on mean value
                x_pos = band_idx - width/2 if dose == 'low' else band_idx + width/2
                y_pos = np.mean(data)
                y_pos = (y_pos + (np.std(data) / np.sqrt(len(data)))) if y_pos >= 0 else (y_pos - (np.std(data) / np.sqrt(len(data))))
                offset = 0.02 if y_pos >= 0 else -0.12
                if agent == 'propofol':
                    if band in ['beta', 'gamma']:
                        if y_pos >= 0:
                            offset = 0.04
                        else:
                            offset = -0.13
                elif agent == 'dexmedetomidine':
                    if band in ['beta', 'gamma']:
                        if y_pos >= 0:
                            offset = 0.04
                        else:
                            offset = -0.16
                ax.text(x_pos, y_pos + offset, stars, ha='center', va='bottom')
        
        # Customize the plot
        ax.set_title(f'{agent.capitalize()} ({monkey_titles[monkey]})', color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45)
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    if layout == 'nhp_rows':
        # NHPs as rows, agents as columns
        for col_idx, agent in enumerate(agents_in_order):
            # Find the data_class for this agent
            data_class = None
            for dc, ag in anesthetic_agent_list:
                if ag == agent:
                    data_class = dc
                    break
            
            if data_class is None or (data_class, agent) not in plot_info:
                continue
            
            colors = curve_colors[agent]
            data_dict = plot_info[(data_class, agent)]
            monkeys = list(data_dict.keys())
            doses = list(data_dict[monkeys[0]].keys())
            
            # Plot first NHP (row 0) and second NHP (row 1)
            for row_idx, monkey in enumerate(monkeys):
                ax = axes[row_idx, col_idx]
                plot_single_subplot(ax, agent, monkey, data_dict, doses, colors)
                # Set ylabel only on leftmost column
                if col_idx == 0:
                    ax.set_ylabel('Correlation Between\nInstability and Power')
    else:  # agent_rows
        # Agents as rows, NHPs as columns
        for row_idx, agent in enumerate(agents_in_order):
            # Find the data_class for this agent
            data_class = None
            for dc, ag in anesthetic_agent_list:
                if ag == agent:
                    data_class = dc
                    break
            
            if data_class is None or (data_class, agent) not in plot_info:
                continue
            
            colors = curve_colors[agent]
            data_dict = plot_info[(data_class, agent)]
            monkeys = list(data_dict.keys())
            doses = list(data_dict[monkeys[0]].keys())
            
            # Process and plot data for each monkey (column)
            for col_idx, monkey in enumerate(monkeys):
                ax = axes[row_idx, col_idx]
                plot_single_subplot(ax, agent, monkey, data_dict, doses, colors)
                # Set ylabel only on leftmost column
                if col_idx == 0:
                    ax.set_ylabel('Correlation Between\nInstability and Power')
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()

def plot_single_session_tracking(session, pct_correct_thresh, time_vals, stab_means, stab_sems, high_perf_times, high_perf_mask, curve_color, img_save_dir):
    plt.plot(time_vals, stab_means, color=curve_color)
    plt.fill_between(time_vals, stab_means - stab_sems, stab_means + stab_sems, alpha=0.2, color=curve_color)
    plt.fill_between(high_perf_times, plt.ylim()[0], plt.ylim()[1], 
                        where=high_perf_mask,
                        color='green', alpha=0.1)
    plt.xlabel('Time Relative to Anesthesia Start (min)')
    plt.ylabel('Mean Instability ($s^{-1}$)')
    plt.savefig(os.path.join(img_save_dir, f'{session}_tracking_pct_correct_{pct_correct_thresh}.pdf'), dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

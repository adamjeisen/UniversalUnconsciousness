import h5py
import itertools
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
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

def _plot_statistics(ax, aligned_data, common_times, curve_color, timescales=True, aligned_data_unnormalized=None, baseline=None, plot_stars=True, baseline_inverse_timescales=None, aligned_data_inverse_timescales=None, label=None):
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
        # For each time point, perform Wilcoxon paired rank test
        # between aligned_data at that time point and the baseline
        ylim = ax.get_ylim()
        star_y_pos = ylim[1] - 0.05 * (ylim[1] - ylim[0])  # Position stars near the top of the plot
        
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

        for t in range(len(p_vals_all)):
            if reject[t]:
                star_marker = '*'
                # if p_val < 0.01:
                #     star_marker = '**'
                # if p_val < 0.001:
                #     star_marker = '***'
                    
                ax.text(common_times[t], star_y_pos, star_marker, 
                        horizontalalignment='center', color=curve_color, 
                        verticalalignment='center', fontsize=8)

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
                                  data_type='delase'):
    is_lever = 'ket' in agent.lower() or 'dex' in agent.lower()

    curve_color = curve_colors[agent]
    loc_roc_color = loc_roc_colors[agent]

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
    fig.suptitle(f'{agent.capitalize()}', c=curve_color, y=0.9)

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
            ax.set_title(f"{monkey_titles[monkey]}") if len(doses) == 1 else ax.set_title(f"{monkey_titles[monkey]} {dose} dose")
            
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
            for area_name in areas_list:
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
                    label=(area_name if multiple_areas else None)
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
            if multiple_areas:
                ax.legend(fontsize=7)

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
    for ax in axs.flat:
        ax.axvline(0, c='k', ls='--', label='Infusion Start')
        if timescales:
            ax.axhline(1, c='k', ls=':', label='Baseline')
        ax.set_xlabel('Time Relative to Infusion Start (min)')
        if timescales:
            ax.set_ylabel('Mean Characteristic Timescale\nRatio to Awake Baseline')
        else:
            ax.set_ylabel('Mean Instability ($s^{-1}$)')
        # ax.tick_params(labelsize=9)
        # ax.legend(fontsize=9)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    if return_data:
        return plot_return_data

def plot_section_stability_boxes(cfg, agent, session_lists, delase_results, top_percent=0.1,
                                  curve_colors={'propofol': 'blue', 'dexmedetomidine': 'purple', 'ketamine': 'red'},
                                  figsize=None,
                                  dose=None,
                                  save_path=None,
                                  section_info_type='plot',
                                  verbose=False,
                                  area='all',
                                  data_type='delase'):
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

    if figsize is None:
        figsize = (4.2, 2)

    # Create figure based on number of doses
    if len(doses) > 1:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Add agent name as title
    fig.suptitle(f'{agent.capitalize()}', c=curve_colors[agent], y=0.9)

    # For each monkey
    for i, monkey in enumerate(session_lists.keys()):
        for j, dose in enumerate(doses):
            if dose not in session_lists[monkey]:
                continue
            # Get subplot
            ax = axs[i, j] if len(doses) > 1 else axs[i]
            
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
                    

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    
    return section_means


def plot_sensory_responses_etdc(agent, curve_colors, epoch_colors, sensory_responses, leadup, response, dt=0.001, n_delays=1, delay_interval=1, plot_legend=False, save_path=None, dims=1, use_mean=False, min_time=None, max_time=None):
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

    if agent == 'propofol':
        n_plots = 1
        fig, axs = plt.subplots(1, 2, figsize=(4.2, 1.5))
    else:
        n_plots = len(responses_etdc[list(responses_etdc.keys())[0]].keys())
        fig, axs = plt.subplots(n_plots, 2, figsize=(4.2, 1.5*n_plots))


    plot_return_data = {}

    for monkey in responses_etdc.keys():
        plot_return_data[monkey] = {}
        for dose in responses_etdc[monkey].keys():
            plot_return_data[monkey][dose] = {}
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
                        ax_idx = (0, 0) if dose == 'low' else (1, 0)
                    else:  # PEDRI
                        ax_idx = (0, 1) if dose == 'low' else (1, 1)
                
            for section in responses_etdc[monkey][dose].keys():
                if dims == 1:
                    mean_trajectory = responses_etdc[monkey][dose][section].mean(axis=0)[:, 0]
                    sem_trajectory = responses_etdc[monkey][dose][section].std(axis=0)[:, 0] / np.sqrt(responses_etdc[monkey][dose][section].shape[0])
                elif dims == 2:
                    mean_trajectory = responses_etdc[monkey][dose][section].mean(axis=0)[:, :dims]
                    sem_trajectory = responses_etdc[monkey][dose][section].std(axis=0)[:, :dims] / np.sqrt(responses_etdc[monkey][dose][section].shape[0])
                
                plot_return_data[monkey][dose][section] = {
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
                for i in range(responses_etdc[monkey][dose][section].shape[0]):
                    if dims == 1:
                        axs[ax_idx].plot(time_vals[plot_indices], responses_etdc[monkey][dose][section][i, plot_indices, 0], 
                                    color=color, alpha=0.1)
                    # if dims == 2:
                    #     axs[ax_idx].plot(responses_etdc[monkey][dose][section][i, :, 0], 
                    #                 responses_etdc[monkey][dose][section][i, :, 1], 
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
            
            
            axs[ax_idx].set_title(f'{monkey_titles[monkey]}' + ('\n' + dose + ' dose' if n_plots > 1 else ''))
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
    
    # Add agent name as title
    fig.suptitle(f'{agent.capitalize()}', c=curve_colors[agent], y=0.9)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    return plot_return_data

def plot_sensory_responses_acf(agent, curve_colors, epoch_colors, sensory_responses, leadup, response, dt=0.001, n_delays=1, delay_interval=1, plot_legend=False, save_path=None, dims=1, method='grouped', use_mean=False, n_lags=50, n_ac_pts=None, verbose=False, data_save_dir=None):
    responses_acf = get_responses_acf(sensory_responses, agent, response, n_delays, delay_interval, method, use_mean, n_lags, n_ac_pts, verbose, data_save_dir)
    time_vals = np.arange(0, n_lags*dt + dt/2, dt)
    
    if agent == 'propofol':
        n_plots = 1
        fig, axs = plt.subplots(1, 2, figsize=(4.2, 1.5))
    else:
        n_plots = len(responses_acf[list(responses_acf.keys())[0]].keys())
        fig, axs = plt.subplots(n_plots, 2, figsize=(4.2, 1.5*n_plots))
    
    plot_return_data = {}

    for monkey in responses_acf.keys():
        plot_return_data[monkey] = {}
        for dose in responses_acf[monkey].keys():
            plot_return_data[monkey][dose] = {}
            
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
                        ax_idx = (0, 0) if dose == 'low' else (1, 0)
                    else:  # PEDRI
                        ax_idx = (0, 1) if dose == 'low' else (1, 1)

            for section in responses_acf[monkey][dose].keys():
                if responses_acf[monkey][dose][section] is not None:    
                    mean_trajectory = responses_acf[monkey][dose][section].mean(axis=0)
                    sem_trajectory = responses_acf[monkey][dose][section].std(axis=0) / np.sqrt(responses_acf[monkey][dose][section].shape[0])
                    plot_return_data[monkey][dose][section] = {
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
            awake_data = responses_acf[monkey][dose].get('awake', None)
            unconscious_data = responses_acf[monkey][dose].get('unconscious', None)
            recovery_data = responses_acf[monkey][dose].get('recovery', None)

            # Calculate y-coordinates for plotting significance stars at the bottom
            y_min, y_max = axs[ax_idx].get_ylim()
            purple_star_y = y_min + 0.04 * (y_max - y_min)   # Slightly above the bottom
            orange_star_y = y_min + 0.08 * (y_max - y_min)  # Offset a bit above purple

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

            axs[ax_idx].set_title(f'{monkey_titles[monkey]}' + ('\n' + dose + ' dose' if n_plots > 1 else ''))
            axs[ax_idx].set_xlabel('Time Lag (s)')
            axs[ax_idx].set_ylabel('Autocorrelation')
            axs[ax_idx].grid(True, alpha=0.3)
        
        if plot_legend:
            # Create empty lines for legend
            line1, = plt.plot([], [], color=epoch_colors['awake'], label='Awake', visible=True)
            line2, = plt.plot([], [], color=epoch_colors['anesthesia'], label='Anesthesia', visible=True)
            line3, = plt.plot([], [], color=epoch_colors['emergence'], label='Emergence', visible=True)
            
            
            # Add legend centered below all subplots
            fig.legend(handles=[line1, line2, line3], loc='center', bbox_to_anchor=(0.5, 0), ncol=3)
    
    # Add agent name as title
    fig.suptitle(f'{agent.capitalize()}', c=curve_colors[agent], y=0.9)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    return plot_return_data

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




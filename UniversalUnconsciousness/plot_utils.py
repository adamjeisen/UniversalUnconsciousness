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

from .data_utils import get_loc_roc, get_section_info
from .sensory_responses import get_responses_etdc, get_responses_acf

# Set up figure based on data class
monkey_titles = {
    'Mary': 'NHP1',
    'MrJones': 'NHP2',
    'SPOCK': 'NHP3',
    'PEDRI': 'NHP4'
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

def get_session_plot_info(cfg, session_list, verbose=False):
    if 'Lvr' in cfg.params.data_class or 'lever' in cfg.params.data_class:
        # Create a dictionary to organize sessions by monkey and dose
        session_lists = {
            'SPOCK': {'low': [], 'high': []},
            'PEDRI': {'low': [], 'high': []}
        }

        # Populate the dictionary
        for session in session_list:
            monkey = session.split('_')[0]
            dose = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))['sessionInfo']['dose'][0, 0]
            if dose in [1.0, 5.0]:
                session_lists[monkey]['low'].append(session)
            if dose in [10.0, 20.0]:
                session_lists[monkey]['high'].append(session)
        
        locs ={}
        rocs = {}
        ropaps = {}
        iterator = tqdm(total=len(session_list), disable=not verbose)
        for monkey in session_lists.keys():
            locs[monkey] = {}
            rocs[monkey] = {}
            ropaps[monkey] = {}
            for dose in session_lists[monkey].keys():
                locs[monkey][dose] = []
                rocs[monkey][dose] = []
                ropaps[monkey][dose] = []
                for session in session_lists[monkey][dose]:
                    loc, roc, ropap = get_loc_roc(cfg, h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat")))
                    infusion_start = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))['sessionInfo']['infusionStart'][0, 0]
                    locs[monkey][dose].append(loc - infusion_start)
                    rocs[monkey][dose].append(roc - infusion_start)
                    ropaps[monkey][dose].append(ropap - infusion_start)

                    iterator.update()
        iterator.close()
    else: # propofol is the data class
        session_lists = {
            'Mary': {'high': []},
            'MrJones': {'high': []}
        }

        for session in session_list:
            monkey = session.split('-')[0]
            session_lists[monkey]['high'].append(session)

        locs = {}
        rocs = {}
        ropaps = {}
        iterator = tqdm(total=len(session_list), disable=not verbose)
        for monkey in session_lists.keys():
            locs[monkey] = {}
            rocs[monkey] = {}
            ropaps[monkey] = {}
            for dose in session_lists[monkey].keys():
                locs[monkey][dose] = []
                rocs[monkey][dose] = []
                ropaps[monkey][dose] = []
                for session in session_lists[monkey][dose]:
                    loc, roc, ropap = get_loc_roc(cfg, h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f"{session}.mat")))
                    infusion_start = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f"{session}.mat"))['sessionInfo']['drugStart'][0, 0]
                    locs[monkey][dose].append(loc - infusion_start)
                    rocs[monkey][dose].append(roc - infusion_start)
                    ropaps[monkey][dose].append(ropap - infusion_start)

                iterator.update()
        iterator.close()
    # everything is relative to the infusion start, so infusion start is 0
    return session_lists, locs, rocs, ropaps

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

def _process_session_data(session, delase_results, session_file, infusion_time, common_times, top_percent, plot_range, interpolate=True, timescales=True):
    """Helper function to process individual session data."""
    time_vals = delase_results[session]['all'].window_start.values
    
    if timescales:
    # Calculate baseline and normalized timescales
        baseline = _calculate_baseline(
            delase_results[session]['all'].stability_params,
            time_vals, 
            infusion_time,
            plot_range,
            top_percent
        )

        if baseline == np.nan:
            return np.nan

        stability_vals = delase_results[session]['all'].stability_params.apply(
            lambda x: _process_stability(x, top_percent)
        ).values / baseline
    else:
        stability_vals = delase_results[session]['all'].stability_params.apply(lambda x: x[:int(top_percent*len(x))].mean())
    
    # Align to infusion start and interpolate
    aligned_times = (time_vals - infusion_time) / 60
    if interpolate:
        interpolated = np.interp(common_times, aligned_times, stability_vals)
    else:
        interpolated = stability_vals
    return interpolated

def _plot_statistics(ax, aligned_data, common_times, curve_color, timescales=True):
    """Helper function to plot geometric mean and standard error."""
    if timescales:
        log_data = np.log(aligned_data)
        mean_log = np.nanmean(log_data, axis=0)
        sem_log = np.nanstd(log_data, axis=0) / np.sqrt(np.sum(~np.isnan(log_data), axis=0))
        
        mean_stability = np.exp(mean_log)
        upper_bound = np.exp(mean_log + sem_log)
        lower_bound = np.exp(mean_log - sem_log)
        ax.plot(common_times, mean_stability, label='Geometric Mean', color=curve_color)
        ax.fill_between(common_times, lower_bound, upper_bound, alpha=0.3, color=curve_color)
    else:
        mean_stability = np.nanmean(aligned_data, axis=0)
        sem_stability = np.nanstd(aligned_data, axis=0) / np.sqrt(np.sum(~np.isnan(aligned_data), axis=0))
        ax.plot(common_times, mean_stability, label='Mean', color=curve_color)
        ax.fill_between(common_times, mean_stability - sem_stability, mean_stability + sem_stability, alpha=0.3, color=curve_color)

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
    common_times = np.arange(plot_range[0], plot_range[1], 1/60)

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
                                                    common_times, top_percent, plot_range, interpolate=False)
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
                                  verbose=False):
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
    common_times = np.arange(plot_range[0], plot_range[1], 1/60)
    
    # Process data for each subplot
    for i, monkey in enumerate(monkeys):
        for j, dose in enumerate(doses):
            ax = axs[i, j] if len(doses) > 1 else axs[i]
            ax.set_title(f"{monkey_titles[monkey]}") if len(doses) == 1 else ax.set_title(f"{monkey_titles[monkey]} {dose} dose")
            
            aligned_data = []
            if dose not in session_lists[monkey]:
                continue
            for session in session_lists[monkey][dose]:
                if is_lever:
                    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))
                    infusion_time = session_file['sessionInfo']['infusionStart'][0, 0]
                else:
                    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f"{session}.mat"))
                    infusion_time = session_file['sessionInfo']['drugStart'][0]
                
                interpolated = _process_session_data(session, delase_results, session_file, infusion_time, 
                                                    common_times, top_percent, plot_range, timescales=timescales)
                
                aligned_data.append(interpolated)
            
            _plot_statistics(ax, np.array(aligned_data), common_times, curve_color, timescales=timescales)
            # _add_roc_ropap_lines(ax, rocs[monkey][dose], ropaps[monkey][dose])
            _add_loc_roc_region(ax, locs[monkey][dose], rocs[monkey][dose], loc_roc_color)
                
    
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

def plot_section_stability_boxes(cfg, agent, session_lists, delase_results, top_percent=0.1,
                                  curve_colors={'propofol': 'blue', 'dexmedetomidine': 'purple', 'ketamine': 'red'},
                                  figsize=None,
                                  dose=None,
                                  save_path=None,
                                  verbose=False):
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
                section_info, section_info_extended, section_colors, infusion_start = get_section_info(session, cfg.params.all_data_dir, cfg.params.data_class)
                session_delase_results = delase_results[session]['all']
                
                # Convert times to minutes relative to infusion
                time_vals = (session_delase_results.window_start - infusion_start) / 60
                
                section_means[monkey][_dose][session] = {}
                for section_name, (start_time, end_time) in section_info:
                    # Get indices for times in this section
                    section_mask = (time_vals >= start_time) & (time_vals < end_time)
                    
                    # Get stability params for times in this section
                    section_stability = session_delase_results.stability_params[section_mask].apply(lambda x: x[:int(len(x)*top_percent)]).values
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
            stat, p_val = wilcoxon(unconscious_vals, awake_vals)
            mean_diff = np.mean(unconscious_vals) - np.mean(awake_vals)
            
            # Determine significance stars based on the p-value
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**' 
            elif p_val < 0.05:
                stars = '*'
            else:
                stars = 'ns'
            
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
                # Draw vertical ticks at the ends
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
                awake_idx = labels.index('awake oddball')
                recovery_idx = labels.index('recovery oddball')
                x1 = awake_idx + 1
                x2 = recovery_idx + 1
                y_bar_max = np.max(box_data_max[awake_idx:recovery_idx + 1])
                y_bar_min = np.min(box_data_min[awake_idx:recovery_idx + 1])
                axis_range = y_bar_max - y_bar_min
                offset = axis_range * 0.05   # For example, 5% of the axis range
                tick_length = offset * 0.3


                awake_vals = [section_means[monkey][dose][session]['awake oddball'] 
                              for session in session_lists[monkey][dose]]
                unconscious_vals = [section_means[monkey][dose][session]['recovery oddball']
                                    for session in session_lists[monkey][dose]]
                
                stat, p_val = wilcoxon(unconscious_vals, awake_vals)
                mean_diff = np.mean(unconscious_vals) - np.mean(awake_vals)
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


def plot_sensory_responses_etdc(agent, curve_colors, sensory_responses, leadup, response, dt=0.001, n_delays=1, delay_interval=1, plot_legend=False, save_path=None, dims=1, use_mean=False, min_time=None, max_time=None):
    if use_mean and dims > 1:
        raise ValueError('use_mean is not supported for dims > 1')
    responses_etdc = get_responses_etdc(sensory_responses, n_delays, delay_interval, use_mean)
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

    for monkey in responses_etdc.keys():
        for dose in responses_etdc[monkey].keys():
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
                
                color = 'green' if 'awake' in section else 'orange' if 'recovery' in section else 'purple'
                
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
    # if plot_legend is True, add one green line (awake) and one purple line (unconscious)
    if plot_legend:
        # Create empty lines for legend
        line1, = plt.plot([], [], color='green', label='awake', visible=True)
        line2, = plt.plot([], [], color='orange', label='recovery', visible=True)
        line3, = plt.plot([], [], color='purple', label='anesthesia', visible=True)
        
        # Add legend centered below all subplots
        fig.legend(handles=[line1, line2, line3], loc='center', bbox_to_anchor=(0.5, 0), ncol=3)
    
    # Add agent name as title
    fig.suptitle(f'{agent.capitalize()}', c=curve_colors[agent], y=0.9)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()

def plot_sensory_responses_acf(agent, curve_colors, sensory_responses, leadup, response, dt=0.001, n_delays=1, delay_interval=1, plot_legend=False, save_path=None, dims=1, method='grouped', use_mean=False, n_lags=50, n_ac_pts=None, verbose=False, data_save_dir=None):
    responses_acf = get_responses_acf(sensory_responses, agent, response, n_delays, delay_interval, method, use_mean, n_lags, n_ac_pts, verbose, data_save_dir)
    time_vals = np.arange(0, n_lags*dt + dt/2, dt)
    
    if agent == 'propofol':
        n_plots = 1
        fig, axs = plt.subplots(1, 2, figsize=(4.2, 1.5))
    else:
        n_plots = len(responses_acf[list(responses_acf.keys())[0]].keys())
        fig, axs = plt.subplots(n_plots, 2, figsize=(4.2, 1.5*n_plots))

    for monkey in responses_acf.keys():
        for dose in responses_acf[monkey].keys():
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

                    if 'awake' in section:
                        color = 'green'
                    elif 'recovery' in section:
                        color = 'orange'
                    else:
                        color = 'purple'
                    
                
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
                        axs[ax_idx].plot(time_vals[t], purple_star_y, marker='*', color='purple', markersize=2, alpha=0.5)

            # 2) Recovery vs. Awake
            if (awake_data is not None) and (recovery_data is not None):
                n_time = min(awake_data.shape[1], recovery_data.shape[1])
                for t in range(1, n_time):
                    _, pval = wilcoxon(awake_data[:, t], recovery_data[:, t])
                    if pval < 0.05:
                        # Place an orange star at the bottom (slightly above purple star)
                        axs[ax_idx].plot(time_vals[t], orange_star_y, marker='*', color='orange', markersize=2, alpha=0.5)
            # --------------------------------------------------

            axs[ax_idx].set_title(f'{monkey_titles[monkey]}' + ('\n' + dose + ' dose' if n_plots > 1 else ''))
            axs[ax_idx].set_xlabel('Time Lag (s)')
            axs[ax_idx].set_ylabel('Autocorrelation')
            axs[ax_idx].grid(True, alpha=0.3)
        
        if plot_legend:
            # Create empty lines for legend
            line1, = plt.plot([], [], color='green', label='awake', visible=True)
            line2, = plt.plot([], [], color='orange', label='recovery', visible=True)
            line3, = plt.plot([], [], color='purple', label='unconscious', visible=True)
            
            # Add legend centered below all subplots
            fig.legend(handles=[line1, line2, line3], loc='center', bbox_to_anchor=(0.5, 0), ncol=3)
    
    # Add agent name as title
    fig.suptitle(f'{agent.capitalize()}', c=curve_colors[agent], y=0.9)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()

def plot_power_analysis(plot_info, data_class, agent, curve_colors, save_path=None):
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2), sharey='row')
    axes = [ax1, ax2]
    monkey_titles = {
        'Mary': 'NHP1',
        'MrJones': 'NHP2',
        'SPOCK': 'NHP3',
        'PEDRI': 'NHP4'
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
                t_stat, p_val = stats.ttest_1samp(data, 0)
                
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




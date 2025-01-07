import h5py
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import os
import scipy
from tqdm.auto import tqdm

from .data_utils import get_loc_roc, get_section_info

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

# def plot_session_stability_separate(cfg, session_lists, delase_results, locs, rocs, ropaps, verbose=False):
#     top_percent = 0.1

#     if 'Lvr' in cfg.params.data_class or 'lever' in cfg.params.data_class:  
#         # Create a 2x2 grid of subplots
#         fig, axs = plt.subplots(2, 2, figsize=(15, 8))
#         fig.suptitle('DeLASE Results by Monkey and Dose', fontsize=14)

#         # Plot for each monkey and dose
#         for i, monkey in enumerate(['SPOCK', 'PEDRI']):
#             for j, dose in enumerate(['low', 'high']):
#                 ax = axs[i, j]
#                 ax.set_title(f"{monkey} - Dose {dose}", fontsize=12)
                
#                 for session in session_lists[monkey][dose]:
#                     session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))
#                     infusion_start = session_file['sessionInfo']['infusionStart'][0, 0]
#                     time_vals = (delase_results[session]['all'].window_start - infusion_start)/60
#                     stability_vals = delase_results[session]['all'].stability_params.apply(lambda x: x[:int(len(x)*top_percent)].mean())
#                     ax.plot(time_vals, stability_vals, label=f"Session {session}")
#                 ax.axvline(0, c='k', ls='--')
                
#                 ax.set_xlabel('Time Relative to Infusion (min)', fontsize=10)
#                 ax.set_ylabel('Mean Stability', fontsize=10)
#                 ax.tick_params(labelsize=9)
#                 ax.legend(loc='upper right', fontsize=9)  # Moved legend to upper right for better visibility


#         for i, monkey in enumerate(['SPOCK', 'PEDRI']):
#             for j, dose in enumerate(['low', 'high']):
#                 ax = axs[i, j]
#                 ylim = ax.get_ylim()
#                 roc_vals = (np.array(rocs[monkey][dose]))/60
#                 ropap_vals = (np.array(ropaps[monkey][dose]))/60

#                 mean_roc = np.mean(roc_vals)
#                 mean_ropap = np.mean(ropap_vals)
#                 sem_roc = np.std(roc_vals) / np.sqrt(len(roc_vals))
#                 sem_ropap = np.std(ropap_vals) / np.sqrt(len(ropap_vals))
#                 ax.axvline(mean_roc, c='g', ls='-', label=f"ROC mean: {mean_roc:.2f} ± {sem_roc:.2f}")
#                 ax.axvline(mean_ropap, c='orange', ls='-', label=f"ROPAP mean: {mean_ropap:.2f} ± {sem_ropap:.2f}")
#                 ax.fill_betweenx(ylim, mean_roc - sem_roc, mean_roc + sem_roc, alpha=0.3, color='g')
#                 ax.fill_betweenx(ylim, mean_ropap - sem_ropap, mean_ropap + sem_ropap, alpha=0.3, color='orange')
#     else: # propofol is the data class
#         fig, axs = plt.subplots(2, 1, figsize=(15, 8))
#         fig.suptitle('DeLASE Results by Monkey and Dose', fontsize=14)  

#         for i, monkey in enumerate(['Mary', 'MrJones']):
#             ax = axs[i]
#             ax.set_title(f"{monkey}", fontsize=12)

#             for session in session_lists[monkey]:
#                 session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f"{session}.mat"))
#                 infusion_start = session_file['sessionInfo']['drugStart'][0, 0]
#                 time_vals = (delase_results[session]['all'].window_start - infusion_start)/60
#                 stability_vals = delase_results[session]['all'].stability_params.apply(lambda x: x[:int(len(x)*top_percent)].mean())
#                 ax.plot(time_vals, stability_vals, label=f"Session {session}")
#             ax.axvline(0, c='k', ls='--')
            
#             ax.set_xlabel('Time Relative to Infusion (min)', fontsize=10)
#             ax.set_ylabel('Mean Stability', fontsize=10)
#             ax.tick_params(labelsize=9)
#             ax.legend(loc='upper right', fontsize=9)
        
#         for i, monkey in enumerate(['Mary', 'MrJones']):
#             ax = axs[i]
#             ylim = ax.get_ylim()
#             #TODO: use infusion start from each individual session
#             roc_vals = (np.array(rocs[monkey]))/60

#             mean_roc = np.mean(roc_vals)
#             # mean_ropap = np.mean(ropap_vals)
#             sem_roc = np.std(roc_vals) / np.sqrt(len(roc_vals))
#             # sem_ropap = np.std(ropap_vals) / np.sqrt(len(ropap_vals))
#             ax.axvline(mean_roc, c='g', ls='-', label=f"ROC mean: {mean_roc:.2f} ± {sem_roc:.2f}")
#             # ax.axvline(mean_ropap, c='orange', ls='-', label=f"ROPAP mean: {mean_ropap:.2f} ± {sem_ropap:.2f}")
#             ax.fill_betweenx(ylim, mean_roc - sem_roc, mean_roc + sem_roc, alpha=0.3, color='g')
#             # ax.fill_betweenx(ylim, mean_ropap - sem_ropap, mean_ropap + sem_ropap, alpha=0.3, color='orange')

#     plt.tight_layout()
#     plt.show()

def _process_stability(x, top_percent):
    """Helper function to process stability parameters into timescales."""
    # Keep only negative roots and convert to timescales
    timescales = -1/np.array([r for r in x if r < 0])
    if len(timescales) == 0:
        return np.nan
    # Take only top percentage of timescales
    n_keep = max(1, int(len(timescales) * top_percent))
    top_timescales = np.sort(timescales)[-n_keep:]
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
    return np.exp(np.mean(np.log(baseline_timescales)))

def _process_session_data(session, delase_results, session_file, infusion_time, common_times, top_percent, plot_range):
    """Helper function to process individual session data."""
    time_vals = delase_results[session]['all'].window_start.values
    
    # Calculate baseline and normalized timescales
    baseline = _calculate_baseline(
        delase_results[session]['all'].stability_params,
        time_vals, 
        infusion_time,
        plot_range,
        top_percent
    )
    
    stability_vals = delase_results[session]['all'].stability_params.apply(
        lambda x: _process_stability(x, top_percent)
    ).values / baseline
    
    # Align to infusion start and interpolate
    aligned_times = (time_vals - infusion_time) / 60
    interpolated = np.interp(common_times, aligned_times, stability_vals)
    return interpolated

def _plot_statistics(ax, aligned_data, common_times, curve_color):
    """Helper function to plot geometric mean and standard error."""
    log_data = np.log(aligned_data)
    mean_log = np.nanmean(log_data, axis=0)
    sem_log = np.nanstd(log_data, axis=0) / np.sqrt(np.sum(~np.isnan(log_data), axis=0))
    
    mean_stability = np.exp(mean_log)
    upper_bound = np.exp(mean_log + sem_log)
    lower_bound = np.exp(mean_log - sem_log)
    
    ax.plot(common_times, mean_stability, label='Geometric Mean', color=curve_color)
    ax.fill_between(common_times, lower_bound, upper_bound, alpha=0.3, color=curve_color)

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


def plot_session_timescales_grouped(cfg, agent, session_lists, delase_results, locs, rocs, ropaps, plot_range=(-15, 85), top_percent=0.1, 
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
            for session in session_lists[monkey][dose]:
                if is_lever:
                    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))
                    infusion_time = session_file['sessionInfo']['infusionStart'][0, 0]
                else:
                    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f"{session}.mat"))
                    infusion_time = session_file['sessionInfo']['drugStart'][0]
                
                interpolated = _process_session_data(session, delase_results, session_file, infusion_time, 
                                                    common_times, top_percent, plot_range)
                aligned_data.append(interpolated)
            
            _plot_statistics(ax, np.array(aligned_data), common_times, curve_color)
            # _add_roc_ropap_lines(ax, rocs[monkey][dose], ropaps[monkey][dose])
            _add_loc_roc_region(ax, locs[monkey][dose], rocs[monkey][dose], loc_roc_color)
                
    
    # Add common elements to all subplots
    for ax in axs.flat:
        ax.axvline(0, c='k', ls='--', label='Infusion Start')
        ax.axhline(1, c='k', ls=':', label='Baseline')
        ax.set_xlabel('Time Relative to Infusion Start (min)')
        ax.set_ylabel('Mean Characteristic Timescale\nRatio to Awake Baseline')
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

    for monkey in session_lists.keys():
        section_means[monkey] = {}
        for dose in session_lists[monkey].keys():
            section_means[monkey][dose] = {}
            for session in session_lists[monkey][dose]:
                section_info, section_info_extended, section_colors, infusion_start = get_section_info(session, cfg.params.all_data_dir, cfg.params.data_class)
                session_delase_results = delase_results[session]['all']
                
                # Convert times to minutes relative to infusion
                time_vals = (session_delase_results.window_start - infusion_start) / 60
                
                section_means[monkey][dose][session] = {}
                for section_name, (start_time, end_time) in section_info:
                    # Get indices for times in this section
                    section_mask = (time_vals >= start_time) & (time_vals < end_time)
                    
                    # Get stability params for times in this section
                    section_stability = session_delase_results.stability_params[section_mask].apply(lambda x: x[:int(len(x)*top_percent)]).values
                    if len(section_stability) > 0:
                        section_stability = np.hstack(section_stability)
                        section_means[monkey][dose][session][section_name] = np.mean(section_stability.mean())

    if figsize is None:
        figsize = (4.2, 2)

    if dose is None:
        doses = session_lists[monkey].keys()
    else:
        doses = [dose]

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
            # Get subplot
            ax = axs[i, j] if len(doses) > 1 else axs[i]
            
            # Collect data for boxplot
            box_data = []
            colors = []
            labels = []
            for section_name, _ in section_info:
                section_values = [section_means[monkey][dose][session][section_name] 
                                for session in session_lists[monkey][dose]]
                box_data.append(section_values)
                colors.append(section_colors[section_name])
                labels.append(section_name)

            # Create boxplot with normal whiskers
            bp = ax.boxplot(box_data, patch_artist=True, 
                        showmeans=False, meanline=False)

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

            # Perform statistical test
            if is_lever:
                # Get awake and unconscious oddball values
                awake_vals = [section_means[monkey][dose][session]['awake oddball'] 
                            for session in session_lists[monkey][dose]]
                unconscious_vals = [section_means[monkey][dose][session]['unconscious oddball']
                                for session in session_lists[monkey][dose]]
                
                # Perform t-test
                t_stat, p_val = scipy.stats.ttest_ind(unconscious_vals, awake_vals)
                
                # Add stars based on p-value
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**' 
                elif p_val < 0.05:
                    stars = '*'
                else:
                    stars = 'ns'
                    
                if t_stat > 0:  # Only show if unconscious > awake
                    # Get positions for the significance bars
                    awake_idx = labels.index('awake oddball')
                    unconscious_idx = labels.index('unconscious oddball')
                    
                    # Get min height (most negative) of boxes and whiskers
                    max_heights = []
                    for i in range(min(awake_idx, unconscious_idx), max(awake_idx, unconscious_idx) + 1):
                        max_heights.extend(box_data[i])
                    y_min = min(max_heights)  # Most negative value
                    
                    # Plot significance bar with more space above and downward-pointing tips
                    if 'dex' in agent.lower():
                        bar_height = y_min * 0.3  # Much smaller multiplier to move bar higher (closer to 0)
                        bar_tips = y_min * 0.35   # Tips extend downward (away from 0)
                    else:
                        bar_height = y_min * 0.5  # Much smaller multiplier to move bar higher (closer to 0)
                        bar_tips = y_min * 0.6   # Tips extend downward (away from 0)
                    ax.plot([awake_idx + 1, unconscious_idx + 1], [bar_height, bar_height], 'k-')
                    ax.plot([awake_idx + 1, awake_idx + 1], [bar_height, bar_tips], 'k-')
                    ax.plot([unconscious_idx + 1, unconscious_idx + 1], [bar_height, bar_tips], 'k-')
                    
                    # Add stars below the bar
                    ax.text((awake_idx + unconscious_idx + 2)/2, bar_height * 0.9,  # Position below bar
                        stars, ha='center', va='top')
            else:
                # Get awake and loading dose values
                awake_vals = [section_means[monkey][dose][session]['awake'] 
                            for session in session_lists[monkey][dose]]
                loading_vals = [section_means[monkey][dose][session]['loading dose']
                            for session in session_lists[monkey][dose]]
                
                # Perform t-test
                t_stat, p_val = scipy.stats.ttest_ind(loading_vals, awake_vals)
                
                # Add stars based on p-value
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**' 
                elif p_val < 0.05:
                    stars = '*'
                else:
                    stars = 'ns'
                    
                if t_stat > 0:  # Only show if loading dose > awake
                    # Get positions for the significance bars
                    awake_idx = labels.index('awake')
                    loading_idx = labels.index('loading dose')
                    
                    # Get min height (most negative) of boxes and whiskers
                    max_heights = []
                    for i in range(min(awake_idx, loading_idx), max(awake_idx, loading_idx) + 1):
                        max_heights.extend(box_data[i])
                    y_min = min(max_heights)  # Most negative value
                    
                    # Plot significance bar with more space above and downward-pointing tips
                    bar_height = y_min * 0.3  # Much smaller multiplier to move bar higher (closer to 0)
                    bar_tips = y_min * 0.35  # Tips extend downward (away from 0)
                    ax.plot([awake_idx + 1, loading_idx + 1], [bar_height, bar_height], 'k-')
                    ax.plot([awake_idx + 1, awake_idx + 1], [bar_height, bar_tips], 'k-')
                    ax.plot([loading_idx + 1, loading_idx + 1], [bar_height, bar_tips], 'k-')
                    
                    # Add stars below the bar
                    ax.text((awake_idx + loading_idx + 2)/2, bar_height * 0.9,  # Position below bar
                        stars, ha='center', va='top')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.show()


# def plot_session_timescales_grouped(cfg, session_lists, delase_results, locs, rocs, ropaps, plot_range=(-15, 85), top_percent=0.1, verbose=False):
#     if 'Lvr' in cfg.params.data_class or 'lever' in cfg.params.data_class:
#         # Create a 2x2 grid of subplots
#         fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
#         fig.suptitle('Average DeLASE Results by Monkey and Dose (Aligned to Infusion)', fontsize=14)

#         # Plot for each monkey and dose
#         for i, monkey in enumerate(['SPOCK', 'PEDRI']):
#             for j, dose in enumerate(['low', 'high']):
#                 ax = axs[i, j]
#                 ax.set_title(f"{monkey} - Dose {dose}", fontsize=12)
                
#                 # Collect aligned data from all sessions
#                 aligned_data = []
#                 # min_start = float('inf')
#                 # max_end = float('-inf')
                
#                 # # First pass - find common time window
#                 # for session in session_lists[monkey][dose]:
#                 #     time_vals = delase_results[session]['all'].window_start.values  # Convert to numpy array
#                 #     session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))
#                 #     infusion_time = session_file['sessionInfo']['infusionStart'][0, 0]
#                 #     aligned_times = (time_vals - infusion_time) / 60  # Convert to minutes
#                 #     min_start = min(min_start, aligned_times[0])
#                 #     max_end = max(max_end, aligned_times[-1])
                    
#                 # # Create common time grid
#                 common_times = np.arange(plot_range[0], plot_range[1], 1/60)  # 1-second intervals converted to minutes
                
#                 # Second pass - interpolate onto common grid
#                 for session in session_lists[monkey][dose]:
#                     time_vals = delase_results[session]['all'].window_start.values  # Convert to numpy array
#                     session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))
#                     infusion_time = session_file['sessionInfo']['infusionStart'][0, 0]
                    
#                     # Process stability params to get timescales
#                     def process_stability(x):
#                         # Keep only negative roots and convert to timescales
#                         timescales = -1/np.array([r for r in x if r < 0])
#                         if len(timescales) == 0:
#                             return np.nan
#                         # Take only top percentage of timescales
#                         n_keep = max(1, int(len(timescales) * top_percent))
#                         top_timescales = np.sort(timescales)[-n_keep:]
#                         return np.exp(np.mean(np.log(top_timescales)))  # geometric mean
                    
#                     # Calculate baseline (pre-infusion) geometric mean
#                     pre_infusion_mask = time_vals < infusion_time
#                     baseline_timescales = []
#                     for params in delase_results[session]['all'].stability_params[pre_infusion_mask]:
#                         timescales = -1/np.array([r for r in params if r < 0])
#                         if len(timescales) > 0:
#                             # Take only top percentage of timescales
#                             n_keep = max(1, int(len(timescales) * top_percent))
#                             top_timescales = np.sort(timescales)[-n_keep:]
#                             baseline_timescales.extend(top_timescales)
#                     baseline = np.exp(np.mean(np.log(baseline_timescales)))
                    
#                     # Calculate normalized timescales for each window
#                     stability_vals = delase_results[session]['all'].stability_params.apply(process_stability).values / baseline
                    
#                     # Align to infusion start and convert to minutes
#                     aligned_times = (time_vals - infusion_time) / 60
                    
#                     # Interpolate onto common grid
#                     interpolated = np.interp(common_times, aligned_times, stability_vals)
#                     aligned_data.append(interpolated)
                    
#                 # Convert to numpy array for easier calculations
#                 aligned_data = np.array(aligned_data)
                
#                 # Calculate geometric mean and standard error
#                 log_data = np.log(aligned_data)
#                 mean_log = np.nanmean(log_data, axis=0)
#                 sem_log = np.nanstd(log_data, axis=0) / np.sqrt(np.sum(~np.isnan(log_data), axis=0))
                
#                 mean_stability = np.exp(mean_log)
#                 upper_bound = np.exp(mean_log + sem_log)
#                 lower_bound = np.exp(mean_log - sem_log)
                
#                 # Plot geometric mean and standard error
#                 ax.plot(common_times, mean_stability, label='Geometric Mean')
#                 ax.fill_between(common_times, 
#                             lower_bound,
#                             upper_bound,
#                             alpha=0.3)
                
#                 ax.axvline(0, c='k', ls='--', label='Infusion Start')
#                 ax.axhline(1, c='k', ls=':', label='Baseline')
#                 ax.set_xlabel('Time Relative to Infusion (min)', fontsize=10)
#                 ax.set_ylabel('Normalized Timescale Ratio', fontsize=10)
#                 ax.tick_params(labelsize=9)
#                 ax.legend(fontsize=9)

#         for i, monkey in enumerate(['SPOCK', 'PEDRI']):
#             for j, dose in enumerate(['low', 'high']):
#                 ax = axs[i, j]
#                 ylim = ax.get_ylim()
#                 roc_vals = np.array(rocs[monkey][dose])/60
#                 ropap_vals = np.array(ropaps[monkey][dose])/60

#                 mean_roc = np.mean(roc_vals)
#                 mean_ropap = np.mean(ropap_vals)
#                 sem_roc = np.std(roc_vals) / np.sqrt(len(roc_vals))
#                 sem_ropap = np.std(ropap_vals) / np.sqrt(len(ropap_vals))
#                 ax.axvline(mean_roc, c='g', ls='-', label=f"ROC mean: {mean_roc:.2f} ± {sem_roc:.2f}")
#                 ax.axvline(mean_ropap, c='orange', ls='-', label=f"ROPAP mean: {mean_ropap:.2f} ± {sem_ropap:.2f}")
#                 ax.fill_betweenx(ylim, mean_roc - sem_roc, mean_roc + sem_roc, alpha=0.3, color='g')
#                 ax.fill_betweenx(ylim, mean_ropap - sem_ropap, mean_ropap + sem_ropap, alpha=0.3, color='orange')

#         plt.tight_layout()
#         plt.show()
#     else: # propofol is the data class
#         # Create figure with 1 row, 2 columns for Mary and MrJones
#         fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#         fig.suptitle('Average DeLASE Results by Monkey (Aligned to Infusion)', fontsize=14)
        
#         for i, monkey in enumerate(['Mary', 'MrJones']):
#             ax = axs[i]
#             ax.set_title(f'{monkey}', fontsize=12)
            
#             # Collect aligned data from all sessions
#             aligned_data = []
            
#             # Create fixed time window: 15 min before to 85 min after infusion start
#             # (60 min infusion + 25 min after)
#             common_times = np.arange(plot_range[0], plot_range[1], 1/60) # stepped by 1 second
            
#             # Process each session
#             for session in session_lists[monkey]:
#                 time_vals = delase_results[session]['all'].window_start.values
#                 session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f"{session}.mat"))
#                 infusion_time = session_file['sessionInfo']['drugStart'][0]
                
#                 # Process stability params to get timescales
#                 def process_stability(x):
#                     # Keep only negative roots and convert to timescales
#                     timescales = -1/np.array([r for r in x if r < 0])
#                     if len(timescales) == 0:
#                         return np.nan
#                     # Take only top percentage of timescales
#                     n_keep = max(1, int(len(timescales) * top_percent))
#                     top_timescales = np.sort(timescales)[-n_keep:]
#                     return np.exp(np.mean(np.log(top_timescales)))  # geometric mean
                
#                 # Calculate baseline (pre-infusion) geometric mean
#                 # pre_infusion_mask = time_vals < infusion_time
#                 pre_infusion_mask = (time_vals >= (infusion_time + plot_range[0]*60)) & (time_vals < infusion_time)
#                 baseline_timescales = []
#                 for params in delase_results[session]['all'].stability_params[pre_infusion_mask]:
#                     timescales = -1/np.array([r for r in params if r < 0])
#                     if len(timescales) > 0:
#                         # Take only top percentage of timescales
#                         n_keep = max(1, int(len(timescales) * top_percent))
#                         top_timescales = np.sort(timescales)[-n_keep:]
#                         baseline_timescales.extend(top_timescales)
#                 baseline = np.exp(np.mean(np.log(baseline_timescales)))
                
#                 # Calculate normalized timescales for each window
#                 stability_vals = delase_results[session]['all'].stability_params.apply(process_stability).values / baseline
                
#                 # Align to infusion start and convert to minutes
#                 aligned_times = (time_vals - infusion_time) / 60
                
#                 # Interpolate onto common grid
#                 interpolated = np.interp(common_times, aligned_times, stability_vals)
#                 aligned_data.append(interpolated)
                
#             # Convert to numpy array for easier calculations
#             aligned_data = np.array(aligned_data)
            
#             # Calculate geometric mean and standard error
#             log_data = np.log(aligned_data)
#             mean_log = np.nanmean(log_data, axis=0)
#             sem_log = np.nanstd(log_data, axis=0) / np.sqrt(np.sum(~np.isnan(log_data), axis=0))
            
#             mean_stability = np.exp(mean_log)
#             upper_bound = np.exp(mean_log + sem_log)
#             lower_bound = np.exp(mean_log - sem_log)
            
#             # Plot geometric mean and standard error
#             ax.plot(common_times, mean_stability, label='Geometric Mean')
#             ax.fill_between(common_times, 
#                         lower_bound,
#                         upper_bound,
#                         alpha=0.3)
            
#             ax.axvline(0, c='k', ls='--', label='Infusion Start')
#             ax.axhline(1, c='k', ls=':', label='Baseline')
#             ax.set_xlabel('Time Relative to Infusion (min)', fontsize=10)
#             ax.set_ylabel('Normalized Timescale Ratio', fontsize=10)
#             ax.tick_params(labelsize=9)
#             ax.legend(fontsize=9)
            
#             # Add ROC and ROPAP lines
#             ylim = ax.get_ylim()
#             roc_vals = (np.array(rocs[monkey]))/60
#             ropap_vals = (np.array(ropaps[monkey]))/60

#             mean_roc = np.mean(roc_vals)
#             # mean_ropap = np.mean(ropap_vals)
#             sem_roc = np.std(roc_vals) / np.sqrt(len(roc_vals))
#             # sem_ropap = np.std(ropap_vals) / np.sqrt(len(ropap_vals))
#             ax.axvline(mean_roc, c='g', ls='-', label=f"ROC mean: {mean_roc:.2f} ± {sem_roc:.2f}")
#             # ax.axvline(mean_ropap, c='orange', ls='-', label=f"ROPAP mean: {mean_ropap:.2f} ± {sem_ropap:.2f}")
#             ax.fill_betweenx(ylim, mean_roc - sem_roc, mean_roc + sem_roc, alpha=0.3, color='g')
#             # ax.fill_betweenx(ylim, mean_ropap - sem_ropap, mean_ropap + sem_ropap, alpha=0.3, color='orange')

#         plt.tight_layout()
#         plt.show()



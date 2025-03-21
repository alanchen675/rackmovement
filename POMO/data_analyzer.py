import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Tuple, Union

class DataAnalyzer:
    def __init__(self, epoch_filename: str, step_filename: str):
        self.epoch_filename = epoch_filename
        self.step_filename = step_filename
        self.title_size = 26
        self.label_size = 26
        self.legend_size = 26
        if self.epoch_filename:
            self.read_epoch_data()
            self.action_obj_relation = self.get_action_obj_relation()

    def read_epoch_data(self):
        epoch_data = pd.read_csv(self.epoch_filename)
        for column_name in epoch_data.columns:
            if 'reward' in column_name or 'demand' in column_name\
                    or 'move_cost' in column_name or 'action_limit' in column_name\
                    or 'selected_rts' in column_name:
                epoch_data[column_name] = epoch_data[column_name].apply(ast.literal_eval)
        self.epoch_df = epoch_data

    def read_step_data(self):
        step_data = pd.read_csv(self.step_filename)
        self.step_df = step_data

    def get_action_obj_relation(self) -> Dict[int, List[Dict[str, Tuple[float, int]]]]:
        result = defaultdict(lambda: defaultdict(list))
        #idx_heatmap = 0
        for index, row in self.epoch_df.iterrows():
            rewards = row['reward_pomo']
            actions = row['selected_rts']
            sorted_rewards = sorted(rewards, reverse=True)

            for pomo, action in enumerate(actions):
                for idx, rt in enumerate(action):
                    rank = sorted_rewards.index(rewards[pomo]) + 1
                    #result[rt]['epoch'].append(index)
                    result[rt]['reward'].append(rewards[pomo])
                    result[rt]['rt_order'].append(idx)
                    result[rt]['rew_rank'].append(rank)
        return result

    def plot_scatter(self, data: pd.DataFrame, x_col: str, y_col: str,\
                     title: str, xlabel: str, ylabel: str, save_path: str):
        plt.figure(figsize=(10, 6))
        plt.scatter(data[x_col], data[y_col], alpha=0.5)
        plt.title(title, fontsize=self.title_size)
        plt.xlabel(xlabel, fontsize=self.label_size)
        plt.ylabel(ylabel, fontsize=self.label_size)
        plt.xticks(fontsize=self.legend_size)
        plt.yticks(fontsize=self.legend_size)
        plt.grid(True)
        plt.savefig(save_path, format='png')

    def plot_box(self, data: pd.DataFrame, x_col: str, y_col: str,\
                 title: str, xlabel: str, ylabel: str, save_path: str):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=x_col, y=y_col, data=data)
        plt.title(title, fontsize=self.title_size)
        plt.xlabel(xlabel, fontsize=self.label_size)
        plt.ylabel(ylabel, fontsize=self.label_size)
        plt.xticks(fontsize=self.legend_size)
        plt.yticks(fontsize=self.legend_size)
        plt.savefig(save_path, format='png')

    def plot_heatmap(self, data: pd.DataFrame, columns_col: str, val_col: str,\
                     aggfunc: str, title: str, xlabel: str, ylabel: str, save_path: str):

        heatmap_data = data.pivot_table(columns=columns_col, values=val_col,\
                                              aggfunc=aggfunc, fill_value=0)

        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis')
        plt.title(title, fontsize=self.title_size)
        plt.xlabel(xlabel, fontsize=self.label_size)
        plt.ylabel(ylabel, fontsize=self.label_size)
        plt.xticks(fontsize=self.legend_size)
        plt.yticks(fontsize=self.legend_size)
        plt.savefig(save_path, format='png')

    def plot_heatmap_v2(self, data: pd.DataFrame, index_col: str, columns_col: str,\
                     aggfunc: str, title: str, xlabel: str, ylabel: str, save_path: str):

        heatmap_data = data.pivot_table(index=index_col, columns=columns_col,\
                                              aggfunc=aggfunc, fill_value=0)

        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='d')
        plt.title(title, fontsize=self.title_size)
        plt.xlabel(xlabel, fontsize=self.label_size)
        plt.ylabel(ylabel, fontsize=self.label_size)
        plt.xticks(fontsize=self.legend_size)
        plt.yticks(fontsize=self.legend_size)
        plt.savefig(save_path, format='png')

    def plot_bar_graph(self, data: pd.DataFrame, x_col: str, y_col: str,\
                 title: str, xlabel: str, ylabel: str, save_path: str):
        # Filter rows where rew_rank <= 3
        filtered_df = data[data['rew_rank'] <= 3]
        # Create an empty DataFrame to store the counts for rt_order (0-9) for each rew_rank
        all_rt_order = pd.Series(range(10), name='rt_order')
        counts_per_rank = pd.DataFrame()

        # Count for each rew_rank (1, 2, 3) and store it
        for rank in [1, 2, 3]:
            rank_counts = filtered_df[filtered_df['rew_rank'] == rank]['rt_order'].value_counts()
            rank_counts = rank_counts.reindex(all_rt_order, fill_value=0)
            counts_per_rank[f'rank_{rank}'] = rank_counts
        # Plotting the results
        plt.figure(figsize=(8, 6))
        counts_per_rank.plot(kind='bar', width=0.8)
        plt.title(title, fontsize=self.title_size)
        plt.xlabel(xlabel, fontsize=self.label_size)
        plt.ylabel(ylabel, fontsize=self.label_size)
        plt.xticks(rotation=0, fontsize=self.legend_size)  # Ensure x-axis labels are horizontal
        plt.yticks(fontsize=self.legend_size)
        plt.legend(title='Reward Rank', fontsize=self.legend_size)
        plt.tight_layout()
        plt.savefig(save_path, format='png')

    def parse_training_log(self, file_path: str):
        # Initialize empty dictionaries to store the data
        epoch_dict = {}
        scores = []
        losses = []
        # Define regular expressions to match lines with Epoch, Score, and Loss
        epoch_pattern = re.compile(r'Epoch\s+(\d+)')
        score_pattern = re.compile(r'Score:\s+([+-]?([0-9]*[.])?[0-9]+)')
        loss_pattern = re.compile(r'Loss:\s+([+-]?([0-9]*[.])?[0-9]+)')
        # Open and read the log file line by line
        with open(file_path, 'r') as file:
            skip_count = 10
            for line in file:
                # Search for epoch matches
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    if epoch not in epoch_dict:
                        if skip_count>0:
                            skip_count -= 1
                            continue
                        epoch_dict[epoch] = {'score': None, 'loss': None}
                # Search for score matches
                score_match = score_pattern.search(line)
                if score_match:
                    score = float(score_match.group(1))
                    if epoch is not None and epoch_dict[epoch]['score'] is None:
                        epoch_dict[epoch]['score'] = score
                # Search for loss matches
                loss_match = loss_pattern.search(line)
                if loss_match:
                    loss = float(loss_match.group(1))
                    if epoch is not None and epoch_dict[epoch]['loss'] is None:
                        epoch_dict[epoch]['loss'] = loss
        # Extract epochs, scores, and losses from the dictionary
        epochs = sorted(epoch_dict.keys())
        scores = [epoch_dict[epoch]['score'] for epoch in epochs]
        losses = [epoch_dict[epoch]['loss'] for epoch in epochs]
        return epochs, scores, losses

    def moving_average(self, data, window_size: int):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def _plot_training_log(self, epochs, scores, losses, score_filename, loss_filename, window_size=5):
        # Calculate moving averages
        scores_ma = self.moving_average(scores, window_size)
        losses_ma = self.moving_average(losses, window_size)
        # Adjust epochs for the moving average plot
        ma_epochs = epochs[window_size-1:]
        # Plot Scores
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, scores, marker='o', color='b', label='Score')
        plt.plot(ma_epochs, scores_ma, marker='x', color='orange', label='Moving Average (Score)')
        plt.xlabel('Epochs', fontsize=self.label_size)
        plt.ylabel('Scores', fontsize=self.label_size)
        plt.title('Scores vs Epochs', fontsize=self.title_size)
        plt.legend(fontsize=self.legend_size)
        plt.xticks(fontsize=self.legend_size)
        plt.yticks(fontsize=self.legend_size)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./{score_filename}')  # Save the scores plot as an image
        plt.close()
        # Plot Losses
        plt.figure(figsize=(7, 5))  # Create a new figure for losses
        plt.plot(epochs, losses, marker='o', color='r', label='Loss')
        plt.plot(ma_epochs, losses_ma, marker='x', color='green', label='Moving Average (Loss)')
        plt.xlabel('Epochs', fontsize=self.label_size)
        plt.ylabel('Losses', fontsize=self.label_size)
        plt.title('Losses vs Epochs', fontsize=self.title_size)
        plt.legend(fontsize=self.legend_size)
        plt.xticks(fontsize=self.legend_size)
        plt.yticks(fontsize=self.legend_size)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./{loss_filename}')  # Save the losses plot as an image
        plt.close()  # Close the figure to avoid overlap

    def plot_training_log(self):
        # Example usage
        file_path = 'data/train_data/a4_b64-8_v2heur_new_reslimit_show_orig_loss.o8099903'
        epochs, scores, losses = self.parse_training_log(file_path)
        print('Epochs:', epochs)
        print('Scores:', scores)
        print('Losses:', losses)
        moving_avgs = [5, 10, 15, 20, 25, 30, 35, 40]
        # Plot the parsed data
        for mv in moving_avgs:
            score_filename = f'fig/train/score_plot_mavg_{mv}.png'
            loss_filename = f'fig/train/loss_plot_mavg_{mv}.png'
            self._plot_training_log(epochs, scores, losses, score_filename, loss_filename, mv)

    # Function to handle status and plot comparison
    def plot_comparison(self, pomo_df, heur_df, mip_df, filename, metric):
        if metric == 'reward2':
            metric_name = 'Resource Spread'
        else:
            metric_name = 'Movement Cost'
        # Group by 'batch' in the first dataset and get the first row for each 'batch'
        pomo_objs = pomo_df.loc[pomo_df.groupby('batch')['obj'].idxmin()]
        heur_objs = heur_df.groupby('batch').first().reset_index()
        # For the second dataset, replace 'obj' with 'reward1' if status is 3
        mip_df['obj'] = np.where(mip_df['status'] == 3, mip_df['reward1'], mip_df['obj'])
        # Plot the first dataset's metric
        plt.figure(figsize=(10, 8))
        plt.plot(pomo_objs['batch'], pomo_objs[metric], marker='o', linestyle='-', color='b', label='Proposed alg')
        plt.plot(heur_objs['batch'], heur_objs[metric], marker='d', linestyle='-', color='g', label='Greedy heuristics')
        # Plot the second dataset's metric, with modified values for status 3
        plt.plot(range(len(mip_df)), mip_df[metric], marker='x', linestyle='--', color='r', label='Gurobi')
        # Labels and legend
        plt.xlabel('Problem Instances', fontsize=self.label_size)
        plt.ylabel(f'{metric_name}', fontsize=self.label_size)
        plt.title(f'Comparison of {metric_name}', fontsize=self.title_size)
        plt.legend(fontsize=self.legend_size)
        plt.xticks(fontsize=self.legend_size)
        plt.yticks(fontsize=self.legend_size)
        plt.grid(True)
        plt.savefig(f'./{filename}')

    # Function to handle status and plot comparison
    def plot_comparison_broken_axis(self, pomo_df, heur_df, mip_df, filename, metric, ylim):
        if metric not in {'obj', 'penalty'}:
            self.plot_comparison(pomo_df, heur_df, mip_df, filename, metric)
            return
        elif metric == 'obj':
            metric_name = 'Objective Value'
        else:
            metric_name = 'Penalty'
        ylim1, ylim2 = ylim
        # Group by 'batch' in the first dataset and get the first row for each 'batch'
        pomo_objs = pomo_df.loc[pomo_df.groupby('batch')['obj'].idxmin()]
        heur_objs = heur_df.groupby('batch').first().reset_index()
        # For the second dataset, replace 'obj' with 'reward1' if status is 3
        mip_df['obj'] = np.where(mip_df['status'] == 3, mip_df['reward1'], mip_df['obj'])

        plt.plot(pomo_objs['batch'], pomo_objs[metric], marker='o', linestyle='-', color='b', label='Proposed alg')
        plt.plot(heur_objs['batch'], heur_objs[metric], marker='d', linestyle='-', color='g', label='Greedy heuristics')
        # Plot the second dataset's metric, with modified values for status 3
        plt.plot(range(len(mip_df)), mip_df[metric], marker='x', linestyle='--', color='r', label='Gurobi')

        # Create subplots with shared x-axis but different y-limits
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
        # Plot for the upper part (obj and reward1 in large range)
        ax1.plot(pomo_objs['batch'], pomo_objs[metric], marker='o', linestyle='-', color='b', label='Proposed alg')
        ax1.plot(heur_objs['batch'], heur_objs[metric], marker='d', linestyle='-', color='g', label='Greedy heuristics')
        ax1.plot(range(len(mip_df)), mip_df[metric], marker='x', linestyle='--', color='r', label='Gurobi')
        ax1.set_ylim(ylim2)  # Set the upper limit first
        ax1.set_yscale('log')
        ax1.legend(fontsize=self.legend_size)
        ax1.grid(True)
        # Plot for the lower part (reward1 values that are very small)
        ax2.plot(pomo_objs['batch'], pomo_objs[metric], marker='o', linestyle='-', color='b', label='Proposed alg')
        ax2.plot(heur_objs['batch'], heur_objs[metric], marker='d', linestyle='-', color='g', label='Greedy heuristics')
        ax2.plot(range(len(mip_df)), mip_df[metric], marker='x', linestyle='--', color='r', label='Gurobi')
        ax2.set_ylim(ylim1)  # Set the lower limit next
        ax2.grid(True)
        # Adding breaks to the y-axis
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.tick_params(labeltop=False)  # Don't put tick labels at the top of ax1
        ax2.xaxis.tick_bottom()

        # Adding diagonal lines to indicate the break
        d = .015  # How big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # Switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

        # Set a shared ylabel using fig.text
        fig.text(0.02, 0.5, f'{metric_name}', va='center', rotation='vertical', fontsize=self.label_size)
        # Adjust tick size for both subplots (ax1 and ax2)
        ax1.tick_params(axis='both', which='major', labelsize=self.label_size)  # Adjust tick size for the upper subplot
        ax2.tick_params(axis='both', which='major', labelsize=self.label_size)  # Adjust tick size for the lower subplot

        # Labels and title
        ax2.set_xlabel('Problem Instances', fontsize=self.label_size)
        #ax1.set_ylabel('Objective Value (obj)')
        #ax2.set_ylabel('Objective Value (obj)')
        plt.suptitle(f'Comparison of {metric_name}', y=0.92, fontsize=self.title_size)

        plt.savefig(f'./{filename}')

    def plot_comparison_boxplot(self, pomo_df, heur_df, mip_df, filename, metric):
        if metric == 'reward2':
            metric_name = 'Resource Spread'
        elif metric == 'move_cost':
            metric_name = 'Movement Cost'
        elif metric == 'obj':
            metric_name = 'Objective Value'
        else:
            metric_name = 'Penalty'
        # Group by 'batch' in the first dataset and get the first row for each 'batch'
        pomo_objs = pomo_df.loc[pomo_df.groupby('batch')['obj'].idxmin()]
        heur_objs = heur_df.groupby('batch').first().reset_index()
        # For the MIP dataset, replace 'obj' with 'reward1' if status is 3
        mip_df['obj'] = np.where(mip_df['status'] == 3, mip_df['reward1'], mip_df['obj'])

        # Prepare data for the box plot
        pomo_data = pomo_objs[metric]
        heur_data = heur_objs[metric]
        mip_data = mip_df[metric]

        # Create a list of the three datasets for the box plot
        data_to_plot = [pomo_data, heur_data, mip_data]

        # Plot the box plot
        plt.figure(figsize=(10, 8))
        plt.boxplot(data_to_plot, patch_artist=True, tick_labels=['Proposed alg', 'Greedy heuristics', 'Gurobi'])
        # Set a logarithmic scale on the y-axis
        plt.yscale('log')

        # Add labels and title
        plt.xlabel('Algorithm', fontsize=self.label_size)
        plt.ylabel(f'{metric_name}', fontsize=self.label_size)
        plt.title(f'Comparison of {metric_name}', fontsize=self.title_size)
        plt.xticks(fontsize=self.legend_size)
        plt.yticks(fontsize=self.legend_size)
        plt.grid(True)

        # Save the plot
        plt.savefig(f'./{filename}')

if __name__=='__main__':
    compare_filenames = {
        'pomo': 'data/compare/pomo/perf_pomo.csv',
        'heur': 'data/compare/heur/perf_heur.csv',
        'mips': 'data/compare/mips/perf_mip.csv',
    }
    pomo_df = pd.read_csv(compare_filenames['pomo'])
    heur_df = pd.read_csv(compare_filenames['heur'])
    mip_df = pd.read_csv(compare_filenames['mips'])
    pomo_df['penalty'] = pomo_df['reward1']+pomo_df['reward3']+pomo_df['reward4']
    heur_df['penalty'] = heur_df['reward1']+heur_df['reward3']+heur_df['reward4']
    mip_df['penalty'] = mip_df['reward1']+mip_df['reward3']+mip_df['reward4']
    mip_df['reward2'] *= 100
    da = DataAnalyzer('data/epoch_data/probs_in_epoch_50.csv', '')
    for rt in range(10):
        rt_df = pd.DataFrame(da.action_obj_relation[rt])
        da.plot_scatter(rt_df, 'rt_order','rew_rank', f'Rack Type {rt} Order vs Performance',\
                        'Rack Type Order', 'Objective Rank', f'./fig/rank/rt{rt}/reward_rank_scatter_vs_rt_order.png')
        da.plot_box(rt_df, 'rt_order', 'rew_rank', f'Rack Type {rt} Order vs Performance',\
                    'Rack Type Order', 'Objective Rank', f'./fig/rank/rt{rt}/reward_rank_boxplot_vs_rt_order.png')
        da.plot_heatmap(rt_df, 'rt_order', 'rew_rank', 'mean',\
                        f'Rack Type {rt} Order vs Performance', 'Rack Type Order', 'Objective Rank',\
                        f'./fig/rank/rt{rt}/reward_rank_heatmap_vs_rt_order.png')
        da.plot_heatmap_v2(rt_df, 'rew_rank', 'rt_order', 'size',\
                        f'Rack Type {rt} Order vs Performance', 'Rack Type Order', 'Objective Rank',\
                        f'./fig/rank/rt{rt}/reward_rank_count_heatmap_vs_rt_order.png')
        da.plot_bar_graph(rt_df, 'rt_order', 'rew_rank', f'Rack Type {rt} Top Three Results Count',\
                    'Rack Type Order', 'Number of Instances', f'./fig/rank/rt{rt}/reward_rank_barplot_top3_vs_rt_order.png')
    da.plot_training_log()

    ylim_dict = {
        'obj': [(500, 1500), (2000, 35000)],
        'move_cost': [(10, 10), (10, 10)],
        'reward2': [(10, 10), (10, 10)],
        'penalty': [(0, 200), (300, 35000)]
    }
    for metric in ['obj', 'move_cost', 'reward2', 'penalty']:
        filename = f'fig/compare/{metric}/comparison.png'
        filename_bp = f'fig/compare/{metric}/comparison_bp.png'
        da.plot_comparison_broken_axis(pomo_df, heur_df, mip_df, filename, metric, ylim_dict[metric])
        da.plot_comparison_boxplot(pomo_df, heur_df, mip_df, filename_bp, metric)

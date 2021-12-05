"""
this one program will be used to basically generate all REDQ related figures for ICLR 2021.
NOTE: currently only works with seaborn 0.8.1 the tsplot function is deprecated in the newer version
the plotting function is originally based on the plot function in OpenAI spinningup
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json

def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    the "condition" here can be a string, when plotting, can be used as label on the legend
    """
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
                exp_name = None
                try:
                    config_path = open(os.path.join(root, 'config.json'))
                    config = json.load(config_path)
                    if 'exp_name' in config:
                        exp_name = config['exp_name']
                except:
                    print('No file named config.json')
                condition1 = condition or exp_name or 'exp'
                print(os.path.join(root, 'progress.txt'))

                # exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
                performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
                exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
                if performance in exp_data:
                    exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
                datasets.append(exp_data)
            except Exception as e:
                print(e)
    return datasets

def do_smooth(x, smooth):
    y = np.ones(smooth)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x

def seeds_to_x_y(list_of_seeds, value, xvalue='TotalEnvInteracts', smooth=1, cut_off=-1, skip=1,):
    # expect each entry in list of seeds to be a dictionary or pandas dataframe
    # returns x, y that can be directly used in seaborn lineplot
    xs = []
    ys = []
    for seed in list_of_seeds:
        x = seed[xvalue].to_numpy()
        y = seed[value].to_numpy()
        smoothed_y = do_smooth(y.reshape(-1)[::skip], smooth)
        xs.append(x)
        ys.append(smoothed_y)
    if len(ys) > 1:
        ys = np.concatenate(ys, 0)
        xs = np.concatenate(xs, 0)
    return xs, ys

def plot_one_curve(data_seeds, value='Performance', color=None, linestyle=None, label=None, smooth=1, n_boot=50):
    x, y = seeds_to_x_y(data_seeds, value, smooth=smooth)
    ax = sns.lineplot(x=x, y=y, n_boot=n_boot, label=label, color=color, linestyle=linestyle)

def plot_figure(list_of_data_seeds, label_list=None, value='Performance', smooth=1, save_loc=None, xlabel=None, ylabel=None):
    # plot a figure, which will have a few curves
    for i, data_seeds in enumerate(list_of_data_seeds):
        label = label_list[i] if label_list is not None else None
        plot_one_curve(data_seeds, value, smooth=smooth, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if save_loc is not None:
        fig = plt.gcf()
        fig.savefig(save_loc)
        plt.close(fig)
    else:
        plt.show()

# the path leading to where the experiment file are located
base_path = '../data/1-1-1'

# for each figure, students will specify the data and label for each variant.
env_names = ['Hopper-v2', 'Ant-v2']
value_list = ['Performance', 'AverageNormQBias', 'StdNormQBias']
value2ylabel = {
'Performance':'Performance',
'AverageNormQBias':'Average normalized Q bias',
'StdNormQBias':'Std of normalized Q bias',
}

polyak_list = [0, 0.9, 0.995]
variants = ['sac_poly0', 'sac_poly0.9', 'sac_poly0.995']
for env in env_names:
    for value in value_list:
        list_of_data_seeds = []
        label_list = variants
        for variant in variants:
            variant_data_path = os.path.join(base_path, variant + '_' + env)
            data_list = get_datasets(variant_data_path)
            list_of_data_seeds.append(data_list)

        try:
            os.mkdir('../figures/1-1-1/')
        except:
            pass
        save_loc = '../figures/1-1-1/sac_polyak_%s_%s.png' % (env, value)
        plot_figure(list_of_data_seeds, label_list=label_list, value=value, smooth=1, save_loc=save_loc,
                    xlabel='Environment Interaction', ylabel=value2ylabel[value])

from pathlib import Path
from pprint import pprint
import os

import numpy as np
import pandas as pd
import tree
import matplotlib.pyplot as plt
import seaborn as sns

from experiments import load_results
from .create_tables import (
    tasks,
    TASK_TO_INDEX_MAP,
    METHODS_TO_REPORT,
    TASK_METHOD_TO_LABEL_MAP)


# TASK_TO_INDEX_MAP = {
#     'boyan': 'Boyan Chain',
#     'baird': 'Baird Star',
#     'disc_random_on': '400-State MDP (On-pol.)',
#     'disc_random_off': '400-State MDP (Off-pol.)',
#     'lqr_imp_onpolicy': 'Cart-Pole (On-pol., Imp. Feat.)',
#     'lqr_imp_offpolicy': 'Cart-Pole (Off-pol., Imp. Feat.)',
#     'lqr_full_onpolicy': 'Cart-Pole (On-pol., Perf. Feat.)',
#     'lqr_full_offpolicy': 'Cart-Pole (Off-pol., Perf. Feat.)',
#     # '': '9. Cart-Pole Swingup On-policy',
#     # '': '10. Cart-Pole Swingup Off-policy',
#     'link20_imp_onpolicy': '20-link Pole (On-pol.)',
#     'link20_imp_offpolicy': '20-link Pole (Off-pol.)',
# }


def plot_task(task_dataframe, metric_to_use, axis, labels, legend=False):
    assert set(labels) == set(task_dataframe['method'].unique())

    X_UNITS = 'thousands'
    # if X_UNITS == 'thousands':
    #     task_dataframe['training_steps'] /= 1000
    # else:
    #     raise ValueError

    # breakpoint()

    sns.lineplot(
        x='training_steps',
        y=f'{metric_to_use}-mean',
        # ci=f'{metric_to_use}-std',
        # ci=task_dataframe[f'{metric_to_use}-std'],
        hue='method',
        hue_order=labels,
        data=task_dataframe,
        legend=legend,
        ax=axis)

    if legend:
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles=(handles[-1], *handles[1:-1]),
                    labels=(labels[-1], *labels[1:-1]))

    # axis.grid(linestyle="--", linewidth=0.1, color='.25', zorder=-10)

    # axis.legend().set_title('')
    # legend_1 = axis.legend(
    #     loc='center right',
    #     bbox_to_anchor=(-0.15, 0.5),
    #     ncol=1,
    #     borderaxespad=0.0)
    # legend_2 = axis.legend(
    #     loc='best',
    #     # bbox_to_anchor=(1.05, 0.5),
    #     # ncol=1,
    #     # borderaxespad=0.0,
    # )

    # axis.set_yscale('symlog')
    axis.set_ylim(np.clip(axis.get_ylim(), 0.0, float('inf')))

    # if X_UNITS == 'thousands':
    #     axis.set_xlabel('Training Steps [$10^3$]')
    # else:
    #     axis.set_xlabel('Training Steps')


def plot(dataframes_by_task):
    metric_to_use = 'RMSE'

    def validate_dataframe(dataframe):
        if (dataframe[f'{metric_to_use}-mean'].hasnans
            or dataframe[f'{metric_to_use}-std'].hasnans):
            raise ValueError("Found nans from the visualization dataframes.")

    tree.map_structure(validate_dataframe, dataframes_by_task)

    task_ids = tuple(dataframes_by_task.keys())

    num_subplots = len(task_ids)
    default_figsize = np.array(plt.rcParams.get('figure.figsize'))
    figure_scale = 0.55

    num_subplots_per_side = int(np.ceil(np.sqrt(num_subplots)))

    figsize = np.array((
        num_subplots_per_side, 0.75 * num_subplots_per_side
    )) * np.max(default_figsize * figure_scale)
    figure, axes = plt.subplots(
        num_subplots_per_side, num_subplots_per_side, figsize=figsize)
    axes = np.atleast_1d(axes)

    save_dir = Path('/tmp/bbo/').expanduser()
    os.makedirs(save_dir, exist_ok=True)

    all_labels = set(
        method_name for dataframe in dataframes_by_task.values()
        for method_name in dataframe['method'].unique().tolist())
    assert all_labels == {'BRM', 'GTD2', 'LSTD', 'TD', 'BBO', 'TDC'}, all_labels

    all_labels = list(sorted(all_labels))
    bbo_label_index = all_labels.index('BBO')
    bbo_label = all_labels.pop(bbo_label_index)
    all_labels = [*all_labels, bbo_label]

    titles = []
    for ((task_id, task_dataframe), ((i, j), axis)) in zip(
            dataframes_by_task.items(), np.ndenumerate(axes)):
        plot_task(
            task_dataframe,
            metric_to_use,
            axis,
            all_labels,
            legend='brief')

        handles, labels = axis.get_legend_handles_labels()
        labels.pop(labels.index('method'))
        assert labels == all_labels, (labels, all_labels)

        axis.get_legend().remove()

        title = axis.set_title(TASK_TO_INDEX_MAP[task_id], fontsize='medium')
        titles.append(title)

        if i == axes.shape[0] - 1:
            axis.set_xlabel(
                'Training Steps',
                labelpad=10,
                fontsize='large',
                # fontweight='bold',
            )
        else:
            axis.set(xlabel=None)

        if j == 0:
            if metric_to_use == 'RMSE':
                axis.set_ylabel(
                    '$\sqrt{MSE}$',
                    labelpad=10,
                    fontsize='large',
                    # fontweight='bold',
                )
            elif metric_to_use == 'MSE':
                axis.set(ylabel='$MSE$')
            else:
                raise ValueError(metric_to_use)
        else:
            axis.set(ylabel=None)

    handles, labels = axes[-1][-1].get_legend_handles_labels()
    legend = figure.legend(
        handles=(handles[-1], *handles[1:-1]),
        labels=(labels[-1], *labels[1:-1]),
        ncol=num_subplots,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        fontsize='large')

    legend.set_in_layout(True)

    plt.tight_layout()
    plt.savefig(
        # os.path.join(result._experiment_dir, 'result.pdf'),
        save_dir / 'result.pdf',
        # bbox_extra_artists=(legend_1, legend_2),
        # bbox_extra_artists=(legend_2, ),
        bbox_extra_artists=(*titles, legend),
        bbox_inches='tight')

    plt.savefig(
        save_dir / 'result.png',
        # os.path.join(result._experiment_dir, 'result.png'),
        # bbox_extra_artists=(legend_1, legend_2),
        # bbox_extra_artists=(legend_2, ),
        bbox_extra_artists=(*titles, legend),
        bbox_inches='tight')


def main():
    all_task_dataframes = {}

    for task in tasks:
        data = load_results(task)
        criteria = data['criteria']

        task_dataframes = {}

        for method, mean, std in zip(data['methods'], data['mean'], data['std']):
            method_label = TASK_METHOD_TO_LABEL_MAP[task].get(method.name, None)
            if method_label is None:
                continue

            # if task != 'link20_imp_onpolicy':
            #     continue
            # method_label = method.name

            if method_label in task_dataframes:
                if method.name != 'TDC(0.0) $\\alpha$=0.002 $\\mu$=0.0001':
                    raise ValueError(task, method_label)

            if (task == 'lqr_imp_offpolicy'
                and method_label == 'TDC(0.0) $\\alpha$=0.002 $\\mu$=0.0001'
                and type(method).__name__ == 'GeriTDCLambda'):
                continue

            # method_mean = mean[rmse_index]
            # method_std = std[rmse_index]

            if data['episodic']:
                training_steps = np.arange(0, data['n_eps'] * data['l'], data['l']) + data['l']
            else:
                training_steps = (
                    np.arange(0, data['n_eps'] * data['l'], data['error_every'])
                    + data['error_every'])

            task_dataframes[method_label] = pd.DataFrame({
                'task': task,
                'method': method_label,
                'training_steps': training_steps,
                **{
                    f'{criterion}-mean': mean[i]
                    for i, criterion in enumerate(criteria)
                },
                **{
                    f'{criterion}-std': std[i]
                    for i, criterion in enumerate(criteria)
                },
            })

        all_task_dataframes[task] = pd.concat(tree.flatten(task_dataframes))

    plot(all_task_dataframes)


if __name__ == '__main__':
    main()

from pathlib import Path
from pprint import pprint
import os

import numpy as np
import pandas as pd
import tree
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
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


FIGURE_TYPE = 'full'


def plot_task(task_dataframe, metric_to_use, axis, labels, legend=False):
    try:
        assert set(labels) == set(task_dataframe['method'].unique())
    except Exception as e:
        breakpoint()
        pass

    X_UNITS = 'thousands'
    if X_UNITS == 'thousands':
        task_dataframe['training_steps'] /= 1000
    else:
        raise ValueError

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

    x = task_dataframe[f'{metric_to_use}-mean'][task_dataframe[f'{metric_to_use}-mean'].index == 0].median()
    # axis.set_ylim(np.clip(axis.get_ylim(), 0.0, float('inf')))
    axis.set_ylim(np.clip(axis.get_ylim(), 0.0, 1.05 * 10 ** np.round(np.log10(x))))

    if X_UNITS == 'thousands':
        axis.set_xlabel('Training Steps [$10^3$]')
    else:
        axis.set_xlabel('Training Steps')


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
    figure_scale = 0.51

    if FIGURE_TYPE == 'small':
        subplots_shape = (1, num_subplots)

        default_figsize = plt.rcParams.get('figure.figsize')
        # figsize = np.array((num_subplots, 0.4)) * np.max(default_figsize[0] * figure_scale)
        figsize = np.array((
            0.9 * num_subplots, 0.4)) * np.max(default_figsize[0] * figure_scale)

        # figsize = np.array((num_subplots, 0.5)) * np.max(default_figsize[0] * figure_scale)
        # figsize = (
        #     np.array((0.88, 0.6))
        #     # np.array((0.75, 0.55))
        #     * np.array(subplots_shape[::-1])
        #     # * np.array((3, 0.4))
        #     * np.max(default_figsize[0] * figure_scale))

        # figsize = (
        #     np.array((1, 0.4))
        #     # np.array((0.75, 0.55))
        #     * np.array(subplots_shape[::-1])
        #     # * np.array((3, 0.4))
        #     * np.max(default_figsize[0] * figure_scale))
    else:
        num_subplots_per_side = int(np.ceil(np.sqrt(num_subplots)))
        subplots_shape = (num_subplots_per_side, num_subplots_per_side)

        figsize = (
            np.array((1, 0.5))
            # np.array((0.75, 0.55))
            * np.array(subplots_shape[::-1])
            # * np.array((3, 0.4))
            * np.max(default_figsize[0] * figure_scale))

    figure, axes = plt.subplots(
        *subplots_shape,
        figsize=figsize,
        # constrained_layout=True,
        # gridspec_kw={
        #     'wspace': 0,
        #     # 'hspace': 0
        # },
    )

    axes = np.atleast_2d(axes)

    save_dir = Path('/tmp/bbo/').expanduser()
    os.makedirs(save_dir, exist_ok=True)

    all_labels = set(
        method_name for dataframe in dataframes_by_task.values()
        for method_name in dataframe['method'].unique().tolist())
    # expected_labels = {'BRM', 'GTD2', 'LSTD', 'TD', 'BBO', 'TDC'}
    # expected_labels = {'BRM', 'BBO', 'LSTD (w/ reg.)', 'TD', 'GTD2', 'LSTD (w/o reg.)', 'TDC'}
    expected_labels = set(METHODS_TO_REPORT)
    assert all_labels == expected_labels, all_labels

    all_labels = list(sorted(all_labels))[::-1]
    if 'BBO' in all_labels:
        bbo_label_index = all_labels.index('BBO')
        bbo_label = all_labels.pop(bbo_label_index)
        # all_labels = [bbo_label, *all_labels]
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

        title_text = (TASK_TO_INDEX_MAP[task_id]
                      .replace('On-pol.', '$\\oplus$')
                      .replace('Off-pol.', '$\\ominus$')
                      .replace('Perf. Feat.', '$\\dag$')
                      .replace('Imp. Feat.', '$\\ddag$')
                      .replace(', ', ','))

        if FIGURE_TYPE == 'small':
            title_text = title_text.split(' (')[0]

        title = axis.set_title(
            title_text,
            # pad=30,
            # fontsize='medium',
        )
        titles.append(title)

        if i == axes.shape[0] - 1:
            axis.set_xlabel(
                'Training Steps [$10^3$]',
                labelpad=10,
                # fontsize='large',
                # fontweight='bold',
            )
        else:
            axis.set(xlabel=None)

        if j == 0:
            if metric_to_use == 'RMSE':
                axis.set_ylabel(
                    '$\sqrt{MSE}$',
                    # labelpad=10,
                    # fontsize='large',
                    # fontweight='bold',
                )
            elif metric_to_use == 'MSE':
                axis.set(ylabel='$MSE$')
            else:
                raise ValueError(metric_to_use)
        else:
            axis.set(ylabel=None)

    # for tick in axes.flatten()[-1].get_yticklabels():
    #     tick.set_rotation(70)

    def round_tick_formatter(x, pos):
        if x == int(x):
            return str(int(x))
        else:
            return f"{x:.1f}"

    formatter = FuncFormatter(round_tick_formatter)
    for axis in axes.flatten():
        axis.yaxis.set_major_formatter(formatter)

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    # Drop the "title" label and move BBO to front
    handles = (handles[-1], *handles[1:-1])
    labels = (labels[-1], *labels[1:-1])

    if FIGURE_TYPE == 'small':
        extra_legend = None
    else:
        extra_legend_handles = (
            Line2D(
                [],
                [],
                marker=r'$\oplus$',
                linestyle='none',
                color='black',
                label='on-policy',
                # markerfacecolor='black',
                markersize=12,
                markeredgewidth=0.0,
                # markersize='medium',
            ),
            Line2D(
                [],
                [],
                marker=r'$\ominus$',
                linestyle='none',
                color='black',
                label='off-policy',
                # markerfacecolor='black',
                markersize=12,
                markeredgewidth=0.0,
                # markersize='medium',
            ),
            Line2D(
                [],
                [],
                marker=r'$\dag$',
                linestyle='none',
                color='black',
                label='perfect features',
                # markerfacecolor='black',
                markersize=12,
                markeredgewidth=0.0,
                # markersize='medium',
            ),
            Line2D(
                [],
                [],
                marker=r'$\ddag$',
                linestyle='none',
                color='black',
                label='impoverished features',
                # markerfacecolor='black',
                markersize=12,
                markeredgewidth=0.0,
                # markersize='medium',
            ),
        )
        handles = (
            *handles,
        )
        labels = (
            *labels,
        )
        extra_legend = plt.legend(
            handles=extra_legend_handles,
            ncol=4,

            handlelength=1.25,
            handletextpad=0.25,
            # labelspacing=0,
            columnspacing=1.25,
            # loc='best',

            loc='lower center',
            bbox_to_anchor=(0.5, 1.0),
            bbox_transform=figure.transFigure,

            # loc='lower center',
            # bbox_to_anchor=(0.5, 1.0),
            # fontsize='large'
            fontsize='medium'
        )
        extra_legend.set_in_layout(True)
        figure.add_artist(extra_legend)

    legend = figure.legend(
        handles=handles,
        labels=labels,
        # handles=handles[1:],
        # labels=labels[1:],
        ncol=8,
        handlelength=1.25,
        handletextpad=0.25,
        # labelspacing=0,
        columnspacing=1.25,
        # loc='best',

        loc='lower center',
        bbox_to_anchor=(
            (0.475, 1.03)
            if FIGURE_TYPE == 'small'
            else (0.5, 1.05)
        ),
        bbox_transform=figure.transFigure,

        # loc='lower center',
        # bbox_to_anchor=(0.5, 1.0),
        # fontsize='large'
        fontsize='medium'
    )

    legend.set_in_layout(True)

    bbox_extra_artists = (*titles, legend, extra_legend)
    bbox_extra_artists = type(bbox_extra_artists)(
        x for x in bbox_extra_artists if x is not None)

    figure.canvas.draw()
    # plt.tight_layout()
    # figure.set_constrained_layout(False)
    plt.savefig(
        # os.path.join(result._experiment_dir, 'result.pdf'),
        save_dir / 'result.pdf',
        # bbox_extra_artists=(legend_1, legend_2),
        # bbox_extra_artists=(legend_2, ),

        bbox_extra_artists=bbox_extra_artists,
        # bbox_extra_artists=(legend, extra_legend),
        # bbox_extra_artists=(*titles, legend,),
        # bbox_extra_artists=(*titles, ),

        bbox_inches='tight'
    )

    plt.savefig(
        save_dir / 'result.png',
        # os.path.join(result._experiment_dir, 'result.png'),
        # bbox_extra_artists=(legend_1, legend_2),
        # bbox_extra_artists=(legend_2, ),

        bbox_extra_artists=bbox_extra_artists,
        # bbox_extra_artists=(legend, extra_legend),
        # bbox_extra_artists=(*titles, legend),
        # bbox_extra_artists=(*titles, ),

        bbox_inches='tight'
    )


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

            if method_label not in METHODS_TO_REPORT:
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

        try:
            all_task_dataframes[task] = pd.concat(tree.flatten(task_dataframes))
        except Exception as e:
            breakpoint()
            pass

    plot(all_task_dataframes)


if __name__ == '__main__':
    main()

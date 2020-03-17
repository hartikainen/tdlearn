from __future__ import division

# -*- coding: utf-8 -*-
"""
Generic code for running policy evaluation experiments from scripts
and plotting their results.
"""
__author__ = "Christoph Dann <cdann@cdann.de>"
import argparse
import pickle
from pprint import pprint
import numpy as np
import os
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from .utils import set_gpu_memory_growth

plt.ion()

exp_list = ["boyan", "baird",
            "disc_random_on", "disc_random_off",
            "lqr_imp_onpolicy", "lqr_imp_offpolicy",
            "lqr_full_onpolicy", "lqr_full_offpolicy", "swingup_gauss_onpolicy",
            "swingup_gauss_offpolicy", "link20_imp_onpolicy",
            "link20_imp_offpolicy"]


def load_result_file(fn, maxerr=5):
    with open(fn) as f:
        d = pickle.load(f)
    for i in range(d["res"].shape[-1]):
        print d["criteria"][i], np.nanmin(d["res"][..., i])
        best = d["params"][np.nanargmin(d["res"][..., i])]
        for n, v in zip(d["param_names"], best):
            print n, v
    return d


def plot_experiment(experiment, criterion):
    d = load_results(experiment)
    plot_errorbar(ncol=3, criterion=criterion, **d)


def plot_2d_error_grid_experiments(experiments, method, criterion, **kwargs):
    l = []
    for e in experiments:
        fn = "data/{e}/{m}.pck".format(e=e, m=method)
        l.append(plot_2d_error_grid_file(
            fn, criterion, title="{m} {e}".format(m=method, e=e),
            **kwargs))
    return l if len(l) > 1 else l


def plot_2d_error_grid_experiment(experiment, method, criterion, title=None, **kwargs):
    fn = "data/{e}/{m}.pck".format(e=experiment, m=method)
    if title is None:
        title = "{} {}".format(method, experiment)
    return plot_2d_error_grid_file(fn, criterion, title=title, **kwargs)


def plot_2d_error_grid_file(fn, criterion, **kwargs):
    with open(fn) as f:
        d = pickle.load(f)
    d.update(kwargs)
    return plot_2d_error_grid(criterion=criterion, **d)


def plot_2d_error_grid(
    criterion, res, param_names, params, criteria, maxerr=5, transform=lambda x: x,
        title="", cmap="hot", pn1=None, pn2=None, settings={}, ticks=True, figsize=(10, 12), **kwargs):
    if pn1 is None and pn2 is None:
        pn1 = param_names[0]
        pn2 = param_names[1]
    erri = criteria.index(criterion)
    ferr = res[..., erri].copy()
    ferr[ferr > maxerr] = np.nan
    ferr = transform(ferr)
    i = [slice(
        None) if (i == pn1 or i == pn2) else np.flatnonzero(np.array(kwargs[i])== settings[i])[0] for i in param_names]
    ferr = ferr[i]
    if param_names.index(pn1) < param_names.index(pn2):
        ferr = ferr.T
    f = plt.figure(figsize=figsize)
    plt.imshow(ferr, interpolation="nearest", cmap=cmap, norm=LogNorm(
        vmin=np.nanmin(ferr), vmax=np.nanmax(ferr)))
    p1 = kwargs[pn1]
    p2 = kwargs[pn2]
    plt.title(title)
    if ticks:
        plt.yticks(range(len(p2)), p2)
        plt.xticks(range(len(p1)), p1, rotation=45, ha="right")
        plt.xlabel(pn1)
        plt.ylabel(pn2)
        return f
    else:
        return f, p1, p2


def run_experiment(task, methods, n_indep, l, error_every, name, n_eps,
                   mdp, phi, title, verbose=1, n_jobs=1, criteria=None,
                   episodic=False, eval_on_traces=False, n_samples_eval=None, **kwargs):
    a, b, c = task.avg_error_traces(methods, n_indep=n_indep, n_eps=n_eps,
                                    n_samples=l, error_every=error_every,
                                    criteria=criteria, eval_on_traces=eval_on_traces,
                                    n_samples_eval=n_samples_eval,
                                    verbose=verbose, n_jobs=n_jobs, episodic=episodic)
    return a, b, c


def plot_path(path, method_id, methods, criterion, title):
    plt.figure(figsize=(15, 10))
    plt.ylabel(criterion)
    plt.xlabel("Regularization Parameter")
    plt.title(title + " " + methods[method_id].name)

    par, theta, err = zip(*path[criterion][method_id])
    plt.plot(par, err)
    plt.show()


def save_results(name, l, criteria, error_every, n_indep, n_eps, methods,
                 mdp, phi, title, mean, std, raw, gamma, episodic=False, **kwargs):
    if not os.path.exists("data/{name}".format(name=name)):
        os.makedirs("data/{name}".format(name=name))

    with open("data/{name}/setting.pck".format(name=name), "w") as f:
        pickle.dump(dict(l=l, criteria=criteria, gamma=gamma,
                         error_every=error_every,
                         n_indep=n_indep,
                         episodic=episodic,
                         n_eps=n_eps,
                         methods=methods,
                         mdp=mdp, phi=phi, title=title, name=name), f)

    np.savez_compressed("data/{name}/results.npz".format(
        name=name), mean=mean, std=std, raw=raw)


def load_results(name, update_title=False):
    with open("data/{name}/setting.pck".format(name=name), "r") as f:
        d = pickle.load(f)

    if update_title:
        replace_title(name, d)
        with open("data/{name}/setting.pck".format(name=name), "w") as f:
            pickle.dump(file=f, obj=d)
    d2 = np.load("data/{name}/results.npz".format(name=name))
    d.update(d2)

    return d


def replace_title(exp, data):
    exec "from experiments." + exp + " import title"
    data["title"] = title
    return data


def filter_methods(data):
    max_ys = np.median(np.max(data['mean'], axis=-1), axis=0) * 5.0

    for i, method in enumerate(data['methods']):
        if np.any(max_ys < np.max(data['mean'][i, ...], axis=-1)):
            method.hide = True
            print("hiding: ", method.name)
        else:
            print("not hiding: ", method.name)

        # method.hide = (
        #     'BBO' in method.name or 1e-2 < method.alpha.alpha)

        print(max_ys < np.max(data['mean'][i, ...], axis=-1))
        print(max_ys)
        print(np.max(data['mean'][i, ...], axis=-1))

        # names_to_filter = {'TD(0.0)', 'FPKF(0.0)', 'LSTD-JP(0)', 'LSPE(0.0)'}
        # for method in data['methods']:
        #     if any(name in method.name for name in names_to_filter):
        #         method.hide = True
        #         print("hiding: ", method.name)
        #     else:
        #         print("not hiding: ", method.name)


def plot_errorbar(name, title, methods, mean, std, l, error_every, criterion,
                  criteria, n_eps, episodic=False, ncol=1, figsize=(7.5, 5), **kwargs):
    max_items_per_row = 3
    rows = int(np.ceil(len(criteria) / max_items_per_row))
    items_per_row = min(max_items_per_row, len(criteria))
    figsize = (figsize[0] * items_per_row, figsize[1] * rows)
    # figsize = (figsize[0] * len(criteria), figsize[1])
    figure, axes = plt.subplots(
        rows,
        items_per_row,
        figsize=figsize,
        constrained_layout=True)

    axes = np.atleast_1d(axes)

    for (index, axis), criterion in zip(np.ndenumerate(axes), criteria):
        k = criteria.index(criterion)

        if index[0] == axes.shape[0] - 1:
            axis.set_xlabel("Timesteps")

        axis.set_ylabel(criterion)

        # y_max = np.median(np.max(mean[:, k, :], axis=-1), axis=0) * 1.5
        # axis.set_ylim(0, y_max)

        x = (
            range(0, l * n_eps, error_every)
            if not episodic
            else range(n_eps))
        if episodic:
            ee = int(n_eps / 8.)
        else:
            ee = int(l * n_eps / error_every / 8.)
        if ee < 1:
            ee = 1
        lss = ["-", "--", "-.", ":"] * 5
        for i, m in enumerate(methods):
            if hasattr(m, "hide") and m.hide:
                continue
            ls = lss[int(i / 7)]

            # label = "{}, error sum={}".format(m.name, np.sum(mean[i, k, :]))
            label = m.name
            axis.errorbar(x, mean[i, k, :], yerr=std[i, k, :],
                          errorevery=ee, label=label, ls=ls)

    title = figure.suptitle(title, y=1.05)

    handles, labels = axis.get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        bbox_to_anchor=(0.0, 0.0, 1.0, 0.9),
        framealpha=1.0,
        ncol=5,
    )

    plt.savefig(
        'data/{name}/errorbar.png'.format(name=name),
        bbox_extra_artists=(legend, title),
        bbox_inches='tight',
    )


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        choices=('train', 'visualize'),
        default='visualize')

    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)

    return parser


def experiment_main(task, name, criterion, methods, *args, **kwargs):
    argument_parser = get_argument_parser()
    cli_args = argument_parser.parse_args()

    set_gpu_memory_growth(True)

    if cli_args.mode == 'train':
        mean, std, raw = run_experiment(
            *args,
            task=task,
            name=name,
            criterion=criterion,
            methods=methods,
            n_jobs=cli_args.n_jobs,
            verbose=cli_args.verbose,
            **kwargs)

        save_results(
            *args,
            name=name,
            methods=methods,
            mean=mean,
            std=std,
            raw=raw,
            **kwargs)
        plot_errorbar(
            *args,
            name=name,
            criterion=criterion,
            methods=methods,
            mean=mean,
            std=std,
            raw=raw,
            **kwargs)

        pprint({m.name: m.time for m in methods})
    elif cli_args.mode == 'visualize':
        data = load_results(name)
        data['criterion'] = criterion
        filter_methods(data)
        plot_errorbar(**data)
    else:
        raise ValueError(cli_args.mode)

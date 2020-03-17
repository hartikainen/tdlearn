# -*- coding: utf-8 -*-
"""
Boyan Chain Experiment
"""
__author__ = "Christoph Dann <cdann@cdann.de>"

import td
import td_non_linear
import examples
from task import LinearDiscreteValuePredictionTask
import numpy as np
import features
from experiments import experiment_main

n = 14
n_feat = 4
mdp = examples.BoyanChain(n, n_feat)
phi = features.spikes(n_feat, n)
# phi = features.one_hot(n)
gamma = .95
p0 = np.zeros(n_feat)
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, p0)

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

methods = []

alpha = 1.0
prior_mean = 0.0
hidden_layer_sizes = (32, 32)
# prior_stddev = 0.1

for network_lr in (1e-4, 1e-3, 1e-2, ):
# for network_lr in (1e-5, ):
    for V_fn_lr in (1e-4, 1e-3, 1e-2, ):  # 1e-1 fails
    # for V_fn_lr in (0.3, 1.0, 3.0, 10.0):
        # for prior_stddev in (1e-6, 1e-3, 1e-2, 1e-1, 1.0):
        # for prior_stddev in (0.0, ):
        for prior_stddev in (0.0, 1e-3, 1e-2, 1e-1, 1.0):
            bbo = td_non_linear.NonLinearBBO(
                alpha,
                D_s=n,
                D_a=1,
                prior_mean=prior_mean,
                prior_stddev=prior_stddev,
                network_lr=network_lr,
                V_fn_lr=V_fn_lr,
                hidden_layer_sizes=hidden_layer_sizes,
                phi=phi)
            bbo.name = (
                r"BBO $\sigma_0$={prior_stddev}"
                r"$\alpha$={V_fn_lr:.2e}, "
                r"$\beta$={network_lr:.2e}, "
                "".format(
                    prior_mean=prior_mean,
                    prior_stddev=prior_stddev,
                    V_fn_lr=V_fn_lr,
                    network_lr=network_lr,
                ))
            bbo.color = "black"
            methods.append(bbo)


for alpha in (1e-4, 1e-3, 1e-2):
    non_linear_td0 = td_non_linear.NonLinearTD0(
        alpha,
        D_s=n,
        D_a=1,
        hidden_layer_sizes=hidden_layer_sizes,
        phi=phi)
    non_linear_td0.name = r"NonLinearTD0 $\alpha$={}".format(alpha)
    non_linear_td0.color = "black"
    methods.append(non_linear_td0)


l = 20
n_eps = 100
episodic = True
error_every = 1
name = "boyan_non_linear"
n_indep = 200
title = "1. 14-State Boyan Chain".format(n, n_indep)
criterion = "MSE"
criteria = ["MSE"]
# criteria = ["RMSPBE", "RMSBE", "RMSE", "MSPBE", "MSBE", "MSE"]

gs_errorevery = 1


if __name__ == "__main__":
    # for x in methods:
    #     x.hide = (
    #         (0 <= x._network_lr)
    #         # or (
    #         # )
    #     )
    #     print(x.name, x.hide)
    experiment_main(**globals())

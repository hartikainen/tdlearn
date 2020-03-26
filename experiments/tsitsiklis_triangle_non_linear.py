# -*- coding: utf-8 -*-
"""
Boyan Chain Experiment
"""
__author__ = "Christoph Dann <cdann@cdann.de>"

import td
import td_spiral
import examples
from task import LinearDiscreteValuePredictionTask
import numpy as np
import features
from experiments import experiment_main

n = 14
mdp = examples.TsitsiklisTriangle()
phi = features.one_hot(3)
gamma = .95
p0 = np.zeros(3)  # TODO(hartikainen): Check this
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, p0)

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

methods = []

initial_omega = 0.0
epsilon = 1e-1


prior_mean = 0.0
# prior_stddev = 0.1

# for network_lr in (1e-3, 1e-2, 1e-1, 1.0):
for network_lr in (1e-2, ):
# for network_lr in (1e-5, ):
    # for V_fn_lr in (network_lr * x for x in (1.0, 0.3, 0.1)):
    # for V_fn_lr in (1e-3, 1e-2, 1e-1, 1.0):
    for V_fn_lr in (1e-3, ):
    # for V_fn_lr in (network_lr * x for x in (1.0, 3.0, 1.0)):
    # for V_fn_lr in (3e-4, ):
    # for V_fn_lr in (1e-4, ):  # 1e-1 fails
    # for V_fn_lr in (0.3, 1.0, 3.0, 10.0):
        # for prior_stddev in (1e-6, 1e-3, 1e-2, 1e-1, 1.0):
        for prior_stddev in (10.0, ):
        # for prior_stddev in (1e-1, 1.0, 3.0, 10.0):
        # for prior_stddev in (3e-2, 1e-1, 3e-1, 1.0):
            bbo = td_spiral.SpiralNonLinearBilevel(
            # bbo = td_spiral.SpiralNonLinearBBO(
                alpha=0.0,
                D_s=3,
                D_a=1,
                initial_omega=0.0,
                epsilon=epsilon,
                prior_mean=prior_mean,
                prior_stddev=prior_stddev,
                network_lr=network_lr,
                V_fn_lr=V_fn_lr,
                phi=phi)
            bbo.name = (
                r"BBO "
                r"$\sigma_0$={prior_stddev}, "
                r"$\alpha$={network_lr:.2e}, "
                r"$\beta$={V_fn_lr:.2e}, "
                "".format(
                    prior_mean=prior_mean,
                    prior_stddev=prior_stddev,
                    V_fn_lr=V_fn_lr,
                    network_lr=network_lr,
                    # V_fn_lr=network_lr,
                    # network_lr=V_fn_lr,
                ))
            bbo.color = "black"
            methods.append(bbo)


# for alpha in (1e-3, 1e-2, 1e-1, ):
#     non_linear_td0 = td_spiral.SpiralNonLinearTD0(
#         alpha,
#         D_s=3,
#         D_a=1,
#         initial_omega=0.0,
#         epsilon=epsilon,
#         phi=phi)
#     non_linear_td0.name = r"NonLinearTD0 $\alpha$={}".format(alpha)
#     non_linear_td0.color = "black"
#     methods.append(non_linear_td0)

l = 1000
n_eps = 1
episodic = False
error_every = 1
name = "tsitsiklis_triangle_non_linear"
n_indep = 50
title = "Tsitsiklis Triangle".format(n, n_indep)
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

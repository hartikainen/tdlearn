# -*- coding: utf-8 -*-
"""
Uniformly sampled random MDP with discrete states, off-policy case
"""
__author__ = "Christoph Dann <cdann@cdann.de>"

import td
import td_non_linear
import examples
from task import LinearDiscreteValuePredictionTask
import numpy as np
import features
import policies
import regtd
from experiments import experiment_main


n = 400
n_a = 10
n_feat = 200
mdp = examples.RandomMDP(n, n_a)
phi = features.lin_random(n_feat, n, constant=True)
# phi = features.one_hot(n)
gamma = .95
np.random.seed(3)
beh_pol = policies.Discrete(np.random.rand(n, n_a))
tar_pol = policies.Discrete(np.random.rand(n, n_a))
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=beh_pol, target_policy=tar_pol)


methods = []

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

alpha = 1.0
prior_mean = 0.0
# prior_stddev = 0.1

for network_lr in (3e-4, ):
    for V_fn_lr in (1e-3, ):
        # for prior_stddev in (1e-3, 1e-2, 1e-1, 1.0):
        # for prior_stddev in (1e-2, ):
        for prior_stddev in (0.0, ):
            bbo = td_non_linear.NonLinearBBOV2(
                alpha,
                D_s=n_feat,
                D_a=1,
                prior_mean=prior_mean,
                prior_stddev=prior_stddev,
                network_lr=network_lr,
                V_fn_lr=V_fn_lr,
                phi=phi)
            bbo.name = (
                r"{name} $\sigma_0$={prior_stddev}"
                r"$\alpha$={V_fn_lr:.2e}, "
                r"$\beta$={network_lr:.2e}, "
                "".format(
                    name=bbo.__class__.__name__,
                    prior_mean=prior_mean,
                    prior_stddev=prior_stddev,
                    V_fn_lr=V_fn_lr,
                    network_lr=network_lr,
                ))
            bbo.color = "black"
            methods.append(bbo)


# alpha = 1.0
# non_linear_td0 = td_non_linear.NonLinearTD0(alpha, D_s=n_feat, D_a=1, phi=phi)
# non_linear_td0.name = r"NonLinearTD0 $\alpha$={}".format(alpha)
# non_linear_td0.color = "black"
# methods.append(non_linear_td0)


l = 8000
n_eps = 1
n_indep = 200

episodic = False
error_every = 80
name = "disc_random_off_non_linear"
title = "4. {}-State Random MDP Off-policy".format(n, n_indep)
criterion = "MSE"
criteria = ["MSE"]
# criteria = ["RMSPBE", "RMSBE", "RMSE", "MSPBE", "MSBE", "MSE"]


if __name__ == "__main__":
    experiment_main(**globals())

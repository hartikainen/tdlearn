# -*- coding: utf-8 -*-
"""
20-link pole balancing task with impoverished features, off-policy case
"""
__author__ = "Christoph Dann <cdann@cdann.de>"

import td
import td_non_linear
import examples
import numpy as np
import dynamic_prog as dp
import features
import policies
from task import (
    LinearLQRValuePredictionTask, LinearContinuousValuePredictionTask)
from experiments import experiment_main


gamma=0.95
dt = 0.1
dim = 20
sigma = np.ones(2*dim)*0.01
mdp = examples.NLinkPendulumMDP(np.ones(dim)*.5, np.ones(dim)*.6, sigma=sigma, dt=dt)
# phi = features.squared_diag(2*dim)
phi = features.identity(2 * dim)


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p,_,_ = dp.solve_LQR(mdp, gamma=gamma)
theta_p = np.array(theta_p)
policy = policies.LinearContinuous(theta=theta_p, noise=np.ones(dim)*0.01)
target_policy = policies.LinearContinuous(theta=theta_p, noise=np.ones(dim)*0.005)
theta0 =  0.*np.ones(n_feat)

# import ipdb; ipdb.set_trace(context=30)

task = LinearContinuousValuePredictionTask(
    mdp,
    gamma,
    phi,
    theta0,
    policy=policy,
    target_policy=target_policy,
    mu_next=1000,
    normalize_phi=True)

# task = LinearLQRValuePredictionTask(
#     mdp,
#     gamma,
#     phi,
#     theta0,
#     policy=policy,
#     target_policy=target_policy,
#     mu_next=1000,
#     normalize_phi=True)

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
        # for prior_stddev in (0.0, 1e-3, 1e-2, 1e-1, 1.0):
        for prior_stddev in (1e-2, ):
            bbo = td_non_linear.NonLinearBBO(
                alpha,
                D_s=n_feat,
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
        D_s=n_feat,
        D_a=1,
        hidden_layer_sizes=hidden_layer_sizes,
        phi=phi)
    non_linear_td0.name = r"NonLinearTD0 $\alpha$={}".format(alpha)
    non_linear_td0.color = "black"
    methods.append(non_linear_td0)


l = 50000
error_every = 500
n_indep = 50
# n_indep = 1
n_eps = 1
episodic = False
criterion = "MSE"
criteria = ["MSE"]
# criteria = ["RMSPBE", "RMSBE", "RMSE", "MSPBE", "MSBE", "MSE"]
title = "12. 20-link Lin. Pole Balancing Off-pol."
name = "link20_imp_offpolicy"


if __name__ == "__main__":
    experiment_main(**globals())

import td
import td_non_linear
import examples
import numpy as np
import features
import matplotlib.pyplot as plt
from task import LinearDiscreteValuePredictionTask
import policies
from experiments import experiment_main

import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

n = 7
beh_pi = np.ones((n + 1, 2))
beh_pi[:, 0] = float(n) / (n + 1)
beh_pi[:, 1] = float(1) / (n + 1)
beh_pol = policies.Discrete(prop_table=beh_pi)
target_pi = np.zeros((n + 1, 2))
target_pi[:, 0] = 0
target_pi[:, 1] = 1
target_pol = policies.Discrete(prop_table=target_pi)

mdp = examples.BairdStarExample(n)
phi = features.one_hot(n)

methods = []

gamma = 0.99
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi,
                                         np.asarray(n * [1.] + [10., 1.]),
                                         policy=beh_pol,
                                         target_policy=target_pol)

alpha = 1.0
prior_mean = 0.0
prior_stddev = 0.1
bbo = td_non_linear.NonLinearBBO(
    alpha,
    D_s=n,
    D_a=target_pol.dim_A,
    prior_mean=prior_mean,
    prior_stddev=prior_stddev,
    phi=phi)
bbo.name = r"BBO $\mu_0$={prior_mean}, $\sigma_0$={prior_stddev}".format(
    prior_mean=prior_mean,
    prior_stddev=prior_stddev)
bbo.color = "black"
methods.append(bbo)


alpha = 1.0
non_linear_td0 = td_non_linear.NonLinearTD0(
    alpha,
    D_s=n,
    D_a=target_pol.dim_A,
    phi=phi)
non_linear_td0.name = r"NonLinearTD0 $\alpha$={}".format(alpha)
non_linear_td0.color = "black"
methods.append(non_linear_td0)


l = 1000
error_every = 1
n_indep = 200
n_eps = 1
episodic = False
name = "baird_non_linear"
title = "2. Baird Star Example"
criterion = "MSE"
# criteria = ["RMSPBE", "RMSBE", "RMSE", "MSPBE", "MSBE", "MSE"]
criteria = ["MSE"]
gs_errorevery = 10


if __name__ == "__main__":
    experiment_main(**globals())

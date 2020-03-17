# -*- coding: utf-8 -*-
"""
Uniformly sampled random MDP with discrete states, on-policy case
Experiment to determine the fix-points of MSBE, MSPBE, MSTDE
"""
__author__ = "Christoph Dann <cdann@cdann.de>"

import td
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
gamma = .95
np.random.seed(3)
beh_pol = policies.Discrete(np.random.rand(n, n_a))
tar_pol = beh_pol
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=beh_pol)


methods = []

lam = 0.
eps = 100
lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)

brm = td.RecursiveBRMDS(phi=phi, eps=100)
brm.name = "BRMDS"
brm.color = "b"
brm.ls = "--"
methods.append(brm)

brm = td.RecursiveBRM(phi=phi, eps=100)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)


l = 1000010
n_eps = 1
n_indep = 20

episodic = False
error_every = 1000000
name = "disc_random_on_conv"
title = "3. {}-State Random MDP On-policy".format(n, n_indep)
criterion = "MSE"
criteria = ["RMSPBE", "RMSBE", "RMSE", "MSPBE", "MSBE", "MSE"]


if __name__ == "__main__":
    experiment_main(**globals())

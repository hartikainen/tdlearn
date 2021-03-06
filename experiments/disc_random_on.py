# -*- coding: utf-8 -*-
"""
Uniformly sampled random MDP with discrete states, on-policy case
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


alpha = 1.0
bbo_v2 = td.BBOV2(
    alpha,
    D_a=tar_pol.dim_A,
    prior_epsilon=10,
    phi=phi)
bbo_v2.name = r"BBO-v2".format()
bbo_v2.color = "black"
methods.append(bbo_v2)

alpha = 1.0
bbo_v3 = td.BBOV3(
    alpha,
    D_a=tar_pol.dim_A,
    prior_epsilon=0.3,
    phi=phi)
bbo_v3.name = r"BBO-v3".format()
bbo_v3.color = "black"
methods.append(bbo_v3)


# alpha = 0.007
# mu = .0001
# gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
# gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
# gtd.color = "r"
# methods.append(gtd)

alpha, mu = 0.003, 4
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)

# alpha = td.RMalpha(0.09, 0.25)
# lam = .0
# td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
# td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
# td0.color = "k"
# methods.append(td0)

alpha = td.DabneyAlpha()
lam = .0
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$=auto".format(lam, alpha)
td0.color = "k"
methods.append(td0)


# alpha = .001
# lam = .4
# td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
# td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
# td0.color = "k"
# methods.append(td0)

lam = 0.
alpha = 0.007
mu = 0.01
tdc = td.TDCLambda(alpha=alpha, mu=mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

# alpha = .1
# lam = .0
# lstd = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
# lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
# lstd.color = "g"
# methods.append(lstd)

lam = 0.
eps = 10
lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)

lam = 0.0
eps = np.nan
lstd = td.LSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
methods.append(lstd)

#
# alpha = 0.5
# lam = .4
# beta = 10.
# mins = 1000
# lstd = td.FPKF(
#     lam=lam, alpha=alpha, mins=mins, beta=beta, phi=phi, gamma=gamma)
# lstd.name = r"FPKF({}) $\alpha={}$ $\beta={}$ m={}".format(
#     lam, alpha, beta, mins)
# lstd.color = "g"
# lstd.ls = "-."
# methods.append(lstd)

# alpha = .006
# rg = td.ResidualGradientDS(alpha=alpha, phi=phi, gamma=gamma)
# rg.name = r"RG DS $\alpha$={}".format(alpha)
# rg.color = "brown"
# rg.ls = "--"
# methods.append(rg)

# alpha = .001
# rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
# rg.name = r"RG $\alpha$={}".format(alpha)
# rg.color = "brown"
# methods.append(rg)


# brm = td.RecursiveBRMDS(phi=phi, eps=0.1)
# brm.name = "BRMDS"
# brm.color = "b"
# brm.ls = "--"
# methods.append(brm)

brm = td.RecursiveBRMLambda(phi=phi, eps=10, lam=0.)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)


# tau = 0.00003
# lstd = regtd.LSTDl1(tau=tau, lam=0, phi=phi)
# lstd.name = r"LSTD-l1({}) $\tau={}$".format(0, tau)
# lstd.color = "b"
# methods.append(lstd)

# tau = 0.05
# lstd = regtd.LarsTD(tau=tau, lam=0, phi=phi)
# lstd.name = r"LarsTD({}) $\tau={}$".format(0, tau)
# lstd.color = "b"
# methods.append(lstd)

# tau = 0.05
# beta = 0.05
# lstd = regtd.LSTDl21(tau=tau, beta=beta, lam=0, phi=phi)
# lstd.name = r"LSTD({}) $\ell_{{21}}$ $\tau={}$, $\beta={}$".format(
#     0, tau, beta)
# lstd.color = "b"
#methods.append(lstd)


l = 8000
n_eps = 1
n_indep = 200

episodic = False
error_every = 80
name = "disc_random_on"
title = "3. {}-State Random MDP On-policy".format(n, n_indep)
criterion = "MSE"
criteria = ["RMSPBE", "RMSBE", "RMSE", "MSPBE", "MSBE", "MSE"]


if __name__ == "__main__":
    experiment_main(**globals())

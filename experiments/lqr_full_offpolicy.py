# -*- coding: utf-8 -*-
"""
pole balancing experiment with prefect features and off-policy samples
"""
__author__ = "Christoph Dann <cdann@cdann.de>"

import td
import examples
import numpy as np
import dynamic_prog as dp
import features
import policies
from task import LinearLQRValuePredictionTask
from experiments import experiment_main


gamma = 0.95
sigma = np.array([0.] * 3 + [0.01])
dt = 0.1
mdp = examples.PoleBalancingMDP(sigma=sigma, dt=dt)
phi = features.squared_tri(11)


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p, _, _ = dp.solve_LQR(mdp, gamma=gamma)
theta_p = np.array(theta_p).flatten()
theta_o = theta_p.copy()
beh_policy = policies.LinearContinuous(theta=theta_o, noise=np.array([0.01]))
target_policy = policies.LinearContinuous(theta=theta_p, noise=np.array([0.001]))
theta0 = 0. * np.ones(n_feat)

task = LinearLQRValuePredictionTask(
    mdp, gamma, phi, theta0, policy=beh_policy, target_policy=target_policy,
    normalize_phi=True, mu_next=1000)


methods = []


alpha = 1.0
bbo_v2 = td.BBOV2(
    alpha,
    D_a=target_policy.dim_A,
    prior_epsilon=10.0,
    phi=phi)
bbo_v2.name = r"BBO-v2".format()
bbo_v2.color = "black"
methods.append(bbo_v2)

alpha = 1.0
bbo_v3 = td.BBOV3(
    alpha,
    D_a=target_policy.dim_A,
    prior_epsilon=0.3,
    phi=phi)
bbo_v3.name = r"BBO-v3".format()
bbo_v3.color = "black"
methods.append(bbo_v3)


# alpha = 0.001
# mu = .0001
# gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
# gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
# gtd.color = "r"
# methods.append(gtd)

alpha, mu = 0.001, 1.
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)

# alpha = td.RMalpha(0.03, 0.25)
# lam = .0
# td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
# td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
# td0.color = "k"
# methods.append(td0)

alpha = .002
lam = .2
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

# lam = 0.2
# alpha = 0.002
# mu = 0.05
# tdc = td.TDCLambda(alpha=alpha, mu = mu, lam=lam, phi=phi, gamma=gamma)
# tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
# tdc.color = "b"
# methods.append(tdc)

lam = 0.
alpha = 0.003
mu = 0.1
tdc = td.GeriTDCLambda(alpha=alpha, mu = mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({})-TO $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

# alpha = .001
# lam = 0.
# lstd = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
# lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
# lstd.color = "g"
# methods.append(lstd)

# alpha = 1.
# lam = .0
# lstd = td.RecursiveLSPELambdaCO(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
# lstd.name = r"LSPE({})-TO $\alpha$={}".format(lam, alpha)
# lstd.color = "g"
# methods.append(lstd)

lam = 0.
eps = 10
lstd = td.RecursiveLSTDLambdaJP(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD-TO({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)


lam = 0.0
eps = np.nan
lstd = td.LSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
methods.append(lstd)


# lam = 0.
# eps = 0.01
# lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
# lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
# lstd.color = "g"
# lstd.ls = "-."
# methods.append(lstd)
# #
# alpha = 1.
# lam = .4
# beta = 1.
# mins = 500
# lstd = td.FPKF(lam=lam, alpha = alpha, beta=beta, mins=mins, phi=phi, gamma=gamma)
# lstd.name = r"FPKF({}) $\alpha$={}".format(lam, alpha)
# lstd.color = "g"
# lstd.ls = "-."
# methods.append(lstd)

# alpha = .008
# rg = td.ResidualGradientDS(alpha=alpha, phi=phi, gamma=gamma)
# rg.name = r"RG DS $\alpha$={}".format(alpha)
# rg.color = "brown"
# rg.ls = "--"
# methods.append(rg)

# alpha = .005
# rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
# rg.name = r"RG $\alpha$={}".format(alpha)
# rg.color = "brown"
# methods.append(rg)


# brm = td.BRMDS(phi=phi)
# brm.name = "BRMDS"
# brm.color = "b"
# brm.ls = "--"
# methods.append(brm)

brm = td.BRM(phi=phi)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)


l = 30000
error_every = 500
n_indep = 50
n_eps = 1
episodic = False
criterion = "MSE"
criteria = ["RMSPBE", "RMSBE", "RMSE", "MSPBE", "MSBE", "MSE"]
name = "lqr_full_offpolicy"
title = "8. Lin. Cart-Pole Balancing Off-pol. Perf. Feat."


if __name__ == "__main__":
    experiment_main(**globals())

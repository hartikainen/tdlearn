# -*- coding: utf-8 -*-
"""
Boyan Chain Experiment
"""
__author__ = "Christoph Dann <cdann@cdann.de>"

import td
import examples
from task import LinearDiscreteValuePredictionTask
import numpy as np
import features
from experiments import experiment_main

n = 14
n_feat = 4
mdp = examples.BoyanChain(n, n_feat)
phi = features.spikes(n_feat, n)
gamma = .95
p0 = np.zeros(n_feat)
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, p0)

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

methods = []

alpha = 1.0
bbo_v2 = td.BBOV2(
    alpha,
    D_a=1,
    prior_epsilon=10.0,
    phi=phi)
bbo_v2.name = r"BBO-v2".format()
bbo_v2.color = "black"
methods.append(bbo_v2)

alpha = 1.0
bbo_v3 = td.BBOV3(
    alpha,
    D_a=1,
    prior_epsilon=3.0,
    phi=phi)
bbo_v3.name = r"BBO-v3".format()
bbo_v3.color = "black"
methods.append(bbo_v3)

# alpha = .5
# mu = 2.
# gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
# gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
# gtd.color = "#6E086D"
# methods.append(gtd)

alpha = .5
mu = 1.
gtd2 = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd2.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd2.color = "#6E086D"
methods.append(gtd2)

# alpha = 0.2
# lam = 1.
# td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi)
# td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
# methods.append(td0)

alpha = td.RMalpha(10., 0.5)
lam = 0.
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi)
td0.name = r"TD({}) $\alpha={}t^{{-{} }}$".format(lam, alpha.c, alpha.mu)
methods.append(td0)

# alpha = td.DabneyAlpha()
# lam = 0.
# td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi)
# td0.name = r"TD({}) $\alpha$=aut.".format(lam)
# methods.append(td0)

alpha = 0.2
mu = 0.0001
lam = 1.
tdc = td.TDCLambda(lam=lam, alpha=alpha, beta=alpha * mu, phi=phi)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "r"
methods.append(tdc)

lam = .8
eps = 10000
lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
methods.append(lstd)

lam = .0
eps = np.nan
lstd = td.LSTDLambda(lam=lam, eps=eps, phi=phi)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
methods.append(lstd)

# lam = .0
# eps = 100
# lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi)
# lstd.name = r"LSTD({})".format(lam)
# methods.append(lstd)

# lam = .0
# eps = 100
# lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi)
# lstd.name = r"LSTD({})".format(lam)
# methods.append(lstd)

# lam = .8
# alpha = 1.
# lspe = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi)
# lspe.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
# methods.append(lspe)

# lam = .0
# alpha = .01
# beta = 1000
# mins = 0
# fpkf = td.FPKF(lam=lam, alpha=alpha, beta=beta, mins=mins, phi=phi)
# fpkf.name = r"FPKF({}) $\alpha={}$ $\beta={}$".format(lam, alpha, beta)
# fpkf.ls = "--"
# methods.append(fpkf)

# brm_ds = td.RecursiveBRMDS(phi=phi)
# brm_ds.name = "BRMDS"
# brm_ds.color = "b"
# brm_ds.ls = "--"
# methods.append(brm_ds)

brm = td.RecursiveBRM(phi=phi)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)

# alpha = 0.5
# rg_ds = td.ResidualGradientDS(alpha=alpha, phi=phi)
# rg_ds.name = r"RG DS $\alpha$={}".format(alpha)
# rg_ds.ls = "--"
# methods.append(rg_ds)

# alpha = 0.5
# rg = td.ResidualGradient(alpha=alpha, phi=phi)
# rg.name = r"RG $\alpha$={}".format(alpha)
# methods.append(rg)

# eta = 0.001
# reward_noise = 0.001
# P_init = 1.
# ktd = td.KTD(phi=phi, gamma=1., P_init=P_init, theta_noise=None, eta=eta,
#              reward_noise=reward_noise)
# ktd.name = r"KTD $\eta$={}, $\sigma^2$={} $P_0$={}".format(
#     eta, reward_noise, P_init)
# methods.append(ktd)

# sigma = 1.
# gptdp = td.GPTDP(phi=phi, sigma=sigma)
# gptdp.name = r"GPTDP $\sigma$={}".format(sigma)
# gptdp.ls = "--"
# methods.append(gptdp)

# lam = .8
# sigma = 1e-5
# gptdp = td.GPTDPLambda(phi=phi, tau=sigma, lam=lam)
# gptdp.name = r"GPTDP({}) $\sigma$={}".format(lam, sigma)
# gptdp.ls = "--"
# methods.append(gptdp)


l = 20
n_eps = 100
episodic = True
error_every = 1
name = "boyan"
n_indep = 200
title = "1. 14-State Boyan Chain".format(n, n_indep)
criterion = "MSE"
criteria = ["RMSPBE", "RMSBE", "RMSE", "MSPBE", "MSBE", "MSE"]

gs_errorevery = 1


if __name__ == "__main__":
    experiment_main(**globals())

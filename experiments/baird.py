# -*- coding: utf-8 -*-
"""
Experiment that shows arbitrary off-policy behavior of TD
"""
__author__ = "Christoph Dann <cdann@cdann.de>"
import td
import examples
import numpy as np
import features
import matplotlib.pyplot as plt
from task import LinearDiscreteValuePredictionTask
import policies
from experiments import experiment_main

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
phi = features.linear_blended(n + 1)

methods = []

gamma = 0.99
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi,
                                         np.asarray(n * [1.] + [10., 1.]),
                                         policy=beh_pol,
                                         target_policy=target_pol)

alpha = 1.0
bbo_v2 = td.BBOV2(
    alpha,
    D_a=target_pol.dim_A,
    phi=phi)
bbo_v2.name = r"BBO-v2".format()
bbo_v2.color = "black"
methods.append(bbo_v2)

alpha = 1.0
bbo_v3 = td.BBOV3(
    alpha,
    D_a=target_pol.dim_A,
    phi=phi)
bbo_v3.name = r"BBO-v3".format()
bbo_v3.color = "black"
methods.append(bbo_v3)


alpha = 0.004
mu = 4
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

alpha = 0.005
mu = 4.
gtd2 = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd2.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd2.color = "orange"
methods.append(gtd2)


alpha = .1
lam = 0.
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
methods.append(td0)


alpha = 0.006
mu = 16
tdc = td.TDC(alpha=alpha, beta=alpha * mu, phi=phi, gamma=gamma)
tdc.name = r"TDC(0) $\alpha$={} $\mu$={}".format(alpha, mu)
tdc.color = "b"
methods.append(tdc)


alpha = 0.003
mu = 8
geri_tdc = td.GeriTDC(alpha=alpha, beta=alpha * mu, phi=phi, gamma=gamma)
geri_tdc.name = r"GeriTDC $\alpha$={} $\mu$={}".format(alpha, mu)
geri_tdc.color = "c"
methods.append(geri_tdc)

alpha = .02
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

alpha = .01
rg_ds = td.ResidualGradientDS(alpha=alpha, phi=phi, gamma=gamma)
rg_ds.name = r"RG DS $\alpha$={}".format(alpha)
rg_ds.color = "brown"
methods.append(rg_ds)

lam = 0.
lstd = td.RecursiveLSTDLambda(lam=lam, phi=phi, gamma=gamma)
lstd.name = r"LSTD({})".format(lam)
lstd.color = "k"
# methods.append(lstd)

lam = 0.
alpha = .1
lspe = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi)
lspe.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
methods.append(lspe)

lam = 0.
alpha = .1
lspe_co = td.RecursiveLSPELambdaCO(lam=lam, alpha=alpha, phi=phi)
lspe_co.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
methods.append(lspe_co)

lam = 0.
alpha = .1
beta = 100.
fpkf = td.FPKF(lam=lam, alpha=alpha, beta=beta, phi=phi)
fpkf.name = r"FPKF({}) $\alpha={}$ $\beta={}$".format(lam, alpha, beta)
methods.append(fpkf)

brm = td.RecursiveBRM(phi=phi)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)

brmds = td.RecursiveBRMDS(phi=phi)
brmds.name = "BRMDS"
brmds.color = "b"
methods.append(brmds)

lam = 0.0
lstd_jp = td.RecursiveLSTDLambdaJP(lam=lam, phi=phi, gamma=gamma)
lstd_jp.name = r"LSTD-JP({})".format(0)
lstd_jp.color = "k"
methods.append(lstd_jp)


l = 1000
error_every = 1
n_indep = 200
n_eps = 1
episodic = False
name = "baird"
title = "2. Baird Star Example"
criterion = "MSE"
criteria = ["RMSPBE", "RMSBE", "RMSE", "MSPBE", "MSBE", "MSE"]
gs_errorevery = 10


if __name__ == "__main__":
    experiment_main(**globals())

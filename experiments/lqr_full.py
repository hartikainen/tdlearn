import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import features
import policies
from task import LinearLQRValuePredictionTask
import pickle

gamma=0.9
sigma = np.array([0.]*3 + [0.01])
#sigma = 0.
dt = 0.1
#mdp = examples.MiniLQMDP(dt=dt)
mdp = examples.PoleBalancingMDP(sigma=sigma, dt=dt)

phi = features.squared_tri()


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p,_,_ = dp.solve_LQR(mdp, gamma=gamma)
print theta_p
theta_p = np.array(theta_p).flatten()

policy = policies.LinearContinuous(theta=theta_p, noise=np.zeros((1,1)))
#theta0 =  10*np.ones(n_feat)
theta0 =  0.*np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0, policy=policy, normalize_phi=True)
#phi = task.phi
print "V_true", task.V_true
print "theta_true"
theta_true = phi.param_forward(*task.V_true)
print theta_true
#task.theta0 = theta_true
methods = []

#for alpha in [0.01, 0.005]:
#    for mu in [0.05, 0.1, 0.2, 0.01]:
#alpha = 0.1
alpha = 0.005
mu = 0.1
gtd = td.GTD(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

#for alpha in [.005,0.01,0.02]:
#    for mu in [0.01, 0.1]:
alpha, mu = 0.01, 0.1
gtd = td.GTD2(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)


#for alpha in [0.005, 0.01, 0.02, 0.03, 0.04]:
alpha = .01
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)
td0.color = "k"
methods.append(td0)

#for alpha in [0.005, 0.01, 0.02]:
#    for mu in [0.01, 0.1]:
for alpha, mu in [(.01,0.1)]:
    tdc = td.TDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
    tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)
    tdc.color = "b"
    methods.append(tdc)

#methods = []
#for eps in np.power(10,np.arange(-1,4)):
eps=100
lstd = td.RecursiveLSTDLambda(lam=0, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(0, eps)
lstd.color = "g"
methods.append(lstd)
#
#methods = []
#for alpha in [0.01, 0.02, 0.03]:
#alpha = .2
alpha=.01
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

ktd = td.KTD(phi=phi, gamma=gamma, theta_noise=None, eta=0.001, reward_noise=1e-5)
ktd.name = r"KTD"
methods.append(ktd)

sigma=1e-5
gptdp = td.GPTDP(phi=phi, sigma=sigma)
gptdp.name =r"GPTDP $\sigma$={}".format(sigma)
methods.append(gptdp)

l=50000
error_every=2000
n_indep=20
name="lqr_full_onpolicy"
title="4-dim. State Pole Balancing Onpolicy"

if __name__ =="__main__":
    mean, std, raw = task.avg_error_traces(methods, n_indep=n_indep,
        n_samples=l, error_every=error_every,
        criterion="RMSPBE",
        verbose=True)
    import os
    if not os.path.exists("data/{name}".format(name=name)):
        os.makedirs("data/{name}".format(name=name))

    with open("data/{name}/setting.pck".format(name=name), "w") as f:
        pickle.dump(dict(l=l, error_every=error_every, n_indep=n_indep, methods=methods, mdp=mdp, phi=phi),f)

    np.savez_compressed("data/{name}/results.npz".format(name=name), mean=mean, std=std, raw=raw)



    plt.figure(figsize=(15,10))
    plt.ylabel(r"$\sqrt{MSPBE}$")
    plt.xlabel("Timesteps")
    plt.title(title)
    for i, m in enumerate(methods):
        plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], errorevery=l/error_every/8, label=m.name)
        #plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], label=m.name)
    plt.legend()
    plt.show()
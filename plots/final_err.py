from experiments import *
import td
#exp_list = filter(lambda x: x != "baird", exp_list)
tab = np.zeros((len(exp_list), 3))
names = []
methods = (("GTD", td.GTD),
           ("GTD2",td.GTD2),
           ("TD",td.LinearTDLambda),
           ("TDC",td.GeriTDCLambda, td.TDCLambda),
           ("RG",td.ResidualGradient),
           ("RG DS",td.ResidualGradientDS),
           ("BRM",td.RecursiveBRM),
           ("BRM DS",td.RecursiveBRMDS),
           ("LSPE", td.RecursiveLSPELambdaCO, td.RecursiveLSPELambda),
           ("LSTD", td.RecursiveLSTDLambdaJP, td.RecursiveLSTDLambda),
           ("FPKF", td.FPKF))

print r"&"," & ".join([a[0] for a in methods]),r"\\"

for j,exp in enumerate(exp_list):
    d = load_results(exp)
    indices = []
    for i,m in enumerate(methods):
        cur_id = -1
        for t in m[1:]:
            if cur_id >= 0:
                break
            for io,mo in enumerate(d["methods"]):
                if isinstance(mo, t):
                    if t == td.LinearTDLambda and not isinstance(mo.alpha, float):
                        continue
                    cur_id = io
                    break
        indices.append(cur_id)
    k = d["criteria"].index("RMSE")
    e = [d["mean"][i, k, -1] for i in indices]
    print d["title"][0], "&", " & ".join(["{:.2f}".format(a) for a in e]), r"\\"


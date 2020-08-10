from pprint import pprint
import numpy as np
import pandas as pd
import tree

from experiments import load_results


tasks = (
  'boyan',
  # 'baird',
  'disc_random_on',
  'disc_random_off',
  'lqr_imp_onpolicy',
  'lqr_imp_offpolicy',
  'lqr_full_onpolicy',
  'lqr_full_offpolicy',
  # 'lqr_imp_onpolicy_unnorm',

  'link20_imp_onpolicy',
  'link20_imp_offpolicy',
  # 'link20_imp_offpolicy_non_linear',
)


# TASK_TO_INDEX_MAP = {
#     'boyan': '1. 14-State Boyan Chain',
#     'baird': '2. Baird Star Example',
#     'disc_random_on': '3. 400-State Random MDP On-policy',
#     'disc_random_off': '4. 400-State Random MDP Off-policy',
#     'lqr_imp_onpolicy': '5. Lin. Cart-Pole Balancing On-pol. Imp. Feat.',
#     'lqr_imp_offpolicy': '6. Lin. Cart-Pole Balancing Off-pol. Imp. Feat.',
#     'lqr_full_onpolicy': '7. Lin. Cart-Pole Balancing On-pol. Perf. Feat.',
#     'lqr_full_offpolicy': '8. Lin. Cart-Pole Balancing Off-pol. Perf. Feat.',
#     # '': '9. Cart-Pole Swingup On-policy',
#     # '': '10. Cart-Pole Swingup Off-policy',
#     'link20_imp_onpolicy': '11. 20-link Lin. Pole Balancing On-pol.',
#     'link20_imp_offpolicy': '12. 20-link Lin. Pole Balancing Off-pol.',
# }


TASK_TO_INDEX_MAP = {
    'boyan': '14-State Boyan Chain',
    'baird': 'Baird Star Example',
    'disc_random_on': '400-State Random MDP (On-pol.)',
    'disc_random_off': '400-State Random MDP (Off-pol.)',
    'lqr_imp_onpolicy': 'Cart-Pole (On-pol., Imp. Feat.)',
    'lqr_imp_offpolicy': 'Cart-Pole (Off-pol., Imp. Feat.)',
    'lqr_full_onpolicy': 'Cart-Pole (On-pol., Perf. Feat.)',
    'lqr_full_offpolicy': 'Cart-Pole (Off-pol., Perf. Feat.)',
    # '': '9. Cart-Pole Swingup On-policy',
    # '': '10. Cart-Pole Swingup Off-policy',
    'link20_imp_onpolicy': '20-Link Pole (On-pol.)',
    'link20_imp_offpolicy': '20-Link Pole (Off-pol.)',
}

# TASK_TO_INDEX_MAP = {
#     key: " ".join(value.split(" ")[1:])
#     for key, value in TASK_TO_INDEX_MAP.items()
# }


METHODS_TO_REPORT = (
    'TD',
    'GTD2',
    'TDC',

    'BRM',
    'LSTD',
    # 'LSPE'?

    # SOME PROBABILISTIC METHOD?
)


TASK_METHOD_TO_LABEL_MAP = {
    'baird': {
        'BBO': 'BBO',
        'BRM': 'BRM',
        # 'BRMDS': 0.0,
        # 'FPKF(0.0) $\\alpha=0.1$ $\\beta=100.0$': 10779.820754702554,
        # 'GTD $\\alpha$=0.004 $\\mu$=4': 2203.5767079862517,
        'GTD2 $\\alpha$=0.005 $\\mu$=4.0': 'GTD2',
        'GeriTDC $\\alpha$=0.003 $\\mu$=8': 'TDC',
        # 'LSPE(0.0) $\\alpha$=0.1': 8508.711779087396,
        'LSTD-JP(0)': 'LSTD',
        # 'RG $\\alpha$=0.02': 1724.1387750695255,
        # 'RG DS $\\alpha$=0.01': 3513.874007335677,
        'TD(0.0) $\\alpha$=0.1': 'TD',
        # 'TDC(0) $\\alpha$=0.006 $\\mu$=16': 'TDC',
    },
    'boyan': {
        'BBO': 'BBO',
        'BRM': 'BRM',
        # 'BRMDS': 73.51348023353925,
        # 'FPKF(0.0) $\\alpha=0.01$ $\\beta=1000$': 273.5480269701192,
        # 'GPTDP $\\sigma$=1.0': 113.30866064050988,
        # 'GPTDP(0.8) $\\sigma$=1e-05': 25.056131260632558,
        # 'GTD $\\alpha$=0.5 $\\mu$=2.0': 827.5133953339824,
        'GTD2 $\\alpha$=0.5 $\\mu$=1.0': 'GTD2',
        # 'KTD $\\eta$=0.001, $\\sigma^2$=0.001 $P_0$=1.0': 31.825294520747324,
        # 'LSPE(0.8) $\\alpha$=1.0': 25.61833224585434,
        # 'LSTD(0.0)': 32.224482273555594,
        'LSTD(0.8)': 'LSTD (w/ reg.)',
        'LSTD(0.8) $\\epsilon$=nan': 'LSTD (w/o reg.)',
        # 'RG $\\alpha$=0.5': 753.8162545455252,
        # 'RG DS $\\alpha$=0.5': 834.6363948867928,
        # 'TD(0.0) $\\alpha$=aut.': 81.90041554133579,
        'TD(0.0) $\\alpha=10.0t^{-0.5 }$': 'TD',
        # 'TD(1.0) $\\alpha$=0.2': 61.55086894524801,
        'TDC(1.0) $\\alpha$=0.2 $\\mu$=0.0001': 'TDC',
    },
    'disc_random_off': {
        'BBO': 'BBO',
        'BRM': 'BRM',
        # 'BRMDS': 470.39077096883125,
        # 'FPKF(0.0) $\\alpha=0.01$ $\\beta=10.0$ m=500': 979.0994049523174,
        # 'GTD $\\alpha$=0.007 $\\mu$=0.0001': 679.0949814861254,
        'GTD2 $\\alpha$=0.002 $\\mu$=1': 'GTD2',
        # 'LSPE(0.0) $\\alpha$=0.001': 1059.184260076886,
        # 'LSPE(0.2)-CO $\\alpha$=0.01': 252.693201628713,
        # 'LSTD(0.0) $\\epsilon$=10': 'LSTD',
        'LSTD-CO(0.0) $\\epsilon$=10': 'LSTD (w/ reg.)',
        'LSTD(0.0) $\\epsilon$=nan': 'LSTD (w/o reg.)',
        # 'LSTD-l1(0) $\\tau=0.0001$': 71.86247829537707,
        # 'LarsTD(0) $\\tau=0.05$': 104.1449172121861,
        # 'RG $\\alpha$=0.005': 442.84152418662165,
        # 'RG DS $\\alpha$=0.003': 472.60759580455624,
        'TD(0.0) $\\alpha$=RMAlpha(0.01, 0.1)': 'TD',
        # 'TD(0.4) $\\alpha$=0.004': 10865.382061498784,
        # 'TDC(0.0) $\\alpha$=0.002 $\\mu$=0.05': 'TDC',
        'TDC(0.0)-CO $\\alpha$=0.003 $\\mu$=0.05': 'TDC',
    },
    'disc_random_on': {
        'BBO': 'BBO',
        'BRM': 'BRM',
        # 'BRMDS': 257.107698326668,
        # 'FPKF(0.4) $\\alpha=0.5$ $\\beta=10.0$ m=1000': 646.5305920307188,
        # 'GTD $\\alpha$=0.007 $\\mu$=0.0001': 691.4351781077611,
        'GTD2 $\\alpha$=0.003 $\\mu$=4': 'GTD2',
        # 'LSPE(0.0) $\\alpha$=0.1': 43.95466968492962,
        'LSTD(0.0) $\\epsilon$=10': 'LSTD (w/ reg.)',
        'LSTD(0.0) $\\epsilon$=nan': 'LSTD (w/o reg.)',
        # 'LSTD-l1(0) $\\tau=3e-05$': 24.65568442748072,
        # 'LarsTD(0) $\\tau=0.05$': 110.65670906153218,
        # 'RG $\\alpha$=0.001': 716.0848732008917,
        # 'RG DS $\\alpha$=0.006': 374.7352608851127,
        # 'TD(0.0) $\\alpha$=RMAlpha(0.09, 0.25)': 'TD',
        'TD(0.0) $\\alpha$=auto': 'TD',
        # 'TD(0.4) $\\alpha$=0.001': 'TD',
        'TDC(0.0) $\\alpha$=0.007 $\\mu$=0.01': 'TDC',
    },
    'link20_imp_offpolicy': {
        'BBO': 'BBO',
        'BRM': 'BRM',
        # 'BRMDS': 7.671062088137126,
        # 'FPKF(0.0) $\\alpha$=0.3 $\\beta$=10.0 m=0.0': 8.082961359694492,
        # 'GTD $\\alpha$=0.003 $\\mu$=16.0': 7.076109450078713,
        'GTD2 $\\alpha$=0.5 $\\mu$=0.01': 'GTD2',
        # 'GeriTDC(0.0) $\\alpha$=0.06 $\\mu$=0.05': 5.491190942223792,
        # 'LSPE(0.0) $\\alpha$=0.01': 8.082953203055869,
        # 'LSPE(0.0)-CO $\\alpha$=0.5': 5.357474509572003,
        # 'LSTD(0.0) $\\epsilon$=100': 8.082953064992447,
        'LSTD(0.0)-CO $\\epsilon$=10': 'LSTD (w/ reg.)',
        'LSTD(0.0) $\\epsilon$=nan': 'LSTD (w/o reg.)',
        # 'RG $\\alpha$=0.04': 7.778383923537741,
        # 'RG DS $\\alpha$=0.05': 7.656775335176597,
        'TD(0.0) $\\alpha$=0.05': 'TD',
        # 'TD(0.0) $\\alpha$=RMAlpha(0.7, 0.25)': 'TD',
        'TDC(0.0) $\\alpha$=0.05 $\\mu$=0.01': 'TDC',
    },
    'link20_imp_onpolicy': {
        'BBO': 'BBO',
        'BRM': 'BRM',
        # 'BRMDS': 7.448412809781189,
        # 'FPKF(0.2) $\\alpha$=0.0005': 7.3029097916889505,
        # 'GTD $\\alpha$=0.0005 $\\mu$=2.0': 4.285696333769541,
        'GTD2 $\\alpha$=0.0005 $\\mu$=1.0': 'GTD2',
        # 'LSPE(0.0) $\\alpha$=0.01': 4.263460657693718,
        'LSTD(0.0) $\\epsilon$=0.01': 'LSTD (w/ reg.)',
        'LSTD(0.0) $\\epsilon$=nan': 'LSTD (w/o reg.)',
        # 'RG $\\alpha$=0.003': 7.5978241252189935,
        # 'RG DS $\\alpha$=0.0005': 7.436269227775181,
        'TD(0.0) $\\alpha$=0.0005': 'TD',
        # 'TD(0.0) $\\alpha$=RMAlpha(0.06, 0.5)': 4.300875206896804,
        'TDC(0.0) $\\alpha$=0.0005 $\\mu$=0.05': 'TDC',
    },
    'lqr_full_offpolicy': {
        'BBO': 'BBO',
        'BRM': 'BRM',
        # 'BRMDS': 155.45184006601957,
        # 'FPKF(0.4) $\\alpha$=1.0': 251.37685915148825,
        # 'GTD $\\alpha$=0.001 $\\mu$=0.0001': 328.66228835433805,
        'GTD2 $\\alpha$=0.001 $\\mu$=1.0': 'GTD2',
        # 'LSPE(0.0) $\\alpha$=0.001': 181.05705256200315,
        # 'LSPE(0.0)-TO $\\alpha$=1.0': 41673.59399231003,
        # 'LSTD(0.0) $\\epsilon$=0.01': 274.21751459268353,
        'LSTD-TO(0.0) $\\epsilon$=10': 'LSTD (w/ reg.)',
        'LSTD(0.0) $\\epsilon$=nan': 'LSTD (w/o reg.)',
        # 'RG $\\alpha$=0.005': 232.04381572684196,
        # 'RG DS $\\alpha$=0.008': 163.44589125291552,
        # 'TD(0.0) $\\alpha$=RMAlpha(0.03, 0.25)': 'TD',
        'TD(0.2) $\\alpha$=0.002': 'TD',
        'TDC(0.0)-TO $\\alpha$=0.003 $\\mu$=0.1': 'TDC',
        # 'TDC(0.2) $\\alpha$=0.002 $\\mu$=0.05': 'TDC',
    },
    'lqr_full_onpolicy': {
        'BBO': 'BBO',
        'BRM': 'BRM',
        # 'BRMDS': 42.83952114202071,
        # 'FPKF(0.8) $\\alpha$=0.5': 28.732748800889972,
        # 'GTD $\\alpha$=0.0005 $\\mu$=0.001': 163.8951239774888,
        'GTD2 $\\alpha$=0.005 $\\mu$=0.5': 'GTD2',
        # 'LSPE(0.0) $\\alpha$=0.1': 13.695198417158917,
        'LSTD(0.0) $\\epsilon$=10': 'LSTD (w/ reg.)',
        'LSTD(0.0) $\\epsilon$=nan': 'LSTD (w/o reg.)',
        # 'RG $\\alpha$=0.01': 115.33997003792499,
        # 'RG DS $\\alpha$=0.02': 73.41508560079738,
        # 'TD(0.0) $\\alpha$=RMAlpha(0.01, 0.05)': 'TD',
        'TD(0.2) $\\alpha$=0.008': 'TD',
        'TDC(0.0) $\\alpha$=0.007 $\\mu$=0.05': 'TDC',
    },
    'lqr_imp_offpolicy': {
        'BBO': 'BBO',
        'BRM': 'BRM',
        # 'BRMDS': 386.22685365261174,
        # 'FPKF(0.2) $\\alpha$=0.3': 334.5631079389631,
        # 'GTD $\\alpha$=0.002 $\\mu$=0.1': 346.31730709792305,
        'GTD2 $\\alpha$=0.01 $\\mu$=0.1': 'GTD2',
        # 'LSPE(0.0) $\\alpha$=0.001': 332.5518261902775,
        # 'LSPE(0.0)-CO $\\alpha$=0.7': 254.9669229432863,
        # 'LSTD(0.0) $\\epsilon$=0.01': 3218.2376092405575,
        'LSTD-CO(0.0) $\\epsilon$=100000': 'LSTD (w/ reg.)',
        'LSTD(0.0) $\\epsilon$=nan': 'LSTD (w/o reg.)',
        # 'RG $\\alpha$=0.005': 462.20928631270107,
        # 'RG DS $\\alpha$=0.006': 417.41225318123463,
        'TD(0.0) $\\alpha$=0.002': 'TD',
        # 'TD(0.0) $\\alpha$=RMAlpha(0.03, 0.25)': 'TD',
        'TDC(0.0) $\\alpha$=0.002 $\\mu$=0.0001': 'TDC',
    },
    'lqr_imp_onpolicy': {
        'BBO': 'BBO',
        # 'BRM': 'BRM',
        'BRM(0.8)': 'BRM',
        # 'BRMDS': 116.87172908094459,
        # 'FPKF(0.0) $\\alpha$=0.3 $\\beta=100.0$': 83.52330303265077,
        # 'GTD $\\alpha$=0.009 $\\mu$=0.1': 88.75582867874991,
        'GTD2 $\\alpha$=0.02 $\\mu$=0.1': 'GTD2',
        # 'LSPE(0.0) $\\alpha$=0.9': 81.55938521992812,
        'LSTD(0.0) $\\epsilon$=100000': 'LSTD (w/ reg.)',
        'LSTD(0.0) $\\epsilon$=nan': 'LSTD (w/o reg.)',
        # 'RG $\\alpha$=0.06': 134.75149275892065,
        # 'RG DS $\\alpha$=0.06': 116.61673543972402,
        'TD(0.0) $\\alpha$=0.004': 'TD',
        # 'TD(0.0) $\\alpha$=RMAlpha(0.04, 0.25)': 'TD',
        'TDC(0.0) $\\alpha$=0.004 $\\mu$=0.0001': 'TDC',
    },
}


all_final_rmse_data = {}
all_sum_rmse_data = {}
all_results = {}

for task in tasks:
    data = load_results(task)
    rmse_index = data['criteria'].index('RMSE')

    task_final_rmse_data = {}
    task_sum_rmse_data = {}

    for method, mean, std in zip(data['methods'], data['mean'], data['std']):
        method_label = TASK_METHOD_TO_LABEL_MAP[task].get(method.name, None)
        if method_label is None:
            continue

        # if task != 'link20_imp_onpolicy':
        #     continue
        # method_label = method.name

        final_rmse = mean[rmse_index][-1]
        sum_rmse = np.sum(mean[rmse_index])

        if (method_label in task_final_rmse_data
            or method_label in task_sum_rmse_data):
            if method.name != 'TDC(0.0) $\\alpha$=0.002 $\\mu$=0.0001':
                raise ValueError(task, method_label)

        if (task == 'lqr_imp_offpolicy'
            and method_label == 'TDC(0.0) $\\alpha$=0.002 $\\mu$=0.0001'
            and type(method).__name__ == 'GeriTDCLambda'):
            continue

        task_final_rmse_data[method_label] = final_rmse
        task_sum_rmse_data[method_label] = sum_rmse

    all_final_rmse_data[task] = task_final_rmse_data
    all_sum_rmse_data[task] = task_sum_rmse_data

pprint(all_final_rmse_data)
pprint(all_sum_rmse_data)


def format_cell(x):
    if 1e10 < x:
        return '$> 10^{10}$'
    elif 1e3 < x:
        return '$> 10^{3}$'
    return f'{x:.2f}'


# def highlight_max(s):
#     is_max = s == s.max()
#     return ['background-color: red' if v else '' for v in is_max]

# final_rmse_dataframe.style.apply(highlight_max)


final_rmse_dataframe = pd.DataFrame.from_dict(
    all_final_rmse_data, orient='index')
sum_rmse_dataframe = pd.DataFrame.from_dict(
    all_sum_rmse_data, orient='index')

final_rmse_dataframe.index = final_rmse_dataframe.index.map(TASK_TO_INDEX_MAP)
sum_rmse_dataframe.index = sum_rmse_dataframe.index.map(TASK_TO_INDEX_MAP)

to_latex_final_rmse_dataframe = final_rmse_dataframe.applymap(format_cell)
to_latex_sum_rmse_dataframe = sum_rmse_dataframe.applymap(format_cell)

for index, row in final_rmse_dataframe.iterrows():
    min_index = row.round(2) == row.round(2).min()
    to_latex_final_rmse_dataframe.loc[index, min_index] = (
        to_latex_final_rmse_dataframe.loc[index, min_index]
        .apply(lambda x: f'\textbf{{{x}}}'))


for index, row in sum_rmse_dataframe.iterrows():
    min_index = row.round(2) == row.round(2).min()
    to_latex_sum_rmse_dataframe.loc[index, min_index] = (
        to_latex_sum_rmse_dataframe.loc[index, min_index]
        .apply(lambda x: f'\textbf{{{x}}}'))


print(final_rmse_dataframe)
print(sum_rmse_dataframe)

WARN_BAIRD = "\kh{Probably want to drop the Baird experiment since its solution is the same as the prior for many of the methods.} "

FINAL_RMSE_CAPTION = (
    f"MSE of final predictions. The values for all methods"
    " except for BBO are obtained with code provided by~\cite{dann2014policy}."
)

SUM_RMSE_CAPTION = (
    f"Sum of square root MSE over all timesteps. The values for all methods"
    " except for BBO are obtained with code provided by~\cite{dann2014policy}."
)

print(to_latex_final_rmse_dataframe.to_latex(
    escape=False, caption=FINAL_RMSE_CAPTION))
print(to_latex_sum_rmse_dataframe.to_latex(
    escape=False, caption=SUM_RMSE_CAPTION))

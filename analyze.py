import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

metric_options = ["robust_val", "subopt_prop"]
metric_name = metric_options[1]
total_trials = 500

def reject_outliers(data, m=2):
    # print(data[abs(data - np.mean(data)) > m * np.std(data)])
    return data[abs(data - np.mean(data)) < m * np.std(data)]

results_dir = "/home/yppatel/fusioncp/results"
for task_name in ["gaussian_linear", "gaussian_linear_uniform", "gaussian_mixture", "two_moons"]:
    trial_inds = []
    for trial_idx in range(total_trials):
        fn = f"/home/yppatel/fusioncp/results/{task_name}/nominal_{trial_idx}.csv"
        if os.path.exists(fn):
            if float(pd.read_csv(fn).columns[0]) < 0:
                trial_inds.append(trial_idx)
    
    methods = ["nominal", "score_1", "score_2", "sum", "mvcp"]
    df = pd.DataFrame(index=np.arange(total_trials), columns=methods)

    table_line = ""
    # method_to_metric = {}
    for trial_idx in trial_inds:
        nominal_val = float(pd.read_csv(f"/home/yppatel/fusioncp/results/{task_name}/nominal_{trial_idx}.csv").columns[0])
        for method_name in methods:
            # if method_name not in method_to_metric:
            #     method_to_metric[method_name] = []
            if os.path.exists(f"/home/yppatel/fusioncp/results/{task_name}/{method_name}_{trial_idx}.csv"):
                method_val = float(pd.read_csv(f"/home/yppatel/fusioncp/results/{task_name}/{method_name}_{trial_idx}.csv").columns[0])
                if metric_name == "robust_val":
                    metric = method_val
                elif metric_name == "subopt_prop":
                    metric = abs((method_val - nominal_val) / nominal_val)
                df[method_name][trial_idx] = metric
    df_filtered = df[(df["score_1"] < 1) | (df["score_2"] < 1)]

    print(f"----- {task_name} -----")
    print(methods)
    table_line = ""
    for method_name in methods:
        data = df_filtered[method_name]
        if len(data) > 0:
            table_line += f"{np.nanmean(data).round(decimals=3)} ({np.nanstd(data).round(decimals=3)})"
        table_line += " | "
    print(table_line)

    for base in ["score_1", "score_2"]:
        _, ax = plt.subplots(1, 1)
        for method_name in ["score_1", "score_2", "sum", "mvcp"]:
            if method_name == base:
                continue
            diff = df_filtered[method_name] - df_filtered[base]
            z = np.mean(diff) / np.std(diff)
            print(f"{method_name} : z={z} -- p={norm.cdf(z)}")
            sns.kdeplot(diff, label=method_name, ax=ax)
        ax.legend()
        plt.savefig(f"results/{task_name}.png")
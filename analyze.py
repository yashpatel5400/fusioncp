import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

metric_options = ["robust_val", "subopt_prop"]
metric_name = metric_options[1]
total_trials = 100

def reject_outliers(data, m=2):
    # print(data[abs(data - np.mean(data)) > m * np.std(data)])
    return data[abs(data - np.mean(data)) < m * np.std(data)]

results_dir = "/home/yppatel/fusioncp/results"
for task_name in os.listdir(results_dir):
    trial_inds = []
    for trial_idx in range(total_trials):
        fn = f"/home/yppatel/fusioncp/results/{task_name}/nominal_{trial_idx}.csv"
        if os.path.exists(fn):
            if float(pd.read_csv(fn).columns[0]) < 0:
                trial_inds.append(trial_idx)
    
    methods = ["nominal", "score_1", "score_2", "sum", "mvcp"]

    table_line = ""
    method_to_metric = {}
    for trial_idx in trial_inds:
        nominal_val = float(pd.read_csv(f"/home/yppatel/fusioncp/results/{task_name}/nominal_{trial_idx}.csv").columns[0])
        for method_name in methods:
            if method_name not in method_to_metric:
                method_to_metric[method_name] = []
            if os.path.exists(f"/home/yppatel/fusioncp/results/{task_name}/{method_name}_{trial_idx}.csv"):
                method_val = float(pd.read_csv(f"/home/yppatel/fusioncp/results/{task_name}/{method_name}_{trial_idx}.csv").columns[0])
                if metric_name == "robust_val":
                    metric = method_val
                elif metric_name == "subopt_prop":
                    metric = abs((method_val - nominal_val) / nominal_val)
                method_to_metric[method_name].append(metric)
        
    for method_name in methods:
        metric_vals = np.array(method_to_metric[method_name])
        unfiltered_len = len(metric_vals)
        method_to_metric[method_name] = reject_outliers(metric_vals) # TODO: why are there random trials (usually one or two?) that have much higher values?
        # plt.hist(metric_vals, label=method_name)
        
        if method_name == "nominal":
            continue
            print(f"{method_name} | {np.mean(metric_vals).round(decimals=3)} | {np.std(metric_vals).round(decimals=3)} |")
        else:
            if method_name == "mvcp":
                table_line += r"& \textbf{" + f"{np.nanmean(metric_vals).round(decimals=3)} ({np.nanstd(metric_vals).round(decimals=3)})" + r"}"
            else:
                table_line += f"& {np.nanmean(metric_vals).round(decimals=3)} ({np.nanstd(metric_vals,).round(decimals=3)}) "
    
    _, ax = plt.subplots(1, 1)
    for method_name in methods:
        sns.kdeplot(method_to_metric[method_name], label=method_name, ax=ax)
    ax.legend()
    plt.savefig(f"results/{task_name}.png")
    print(f"{task_name} : {table_line}")
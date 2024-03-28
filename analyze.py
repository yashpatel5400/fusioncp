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

methods = ["nominal", "score_1", "score_2", "sum", "mvcp"]
results_dir = "/home/yppatel/fusioncp/results"
task_names = sorted([task_name for task_name in os.listdir(results_dir) 
                    if os.path.isdir(os.path.join(results_dir, task_name))
                    and "_old" not in task_name])
for task_name in task_names:
    trial_inds = []
    for trial_idx in range(total_trials):
        fn = f"/home/yppatel/fusioncp/results/{task_name}/nominal_{trial_idx}.csv"
        if os.path.exists(fn):
            if float(pd.read_csv(fn).columns[0]) < 0:
                trial_inds.append(trial_idx)
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

    table_line = " ".join([word.capitalize() for word in task_name.split("_")]) + " & "
    for method_name in methods:
        if method_name == "nominal":
            continue

        data = df_filtered[method_name]
        data = data[data == data]
        if len(data) > 0:
            if method_name != "mvcp":
                table_line += f"{np.nanmean(data).round(decimals=3)} ({np.nanstd(data).round(decimals=3)}) & "
            else:
                table_line += r"\textbf{" + f"{np.nanmean(data).round(decimals=3)} ({np.nanstd(data).round(decimals=3)})" + r"} \\"
    print(table_line)
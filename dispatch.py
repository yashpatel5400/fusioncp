# VERY hacky script but hey, gets the job done
import os

batch_script = """#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=mvcp
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=05:10:00
#SBATCH --account=tewaria0
#SBATCH --partition=standard

# The application(s) to execute along with its input arguments and options:
LD_PRELOAD=/home/yppatel/anaconda3/envs/chig/lib/libstdc++.so.6 python /home/yppatel/fusioncp/pred_opt.py --tasks {} --models {} --trial {} --fusion {}
"""

dispatch_scripts_dir = "dispatch_jobs"
total_trials = 500

task_to_models = {
    "lotka_volterra": "lotka_volterra_0-10.nf,lotka_volterra_10-20.nf",
    "sir": "sir_0-5.nf,sir_5-10.nf",
    "slcp": "slcp_0-2.nf,slcp_2-4.nf",
    "gaussian_linear": "gaussian_linear_0-5.nf,gaussian_linear_5-10.nf",
}

for task_name in ["lotka_volterra"]:
    for method_name in ["score_1", "score_2", "sum", "mvcp"]:
        for trial_idx in range(total_trials):
            dispatch_fn = os.path.join(dispatch_scripts_dir, f"dispatch_{method_name}_{trial_idx}.sh")
            with open(dispatch_fn, "w") as f:
                f.write(batch_script.format(task_name, task_to_models[task_name], trial_idx, method_name))
            
            print(f"Dispatched : $ sbatch {dispatch_fn}")
            os.system(f"sbatch {dispatch_fn}")
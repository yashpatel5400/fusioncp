# VERY hacky script but hey, gets the job done
import os

batch_script = """#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=mvcp
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=05:10:00
#SBATCH --account=tewaria0
#SBATCH --partition=standard

# The application(s) to execute along with its input arguments and options:
LD_PRELOAD=/home/yppatel/anaconda3/envs/chig/lib/libstdc++.so.6 python /home/yppatel/fusioncp/pred_opt.py --task {} --trial {} --fusion {}
"""

task_name = "slcp"
dispatch_scripts_dir = "dispatch_jobs"
total_trials = 10
for method_name in ["nominal", "score_1", "score_2", "sum", "mvcp"]:
    for trial_idx in range(total_trials):
        dispatch_fn = os.path.join(dispatch_scripts_dir, f"dispatch_{method_name}_{trial_idx}.sh")
        with open(dispatch_fn, "w") as f:
            f.write(batch_script.format(task_name, trial_idx, method_name))
        
        print(f"Dispatched : $ sbatch {dispatch_fn}")
        os.system(f"sbatch {dispatch_fn}")
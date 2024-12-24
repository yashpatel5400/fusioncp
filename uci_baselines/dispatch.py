# VERY hacky script but hey, gets the job done
import libtmux
from uci_datasets import all_datasets

task_names = [(name, n_observations, n_dimensions) for name, (n_observations, n_dimensions) in all_datasets.items() if n_observations > 1000]

# header = (
#     "\\begin{tabular}{ccc}\n"
#     "\\toprule\n"
#     r"Dataset & Observations & Input Dimension \\"
#     "\\midrule"
# )

# body = "\n"
# for task_desc in task_names:
#     task_name, n_observations, n_dimensions = task_desc
#     body += f"{task_name} & {n_observations} & {n_dimensions} \\\\ \n"
# footer = "\\bottomrule\n\\end{tabular}"
# print(header + body + footer)
# exit()

server = libtmux.Server()

for task_desc in task_names:
    task_name, _, _ = task_desc
    
    server.new_session(attach=False)
    session = server.sessions[-1]
    p = session.attached_pane
    p.send_keys("conda activate spectral", enter=True)
    cmd = f"python uci_tasks.py --task {task_name}"
    p.send_keys(cmd, enter=True)
    print(f"{cmd}")
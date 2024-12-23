# VERY hacky script but hey, gets the job done
import libtmux
from uci_datasets import all_datasets

task_names = [name for name, (n_observations, n_dimensions) in all_datasets.items() if n_observations > 1000]
server = libtmux.Server()

for task_name in task_names:
    server.new_session(attach=False)
    session = server.sessions[-1]
    p = session.attached_pane
    p.send_keys("conda activate spectral", enter=True)
    cmd = f"python uci_tasks.py --task {task_name}"
    p.send_keys(cmd, enter=True)
    print(f"{cmd}")
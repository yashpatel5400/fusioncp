# VERY hacky script but hey, gets the job done
import libtmux
import openml
from openml.tasks import TaskType
from uci_datasets import all_datasets

server = libtmux.Server()

suite_ids = [353]
benchmark_task_ids = []
for suite_id in suite_ids:
    benchmark_task_ids += openml.study.get_suite(suite_id).tasks

for task_name in range(10):
    for alpha in [0.025]:
        server.new_session(attach=False)
        session = server.get_by_id(f'${len(server.sessions)-1}')
        p = session.attached_pane
        p.send_keys("conda activate spectral", enter=True)
        cmd = f"python uci_tasks.py --task {task_name} --alpha {alpha}"
        p.send_keys(cmd, enter=True)
        print(f"{cmd}")

# task_names = [(name, n_observations, n_dimensions) for name, (n_observations, n_dimensions) in all_datasets.items() if n_observations > 1000]

# for task_desc in task_names:
#     task_name, _, _ = task_desc
    
#     server.new_session(attach=False)
#     session = server.get_by_id(f'${len(server.sessions)-1}')
#     p = session.attached_pane
#     p.send_keys("conda activate spectral", enter=True)
#     cmd = f"python uci_tasks.py --task {task_name} --alpha 0.05"
#     p.send_keys(cmd, enter=True)
#     print(f"{cmd}")
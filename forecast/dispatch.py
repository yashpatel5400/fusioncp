# VERY hacky script but hey, gets the job done
import libtmux
from uci_datasets import all_datasets

server = libtmux.Server()

methods = ["mvcp","score_1","score_2"]
for method in methods:
    for test_idx in range(80,110):
        server.new_session(attach=False)
        session = server.get_by_id(f'${len(server.sessions)-1}')
        p = session.attached_pane
        p.send_keys("conda activate spectral", enter=True)
        cmd = f"python pred_opt.py --method {method} --idx {test_idx}"
        p.send_keys(cmd, enter=True)
        print(f"{cmd}")
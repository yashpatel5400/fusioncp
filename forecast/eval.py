import os
import numpy as np
from rsome import ro
from rsome import grb_solver as grb
from scipy.stats import ttest_ind, ttest_rel

import osmnx as ox
import networkx as nx

def nominal_solve(c, A, b):
    model = ro.Model()

    w = model.dvar(A.shape[-1])

    model.min(c @ w)
    model.st(w <= 1)
    model.st(w >= 0)
    model.st(A @ w == b)

    model.solve(grb)
    return w.get()

def get_problem_setup():
    G = ox.graph_from_place("Manhattan, New York City, New York", network_type="drive")

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    edges = ox.graph_to_gdfs(G, nodes=False)
    edges["highway"] = edges["highway"].astype(str)
    edges.groupby("highway")[["length", "speed_kph", "travel_time"]].mean().round(1)

    hwy_speeds = {"residential": 35, "secondary": 50, "tertiary": 60}
    G = ox.add_edge_speeds(G, hwy_speeds=hwy_speeds)
    G = ox.add_edge_travel_times(G)

    A = nx.incidence_matrix(G, oriented=True).todense()
    b = np.zeros(len(G.nodes)) # b entries: 1 for source, -1 for target, 0 o.w.
    b[8]   = -1
    b[4350] = 1
    return G, A, b

def main():
    N_cal = 200
    G, A, b = get_problem_setup()
    
    c_fn = os.path.join("traffic", "truths.npy")
    cs = np.array(np.load(c_fn))[N_cal:,0]

    method_results = {
        "nominal": [],
        "score_1": [],
        "score_2": [],
        "mvcp": [],
    }

    solve_nominal = True
    if solve_nominal:
        for test_idx in range(80,110):
            w_star = nominal_solve(cs[test_idx], A, b)
            np.save(os.path.join("results", "nominal", f"{test_idx}.npy"), w_star)

    for test_idx in range(110):
        if test_idx in [16,23,92,95]:
            continue

        for method in method_results:
            w_star = np.load(os.path.join("results", method, f"{test_idx}.npy"))
            if method == "mvcp":
                w_star = w_star[0]
            method_opt = w_star @ cs[test_idx]
            method_results[method].append(method_opt)
        print(f"{method_results['mvcp'][-1]} | {method_results['score_1'][-1]} | {method_results['score_2'][-1]}")

    normalized = {}
    for method in method_results:
        if method == "nominal":
            continue
        normalized[method] = (np.array(method_results[method]) - np.array(method_results["nominal"])) / np.array(method_results["nominal"])
        print(f'{method} -- {np.mean(normalized[method])} ({np.std(normalized[method])})')

    # Perform the t-test
    print(normalized["mvcp"])
    print(normalized["score_1"])
    print(normalized["score_2"])
    _, p_value_1 = ttest_rel(normalized["mvcp"], normalized["score_1"], alternative='less')
    _, p_value_2 = ttest_rel(normalized["mvcp"], normalized["score_2"], alternative='less')
    _, p_value_3 = ttest_rel(normalized["score_1"], normalized["score_2"], alternative='less')
    print(f"p-value: {p_value_1}")
    print(f"p-value: {p_value_2}")
    print(f"p-value: {p_value_3}")

if __name__ == "__main__":
    main()
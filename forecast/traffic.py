import os
import numpy as np
from rsome import ro
from rsome import grb_solver as grb
from scipy.stats import ttest_ind

import osmnx as ox
import networkx as nx

def forecast_to_traffic(G, weathers):
    threshold_mmh = 0.1
    weathers[weathers < threshold_mmh] = 0

    x_coords = np.array(list(nx.get_node_attributes(G, "x").values()))
    y_coords = np.array(list(nx.get_node_attributes(G, "y").values()))

    x_coord_range = (np.min(x_coords), np.max(x_coords))
    y_coord_range = (np.min(y_coords), np.max(y_coords))

    # get rain at each node
    scaled_x = np.clip(((x_coords - x_coord_range[0]) / (x_coord_range[1] - x_coord_range[0])) * weathers.shape[1], 0, weathers.shape[1] - 1).astype(int)
    scaled_y = np.clip(((y_coords - y_coord_range[0]) / (y_coord_range[1] - y_coord_range[0])) * weathers.shape[2], 0, weathers.shape[2] - 1).astype(int)
    scaled_coords = np.vstack([scaled_x, scaled_y]).T

    rain_node_predictions = weathers[:,scaled_coords[:,0],scaled_coords[:,1]]

    # average between node endpoints to get rain along edges
    nodes_idx_to_idx = dict(zip(list(G.nodes), range(len(G.nodes))))
    edges = np.array([(nodes_idx_to_idx[e[0]], nodes_idx_to_idx[e[1]]) for e in list(G.edges)])
    rain_edge_predictions = np.concatenate([
        np.expand_dims(rain_node_predictions[:,edges[:,0]], axis=-1), 
        np.expand_dims(rain_node_predictions[:,edges[:,1]], axis=-1),
    ], axis=-1)

    travel_time = np.array(list(nx.get_edge_attributes(G, name="travel_time").values()))
    forecast_traffic = travel_time * np.exp(np.mean(rain_edge_predictions, axis=-1))

    return forecast_traffic

def get_graph():
    G = ox.graph_from_place("Manhattan, New York City, New York", network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    edges = ox.graph_to_gdfs(G, nodes=False)
    edges["highway"] = edges["highway"].astype(str)
    edges.groupby("highway")[["length", "speed_kph", "travel_time"]].mean().round(1)

    hwy_speeds = {"residential": 35, "secondary": 50, "tertiary": 60}
    G = ox.add_edge_speeds(G, hwy_speeds=hwy_speeds)
    G = ox.add_edge_travel_times(G)
    return G

def compute_traffic(method_fn):
    c_fn = os.path.join("precipitations", method_fn)
    G = get_graph()

    precip = np.nan_to_num(np.load(c_fn))
    if len(precip.shape) == 4: # truth does not have samples
        precip = np.expand_dims(precip, axis=1)

    traffic_samples = []
    for precip_sample in precip:
        traffic_samples.append(forecast_to_traffic(G, precip_sample[:,-1]))
    traffic_samples = np.array(traffic_samples)

    result_fn = os.path.join("traffic", method_fn)
    np.save(result_fn, traffic_samples)

if __name__ == "__main__":
    method_fns = ["truths.npy", "prediff.npy", "prediff_unaligned.npy", "steps.npy"]
    for method_fn in method_fns:
        compute_traffic(method_fn)
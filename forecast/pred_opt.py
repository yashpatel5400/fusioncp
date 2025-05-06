import argparse
import numpy as np
import matplotlib.pyplot as plt
from rsome import ro
from rsome import grb_solver as grb
import rsome as rso
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import cvxpy as cp
import osmnx as ox
import networkx as nx

import os
import pickle
import sys, os
import multiprocessing

import random
import einops

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

device = ("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------- MVCP ------------------------------------------------ #
def gpcp(c, c_hat):
    _, J, _ = c_hat.shape
    c_tiled = np.transpose(np.tile(c, (J, 1, 1)), (1, 0, 2))
    c_diff = c_hat - c_tiled
    c_norms = np.linalg.norm(c_diff, axis=-1)
    return np.min(c_norms, axis=-1)


def get_mvcp_dirs(k, M=1_000):
    directions = []
    for m in range(M):
        angle = (2 * np.pi / 4) * m / M
        direction   = np.array([np.cos(angle), np.sin(angle)])
        directions.append(direction)
    return np.array(directions).T


def get_mvcp_quantile(scores, proj_dirs, alpha):
    # scores: N x k; proj_dirs: k x M
    N_cal_shape = int(len(scores) * .2)
    
    _, k = scores.shape
    proj_scores = scores @ proj_dirs
    proj_scores_shape, proj_scores_quantile = proj_scores[:N_cal_shape], proj_scores[N_cal_shape:]

    # step 1: compute quantile profile with S_C^1
    beta_range = [1-alpha, 1-alpha/(2 * k)]
    coverage = 0.0
    eps = .01

    beta_prev = None
    while not (1-alpha <= coverage and coverage <= 1-alpha + eps):
        beta = (beta_range[0] * .8 + beta_range[1] * .2)
        mvcp_proj_quantiles = np.quantile(proj_scores_shape, q=beta, axis=0)
        train_covered = einops.reduce(proj_scores_shape < mvcp_proj_quantiles, "n p -> n", "prod")
        coverage = np.sum(train_covered) / len(train_covered)

        beta_prev = beta
        if coverage < 1-alpha:
            beta_range[0] = beta
        else:
            beta_range[1] = beta
        if beta_prev == beta:
            break
        print(f"{beta} -- {coverage}")

    # step 2: adjust size with S_C^2
    ts = einops.reduce(proj_scores_quantile / mvcp_proj_quantiles, "n m -> n", "max")
    t_star = np.quantile(ts, 1 - alpha)
    return t_star * mvcp_proj_quantiles


def inner_opt(args):
    w, directions, conformal_quantiles, c_region_centers = args
    N, K, d = np.shape(c_region_centers) # c_region_centers now corresponds to a single vec(j), so in batch_size x K x c_dim
    M       = len(directions)
    
    # Construct the problem.
    c = cp.Variable((N, d))
    objective = cp.Maximize(cp.sum(cp.multiply(c, w)))
    constraints = []
    
    for i in range(N): # iterates over batch size to optimize in parallel
        for m in range(M): # number of constraints is discretizations used for ellipse
            # TODO: manually adding constraints this way bakes in 2-view form: can generalize, but fine for now
            if len(directions[m]) == 1:
                constraints.append(cp.norm(c[i] - c_region_centers[i][0], 2) * directions[m][0] <= conformal_quantiles[m])
            elif len(directions[m]) == 2:
                constraints.append(
                    cp.norm(c[i] - c_region_centers[i][0], 2) * directions[m][0] + 
                    cp.norm(c[i] - c_region_centers[i][1], 2) * directions[m][1] <= conformal_quantiles[m])
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return c.value


# generative model-based prediction regions
def mvcp(mvcp_quantiles, mvcp_directions, c_region_centers, A, b):
    # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)

    N, J, K, d = c_region_centers.shape
    Js = cartesian_product(*([
        np.array(list(range(1))),
        np.array(list(range(4))), 
    ]))

    eta = 5e-3 # learning rate
    T = 10 # optimization steps

    pool = multiprocessing.Pool(J)
    w = np.random.random((N, d)) / 2
    
    # start optimization loop
    for t in range(T):
        raw_maxes = list(pool.map(inner_opt, [(w, mvcp_directions, mvcp_quantiles, c_region_centers[:,j,np.arange(c_region_centers.shape[2]),:]) for j in Js]))
        maxizer_per_region = np.transpose(np.array([maximizer for maximizer in raw_maxes if maximizer is not None]), (1, 0, 2))
        w_tiled = np.transpose(np.tile(w, (maxizer_per_region.shape[1], 1, 1)), (1, 0, 2))
        
        opt_value_per_region = np.sum((maxizer_per_region * w_tiled), axis=-1)
        opt_value = np.max(opt_value_per_region, axis=-1)        
        c_star_idx = np.argmax(opt_value_per_region, axis=-1)        
        c_star = maxizer_per_region[np.arange(maxizer_per_region.shape[0]), c_star_idx] # yuck, is this really the best way?
        
        grad = grad_f(w, c_star)
        w_temp = (w - eta * grad).flatten()

        # projection step: there's probably a better way of doing this?
        model = ro.Model()
        w_d = model.dvar(w_temp.shape)

        model.min(rso.sumsqr(w_d - w_temp))
        model.st(w_d <= 1)
        model.st(w_d >= 0)
        model.st((A @ w_d[i * d:(i+1) * d] == b) for i in range(N))

        model.solve(grb)
        
        w = w_d.get()
        w = w.reshape((N, d))

        print(f"Completed step={t} -- {opt_value}")
    pool.close()
    return w, np.max(opt_value)

# ---------------------------------------------------------------------------------

# *marginal* box constraint (i.e. just ignore contextual information)
def nominal_solve(cal_true_traffic, cal_pred_traffics, alpha, test_pred_traffic, A, b, norm):
    model = ro.Model()

    w = model.dvar(A.shape[-1])
    print(cal_true_traffic.shape)

    model.min(cal_true_traffic @ w)
    model.st(w <= 1)
    model.st(w >= 0)
    model.st(A @ w == b)

    model.solve(grb)
    return 1, model.get()


# current f = w^T c --> grad_w(f) = c
def grad_f(w, c):
    return c


# generative model-based prediction regions
def cpo(conformal_quantile, test_pred_traffic, A, b):
    eta = 5e-3 # learning rate
    T = 10 # optimization steps
    w = np.random.random(A.shape[-1]) / 2
    
    opt_values = []
    for t in range(T):
        maxizer_per_region = []
        opt_value = []

        for c_region_center in test_pred_traffic:
            model = ro.Model()
            c = model.dvar(A.shape[-1])

            model.max(c @ w)
            model.st(rso.norm(c - c_region_center, 2) <= conformal_quantile)
            model.solve(grb)

            maxizer_per_region.append(c.get())
            opt_value.append(model.get())

        opt_values.append(np.max(opt_value))
        c_star = maxizer_per_region[np.argmax(opt_value)]
        grad = grad_f(w, c_star)
        w_temp = w - eta * grad

        # projection step: there's probably a better way of doing this?
        model = ro.Model()
        w_d = model.dvar(A.shape[-1])

        model.min(rso.norm(w_d - w_temp, 2))
        model.st(w_d <= 1)
        model.st(w_d >= 0)
        model.st(A @ w_d == b)

        model.solve(grb)

        w = w_d.get()

        print(f"Completed step={t} -- {opt_value}")
    return w, np.max(opt_value)


def get_calibration_quantiles(c, c_hats, alpha=0.05, N_cal=300):
    cal_scores  = []
    test_scores = []
    
    for c_hat in c_hats:
        scores = gpcp(c, c_hat)
        cur_cal_scores, cur_test_scores = scores[:N_cal], scores[N_cal:]
        
        cal_scores.append(cur_cal_scores)
        test_scores.append(cur_test_scores)
        
    cal_scores  = einops.rearrange(cal_scores, "k n -> n k")
    test_scores = einops.rearrange(test_scores, "k n -> n k")
    
    _, k = cal_scores.shape    
    score1_quantile = np.quantile(cal_scores[:,0], q = 1 - alpha)
    score2_quantile = np.quantile(cal_scores[:,1], q = 1 - alpha)
    proj_dirs = get_mvcp_dirs(k, M=25)
    mvcp_quantiles = get_mvcp_quantile(cal_scores, proj_dirs, alpha)

    in_region = lambda scores : einops.reduce(scores @ proj_dirs < mvcp_quantiles, "n m -> n", "prod")
    score1_coverage = np.mean(test_scores[:,0] < score1_quantile)
    score2_coverage = np.mean(test_scores[:,1] < score2_quantile)
    mvcp_coverage   = np.mean(in_region(test_scores))
    print(f"Score 1: {score1_coverage} | Score 2: {score2_coverage} | MVCP: {mvcp_coverage}")

    return score1_quantile, score2_quantile, mvcp_quantiles, proj_dirs


def main(c_fn, c_hat_fns, calibration_method, test_idx):
    # --------- Calibration --------- # 
    N_cal = 200
    c = np.load(c_fn)[:,0]
    c_hats = [np.load(c_hat_fn) for c_hat_fn in c_hat_fns]
    score1_quantile, score2_quantile, mvcp_quantiles, mvcp_proj_dirs = get_calibration_quantiles(c, c_hats, N_cal=N_cal)
    mvcp_proj_dirs = einops.rearrange(mvcp_proj_dirs, "k m -> m k")
    
    # -------- Problem setup -------- # 
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

    # -------- Solve problem -------- #
    test_preds = np.array([c_hat[N_cal:] for c_hat in c_hats])
    test_preds = einops.rearrange(np.array(test_preds), "k n j d -> n j k d")
    
    J = 1

    test_batch_size = 1
    if calibration_method == "score_1":
        w_star, _ = cpo(score1_quantile, test_preds[test_idx,:1,0], A, b)
    elif calibration_method == "score_2":
        w_star, _ = cpo(score2_quantile, test_preds[test_idx,:,1], A, b)
    elif calibration_method == "mvcp":
        w_star, _ = mvcp(mvcp_quantiles, mvcp_proj_dirs, test_preds[test_idx * test_batch_size:(test_idx+1) * test_batch_size], A, b)

    results_dir = os.path.join("results", calibration_method)
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f"{test_idx}.npy"), w_star)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method")
    parser.add_argument("--idx", type=int)
    args = parser.parse_args()
    
    main(os.path.join("traffic", "truths.npy"), 
         [os.path.join("traffic", "prediff.npy"), os.path.join("traffic", "steps.npy")],
         args.method, args.idx)
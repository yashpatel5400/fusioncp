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
import sbibm
import argparse
import cvxpy as cp

import os
import pickle
import sys, os
import multiprocessing

import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(FeedforwardNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

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

def get_data(prior, simulator, N_test):
    # used for setting up dimensions
    sample_c = prior.sample((1,))
    sample_x = simulator(sample_c)

    N = 2_000
    N_train = 1000

    c_dataset = prior.sample((N,))
    x_dataset = simulator(c_dataset)

    to_tensor = lambda r : torch.tensor(r).to(torch.float32).to(device)
    x_train, x_cal = to_tensor(x_dataset[:N_train]), to_tensor(x_dataset[N_train:])
    c_train, c_cal = to_tensor(c_dataset[:N_train]), to_tensor(c_dataset[N_train:])

    c_test = prior.sample((N_test,))
    x_test = simulator(c_test)
    x_test, c_test = to_tensor(x_test), to_tensor(c_test)

    return (x_train, x_cal, x_test), (c_train, c_cal, c_test)


# current f = -w^T c --> grad_w(f) = -c
def grad_f(w, c):
    return -c


def inner_opt(args):
    w, c_shape, directions, conformal_quantiles, c_region_centers = args
    N, K, _ = np.shape(c_region_centers) # c_region_centers now corresponds to a single vec(j), so in batch_size x K x c_dim
    M       = len(directions)
    
    # Construct the problem.
    c = cp.Variable(c_shape)
    objective = cp.Maximize(-cp.sum(cp.multiply(c, w)))
    constraints = []
    
    for i in range(N): # iterates over batch size to optimize in parallel
        for m in range(M): # number of constraints is discretizations used for ellipse
            # TODO: manually adding constraints this way bakes in 2-view form: can generalize, but fine for now
            constraints.append(
                cp.norm(c[i] - c_region_centers[i][0], 2) * directions[m][0] + 
                cp.norm(c[i] - c_region_centers[i][1], 2) * directions[m][1] <= conformal_quantiles[m])
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return c.value


# generative model-based prediction regions
def mvcp(generative_models, view_dims, alpha, x_cal, c_cal, x_true, c_true, p, B):
    def comp_gpcp_scores(x, c, J = 10):
        scores, samples = [], []
        for generative_model, view_dim in zip(generative_models, view_dims):
            c_hat = generative_model.sample(J, x[:,view_dim[0]:view_dim[1]]).detach().cpu().numpy()
            c_tiled = np.transpose(np.tile(c.detach().cpu().numpy(), (J, 1, 1)), (1, 0, 2))
            c_diff = c_hat - c_tiled
            c_norms = np.linalg.norm(c_diff, axis=-1)
            scores.append(np.min(c_norms, axis=-1))
            samples.append(c_hat)
            print(c_hat.shape)
        return np.array(scores).T, np.transpose(np.array(samples), (1, 0, 2, 3))
    
    def get_mvcp_quantile(J = 10):
        scores, _ = comp_gpcp_scores(x_cal, c_cal, J)
        if len(generative_models) == 1:
            return np.quantile(proj_scores[:,0], q = 1 - alpha)

        beta_int = [0, 2 * alpha]
        coverage = -1
        tolerance = 0.01

        while np.abs(coverage - (1 - alpha)) > tolerance:
            beta = (beta_int[0] + beta_int[1]) / 2

            M = 10 # number of angle discretizations
            directions, quantiles = [], []
            for m in range(M):
                angle = (2 * np.pi / 4) * m / M
                direction   = np.array([np.cos(angle), np.sin(angle)])
                proj_scores = np.dot(scores, direction)
                quantile    = np.quantile(proj_scores, q = 1 - beta)
                directions.append(direction)
                quantiles.append(quantile)
            directions, quantiles = np.array(directions), np.array(quantiles)

            proj_cal_scores = np.array([np.dot(scores, direction) for direction in directions]).T
            coverage = np.sum(np.all(proj_cal_scores < quantiles, axis=1)) / len(proj_cal_scores)
            if coverage > (1 - alpha):
                beta_int[0] = beta
            else:
                beta_int[1] = beta
        return directions, quantiles
    
    # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)

    J = 10
    directions, quantiles = get_mvcp_quantile(J)
    c_score, c_region_centers = comp_gpcp_scores(x_true, c_true)
    proj_score = np.array([np.dot(c_score, direction) for direction in directions]).T
    contained = np.sum(np.all(proj_score < quantiles, axis=1)) / len(c_score)
    print(f"Contained: {contained}")

    N, K, J, d = c_region_centers.shape
    Js = cartesian_product(*([np.array(list(range(J)))] * K))
    
    eta = 5e-3 # learning rate
    T = 2_000 # optimization steps

    pool = multiprocessing.Pool(J)

    w = np.random.random(c_true.shape) / 2
    
    # start optimization loop
    for t in range(T):
        # c_region_centers is batch_size x K x J x c_dim
        raw_maxes = list(pool.map(inner_opt, [(w, c_true.shape, directions, quantiles, c_region_centers[:,np.arange(2),j,:]) for j in Js]))
        maxizer_per_region = np.transpose(np.array([maximizer for maximizer in raw_maxes if maximizer is not None]), (1, 0, 2))
        w_tiled = np.transpose(np.tile(w, (maxizer_per_region.shape[1], 1, 1)), (1, 0, 2))
        
        opt_value_per_region = np.sum(-(maxizer_per_region * w_tiled), axis=-1)
        opt_value = np.max(opt_value_per_region, axis=-1)        
        c_star_idx = np.argmax(opt_value_per_region, axis=-1)        
        c_star = maxizer_per_region[np.arange(maxizer_per_region.shape[0]), c_star_idx] # yuck, is this really the best way?
        
        # c_star = maxizer_per_region[np.argmax(opt_value)]
        grad = grad_f(w, c_star)
        w_temp = (w - eta * grad).flatten()

        # projection step: there's probably a better way of doing this?
        model = ro.Model()
        w_d = model.dvar(w_temp.shape)

        model.min(rso.sumsqr(w_d - w_temp))
        model.st(w_d <= 1)
        model.st(w_d >= 0)

        model.st(p[i] @ w_d[i * p.shape[-1]:(i+1) * p.shape[-1]] <= B[i] for i in range(len(B)))
        blockPrint()
        model.solve(grb)
        enablePrint()

        w = w_d.get()
        w = w.reshape(c_true.shape)

        print(f"Completed step={t} -- {opt_value}")
    pool.close()
    return contained, np.max(opt_value)


def run_mvcp(exp_config, task_name, trial_idx):
    x_train, x_cal, x_test = exp_config["x_train"], exp_config["x_cal"], exp_config["x_test"]
    c_train, c_cal, c_test = exp_config["c_train"], exp_config["c_cal"], exp_config["c_test"]
    ps = exp_config["ps"]
    Bs = exp_config["Bs"]

    cached_dir = "trained"
    trained_model_names = os.listdir(cached_dir)
    model_names = [trained_model_name for trained_model_name in trained_model_names if trained_model_name.startswith(task_name)]
    view_dims = [[int(dim) for dim in model_name.split("_")[-1].split(".")[0].split("-")] for model_name in model_names]

    generative_models = []
    for model_name in model_names:
        cached_fn = os.path.join(cached_dir, model_name)
        with open(cached_fn, "rb") as f:
            generative_model = pickle.load(f)
        generative_model.to(device)
        generative_models.append(generative_model)

    result_dir = os.path.join("results", task_name)
    os.makedirs(result_dir, exist_ok=True)
    
    alphas = [0.05]
    name_to_method = {
        "MVCP": mvcp,
    }
    
    for method_name in name_to_method:
        print(f"Running: {method_name}")
        for alpha in alphas:
            covered = 0
            x = x_test[trial_idx:(trial_idx + 1)]
            c = c_test[trial_idx:(trial_idx + 1)]
            p =     ps[trial_idx:(trial_idx + 1)]
            B =     Bs[trial_idx:(trial_idx + 1)]
                
            (covered_trial, value_trial) = name_to_method[method_name](
                generative_models, view_dims, alpha, x_cal, c_cal, x, c, p, B
            )
            covered += covered_trial
            trial_df = pd.DataFrame([value_trial])
            trial_df.to_csv(os.path.join(result_dir, f"{method_name}_{trial_idx}.csv"), mode="a", index=False, header=False)


def generate_data(cached_fn, task_name):
    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    n_trials   = 10
    trial_size = 1
    N_test = n_trials * trial_size

    (x_train, x_cal, x_test), (c_train, c_cal, c_test) = get_data(prior, simulator, N_test=N_test)
    c_dataset = torch.vstack([c_train, c_cal]).detach().cpu().numpy() # for cases where only marginal draws are used, no splitting occurs
    
    # want these to be consistent for comparison between methods, so generate once beforehand
    ps = np.random.randint(low=0, high=1000, size=(N_test, c_dataset.shape[-1]))
    us = np.random.uniform(low=0, high=1, size=N_test)
    Bs = np.random.uniform(np.max(ps, axis=1), np.sum(ps, axis=1) - us * np.max(ps, axis=1))

    with open(cached_fn, "wb") as f:
        pickle.dump({
            "x_train" : x_train, 
            "x_cal"   : x_cal, 
            "x_test"  : x_test, 
            "c_train" : c_train, 
            "c_cal"   : c_cal,
            "c_test"  : c_test, 
            "ps"      : ps, 
            "Bs"      : Bs,
        }, f)


def main(args):
    data_cache = "exp_configs"
    cached_fn  = os.path.join(data_cache, args.task)
    if not os.path.exists(cached_fn):
        generate_data(cached_fn, args.task)
    with open(cached_fn, "rb") as f:
        exp_config = pickle.load(f)
    run_mvcp(exp_config, args.task, int(args.trial))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--trial")
    args = parser.parse_args()
    main(args)
    
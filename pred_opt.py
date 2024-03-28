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

def sample(priors, simulators, N):
    thetas, xs, view_dims = [], [], []
    for prior, simulator in zip(priors, simulators):
        theta = prior(num_samples=N)
        x = simulator(theta)

        thetas.append(theta)
        xs.append(x)
        view_dims.append(x.shape[-1])

    combination_method = "repeat"
    if combination_method is None or combination_method == "stack":
        theta, x = torch.hstack(thetas), torch.hstack(xs)
    elif combination_method == "sum":
        theta, x = (thetas[0][:,:2] + thetas[1][:,:2]), torch.hstack(xs)
    elif combination_method == "repeat":
        theta, x = theta, x # torch.tile(theta, (1,3)), torch.hstack(xs)
    
    proj_dim = 2 # to consider a projected, lower-dimensional version of the problem
    if proj_dim is not None:
        theta = theta[:,:proj_dim]
    
    return theta, x

def get_data(priors, simulators, N_test):
    N = 2_000
    N_train = 1000

    c_dataset, x_dataset = sample(priors, simulators, N)

    to_tensor = lambda r : torch.tensor(r).to(torch.float32).to(device)
    x_train, x_cal = to_tensor(x_dataset[:N_train]), to_tensor(x_dataset[N_train:])
    c_train, c_cal = to_tensor(c_dataset[:N_train]), to_tensor(c_dataset[N_train:])

    c_test, x_test = sample(priors, simulators, N_test)
    x_test, c_test = to_tensor(x_test), to_tensor(c_test)
    return (x_train, x_cal, x_test), (c_train, c_cal, c_test)


def nominal_solve(c_true, p, B):
    model = ro.Model()

    w = model.dvar(c_true.shape)
    c = c_true.detach().cpu().numpy()
    
    model.min(-(c * w).sum())
    model.st(w <= 1)
    model.st(w >= 0)
    model.st(p[i] @ w[i,:] <= B[i] for i in range(len(B)))
    
    # blockPrint()
    model.solve(grb)
    # enablePrint()
    
    w_star = w.get()
    optima = np.sum(-(c * w_star), axis=-1)
    return 1, optima


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
def mvcp(generative_models, view_dims, alpha, x_cal, c_cal, x_true, c_true, p, B, fusion_technique):
    def comp_gpcp_scores(x, c, J = 10):
        scores, samples = [], []
        for generative_model, view_dim in zip(generative_models, view_dims):
            c_hat = generative_model.sample(J, x[:,view_dim[0]:view_dim[1]]).detach().cpu().numpy()
            c_tiled = np.transpose(np.tile(c.detach().cpu().numpy(), (J, 1, 1)), (1, 0, 2))
            c_diff = c_hat - c_tiled
            c_norms = np.linalg.norm(c_diff, axis=-1)
            scores.append(np.min(c_norms, axis=-1))
            samples.append(c_hat)
        return np.array(scores).T, np.transpose(np.array(samples), (1, 0, 2, 3))
    
    # if directions are not specified, we sample M (fixed to 10 below) uniform angles from [0, 2*pi) 
    def get_mvcp_quantile(directions=None, J = 10):
        scores, _ = comp_gpcp_scores(x_cal, c_cal, J)
        if len(generative_models) == 1:
            return np.quantile(proj_scores[:,0], q = 1 - alpha)

        beta_int = [0, 2 * alpha]
        coverage = -1
        tolerance = 0.01

        while np.abs(coverage - (1 - alpha)) > tolerance:
            beta = (beta_int[0] + beta_int[1]) / 2

            if directions is None:
                directions = []
                M = 10 # number of angle discretizations
                for m in range(M):
                    angle = (2 * np.pi / 4) * m / M
                    direction   = np.array([np.cos(angle), np.sin(angle)])
                    directions.append(direction)

            quantiles = []    
            for direction in directions:
                proj_scores = np.dot(scores, direction)
                quantile    = np.quantile(proj_scores, q = 1 - beta)
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

    if fusion_technique   == "score_1": directions = [[1,0]]
    elif fusion_technique == "score_2": directions = [[0,1]]
    elif fusion_technique == "sum":     directions = [[np.cos(np.pi / 4), np.sin(np.pi / 4)]]
    elif fusion_technique == "mvcp":    directions = None

    J = 10
    directions, quantiles = get_mvcp_quantile(directions, J)
    c_score, c_region_centers = comp_gpcp_scores(x_true, c_true)
    proj_score = np.array([np.dot(c_score, direction) for direction in directions]).T
    contained = np.sum(np.all(proj_score < quantiles, axis=1)) / len(c_score)
    print(f"Contained: {contained}")

    N, K, J, d = c_region_centers.shape
    Js = cartesian_product(*([np.array(list(range(J)))] * K))
    
    eta = 5e-3 # learning rate
    T = 500 # optimization steps

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
        # blockPrint()
        model.solve(grb)
        # enablePrint()

        w = w_d.get()
        w = w.reshape(c_true.shape)

        print(f"Completed step={t} -- {opt_value}")
    pool.close()
    return contained, np.max(opt_value)


def run_mvcp(exp_config, task_name, trial_idx, trial_size, method_name):
    x_train, x_cal, x_test = exp_config["x_train"], exp_config["x_cal"], exp_config["x_test"]
    c_train, c_cal, c_test = exp_config["c_train"], exp_config["c_cal"], exp_config["c_test"]
    ps = exp_config["ps"]
    Bs = exp_config["Bs"]

    cached_dir = "trained_cpu"
    trained_model_names = os.listdir(cached_dir)
    # model_names = [trained_model_name for trained_model_name in trained_model_names if trained_model_name.startswith(task_name)]
    model_names = [f"{task_name}_0-1.nf", f"{task_name}_1-2.nf"]
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
    
    alpha = 0.05
    fusion_method_to_func = {
        "nominal": nominal_solve,
        "score_1": mvcp,
        "score_2": mvcp,
        "sum": mvcp,
        "mvcp": mvcp,
    }
    
    print(f"Running: {method_name}")
    covered = 0

    start_trial_idx = trial_size * trial_idx
    for trial_idx in range(start_trial_idx, start_trial_idx + trial_size):
        x = x_test[trial_idx:(trial_idx + 1)]
        c = c_test[trial_idx:(trial_idx + 1)]

        # fix problem specification (p, B here) other than c across trials to reduce unrelatedd variance 
        p = ps[0:(0 + 1)] # ps[trial_idx:(trial_idx + 1)]
        B = Bs[0:(0 + 1)] # Bs[trial_idx:(trial_idx + 1)] # 

        _, nominal_val = nominal_solve(c, p, B)
        if nominal_val >= 0: # only want to consider those problem setups where the solution is non-trivial (i.e. not just w^* = 0)
            continue
        
        nominal_df = pd.DataFrame([nominal_val])
        nominal_df.to_csv(os.path.join(result_dir, f"nominal_{trial_idx}.csv"), index=False, header=False)

        if method_name == "nominal":
            (covered_trial, value_trial) = (1, nominal_val)
        else:
            (covered_trial, value_trial) = fusion_method_to_func[method_name](
                generative_models, view_dims, alpha, x_cal, c_cal, x, c, p, B, method_name
            )
        covered += covered_trial
        trial_df = pd.DataFrame([value_trial])
        trial_df.to_csv(os.path.join(result_dir, f"{method_name}_{trial_idx}.csv"), index=False, header=False)


def generate_data(cached_fn, task_names):
    task_names = task_names.split(",")
    tasks      = [sbibm.get_task(task_name) for task_name in task_names]
    priors     = [task.get_prior() for task in tasks]
    simulators = [task.get_simulator() for task in tasks]

    n_trials   = 500
    trial_size = 1
    N_test = n_trials * trial_size

    (x_train, x_cal, x_test), (c_train, c_cal, c_test) = get_data(priors, simulators, N_test=N_test)
    c_dataset = torch.vstack([c_train, c_cal]).detach().cpu().numpy() # for cases where only marginal draws are used, no splitting occurs
    
    # want these to be consistent for comparison between methods, so generate once beforehand
    ps = np.random.randint(low=0, high=1000, size=(N_test, c_dataset.shape[-1]))
    us = np.random.uniform(low=0, high=1, size=N_test)
    Bs = np.random.uniform(np.max(ps, axis=1), np.sum(ps, axis=1) - us * np.max(ps, axis=1))

    with open(cached_fn, "wb") as f:
        pickle.dump({
            "x_train" : x_train.to("cpu"), 
            "x_cal"   : x_cal.to("cpu"), 
            "x_test"  : x_test.to("cpu"), 
            "c_train" : c_train.to("cpu"), 
            "c_cal"   : c_cal.to("cpu"),
            "c_test"  : c_test.to("cpu"), 
            "ps"      : ps, 
            "Bs"      : Bs,
        }, f)
    exit()


def main(args):
    data_cache = "exp_configs"
    os.makedirs(data_cache, exist_ok=True)
    cached_fn  = os.path.join(data_cache, args.tasks)
    if not os.path.exists(cached_fn):
        generate_data(cached_fn, args.tasks)
    with open(cached_fn, "rb") as f:
        exp_config = pickle.load(f)
    run_mvcp(exp_config, args.tasks, int(args.trial), 1, args.fusion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks")
    parser.add_argument("--trial")
    parser.add_argument("--fusion")
    args = parser.parse_args()
    main(args)

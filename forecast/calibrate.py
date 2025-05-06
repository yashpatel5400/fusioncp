import argparse
import einops
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# scikit-learn imports
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

np.set_printoptions(precision=3)

def gpcp(c, c_hat):
    _, J, _ = c_hat.shape
    c_tiled = np.transpose(np.tile(c, (J, 1, 1)), (1, 0, 2))
    c_diff = c_hat - c_tiled
    c_norms = np.linalg.norm(c_diff, axis=-1)
    return np.min(c_norms, axis=-1)


def get_mvcp_dirs(k, M=1_000):
    unnorm_dirs = np.abs(np.random.multivariate_normal(np.zeros(k), np.eye(k), size=M))
    return (unnorm_dirs / np.linalg.norm(unnorm_dirs, axis=1, keepdims=True)).T


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


def compute_mvcp_coverage(proj_dirs, mvcp_quantile, test_scores):
    in_region = lambda scores : einops.reduce(scores @ proj_dirs < mvcp_quantile, "n m -> n", "prod")
    mvcp_coverage = np.mean(in_region(test_scores))
    return mvcp_coverage


def main(c_fn, c_hat_fns):
    c = np.load(c_fn)[:,0]
    
    cal_scores = []
    test_scores = []

    N_cal = 300
    for c_hat_fn in c_hat_fns:
        c_hat = np.load(c_hat_fn)
        scores = gpcp(c, c_hat)
        cur_cal_scores, cur_test_scores = scores[:N_cal], scores[N_cal:]
        cal_scores.append(cur_cal_scores)
        test_scores.append(cur_test_scores)
    
    cal_scores = np.array(cal_scores).T
    test_scores = np.array(test_scores).T
    
    alpha = 0.05
    
    _, k = cal_scores.shape
    
    proj_dirs = get_mvcp_dirs(k, M=1_000)
    mvcp_quantile = get_mvcp_quantile(cal_scores, proj_dirs, alpha)
    mvcp_coverage = compute_mvcp_coverage(proj_dirs, mvcp_quantile, test_scores)


if __name__ == "__main__":
    # fns = ["prediff.npy", "steps.npy", "truths.npy"]
    main(os.path.join("traffic", "truths.npy"), 
         [os.path.join("traffic", "prediff.npy"), os.path.join("traffic", "steps.npy")])
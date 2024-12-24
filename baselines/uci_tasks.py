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

from uci_datasets import Dataset

np.set_printoptions(precision=3)

###############################################################################
# 1) Helper Functions (placeholders for the ones from utils.R in R)
###############################################################################

def lm_train(X, y):
    """Mimic a simple linear model (no CV)."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def lm_predict(model, X):
    return model.predict(X)

def lm_funs():
    return {
        'train': lm_train,
        'predict': lm_predict
    }

def lasso_train_cv(X, y):
    """
    Mimic lasso.funs(standardize=False, cv=True, cv.rule='min').
    For brevity, we do a naive approach: a single Lasso with alpha=0.01
    or do a small grid search. 
    In real usage, you might do cross-validation with e.g. LassoCV.
    """
    from sklearn.linear_model import LassoCV
    # A quick LassoCV example:
    model = LassoCV(cv=5, random_state=123, fit_intercept=True).fit(X, y)
    return model

def lasso_predict_cv(model, X):
    return model.predict(X)

def lasso_funs(standardize=False, cv=True, cv_rule="min"):
    # Return a minimal workable approach
    return {
        'train': lasso_train_cv,
        'predict': lasso_predict_cv
    }

def rf_train(X, y, ntree=100, mtry=None):
    """
    Mimic rf.funs(ntree=..., varfrac=1, etc.
    mtry in R is the number of variables at each split.
    scikit-learn calls it max_features.
    If varfrac=1 => max_features = X.shape[1].
    """
    if mtry is None:
        mtry = X.shape[1]
    model = RandomForestRegressor(
        n_estimators=ntree,
        max_features=mtry,
        random_state=123
    )
    model.fit(X, y)
    return model

def rf_predict(model, X):
    return model.predict(X)

def rf_funs(ntree=100, varfrac=1):
    """
    varfrac = fraction of total features
    """
    # mtry in R = #features per split. 
    # If varfrac=1 => mtry = all features
    def _train(X, y):
        mtry_ = int(varfrac * X.shape[1])
        return rf_train(X, y, ntree=ntree, mtry=mtry_)
    
    return {
        'train': _train,
        'predict': rf_predict
    }

def nnet_train(X, y, size=12, decay=1.0, maxit=2000, linout=True):
    """
    Emulate nnet. size => hidden_layer_sizes, 
    decay => alpha (L2 reg), linout => regression (identity) in R terms.
    """
    model = MLPRegressor(
        hidden_layer_sizes=(size,),
        alpha=decay,
        max_iter=maxit,
        activation='relu',      # approximate
        solver='adam',
        random_state=123
    )
    model.fit(X, y)
    return model

def nnet_predict(model, X):
    return model.predict(X)

def nnet_funs_factory(size=12, decay=1.0, maxit=2000):
    """Return a dictionary for training/predicting with a custom NN config."""
    return {
        'train': lambda X, y: nnet_train(X, y, size=size, decay=decay, maxit=maxit),
        'predict': nnet_predict
    }

###############################################################################
# Conformal placeholder
###############################################################################
def conformal_pred_split(X_train, y_train, 
                         X_cal, y_cal, 
                         X_test, y_test, alpha=0.05, 
                         train_fun=None, predict_fun=None, 
                         seed=None):
    """
    Here, we do:
      1) Train model
      2) residuals = y_cal - preds_cal
      3) q = quantile(|residuals|, 1-alpha)
      4) lo = pred_test - q
         up = pred_test + q
    If X_test has multiple rows, we return arrays of lo, up.
    """
    if seed is not None:
        np.random.seed(seed)
    model = train_fun(X_train, y_train)
    
    preds_cal = predict_fun(model, X_cal)
    cal_scores = np.abs(y_cal - preds_cal)
    q = np.quantile(cal_scores, 1 - alpha)

    test_preds = predict_fun(model, X_test)
    test_scores = np.abs(y_test - test_preds)

    # If X_test is 1D or single row, shape might be (). Convert to array
    test_preds = np.array(test_preds).ravel()
    lo = test_preds - q
    up = test_preds + q
    
    # Return a structure that has 'lo' and 'up' as arrays
    return {'lo': lo, 'up': up, 'scores': cal_scores, 'test_scores': test_scores, 'test_preds': test_preds}

###############################################################################
# Majority vote placeholder
###############################################################################
def majority_vote(M, w, rho=0.5):
    """
    M : 2D array of shape (k, 2) => each row is [lo, up]
    w : weights array of shape (k,)
    rho: threshold
    Returns a 2D array of intervals or None
    (This is a simplistic translation of your earlier 'majority_vote' logic.)
    """
    if len(M.shape) != 2 or M.shape[1] != 2:
        return None
    # unique sorted breakpoints
    all_vals = M.flatten()
    unique_breaks = np.unique(all_vals)
    unique_breaks.sort()
    
    intervals = []
    i = 0
    while i < len(unique_breaks) - 1:
        mid = 0.5*(unique_breaks[i] + unique_breaks[i+1])
        # compute coverage:
        coverage = 0
        for idx, (lo, up) in enumerate(M):
            if lo <= mid <= up:
                coverage += w[idx]
        if coverage > rho:
            start_ = unique_breaks[i]
            j = i
            cond = True
            while j < len(unique_breaks)-1 and cond:
                j += 1
                if j < len(unique_breaks)-1:
                    mid_j = 0.5*(unique_breaks[j] + unique_breaks[j+1])
                    c_j = 0
                    for idx2, (lo2, up2) in enumerate(M):
                        if lo2 <= mid_j <= up2:
                            c_j += w[idx2]
                    if c_j <= rho:
                        cond = False
            end_ = unique_breaks[j]
            intervals.append([start_, end_])
            i = j
        i += 1
    if len(intervals) == 0:
        return None
    return np.array(intervals)


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
    return mvcp_proj_quantiles, t_star * mvcp_proj_quantiles


def compute_mvcp_coverage_len(proj_dirs, mvcp_quantile, test_scores, test_preds):
    in_region = lambda scores : einops.reduce(scores @ proj_dirs < mvcp_quantile, "n m -> n", "prod")
    mvcp_lens = []
    for i in range(len(test_preds)):
        test_pred = test_preds[i]
        range_ = np.max(test_pred) - np.min(test_pred)
        delta  = range_ / 100
        candidate_y = np.arange(np.min(test_pred) - 5 * range_, np.max(test_pred) + 5 * range_, delta)

        tiled_pred = einops.repeat(test_pred, "n -> n s", s=len(candidate_y))
        candidate_scores = np.abs(tiled_pred - candidate_y).T
        scores_in_region = in_region(candidate_scores)
        
        mvcp_len = np.sum(scores_in_region) * delta / range_
        mvcp_lens.append(mvcp_len)
    mvcp_lens = np.array(mvcp_lens)
    mvcp_coverage = np.mean(in_region(test_scores))
    return mvcp_lens, mvcp_coverage


def prep_data(X, y):
    N = len(y)
    N_train = int(0.8 * N)
    N_cal   = int(0.1 * N)

    full_indices = np.arange(len(y))
    train_indices = np.random.choice(full_indices, size=N_train, replace=False)
    remaining_indices = np.setdiff1d(full_indices, train_indices)
    cal_indices = np.random.choice(remaining_indices, size=N_cal, replace=False)
    test_indices = np.setdiff1d(remaining_indices, cal_indices)
    
    Xtrain, ytrain = X[train_indices], y[train_indices].ravel()
    Xcal, ycal = X[cal_indices], y[cal_indices].ravel()
    Xtest, ytest = X[test_indices], y[test_indices].ravel()
    return (Xtrain, ytrain), (Xcal, ycal), (Xtest, ytest)


def trial(X, y):
    (Xtrain, ytrain), (Xcal, ycal), (Xtest, ytest) = prep_data(X, y)

    def final_nnet_train(x, y, **kwargs):
        return nnet_train(x, y, size=12, decay=1.0, maxit=2000, linout=True)
    
    def final_nnet_predict(model, X):
        return model.predict(X)
    
    nnet_funs = {
        'train': final_nnet_train,
        'predict': final_nnet_predict
    }
    
    def train_lm(X, y):
        lm_ = LinearRegression()
        lm_.fit(X, y)
        return lm_

    def predict_lm(m, X):
        return m.predict(X)
    
    def train_lasso_cv(X, y):
        from sklearn.linear_model import LassoCV
        model_ = LassoCV(cv=5, random_state=123).fit(X, y)
        return model_
    
    def predict_lasso_cv(m, X):
        return m.predict(X)
    
    lmF = {'train': train_lm, 'predict': predict_lm}
    lassoF = {'train': train_lasso_cv, 'predict': predict_lasso_cv}
    rfF = rf_funs(ntree=400, varfrac=1)

    funs = [lmF, lassoF, rfF, nnet_funs]
    
    # apply conformal
    alpha = 0.05
    conf_ints1 = []
    for f_ in funs:
        out_ = conformal_pred_split(
            Xtrain, ytrain, 
            Xcal, ycal, 
            Xtest, ytest,
            alpha=alpha,
            train_fun=f_['train'],
            predict_fun=f_['predict'],
        )
        conf_ints1.append(out_)
    
    test_scores = np.array([out['test_scores'] for out in conf_ints1]).T
    test_preds = np.array([out['test_preds'] for out in conf_ints1]).T

    stacked_scores = np.array([conf_int['scores'] for conf_int in conf_ints1]).T
    _, k = stacked_scores.shape
    proj_dirs = get_mvcp_dirs(k, M=1_000)
    
    # no_recal_mvcp_quantile is just used to demonstrate recalibration is necessary for coverage
    no_recal_mvcp_quantile, mvcp_quantile = get_mvcp_quantile(stacked_scores, proj_dirs, alpha)
    no_recal_mvcp_lens, no_recal_mvcp_coverage = compute_mvcp_coverage_len(proj_dirs, no_recal_mvcp_quantile, test_scores, test_preds)
    mvcp_lens, mvcp_coverage = compute_mvcp_coverage_len(proj_dirs, mvcp_quantile, test_scores, test_preds)

    coverage_bools = []
    for c_ in conf_ints1:
        coverage_each = ((c_['lo'] <= ytest) & (ytest <= c_['up'])).astype(float)
        coverage_bools.append(coverage_each)
    
    coverage_bools = np.column_stack(coverage_bools)
    method_cov = coverage_bools.mean(axis=0)
    
    # Randomized majority vote coverage
    # rowMeans coverage => fraction of methods covering each point
    row_means = coverage_bools.mean(axis=1)
    
    np.random.seed(1234)
    our_out1 = (row_means > 0.5).astype(float)
    u1 = np.random.uniform(low=0.0, high=0.5, size=len(row_means))
    our_out2 = (row_means > (0.5 + u1)).astype(float)
    u2 = np.random.uniform(low=0.0, high=1.0, size=len(row_means))
    our_out3 = (row_means > u2).astype(float)
    
    coverages_rand = [
        our_out1.mean(), 
        our_out2.mean(), 
        our_out3.mean()
    ]
    
    # Length of each method's intervals
    avg_lengths = []
    for c_ in conf_ints1:
        lengths_ = c_['up'] - c_['lo']
        avg_lengths.append(lengths_.mean())
        
    res_len = np.zeros((len(ytest), 3))
    res_dou = np.zeros((len(ytest), 3))
    
    # Build M: 4 intervals -> shape (4,2)
    w_ = np.full(4, 1.0/4)  # uniform weights for 4 methods
    
    for i in range(len(ytest)):
        M = np.zeros((4,2))
        for j in range(4):
            M[j,0] = conf_ints1[j]['lo'][i]
            M[j,1] = conf_ints1[j]['up'][i]
        
        # majority_vote with rho=0.5
        ci1 = majority_vote(M, w_, 0.5)
        if ci1 is not None:
            res_dou[i,0] = ci1.shape[0]
            # sum of intervals' lengths
            res_len[i,0] = sum(ci1[:,1] - ci1[:,0])
        
        # majority_vote with rho = 0.5 + u1[i]
        ci2 = majority_vote(M, w_, 0.5 + u1[i])
        if ci2 is not None:
            res_dou[i,1] = ci2.shape[0]
            res_len[i,1] = sum(ci2[:,1] - ci2[:,0])
        
        # majority_vote with rho = u2[i]
        ci3 = majority_vote(M, w_, u2[i])
        if ci3 is not None:
            res_dou[i,2] = ci3.shape[0]
            res_len[i,2] = sum(ci3[:,1] - ci3[:,0])
    
    # combine coverage
    coverages = np.concatenate([method_cov, coverages_rand, [no_recal_mvcp_coverage, mvcp_coverage]])
    # combine lengths
    # The R code does: avg_length of each method, plus colMeans(res_len)
    method_lengths = avg_lengths  # 4 methods
    # colMeans(res_len) => average interval size for each scenario
    # shape (len(ytest), 3)
    majority_lengths = res_len.mean(axis=0)
    avg_no_recal_mvcp_length = np.mean(no_recal_mvcp_lens)
    avg_mvcp_length = np.mean(mvcp_lens)
    lengths_combined = np.concatenate([method_lengths, majority_lengths, [avg_no_recal_mvcp_length, avg_mvcp_length]])
    
    return coverages, lengths_combined


###############################################################################
# Main script
###############################################################################
def main(task_name, X, y):
    np.random.seed(123)
    num_trials = 10
    trial_coverages, trial_lengths_combined = [], []
    for _ in range(num_trials):
        coverages, lengths_combined = trial(X, y)
        trial_coverages.append(coverages)
        trial_lengths_combined.append(lengths_combined)
    trial_coverages = np.array(trial_coverages)
    trial_lengths_combined = np.array(trial_lengths_combined)
    
    methods_name = [
        "Linear Model", "LASSO", "Random Forest", "Neural Net",
        r"\mathcal{C}^{M}", 
        r"\mathcal{C}^{R}",
        r"\mathcal{C}^{U}", 
        "DECP (Single-Stage)",
        "DECP",
    ]
    
    df_table = pd.DataFrame({
        "methods": methods_name,

        "mean_cov": np.mean(trial_coverages, axis=0),
        "std_cov": np.std(trial_coverages, axis=0),

        "mean_len": np.mean(trial_lengths_combined, axis=0),
        "std_len": np.std(trial_lengths_combined, axis=0),
    }).T

    df_table.loc["cov"] = df_table.loc["mean_cov"].astype(float).round(3).astype(str) + " (" + df_table.loc["std_cov"].astype(float).round(3).astype(str) + ")"
    df_table.loc["len"] = df_table.loc["mean_len"].astype(float).round(3).astype(str) + " (" + df_table.loc["std_len"].astype(float).round(3).astype(str) + ")"
    df_table.at["len", 8] = r"\textbf{" +  df_table.at["len", 8] + "}"

    latex_output = df_table.loc[["cov", "len"]].to_latex(float_format="{:.3f}".format)
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{task_name}.txt"), "w") as f:
        f.write(latex_output)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    args = parser.parse_args()

    data = Dataset(args.task)
    main(args.task, data.x, data.y)
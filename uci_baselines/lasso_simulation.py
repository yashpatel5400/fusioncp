import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import Lasso

##############################################################################
# Placeholder conformal prediction function
##############################################################################
def conformal_pred_split(X_train, y_train, X_test, alpha=0.05,
                         train_fun=None, predict_fun=None, seed=None):
    """
    Mimics R's conformalInference::conformal.pred.split().
    This is a placeholder. In practice, you can:
    - use MAPIE (https://github.com/scikit-learn-contrib/MAPIE)
    - implement split conformal logic manually.
    
    Parameters
    ----------
    X_train : array of shape (n, p)
    y_train : array of shape (n,)
    X_test  : array of shape (n_test, p)
    alpha   : significance level (1 - coverage)
    train_fun : callable that trains a model (X, y) -> model
    predict_fun : callable that predicts from model (model, X) -> predictions
    seed : random seed (if needed)
    
    Returns
    -------
    A dict with keys:
      'lo': float (lower bound),
      'up': float (upper bound)
    for each test point. Here we assume n_test=1. If >1, you'd return arrays.
    """
    if seed is not None:
        np.random.seed(seed)
    # Train the model
    model = train_fun(X_train, y_train)
    preds_train = predict_fun(model, X_train)
    # Compute residuals (naive approach)
    residuals = y_train - preds_train
    scores = np.abs(residuals)
    
    # quantile of absolute residuals
    q = np.quantile(scores, 1 - alpha)
    
    # Predict on test
    preds_test = predict_fun(model, X_test)
    # If X_test has shape (1, p), preds_test is a single float or shape (1,)
    # We'll assume n0=1 as in your R script. If more, adjust accordingly.
    if preds_test.shape == ():  # single float
        preds_test = np.array([preds_test])
    
    lo = preds_test[0] - q
    up = preds_test[0] + q
    
    return {'lo': lo, 'up': up, 'scores': scores}


##############################################################################
# Lasso "funs" (mimicking lasso.funs(...))
##############################################################################
def lasso_train(X, y, alpha=1.0, fit_intercept=False):
    """
    Train a Lasso model with no standardization (unless you do it outside)
    and possibly no intercept (fit_intercept=False).
    """
    model = Lasso(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X, y)
    return model

def lasso_predict(model, X):
    return model.predict(X)

def lasso_funs(lambda_val=1.0, standardize=False, intercept=False):
    """
    Returns a dict with 'train' and 'predict' to mimic R's:
      lasso.funs(lambda = lambda_val, standardize = F, intercept = F)
    """
    # In scikit-learn, set fit_intercept=not intercept
    # 'standardize' = F means we skip scaling here; user can scale externally if needed
    return {
        'train': lambda X, y: lasso_train(X, y, alpha=lambda_val, fit_intercept=intercept),
        'predict': lasso_predict
    }


##############################################################################
# Majority Vote & Exchangeable Majority Vote
# (Assuming you have them from previous translations; placeholders below.)
##############################################################################
def majority_vote(M, w, rho=0.5):
    """
    M : 2D array of shape (k, 2)
    w : 1D array of shape (k,) for weights
    rho: threshold
    Returns 2D array of intervals or None
    """
    # (Placeholder logic; adapt your previously translated code)
    # unique sorted breakpoints
    unique_breaks = np.unique(M.flatten())
    unique_breaks.sort()

    i = 0
    lower = []
    upper = []
    while i < len(unique_breaks) - 1:
        # counts_int-like check
        mid = 0.5 * (unique_breaks[i] + unique_breaks[i+1])
        # Weighted coverage:
        coverage_indicators = [(1 if row[0] <= mid <= row[1] else 0) for row in M]
        coverage = np.average(coverage_indicators, weights=w)
        cond = (coverage > rho)

        if cond:
            lower.append(unique_breaks[i])
            j = i
            while j < len(unique_breaks) - 1 and cond:
                j += 1
                if j < len(unique_breaks) - 1:
                    mid2 = 0.5 * (unique_breaks[j] + unique_breaks[j+1])
                    coverage2 = np.average(
                        [(1 if row[0] <= mid2 <= row[1] else 0) for row in M],
                        weights=w
                    )
                    cond = (coverage2 > rho)
            upper.append(unique_breaks[j])
            i = j
        i += 1

    if len(lower) == 0:
        return None
    return np.column_stack((lower, upper))

def exch_majority_vote(M, tau=0.5):
    """
    Exchangeable majority vote
    M: 2D array (k, 2)
    tau: threshold
    """
    # (Placeholder logic.)
    k = M.shape[0]
    if k == 1:
        return M
    perm_indices = np.random.permutation(k)
    permM = M[perm_indices, :]
    newM = [None]*k
    newM[0] = permM[0,:].reshape(1,2)
    for i in range(1, k):
        subset = permM[:i+1, :]
        w = np.full(i+1, 1/(i+1))
        mv = majority_vote(subset, w, rho=tau)
        if mv is None:
            return None
        newM[i] = mv
    
    # We gather the final intersection across newM
    # The R code's logic checks if intervals appear in *all* sets.
    # We'll do a simplified approach: take the union in newM and only keep those 
    # that appear in each list. 
    # For brevity, return one combined interval or None.

    # If you already have a fully working version from earlier translations, use that.
    # Here is a minimal approach to avoid an overly long snippet:
    if any(x is None for x in newM):
        return None
    # Flatten all intervals
    all_breaks = []
    for arr in newM:
        all_breaks.extend(arr.flatten())
    all_breaks = np.unique(all_breaks)
    all_breaks.sort()

    if len(all_breaks) < 2:
        return None

    # We'll just return everything as one big bracket or None
    return np.array([[all_breaks[0], all_breaks[-1]]])

def exch_rand_majority_vote(M):
    """
    Placeholder for exch_rand_majority_vote(cis),
    which is not defined in your snippet.
    
    We'll guess it does the same as exch_majority_vote but uses 
    a random threshold 'u' somewhere inside?
    """
    # Minimal stand-in:
    # e.g. pick a random threshold in [0.5,1], then call exch_majority_vote
    u = np.random.uniform(0.5, 1)
    return exch_majority_vote(M, tau=u)


##############################################################################
# Main simulation code (translated from your R script)
##############################################################################
def main():
    # simulation p > n
    # lasso with different lambda values
    n  = 100    # number of obs. in the train set
    n0 = 1      # number of obs. in the test set
    p  = 120    # number of predictors
    m  = 10     # number of active predictors

    # beta values
    np.random.seed(123)
    beta_vals = np.concatenate((np.random.normal(loc=0, scale=2, size=m), 
                                np.zeros(p - m)))

    # design matrix and outcome
    X = np.random.randn(n, p)
    y = X @ beta_vals + np.random.randn(n)

    # design matrix and outcome (test)
    X0 = np.random.randn(n0, p)
    y0 = X0 @ beta_vals + np.random.randn(n0)

    # lasso for all parameters
    lambda_vals = np.exp(np.linspace(-4, 1.5, 20))  # seq(-4, 1.5, length=20)
    k = len(lambda_vals)
    funs = []
    for lam in lambda_vals:
        funs.append(lasso_funs(lambda_val=lam, standardize=False, intercept=False))

    # Obtain a conformal prediction interval for each X0 with level (1-alpha/2)
    alpha = 0.1
    conf_pred_ints = []
    for f in funs:
        # X0 is shape (1, p), we assume n0=1
        out = conformal_pred_split(X, y, X0, alpha=alpha/2,
                                   train_fun=f['train'],
                                   predict_fun=f['predict'],
                                   seed=123)
        conf_pred_ints.append(out)

    us = 0.5

    # Check coverage: out -> booleans (lo <= y0 <= up)
    coverage_bools = []
    for cpi in conf_pred_ints:
        in_cov = (cpi['lo'] <= y0[0] <= cpi['up'])
        coverage_bools.append(in_cov)

    coverage_bools = np.array(coverage_bools, dtype=float)
    # rowMeans(out) in R is just the mean of coverage_bools if n0=1
    # This is a single value if n0=1
    vote = coverage_bools.mean()
    # If you had n0 > 1, you'd handle arrays instead.

    print("vote > 0.5:", (vote > 0.5))

    # Build cis matrix (k x 2)
    cis = np.zeros((k, 2))
    for i in range(k):
        cis[i, 0] = conf_pred_ints[i]['lo']
        cis[i, 1] = conf_pred_ints[i]['up']

    # various majority votes
    w = np.full(k, 1.0/k)
    cis_mv  = majority_vote(cis, w, rho=0.5)
    cis_rmv = majority_vote(cis, w, rho=0.5 + us/2)
    cis_u   = majority_vote(cis, w, rho=us)
    cis_exc = exch_majority_vote(cis)

    # For demonstration, we place them all in one array:
    # We'll label them by type.
    # The R code does some data frame manipulations for plotting in ggplot2
    # We'll mimic that logic in Python via pandas:
    all_cis_list = []
    # Original C_k intervals
    for i in range(k):
        all_cis_list.append([cis[i,0], cis[i,1], i+1, "a"])  # type "a"
    # MV
    if cis_mv is not None:
        for row in cis_mv:
            # in R code, it was "b" with "n" repeated
            all_cis_list.append([row[0], row[1], k+1, "b"])
    # RMV
    if cis_rmv is not None:
        for row in cis_rmv:
            all_cis_list.append([row[0], row[1], k+2, "c"])
    # U
    if cis_u is not None:
        for row in cis_u:
            all_cis_list.append([row[0], row[1], k+3, "d"])
    # EXC
    if cis_exc is not None:
        for row in cis_exc:
            all_cis_list.append([row[0], row[1], k+4, "e"])

    df_all = pd.DataFrame(all_cis_list, columns=["low","up","n","type"])
    print("All intervals:\n", df_all.head(25))

    # We won't replicate the exact ggplot calls in Python; 
    # here's a simple scatter + vertical line approach in matplotlib:
    colors_map = {"a":"black","b":"red","c":"blue","d":"forestgreen","e":"purple"}
    plt.figure(figsize=(8,5))
    for _, row in df_all.iterrows():
        n_ = row["n"]
        low_ = row["low"]
        up_  = row["up"]
        t_   = row["type"]
        c_   = colors_map.get(t_, "gray")
        # draw line
        plt.plot([n_, n_], [low_, up_], color=c_)
        # draw markers at endpoints
        plt.plot(n_, low_, marker="o", color=c_)
        plt.plot(n_, up_,  marker="o", color=c_)

    plt.xlabel("Tuning parameter index (or label)")
    plt.ylabel("Prediction Intervals")
    plt.title("Intervals for Each Lambda + Majority Votes")
    plt.show()

    #########################################################################
    # Simulation
    #########################################################################
    B = 1000
    # We'll store coverage results in res and interval counts in length
    # res has shape (B, 5)
    # len has shape (B, 5)
    res = np.zeros((B, 5))
    length_mat = np.zeros((B, 5))

    np.random.seed(123)
    for i in range(B):
        # design matrix and outcome
        X = np.random.randn(n, p)
        y = X @ beta_vals + np.random.randn(n)
        # test
        X0 = np.random.randn(n0, p)
        y0 = X0 @ beta_vals + np.random.randn(n0)

        # get intervals from each fun in funs
        conf_pred_ints_B = []
        for ff in funs:
            out_ = conformal_pred_split(X, y, X0, alpha=alpha/2,
                                        train_fun=ff['train'],
                                        predict_fun=ff['predict'])
            conf_pred_ints_B.append(out_)
        # build cis
        los = np.array([o['lo'] for o in conf_pred_ints_B])
        ups = np.array([o['up'] for o in conf_pred_ints_B])
        cis_B = np.column_stack((los, ups))

        # coverage booleans
        coverage_B = []
        for cpi in conf_pred_ints_B:
            coverage_B.append(1 if (cpi['lo'] <= y0[0] <= cpi['up']) else 0)
        coverage_B = np.array(coverage_B, dtype=float)
        vote_ = coverage_B.mean()

        # random thresholds:
        u1 = np.random.uniform(0.5, 1)
        u2 = np.random.uniform()

        # res[i,1] = I(vote > 0.5)
        res[i,0] = 1 if (vote_ > 0.5) else 0
        # res[i,2] = I(vote > u1)
        res[i,1] = 1 if (vote_ > u1) else 0
        # res[i,3] = I(vote > u2)
        res[i,2] = 1 if (vote_ > u2) else 0

        # majority votes
        ci_m = majority_vote(cis_B, np.ones(k), 0.5)
        ci_r = majority_vote(cis_B, np.ones(k), u1)
        ci_u = majority_vote(cis_B, np.ones(k), u2)

        # lengths: if None, 0 rows
        length_mat[i,0] = 0 if ci_m is None else ci_m.shape[0]
        length_mat[i,1] = 0 if ci_r is None else ci_r.shape[0]
        length_mat[i,2] = 0 if ci_u is None else ci_u.shape[0]

        # exch majority vote
        ci_e = exch_majority_vote(cis_B)
        if ci_e is None:
            res[i,3] = 0
            length_mat[i,3] = 0
        else:
            # coverage is how many intervals in ci_e contain y0
            # each row is [lo, up]
            cov_cnt = 0
            for row in ci_e:
                if row[0] <= y0[0] <= row[1]:
                    cov_cnt += 1
            res[i,3] = cov_cnt
            length_mat[i,3] = ci_e.shape[0]

        ci_er = exch_rand_majority_vote(cis_B)
        if ci_er is None:
            res[i,4] = 0
            length_mat[i,4] = 0
        else:
            cov_cnt = 0
            for row in ci_er:
                if row[0] <= y0[0] <= row[1]:
                    cov_cnt += 1
            res[i,4] = cov_cnt
            length_mat[i,4] = ci_er.shape[0]

        if (i+1) % 50 == 0:
            print(f"Iter: {i+1}")

    # colMeans(res); colMeans(len>1) in R
    # => we do res.mean(axis=0), (length_mat>1).mean(axis=0) in Python
    coverage_means = res.mean(axis=0)
    length_exceed_1 = (length_mat > 1).mean(axis=0)
    len_means = length_mat.mean(axis=0)

    print("Coverage means:", coverage_means)
    print("Length means:", len_means)
    print("Fraction of times length > 1:", length_exceed_1)


if __name__ == "__main__":
    main()

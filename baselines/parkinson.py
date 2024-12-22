import einops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# scikit-learn imports
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

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
    A naive stand-in for conformal.pred.split from R's conformalInference.
    If you want a real conformal approach, 
    use e.g. MAPIE or your own split-conformal logic.
    Here, we do:
      1) Train model
      2) residuals = y_train - preds_train
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
    _, k = scores.shape
    proj_scores = scores @ proj_dirs

    beta_range = [1-alpha, 1-alpha/k]
    coverage = 0.0
    eps = .005

    while not (1-alpha <= coverage and coverage <= 1-alpha + eps):
        beta = (beta_range[0] * .8 + beta_range[1] * .2)
        mvcp_proj_quantiles = np.quantile(proj_scores, q=beta, axis=0)
        train_covered = einops.reduce(proj_scores < mvcp_proj_quantiles, "n p -> n", "prod")
        coverage = np.sum(train_covered) / len(train_covered)
        if coverage < 1-alpha:
            beta_range[0] = beta
        else:
            beta_range[1] = beta
        print(f"{beta} -- {coverage}")
    return mvcp_proj_quantiles


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


###############################################################################
# Main script
###############################################################################
def main():
    # ------------------
    # Load data
    # ------------------
    # In R: read.table("merged_dataset.csv", ...)
    # Adjust path/sep as needed
    df = pd.read_csv("merged_dataset.csv", sep=",")
    
    # creation of the outcome and the X matrix
    y = df["total_UPDRS"].values
    # remove first two columns from X (like [-c(1,2)])
    X = df.drop(columns=["total_UPDRS", df.columns[0]]).copy()
    
    # scale columns 0..17 (like X[,1:18] in R, but be mindful of indexing)
    # If X has exactly 18 numeric columns to scale, do that:
    cols_to_scale = X.columns[:18]
    X[cols_to_scale] = (X[cols_to_scale] - X[cols_to_scale].mean()) / X[cols_to_scale].std(ddof=0)
    
    # train/test split
    np.random.seed(123)
    full_indices = np.arange(len(y))
    train_indices = np.random.choice(full_indices, size=4500, replace=False)
    remaining_indices = np.setdiff1d(full_indices, train_indices)
    cal_indices = np.random.choice(remaining_indices, size=500, replace=False)
    test_indices = np.setdiff1d(remaining_indices, cal_indices)
    
    ytrain = y[train_indices]
    Xtrain = X.iloc[train_indices].values  # turn to NumPy
    ycal = y[cal_indices]
    Xcal = X.iloc[cal_indices].values  # turn to NumPy
    y0 = y[test_indices]
    X0 = X.iloc[test_indices].values
    
    # histogram of ytrain
    plt.figure()
    plt.hist(ytrain, bins=10, color='lightblue', edgecolor='black')
    plt.title("Histogram of total UPDRS")
    plt.xlabel("total UPDRS")
    plt.ylabel("Freq")
    plt.show()
    
    # correlation plot using seaborn
    corr_mat = np.corrcoef(Xtrain, rowvar=False)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_mat, cmap='coolwarm')
    plt.title("Correlation matrix for Xtrain")
    plt.show()
    
    # We mimic R's approach:
    # find correlations > 0.975 in absolute value
    # Then remove columns "Jitter.RAP", etc. if they exist
    # We'll do a small example of how you might do it:
    corr_df = pd.DataFrame(corr_mat)
    # We'll just show how to find large correlations (similar to your approach).
    # In R code: filter cor < -0.975 | cor > 0.975
    # The user specifically removed c("Jitter.RAP", "Jitter.DDP", "Shimmer", "Shimmer.dB.", "Shimmer.APQ3").
    # We'll do the same if those columns exist:
    cols_to_remove = ["Jitter.RAP","Jitter.DDP","Shimmer","Shimmer.dB.","Shimmer.APQ3"]
    existing_cols_to_remove = [c for c in cols_to_remove if c in X.columns]
    # remove them
    if existing_cols_to_remove:
        Xtrain_df = pd.DataFrame(Xtrain, columns=X.columns)
        Xtrain_df.drop(columns=existing_cols_to_remove, inplace=True)
        Xcal_df = pd.DataFrame(Xcal, columns=X.columns)
        Xcal_df.drop(columns=existing_cols_to_remove, inplace=True)
        X0_df = pd.DataFrame(X0, columns=X.columns)
        X0_df.drop(columns=existing_cols_to_remove, inplace=True)
        
        Xtrain = Xtrain_df.values
        Xcal = Xcal_df.values
        X0 = X0_df.values
        # update X.columns if you want to keep track
        X = X.drop(columns=existing_cols_to_remove)
    
    # correlation again after removal (optional)
    corr_mat2 = np.corrcoef(Xtrain, rowvar=False)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_mat2, cmap='coolwarm')
    plt.title("Correlation matrix (after removal)")
    plt.show()
    
    # define final nnet.funs
    # from R code: nnet.train(x, y, size=12, decay=1, maxit=2000, linout=T)
    def final_nnet_train(x, y, **kwargs):
        return nnet_train(x, y, size=12, decay=1.0, maxit=2000, linout=True)
    
    def final_nnet_predict(model, X):
        return model.predict(X)
    
    nnet_funs = {
        'train': final_nnet_train,
        'predict': final_nnet_predict
    }
    
    # Conformal predictions
    # Build funs: 
    #   LM, Lasso(CV), RF(ntree=400, varfrac=1 => mtry=all), NNet
    def train_lm(X, y):
        lm_ = LinearRegression()
        lm_.fit(X, y)
        return lm_

    def predict_lm(m, X):
        return m.predict(X)
    
    lmF = {'train': train_lm, 'predict': predict_lm}
    
    def train_lasso_cv(X, y):
        from sklearn.linear_model import LassoCV
        model_ = LassoCV(cv=5, random_state=123).fit(X, y)
        return model_
    
    def predict_lasso_cv(m, X):
        return m.predict(X)
    
    lassoF = {'train': train_lasso_cv, 'predict': predict_lasso_cv}
    
    rfF = rf_funs(ntree=400, varfrac=1)  # mimic best found
    # nnet_funs is already defined
    
    funs = [lmF, lassoF, rfF, nnet_funs]
    
    # apply conformal
    split_ind = np.random.choice(np.arange(5000), size=2500, replace=False)
    alpha = 0.05
    conf_ints1 = []
    for f_ in funs:
        out_ = conformal_pred_split(
            Xtrain, ytrain, 
            Xcal, ycal, 
            X0, y0,
            alpha=alpha,
            train_fun=f_['train'],
            predict_fun=f_['predict'],
            seed=split_ind  # storing seed for demonstration
        )
        conf_ints1.append(out_)
    
    test_scores = np.array([out['test_scores'] for out in conf_ints1]).T
    test_preds = np.array([out['test_preds'] for out in conf_ints1]).T

    stacked_scores = np.array([conf_int['scores'] for conf_int in conf_ints1]).T
    _, k = stacked_scores.shape
    proj_dirs = get_mvcp_dirs(k, M=1_000)
    mvcp_quantile = get_mvcp_quantile(stacked_scores, proj_dirs, alpha)
    mvcp_lens, mvcp_coverage = compute_mvcp_coverage_len(proj_dirs, mvcp_quantile, test_scores, test_preds)

    # Coverage of each method
    # out[[i]] => c(lo <= y0 & y0 <= up)
    coverage_bools = []
    for c_ in conf_ints1:
        coverage_each = ((c_['lo'] <= y0) & (y0 <= c_['up'])).astype(float)
        coverage_bools.append(coverage_each)
    
    coverage_bools = np.column_stack(coverage_bools)
    # coverage_bools shape is (len(y0), 4)
    # colMeans in R => coverage_bools.mean(axis=0)
    method_cov = coverage_bools.mean(axis=0)
    print("Method-by-method coverage:", method_cov)

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
    print("Randomized coverage thresholds:", coverages_rand)
    
    # Length of each method's intervals
    avg_lengths = []
    for c_ in conf_ints1:
        lengths_ = c_['up'] - c_['lo']
        avg_lengths.append(lengths_.mean())
    
    # majority vote intervals
    # res_len => keep track of lengths
    # res_dou => keep track of number of intervals?
    res_len = np.zeros((len(y0), 3))
    res_dou = np.zeros((len(y0), 3))
    
    # Build M: 4 intervals -> shape (4,2)
    w_ = np.full(4, 1.0/4)  # uniform weights for 4 methods
    
    for i in range(len(y0)):
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
    coverages = np.concatenate([method_cov, coverages_rand, [mvcp_coverage]])
    # combine lengths
    # The R code does: avg_length of each method, plus colMeans(res_len)
    method_lengths = avg_lengths  # 4 methods
    # colMeans(res_len) => average interval size for each scenario
    # shape (len(y0), 3)
    majority_lengths = res_len.mean(axis=0)
    avg_mvcp_length = np.mean(mvcp_lens)
    lengths_combined = np.concatenate([method_lengths, majority_lengths, [avg_mvcp_length]])
    
    # fraction of times # intervals > 1
    frac_gt1 = (res_dou > 1).mean(axis=0)
    
    print("\nCoverage of 4 methods + 3 random thresholds:\n", coverages)
    print("Lengths of 4 methods + 3 majority-vote scenarios:\n", lengths_combined)
    print("Fraction (#intervals>1) for each majority vote scenario:", frac_gt1)
    
    methods_name = [
        "Linear Model", "Lasso (CV)", "Random Forest", "Neural Net",
        "Majority Vote (>0.5)", "Rand Majority Vote (>0.5 + u1)",
        "Rand Vote (>u2)", "MVCP"
    ]
    
    # Build final table
    # coverage => first 7
    # lengths => first 4 from methods, then 3 from majority
    df_table = pd.DataFrame({
        "Methods": methods_name,
        "Coverage": coverages,
        "Lengths": lengths_combined
    })
    
    print("\nFinal Table of Coverage / Length:\n", df_table)

if __name__ == "__main__":
    main()

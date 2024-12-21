import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# -------------------------------------------------------------
# nnet equivalents
# -------------------------------------------------------------
def nnet_train(X, y, **kwargs):
    """
    Trains a neural network model similar to R's nnet().
    Using scikit-learn's MLPRegressor as an approximate analog.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    kwargs : additional keyword arguments passed to MLPRegressor
    
    Returns
    -------
    model : fitted MLPRegressor
    """
    model = MLPRegressor(**kwargs)
    model.fit(X, y)
    return model

def nnet_preds(model, newX, **kwargs):
    """
    Generates predictions from the trained model, analogous to predict(out, newx, ...).
    
    Parameters
    ----------
    model : trained MLPRegressor
    newX : array-like of shape (n_samples, n_features)
    kwargs : additional predict args (ignored by MLPRegressor.predict)
    
    Returns
    -------
    preds : array-like
    """
    return model.predict(newX)

nnet_funs = {
    'train': nnet_train,
    'predict': nnet_preds
}

# -------------------------------------------------------------
# rpart equivalents
# -------------------------------------------------------------
def rpart_train(X, y, **kwargs):
    """
    Trains a decision tree model similar to R's rpart().
    Using scikit-learn's DecisionTreeRegressor as an approximate analog.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    kwargs : additional keyword arguments passed to DecisionTreeRegressor
    
    Returns
    -------
    model : fitted DecisionTreeRegressor
    """
    model = DecisionTreeRegressor(**kwargs)
    model.fit(X, y)
    return model

def rpart_preds(model, newX, **kwargs):
    """
    Generates predictions from the trained tree, analogous to predict(out, newx, ...).
    
    Parameters
    ----------
    model : trained DecisionTreeRegressor
    newX : array-like of shape (n_samples, n_features)
    kwargs : ignored
    """
    return model.predict(newX)

# -------------------------------------------------------------
# Majority Vote and related helpers
# -------------------------------------------------------------
def counts_int(M, a, b, w):
    """
    M : 2D NumPy array of shape (K, 2). Each row is [lo, up].
    a, b : floats, endpoints for checking midpoint
    w : 1D array of shape (K,) for weights
    
    Returns
    -------
    num_int : float, the weighted mean of coverage indicators
    """
    mid = 0.5 * (a + b)
    # Indicator: mid ∈ [lo, up]
    indicator = np.array([(1 if (row[0] <= mid <= row[1]) else 0) for row in M], dtype=float)
    # Weighted mean:
    num_int = np.average(indicator, weights=w)
    return num_int

def counts_set(M_list, a, b):
    """
    M_list : list of arrays, each array has shape (n_intervals, 2) or is None/NaN
    a, b   : floats
    
    Returns
    -------
    num_int : sum of coverage counts across all intervals in M_list
    """
    mid = 0.5 * (a + b)
    total_coverage = 0
    for Mi in M_list:
        if Mi is None:
            # Equivalent to an NA interval in R
            continue
        if np.any(np.isnan(Mi)):
            # If the entire set is "NA" in R
            continue
        
        # Count how many sub-intervals in Mi contain mid
        count_in = 0
        for row in Mi:
            if row[0] <= mid <= row[1]:
                count_in += 1
        total_coverage += count_in
    return total_coverage

def majority_vote(M, w, rho=0.5):
    """
    M : 2D array of shape (k, 2), each row [lo, up]
    w : 1D array of shape (k,) for weights
    rho : float, threshold (default 0.5)
    
    Returns
    -------
    intervals : 2D array (n_intervals, 2) or None
                The "majority vote" intervals, or None if none found.
    """
    # unique sorted breakpoints
    unique_breaks = np.unique(M.flatten())
    unique_breaks.sort()

    i = 0
    lower = []
    upper = []

    while i < len(unique_breaks) - 1:
        cond = (counts_int(M, unique_breaks[i], unique_breaks[i+1], w) > rho)
        if cond:
            # start interval
            lower.append(unique_breaks[i])
            j = i
            # keep going while condition is true
            while j < (len(unique_breaks) - 1) and cond:
                j += 1
                if j < (len(unique_breaks) - 1):
                    cond = (counts_int(M, unique_breaks[j], unique_breaks[j+1], w) > rho)
            upper.append(unique_breaks[j])
            i = j
        i += 1

    if len(lower) == 0:
        return None
    else:
        return np.column_stack((lower, upper))

def exch_majority_vote(M, tau=0.5):
    """
    Exchangeable majority vote:
    M   : 2D array (k, 2)
    tau : threshold (default=0.5)
    
    Returns
    -------
    intervals : 2D array or None
    """
    k = M.shape[0]
    if k == 1:
        return M

    # permute
    perm_indices = list(range(k))
    random.shuffle(perm_indices)
    permM = M[perm_indices, :]  # shape (k,2)

    # newM is a list of intervals (2D arrays), possibly None
    newM = [None] * k
    # newM[0] is just the single row (turn it into shape (1,2))
    newM[0] = permM[0, :].reshape(1, 2)

    # build up
    for i in range(1, k):
        # majority vote on first i+1 intervals
        subset = permM[:i+1, :]
        w = np.full(i+1, 1.0/(i+1))  # uniform weights
        mv = majority_vote(subset, w, rho=tau)
        if mv is None:
            return None
        newM[i] = mv

    # gather all intervals from newM
    all_breaks = []
    for intervals_i in newM:
        if intervals_i is None:
            continue
        all_breaks.extend(intervals_i.flatten())
    all_breaks = np.unique(all_breaks)
    all_breaks.sort()

    i = 0
    lower = []
    upper = []

    # The R code requires that the sub-interval is in *all* newM sets
    while i < len(all_breaks) - 1:
        c_ = counts_set(newM, all_breaks[i], all_breaks[i+1])
        cond = (c_ == k)
        if cond:
            lower.append(all_breaks[i])
            j = i
            while j < (len(all_breaks) - 1) and cond:
                j += 1
                if j < (len(all_breaks) - 1):
                    c_ = counts_set(newM, all_breaks[j], all_breaks[j+1])
                    cond = (c_ == k)
            upper.append(all_breaks[j])
            i = j
        i += 1

    if len(lower) == 0:
        return None
    else:
        return np.column_stack((lower, upper))

# -------------------------------------------------------------
# Loss and size utility functions
# -------------------------------------------------------------
def loss_fun(c, sets_list, a, b):
    """
    c : float (target point)
    sets_list : list of dicts or objects with 'lo' and 'up' attributes
    a, b : floats
    
    Returns
    -------
    loss_vec : 1D numpy array of length len(sets_list)
               = a * interval_size - b * indicator(c in interval)
    """
    size_int = []
    cov_int = []
    for s in sets_list:
        # s is something like {'lo':..., 'up':...}
        interval_size = s['up'] - s['lo']
        in_cov = 1.0 if (s['lo'] <= c <= s['up']) else 0.0
        size_int.append(interval_size)
        cov_int.append(in_cov)

    size_int = np.array(size_int)
    cov_int = np.array(cov_int)
    loss_vec = a * size_int - b * cov_int
    return loss_vec

def sizes(sets_list):
    """
    sets_list: list of dicts, each with {'lo': ..., 'up': ...}
    Return a NumPy array of interval sizes.
    """
    return np.array([s['up'] - s['lo'] for s in sets_list])

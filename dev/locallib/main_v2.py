from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from scipy.stats import norm
from sklearn.utils import check_X_y, check_array
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold
import warnings
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted

class BoostedOrdinal(BaseEstimator, ClassifierMixin):
    """
    Ordinal Regression using Gradient Boosting

    This class implements a mathematical framework for adapting machine learning (ML) regression models to handle ordinal response variables.

    Parameters
    ----------
    base_learner : estimator object, default=DecisionTreeRegressor()
        The base estimator used as weak learner in a gradient boosting context.
        
    max_iter : int, default=100
        The maximum number of boosting iterations to perform.
        
    lr_g : float, default=1e-1
        Learning rate - or shirnkage factor - applied to predictions of base learners.
        
    lr_theta : float, default=1e-3
        Starting value for the learning rate for the threshold updates. This parameter is auto-tuned during model training.
        
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for early stopping.
        
    n_iter_no_change : int or None, default=None
        Number of iterations with no improvement to wait before stopping early.
        
    reltol : float, default=1e-2
        Relative tolerance to determine if early stopping is triggered.
        
    validation_stratify : bool, default=True
        Whether to stratify the validation set.

    Methods
    -------
    fit(X, y)
        Fit the BoostedOrdinal model according to the given training data.
        
    predict(X, y=None, path=False, class_labels=True)
        Predict ordinal class labels for the given data.
        
    predict_proba(X)
        Predict class probabilities for the given data.
    """    
    
    def __init__(
        self
        , base_learner = DecisionTreeRegressor(max_depth=3)
        , max_iter=100
        , lr_g = 1e-1
        , lr_theta = 1e-3
        , validation_fraction = 0.1
        , n_iter_no_change = None
        , reltol = 1e-2
        , validation_stratify = True
        , n_class = None
        , predict_type = 'labels'
    ):
        self.base_learner = base_learner
        self.max_iter = max_iter
        self.lr_g = lr_g
        self.lr_theta = lr_theta
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.reltol = reltol
        self.validation_stratify = validation_stratify
        self.n_class = n_class
        self.predict_type = predict_type

    def fit(self, X, y):
        """
        Fit the BoostedOrdinal model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers. Must be 0, 1, ..., M-1 where, M is the number of ordinal classes.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)

        if self.n_iter_no_change:
            X, X_holdout, y, y_holdout = train_test_split(X, y, test_size = self.validation_fraction, stratify = y if self.validation_stratify else None)
        
        ylist = self._validate_ordinal(y) # sets self._n_class if not set
        #self.classes_ = np.arange(self._n_class)

        g_init, theta_init = BoostedOrdinal._initialize(y, n_class = self._n_class, laplace_smoothing = True)
        loss_init = BoostedOrdinal._loss_function(X, y, g_init, theta_init)

        g, theta, loss = g_init, theta_init, loss_init
        loss_all = []
        learner_all = []
        intercept_all = []
        g_all = []
        theta_all = []

        if self.n_iter_no_change:
            loss_holdout = BoostedOrdinal._loss_function(X_holdout, y_holdout, g_init, theta_init)
            loss_all_holdout = []
            loss_all_holdout.append(loss_holdout)
            g_holdout = g_init

        loss_all.append(loss)
        g_all.append(np.repeat(g, y.size))
        theta_all.append(theta)

        no_change = False
        
        lr_theta = self.lr_theta
        lr_theta_all = [lr_theta]
        
        #lr_g = self.lr_g
        #lr_g_all = [lr_g]

        for p in range(self.max_iter):
            
            # update regression function
            dg = BoostedOrdinal._derivative_g(X, y, theta, g)
            weak_learner, h, intercept = BoostedOrdinal._fit_weak_learner(X, -dg, clone(self.base_learner))
            g = BoostedOrdinal._update_g(g, h, lr = self.lr_g)
            #g, lr_g = BoostedOrdinal._update_g_dev(g, h, lr_g, X, y, theta, frac = 0.5)
            
            # update loss
            loss = BoostedOrdinal._loss_function(X, y, g, theta)
            loss_all.append(loss)
            
            # update threshold vector
            dtheta = BoostedOrdinal._derivative_threshold(X, ylist, theta, g)
            #theta = BoostedOrdinal._update_thresh_naive(theta, dtheta, lr = lr_theta)
            theta, lr_theta = BoostedOrdinal._update_thresh(theta, dtheta, lr_theta, X, y, g, frac = 0.5)

            # update loss
            loss = BoostedOrdinal._loss_function(X, y, g, theta)
            loss_all.append(loss)
            
            learner_all.append(weak_learner)
            intercept_all.append(intercept)
            g_all.append(g)
            theta_all.append(theta)

            lr_theta_all.append(lr_theta)
            #lr_g_all.append(lr_g)

            if self.n_iter_no_change:
                h_holdout = weak_learner.predict(X_holdout) + intercept
                g_holdout = BoostedOrdinal._update_g(g_holdout, h_holdout, lr = self.lr_g)
                #g_holdout += lr_g * h_holdout
                loss_holdout = BoostedOrdinal._loss_function(X_holdout, y_holdout, g_holdout, theta)
                loss_all_holdout.append(loss_holdout)
                if len(loss_all_holdout) > self.n_iter_no_change:
                    if ((loss_all_holdout[-(1+self.n_iter_no_change)] - loss_all_holdout[-1]) / loss_all_holdout[-(1+self.n_iter_no_change)] < self.reltol):
                        no_change = True
                        break

        self._n_iter_ = p + 1 - self.n_iter_no_change if no_change else self.max_iter
        self._init = {'g': g_init, 'theta': theta_init, 'loss': loss_init}
        self._final = {'g': g, 'theta': theta, 'loss': loss_all[-1]}
        self._path = {
            'g': np.array(g_all)
            , 'theta': np.array(theta_all)
            , 'loss': np.array(loss_all) / X.shape[0]
            , 'learner': learner_all
            , 'intercept': np.array(intercept_all)
            , 'lr_theta': np.array(lr_theta_all)
        }
        if self.n_iter_no_change:
            self.path['loss_holdout'] = np.array(loss_all_holdout) / X_holdout.shape[0]
        
        return self
    
    def predict(self, X, y=None, path=False, type=None):
        """
        Predict method that can return ordinal class labels, probabilities, or latent values,
        either for the final iteration or step-by-step (path).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,), optional
            True labels (not strictly required for prediction unless we do special calculations).
        path : bool, default=False
            If True, returns step-by-step predictions for all boosting iterations.
        type : {'labels', 'probs', 'latent'} or None, default=None
            The format of the returned prediction:
                - 'labels': Return class labels (default behavior).
                - 'probs':  Return predicted class probabilities.
                - 'latent': Return the raw, latent (continuous) scores before thresholding.
            If None, the class attribute `self.predict_type` is used instead.

        Returns
        -------
        Depending on `path` and `type`, returns either a single ndarray (final prediction)
        or a list of ndarrays (one for each iteration).
        """
        
        # check if the model has been fitted
        check_is_fitted(self)
        
        # 1) Resolve the user-specified or default prediction type
        if type is None:
            type = self.predict_type  # fallback to class-level default

        X = check_array(X)

        # 2) Collect per-iteration raw predictions
        per_iter_raw = np.array([
            learner.predict(X) + self._path['intercept'][p]
            for p, learner in enumerate(self._path['learner'])
        ])

        # 3) Handle path = True vs. path = False
        if path:
            # Return a list of predictions, one per iteration
            cum_preds = np.cumsum(per_iter_raw, axis=0) * self.lr_g + self._init['g']
            all_preds = []
            for iteration_idx in range(cum_preds.shape[0]):
                g_iter = cum_preds[iteration_idx, :]  # latent for iteration_idx

                if type == 'latent':
                    all_preds.append(g_iter)

                elif type == 'probs':
                    probs_iter = BoostedOrdinal._probabilities(
                        g_iter,
                        self._path['theta'][iteration_idx + 1],  # threshold after iteration_idx-th update
                        y=None
                    )
                    all_preds.append(probs_iter)

                elif type == 'labels':
                    probs_iter = BoostedOrdinal._probabilities(
                        g_iter,
                        self._path['theta'][iteration_idx + 1],
                        y=None
                    )
                    labels_iter = BoostedOrdinal._class_labels(probs_iter)
                    all_preds.append(labels_iter)

                else:
                    raise ValueError("type must be one of {'labels', 'probs', 'latent'}")

            return all_preds

        else:
            # Return final predictions only
            final_raw = np.sum(per_iter_raw[:self._n_iter_], axis=0) * self.lr_g + self._init['g']

            if type == 'latent':
                return final_raw

            elif type == 'probs':
                final_probs = BoostedOrdinal._probabilities(final_raw, self._path['theta'][-1], y=None)
                return final_probs

            elif type == 'labels':
                final_probs = BoostedOrdinal._probabilities(final_raw, self._path['theta'][-1], y=None)
                final_labels = BoostedOrdinal._class_labels(final_probs)
                return final_labels

            else:
                raise ValueError("type must be one of {'labels', 'probs', 'latent'}")
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the given data. This calls the predict method, with y set to None, path set to False, and class_labels set to False.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Predicted class probabilities for the input data.
        """
        
        check_is_fitted(self)
        
        return self.predict(X, y = None, path = False, type = 'probs')

    def plot_cross_entropy_loss(self):
        cross_entropy_train, cross_entropy_validation = self._path['loss'][::2] , self._path['loss_holdout']
        
        indices = np.arange(len(cross_entropy_train))
        
        # Create the plot
        plt.figure(figsize=(10, 5))
        
        # Plot the first array
        plt.plot(indices, cross_entropy_train, label='Training set')#, marker='o')
        
        # Plot the second array
        plt.plot(indices, cross_entropy_validation, label='Validation set')#, marker='x')
        
        # Add labels and title
        plt.xlabel('Iteration No')
        plt.ylabel('Cross-entropy')
        plt.title('Training vs. Validation Cross-entropy Loss')

        # add vertical line to indicate selected number of iterations
        plt.axvline(x = self._n_iter_, color = 'r', label = 'Number of Iterations Seleced', linestyle = 'dashed')
        
        plt.legend()
        
        # Show the plot
        plt.show()

        pass
        
    
    def _try_thresh(thresh_i, thresh_f, X, y, g):
        #try:
        with warnings.catch_warnings(record=True) as w:
            f_f = BoostedOrdinal._loss_function(X, y, g, thresh_f)
        if w:
            return False
        #except:
        #    return False
        
        return (BoostedOrdinal._loss_function(X, y, g, thresh_f) < BoostedOrdinal._loss_function(X, y, g, thresh_i)) and (np.all(np.diff(thresh_f) > 0))
    
    def _update_thresh(thresh, dthresh, lr, X, y, g, frac = 0.5):
        this_accept = BoostedOrdinal._try_thresh(thresh, thresh - lr * dthresh, X, y, g)
        if this_accept:
            # keep doubling till reject
            lr_proposed = lr
            while this_accept:
                lr = lr_proposed
                lr_proposed = lr / frac
                this_accept = BoostedOrdinal._try_thresh(thresh - lr * dthresh, thresh - lr_proposed * dthresh, X, y, g)
        else:
            # keep halving till accept
            while not this_accept:
                lr = lr * frac
                this_accept = BoostedOrdinal._try_thresh(thresh, thresh - lr * dthresh, X, y, g)

        return (thresh - lr * dthresh, lr)
    
    def _try_g(g_i, g_f, X, y, theta):
        with warnings.catch_warnings(record=True) as w:
            f_f = BoostedOrdinal._loss_function(X, y, g_f, theta)
        if w:
            return False
        
        return (BoostedOrdinal._loss_function(X, y, g_f, theta) < BoostedOrdinal._loss_function(X, y, g_i, theta))
    
    # conventions are different for theta and g, hence we subtract in theta and add in g (the learning rate term)
    def _update_g_dev(regfun, dregfun, lr, X, y, theta, frac = 0.5):
        this_accept = BoostedOrdinal._try_g(regfun, regfun + lr * dregfun, X, y, theta)
        if this_accept:
            # keep doubling till reject
            lr_proposed = lr
            while this_accept:
                lr = lr_proposed
                lr_proposed = lr / frac
                this_accept = BoostedOrdinal._try_g(regfun + lr * dregfun, regfun + lr_proposed * dregfun, X, y, theta)
        else:
            # keep halving till accept
            while not this_accept:
                lr = lr * frac
                this_accept = BoostedOrdinal._try_g(regfun, regfun + lr * dregfun, X, y, theta)

        return (regfun + lr * dregfun, lr)
        
    def _class_labels(probs, axis = 1):
        return np.argmax(probs, axis = axis)
    
    def _probabilities(g, theta, y = None):
        probs = np.array([np.diff(norm.cdf(BoostedOrdinal._pad_thresholds(theta - x))) for x in g])

        if y is None:
            return probs
        
        loglike = sum([np.log(probs[n, yn]) for n, yn in enumerate(y)])
        return probs, loglike
    
    def _check_loss_change(loss):
        x = np.diff(loss)
        return (x[::2], x[1::2]) # (g, theta)
    
    def _validate_ordinal(self, arr):
    
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if arr.dtype.kind not in {'i', 'u'}:
            raise ValueError("Input array must contain integers")
        
        unique_values = np.unique(arr) # we rely on numpy.unique returning a sorted array
        min_value, max_value = unique_values[0], unique_values[-1]
    
        if min_value < 0:
            raise ValueError("Minimum of arr cannot be less than 0")
    
        if not self.n_class:
            check_gap = True
            self._n_class = max_value + 1
        else:
            check_gap = False
            self._n_class = self.n_class
        
        if max_value >= self._n_class:
            raise ValueError("Maximum of arr cannot be more than n_class-1")
        
        expected_values = np.arange(self._n_class)
        
        if check_gap:
            if not np.array_equal(expected_values, unique_values):
                raise ValueError("Unique values in arr have gaps")
    
        return [np.where(arr == m) for m in expected_values]
        
    def _initialize(y, **kwargs):
        return (BoostedOrdinal._initialize_g(y), BoostedOrdinal._initialize_thresholds(y, **kwargs))
    
    def _initialize_g(y):
        #return np.zeros(y.size)
        return 0
    
    def _initialize_thresholds(y, n_class = None, laplace_smoothing = False):
        # Calculate the initial threshold vector
        n_samples = len(y)
        
        if not n_class:
            n_class = np.max(y) + 1
        else:
            if np.max(y) + 1 > n_class:
                raise ValueError('Number of classes cannot be smaller than number of distinct values in y')
        
        P = np.array([np.sum(y == i) + laplace_smoothing for i in range(n_class)]) / (n_samples + laplace_smoothing * n_class)
        return norm.ppf(np.cumsum(P[:-1]))
    
    def _pad_thresholds(theta):
        return np.insert(theta, [0, theta.size], [-np.inf, np.inf])
    
    def _derivative_threshold(X, ylist, thresh, g, return_mean = False):
        # added return_mean but not tested yet, intended to make the gradient signal insensitive to data size
        thresh_padded = BoostedOrdinal._pad_thresholds(thresh)
        M = len(thresh)
        ret = []
        for m in range(M):
            S_m = ylist[m]
            S_mp1 = ylist[m+1]
            v1 = np.sum(norm.pdf(thresh_padded[m+1] - g[S_m]) / (norm.cdf(thresh_padded[m+1] - g[S_m]) - norm.cdf(thresh_padded[m] - g[S_m])))
            v2 = np.sum(norm.pdf(thresh_padded[m+1] - g[S_mp1]) / (norm.cdf(thresh_padded[m+2] - g[S_mp1]) - norm.cdf(thresh_padded[m+1] - g[S_mp1])))
            tmp = -v1 + v2
            if return_mean:
                tmp = tmp / X.shape[0]
            ret.append(tmp)
        return np.array(ret)

    def _derivative_g(X, y, thresh, g):
        thresh_padded = BoostedOrdinal._pad_thresholds(thresh)
        ret = (norm.pdf(thresh_padded[y+1] - g) - norm.pdf(thresh_padded[y] - g)) / (norm.cdf(thresh_padded[y+1] - g) - norm.cdf(thresh_padded[y] - g))
        return ret

    def _fit_weak_learner(X, pseudo_resids, learner):
        learner.fit(X, pseudo_resids)
        pred = learner.predict(X)
        intercept = -np.mean(pred) # we could also perform intercept adjustment in _update_g but mathematically the effect is the same
        return (learner, pred + intercept, intercept)
    
    # replace with more sophisticated version that performs line search
    def _update_g(g, h, lr = 1e-1):
        return g + lr * h
    
    # we need to check if updated thresh is valid (must be sorted) and handle invalid ones
    def _update_thresh_naive(thresh, dthresh, lr = 1e-3):
        new_thresh = thresh - lr * dthresh
        if not np.all(np.diff(new_thresh) > 0):
            raise ValueError("updated threshold vector invalid (must have strict ascending order)")
        return new_thresh
    
    # this can be fused with _probabilities, though this is likely more efficient is the goal is only loss and not the prob matrix
    def _loss_function(X, y, g, theta):
        theta_padded = BoostedOrdinal._pad_thresholds(theta)
        return -np.sum(np.log(norm.cdf(theta_padded[y + 1] - g) - norm.cdf(theta_padded[y] - g)))

from sklearn.metrics import make_scorer
from itertools import combinations
import numpy as np

def concordance_index(y_true, y_pred):
    """
    Compute the concordance index (C-index) for ordinal predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels.
        
    y_pred : array-like of shape (n_samples,)
        Predicted scores or continuous outputs from the model.

    Returns
    -------
    float
        Concordance index, ranging from 0 to 1. A higher value indicates better concordance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check that the input sizes match
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    
    # Pairwise comparisons
    pairs = list(combinations(range(len(y_true)), 2))
    concordant, permissible = 0, 0
    
    for i, j in pairs:
        if y_true[i] != y_true[j]:  # Skip ties
            permissible += 1
            if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
               (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                concordant += 1
            elif y_pred[i] == y_pred[j]:  # Handle ties in predictions
                concordant += 0.5

    return concordant / permissible if permissible > 0 else 0.0

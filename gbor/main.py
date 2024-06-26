from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from scipy.stats import norm
from sklearn.utils import check_X_y, check_array
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold
import warnings
import matplotlib.pyplot as plt

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
        , base_learner = DecisionTreeRegressor()
        , max_iter=100
        , lr_g = 1e-1
        , lr_theta = 1e-3
        , validation_fraction = 0.1
        , n_iter_no_change = None
        , reltol = 1e-2
        , validation_stratify = True
        , n_class = None
        , cv = None
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
        self.cv = cv

    def fit_cv(self, X, y):
        ncv = self.cv
        kf = KFold(n_splits = ncv, shuffle = True)
        cv_indices = [x for x in kf.split(X)]
        if self.n_class:
            n_class = self.n_class
        else:
            n_class = np.max(y) + 1

        ylist_list = [BoostedOrdinal._validate_ordinal_2(y[train_index], n_class = n_class) for train_index, _ in cv_indices]
        #g_theta_init_list = [BoostedOrdinal._initialize(y[train_index], n_class = n_class, laplace_smoothing = True) for train_index, _ in cv_indices]
        g_init_list, theta_init_list = zip(*[BoostedOrdinal._initialize(y[train_index], n_class = n_class, laplace_smoothing = True) for train_index, _ in cv_indices])
        
        g_list, theta_list = g_init_list, theta_init_list
        g_holdout = np.empty(len(y))
        for k, (train_index, test_index) in enumerate(cv_indices):
            g_holdout[test_index] = g_init_list[k]
        
        loss_holdout = sum([BoostedOrdinal._loss_function(X[test_index, :], y[test_index], g_holdout[test_index], theta_list[k]) for k, (train_index, test_index) in enumerate(cv_indices)])
        loss_holdout_all = [loss_holdout]
        
        no_change = False
        lr_theta_list = list(np.repeat(self.lr_theta, ncv))

        for p in range(self.max_iter):
            # update regression function
            dg_list = [BoostedOrdinal._derivative_g(X[train_index, :], y[train_index], theta_list[k], g_list[k]) for k, (train_index, test_index) in enumerate(cv_indices)]
            weak_learner_list, h_list, intercept_list = zip(*[BoostedOrdinal._fit_weak_learner(X[train_index, :], -dg_list[k], clone(self.base_learner)) for k, (train_index, test_index) in enumerate(cv_indices)])
            g_list = [BoostedOrdinal._update_g(g_list[k], h_list[k], lr = self.lr_g) for k, (train_index, test_index) in enumerate(cv_indices)]
            
            # update holdout loss
            h_holdout = np.empty(len(y))
            for k, (train_index, test_index) in enumerate(cv_indices):
                h_holdout[test_index] = weak_learner_list[k].predict(X[test_index, :]) + intercept_list[k]
            g_holdout = BoostedOrdinal._update_g(g_holdout, h_holdout, lr = self.lr_g)
            loss_holdout = sum([BoostedOrdinal._loss_function(X[test_index, :], y[test_index], g_holdout[test_index], theta_list[k]) for k, (train_index, test_index) in enumerate(cv_indices)])
            
            # update threshold vector
            dtheta_list = [BoostedOrdinal._derivative_threshold(X[train_index, :], ylist_list[k], theta_list[k], g_list[k]) for k, (train_index, test_index) in enumerate(cv_indices)]
            theta_list, lr_theta_list = zip(*[BoostedOrdinal._update_thresh(theta_list[k], dtheta_list[k], lr_theta_list[k], X[train_index, :], y[train_index], g_list[k], frac = 0.5) for k, (train_index, test_index) in enumerate(cv_indices)])
            # update loss
            loss_holdout = sum([BoostedOrdinal._loss_function(X[test_index, :], y[test_index], g_holdout[test_index], theta_list[k]) for k, (train_index, test_index) in enumerate(cv_indices)])
            loss_holdout_all.append(loss_holdout)
            
            # check for change in holdout loss
            if len(loss_holdout_all) > self.n_iter_no_change:
                if ((loss_holdout_all[-(1+self.n_iter_no_change)] - loss_holdout_all[-1]) / loss_holdout_all[-(1+self.n_iter_no_change)] < self.reltol):
                    no_change = True
                    break

        n_iter_no_change = self.n_iter_no_change
        self.n_iter_no_change = None
        self.max_iter = p + 1 - n_iter_no_change if no_change else self.max_iter
        self.fit(X, y)
        self.path['loss_holdout'] = np.array(loss_holdout_all[:(self.max_iter + 1)]) / X.shape[0]
        self.n_iter_no_change = n_iter_no_change
        
        return self
    
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
            if self.cv:
                return self.fit_cv(X, y)
            X, X_holdout, y, y_holdout = train_test_split(X, y, test_size = self.validation_fraction, stratify = y if self.validation_stratify else None)
        
        #ylist = BoostedOrdinal._validate_ordinal(y)
        ylist = BoostedOrdinal._validate_ordinal_2(y, n_class = self.n_class)

        g_init, theta_init = BoostedOrdinal._initialize(y, n_class = self.n_class, laplace_smoothing = True)
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

        self.n_iter = p + 1 - self.n_iter_no_change if no_change else self.max_iter
        self.init = {'g': g_init, 'theta': theta_init, 'loss': loss_init}
        self.final = {'g': g, 'theta': theta, 'loss': loss_all[-1]}
        self.path = {
            'g': np.array(g_all)
            , 'theta': np.array(theta_all)
            , 'loss': np.array(loss_all) / X.shape[0]
            , 'learner': learner_all
            , 'intercept': np.array(intercept_all)
            , 'lr_theta': np.array(lr_theta_all)
            #, 'lr_g': np.array(lr_g_all)
        }
        if self.n_iter_no_change:
            self.path['loss_holdout'] = np.array(loss_all_holdout) / X_holdout.shape[0]
        
        return self
    
    def predict(self, X, y = None, path = False, class_labels = True):
        """
        Predict ordinal class labels for the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        y : array-like of shape (n_samples,), optional
            The true labels for X. If provided, log-likleihood of data is also returned
            
        path : bool, default=False
            If True, returns the prediction path (step-by-step predictions from all boosting iterations).
            
        class_labels : bool, default=True
            If True, returns the predicted class labels.
            If False, returns the predicted probabilities.

        Returns
        -------
        array-like
            Predicted class labels or probabilities for the input data. However, 1) if y is provided, a tuple is returned, with the second element being the log-liklihood of the data; 2) if 'path' is set to True, a list of such objects is returned.
        """
        check_array(X)
        arr = np.array([learner.predict(X) + self.path['intercept'][p] for p, learner in enumerate(self.path['learner'])])
        if path:
            arr = np.cumsum(arr, 0) * self.lr_g + self.init['g']
            if class_labels:
                tmp = [BoostedOrdinal._probabilities(arr[p, :], self.path['theta'][p+1], y) for p in range(arr.shape[0])]
                if class_labels:
                    if y is None:
                        return [BoostedOrdinal._class_labels(u) for u in tmp]
                    else:
                        return [(BoostedOrdinal._class_labels(u[0]), u[1]) for u in tmp]
                else:
                    return tmp
            else:
                return [BoostedOrdinal._probabilities(arr[p, :], self.path['theta'][p+1], y) for p in range(arr.shape[0])]
        else:
            arr = np.sum(arr[:self.n_iter, :], 0) * self.lr_g + self.init['g']
            tmp = BoostedOrdinal._probabilities(arr, self.path['theta'][-1], y)
            if class_labels:
                if y is None:
                    return BoostedOrdinal._class_labels(tmp)
                else:
                    return BoostedOrdinal._class_labels(tmp[0]), tmp[1]
                    
            else:
                return tmp

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
        return self.predict(X, y = None, path = False, class_labels = False)

    def plot_cross_entropy_loss(self):
        cross_entropy_train, cross_entropy_validation = self.path['loss'][::2] , self.path['loss_holdout']
        
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
        plt.axvline(x = self.n_iter, color = 'r', label = 'Number of Iterations Seleced', linestyle = 'dashed')
        
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
    
    def _validate_ordinal_2(arr, n_class = None):
    
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if arr.dtype.kind not in {'i', 'u'}:
            raise ValueError("Input array must contain integers")
        
        unique_values = np.unique(arr) # we rely on numpy.unique returning a sorted array
        min_value, max_value = unique_values[0], unique_values[-1]
    
        if min_value < 0:
            raise ValueError("Minimum of arr cannot be less than 0")
    
        if not n_class:
            check_gap = True
            n_class = max_value + 1
        else:
            check_gap = False
        
        if max_value >= n_class:
            raise ValueError("Maximum of arr cannot be more than n_class-1")
        
        expected_values = np.arange(n_class)
        
        if check_gap:
            if not np.array_equal(expected_values, unique_values):
                raise ValueError("Unique values in arr have gaps")
    
        return [np.where(arr == m) for m in expected_values]
    
    def _validate_ordinal(arr):
        """
        Check if the unique values in a numpy integer vector are 0, 1, ..., M with M >= 2.
    
        Parameters:
        arr (numpy.ndarray): Input numpy integer vector.
    
        Returns:
        bool: True if unique values are 0, 1, ..., M with M >= 2, False otherwise.
        """
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if arr.dtype.kind not in {'i', 'u'}:
            raise ValueError("Input array must contain integers")
    
        unique_values = np.unique(arr)
        
        if unique_values[0] != 0:
            return []
        
        M = unique_values[-1]

        if M < 2:
            return []
        
        expected_values = np.arange(M + 1)

        if np.array_equal(unique_values, expected_values):
            #return M + 1
            return [np.where(arr == m) for m in unique_values]
        else:
            return []
    
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

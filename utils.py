##############################################################################
"""
Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import itertools
import time

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        return self.elapsed(time.time() - self.start_time)
    
def gen_grid(d, points_per_axis, lb=0., rb=1.):
    ''' Generate a grid in a d-dimensional space 
        within the range [lb, rb] for each axis '''
    
    lincoords = []
    for i in range(0, d):
        lincoords.append(np.linspace(lb, rb, points_per_axis))
    coords = list(itertools.product(*lincoords))
    
    return np.array(coords)

def mean_err(metric_list):
    n = len(metric_list)
    mean = np.mean(metric_list)
    std = np.std(metric_list)
    err = 1.96*std/n**.5
    return mean, err

def optimize_kde(X):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(-3, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, n_jobs=8, cv=5, verbose=1)
    grid.fit(X)
    
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    
    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_
    
    return kde

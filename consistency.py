"""
Estimates likelihood of generated data using kernel density estimation 

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np

def sample_line(d, m):
    # Sample m points along a line parallel to a d-dimensional space's basis
    basis = np.random.choice(d)
    c = np.zeros((m, d))
    c[:,:] = np.random.rand(d)
    c[:,basis] = np.linspace(0.0, 1.0, m)
    return c

def consistency(gen_func, child=False, X_parent=None):
    
    n_eval = 100
    n_points = 50
    mean_cor = 0
    
    for i in range(n_eval):
        
        c = sample_line(2, n_points)
        dist_c = np.linalg.norm(c - c[0], axis=1)
        
#        from matplotlib import pyplot as plt
#        plt.scatter(c[:,0], c[:,1])
        
        if child:
            X_p = X_parent[np.random.choice(X_parent.shape[0])]
            X = gen_func(c, X_p)[1]
        else:
            X = gen_func(c)
        X = X.reshape((n_points, -1))
        dist_X = np.linalg.norm(X - X[0], axis=1)
        
        mean_cor += np.corrcoef(dist_c, dist_X)[0,1]
        
    return mean_cor/n_eval
        

def ci_cons(n, gen_func, child=False, X_parent=None):
    conss = np.zeros(n)
    for i in range(n):
        conss[i] = consistency(gen_func, child=child, X_parent=X_parent)
    mean = np.mean(conss)
    std = np.std(conss)
    err = 1.96*std/n**.5
    return mean, err
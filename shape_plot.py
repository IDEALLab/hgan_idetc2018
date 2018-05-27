"""
Plots samples or new shapes in the semantic space.

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""

from matplotlib import pyplot as plt
import numpy as np
from utils import gen_grid

def plot_shape(xys, z1, z2, ax, scale=.05, **kwargs):
    
#    mx = max([y for (x, y) in m])
#    mn = min([y for (x, y) in m])
    xscl = scale# / (mx - mn)
    yscl = scale# / (mx - mn)
#    ax.scatter(z1, z2)
    ax.plot( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), **kwargs)

def plot_samples(Z, X1, X2=None, scale=.05, annotate=False, save_path=None):
    
    ''' Plot shapes given design sapce and latent space coordinates '''
    
    plt.rc("font", size=12)
    
    if Z is None:
        N = X1.shape[0]
        points_per_axis = int(N**.5)
        bounds = (-1., 1.)
        Z = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
        scale = 0.8*2.0/points_per_axis

    # Create a 2D plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
            
    for (i, z) in enumerate(Z):
        if annotate:
            label = '{0}'.format(i+1)
            plt.annotate(label, xy = (z[0], z[1]), size=10)
            
        if X2 is None:
            plot_shape(X1[i], z[0], z[1], ax, scale=scale, lw=1.0, ls='-', c='k', alpha=1)
        
        else:
            plot_shape(X1[i], z[0], z[1], ax, scale=scale, lw=1.0, ls=':', c='k', alpha=1)
            plot_shape(X2[i], z[0], z[1], ax, scale=scale, lw=1.0, ls='-', c='k', alpha=1)
    
#    plt.xlabel('c1')
#    plt.ylabel('c2')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_synthesized(Z, gen_func, d=2, scale=.05, alpha=1.0, save_path=None):
    
    ''' Synthesize shapes given latent space coordinates and plot them '''
    
    if d == 2:
        latent = Z
    else:
        latent = np.random.normal(scale=0.5, size=(Z.shape[0], d))
    X = gen_func(latent)
    
    if isinstance(X, tuple):
        X1 = X[0]
        X2 = X[1]
        plot_samples(Z, X1, X2, scale=scale, save_path=save_path)
    else:
        plot_samples(Z, X, scale=scale, save_path=save_path)

def plot_grid(points_per_axis, gen_func, d=2, bounds=(0.0, 1.0), scale=.8, save_path=None):
    
    ''' Uniformly plots synthesized shapes in the latent space
        K : number of samples for each point in the latent space '''
        
    scale *= (bounds[1]-bounds[0])/points_per_axis
    
    Z = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
    plot_synthesized(Z, gen_func, d=d, scale=scale, save_path=save_path)
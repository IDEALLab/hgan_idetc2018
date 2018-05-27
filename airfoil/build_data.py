"""
Creates dataset of airfoils with holes

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import pyplot as plt
import math

import sys
sys.path.append("../")
from shape_plot import plot_shape

def points_on_circle(circle, n=100):
    x = circle[0]
    y = circle[1]
    r = circle[2]
    C = [(x + math.cos(2*np.pi/n*i)*r, y + math.sin(2*np.pi/n*i)*r) for i in xrange(n)]
    return np.array(C)

def filt(circles, shape):
    centers = circles[:,:2]
    radii = circles[:,2]
    # The center should be inside the airfoil
    ubs = np.interp(centers[:,0], shape[range(49,-1,-1),0], shape[range(49,-1,-1),1])
    lbs = np.interp(centers[:,0], shape[50:,0], shape[50:,1])
    ind1 = np.logical_and(centers[:,1]<ubs, centers[:,1]>lbs)
    # All points on the airfoil contour should be outside the circle
    distances = pairwise_distances(shape, centers)
    ind2 = np.all(distances-radii.reshape((1,-1))>0.01, axis=0)
    return np.logical_and(ind1, ind2)

def build_data():
    # Airfoils
    A = np.load('airfoil/airfoils.npy')
    A = A.reshape((-1, 100, 2))
    
    # Holes
    C = np.random.uniform(low=[0.0,-0.2,0.04], high=[1.0,0.3,0.5], 
                          size=(100000, 3)) # random circles (x0, y0, r)
    X = []
    count = 0
    for (i, a) in enumerate(A):
        ind = filt(C, a)
        Ci = C[ind] # circles inside the airfoil
        for (j, c) in enumerate(Ci):
            h = points_on_circle(c, 100)
            ah = np.concatenate((a, h))
            X.append(ah)
            if j > 500:
                break
        print '%d : %d' % (i, Ci.shape[0])
        if Ci.shape[0] != 0:
            count += 1
    print 'Total valid airfoils: ', count
    
    X = np.array(X)
    np.random.shuffle(X)
    np.save('airfoil_hole.npy', X)
    return X
    
if __name__ == "__main__":
    
    X = build_data()
    
    # Visualize
    for i in np.random.randint(1, X.shape[0], size=10):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_shape(X[i,:100], 0, 0, ax, mirror=False, linewidth=1.5, color='blue', linestyle='-', alpha=1, scale=1)
        plot_shape(X[i,100:], 0, 0, ax, mirror=False, linewidth=1.5, color='blue', linestyle='-', alpha=1, scale=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([-0.5, 0.5])
        plt.show()
    
    
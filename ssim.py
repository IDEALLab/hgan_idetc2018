import numpy as np
from utils import mean_err

# 100 x 2 image
def ssim(m1, m2):
    u1 = np.mean(m1)
    u2 = np.mean(m2)

    C1 = (np.ptp(m1) * 0.01) ** 2
    C2 = (np.ptp(m2) * 0.03) ** 2

    # each row in cov represents a variable so convert each matrix into dimension 1
    # flatten into vector 200 dimensions
    COV = np.cov(m1.flatten(), m2.flatten())
    v1 = COV[0,0]
    v2 = COV[1,1]
    cov = COV[0,1]

    return (2 * u1 * u2 + C1) * (2 * cov + C2) / ((u1**2 + u2**2 + C1) * (v1 + v2 + C2))

# or just use scipy's ssim of x * scipy's ssim of y

#def avg_dist(m1, m2, order=True): # avg dist between corresponding points
#    if not order:
#        m1 = sorted(m1)
#        m2 = sorted(m2)
#    d = 0
#    for p1, p2 in zip(m1, m2):
#        d += np.linalg.norm(p1-p2)
#
#    return d / len(m1)

def rssim(X_train, X_gen):
    ''' Relative SSIM '''
    X_train = np.squeeze(X_train)
    n = 100
    gen_ssim = train_ssim = 0
    for i in range(n):
        a, b = np.random.choice(X_gen.shape[0], 2, replace=False)
        gen_ssim += ssim(X_gen[a], X_gen[b])
        c, d = np.random.choice(X_train.shape[0], 2, replace=False)
        train_ssim += ssim(X_train[c], X_train[d])
    rssim = train_ssim/gen_ssim
    return rssim

def ci_rssim(n, X_train, gen_func):
    rssims = np.zeros(n)
    for i in range(n):
        X_gen = gen_func(X_train.shape[0])
        rssims[i] = rssim(X_train, X_gen)
    mean, err = mean_err(rssims)
    return mean, err


#a = np.load('airfoil/airfoil_hole.npy')[0]
#ax = a[:,0]
#ay = a[:,1]
#
#b = np.load('airfoil/airfoil_hole.npy')[1]
#bx = b[:,0]
#by = b[:,1]
#
#print(ssim(np.array([-1.2, -1]), np.array([1.2, -0.9])))

#print(avg_dist(a,b))
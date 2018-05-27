"""
Trains an HGAN, and visulizes results

Author(s): Wei Chen (wchen459@umd.edu)
"""

import sys
import argparse
import os.path
import pickle
import numpy as np
from importlib import import_module
from matplotlib import pyplot as plt

import shape_plot
from ssim import ci_rssim
from likelihood import ci_mll
from consistency import ci_cons
from utils import ElapsedTimer


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='startover', help='startover, continue or evaluate')
    parser.add_argument('data', type=str, default='airfoil', help='airfoil or superformula')
    parser.add_argument('model', type=str, default='2g1d', help='2g1d, naive, wo_info')
    parser.add_argument('--train_steps', type=int, default=100000, help='training steps')
    parser.add_argument('--save_interval', type=int, default=500, help='save interval')
    args = parser.parse_args()
    assert args.mode in ['startover', 'continue', 'evaluate']
    assert args.data in ['airfoil', 'superformula']
    assert args.model in ['2g1d', 'naive', 'wo_info']
    
    data_directory = '../hgan_idetc2018_data/%s/%s' % (args.data, args.model)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    if args.data == 'airfoil':
        data_fname = '../hgan_idetc2018_data/airfoil/airfoil_hole.npy'
    else:
        data_fname = '../hgan_idetc2018_data/superformula/superformula_ellipse.npy'
    
    sys.path.append('./%s' % args.data)
    sys.path.append('./%s/%s' % (args.data, args.model))
    
    # Read dataset
    if os.path.isfile(data_fname):
        X = np.load(data_fname)
    else:
        dt = import_module('build_data')
        X = dt.build_data()
        
    X = X.reshape((X.shape[0],-1,2,1)).astype(np.float32)
    latent_dim = 2
    noise_dim = 100
    
    # Split training and test data
    test_split = 0.8
    N = X.shape[0]
    split = int(N*test_split)
    X_train = X[:split]
    X_test = X[split:]
    
    # Train
    h = import_module('hgan')
    hgan = h.HGAN(X_train, X_test, latent_dim=latent_dim, noise_dim=noise_dim)
    timer = ElapsedTimer()
    hgan.train(batch_size=100, train_steps=args.train_steps, save_interval=args.save_interval, mode=args.mode)
    elapsed_time = timer.elapsed_time()
    
    runtime_mesg = 'Wall clock time: %s' % elapsed_time
    print(runtime_mesg)
    
    print('Plotting training samples ...')
    samples1 = np.squeeze(X)[:64, :100]
    samples2 = np.squeeze(X)[:64, 100:]
    shape_plot.plot_samples(None, samples1, samples2, save_path='%s/samples.svg' % args.data)
    
    print('Plotting synthesized parent ...')
    shape_plot.plot_grid(5, gen_func=hgan.synthesize_parent, d=latent_dim, scale=.95, 
                         save_path='%s/%s/parent.svg' % (args.data, args.model))
    
    print('Plotting synthesized child ...')
    shape_plot.plot_grid(5, gen_func=hgan.synthesize_child, d=latent_dim, scale=.95, 
                         save_path='%s/%s/child.svg' % (args.data, args.model))
    
    n_runs = 10
    
    mll_mean, mll_err = ci_mll(n_runs, hgan.synthesize_assembly, X_test)
    rssim_mean, rssim_err = ci_rssim(n_runs, X_train, hgan.synthesize_assembly)
    cons1_mean, cons1_err = ci_cons(n_runs, hgan.synthesize_parent)
    cons_mean, cons_err = ci_cons(n_runs, hgan.synthesize_child, child=True, X_parent=X_test[:,:100])
    
    results_mesg_1 = 'Mean log likelihood for assembly: %.1f +/- %.1f' % (mll_mean, mll_err)
    results_mesg_2 = 'Relative diversity for assembly: %.3f +/- %.3f' % (rssim_mean, rssim_err)
    results_mesg_3 = 'Consistency for parent latent space: %.3f +/- %.3f' % (cons1_mean, cons1_err)
    results_mesg_4 = 'Consistency for child latent space: %.3f +/- %.3f' % (cons_mean, cons_err)
    
    print(results_mesg_1)
    print(results_mesg_2)
    print(results_mesg_3)
    print(results_mesg_4)
    
    if args.mode == 'startover':
        # Save results to text file
        text_file = open('./%s/%s/results.txt' % (args.data, args.model), 'w')
        text_file.write('%s\n' % runtime_mesg)
        text_file.write('%s\n' % results_mesg_1)
        text_file.write('%s\n' % results_mesg_2)
        text_file.write('%s\n' % results_mesg_3)
        text_file.write('%s\n' % results_mesg_4)
        text_file.close()
    
    if args.model == '2g1d':
        # Plot losses
        with open('%s/losses.pkl' % data_directory, 'rb') as f:
            losses = pickle.load(f)
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.set_title('Training losses')
        plt.plot(losses['d_loss_real'], 'r-', label='D loss for real data')
        plt.plot(losses['d_loss_fake'], 'g--', label='D loss for synthetic data')
        plt.plot(losses['g_loss_parent'], 'b:', label='G loss for parent shapes')
        plt.plot(losses['g_loss_child'], 'c-.', label='G loss for child shapes')
        plt.plot(losses['q_loss_parent'], 'm-', label='Q loss for parent shapes')
        plt.plot(losses['q_loss_child'], 'k--', label='Q loss for child shapes')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Losses')
        ax.legend(loc='best')
        plt.savefig('%s/%s/losses.svg' % (args.data, args.model), dpi=600)
        plt.close()

    print 'All completed :)'

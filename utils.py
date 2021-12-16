import seaborn as sns
import numpy as np
import torch
import matplotlib as plt

def plot_elbo(bints):

    """
    Util function to plot the ELBO track of an instance of BinaryTS.
    """

    elbo = bints.elbo_track
    elbo = elbo.detach().numpy()
    sns.set_context('talk')
    sns.set_style('whitegrid')

    f, ax = plt.subplots(figsize=(15, 8))

    ax.set_ylabel('ELBO', fontsize=25, fontweight='bold')

    ax.set_xlabel('Iterations', fontsize=25, fontweight='bold')
    ax.plot(elbo[1:], linewidth=4)

def plot_values(bints):

    """
    Util function to plot the error between true parameter values and estimated parameters through iterations of
    the algorithm.
    """

    f, ax = plt.subplots(1, 3, figsize=(30, 10))

    f.suptitle('Training error of estimated $\mathbf{A,B,\Sigma}$', fontsize=45)

    ax[0].plot(bints.A_error_track[1:], linewidth=4)
    ax[0].set_ylabel('$||\mathbf{A}_{true}-\mathbf{A}_{estimated}||_F$', fontsize=35)
    ax[1].plot(bints.B_error_track[1:], linewidth=4)
    ax[1].set_ylabel('$||\mathbf{B}_{true}-\mathbf{B}_{estimated}||_F$', fontsize=35)
    ax[2].plot(bints.sigma_inv_error_track[1:], linewidth=4)
    ax[2].set_ylabel('$||\mathbf{\Sigma}_{true}-\mathbf{\Sigma}_{estimated}||_F$', fontsize=35)

    for i in [0, 1, 2]:
        ax[i].set_xlabel('Iterations')
    plt.tight_layout()

def plot_gradients(bints):

    """
    Util function to plot the gradient's norms through iterations of the algorithm.
    """

    f, ax = plt.subplots(1, 3, figsize=(30, 15))

    f.suptitle('Norm of ELBO gradient w.r.t. $\mathbf{A,B,\Sigma}$', fontsize=45)

    ax[0].plot(bints.A_grad_track[1:], c='red', linewidth=4)
    ax[0].set_ylabel('Norm of $\mathbf{A}$-gradient', fontsize=35)
    ax[0].set_yscale('log')
    ax[1].plot(bints.B_grad_track[1:], c='blue', linewidth=4)
    ax[1].set_ylabel('Norm of $\mathbf{B}$-gradient', fontsize=35)
    ax[1].set_yscale('log')
    ax[2].plot(bints.sigma_inv_grad_track[1:], c='purple', linewidth=4)
    ax[2].set_ylabel('Norm of $\mathbf{\Sigma}$-gradient', fontsize=35)
    ax[2].set_yscale('log')

    for i in [0, 1, 2]:
        ax[i].set_xlabel('Iterations', fontsize=35)
        ax[i].tick_params(axis='both', which='major', labelsize=30)

    plt.tight_layout()
    plt.show()

    f, ax = plt.subplots(1, 2, figsize=(30, 10))

    f.suptitle('Norm of ELBO gradient w.r.t. $\mathbf{\mu}, \mathbf{\Lambda}$', fontsize=45)

    ax[0].plot(bints.mu_approx_grad_track[1:], c='red', linewidth=4)
    ax[0].set_ylabel('Norm of $\mathbf{\mu}$-gradient', fontsize=35)
    ax[0].set_yscale('log')
    ax[1].plot(bints.var_approx_grad_track[1:], c='blue', linewidth=4)
    ax[1].set_ylabel('Norm of $\mathbf{\Lambda}$-gradient', fontsize=35)
    ax[1].set_yscale('log')

    for i in [0, 1]:
        ax[i].set_xlabel('Iterations', fontsize=35)
        ax[i].tick_params(axis='both', which='major', labelsize=30)

    plt.tight_layout()
    plt.show()
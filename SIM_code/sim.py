# Simulate multi-output data
import numpy as np
from scipy.stats import multivariate_normal
import pyGPs
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma 

# import private libraries
import sys
sys.path.append("..")
from Utility import kernels
from Utility import settings
from Utility import logpos
from Utility import utils
from Utility import kronecker_operation
from Utility import posterior_analysis

def SIM_MNDATA(M, N, x, pertentage=0.5):
    """
    Randomly simulate multi-output non-stationary data for M tasks and N individuals
    :param M: the number of tasks
    :param N: the number of inputs
    :param: x 1d array with length N
    :return: 2d array with dim N by M
    """
    # randomly generate covariance matrix for tasks and individuals
    L_M = np.random.randn(M, M)
    K_M = np.matmul(L_M, L_M.T)
    # we propose a blocked covariance matrix
    N0 = int(N*pertentage)
    N1 = int(N-N0)
    K_N = np.zeros([N, N])
    kernel = pyGPs.cov.RBF()
    kernel.hyp = [-1., 1.]
    K_N[:N0, :N0] = kernel.getCovMatrix(x=x[:N0].reshape([-1,1]), mode="train")
    kernel.hyp = [-4., 2.]
    K_N[-N1:, -N1:] = kernel.getCovMatrix(x=x[-N1:].reshape([-1,1]), mode="train")
    K = np.kron(K_M, K_N)
    y = multivariate_normal.rvs(mean=np.zeros(M*N), cov=K)
    Y = y.reshape([M,N]).T
    return Y, K_M, K_N

def SIM_USDATA(N, x):
    """
    Randomly simulate univariate stationary data
    :param N: the number of inputs
    :param: x 1d array with length N
    :return: 1d array with length N
    """
    kernel = pyGPs.cov.RBF()
    kernel.hyp = [-1, 1.]
    K = kernel.getCovMatrix(x=x.reshape([-1,1]), mode="train")
    y = multivariate_normal.rvs(mean=np.zeros(N), cov=K)
    return y, K

def SIM_UNDATA(N, x):
    """
    Randomly simulate univariate non-stationary data
    :param N: the number of inputs
    :param: x 1d array with length N
    :return: 1d array with length
    """
    kernel = pyGPs.cov.RBF()
    # we propose a blocked covariance matrix
    N0 = int(N / 2)
    N1 = int(N - N0)
    K = np.zeros([N, N])
    kernel = pyGPs.cov.RBF()
    kernel.hyp = [-1., 1.]
    K[:N0, :N0] = kernel.getCovMatrix(x=x[:N0].reshape([-1, 1]), mode="train")
    kernel.hyp = [-3., 2.]
    K[-N1:, -N1:] = kernel.getCovMatrix(x=x[-N1:].reshape([-1, 1]), mode="train")
    y = multivariate_normal.rvs(mean=np.zeros(N), cov=K)
    return y, K

def SIM_MSDATA(M, N, x):
    """
    Randomly simulate multi-output stationary data for M tasks and N individuals
    :param M: the number of tasks
    :param N: the number of inputs
    :param: x 1d array with length N
    :return: 2d array with dim N by M
    """
    # randomly generate covariance matrix for tasks and individuals
    L_M = np.random.randn(M, M)
    K_M = np.matmul(L_M, L_M.T)
    # we propose a blocked covariance matrix
    K_N = np.zeros([N, N])
    kernel = pyGPs.cov.RBF()
    kernel.hyp = [-1., 1.]
    K_N = kernel.getCovMatrix(x=x.reshape([-1,1]), mode="train")
    K = np.kron(K_M, K_N)
    y = multivariate_normal.rvs(mean=np.zeros(M*N), cov=K)
    Y = y.reshape([M,N]).T
    return Y, K_M, K_N

def SIM_MNTS_S(M, N, save_dir=None, folder_name=None, file_name="sim_MNTS.pickle", verbose=True, seed = 0):
    # Generate N time stamps on (0,1)
    x = torch.from_numpy(np.sort(np.random.rand(N))).type(settings.torchType)
    # Genetate length-scale function on x
    # tilde_l = 8*(x-1)**3
    tilde_l = 3*(x-1)**3 - 3
    l = torch.exp(tilde_l)
    if verbose:
        fig = plt.figure()
        plt.plot(x.numpy(), tilde_l.numpy())
        plt.savefig(save_dir + folder_name + "true_log_l.png")
        plt.close(fig)
    # Generate std process
    L11 = 1
    L22 = 2
    std = 1 + x**2
    stds = torch.stack([std*L11, std*L22], axis=1)
    std_array = stds.numpy()
    if verbose:
        fig = plt.figure()
        for m in range(M):
            plt.plot(x.numpy(), std_array[:, m], label="Dim {}".format(m+1))
        plt.legend()
        plt.savefig(save_dir + folder_name + "std.png")
        plt.close(fig)
    # Generate correlation process using cos function
    # cors = torch.cos(x*np.pi) 
    cors = torch.ones_like(x)* 0.5
    if verbose:
        fig = plt.figure()
        plt.plot(x.numpy(), cors.numpy())
        plt.savefig(save_dir + folder_name + "true_R_{}{}.png".format(0, 1))
        plt.close(fig)
    # Generate covariance process via combining std process and correlation process
    L_f_list = []
    for n in range(N):
        D_f = torch.diag(stds[n, :])
        R_f = torch.eye(M).type(settings.torchType)
        R_f[0,1] = cors[n]
        R_f[1,0] = cors[n]
        B_f = D_f.mm(R_f).mm(D_f)
        L_f = torch.cholesky(B_f)
        L_f_list.append(L_f)
    L_vecs = torch.cat([L_f[[0,1,1], [0,0,1]] for L_f in L_f_list])

    # Generate sigma2_err
    # sigma2_err = 1./Gamma(a, b).sample()
    # sigma2_err = 1e-6
    sigma2_err = 1e-2
    # Generare y
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l)
    K_i = logpos.generate_K_index_SVC(L_f_list)
    neworder = torch.arange(N*M).view([N, M]).t().contiguous().view(-1)
    K_i = K_i[:, neworder][neworder]
    K = kronecker_operation.kronecker_product(torch.ones([M, M]).type(settings.torchType), K_x) * K_i
    torch.manual_seed(seed)
    y = MultivariateNormal(loc=torch.zeros(M*N).type(settings.torchType), covariance_matrix=K + sigma2_err *
                                            torch.diag(torch.ones(M*N).type(settings.torchType))).sample()
    Y = y.view([M, N])
    if verbose:
        fig = plt.figure()
        for m in range(M):
            plt.plot(x.numpy(), Y.numpy()[m, :], label="Dim {}".format(m+1))
        plt.legend()
        plt.savefig(save_dir + folder_name + "Y.png")
        plt.close(fig)

    with open(save_dir + folder_name + file_name, "wb") as res:
        pickle.dump([x.numpy(), l.numpy(), L_vecs.numpy(), sigma2_err, Y.numpy().T], res)
    return Y

def SIM_MNTS(M, N, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L, a, b, save_dir=None, folder_name=None, file_name="sim_MNTS.pickle", verbose=True, seed = 0):
    # np.random.seed(22)
    # torch.manual_seed(22)
    # Generate N time stamps on (0,1)
    x = torch.from_numpy(np.sort(np.random.rand(N))).type(settings.torchType)
    # Generate length-scale function on x
    # tilde_l = 8*(x-1)**3
    tilde_l = 3*(x-1)**3- 3
    # Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    # l_dist = MultivariateNormal(loc=mu_tilde_l*torch.ones(N).type(settings.torchType), covariance_matrix=Sigma_l)
    # tilde_l = l_dist.sample()
    

    l = torch.exp(tilde_l)
    if verbose:
        fig = plt.figure()
        plt.plot(x.numpy(), tilde_l.numpy())
        plt.savefig(save_dir + folder_name + "true_log_l.png")
        plt.close(fig)

    # # Generate L_vecs (Approach 1)
    # Sigma_L = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_L, beta=beta_L)
    # L_dist = MultivariateNormal(loc=mu_L*torch.ones(N).type(settings.torchType), covariance_matrix=Sigma_L)
    # L_vecs = L_dist.sample([int(M*(M+1)/2)]).t().contiguous().view(-1)
    # print(tilde_l.shape, L_vecs.shape)
    # L_vec_list = [L_vecs[n*int(M*(M+1)/2):(n+1)*int(M*(M+1)/2)] for n in range(N)]
    # L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list]
    # std_array = np.stack([np.sqrt(np.diag(torch.mm(L_f, L_f.t()).numpy())) for L_f in L_f_list]) # array
    # if verbose:
    #     fig = plt.figure()
    #     for m in range(M):
    #         plt.plot(x.numpy(), std_array[:, m], label="Dim {}".format(m+1))
    #     plt.legend()
    #     plt.savefig(save_dir + folder_name + "std.png")
    #     plt.close((fig))
    # R_f_list = [posterior_analysis.cov2cor(torch.mm(L_f, L_f.t()).numpy()) for L_f in L_f_list] # array
    # if verbose:
    #     for i in range(M):
    #         for j in range(i, M):
    #             fig = plt.figure()
    #             R_ij = np.stack([R_f[i,j] for R_f in R_f_list])
    #             plt.plot(x.numpy(), R_ij)
    #             plt.savefig(save_dir + folder_name + "true_log_R_{}{}.png".format(i, j))
    #             plt.close(fig)

    # Generate L_vecs (Approach 2) Approach 2 generates specific covariance process
    # generate std process
    stds = torch.stack([1+x**2, 2-x**2], axis=1)
    # Sigma_std = kernels.RBF_cov(x.view([-1, 1]), alpha=1., beta=1.)
    # std_dist = MultivariateNormal(loc=0 * torch.ones(N).type(settings.torchType), covariance_matrix=Sigma_std)
    # stds = torch.exp(std_dist.sample([M]).t()) # size N, 2
    std_array = stds.numpy()
    if verbose:
        fig = plt.figure()
        for m in range(M):
            plt.plot(x.numpy(), std_array[:, m], label="Dim {}".format(m+1))
        plt.legend()
        plt.savefig(save_dir + folder_name + "std.png")
        plt.close(fig)
    # generate correlation process using cos function
    cors = torch.cos(x*np.pi)
    if verbose:
        fig = plt.figure()
        plt.plot(x.numpy(), cors.numpy())
        plt.savefig(save_dir + folder_name + "true_log_R_{}{}.png".format(0, 1))
        plt.close(fig)
    # generate covariance process via combining std process and correlation process
    L_f_list = []
    for n in range(N):
        D_f = torch.diag(stds[n, :])
        R_f = torch.eye(M).type(settings.torchType)
        R_f[0,1] = cors[n]
        R_f[1,0] = cors[n]
        B_f = D_f.mm(R_f).mm(D_f)
        L_f = torch.cholesky(B_f)
        L_f_list.append(L_f)
    L_vecs = torch.cat([L_f[[0,1,1], [0,0,1]] for L_f in L_f_list])

    # Generate sigma2_err
    # sigma2_err = 1./Gamma(a, b).sample()
    # sigma2_err = 1e-6
    sigma2_err = 1e-2
    # Generare y
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l)
    K_i = logpos.generate_K_index_SVC(L_f_list)
    neworder = torch.arange(N*M).view([N, M]).t().contiguous().view(-1)
    K_i = K_i[:, neworder][neworder]
    K = kronecker_operation.kronecker_product(torch.ones([M, M]).type(settings.torchType), K_x) * K_i
    torch.manual_seed(seed)
    y = MultivariateNormal(loc=torch.zeros(M*N).type(settings.torchType), covariance_matrix=K + sigma2_err *
                                            torch.diag(torch.ones(M*N).type(settings.torchType))).sample()
    Y = y.view([M, N])
    if verbose:
        fig = plt.figure()
        for m in range(M):
            plt.plot(x.numpy(), Y.numpy()[m, :], label="Dim {}".format(m+1))
        plt.legend()
        plt.savefig(save_dir + folder_name + "Y.png")
        plt.close(fig)

    with open(save_dir + folder_name + file_name, "wb") as res:
        pickle.dump([x.numpy(), l.numpy(), L_vecs.numpy(), sigma2_err, Y.numpy().T], res)
    return Y


if __name__ == "__main__":
    # M = 3
    # N = 150
    # x = np.linspace(0, 1, N)
    # F, K_M, K_N = SIM_MNDATA(M, N, x, pertentage=0.75)
    # # add noise
    # Y = F + np.random.randn(N,M)*2.
    # fig = plt.figure()
    # plt.scatter(x, Y[:, 0], color="b")
    # plt.scatter(x, Y[:, 1], color="r")
    # plt.scatter(x, Y[:, 2], color="g")
    # plt.plot(x, F)
    # plt.savefig("true_data.png")
    # plt.show()
    # with open("../data/simIV.pickle", "wb") as res:
    #     pickle.dump([x, Y, K_M, K_N], res)
    # with open("../data/sim.pickle", "rb") as res:
    #     x, Y, K_M, K_N = pickle.load(res)

    # # simulate stationary univariate data
    # N = 200
    # x = np.linspace(0, 5, N)
    # F, _ = SIM_USDATA(N, x)
    # fig = plt.figure()
    # plt.plot(x, F)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.savefig("USSIM.png")
    # # plt.close(fig)
    #
    # # simulate nonstationary univariate data
    # N = 200
    # x = np.linspace(0, 5, N)
    # F, _ = SIM_UNDATA(N, x)
    # fig = plt.figure()
    # plt.plot(x, F)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.savefig("UNSIM.png")

    # # simulation stationary multivariate data
    # N = 200
    # M = 3
    # x = np.linspace(0, 5, N)
    # F, K_M, K_N = SIM_MSDATA(M, N, x)
    # fig = plt.figure()
    # plt.plot(x, F[:, 0]+20, color="b", label="Task 1")
    # plt.plot(x, F[:, 1], color="r", label="Task 2")
    # plt.plot(x, F[:, 2]-20, color="g", label="Task 3")
    # s = np.max(F) - np.min(F) + 40
    # plt.ylim(np.min(F)-20, np.max(F)+20 + 0.2*s)
    # plt.legend(loc=1, fontsize=12)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.savefig("MSSIM.png")

    # # simulation stationary multivariate data
    # N = 200
    # M = 3
    # x = np.linspace(0, 5, N)
    # F, K_M, K_N = SIM_MNDATA(M, N, x)
    # fig = plt.figure()
    # plt.plot(x, F[:, 0]+40, color="b", label="Task 1")
    # plt.plot(x, F[:, 1], color="r", label="Task 2")
    # plt.plot(x, F[:, 2]-40, color="g", label="Task 3")
    # s = np.max(F) - np.min(F) + 80
    # plt.ylim(np.min(F) - 40, np.max(F) + 40 + 0.2 * s)
    # plt.legend(loc=1, fontsize=12)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.savefig("MNSIM.png")
    # pass

    # simulate nonstationary multivariate time series with varying covariance process
    N = 200
    M = 2
    hyper_pars = {"mu_tilde_l": -3, "alpha_tilde_l": 3., "beta_tilde_l": 0.4, "mu_L": 0., "alpha_L": 5., "beta_L": 1
        , "a": 1., "b": 1.}

    save_dir = "../res/"
    folder_name = "simulation/"
    ts = SIM_MNTS(M, N, save_dir=save_dir, folder_name=folder_name, file_name="sim_MNTS.pickle", seed=2222, verbose=True, **hyper_pars)
    N_rep = 100
    for n in range(N_rep):
        print(n)
        ts = SIM_MNTS(M, N, save_dir=save_dir, folder_name=folder_name, file_name="sim_MNTS_{}.pickle".format(n), seed=n, verbose=False, **hyper_pars)
    # ts = SIM_MNTS_S(M, N, save_dir=save_dir, folder_name=folder_name, file_name="sim_MNTS_S.pickle", seed=2222, verbose=True)
    # N_rep = 1000
    # for n in range(N_rep):
    #     print(n)
    #     ts = SIM_MNTS_S(M, N, save_dir=save_dir, folder_name=folder_name, file_name="sim_MNTS_S_{}.pickle".format(n), seed=n, verbose=False)

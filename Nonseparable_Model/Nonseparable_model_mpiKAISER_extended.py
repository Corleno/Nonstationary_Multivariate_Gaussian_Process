import numpy as np
import torch
from torch import optim, autograd
from torch.autograd import Variable
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import time

# import private library
import sys
sys.path.append("../../Hamiltonian_Monte_Carlo")
import HMC_Sampler

sys.path.append("..")
from Utility import logpos
from Utility import settings
from Utility import model_validation
from Utility import prediction
from Utility import visualization
from Utility import posterior_analysis
from Utility import preprocess_realdata
from Utility import utils
from Utility import empirical_estimation

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def load_syndata(test_size=0, directory="../data/sim/sim_MNTS.pickle"):
    # Load synthetic data
    with open(directory, "rb") as res:
        x, true_l, true_L_vecs, true_sigma2_err, Y = pickle.load(res)
    true_tilde_l = np.log(true_l)
    N, M = Y.shape
    true_L_vec_list = [true_L_vecs[n * int(M * (M + 1) / 2):(n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
    true_L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in true_L_vec_list]
    true_B_list = [np.matmul(L_f, L_f.T) for L_f in true_L_f_list]
    true_R_list = [posterior_analysis.cov2cor(B) for B in true_B_list]
    true_stds = np.stack([np.sqrt(np.diag(B)) for B in true_B_list])
    true_R_01 = np.cos(x * np.pi)
    fig = plt.figure()
    plt.plot(x, true_tilde_l)
    plt.savefig(save_dir + folder_name + subfolder_name + "true_log_l.png")
    fig = plt.figure()
    for m in range(M):
        plt.plot(x, true_stds[:,m], label="Dim {}".format(m))
    plt.savefig(save_dir + folder_name + subfolder_name + "true_std.png")
    fig = plt.figure()
    plt.plot(x, true_R_01)
    plt.savefig(save_dir + folder_name + subfolder_name + "true_RDim 1_DIM 2.png")
    fig = plt.figure()
    for m in range(M):
        plt.plot(x, Y[:,m], label="Dim {}".format(m))
    plt.savefig(save_dir + folder_name + subfolder_name + "true_data.png")
    trend = 0.
    scale = 1.
    x_scale = 1.
    if test_size == 0:
        x_train = torch.from_numpy(x).type(settings.torchType)
        Y_train = torch.from_numpy(Y).type(settings.torchType)
        x_test = None
        Y_test = None
    else:
        from sklearn.model_selection import train_test_split
        x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=test_size, random_state=42)
        # reorder x and Y
        x_train_index = np.argsort(x_train)
        x_test_index = np.argsort(x_test)
        x_train = x_train[x_train_index]
        Y_train = Y_train[x_train_index]
        x_test = x_test[x_test_index]
        Y_test = Y_test[x_test_index]
        # convert numpy to torch
        x_train = torch.from_numpy(x_train).type(settings.torchType)
        Y_train = torch.from_numpy(Y_train).type(settings.torchType)
        x_test = torch.from_numpy(x_test).type(settings.torchType)
        Y_test = torch.from_numpy(Y_test).type(settings.torchType)
    return x_train, Y_train, x_test, Y_test, trend, scale, x_scale

# def load_kaiser_data(data_dir="../data/KAISER/", ID_dir="../data/KAISER/IDs_sampled.pickle", group="sepsis", random_seed=22):
#     # with open(ID_dir, "rb") as res:
#     #     ID_sepsis, ID_nonsepsis = pickle.load(res)
#     ID_sepsis = ["41283451", "41304767", "41309821", "43215388"]
#     ID_nonsepsis = ["43120202", "43120381", "43121070", "43122359"]
#     ID_illustration = ["12978238", "12986958", "13296382", "41168468", "13039442"]
#     if group=="sepsis":
#         with open(data_dir + "sepsis/ID_{}.pickle".format(ID_sepsis[rank]), "rb") as res:
#             origx, origY, attributes = pickle.load(res)
#     if group=="nonsepsis":
#         with open(data_dir + "nonsepsis/ID_{}.pickle".format(ID_nonsepsis[rank]), "rb") as res:
#             origx, origY, attributes = pickle.load(res)
#     if group=="illustration":
#         with open(data_dir + "illustration/ID_{}.pickle".format(ID_illustration[rank]), "rb") as res:
#             origx, origY, attributes = pickle.load(res)
#     Y, trend, scale = preprocess_realdata.orig2adj(origx, origY)
#     x_scale = np.max(origx)
#     x = origx / x_scale
#     x_train, x_test, Y_train, Y_test = utils.data_split(x, Y, test_size=0.33, random_state=random_seed)
#     # convert numpy to torch
#     x_train = torch.from_numpy(x_train).type(settings.torchType)
#     Y_train = torch.from_numpy(Y_train).type(settings.torchType)
#     x_test = torch.from_numpy(x_test).type(settings.torchType)
#     Y_test = torch.from_numpy(Y_test).type(settings.torchType)
#     return x_train, x_test, Y_train, Y_test, trend, scale, x_scale, attributes
#

def load_kaiser_data(data_dir="../data/KAISER/", ID_dir=None, group="sepsis", random_seed=22):
    global subfolder_name, ID
    if ID_dir is None:
        ID_sepsis = ["41283451", "41304767", "41309821", "43215388"]
        # ID_sepsis = ["15649333", "16719597"]
        # ID_nonsepsis = ["43120202", "43120381", "43121070", "43122359"]
        ID_nonsepsis = ["28722559", "18566513", "22080788", "27452122"]
    else:
        with open(ID_dir, "rb") as res:
            ID_sepsis, ID_nonsepsis = pickle.load(res)
    ID_illustration = ["12978238", "12986958", "13296382", "41168468"]
    if group=="sepsis":
        ID = ID_sepsis[rank]
        subfolder_name = "ID_{}/".format(ID)
        with open(data_dir + "sepsis/ID_{}.pickle".format(ID_sepsis[rank]), "rb") as res:
            origx, origY, attributes = pickle.load(res)
    if group=="nonsepsis":
        ID = ID_nonsepsis[rank]
        subfolder_name = "ID_{}/".format(ID)
        with open(data_dir + "nonsepsis/ID_{}.pickle".format(ID_nonsepsis[rank]), "rb") as res:
            origx, origY, attributes = pickle.load(res)
    if group=="illustration":
        ID = ID_illustration[rank]
        subfolder_name = "ID_{}/".format(ID)
        with open(data_dir + "illustration/ID_{}.pickle".format(ID_illustration[rank]), "rb") as res:
            origx, origY, attributes = pickle.load(res)
    Y, trend, scale = preprocess_realdata.orig2adj(origx, origY)
    x_scale = np.max(origx)
    x = origx / x_scale
    x_train, x_test, Y_train, Y_test = utils.data_split(x, Y, test_size=0.33, random_state=random_seed)
    # convert numpy to torch
    x_train = torch.from_numpy(x_train).type(settings.torchType)
    Y_train = torch.from_numpy(Y_train).type(settings.torchType)
    x_test = torch.from_numpy(x_test).type(settings.torchType)
    Y_test = torch.from_numpy(Y_test).type(settings.torchType)
    return x_train, x_test, Y_train, Y_test, trend, scale, x_scale, attributes

def vec2pars_SVC(pars, N, M):
    """
    covert parameter vector to parameters
    :param pars: parameter vector
    :param N: number of input vectors
    :param M: number of tasks
    :return: three vectorized parameters
    """
    tilde_l = pars[:N]
    L_vecs = pars[N: N + N * int(M * (M + 1) / 2)]
    tilde_sigma2_err = pars[-N*M:]
    return tilde_l, L_vecs, tilde_sigma2_err

def nlogpos_obj_SVC(pars, Y, x, mu_tilde_l=0., alpha_tilde_l=5., beta_tilde_l=1., mu_L=0., alpha_L=5., beta_L=1., a=1, b=1, verbose=False, Prior=True):
    """
    Objective function w.r.t log posterior for SVC version
    :param pars: parameters including tilde_l, uL_vec, ilde_sigma2_err
    :param Y: 2d tensor with dim N, M
    :param x: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l: hyper-parametes of GP for tilde_l
    :param mu_L, alpha_L, beta_L: hyper-parameters of GP for L_ij
    :param a, b: hyper-parameters for sigma2_err
    :return: scalar tensor
    """
    N, M = Y.size()
    tilde_l, uL_vecs, tilde_sigma2_err = vec2pars_SVC(pars, N, M)
    if verbose:
        res, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err = logpos_SVC(tilde_l, uL_vecs,
                                                                                                    tilde_sigma2_err, Y,
                                                                                                    x, mu_tilde_l,
                                                                                                    alpha_tilde_l,
                                                                                                    beta_tilde_l, mu_L,
                                                                                                    alpha_L, beta_L, a,
                                                                                                    b, verbose, Prior)
        return -res, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err
    else:
        return -logpos_SVC(tilde_l, uL_vecs, tilde_sigma2_err, Y, x, mu_tilde_l, alpha_tilde_l, beta_tilde_l,
                             mu_L, alpha_L, beta_L, a, b, verbose, Prior)

def logpos_SVC(tilde_l, uL_vecs, tilde_sigma2_err, Y, x, mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_L, alpha_L, beta_L,mu_error, alpha_err, beta_error, verbose=False, Prior=True):
    """
    Compute the log joint posterior distribution
    :param tilde_l: 1d tensor with length N
    :param uL_vecs: 1d tensor with length NM(M+1)/2
    :param tilde_sigma2_err: scalar tensor
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :param mu_tilde_l, alpha_tilde_l, beta_tilde_l, mu_tilde_sigma, alpha_tilde_sigma, beta_tilde_sigma, a, b: hyper-parameters
    :return: scalar tensor
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L_vecs = utils.uLvecs2Lvecs(uL_vecs, N, M)
    L_vec_list = [L_vecs[n * int(M * (M + 1) / 2): (n+1) * int(M * (M + 1) / 2)] for n in range(N)]
    L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list]
    l = torch.exp(tilde_l)
    sigma2_err = torch.exp(tilde_sigma2_err)
    res = 0
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), ell1=l) # dim: N, N
    K_i = generate_K_index_SVC(L_f_list) # dim: NM , NM
    neworder = torch.arange(N * M).view([N, M]).t().contiguous().view(-1)
    K_i = K_i[:, neworder][neworder] # dim: MN. MN
    K = kronecker_operation.kronecker_product(torch.ones([M, M]).type(settings.torchType), K_x) * K_i
    # print("K_x:", K_x, "K_i:", K_i, "K:", K)
    # Compute log likelihood
    invS = torch.inverse(K + torch.diag(sigma2_err))
    logdetS = torch.logdet(K + torch.diag(sigma2_err))
    loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    res += loglik
    # Compute log prob for tilde_l
    Sigma_l = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_tilde_l, beta=beta_tilde_l)
    log_prior_tilde_l = MultivariateNormal(mu_tilde_l*torch.ones_like(x), covariance_matrix=Sigma_l).log_prob(tilde_l)
    if Prior:
        res += log_prior_tilde_l
    # Compute log prob for uL_vecs
    Sigma_L = kernels.RBF_cov(x.view([-1, 1]), alpha=alpha_L, beta=beta_L)
    order_matrix = torch.arange(N*int(M*(M+1)/2)).view([N, int(M*(M+1)/2)])

    log_prior_uL_vecs = torch.sum(torch.stack([MultivariateNormal(mu_L*torch.ones_like(x), covariance_matrix=Sigma_L).log_prob(uL_vecs[order_matrix[:, m]]) for m in range(int(M*(M+1)/2))]))
    #import pdb
    #pdb.set_trace()
    if Prior:
        res += log_prior_uL_vecs
    # Compute log prob for tilde_sigma2_err
    Sigma_err = kernel.RBF_cov(x.view([-1, 1]), alpha=alpha_err, beta=beta_err)
    log_prior_sigma2_err = 0
    for m in range(M):
        log_prior_sigma2_err = MultivariateNormal(mu_error*torch.ones_like(x), covariance_matrix=Sigma_err).log_prob(tilde_sigma2_err[m*N:m*(N+1)])
    # log_prior_sigma2_err = distributions.gamma_logpdf(sigma2_err, alpha=a, beta=b)
    if Prior:
        res += log_prior_sigma2_err
        # Add jacobian
        res += tilde_sigma2_err
    if verbose:
        return res, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err
    else:
        return res

def train(x, Y, N_opt=None, err_opt=None, N_hmc=1000, init_position=None, M_hmc=None, step_size=1e-4, adaptive_step_size=True, num_steps_in_leap=20, learning_rate=1e-1, do_initialization=True, use_separable_res=False, use_empirical_res=False, use_random_res=False, use_combined_res=False, use_last_res=False, do_MAP=True, do_HMC=True, hyper_pars=None, save_dir=None, folder_name=None, folder_name_stationary=None, folder_name_separable=None, subfolder_name=None, verbose=False):
    """
    Using non-stationary multivariate Gaussian process to train data (x, Y)
    :param x: 1D tensor with length N
    :param Y: 2D tensor with dim N. M
    :param N_opt: number of iterations in the optimization
    :return: est_tilde_l, est_L_vecs, est_tilde_sigma2_err
    """
    # print("start trainining!")
    N, M = Y.size()
    if do_initialization:
        # use separable initializaion
        if use_separable_res:
            subname="separable"
            if subfolder_name is None:
                with open(save_dir + folder_name_separable + "MAP.dat", "rb") as res:
                    initPars = pickle.load(res)
            else:
                with open(save_dir + folder_name_separable + subfolder_name + "MAP.dat", "rb") as res:
                    initPars = pickle.load(res)
            tilde_l = torch.from_numpy(initPars[0: N]).type(settings.torchType)
            # tilde_l += torch.randn(N).type(settings.torchType)*1e-3
            tilde_sigma = torch.tensor(initPars[N: 2*N]).type(settings.torchType)
            L_vec = torch.from_numpy(initPars[2*N: -1]).type(settings.torchType)
            L_f = utils.vec2lowtriangle(L_vec, M)
            # print("initialized B_f: {}".format(torch.matmul(L_f, L_f.t())))
            L_vecs = torch.cat([L_vec*s for s in torch.exp(tilde_sigma)])
            # L_vecs += torch.randn(int(M*(M+1)/2)*N).type(settings.torchType)*1e-3
            tilde_sigma2_err = torch.from_numpy(initPars[-1]*np.ones(N*M)).type(settings.torchType)
        # use empirical initialization
        if use_empirical_res:
            subname="empirical"
            if subfolder_name is None:
                with open(save_dir + folder_name + "empirical_est.pickle", "rb") as res:
                    est_tilde_l, smooth_tilde_l, est_L_vecs, est_tilde_sigma2_err = pickle.load(res)
            else:
                with open(save_dir + folder_name + subfolder_name + "empirical_est.pickle", "rb") as res:
                    est_tilde_l, smooth_tilde_l, est_L_vecs, est_tilde_sigma2_err = pickle.load(res)
            tilde_l = torch.from_numpy(est_tilde_l).type(settings.torchType)
            L_vecs = torch.from_numpy(est_L_vecs).type(settings.torchType)
            tilde_sigma2_err = torch.from_numpy(est_tilde_sigma2_err*np.ones(N*M)).type(settings.torchType)
        # use random initialization
        if use_random_res:
            subname="random"
            tilde_l = -4*torch.ones(N).type(settings.torchType)
            uL_vecs = torch.randn(int(M*(M+1)/2)*N).type(settings.torchType)
            L_vecs = utils.uLvecs2Lvecs(uL_vecs, N, M)
            tilde_sigma2_err = torch.log(torch.rand(N*M)).type(settings.torchType)
        # use combined initialization
        if use_combined_res:
            subname="combined"
            if subfolder_name is None:
                with open(save_dir + folder_name_stationary + "MAP.dat", "rb") as res:
                    initPars = pickle.load(res)
            else:
                # import pdb
                # pdb.set_trace()
                with open(save_dir + folder_name_stationary + subfolder_name + "MAP.dat", "rb") as res:
                    initPars = pickle.load(res)
            if subfolder_name is None:
                with open(save_dir + folder_name + "empirical_est.pickle", "rb") as res:
                    est_tilde_l, smooth_tilde_l, est_L_vecs, est_tilde_sigma2_err = pickle.load(res)
            else:
                with open(save_dir + folder_name + subfolder_name + "empirical_est.pickle", "rb") as res:
                    est_tilde_l, smooth_tilde_l, est_L_vecs, est_tilde_sigma2_err = pickle.load(res) 
            tilde_l = torch.from_numpy(initPars[0] * np.ones(N) + 0.1 * np.random.randn(N)).type(settings.torchType)
            L_vecs = torch.from_numpy(est_L_vecs).type(settings.torchType)
            tilde_sigma2_err = torch.from_numpy(est_tilde_sigma2_err*np.ones(N*M)).type(settings.torchType)
        if use_last_res:
            subname="last"
            if subfolder_name is None:
                with open(save_dir + folder_name + "MAP.dat", "rb") as res:
                    initPars = pickle.load(res)
            else:
                with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
                    initPars = pickle.load(res)
            init_tilde_l, init_uL_vecs, init_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(initPars, N)
            tilde_l = torch.from_numpy(init_tilde_l).type(settings.torchType)
            uL_vecs = torch.from_numpy(init_uL_vecs).type(settings.torchType)
            tilde_sigma2_err = torch.from_numpy(init_tilde_sigma2_err).type(settings.torchType)
    # Pars = Variable(torch.cat([tilde_l, L_vecs, tilde_sigma2_err.view(1)]), requires_grad=True)
    
    tilde_l = torch.ones(N).type(settings.torchType) * -5.2
    tilde_l = Variable(tilde_l, requires_grad=True)


    try:
        uL_vecs = Variable(uL_vecs, requires_grad=True)
    except:
        uL_vecs = Variable(utils.Lvecs2uLvecs(L_vecs, N, M), requires_grad=True)
    tilde_sigma2_err = Variable(tilde_sigma2_err, requires_grad=True)
    Pars = torch.cat([tilde_l, uL_vecs, tilde_sigma2_err])
    import pdb
    pdb.set_trace()
    # print(Pars)
    # print("Negative log of posterior: {}".format(nlogpos_obj_SVC(Pars, Y, x, verbose=True, **hyper_pars)))

    # print("do_map: ", do_MAP)
    if do_MAP:
        # MAP inference
        # print("start MAP")
        # optimizer = optim.Adam([{'params': tilde_l, 'lr': 1e-1}, {"params": [uL_vecs, tilde_sigma2_err], 'lr': 1e-1}])
        optimizer = optim.Adam([{'params': tilde_l, 'lr': learning_rate}, {"params": [uL_vecs, tilde_sigma2_err], 'lr': learning_rate}])
        target_value_hist = np.zeros(N_opt)
        if N_opt is not None:
            for i in range(N_opt):
                optimizer.zero_grad()
                # with autograd.detect_anomaly():
                #     Pars = torch.cat([tilde_l, uL_vecs, tilde_sigma2_err.view(1)])
                #     NegLog, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err = \
                #         nlogpos_obj_SVC(Pars, Y, x, **hyper_pars, verbose=True)
                #     NegLog.backward()
                Pars = torch.cat([tilde_l, uL_vecs, tilde_sigma2_err])
                NegLog, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err = \
                    nlogpos_obj_SVC(Pars, Y, x, **hyper_pars, verbose=True)
                NegLog.backward()
                # gradient correction
                # print("Pars: {}, grad: ".format(Pars, Pars.grad))
                # Pars.grad.data[torch.isnan(Pars.grad)] = 0
                optimizer.step()
                if verbose and (i % 100 == 99):
                    print(
                        "{}/{} iteration with target value {}, loglik: {},s log_prior_tilde_l: {}, log_prior_uL_vecs: {}, log_prior_sigma2_err: {}".format(
                            i+1, N_opt, NegLog, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err))
                    # print(tilde_sigma2_err)
                target_value_hist[i] = -NegLog

                # save results every 100 iterations
                if i % 100 == 99:
                    # Save MAP estimate results
                    Pars = torch.cat([tilde_l, uL_vecs, tilde_sigma2_err.view(1)])
                    if subfolder_name is None:
                        with open(save_dir + folder_name + "MAP_" + subname + ".dat", "wb") as res:
                            pickle.dump(Pars.data.numpy(), res)
                    else:
                        with open(save_dir + folder_name + subfolder_name + "MAP_" + subname + ".dat", "wb") as res:
                            pickle.dump(Pars.data.numpy(), res)
        if err_opt is not None:
            gap = np.inf
            curr_obj = np.inf
            i = 0
            ts = time.time()
            while gap > err_opt:
                i += 1
                optimizer.zero_grad()
                with autograd.detect_anomaly():
                    Pars = torch.cat([tilde_l, uL_vecs, tilde_sigma2_err.view(1)])
                    NegLog, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err = \
                        nlogpos_obj_SVC(Pars, Y, x, **hyper_pars, verbose=True)
                    NegLog.backward()
                optimizer.step()
                if i % 100 == 99:
                    gap = curr_obj - NegLog
                    grad = np.concatenate([tilde_l.grad.data, uL_vecs.grad.data, tilde_sigma2_err.grad.data])
                    print(np.linalg.norm(grad, ord=np.inf))
                    # upate objective
                    curr_obj = NegLog
                if verbose and (i % 100 == 99):
                    print("{}/{} iteration with target value {} which costs {}s, loglik: {}, log_prior_tilde_l: {}, log_prior_uL_vecs: {}, log_prior_sigma2_err: {}".format(
                            i+1, N_opt, NegLog, time.time()-ts, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err))
        fig = plt.figure()
        plt.plot(target_value_hist)
        if subfolder_name is None:
            plt.savefig(save_dir + folder_name + "target_trace.png")
        else:
            plt.savefig(save_dir + folder_name + subfolder_name + "target_trace.png")
        plt.close(fig)
        # Save MAP estimate results
        Pars = torch.cat([tilde_l, uL_vecs, tilde_sigma2_err.view(1)])
        if subfolder_name is None:
            with open(save_dir + folder_name + "MAP_" + subname + ".dat", "wb") as res:
                pickle.dump(Pars.data.numpy(), res)
        else:
            with open(save_dir + folder_name + subfolder_name + "MAP_" + subname + ".dat", "wb") as res:
                pickle.dump(Pars.data.numpy(), res)
        return NegLog

    if do_HMC:
        # HMC inference
        if init_position is None:
            # Load MAP result
            if subfolder_name is None:
                with open(save_dir + folder_name + "MAP.dat", "rb") as res:
                    estPars = pickle.load(res)
            else:
                with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
                   estPars = pickle.load(res)
            init_position = estPars
        hmc = HMC_Sampler.HMC_sampler.sampler(sample_size=N_hmc, potential_func=nlogpos_obj_SVC, init_position=init_position,
                                              step_size=step_size, adaptive_step_size=adaptive_step_size, num_steps_in_leap=num_steps_in_leap, M=M_hmc, x=x, Y=Y, duplicate_samples=True, TensorType=settings.torchType,
                                              **hyper_pars)
        sample, _ = hmc.main_hmc_loop()
        if subfolder_name is None:
            with open(save_dir + folder_name + "HMC_sample.pickle", "wb") as res:
                pickle.dump(sample, res)
        else:
            with open(save_dir + folder_name + subfolder_name + "HMC_sample.pickle", "wb") as res:
                pickle.dump(sample, res)
        # print(np.mean(sample, axis=0))
        # print(np.std(sample, axis=0))
        return sample

if __name__ == "__main__":
    rank = 4
    parser = argparse.ArgumentParser() 
    parser.add_argument("--group", type=str, default="sepsis")
    args = parser.parse_args()
    save_dir = "../res/"
    group = args.group
    folder_name_stationary = "kaiser_stationary/"+group+"/"
    folder_name_separable = "kaiser_separable/"+group+"/"
    folder_name = "kaiser_nonseparable/"+group+"/"

    do_empirical_estimation = True
    do_train = True
    do_initialization = True
    do_continue_train = False
    do_MAP = True
    do_HMC = False
    do_use_mass_matrix = True
    do_map_analysis = True
    do_hmc_analysis = False
    do_post_analysis = False
    do_bayes_pred = False
    do_freq_pred = True
    do_pred_grids = True
    do_model_evaluation = True
    do_pred_test = True
    do_vis_bayes = False
    do_vis_freq = True

    x, x_test, Y, Y_test, trend, scale, x_scale, attributes = load_kaiser_data(group=group, random_seed=222)

    # subfolder_name = "{}/".format(rank)
    if not os.path.exists(save_dir + folder_name + subfolder_name):
        os.mkdir(save_dir + folder_name + subfolder_name)

    import pdb; pdb.set_trace()

    # Initialization
    N, M = Y.shape
    hyper_pars = {"mu_tilde_l": 0., "alpha_tilde_l": 10., "beta_tilde_l": 1, "mu_L": 0.,
                  "alpha_L": 1., "beta_L": 1, "a": 1, "b": 1}

    if do_empirical_estimation:
        est_sigmas, est_ls, smooth_ls, est_stds, est_R, est_B, est_L_vecs, est_tilde_sigma2_err = empirical_estimation.local_estimation(x.numpy(), Y.numpy())
        empirical_estimation.visualization(x.numpy(), Y.numpy(), est_ls, smooth_ls, est_stds, est_R, est_L_vecs, save_dir=save_dir,
                      folder_name=folder_name, subfolder_name=subfolder_name)
        empirical_estimation.save_res(est_ls, smooth_ls, est_L_vecs, est_tilde_sigma2_err, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name)

    if do_train:
        if do_MAP:
            # train non-stationary model
            if do_continue_train:
                try:
                    NegLog = train(x, Y, N_opt=20000, N_hmc=1000, learning_rate=1e-2, do_initialization=do_initialization, use_last_res=True, do_MAP=do_MAP, do_HMC=do_HMC, 
                    hyper_pars=hyper_pars, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, folder_name_separable=folder_name_separable, verbose=True)
                except:
                    print("optimization failed for continuous train.")
                with open(save_dir + folder_name + subfolder_name + "MAP_last.dat", "rb") as res:
                    estPars = pickle.load(res)
            else:
                try:
                    NegLog_combined = train(x, Y, N_opt=10000, N_hmc=100, learning_rate=1e-2, do_initialization=do_initialization, use_combined_res=True, do_MAP=do_MAP, do_HMC=do_HMC,
                    hyper_pars=hyper_pars, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, folder_name_stationary=folder_name_stationary, folder_name_separable=folder_name_separable, verbose=True)
                except:
                    print("optimization failed for combined.")
                    NegLog_combined = np.inf
                # NegLog_combined = train(x, Y, N_opt=10000, N_hmc=100, learning_rate=1e-2, do_initialization=do_initialization, use_combined_res=True, do_MAP=do_MAP, do_HMC=do_HMC,
                #     hyper_pars=hyper_pars, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, folder_name_separable=folder_name_separable, verbose=True)

                # NegLog_combined = np.inf
                # try:
                #     NegLog_empirical = train(x, Y, N_opt=20000, N_hmc=100, learning_rate=0.01, do_initialization=do_initialization, use_empirical_res=True, do_MAP=do_MAP, do_HMC=do_HMC,
                #         hyper_pars=hyper_pars, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, folder_name_separable=folder_name_separable, verbose=True)
                # except:
                #     print("optimization failed for empirical.")
                #     NegLog_empirical = np.inf
                NegLog_empirical = np.inf
                # try:
                #     NegLog_separable = train(x, Y, err_opt=10, N_hmc=100, learning_rate=1e-2, do_initialization=do_initialization, use_separable_res=True, do_MAP=do_MAP, do_HMC=do_HMC,
                #         hyper_pars=hyper_pars, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, folder_name_separable=folder_name_separable, verbose=True)
                # :
                #     NegLog_separable = np.inf
                NegLog_separable = np.inf
                optimal_approach = ["combined", "empirical", "separable"][np.argmin([NegLog_combined, NegLog_empirical, NegLog_separable])]
                with open(save_dir + folder_name + subfolder_name + "MAP_" + optimal_approach + ".dat", "rb") as res:
                    estPars = pickle.load(res)
            
            with open(save_dir + folder_name + subfolder_name + "MAP.dat", "wb") as res:
                pickle.dump(estPars, res)
            print("training completed.")
        if do_HMC:
            if do_use_mass_matrix:
            # import mass matrix

                if subfolder_name is None:
                    with open(save_dir + folder_name + "HMC_sample_res.pickle", "rb") as res:
                        sample_cov, est_pars = pickle.load(res)
                else:
                    with open(save_dir + folder_name + subfolder_name + "HMC_sample_res.pickle", "rb") as res:
                        sample_cov, est_pars = pickle.load(res)
                init_position = est_pars
                step_size=1e-1
                adaptive_step_size=False
                num_steps_in_leap=5
                # M_hmc = np.linalg.pinv(sample_cov/np.sqrt(np.outer(np.diag(sample_cov), np.diag(sample_cov))))
                print("sample_cov: {}".format(sample_cov))
                sample_cov = sample_cov + 1e-10*np.eye(sample_cov.shape[0])
                M_hmc = np.linalg.inv(sample_cov)
                print("M_hmc: {}".format(M_hmc))
                print("init_position: {}".format(init_position))
                # import pdb
                # pdb.set_trace()
            else:
                print("Mass matrix is an identity matrix.")
                M_hmc = None
                init_position = None
                step_size=1e-4
                adaptive_step_size=False
                num_steps_in_leap=20
                init_position=None
            sample = train(x, Y, learning_rate=0.01, N_opt=10000, N_hmc=10000, M_hmc=M_hmc, init_position=init_position, step_size=step_size, adaptive_step_size=adaptive_step_size, num_steps_in_leap=num_steps_in_leap, do_MAP=do_MAP, do_HMC=do_HMC, use_last_res=True, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, folder_name_stationary=folder_name_stationary, hyper_pars=hyper_pars, verbose=True)

    if do_map_analysis:
        # Load MAP result
        with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_uL_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        est_L_vecs = utils.uLvecs2Lvecs(est_uL_vecs, N, M)
        est_L_vec_list = [est_L_vecs[n * int(M * (M + 1) / 2):(n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
        est_L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in est_L_vec_list]
        est_B_list = [np.matmul(L_f, L_f.T) for L_f in est_L_f_list]
        est_R_list = [posterior_analysis.cov2cor(B) for B in est_B_list]
        est_stds = np.stack([np.sqrt(np.diag(B)) for B in est_B_list])
        print(est_tilde_l)
        print(x.numpy())
        # plot estimated log length-scale process
        fig = plt.figure()
        plt.plot(x.numpy(), est_tilde_l)
        plt.savefig(save_dir + folder_name + subfolder_name + "est_log_l.png")
        plt.close(fig)
        # plot estimated std process
        fig = plt.figure()
        for m in range(M):
            plt.plot(x.numpy(), est_stds[:, m], label="Dim {}".format(m+1))
        plt.legend()
        plt.savefig(save_dir + folder_name + subfolder_name + "est_std.png")
        plt.close((fig))
        # plot correlation process
        for i in range(M):
            for j in range(i+1, M):
                fig = plt.figure()
                R_ij = np.stack([R_f[i, j] for R_f in est_R_list])
                plt.plot(x.numpy(), R_ij)
                plt.savefig(save_dir + folder_name + subfolder_name + "est_log_R_{}{}.png".format(i, j))
                plt.close(fig)
        # print error measurement
        print("sigma2_error: ", est_tilde_sigma2_err)

    if do_hmc_analysis:
        if subfolder_name is None:
            with open(save_dir + folder_name + "HMC_sample.pickle", "rb") as res:
                sample = pickle.load(res)
        else:
            with open(save_dir + folder_name + subfolder_name + "HMC_sample.pickle", "rb") as res:
                sample = pickle.load(res)
        # check convergence
        fig = plt.figure()
        plt.plot(sample[:, 0])
        plt.savefig("log_l_trace0.png")
        from statsmodels.graphics.tsaplots import plot_acf
        fig = plt.figure()
        plot_acf(sample[:, 0])
        plt.savefig("log_l_trace0_acf.png")
        tilde_ls, tilde_sigmas, uL_vecs, tilde_sigma2_errs = sample[:, :N], sample[:, N:2*N], sample[:, 2*N:-1], sample[:, -1] 
        tilde_ls = torch.from_numpy(tilde_ls).type(settings.torchType)
        tilde_sigmas = torch.from_numpy(tilde_sigmas).type(settings.torchType)
        uL_vecs = torch.from_numpy(uL_vecs).type(settings.torchType)
        tilde_sigma2_errs = torch.from_numpy(tilde_sigma2_errs).type(settings.torchType)
        # compute the sample covariance matrix of parameters
        sample_cov = np.cov(sample.T)
        est_pars = sample[-1, :]
        if subfolder_name is None:
            with open(save_dir + folder_name + "HMC_sample_res.pickle", "wb") as res:
                pickle.dump([sample_cov, est_pars], res)
        else:
            with open(save_dir + folder_name + subfolder_name + "HMC_sample_res.pickle", "wb") as res:
                pickle.dump([sample_cov, est_pars], res)
        import pdb
        pdb.set_trace()

    if do_post_analysis:
        # Posterior analysis
        with open(save_dir + folder_name + subfolder_name + "HMC_sample.pickle", "rb") as res:
            sample = pickle.load(res)
        tilde_l_hist, uL_vecs_hist, tilde_sigma2_err_hist = posterior_analysis.vec2pars_SVC(sample, N, M)
        # ....
        posterior_analysis.visualization_pos(x.numpy(), tilde_l_hist, L_vecs_hist=L_vecs_hist, N=N, M=M, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)

    grids = np.linspace(0., 1., 201)
    grids = torch.from_numpy(grids).type(settings.torchType)

    if do_bayes_pred:
        N_sample = 1000
        if subfolder_name is None:
            with open(save_dir + folder_name + "HMC_sample.pickle", "rb") as res:
                sample = pickle.load(res)
        else:
            with open(save_dir + folder_name + subfolder_name + "HMC_sample.pickle", "rb") as res:
                sample = pickle.load(res)
        tilde_ls, uL_vecs, tilde_sigma2_errs = posterior_analysis.vec2pars_SVC(torch.from_numpy(sample).type(settings.torchType), N, M)
        
        if do_pred_grids:
            # Predictive inference            
            sampled_ys = prediction.pointwise_predsample_inhomogeneous(tilde_ls, uL_vecs, tilde_sigma2_errs, Y, x, grids, N_sample=N_sample, **hyper_pars)
            gridy_quantiles, gridy_mean, gridy_std = np.percentile(sampled_ys, q = [2.5, 97.5], axis=1), np.mean(sampled_ys, axis=1), np.std(sampled_ys, axis=1)
            with open(save_dir + folder_name + subfolder_name + "pred_grid_hmc.pickle", "wb") as res:
                pickle.dump([gridy_quantiles, gridy_mean, gridy_std], res)
        if do_model_evaluation:
            # Predict testing data
            sampled_repys = prediction.test_predsample_inhomogeneous(tilde_ls, uL_vecs, tilde_sigma2_errs, Y, x, x, N_sample=N_sample, **hyper_pars)
            repy_quantiles, repy_mean, repy_std = np.percentile(sampled_repys, q = [2.5, 97.5], axis=1), np.mean(sampled_repys, axis=1), np.std(sampled_repys, axis=1)
            G = np.sum(np.linalg.norm(Y.numpy() - repy_mean, axis = 1)**2)
            P = np.sum(repy_std**2)
            D = G+P
            print("G", G)
            print("P", P)
            print("D", D)
            with open(save_dir + folder_name + subfolder_name + "model_evaluation_hmc.pickle", "wb") as res:
                pickle.dump([repy_quantiles, repy_mean, repy_std, G, P, D], res)
        if do_pred_test:
            # Prediction on testing data
            sampled_predys = prediction.test_predsample_inhomogeneous(tilde_ls, uL_vecs, tilde_sigma2_errs, Y, x, x_test, N_sample=N_sample, **hyper_pars)
            predy_quantiles, predy_mean, predy_std = np.percentile(sampled_predys, q = [2.5, 97.5], axis=1), np.mean(sampled_predys, axis=1), np.std(sampled_predys, axis=1)
            PMSE = np.sum(np.linalg.norm(Y_test.numpy() - predy_mean, axis = 1)**2)
            print("PMSE", PMSE)
            with open(save_dir + folder_name + subfolder_name + "pred_test_hmc.pickle", "wb") as res:
                pickle.dump([predy_quantiles, predy_mean, predy_std, PMSE], res)

    if do_freq_pred:
        ###### Frequentist Inference##########
        # Load MAP result
        with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_uL_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        est_uL_vecs = torch.from_numpy(est_uL_vecs).type(settings.torchType)
        est_L_vecs = utils.uLvecs2Lvecs(est_uL_vecs, N, M)
        est_tilde_l = torch.from_numpy(est_tilde_l).type(settings.torchType)
        est_tilde_sigma2_err = torch.tensor(est_tilde_sigma2_err).type(settings.torchType)

        if do_pred_grids:
            gridy_quantiles, gridy_mean, gridy_std = prediction.pointwise_predmap_inhomogeneous_sampling(100, est_tilde_l, est_uL_vecs, est_tilde_sigma2_err, Y, x, grids, **hyper_pars)
            with open(save_dir + folder_name + subfolder_name + "pred_grid_map.pickle", "wb") as res:
                pickle.dump([gridy_quantiles, gridy_mean, gridy_std], res)

            gridy_Y = np.stack([gridy_quantiles[:,0,:], gridy_mean, gridy_quantiles[:,1,:]])
            visualization.Plot_posterior(x.numpy(), Y.numpy(), grids.numpy(),
                                         gridy_Y,
                                         save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes, type="MAP")
        
        if do_model_evaluation:
            repy_quantiles, repy_mean, repy_std = prediction.test_predmap_inhomogeneous_sampling(100, est_tilde_l, est_uL_vecs, est_tilde_sigma2_err, Y, x, x, **hyper_pars)
            G = np.sum(np.linalg.norm(Y.numpy() - repy_mean, axis = 1)**2)
            P = np.sum(repy_std**2)
            D = G+P
            print("G", G)
            print("P", P)
            print("D", D)
            with open(save_dir + folder_name + subfolder_name + "model_evaluation_map.pickle", "wb") as res:
                pickle.dump([repy_quantiles, repy_mean, repy_std, G, P, D], res)

        if do_pred_test:
            # Prediction on testing data
            predy_quantiles, predy_mean, predy_std = prediction.test_predmap_inhomogeneous_sampling(100, est_tilde_l, est_uL_vecs, est_tilde_sigma2_err, Y, x, x_test, **hyper_pars)
            PMSE = np.sum(np.linalg.norm(Y_test.numpy() - predy_mean, axis = 1)**2)
            print("PMSE", PMSE)
            with open(save_dir + folder_name + subfolder_name + "pred_test_map.pickle", "wb") as res:
                pickle.dump([predy_quantiles, predy_mean, predy_std, PMSE], res)

    if do_vis_bayes:
        print("HMC result:")
        # Visualization
        with open(save_dir + folder_name + subfolder_name + "pred_grid_hmc.pickle", "rb") as res:
            gridy_quantiles, gridy_mean, gridy_std = pickle.load(res)
        with open(save_dir + folder_name + subfolder_name + "pred_test_hmc.pickle", "rb") as res:
            predy_quantiles, predy_mean, predy_std, _ = pickle.load(res)
        # import pdb
        # pdb.set_trace()
        gridy_Y = np.stack([gridy_quantiles[0], gridy_mean, gridy_quantiles[1]])
        predy_Y = np.stack([predy_quantiles[0], predy_mean, predy_quantiles[1]])

        # convert back to original data
        gridy_Y_orig = preprocess_realdata.adj2orig(gridy_Y, trend, scale)
        # print(sampled_y_quantile_orig.shape)
        Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
        predy_mean_orig = preprocess_realdata.adj2orig(predy_mean, trend, scale)
        predy_std_orig = predy_std*scale
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        visualization.Plot_posterior_trainandtest(x.numpy()*x_scale, Y_orig, grids.numpy()*x_scale, gridy_Y_orig, 
            x_test=x_test.numpy()*x_scale, Y_test=Y_test_orig, Y_pred=predy_mean_orig,
            save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes, type="HMC")
        # compute RMSE for tasks separately
        pred_test_rmse = utils.RMSE(predy_mean_orig, Y_test_orig, axis=0)
        print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(predy_mean_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(predy_mean_orig, predy_std_orig, Y_test_orig)
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))


        # ########### Visualization (Bayesian)
        # with open(save_dir + folder_name + "pred_res.pickle", "rb") as res:
        #     sampled_y_hist = pickle.load(res)
        # with open(save_dir + folder_name + "pred_test_res.pickle", "rb") as res:
        #     sampled_test_hist = pickle.load(res)
        # print(sampled_y_hist.size())
        # print(sampled_test_hist.size())
        # sampled_y_hist = sampled_y_hist.data.numpy()
        # sampled_y_quantile = visualization.samples2quantiles(sampled_y_hist)
        # print(sampled_y_quantile.shape)
        # # visualization.Plot_posterior_hadamard(x.data.numpy(), indx.data.numpy(), y.data.numpy(), grids.data.numpy(), sampled_y_quantile)
        # pred_test = torch.mean(sampled_test_hist, dim=1)
        #
        # # convert back to original scale
        # sampled_y_quantile_list = [sampled_y_quantile[:, :, m] for m in range(M)]
        # orig_sampled_y_quantile_list = preprocess_realdata.adj2orig_non(sampled_y_quantile_list, trend_list, scale_list)
        # # orig_sampled_y_quantile = np.stack(orig_sampled_y_quantile_list, axis=2)
        # pred_test_list = prediction.vec2list(pred_test.detach().numpy(), indx_test.detach().numpy())
        # orig_pred_test_list = preprocess_realdata.adj2orig_non(pred_test_list, trend_list, scale_list)
        # x_list = prediction.vec2list(x.detach().numpy(), indx.detach().numpy())
        # origx_list = [x_scale * origx for origx in x_list]
        # y_list = prediction.vec2list(y.detach().numpy(), indx.detach().numpy())
        # origy_list = preprocess_realdata.adj2orig_non(y_list, trend_list, scale_list)
        # x_test_list = prediction.vec2list(x_test.detach().numpy(), indx_test.detach().numpy())
        # origx_test_list = [x_scale * x_test for x_test in x_test_list]
        # y_test_list = prediction.vec2list(y_test.detach().numpy(), indx_test.detach().numpy())
        # origy_test_list = preprocess_realdata.adj2orig_non(y_test_list, trend_list, scale_list)
        #
        # visualization.Plot_posterior_trainandtest_non(origx_list, origy_list, grids.numpy()*x_scale, orig_sampled_y_quantile_list, origx_test_list, origy_test_list, orig_pred_test_list)
        #
        # pred_test_mse_list = [utils.MSE(origy_test_list[m], orig_pred_test_list[m]) for m in range(M)]
        # print("MSE_feature: {}".format(pred_test_mse_list))
        # pred_test_mse = utils.MSE(np.concatenate(origy_test_list), np.concatenate(orig_pred_test_list))
        # print("MSE = {}".format(pred_test_mse))
        pass

    if do_vis_freq:
        ############ Visualization (Frequentist)
        with open(save_dir + folder_name + subfolder_name + "pred_grid_map.pickle", "rb") as res:
            gridy_quantiles, gridy_mean, gridy_std = pickle.load(res)
        with open(save_dir + folder_name + subfolder_name + "pred_test_map.pickle", "rb") as res:
            predy_quantiles, predy_mean, predy_std, _= pickle.load(res)
        gridy_Y = np.stack([gridy_quantiles[:,0,:], gridy_mean, gridy_quantiles[:,1,:]])
        predy_Y = np.stack([predy_quantiles[:,0,:], predy_mean, predy_quantiles[:,1,:]])

        # convert back to original data
        gridy_Y_orig = preprocess_realdata.adj2orig(gridy_Y, trend, scale)
        # print(sampled_y_quantile_orig.shape)
        Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
        predy_mean_orig = preprocess_realdata.adj2orig(predy_mean, trend, scale)
        predy_std_orig = predy_std*scale
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        visualization.Plot_posterior_trainandtest(x.numpy()*x_scale, Y_orig, grids.numpy()*x_scale, gridy_Y_orig, 
            x_test=x_test.numpy()*x_scale, Y_test=Y_test_orig, Y_pred=predy_mean_orig,
            save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)
        # compute RMSE for tasks separately
        pred_test_rmse = utils.RMSE(predy_mean_orig, Y_test_orig, axis=0)
        print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(predy_mean_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(predy_mean_orig, predy_std_orig, Y_test_orig)
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))

        with open(save_dir + folder_name + subfolder_name + "pred_test_map_orig.pickle", "wb") as res:
            pickle.dump([pred_test_rmse, pred_test_lpd], res)

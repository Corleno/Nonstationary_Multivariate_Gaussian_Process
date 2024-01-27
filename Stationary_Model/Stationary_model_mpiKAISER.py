"""
pretrain the nonstationary multi-task model using stationary multi-task model
"""
import torch
from torch.autograd import Variable
import pickle
from torch import optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import argparse

# import private library
import sys
sys.path.append("../../Hamiltonian_Monte_Carlo")
import HMC_Sampler

# import private library
import sys
sys.path.append("..")
from Utility import settings
from Utility import utils
from Utility import kernels
from Utility import distributions
from Utility import kronecker_operation
from Utility import posterior_analysis
from Utility import visualization
from Utility import preprocess_realdata
from Utility import logpos
from Utility import prediction
from Utility import empirical_estimation

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def deviance_obj(pars, Y, x):
    """
    Objective function w.r.t. deviance
    :param pars: parameters including tilde_l tilde_sigma, L_vec, tilde_sigma2_err
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :return: scalar tensor
    """
    N, M = Y.size()
    tilde_l, tilde_sigma, L_vec, tilde_sigma2_err = vec2pars(pars, M)
    return deviance(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, Y, x)


def deviance(tilde_l, tilde_sigma, L_vec, tilde_sigma2_err, Y, x):
    """
    Deviance fuction
    :param tilde_l: scale tensor
    :param tilde_sigma: scale tensor
    :param L_vec: 1d tensor with length M(M+1)/2
    :param tilde_sigma_err: scalar tensor
    :param Y: 2d tensor with dim N by M
    :param x: 1d tensor with length N
    :return: scalar tensor
    """
    N, M = Y.size()
    y = Y.t().contiguous().view(-1)
    L = utils.vec2lowtriangle(L_vec, M)
    B_f = torch.mm(L, L.t())
    l = torch.exp(tilde_l * torch.ones(N)).type(settings.torchType)
    sigma = torch.exp(tilde_sigma * torch.ones(N).type(settings.torchType))
    sigma2_err = torch.exp(tilde_sigma2_err)
    K_x = kernels.Nonstationary_RBF_cov(x.view([-1, 1]), sigma1=sigma, ell1=l)
    # Compute log likelihood
    invS = kronecker_operation.kron_inv(sigma2_err, B_f, K_x)
    logdetS = kronecker_operation.kron_logdet(sigma2_err, B_f, K_x)
    loglik = distributions.multivariate_normal_logpdf(y, torch.zeros_like(y), logdetS, invS)
    dev = -2*loglik
    return dev


def train(x, Y, N_opt=None, err_opt=None, N_hmc=1000, learning_rate=1e-1, M_hmc=None, init_position=None, step_size=1e-2, adaptive_step_size=False, num_steps_in_leap=20, do_MAP=False, do_HMC=False, use_empirical_res=True, save_dir=None, folder_name=None, subfolder_name=None, verbose=False):
    """
    Using stationary multivariate Gaussian process to train data (x, Y)
    :param x: 1D tensor with length N
    :param Y: 2D tensor with dim N, M
    :param N_opt: number of iterations in the optimization
    :return: est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err
    """
    # Initialization
    hyper_pars = {"mu_tilde_l": 0, "sigma_tilde_l": 1., "a": 1, "b": 1, "c": 1.}
    N, M = Y.size()

    # # randomly initialize starting point
    # tilde_sigma2_err = torch.log(torch.rand(1))[0]
    # tilde_l = torch.randn(1)
    # tilde_sigma = torch.randn(1)
    # L_vec = torch.randn(int(M * (M + 1) / 2))
    # initialize starting point using informative informatio
    tilde_sigma = torch.FloatTensor([0]).type(settings.torchType) # fixed for correlation
    if use_empirical_res:
        if subfolder_name is None:
            with open(save_dir + folder_name + "empirical_est.pickle", "rb") as res:
                est_tilde_l, smooth_tilde_l, est_L_vecs, est_tilde_sigma2_err = pickle.load(res)
        else:
            with open(save_dir + folder_name + subfolder_name + "empirical_est.pickle", "rb") as res:
                est_tilde_l, smooth_tilde_l, est_L_vecs, est_tilde_sigma2_err = pickle.load(res)
        tilde_l = torch.from_numpy(np.array([np.mean(est_tilde_l)])).type(settings.torchType)
        L_vecs = torch.from_numpy(est_L_vecs).type(settings.torchType)
        uL_vecs = utils.Lvecs2uLvecs(L_vecs, N, M)
        uL_vec = torch.mean(uL_vecs.view([N, -1]), 0)
        tilde_sigma2_err = torch.from_numpy(np.array([est_tilde_sigma2_err])).type(settings.torchType)
    else:
        tilde_sigma2_err = torch.log(torch.tensor(0.1)).type(settings.torchType)
        tilde_l = torch.FloatTensor([-3]).type(settings.torchType)
        uL_vec = torch.rand(int(M * (M + 1) / 2)).type(settings.torchType)

    tilde_l = torch.FloatTensor([-2]).type(settings.torchType)
    tilde_l = Variable(tilde_l, requires_grad=True)
    uL_vec = Variable(uL_vec, requires_grad=True)
    tilde_sigma2_err = Variable(tilde_sigma2_err, requires_grad=True)
    Pars = torch.cat([tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err.view(1)])

    if verbose:
        print("Negative log of posterior: {}".format(logpos.nlogpos_obj_S(Pars, Y, x, verbose=True, **hyper_pars)))

    if do_MAP:
        # MAP inference
        optimizer = optim.Adam([tilde_l, uL_vec, tilde_sigma2_err], lr=0.1)
        
        if N_opt is not None:
            target_value_hist = np.zeros(N_opt)
            for i in range(N_opt):
                ts = time.time()
                optimizer.zero_grad()
                Pars = torch.cat([tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err.view(1)])
                # with autograd.detect_anomaly():
                #     NegLog, loglik, log_prior_tilde_l, log_prior_uL_vec, log_prior_sigma2_err = logpos.nlogpos_obj_S(Pars, Y, x, **hyper_pars, verbose=True)
                #     NegLog.backward()
                NegLog, loglik, log_prior_tilde_l, log_prior_uL_vec, log_prior_sigma2_err = logpos.nlogpos_obj_S(Pars, Y, x, **hyper_pars, verbose=True)
                NegLog.backward()        
                # Pars.grad.data[torch.isnan(Pars.grad)] = 0
                tilde_l.grad.data[torch.isnan(tilde_l.grad)] = 0
                uL_vec.grad.data[torch.isnan(uL_vec.grad)] = 0
                tilde_sigma2_err.grad.data[torch.isnan(tilde_sigma2_err.grad)] = 0
                optimizer.step()
                # print("{}th iteration with target value {}, deviance {} and Pars {}".format(i+1, -NegLog, deviance_obj(Pars, Y, x), Pars))
                if verbose and (i % 100 == 99):
                    print("{}/{}th iteration with target value {}.".format(i + 1, N_opt, NegLog))
                    print("loglik: {}, log_prior_tilde_l: {}, log_prior_uL_vec: {}, log_prior_sigma2_err: {}".format(loglik, log_prior_tilde_l, log_prior_uL_vec, log_prior_sigma2_err))
                target_value_hist[i] = -NegLog
            fig = plt.figure()
            plt.plot(target_value_hist)
            if subfolder_name is None:
                plt.savefig(save_dir + folder_name + "target_trace.png")
            else:
                plt.savefig(save_dir + folder_name + subfolder_name + "target_trace.png")
            plt.close(fig)

        if err_opt is not None:
            gap = np.inf
            curr_obj = np.inf
            i = 0
            while gap > err_opt:
                i += 1
                optimizer.zero_grad()
                Pars = torch.cat([tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err.view(1)])
                # with autograd.detect_anomaly():
                #     NegLog, loglik, log_prior_tilde_l, log_prior_uL_vec, log_prior_sigma2_err = logpos.nlogpos_obj_S(Pars, Y, x, **hyper_pars, verbose=True)
                #     NegLog.backward()
                NegLog, loglik, log_prior_tilde_l, log_prior_uL_vec, log_prior_sigma2_err = logpos.nlogpos_obj_S(Pars, Y, x, **hyper_pars, verbose=True)
                NegLog.backward()        
                # Pars.grad.data[torch.isnan(Pars.grad)] = 0
                tilde_l.grad.data[torch.isnan(tilde_l.grad)] = 0
                uL_vec.grad.data[torch.isnan(uL_vec.grad)] = 0
                tilde_sigma2_err.grad.data[torch.isnan(tilde_sigma2_err.grad)] = 0
                optimizer.step()
                if i % 100 == 99:
                    gap = curr_obj - NegLog
                    # upate objective
                    curr_obj = NegLog
                if verbose:
                    if i % 100 == 99:
                        print("{}/{}th iteration with target value {}.".format(i + 1, N_opt, NegLog))
                        print("loglik: {}, log_prior_tilde_l: {}, log_prior_uL_vec: {}, log_prior_sigma2_err: {}".format(loglik, log_prior_tilde_l, log_prior_uL_vec, log_prior_sigma2_err))

        # Save MAP estimate results
        if subfolder_name is None:
            with open(save_dir + folder_name + "MAP.dat", "wb") as res:
                pickle.dump(Pars.data.numpy(), res)
        else:
            with open(save_dir + folder_name + subfolder_name + "MAP.dat", "wb") as res:
                pickle.dump(Pars.data.numpy(), res)
        return NegLog

    if do_HMC:
        # HMC inference
        # Load MAP result
        if subfolder_name is None:
            with open(save_dir + folder_name + "MAP.dat", "rb") as res:
                estPars = pickle.load(res)
        else:
            with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
               estPars = pickle.load(res)
        hmc = HMC_Sampler.HMC_sampler.sampler(sample_size=N_hmc, potential_func=logpos.nlogpos_obj_S, init_position=init_position, step_size=step_size, adaptive_step_size=adaptive_step_size, num_steps_in_leap=num_steps_in_leap, M=M_hmc, Y=Y, x=x, duplicate_samples=True, TensorType=settings.torchType, **hyper_pars)
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


def load_syndata(test_size=0, directory="../data/sim/sim_MNTS.pickle"):
    # Load synthetic data
    with open(directory, "rb") as res:
        x, true_l, true_L_vecs, true_sigma2_err, Y = pickle.load(res)
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


def load_kaiser_data(data_dir="../data/KAISER/", ID_dir=None, group="sepsis", test_size=0.33, random_seed = 22):
    global subfolder_name
    if ID_dir is None:
        # ID_sepsis = ["41283451", "41304767", "41309821", "43215388"]
        ID_sepsis = ["41259111"]
        ID_nonsepsis = ["43120202", "43120381", "43121070", "43122359"]
    else:
        with open(ID_dir, "rb") as res:
            ID_sepsis, ID_nonsepsis = pickle.load(res)
    if group=="sepsis":
        subfolder_name = "ID_{}/".format(ID_sepsis[rank])
        with open(data_dir + "sepsis/ID_{}.pickle".format(ID_sepsis[rank]), "rb") as res:
            origx, origY, attributes = pickle.load(res)
    if group=="nonsepsis":
        subfolder_name = "ID_{}/".format(ID_nonsepsis[rank])
        with open(data_dir + "nonsepsis/ID_{}.pickle".format(ID_nonsepsis[rank]), "rb") as res:
            origx, origY, attributes = pickle.load(res)    
    if group=="illustration":
        ID_illustration = ["12978238", "12986958", "13296382", "41168468"]
        subfolder_name = "ID_{}/".format(ID_illutration[rank])
        with open(data_dir + "illustration/ID_{}.pickle".format(ID_illustration[rank]), "rb") as res:
            origx, origY, attributes = pickle.load(res)            
    Y, trend, scale = preprocess_realdata.orig2adj(origx, origY)
    x_scale = np.max(origx)
    x = origx / x_scale
    # convert numpy to torch
    if test_size == 0:
        x_train = torch.from_numpy(x).type(settings.torchType)
        Y_train = torch.from_numpy(Y).type(settings.torchType)
        x_test = None
        Y_test = None
    else:
        x_train, x_test, Y_train, Y_test = utils.data_split(x, Y, test_size=test_size, random_state=random_seed)
        x_train = torch.from_numpy(x_train).type(settings.torchType)
        Y_train = torch.from_numpy(Y_train).type(settings.torchType)
        x_test = torch.from_numpy(x_test).type(settings.torchType)
        Y_test = torch.from_numpy(Y_test).type(settings.torchType)
    return x_train, x_test, Y_train, Y_test, trend, scale, x_scale, attributes


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--group", type=str, default="sepsis")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--node", type=int, default=0)
    args = parser.parse_args()
    rank = rank + args.node*24
    save_dir = "../res/"
    group = args.group
    folder_name = "kaiser_stationary/"+group+"/"
    
    # ID_dir = "../data/KAISER/IDs_sampled.pickle"
    ID_dir = "../data/KAISER/IDs_sampled_seed{}_updated.pickle".format(args.seed)
    # ID_dir = None
    
    do_empirical_estimation = True
    do_train = True
    do_MAP = True
    do_HMC = False
    do_use_mass_matrix = False
    do_map_analysis = True
    do_hmc_analysis = False
    do_bayes_pred = False
    do_freq_pred = False
    do_pred_grids = False
    do_model_evaluation = False
    do_pred_test = False
    do_vis_bayes = False
    do_vis_freq = False

    x_train, x_test, Y_train, Y_test, trend, scale, x_scale, attributes = load_kaiser_data(group=group, ID_dir=ID_dir, test_size=0, random_seed=222)

    # subfolder_name = "{}/".format(rank)
    print(subfolder_name) 
    if not os.path.exists(save_dir + folder_name + subfolder_name):
        os.mkdir(save_dir + folder_name + subfolder_name)

    if do_empirical_estimation:
        est_sigmas, est_ls, smooth_ls, est_stds, est_R, est_B, est_L_vecs, est_tilde_sigma2_err = empirical_estimation.local_estimation(x_train.numpy(), Y_train.numpy())
        empirical_estimation.visualization(x_train.numpy(), Y_train.numpy(), est_ls, smooth_ls, est_stds, est_R, est_L_vecs, save_dir=save_dir,
         folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)
        empirical_estimation.save_res(est_ls, smooth_ls, est_L_vecs, est_tilde_sigma2_err, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name)

    N, M = Y_train.size() 

    if do_train:
        if do_MAP:
            NegLog = train(x_train, Y_train, N_opt=2000, N_hmc=10000, learning_rate=1e-2, do_MAP=do_MAP, do_HMC=do_HMC, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, verbose=True)

        if do_HMC:
            # Load MAP result
            if subfolder_name is None:
                with open(save_dir + folder_name + "MAP.dat", "rb") as res:
                    estPars = pickle.load(res)
            else:
                with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
                   estPars = pickle.load(res)
            if do_use_mass_matrix:
            # import mass matrix
                if subfolder_name is None:
                    with open(save_dir + folder_name + "HMC_sample_cov.pickle", "rb") as res:
                        sample_cov = pickle.load(res)
                else:
                    with open(save_dir + folder_name + subfolder_name + "HMC_sample_cov.pickle", "rb") as res:
                        sample_cov = pickle.load(res)
                # M_hmc = np.linalg.inv(sample_cov)
                M_hmc = np.linalg.inv(sample_cov/np.sqrt(np.outer(np.diag(sample_cov), np.diag(sample_cov))))
                # import pdb
                # pdb.set_trace()
                print("M_hmc: {}".format(M_hmc))
            else:
                print("Mass matrix is not available.")
                M_hmc = None
            init_position = estPars
            step_size=1e-2
            adaptive_step_size=False
            num_steps_in_leap=20
            sample = train(x_train, Y_train, N_opt=10000, N_hmc=1000, M_hmc=M_hmc, init_position=init_position, step_size=step_size, adaptive_step_size=adaptive_step_size, num_steps_in_leap=num_steps_in_leap, do_MAP=do_MAP, do_HMC=do_HMC, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, verbose=True)

    if do_map_analysis:
        with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)

        est_tilde_l = estPars[0]
        est_tilde_sigma = estPars[1]
        est_uL_vec = estPars[2:-1]
        est_L_vec = utils.uLvec2Lvec(est_uL_vec, M)
        est_L_f = utils.vec2lowtriangle(est_L_vec, M)
        est_B = np.matmul(est_L_f, est_L_f.T)
        est_real_B = est_B*np.exp(est_tilde_sigma)**2
        est_std = np.sqrt(np.diag(est_real_B))
        est_R = posterior_analysis.cov2cor(est_B)
        est_tilde_sigma2_err = estPars[-1]
        print(est_tilde_l)
        print(est_std)
        print(est_R)
        print(est_tilde_sigma2_err)
        est_tilde_l = torch.from_numpy(np.array([est_tilde_l])).type(settings.torchType)
        est_tilde_sigma = torch.from_numpy(np.array([est_tilde_sigma])).type(settings.torchType)
        est_uL_vec = torch.from_numpy(est_uL_vec).type(settings.torchType)
        est_tilde_sigma2_err = torch.from_numpy(np.array([est_tilde_sigma2_err])).type(settings.torchType)  

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
        plt.savefig("log_l_trace.png")
        from statsmodels.graphics.tsaplots import plot_acf
        fig = plt.figure()
        plot_acf(sample[:, 0])
        plt.savefig("log_l_trace_acf.png")
        tilde_ls, tilde_sigmas, uL_vecs, tilde_sigma2_errs = sample[:, 0], sample[:, 1], sample[:, 2:-1], sample[:, -1] 
        tilde_ls = torch.from_numpy(tilde_ls).type(settings.torchType)
        tilde_sigmas = torch.from_numpy(tilde_sigmas).type(settings.torchType)
        uL_vecs = torch.from_numpy(uL_vecs).type(settings.torchType)
        tilde_sigma2_errs = torch.from_numpy(tilde_sigma2_errs).type(settings.torchType)
        # compute the sample covariance matrix of parameters
        sample_cov = np.cov(sample.T)
        if subfolder_name is None:
            with open(save_dir + folder_name + "HMC_sample_cov.pickle", "wb") as res:
                pickle.dump(sample_cov, res)
        else:
            with open(save_dir + folder_name + subfolder_name + "HMC_sample_cov.pickle", "wb") as res:
                pickle.dump(sample_cov, res) 

    if do_freq_pred:
        if do_pred_grids:
            # Predictive inference
            grids = np.linspace(0., 1., 201)
            grids = torch.from_numpy(grids).type(settings.torchType)
            pred_grids_percentiles = prediction.pointwise_predmap_S(est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err, Y_train, x_train, grids)
            pred_grids_percentiles = torch.transpose(pred_grids_percentiles, 0, 1)
            visualization.Plot_posterior(x_train.numpy(), Y_train.numpy(), grids.numpy(), pred_grids_percentiles.data.numpy(),
                                         save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)

        if do_model_evaluation:
            # compute distribution for replicationss
            repy_mean, repy_std = prediction.test_predmap_S(est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err, Y_train, x_train, x_train)
            G = np.sum(np.linalg.norm(Y_train.numpy() - repy_mean.numpy(), axis = 1)**2)
            P = np.sum(repy_std.numpy()**2)
            D = G+P
            print("G", G)
            print("P", P)
            print("D", D)
            with open(save_dir + folder_name + subfolder_name + "model_evaluation_map.pickle", "wb") as res:
                pickle.dump([repy_mean, repy_std, G, P, D], res)

        if do_pred_test:
            # Predict testing data
            pred_test_mean, pred_test_std = prediction.test_predmap_S(est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err, Y_train, x_train, x_test)
            PMSE = np.sum(np.linalg.norm(Y_test.numpy() - pred_test_mean.numpy(), axis = 1)**2)
            print("PMSE", PMSE)
            with open(save_dir + folder_name + subfolder_name + "pred_test_map.pickle", "wb") as res:
                pickle.dump([pred_test_mean, pred_test_std, PMSE], res)

    if do_bayes_pred:
        N_sample = 10000
        if subfolder_name is None:
            with open(save_dir + folder_name + "HMC_sample.pickle", "rb") as res:
                sample = pickle.load(res)
        else:
            with open(save_dir + folder_name + subfolder_name + "HMC_sample.pickle", "rb") as res:
                sample = pickle.load(res)
        tilde_ls, tilde_sigmas, uL_vecs, tilde_sigma2_errs = posterior_analysis.vec2pars_S(torch.from_numpy(sample).type(settings.torchType), M) # tensor
        if do_pred_grids:
            grids = np.linspace(0., 1., 201)
            grids = torch.from_numpy(grids).type(settings.torchType)
            # Predictive inference            
            sampled_ys = prediction.pointwise_predsample_S(tilde_ls, tilde_sigmas, uL_vecs, tilde_sigma2_errs, Y_train, x_train, grids, N_sample=N_sample)
            gridy_quantiles, gridy_mean, gridy_std = np.percentile(sampled_ys, q = [2.5, 97.5], axis=0), np.mean(sampled_ys, axis=0), np.std(sampled_ys, axis=0)
            with open(save_dir + folder_name + subfolder_name + "pred_grid_hmc.pickle", "wb") as res:
                pickle.dump([gridy_quantiles, gridy_mean, gridy_std], res)
        if do_model_evaluation:
            # Predict testing data
            sampled_repys = prediction.test_predsample_S(tilde_ls, tilde_sigmas, uL_vecs, tilde_sigma2_errs, Y_train, x_train, x_train, N_sample=N_sample)
            repy_quantiles, repy_mean, repy_std = np.percentile(sampled_repys, q = [2.5, 97.5], axis=0), np.mean(sampled_repys, axis=0), np.std(sampled_repys, axis=0)
            G = np.sum(np.linalg.norm(Y_train.numpy() - repy_mean, axis = 1)**2)
            P = np.sum(repy_std**2)
            D = G+P
            print("G", G)
            print("P", P)
            print("D", D)
            with open(save_dir + folder_name + subfolder_name + "model_evaluation_hmc.pickle", "wb") as res:
                pickle.dump([repy_quantiles, repy_mean, repy_std, G, P, D], res)
        if do_pred_test:
            # Prediction on testing data
            sampled_predys = prediction.test_predsample_S(tilde_ls, tilde_sigmas, uL_vecs, tilde_sigma2_errs, Y_train, x_train, x_test, N_sample=N_sample)
            predy_quantiles, predy_mean, predy_std = np.percentile(sampled_predys, q = [2.5, 97.5], axis=0), np.mean(sampled_predys, axis=0), np.std(sampled_predys, axis=0)
            # import pdb
            # pdb.set_trace()
            PMSE = np.sum(np.linalg.norm(Y_test.numpy() - predy_mean, axis = 1)**2)
            print("PMSE", PMSE)
            with open(save_dir + folder_name + subfolder_name + "pred_test_hmc.pickle", "wb") as res:
                pickle.dump([predy_quantiles, predy_mean, predy_std, PMSE], res)

    grids = np.linspace(0., 1., 201)
    grids = torch.from_numpy(grids).type(settings.torchType)

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
        Y_train_orig = preprocess_realdata.adj2orig(Y_train.data.numpy(), trend, scale)
        predy_mean_orig = preprocess_realdata.adj2orig(predy_mean, trend, scale)
        predy_std_orig = predy_std*scale
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        visualization.Plot_posterior_trainandtest(x_train.numpy()*x_scale, Y_train_orig, grids.numpy()*x_scale, gridy_Y_orig, 
            x_test=x_test.numpy()*x_scale, Y_test=Y_test_orig, Y_pred=predy_mean_orig,
            save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes, type="HMC")
        # compute RMSE for tasks separately
        pred_test_rmse = utils.RMSE(predy_mean_orig, Y_test_orig, axis=0)
        print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(predy_mean_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(predy_mean_orig, predy_std_orig, Y_test_orig)
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))

    if do_vis_freq:
        # convert back to original data
        pred_grids_quantile_orig = preprocess_realdata.adj2orig(pred_grids_percentiles.data.numpy(), trend, scale)
        Y_train_orig = preprocess_realdata.adj2orig(Y_train.data.numpy(), trend, scale)
        pred_test_orig = preprocess_realdata.adj2orig(pred_test_mean.data.numpy(), trend, scale)
        pred_std_orig = pred_test_std.data.numpy() * scale
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        # visualization.Plot_posterior(x.data.numpy()*x_scale, Y_orig, grids.data.numpy()*x_scale, pred_grids_quantile_orig)
        visualization.Plot_posterior_trainandtest(x_train.numpy()*x_scale, Y_train_orig, grids.numpy()*x_scale, pred_grids_quantile_orig,
                                                  x_test=x_test.numpy()*x_scale, Y_test=Y_test_orig, Y_pred=pred_test_orig,
                                                  save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)
        # compute RMSE for tasks separately
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig, axis=0)
        print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))

        with open(save_dir + folder_name + subfolder_name + "pred_test_map_orig.pickle", "wb") as res:
            pickle.dump([pred_test_rmse, pred_test_lpd], res)

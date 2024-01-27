"""
pretrain the nonstationary multi-task model using stationary multi-task model
"""
import torch
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import pickle
from torch import optim, autograd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

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


def train(x, Y, N_opt=None, err_opt=None, learning_rate=1e-1, do_MAP=True, use_empirical_res=True, save_dir=None, folder_name=None, subfolder_name=None, verbose=False):
    """
    Using stationary multivariate Gaussian process to train data (x, Y)
    :param x: 1D tensor with length N
    :param Y: 2D tensor with dim N, M
    :param N_opt: number of iterations in the optimization
    :return: est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err
    """
    # Initialization
    hyper_pars = {"mu_tilde_l": 0, "sigma_tilde_l": 10., "a": 1e-6, "b": 1e-6, "c": 1.}
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

    # tilde_l = torch.FloatTensor([-2]).type(settings.torchType)
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
        # import pdb
        # pdb.set_trace()

    # Model validation
    if subfolder_name is None:
        with open(save_dir + folder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
    else:
        with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
    # dev = deviance_obj(torch.from_numpy(estPars).type(settings.torchType), Y, x)
    # AIC = model_validation.get_AIC(torch.from_numpy(estPars).type(settings.torchType), deviance_obj, Y=Y, x=x)
    # BIC = model_validation.get_BIC(torch.from_numpy(estPars).type(settings.torchType), deviance_obj, Y=Y, x=x)
    # print("Deviance:{}, AIC: {}, BIC: {}.".format(dev, AIC, BIC))

    estPars = torch.from_numpy(estPars).type(settings.torchType)
    est_tilde_l = estPars[0]
    est_tilde_sigma = estPars[1]
    est_L_vec = estPars[2:-1]
    est_tilde_sigma2_err = estPars[-1]
    if verbose:
        print(est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err)
        print("Negative log of posterior: {}".format(logpos.nlogpos_obj_S(estPars, Y, x, verbose=True, **hyper_pars)))

    return est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err


def load_syndata(test_size=0, directory="../data/sim/sim_MNTS_S.pickle"):
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


if __name__ == "__main__":
    rank = 22
    save_dir = "../res/"
    folder_name = "sim_stationary/"
    subfolder_name = "{}/".format(rank)
    if not os.path.exists(save_dir + folder_name + subfolder_name):
        os.mkdir(save_dir + folder_name + subfolder_name)

    do_empirical_estimation = True
    do_MAP = False
    do_map_analysis = True
    do_pred_grids = True
    do_model_evaluation = True
    do_pred_test = True
    do_vis = False

    # x, Y, x_test, Y_test, trend, scale, x_scale = load_realdata()
    # attributes = ["BPDIA", "BPSYS", "PP"]
    # x, Y, x_test, Y_test, trend, scale, x_scale = load_syndata(directory="../data/sim/sim_MNTS_S_{}.pickle".format(rank), test_size = 0)
    x, Y, x_test, Y_test, trend, scale, x_scale = load_syndata(directory="../data/sim/sim_MNTS_{}.pickle".format(rank), test_size = 0.33)
    attributes = ["Dim 1", "Dim 2"]
    M = 2

    if do_empirical_estimation:
        est_sigmas, est_ls, smooth_ls, est_stds, est_R, est_B, est_L_vecs, est_tilde_sigma2_err = empirical_estimation.local_estimation(x.numpy(), Y.numpy())
        empirical_estimation.visualization(x.numpy(), Y.numpy(), est_ls, smooth_ls, est_stds, est_R, est_L_vecs, save_dir=save_dir,
                      folder_name=folder_name, subfolder_name=subfolder_name)
        empirical_estimation.save_res(est_ls, smooth_ls, est_L_vecs, est_tilde_sigma2_err, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name)

    est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err = train(x, Y, N_opt=5000, learning_rate=1e-2, do_MAP=do_MAP, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, verbose=True)

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

    if do_pred_grids:
        # Predictive inference
        grids = np.linspace(0., 1., 201)
        grids = torch.from_numpy(grids).type(settings.torchType)
        pred_grids_percentiles = prediction.pointwise_predmap_S(est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err,
                                            Y, x, grids)
        pred_grids_percentiles = torch.transpose(pred_grids_percentiles, 0, 1)
        visualization.Plot_posterior(x.numpy(), Y.numpy(), grids.numpy(), pred_grids_percentiles.data.numpy(),
                                     save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)

    if do_model_evaluation:
        # compute distribution for replicationss
        repy_mean, repy_std = prediction.test_predmap_S(est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err, Y, x, x)
        G = np.sum(np.linalg.norm(Y.numpy() - repy_mean.numpy(), axis = 1)**2)
        P = np.sum(repy_std.numpy()**2)
        D = G + P
        print("G", G)
        print("P", P)
        print("D", D)
        with open(save_dir + folder_name + subfolder_name + "model_evaluation_map.pickle", "wb") as res:
            pickle.dump([repy_mean, repy_std, G, P, D], res)

    if do_pred_test:
        # Predict testing data
        pred_test_mean, pred_test_std = prediction.test_predmap_S(est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err, Y, x, x_test)
        PMSE = np.mean(np.linalg.norm(Y_test.numpy() - pred_test_mean.numpy(), axis = 1)**2)        
        print("PMSE", PMSE)
        with open(save_dir + folder_name + subfolder_name + "pred_test_map.pickle", "wb") as res:
            pickle.dump([pred_test_mean, pred_test_std, PMSE], res)

    if do_vis:
        # convert back to original data
        pred_grids_quantile_orig = preprocess_realdata.adj2orig(pred_grids_percentiles.data.numpy(), trend, scale)
        Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
        pred_test_orig = preprocess_realdata.adj2orig(pred_testdata.data.numpy(), trend, scale)
        pred_std_orig = pred_teststd.data.numpy() * scale
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        # visualization.Plot_posterior(x.data.numpy()*x_scale, Y_orig, grids.data.numpy()*x_scale, pred_grids_quantile_orig)
        visualization.Plot_posterior_trainandtest(x.numpy()*x_scale, Y_orig, grids.numpy()*x_scale, pred_grids_quantile_orig,
                                                  x_test=x_test.numpy()*x_scale, Y_test=Y_test_orig, Y_pred=pred_test_orig,
                                                  save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)
        # compute RMSE for tasks separately
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig, axis=0)
        print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))

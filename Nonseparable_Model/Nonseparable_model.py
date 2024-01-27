"""
"Hadamard" multitask nonstationary Gaussion process with inhomogeneous tasks

This differs from the main.py in one key way: Here, we assume that we have observations for one task per input. For each
input, we specify the task of the input that we observe. (The kernel that we learn is expressed as a Hadamard product of
an input kernel and a task kernel).
k([x, i], [x', j]) = k_{inputs}(x, x') * k_{tasks}(i,j)
Moreover, we assume the correlation across the tasks depends on time.
"""

import numpy as np
import torch
from torch import optim, autograd
from torch.autograd import Variable
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


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
# import extract_LAPS

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


def Load_realdata(data="kaiser"):
    if data == "mimic":
        # Load real data
        with open("../data/mimic_1p.pickle", "rb") as res:
            x, origY = pickle.load(res)
        Y, trend, scale = preprocess_realdata.orig2adj(x, origY)
        x_scale = 48.
        # # add some errors to x or Y
        # x += np.random.randn(*x.shape) * settings.precision
        # Y += np.random.randn(*Y.shape) * settings.precision
    elif data == "kaiser":
        # Load Kaiser data
        with open("../data/kaiser.pickle", "rb") as res:
            origx, origY = pickle.load(res)
        Y, trend, scale = preprocess_realdata.orig2adj(origx, origY)
        x_scale = np.max(origx)
        x = origx / x_scale

    # Split data for training and testing
    x_train, x_test, Y_train, Y_test = utils.data_split(x, Y, random_state=22)

    # # Visualize normalized data
    # figure = plt.figure()
    # lines = plt.plot(x, Y)
    # plt.legend(lines, ["BPDIA", "BPSYS", "PP"])
    # plt.savefig("normalized_real_data.png")
    # plt.close(figure)

    # convert numpy to torch
    x_train = torch.from_numpy(x_train).type(settings.torchType)
    Y_train = torch.from_numpy(Y_train).type(settings.torchType)
    x_test = torch.from_numpy(x_test).type(settings.torchType)
    Y_test = torch.from_numpy(Y_test).type(settings.torchType)
    return x_train, Y_train, x_test, Y_test, trend, scale, x_scale


def train(x, Y, N_opt=1000, N_hmc=1000, do_initialization=True, use_separable_res=False, use_empirical_res=True, do_MAP=True, do_HMC=True, hyper_pars=None, save_dir=None, folder_name=None, folder_name_separable=None, subfolder_name=None, verbose=False):
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
            if subfolder_name is None:
                with open(save_dir + folder_name_separable + "MAP.dat", "rb") as res:
                    initPars = pickle.load(res)
            else:
                with open(save_dir + folder_name_separable + subfolder_name + "MAP.dat", "rb") as res:
                    initPars = pickle.load(res)
            tilde_l = torch.from_numpy(initPars[0: N]).type(settings.torchType)
            tilde_sigma = torch.tensor(initPars[N: 2*N]).type(settings.torchType)
            L_vec = torch.from_numpy(initPars[2*N: -1]).type(settings.torchType)
            L_f = utils.vec2lowtriangle(L_vec, M)
            # print("initialized B_f: {}".format(torch.matmul(L_f, L_f.t())))
            L_vecs = torch.cat([L_vec*s for s in torch.exp(tilde_sigma)])
            tilde_sigma2_err = torch.from_numpy(initPars[-1].reshape(-1)).type(settings.torchType)
        # use empirical initialization
        if use_empirical_res:
            if subfolder_name is None:
                with open(save_dir + folder_name + "empirical_est.pickle", "rb") as res:
                    est_tilde_l, smooth_tilde_l, est_L_vecs, est_tilde_sigma2_err = pickle.load(res)
            else:
                with open(save_dir + folder_name + subfolder_name + "empirical_est.pickle", "rb") as res:
                    est_tilde_l, smooth_tilde_l, est_L_vecs, est_tilde_sigma2_err = pickle.load(res)
            tilde_l = torch.from_numpy(est_tilde_l).type(settings.torchType)
            L_vecs = torch.from_numpy(est_L_vecs).type(settings.torchType)
            tilde_sigma2_err = torch.from_numpy(np.array([est_tilde_sigma2_err])).type(settings.torchType)
    else:
        # use random initialization
        tilde_l = -4*torch.ones(N).type(settings.torchType)
        L_vecs = torch.randn(int(M*(M+1)/2)*N).type(settings.torchType)
        tilde_sigma2_err = torch.log(torch.rand(1))[0].type(settings.torchType)
    # Pars = Variable(torch.cat([tilde_l, L_vecs, tilde_sigma2_err.view(1)]), requires_grad=True)
    tilde_l = Variable(tilde_l, requires_grad=True)
    uL_vecs = Variable(utils.Lvecs2uLvecs(L_vecs, N, M), requires_grad=True)
    tilde_sigma2_err = Variable(tilde_sigma2_err, requires_grad=True)
    Pars = torch.cat([tilde_l, uL_vecs, tilde_sigma2_err.view(1)])
    # import pdb
    # pdb.set_trace()
    # print(Pars)
    # print("Negative log of posterior: {}".format(logpos.nlogpos_obj_SVC(Pars, Y, x, verbose=True, **hyper_pars)))

    # print("do_map: ", do_MAP)
    if do_MAP:
        # MAP inference
        # print("start MAP")
        optimizer = optim.Adam([{'params': tilde_l, 'lr': 2e-1}, {"params": [uL_vecs, tilde_sigma2_err], 'lr': 2e-1}])
        target_value_hist = np.zeros(N_opt)
        for i in range(N_opt):
            optimizer.zero_grad()
            with autograd.detect_anomaly():
                Pars = torch.cat([tilde_l, uL_vecs, tilde_sigma2_err.view(1)])
                # import pdb
                # pdb.set_trace()
                NegLog, loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err = \
                    logpos.nlogpos_obj_SVC(Pars, Y, x, **hyper_pars, verbose=True)
                NegLog.backward()
            # gradient correction
            # print("Pars: {}, grad: ".format(Pars, Pars.grad))
            # Pars.grad.data[torch.isnan(Pars.grad)] = 0
            optimizer.step()
            # import pdb
            # pdb.set_trace()
            if verbose:
                print(
                    "loglik: {},s log_prior_tilde_l: {}, log_prior_uL_vecs: {}, log_prior_sigma2_err: {}".format(
                        loglik, log_prior_tilde_l, log_prior_uL_vecs, log_prior_sigma2_err))
                print(NegLog)
            target_value_hist[i] = -NegLog

            # save results every 100 iterations
            if i % 100 == 99:
                # Save MAP estimate results
                Pars = torch.cat([tilde_l, uL_vecs, tilde_sigma2_err.view(1)])
                if subfolder_name is None:
                    with open(save_dir + folder_name + "MAP.dat", "wb") as res:
                        pickle.dump(Pars.data.numpy(), res)
                else:
                    with open(save_dir + folder_name + subfolder_name + "MAP.dat", "wb") as res:
                        pickle.dump(Pars.data.numpy(), res)

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
            with open(save_dir + folder_name + "MAP.dat", "wb") as res:
                pickle.dump(Pars.data.numpy(), res)
        else:
            with open(save_dir + folder_name + subfolder_name + "MAP.dat", "wb") as res:
                pickle.dump(Pars.data.numpy(), res)
       
    if do_HMC:
        # HMC inference
        # Load MAP result
        if subfolder_name is None:
            with open(save_dir + folder_name + "MAP.dat", "rb") as res:
                estPars = pickle.load(res)
        else:
            with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
                estPars = pickle.load(res)
        # est_tilde_l, est_L_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        # est_L = utils.vec2lowtriangle(torch.from_numpy(est_L_vec).type(settings.torchType), M).data.numpy()
        # est_S = np.matmul(est_L, est_L.T)
        # print(est_S)
        # print("Estimated correlation matrix: {}".format(posterior_analysis.cov2cor(est_S)))
        # print("Estimated Negative log of posterior: {}".format(logpos.nlogpos_obj(torch.from_numpy(estPars), Y, x, **hyper_pars)))

        hmc = HMC_Sampler.HMC_sampler.sampler(sample_size=N_hmc, potential_func=logpos.nlogpos_obj_SVC, init_position=estPars,
                                              step_size=1e-4, num_steps_in_leap=20, x=x, Y=Y, duplicate_samples=True, TensorType=settings.torchType,
                                              **hyper_pars)
        sample, _ = hmc.main_hmc_loop()
        if subfolder_name is not None:
            with open(save_dir + folder_name + subfolder_name + "HMC_sample.pickle", "wb") as res:
                pickle.dump(sample, res)
        else:
            with open(save_dir + folder_name + "HMC_sample.pickle", "wb") as res:
                pickle.dump(sample, res)


if __name__ == "__main__":
    save_dir = "../res/"
    # folder_name_stationary = "sim_stationaryIV/train_test/"
    # folder_name_nonstationary = "sim0.1IV/train_test/"
    # folder_name = "sim0.1IV/train_test_inhomogeneous/"

    folder_name_stationary = "sim_stationary/"
    folder_name_separable = "sim_separable/"
    folder_name = "sim_nonseparable/"
    # folder_name_stationary = "sim_large_stationary/"
    # folder_name_separable = "sim_large_separable/"
    # folder_name = "sim_large_nonseparable/"

    do_empirical_estimation = True
    do_initialization = True
    do_MAP =True
    do_HMC = False
    do_map_analysis = True
    do_post_analysis = False
    do_bayes_pred = False
    do_freq_pred = True
    do_pred_grids = True
    do_model_evaluation = True
    do_pred_test = False
    do_bayes_visualization = False
    do_freq_visualization = False

    # x, Y, x_test, Y_test, trend, scale, x_scale = load_realdata()
    # attributes = ["BPDIA", "BPSYS", "PP"]
    x, Y, x_test, Y_test, trend, scale, x_scale = load_syndata(test_size = 0.)
    attributes = ["Dim 1", "Dim 2"]

    # Initialization
    N, M = Y.shape
    hyper_pars = {"mu_tilde_l": 0., "alpha_tilde_l": 10., "beta_tilde_l": 1, "mu_L": 0.,
                  "alpha_L": 10., "beta_L": 1, "a": 1., "b": 1.}

    if do_empirical_estimation:
        est_sigmas, est_ls, smooth_ls, est_stds, est_R, est_B, est_L_vecs, est_tilde_sigma2_err = empirical_estimation.local_estimation(x.numpy(), Y.numpy())
        empirical_estimation.visualization(x.numpy(), Y.numpy(), est_ls, smooth_ls, est_stds, est_R, est_L_vecs, save_dir=save_dir,
                      folder_name=folder_name)
        empirical_estimation.save_res(est_ls, smooth_ls, est_L_vecs, est_tilde_sigma2_err, save_dir=save_dir, folder_name=folder_name)
    # import pdb
    # pdb.set_trace()

    # train non-stationary model
    train(x, Y, N_opt=1000, N_hmc=100, do_initialization=do_initialization, do_MAP=do_MAP, do_HMC=do_HMC,
          hyper_pars=hyper_pars, save_dir=save_dir, folder_name=folder_name, folder_name_separable=folder_name_separable, verbose=True)
    print("training completed.")

    if do_map_analysis:
        # Load MAP result
        with open(save_dir + folder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_uL_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        est_L_vecs = utils.uLvecs2Lvecs(est_uL_vecs, N, M)
        est_L_vec_list = [est_L_vecs[n * int(M * (M + 1) / 2):(n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
        est_L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in est_L_vec_list]
        est_B_list = [np.matmul(L_f, L_f.T) for L_f in est_L_f_list]
        est_R_list = [posterior_analysis.cov2cor(B) for B in est_B_list]
        est_stds = np.stack([np.sqrt(np.diag(B)) for B in est_B_list])
        # plot estimated log length-scale process
        fig = plt.figure()
        plt.plot(x.numpy(), est_tilde_l)
        plt.savefig(save_dir + folder_name + "est_log_l.png")
        plt.close(fig)
        # plot estimated std process
        fig = plt.figure()
        for m in range(M):
            plt.plot(x.numpy(), est_stds[:, m], label="Dim {}".format(m+1))
        plt.legend()
        plt.savefig(save_dir + folder_name + "est_std.png")
        plt.close((fig))
        # plot correlation process
        for i in range(M):
            for j in range(i+1, M):
                fig = plt.figure()
                R_ij = np.stack([R_f[i, j] for R_f in est_R_list])
                plt.plot(x.numpy(), R_ij)
                plt.savefig(save_dir + folder_name + "est_log_R_{}{}.png".format(i, j))
                plt.close(fig)
        # print error measurement
        print("sigma2_error: ", est_tilde_sigma2_err)


    if do_post_analysis:
        # Posterior analysis
        with open(save_dir + folder_name + "HMC_sample.pickle", "rb") as res:
            sample = pickle.load(res)
        tilde_l_hist, uL_vecs_hist, tilde_sigma2_err_hist = posterior_analysis.vec2pars_SVC(sample, N, M)
        # ....
        posterior_analysis.visualization_pos(x.numpy(), tilde_l_hist, L_vecs_hist=L_vecs_hist, N=N, M=M, save_dir=save_dir, folder_name=folder_name, attributes=attributes)

    grids = np.linspace(0., 1., 201)
    grids = torch.from_numpy(grids).type(settings.torchType)

    if do_bayes_pred:
        #######Bayesian Inference##########
        # # Prediction on grids
        # tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist = prediction.vec2pars(
        #     torch.from_numpy(sample).type(settings.torchType), N, M)
        # sampled_y_hist = prediction.pointwise_predsample_hadamard(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, x, indx, y, grids, **hyper_pars)
        # print(sampled_y_hist.size())
        # with open(save_dir + folder_name + "pred_res.pickle", "wb") as res:
        #     pickle.dump(sampled_y_hist, res)
        #
        # # Prediction on testing data
        # tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist = prediction.vec2pars(
        #     torch.from_numpy(sample).type(settings.torchType), N, M)
        # sampled_test_hist = prediction.test_predsample_hadamard(tilde_l_hist, tilde_sigma_hist, L_vec_hist,
        #                                                tilde_sigma2_err_hist, x, indx, y, x_test, indx_test, **hyper_pars)
        # print(sampled_test_hist.size())
        # with open(save_dir + folder_name + "pred_test_res.pickle", "wb") as res:
        #     pickle.dump(sampled_test_hist, res)
        pass

    if do_freq_pred:
        ###### Frequentist Inference##########
        # Load MAP result
        with open(save_dir + folder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_uL_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        est_uL_vecs = torch.from_numpy(est_uL_vecs).type(settings.torchType)
        est_L_vecs = utils.uLvecs2Lvecs(est_uL_vecs, N, M)
        # import pdb
        # pdb.set_trace()
        est_tilde_l = torch.from_numpy(est_tilde_l).type(settings.torchType)
        est_tilde_sigma2_err = torch.tensor(est_tilde_sigma2_err).type(settings.torchType)

        if do_pred_grids:
            # Prediction on grids
            # gridy_quantiles, _ = prediction.pointwise_predmap_inhomogeneous(est_tilde_l, est_uL_vecs, est_tilde_sigma2_err, Y, x, grids, **hyper_pars)
            # with open(save_dir + folder_name + "pred_grid_map.pickle", "wb") as res:
            #     pickle.dump(gridy_quantiles, res)
            # visualization.Plot_posterior(x.numpy(), Y.numpy(), grids.numpy(),
            #                              torch.transpose(gridy_quantiles, 1, 0).numpy(),
            #                              save_dir=save_dir, folder_name=folder_name, attributes=attributes, type="MAP")
            gridy_quantiles, gridy_mean, gridy_std = prediction.pointwise_predmap_inhomogeneous_sampling(100, est_tilde_l, est_uL_vecs, est_tilde_sigma2_err, Y, x, grids, **hyper_pars)
            with open(save_dir + folder_name + "pred_grid_map.pickle", "wb") as res:
                pickle.dump([gridy_quantiles, gridy_mean, gridy_std], res)

            gridy_Y = np.stack([gridy_quantiles[:,0,:], gridy_mean, gridy_quantiles[:,1,:]])
            visualization.Plot_posterior(x.numpy(), Y.numpy(), grids.numpy(),
                                         gridy_Y,
                                         save_dir=save_dir, folder_name=folder_name, attributes=attributes, type="MAP")
        
        if do_model_evaluation:
            repy_quantiles, repy_mean, repy_std = prediction.test_predmap_inhomogeneous_sampling(100, est_tilde_l, est_uL_vecs, est_tilde_sigma2_err, Y, x, x, **hyper_pars)
            G = np.sum(np.linalg.norm(Y.numpy() - repy_mean, axis = 1)**2)
            P = np.sum(repy_std**2)
            D = G+P
            print("G", G)
            print("P", P)
            print("D", D)
            with open(save_dir + folder_name + "model_evaluation_map.pickle", "wb") as res:
                pickle.dump([repy_quantiles, repy_mean, repy_std, G, P, D], res)

        if do_pred_test:
            # Prediction on testing data
            predy_quantiles, predy_mean, predy_std = prediction.test_predmap_inhomogeneous_sampling(100, est_tilde_l, est_uL_vecs, est_tilde_sigma2_err, Y, x, x_test, **hyper_pars)
            G = np.sum(np.linalg.norm(Y_test.numpy() - predy_mean, axis = 1)**2)
            P = np.sum(predy_std**2)
            D = G+P
            print("G", G)
            print("P", P)
            print("D", D)
            with open(save_dir + folder_name + "pred_test_map.pickle", "wb") as res:
                pickle.dump([predy_quantiles, predy_mean, predy_std, G, P, D], res)

    if do_bayes_visualization:
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

    if do_freq_visualization:
        ############ Visualization (Frequentist)
        with open(save_dir + folder_name + "pred_grid_map.pickle", "rb") as res:
            gridy_quantiles = pickle.load(res)
        with open(save_dir + folder_name + "pred_test_map.pickle", "rb") as res:
            predy_quantiles = pickle.load(res)
        gridy_quantiles = np.transpose(gridy_quantiles.data.numpy(), axes=[1,0,2])

        predy_quantiles = predy_quantiles.data.numpy()
        # visualization.Plot_posterior_hadamard(x.data.numpy(), indx.data.numpy(), y.data.numpy(), grids.data.numpy(), sampled_y_quantile)
        pred_test = predy_quantiles[:, 1, :]
        pred_test_orig = preprocess_realdata.adj2orig(pred_test, trend, scale)
        pred_std = (predy_quantiles[:, 1, :] - predy_quantiles[:, 0, :])/1.96
        pred_std_orig = pred_std * scale

        # convert back to original scale
        gridy_quantiles_orig = preprocess_realdata.adj2orig(gridy_quantiles, trend, scale)
        # print(sampled_y_quantile_orig.shape)
        Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        print(Y_orig.shape, gridy_quantiles_orig.shape, Y_test_orig.shape, pred_test_orig.shape)
        visualization.Plot_posterior_trainandtest(x.numpy() * x_scale, Y_orig, grids.numpy() * x_scale,
                                                  gridy_quantiles_orig, x_test=x_test.numpy() * x_scale,
                                                  Y_test=Y_test_orig, Y_pred=pred_test_orig, save_dir=save_dir, folder_name=folder_name)
        # compute RMSE for tasks separately
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig, axis=0)
        print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))
        import pdb
        pdb.set_trace()

import numpy as np
import torch
from torch import optim, autograd
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt

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


def load_syndata(test_size=0):
    # Load synthetic data
    with open("../data/sim/sim_MNTS.pickle", "rb") as res:
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


def Load_realdata():
    # # Load real data
    # with open("../data/mimic_1p.pickle", "rb") as res:
    #     x, origY = pickle.load(res)
    # N, M = origY.shape
    # Y, trend, scale = preprocess_realdata.orig2adj(x, origY)
    # x_scale = 48.
    # # # add some errors to x or Y
    # # x += np.random.randn(*x.shape) * settings.precision
    # # Y += np.random.randn(*Y.shape) * settings.precision
    # Load Kaiser data
    with open("../data/kaiser.pickle", "rb") as res:
        origx, origY = pickle.load(res)
    Y, trend, scale = preprocess_realdata.orig2adj(origx, origY)
    x_scale = np.max(origx)
    x = origx / x_scale

    # Split data for training and testing
    x_train, x_test, Y_train, Y_test = utils.data_split(x, Y, random_state=22)

    # convert numpy to torch
    x_train = torch.from_numpy(x_train).type(settings.torchType)
    Y_train = torch.from_numpy(Y_train).type(settings.torchType)
    x_test = torch.from_numpy(x_test).type(settings.torchType)
    Y_test = torch.from_numpy(Y_test).type(settings.torchType)

    return x_train, Y_train, x_test, Y_test, trend, scale, x_scale


def train(x, Y, N_opt=2000, N_hmc=1000, use_stationary_res = False, use_empirical_res=False, use_combined_res=False, do_MAP=True, do_HMC=True, save_dir=None, folder_name=None, folder_name_stationary=None, subfolder_name=None, hyper_pars=None, verbose=False):
    """
    Using non-stationary multivariate Gaussian process to train data (x, Y)
    :param x: 1D tensor with length N
    :param Y: 2D tensor with dim N, M
    :param N_opt: number of iterations in the optimization
    :param do_MAP: boolean for MAP
    :param do_HNC: boolean for HMC
    :return: est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err
    """
    N, M = Y.size()

    # # use random initialization
    # tilde_l = torch.randn(N).type(settings.torchType)
    # tilde_sigma = torch.randn(N).type(settings.torchType)
    # L_vec = torch.randn(int(M*(M+1)/2)).type(settings.torchType)
    # tilde_sigma2_err = torch.log(torch.rand(1))[0].type(settings.torchType)
    # use informative parameter estimates from stationary model
    if use_stationary_res:
        if subfolder_name is None:
            with open(save_dir + folder_name_stationary + "MAP.dat", "rb") as res:
                initPars = pickle.load(res)
        else:
            with open(save_dir + folder_name_stationary + subfolder_name + "MAP.dat", "rb") as res:
                initPars = pickle.load(res)
        tilde_l = torch.from_numpy(initPars[0] * np.ones(N) + 0.1 * np.random.randn(N)).type(settings.torchType)
        tilde_sigma = torch.from_numpy(initPars[1] * np.ones(N) + 0.1 * np.random.randn(N)).type(settings.torchType)
        uL_vec = torch.from_numpy(initPars[2:-1]).type(settings.torchType)
        tilde_sigma2_err = torch.from_numpy(initPars[-1].reshape(-1)).type(settings.torchType)
    if use_empirical_res:
        if subfolder_name is None:
            with open(save_dir + folder_name + "empirical_est.pickle", "rb") as res:
                est_tilde_l, smooth_tilde_l, est_L_vecs, est_tilde_sigma2_err = pickle.load(res)
        else:
            with open(save_dir + folder_name + subfolder_name + "empirical_est.pickle", "rb") as res:
                est_tilde_l, smooth_tilde_l, est_L_vecs, est_tilde_sigma2_err = pickle.load(res)
        tilde_l = torch.from_numpy(est_tilde_l).type(settings.torchType)
        L_vecs = torch.from_numpy(est_L_vecs).type(settings.torchType)
        uL_vecs = utils.Lvecs2uLvecs(L_vecs, N, M)
        uL_vec = torch.mean(uL_vecs.view([N, -1]), 0)
        tilde_sigma = torch.ones(N).type(settings.torchType)
        tilde_sigma2_err = torch.from_numpy(np.array([est_tilde_sigma2_err])).type(settings.torchType)
    if use_combined_res:
        if subfolder_name is None:
            with open(save_dir + folder_name_stationary + "MAP.dat", "rb") as res:
                initPars = pickle.load(res)
        else:
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
        uL_vecs = utils.Lvecs2uLvecs(L_vecs, N, M)
        uL_vec = torch.mean(uL_vecs.view([N, -1]), 0)
        tilde_sigma = torch.ones(N).type(settings.torchType)
        tilde_sigma2_err = torch.from_numpy(np.array([est_tilde_sigma2_err])).type(settings.torchType)


    tilde_l = Variable(tilde_l, requires_grad=True)
    tilde_sigma = Variable(tilde_sigma, requires_grad=True)
    uL_vec = Variable(uL_vec, requires_grad=True)
    tilde_sigma2_err = Variable(tilde_sigma2_err, requires_grad=True)
    Pars = torch.cat([tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err.view(1)])
    L_vec = utils.uLvec2Lvec(uL_vec, M)
    # logpos.show_covs(Pars, Y, x)

    if do_MAP:
        # MAP inference
        # print("start to do MAP inference")
        optimizer = optim.Adam([{"params": [tilde_sigma, uL_vec, tilde_sigma2_err], 'lr': 0.2}, {"params": tilde_l, 'lr': 0.2}])
        target_value_hist = np.zeros(N_opt)
        for i in range(N_opt):
            optimizer.zero_grad()
            with autograd.detect_anomaly():
                Pars = torch.cat([tilde_l, tilde_sigma, uL_vec, tilde_sigma2_err.view(1)])
                NegLog, loglik, log_prior_tilde_l, log_prior_tilde_sigma, log_prior_uL_vec, log_prior_sigma2_err = logpos.nlogpos_obj(Pars, Y, x, **hyper_pars, verbose=True)
                NegLog.backward()
            optimizer.step()
            if verbose:
                if i % 100 == 99:
                    # print("{}/{} iterations have completed with Pars {}".format(i+1, N_opt, Pars))
                    print("{}/{}th iteration with target value {}".format(i+1, N_opt, NegLog))
                    print("loglik: {},s log_prior_tilde_l: {}, log_prior_tilde_sigma: {}, log_prior_uL_vec: {}, log_prior_sigma2_err: {}".format(loglik, log_prior_tilde_l, log_prior_tilde_sigma, log_prior_uL_vec, log_prior_sigma2_err))
            target_value_hist[i] = -NegLog
        fig = plt.figure()
        plt.plot(target_value_hist)
        if subfolder_name is None:
            plt.savefig(save_dir + folder_name + "target_trace.png")
        else:
            plt.savefig(save_dir + folder_name + subfolder_name + "target_trace.png")
        plt.close(fig)
        # Save MAP estimate results
        if subfolder_name is None:
            with open(save_dir + folder_name + "MAP.dat", "wb") as res:
                pickle.dump(Pars.data.numpy(), res)
        else:
            with open(save_dir + folder_name + subfolder_name + "MAP.dat", "wb") as res:
                pickle.dump(Pars.data.numpy(), res)
        # logpos.show_covs(Pars, Y, x)
        # import pdb
        # pdb.set_trace()

    if do_HMC:
        # HMC inference
        # Load MAP result
        if subfolder_name is None:
            with open(save_dir + folder_name + "MAP.dat", "rb") as res:
                estPars = pickle.load(res)
        else:
            with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
                estPars = pickle.load(res)
        est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err = posterior_analysis.vec2pars_est(estPars, N, M)
        est_L = utils.vec2lowtriangle(torch.from_numpy(est_L_vec).type(settings.torchType), M).data.numpy()
        est_S = np.matmul(est_L, est_L.T)
        # print(est_S)
        print("Estimated correlation matrix: {}".format(posterior_analysis.cov2cor(est_S)))
        # print("Estimated Negative log of posterior: {}".format(logpos.nlogpos_obj(torch.from_numpy(estPars), Y, x, **hyper_pars)))

        # import pdb
        # pdb.set_trace()
        hmc = HMC_Sampler.HMC_sampler.sampler(sample_size=N_hmc, potential_func=logpos.nlogpos_obj, init_position=estPars, step_size=0.0002, num_steps_in_leap=20, Y=Y, x=x, duplicate_samples=True, TensorType=settings.torchType, **hyper_pars)
        sample, _ = hmc.main_hmc_loop()
        if subfolder_name is None:
            with open(save_dir + folder_name + "HMC_sample.pickle", "wb") as res:
                pickle.dump(sample, res)
        else:
            with open(save_dir + folder_name + subfolder_name + "HMC_sample.pickle", "wb") as res:
                pickle.dump(sample, res)
        print(np.mean(sample, axis=0))
        print(np.std(sample, axis=0))
        # import pdb
        # pdb.set_trace()


if __name__ == "__main__":
    save_dir = "../res/"
    # folder_name = "mimic_1p/"
    # folder_name_stationary = "sim_stationaryIV/train_test/"
    # folder_name = "sim0.1IV/train_test/"
    folder_name_stationary = "sim_stationary/"
    folder_name = "sim_separable/"
    do_empirical_estimation = True
    do_MAP = True
    do_HMC = False
    do_map_analysis = True
    do_post_analysis = False
    do_bayes_pred = False
    do_freq_pred = True
    do_pred_grids = True
    do_model_evaluation = True
    do_pred_test = True
    do_vis_bayes = False
    do_vis_freq = False

    # x, Y, x_test, Y_test, trend, scale, x_scale = load_realdata()
    # attributes = ["BPDIA", "BPSYS", "PP"]
    x, Y, x_test, Y_test, trend, scale, x_scale = load_syndata(test_size=0.33)
    attributes = ["Dim 1", "Dim 2"]

    # Initialization
    N, M = Y.shape
    hyper_pars = {"mu_tilde_l": 0., "alpha_tilde_l": 10., "beta_tilde_l": 1., "mu_tilde_sigma": 0.,
                  "alpha_tilde_sigma": 10., "beta_tilde_sigma": 1., "a": 1., "b": 1., "c": 10.}

    if do_empirical_estimation:
        est_sigmas, est_ls, smooth_ls, est_stds, est_R, est_B, est_L_vecs, est_tilde_sigma2_err = empirical_estimation.local_estimation(x.numpy(), Y.numpy())
        empirical_estimation.visualization(x.numpy(), Y.numpy(), est_ls, smooth_ls, est_stds, est_R, est_L_vecs, save_dir=save_dir,
                      folder_name=folder_name)
        empirical_estimation.save_res(est_ls, smooth_ls, est_L_vecs, est_tilde_sigma2_err, save_dir=save_dir, folder_name=folder_name)

    # # train non-stationary model
    train(x, Y, N_opt=1000, N_hmc=1000, do_MAP=do_MAP, do_HMC=do_HMC, use_combined_res=True, save_dir=save_dir, folder_name=folder_name, folder_name_stationary=folder_name_stationary, hyper_pars=hyper_pars, verbose=True)
    # train(x, Y, use_stationary_res = True, use_empirical_res=False, N_opt=1000, N_hmc=1000, do_MAP=do_MAP, do_HMC=do_HMC, save_dir=save_dir, folder_name=folder_name, folder_name_stationary=folder_name_stationary, hyper_pars=hyper_pars, verbose=True)


    if do_map_analysis:
        with open(save_dir + folder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err = posterior_analysis.vec2pars_est(estPars, N, M)
        est_L_vec = utils.uLvec2Lvec(est_uL_vec, M)
        est_L = utils.vec2lowtriangle(torch.from_numpy(est_L_vec).type(settings.torchType), M).data.numpy()
        est_B = np.matmul(est_L, est_L.T)
        # plot estimated log length-scale process
        fig = plt.figure()
        plt.plot(x.numpy(), est_tilde_l)
        plt.savefig(save_dir + folder_name + "est_log_l.png")
        plt.close(fig)
        # plot estimated std process
        est_std = np.sqrt(np.diag(est_B))
        est_sigma = np.exp(est_tilde_sigma)
        fig = plt.figure()
        for m in range(M):
            plt.plot(x.numpy(), est_sigma * est_std[m], label="Dim {}".format(m + 1))
        plt.legend()
        plt.savefig(save_dir + folder_name + "std.png")
        plt.close(fig)
        # print correlation
        est_r = posterior_analysis.cov2cor(est_B)[0, 1]
        print("correlation coefficient: ", est_r)
        print("sigma2_error: ", est_tilde_sigma2_err)
        # import pdb
        # pdb.set_trace()

    if do_post_analysis:
        # Posterior analysis
        with open(save_dir + folder_name + "HMC_sample.pickle", "rb") as res:
            sample = pickle.load(res)
        tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist = posterior_analysis.vec2pars(sample, N, M) # array
        posterior_analysis.visualization_pos(x.numpy(), tilde_l_hist, tilde_sigma_hist, save_dir=save_dir, folder_name=folder_name, attributes=attributes)

    grids = np.linspace(0., 1., 201)
    grids = torch.from_numpy(grids).type(settings.torchType)
    # import pdb
    # pdb.set_trace()

    if do_bayes_pred:
        if do_pred_grids:
            # Predictive inference
            tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist = prediction.vec2pars(torch.from_numpy(sample).type(settings.torchType), N, M) # tensor
            sampled_y_hist = prediction.pointwise_predsample(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, Y, x, grids, **hyper_pars)
            print(sampled_y_hist.size())
            with open(save_dir + folder_name + "pred_res.pickle", "wb") as res:
                pickle.dump(sampled_y_hist, res)

        if do_pred_test:
            # Predict testing data
            tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist = prediction.vec2pars(torch.from_numpy(sample).type(settings.torchType), N, M) # tensor
            sampled_test_hist = prediction.test_predsample(tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist, Y, x, x_test, **hyper_pars)
            print(sampled_test_hist.size())
            with open(save_dir + folder_name + "pred_test_res.pickle", "wb") as res:
                pickle.dump(sampled_test_hist, res)

    if do_freq_pred:
        # # Load Stationary result
        # with open(save_dir + folder_name_stationary + "MAP.dat", "rb") as res:
        #     initPars = pickle.load(res)
        # est_tilde_l = torch.from_numpy(initPars[0] * np.ones(N)).type(settings.torchType)
        # est_tilde_sigma = torch.from_numpy(initPars[1] * np.ones(N)).type(settings.torchType)
        # est_L_vec = torch.from_numpy(initPars[2:-1]).type(settings.torchType)
        # est_tilde_sigma2_err = torch.from_numpy(initPars[-1].reshape(-1)).type(settings.torchType)
        # Load MAP result
        with open(save_dir + folder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err = posterior_analysis.vec2pars_est(estPars, N, M)
        est_tilde_l = torch.from_numpy(est_tilde_l).type(settings.torchType)
        est_tilde_sigma = torch.from_numpy(est_tilde_sigma).type(settings.torchType)
        est_uL_vec = torch.from_numpy(est_uL_vec).type(settings.torchType)
        est_tilde_sigma2_err = torch.FloatTensor([est_tilde_sigma2_err]).type(settings.torchType)
        est_L_vec = utils.uLvec2Lvec(est_uL_vec, M)
        if do_pred_grids:
            # Predictive inference
            gridy_quantiles, gridy_mean, gridy_std = prediction.pointwise_predmap_sampling(100, est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err, Y, x, grids, **hyper_pars)
            gridy_Y = np.stack([gridy_quantiles[:,0,:], gridy_mean, gridy_quantiles[:,1,:]])
            visualization.Plot_posterior(x.numpy(), Y.numpy(), grids.numpy(),
                                         gridy_Y,
                                         save_dir=save_dir, folder_name=folder_name, attributes=attributes, type="MAP")

            # print(gridy_quantiles.size())
            with open(save_dir + folder_name + "pred_grid_map.pickle", "wb") as res:
                pickle.dump([gridy_quantiles, gridy_mean, gridy_std], res)

        if do_model_evaluation:
            repy_quantiles, repy_mean, repy_std = prediction.test_predmap_sampling(100, est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err, Y, x, x, **hyper_pars)
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
            predy_quantiles, predy_mean, predy_std = prediction.test_predmap_sampling(100, est_tilde_l, est_tilde_sigma, est_uL_vec, est_tilde_sigma2_err, Y, x, x_test, **hyper_pars)
            G = np.sum(np.linalg.norm(Y_test.numpy() - predy_mean, axis = 1)**2)
            P = np.sum(predy_std**2)
            D = G+P
            print("G", G)
            print("P", P)
            print("D", D)
            with open(save_dir + folder_name + "pred_test_map.pickle", "wb") as res:
                pickle.dump([predy_quantiles, predy_mean, predy_std, G, P, D], res)
                

    if do_vis_bayes:
        print("HMC result:")
        # Visualization
        with open(save_dir + folder_name + "pred_res.pickle", "rb") as res:
            sampled_y_hist = pickle.load(res)
        with open(save_dir + folder_name + "pred_test_res.pickle", "rb") as res:
            sampled_test_hist = pickle.load(res)
        # print(sampled_y_hist.size())
        # print(sampled_test_hist.size())
        sampled_y_hist = sampled_y_hist.data.numpy()
        sampled_y_quantile = visualization.samples2quantiles(sampled_y_hist)
        # print(sampled_y_quantile.shape)
        # visualization.Plot_posterior(x.data.numpy(), Y.data.numpy(), grids.data.numpy(), sampled_y_quantile)

        pred_test = torch.mean(sampled_test_hist, dim=1)
        pred_std = torch.std(sampled_test_hist, dim=1)

        # convert back to original data
        sampled_y_quantile_orig = preprocess_realdata.adj2orig(sampled_y_quantile, trend, scale)
        # print(sampled_y_quantile_orig.shape)
        Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
        pred_test_orig = preprocess_realdata.adj2orig(pred_test.data.numpy(), trend, scale)
        pred_std_orig = pred_std.data.numpy()*scale
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        visualization.Plot_posterior_trainandtest(x.numpy()*x_scale, Y_orig, grids.numpy()*x_scale, sampled_y_quantile_orig, x_test=x_test.numpy()*x_scale, Y_test=Y_test_orig, Y_pred=pred_test_orig)
        # compute RMSE for tasks separately
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig, axis=0)
        print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))

    if do_vis_freq:
        print("MAP results:")
        # Visualization
        with open(save_dir + folder_name + "pred_resmap.pickle", "rb") as res:
            gridy_quantile = pickle.load(res)
        with open(save_dir + folder_name + "pred_test_resmap.pickle", "rb") as res:
            testy_quantile = pickle.load(res)
        gridy_quantile = gridy_quantile.data.numpy()
        # print(sampled_y_quantile.shape)
        # visualization.Plot_posterior(x.data.numpy(), Y.data.numpy(), grids.data.numpy(), sampled_y_quantile)

        pred_test = testy_quantile[:, 1, :]
        pred_std = (testy_quantile[:, 1, :] - testy_quantile[:, 0, :])/1.96

        # convert back to original data
        gridy_quantile_orig = preprocess_realdata.adj2orig(gridy_quantile, trend, scale)
        # print(sampled_y_quantile_orig.shape)
        Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
        pred_test_orig = preprocess_realdata.adj2orig(pred_test.data.numpy(), trend, scale)
        pred_std_orig = pred_std.data.numpy()*scale
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        visualization.Plot_posterior_trainandtest(x.numpy()*x_scale, Y_orig, grids.numpy()*x_scale, np.transpose(gridy_quantile_orig, axes=(1,0,2)), x_test=x_test.numpy()*x_scale, Y_test=Y_test_orig, Y_pred=pred_test_orig)
        # compute RMSE for tasks separately
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig, axis=0)
        print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))

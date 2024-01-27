# import standard libraries
import os
import time
import matplotlib
matplotlib.use('Agg')
import argparse
# import pickle

# import private libraries
from Nonseparable_model import *


def nonseparable_model(x, Y, x_test, Y_test, trend, scale, x_scale, n_opt=1000, n_hmc=100, attributes=None):
    # Initialization
    N, M = Y.shape
    hyper_pars = {"mu_tilde_l": 0., "alpha_tilde_l": 5., "beta_tilde_l": 0.1, "mu_L": 0.,
                  "alpha_L": 5., "beta_L": 0.2, "a": 1., "b": 1.}

    # train non-stationary model
    train(x, Y, N_opt=n_opt, N_hmc=n_hmc, do_initialization=do_initialization, do_MAP=do_MAP, do_HMC=do_HMC,
          hyper_pars=hyper_pars, save_dir=save_dir, folder_name=folder_name, folder_name_separable=folder_name_separable, subfolder_name=subfolder_name, verbose=True)

    if do_map_analysis:
        # Load MAP result
        with open(save_dir + folder_name + subfolder_name + "MAP_old.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_L_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        posterior_analysis.visualization_pos_map(x.numpy() * x_scale, est_tilde_l, est_L_vecs, N=N, M=M, save_dir=save_dir,
                                               folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)

    if do_post_analysis:
        # Posterior analysis
        with open(save_dir + folder_name + subfolder_name + "HMC_sample.pickle", "rb") as res:
            sample = pickle.load(res)
        tilde_l_hist, L_vecs_hist, tilde_sigma2_err_hist = posterior_analysis.vec2pars_SVC(sample, N, M)
        posterior_analysis.visualization_pos(x.numpy() * x_scale, tilde_l_hist, L_vecs_hist=L_vecs_hist, N=N, M=M,
                                             save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)

    grids = np.linspace(0., 1., 201)
    grids = torch.from_numpy(grids).type(settings.torchType)

    if do_bayes_pred:
        #######Bayesian Inference##########
        tilde_l_hist, L_vecs_hist, tilde_sigma2_err_hist = posterior_analysis.vec2pars_SVC(sample, N, M)
        tilde_l_hist = torch.from_numpy(tilde_l_hist).type(settings.torchType)
        L_vecs_hist = torch.from_numpy(L_vecs_hist).type(settings.torchType)
        tilde_sigma2_err_hist = torch.from_numpy(tilde_sigma2_err_hist).type(settings.torchType)
        if do_pred_grid:
            # Prediction on grids
            sampled_grid_hist = prediction.pointwise_predsample_inhomogeneous(tilde_l_hist, L_vecs_hist, tilde_sigma2_err_hist, Y, x, grids, **hyper_pars)
            # print(sampled_grid_hist.size())
            if subfolder_name is None:
                with open(save_dir + folder_name + "pred_grid_hmc.pickle", "wb") as res:
                    pickle.dump(sampled_grid_hist, res)
            else:
                with open(save_dir + folder_name + subfolder_name + "pred_grid_hmc.pickle", "wb") as res:
                    pickle.dump(sampled_grid_hist, res)
        if do_pred_test:
            # Prediction on testing data
            sampled_test_hist = prediction.test_predsample_inhomogeneous(tilde_l_hist, L_vecs_hist, tilde_sigma2_err_hist, Y, x, x_test, **hyper_pars)
            # print(sampled_test_hist.size())
            if subfolder_name is None:
                with open(save_dir + folder_name + "pred_test_hmc.pickle", "wb") as res:
                    pickle.dump(sampled_test_hist, res)
            else:
                with open(save_dir + folder_name + subfolder_name + "pred_test_hmc.pickle", "wb") as res:
                    pickle.dump(sampled_test_hist, res)

    if do_freq_pred:
        ###### Frequentist Inference##########
        # Load MAP result
        with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_L_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        est_tilde_l = torch.from_numpy(est_tilde_l).type(settings.torchType)
        est_L_vecs = torch.from_numpy(est_L_vecs).type(settings.torchType)
        est_tilde_sigma2_err = torch.tensor(est_tilde_sigma2_err).type(settings.torchType)
        if do_pred_grid:
            # Prediction on grids
            gridy_quantiles = prediction.pointwise_predmap_inhomogeneous(est_tilde_l, est_L_vecs, est_tilde_sigma2_err, Y,
                                                                         x, grids, **hyper_pars)
            with open(save_dir + folder_name + subfolder_name + "pred_grid_map.pickle", "wb") as res:
                pickle.dump(gridy_quantiles, res)
        if do_pred_test:
            # Prediction on testing data
            predy_quantiles = prediction.test_predmap_inhomogeneous(est_tilde_l, est_L_vecs, est_tilde_sigma2_err, Y, x,
                                                                    x_test, **hyper_pars)
            with open(save_dir + folder_name + subfolder_name + "pred_test_map.pickle", "wb") as res:
                pickle.dump(predy_quantiles, res)

    if do_bayes_visualization:
        print("HMC result:")
        ########### Visualization (Bayesian)
        if subfolder_name is None:
            with open(save_dir + folder_name + "pred_grid_hmc.pickle", "rb") as res:
                sampled_grid_hist = pickle.load(res)
        else:
            with open(save_dir + folder_name + subfolder_name + "pred_grid_hmc.pickle", "rb") as res:
                sampled_grid_hist = pickle.load(res)
        if subfolder_name is None:
            with open(save_dir + folder_name + "pred_test_hmc.pickle", "rb") as res:
                sampled_test_hist = pickle.load(res)
        else:
            with open(save_dir + folder_name + subfolder_name + "pred_test_hmc.pickle", "rb") as res:
                sampled_test_hist = pickle.load(res)
        # print(sampled_grid_hist.size())
        # print(sampled_test_hist.size())
        sampled_grid_hist = sampled_grid_hist.data.numpy()
        sampled_grid_quantile = visualization.samples2quantiles(sampled_grid_hist)
        # print(sampled_y_quantile.shape)
        # visualization.Plot_posterior_hadamard(x.data.numpy(), indx.data.numpy(), y.data.numpy(), grids.data.numpy(), sampled_y_quantile)
        
        pred_test = torch.mean(sampled_test_hist, dim=1)
        pred_std = torch.std(sampled_test_hist, dim=1)
        
        # convert back to original data
        sampled_grid_quantile_orig = preprocess_realdata.adj2orig(sampled_grid_quantile, trend, scale)
        # print(sampled_grid_quantile_orig.shape)
        Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
        pred_test_orig = preprocess_realdata.adj2orig(pred_test.data.numpy(), trend, scale)
        pred_std_orig = pred_std.data.numpy() * scale
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale) 
        visualization.Plot_posterior_trainandtest(x.numpy() * x_scale, Y_orig, grids.numpy() * x_scale,
                                                  sampled_grid_quantile_orig, x_test=x_test.numpy() * x_scale,
                                                  Y_test=Y_test_orig, Y_pred=pred_test_orig, save_dir=save_dir,
                                                  folder_name=folder_name, subfolder_name=subfolder_name,
                                                  attributes=attributes, type="HMC")
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)   
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))
        if subfolder_name is None:
            with open(save_dir + folder_name + "bayes_res.pickle", "wb") as res:
                pickle.dump([pred_test_rmse, pred_test_lpd], res)  
        else:
            with open(save_dir + folder_name + subfolder_name + "bayes_res.pickle", "wb") as res:
                pickle.dump([pred_test_rmse, pred_test_lpd], res)  

    if do_freq_visualization:
        print("Freq result:")
        ############ Visualization (Frequentist)
        with open(save_dir + folder_name + subfolder_name + "pred_grid_map.pickle", "rb") as res:
            gridy_quantiles = pickle.load(res)
        with open(save_dir + folder_name + subfolder_name + "pred_test_map.pickle", "rb") as res:
            predy_quantiles = pickle.load(res)
        gridy_quantiles = np.transpose(gridy_quantiles.data.numpy(), axes=[1, 0, 2])

        predy_quantiles = predy_quantiles.data.numpy()
        # visualization.Plot_posterior_hadamard(x.data.numpy(), indx.data.numpy(), y.data.numpy(), grids.data.numpy(), sampled_y_quantile)
        pred_test = predy_quantiles[:, 1, :]
        pred_test_orig = preprocess_realdata.adj2orig(pred_test, trend, scale)
        pred_std = (predy_quantiles[:, 1, :] - predy_quantiles[:, 0, :]) / 1.96
        pred_std_orig = pred_std * scale

        # convert back to original scale
        gridy_quantiles_orig = preprocess_realdata.adj2orig(gridy_quantiles, trend, scale)
        # print(sampled_y_quantile_orig.shape)
        Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        # print(Y_orig.shape, gridy_quantiles_orig.shape, Y_test_orig.shape, pred_test_orig.shape)
        visualization.Plot_posterior_trainandtest(x.numpy() * x_scale, Y_orig, grids.numpy() * x_scale,
                                                  gridy_quantiles_orig, x_test=x_test.numpy() * x_scale,
                                                  Y_test=Y_test_orig, Y_pred=pred_test_orig, save_dir=save_dir,
                                                  folder_name=folder_name, subfolder_name=subfolder_name,
                                                  attributes=attributes, type="MAP")
        # # compute RMSE for tasks separately
        # pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig, axis=0)
        # print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))
        with open(save_dir + folder_name + subfolder_name + "freq_res.pickle", "wb") as res:
            pickle.dump([pred_test_rmse, pred_test_lpd], res)

    if do_real_visualization:
        x_orig = x.numpy() * x_scale
        Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
        fig = plt.figure()
        plt.scatter(x_orig, np.zeros_like(x_orig), marker="^", label="sampling time")
        for m in range(M):
            plt.plot(x_orig, Y_orig[:, m], label=attributes[m])
        plt.legend(fontsize=16)
        plt.xlabel("time (hour)", fontsize=22)
        plt.ylabel("vital", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(save_dir + folder_name + subfolder_name + "real_data.png")
        plt.close(fig)


def load_data(rank):
    # Load Kaiser data
    with open("../data/KAISER/kaiser_distributed_small.pickle", "rb") as res:
        data = pickle.load(res)
    origx, origY, attributes = data[rank]
    import pdb; pdb.set_trace()
    Y, trend, scale = preprocess_realdata.orig2adj(origx, origY)
    x_scale = np.max(origx)
    x = origx / x_scale

    # Split data for training and testing
    if do_extrapolation:
        x, x_test, Y, Y_test = utils.data_split_extrapolation(x, Y)
    else:
        x, x_test, Y, Y_test = utils.data_split(x, Y, random_state=22)
    # print(x, x_test)

    # convert numpy to torch
    x = torch.from_numpy(x).type(settings.torchType)
    Y = torch.from_numpy(Y).type(settings.torchType)
    x_test = torch.from_numpy(x_test).type(settings.torchType)
    Y_test = torch.from_numpy(Y_test).type(settings.torchType)

    return x, Y, x_test, Y_test, trend, scale, x_scale, attributes


def save_res(rmse, lpd):
    with open(save_dir + folder_name + subfolder_name + "res.pickle", "wb") as res:
        pickle.dump([rmse, lpd], res)


if __name__ == "__main__":
    do_fully_inference = True 
    do_extrapolation = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", type=int, help="ID number", default="12978238")
    args = parser.parse_args()
    save_dir = "../res/"
    # if do_fully_inference:
    #     folder_name_stationary = "kaiser_stationary_distributed/"
    #     folder_name_separable = "kaiser_separable_distributed/"
    #     folder_name = "kaiser_nonseparable_distributed/"
    # if do_extrapolation:
    #     folder_name_stationary = "kaiser_stationary_distributed_extrapolation/"
    #     folder_name_separable = "kaiser_separable_distributed_extrapolation/"
    #     folder_name = "kaiser_nonseparable_distributed_extrapolation/"

    # Load ID_dict

    folder_name = "kaiser_nonseparable/illustration/"

    with open("../data/KAISER/IDs_small.pickle", "rb") as res:
        ID_dict = pickle.load(res)
    ID2index = {ID: index for index, ID in enumerate(ID_dict)}

    ID = args.ID
    subfolder_name = "ID_{}/".format(ID)
    if not os.path.exists(save_dir + folder_name + subfolder_name):
        os.mkdir(save_dir + folder_name + subfolder_name)

    do_initialization = False
    do_MAP = False 
    do_HMC = False
    do_map_analysis = True
    do_post_analysis = True
    do_bayes_pred = False
    do_freq_pred = False
    do_pred_grid = True
    do_pred_test = True
    do_bayes_visualization = False 
    do_freq_visualization = False 
    do_real_visualization = True

    x, Y, x_test, Y_test, trend, scale, x_scale, attributes = load_data(ID2index[ID])

    ts = time.time()
    nonseparable_model(x, Y, x_test, Y_test, trend, scale, x_scale, n_opt=1000, n_hmc=100, attributes=attributes)
    print("ID_{} training costs {}s".format(ID, time.time() - ts))

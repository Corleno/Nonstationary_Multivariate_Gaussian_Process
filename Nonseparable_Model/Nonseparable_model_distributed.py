# import standard libraries
import os
import time
import matplotlib
matplotlib.use('Agg')

# import private libraries
from Nonseparable_model import *

# import parallelism libraries
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def nonseparable_model(x, Y, x_test, Y_test, trend, scale, x_scale, n_opt=1000, n_hmc=100, attributes=None):
    N, M = Y.shape
    if do_real_visualization:
        x_orig = x * x_scale
        Y_orig = preprocess_realdata.adj2orig(Y, trend, scale)
        fig = plt.figure()
        for m in range(M):
            plt.plot(x_orig, Y_orig[:, m], label=attributes[m])
        plt.legend()
        plt.savefig(save_dir + folder_name + subfolder_name + "real_data.png")
        plt.close(fig)

    # empirical estimation
    if do_empirical_estimation:
        est_sigmas, est_ls, smooth_ls, est_stds, est_R, est_B, est_L_vecs, est_tilde_sigma2_err = empirical_estimation.local_estimation(x, Y, window_size=15, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name)
        empirical_estimation.visualization(x, Y, est_ls, smooth_ls, est_stds, est_R, est_L_vecs, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)
        empirical_estimation.save_res(est_ls, smooth_ls, est_L_vecs, est_tilde_sigma2_err, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name)
        print("empirical estimation complete.")
        # import pdb
        # pdb.set_trace()

    # convert numpy to torch
    x = torch.from_numpy(x).type(settings.torchType)
    Y = torch.from_numpy(Y).type(settings.torchType)
    if not do_fully_inference:
        x_test = torch.from_numpy(x_test).type(settings.torchType)
        Y_test = torch.from_numpy(Y_test).type(settings.torchType)

    # Initialization
    hyper_pars = {"mu_tilde_l": 0., "alpha_tilde_l": 5., "beta_tilde_l": 0.1, "mu_L": 0.,
                  "alpha_L": 5., "beta_L": 0.2, "a": 1., "b": 1.}

    # train non-stationary model
    if do_training:
        train(x, Y, N_opt=n_opt, N_hmc=n_hmc, use_separable_res=False, use_empirical_res=True, do_initialization=do_initialization, do_MAP=do_MAP, do_HMC=do_HMC,
          hyper_pars=hyper_pars, save_dir=save_dir, folder_name=folder_name, folder_name_separable=folder_name_separable, subfolder_name=subfolder_name, verbose=False)
        # print("training completed.")

    if do_map_analysis:
        # Load MAP result
        with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_L_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        posterior_analysis.visualization_pos_map(x.numpy(), est_tilde_l, L_vecs=est_L_vecs, N=N, M=M, save_dir=save_dir,
                                               folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)

    if do_post_analysis:
        # Posterior analysis
        with open(save_dir + folder_name + subfolder_name + "HMC_sample.pickle", "rb") as res:
            sample = pickle.load(res)
        tilde_l_hist, L_vecs_hist, tilde_sigma2_err_hist = posterior_analysis.vec2pars_SVC(sample, N, M)
        posterior_analysis.visualization_pos(x.numpy(), tilde_l_hist, L_vecs_hist=L_vecs_hist, N=N, M=M,
                                             save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name,
                                             attributes=attributes)

    # use 201 equal-spaced grids for predictive processes 
    # grids = np.linspace(0., 1., 201)
    # use timestamps in LAPS2 for predictive processes
    grids = x_grids
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
        print("frequentist prediction start.")
        # Load MAP result
        with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_L_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        est_tilde_l = torch.from_numpy(est_tilde_l).type(settings.torchType)
        est_L_vecs = torch.from_numpy(est_L_vecs).type(settings.torchType)
        est_tilde_sigma2_err = torch.tensor(est_tilde_sigma2_err).type(settings.torchType)
        print("MAP estimation load complete.") 
        
        if do_pred_grids:
            print("prediction on grids start.")
            # Prediction on grids
            gridy_quantiles, grid_L_vecs = prediction.pointwise_predmap_inhomogeneous(est_tilde_l, est_L_vecs, est_tilde_sigma2_err, Y, x, grids, **hyper_pars)
            with open(save_dir + folder_name + subfolder_name + "pred_grid_map.pickle", "wb") as res:
                pickle.dump([gridy_quantiles, grid_L_vecs], res)
            Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
            gridy_quantiles = np.transpose(gridy_quantiles.data.numpy(), axes=[1, 0, 2])
            gridy_quantiles_orig = preprocess_realdata.adj2orig(gridy_quantiles, trend, scale)
            visualization.Plot_posterior(x.numpy()*x_scale, Y_orig, grids.numpy()*x_scale, gridy_quantiles_orig,
                                         save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)

        if do_pred_test:
            # Prediction on testing data
            predy_quantiles, _ = prediction.test_predmap_inhomogeneous(est_tilde_l, est_L_vecs, est_tilde_sigma2_err, Y, x,
                                                                    x_test, **hyper_pars)
            with open(save_dir + folder_name + subfolder_name + "pred_test_map.pickle", "wb") as res:
                pickle.dump(predy_quantiles, res)

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
        with open(save_dir + folder_name + subfolder_name + "pred_grid_map.pickle", "rb") as res:
            gridy_quantiles, _ = pickle.load(res)
        with open(save_dir + folder_name + subfolder_name + "pred_test_map.pickle", "rb") as res:
            predy_quantiles = pickle.load(res)
        gridy_quantiles = np.transpose(gridy_quantiles.data.numpy(), axes=[1, 0, 2])

        predy_quantiles = predy_quantiles.data.numpy()
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
                                                  attributes=attributes)
        # # compute RMSE for tasks separately
        # pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig, axis=0)
        # print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)
        # print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))
        with open(save_dir + folder_name + subfolder_name + "freq_res.pickle", "wb") as res:
            pickle.dump([pred_test_rmse, pred_test_lpd], res)


def load_data(rank, ID, extrapolation=False, fully_inference=False, do_small = False, save_dir=None, folder_name=None):
    # Load Kaiser data
    if do_small:
        with open("../data/kaiser_distributed_small.pickle", "rb") as res:
            data = pickle.load(res)
    else:  
        with open("../data/kaiser_distributed.pickle", "rb") as res:
            data = pickle.load(res)
    origx, origY, attributes = data[rank]
    Y, trend, scale = preprocess_realdata.orig2adj(origx, origY)
    x_scale = np.max(origx)
    x = origx / x_scale
    # Load LAPS2 data
    df = pd.read_csv("../data/LLNL_HOURLY_LAPS2.CSV")
    hour, _ = extract_LAPS.extract_LAPS(ID, df, save_dir=save_dir, folder_name=folder_name, verbose=False)
    x_grids = hour/x_scale
    # Split data for training and testing
    if extrapolation:
        x_train, x_test, Y_train, Y_test = utils.data_split_extrapolation(x, Y)
    elif fully_inference:
        x_train, x_test, Y_train, Y_test = x, None, Y, None
    else:
        x_train, x_test, Y_train, Y_test = utils.data_split(x, Y, random_state=22)
    return x_train, Y_train, x_test, Y_test, trend, scale, x_scale, x_grids, attributes


def save_res(rmse, lpd):
    with open(save_dir + folder_name + subfolder_name + "res.pickle", "wb") as res:
        pickle.dump([rmse, lpd], res)


if __name__ == "__main__":
    do_sample = False 
    do_small = False 
    if do_sample:
        np.random.seed(22)
        rank2index = np.random.choice(2451, size, replace=False)
    else:
        rank2index = np.arange(2451)
    do_extrapolation = False 
    do_fully_inference = True

    save_dir = "../res/"
    if do_extrapolation:
        folder_name_stationary = "kaiser_stationary_distributed_extrapolation0/"
        folder_name_separable = "kaiser_separable_distributed_extrapolation0/"
        folder_name = "kaiser_nonseparable_distributed_extrapolation0/"
    if do_fully_inference:
        folder_name_stationary = "kaiser_stationary_distributed/"
        folder_name_separable = "kaiser_separable_distributed/"
        folder_name = "kaiser_nonseparable_distributed/"

    # Load ID_dict
    if do_small:
        with open("../data/IDs_small.pickle", "rb") as res:
            ID_dict = pickle.load(res)
    else:
        with open("../data/IDs.pickle", "rb") as res:
            ID_dict = pickle.load(res)
    ID2index = {ID: index for index, ID in enumerate(ID_dict)}

    ID = ID_dict[rank2index[rank]]
    subfolder_name = "ID_{}/".format(ID)
    if not os.path.exists(save_dir + folder_name + subfolder_name):
        os.mkdir(save_dir + folder_name + subfolder_name)

    do_empirical_estimation = False 
    do_initialization = True
    do_training = False
    do_MAP = False 
    do_HMC = False
    do_map_analysis = False
    do_post_analysis = False
    do_bayes_pred = False
    do_freq_pred = True 
    do_pred_grids = True
    do_pred_test = False
    do_bayes_visualization = False
    do_freq_visualization = False
    do_real_visualization = False 

    rank = ID2index[ID]
    
    x_train, Y_train, x_test, Y_test, trend, scale, x_scale, x_grids, attributes = load_data(rank, ID, extrapolation=do_extrapolation, fully_inference=do_fully_inference, do_small = do_small, save_dir = save_dir, folder_name = folder_name)
    print("Data size:{} by {}".format(*(Y_train.shape)))

    ts = time.time()
    nonseparable_model(x_train, Y_train, x_test, Y_test, trend, scale, x_scale, n_opt=1000, n_hmc=1000, attributes=attributes)
    print("ID_{} training costs {}s".format(ID, time.time() - ts))

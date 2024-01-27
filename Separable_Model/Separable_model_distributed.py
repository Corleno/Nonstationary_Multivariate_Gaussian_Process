# import standard libraries
import os
import time
import matplotlib
matplotlib.use('Agg')

# import private libraries
from Separable_model import *

# import parallelism libraries
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def separable_model(x, x_test, Y, Y_test, trend, scale, x_scale, n_opt=1000, n_hmc=1000):
    # convert numpy to torch
    x = torch.from_numpy(x).type(settings.torchType)
    Y = torch.from_numpy(Y).type(settings.torchType)
    x_test = torch.from_numpy(x_test).type(settings.torchType)
    Y_test = torch.from_numpy(Y_test).type(settings.torchType)

    # Initialization
    N, M = Y.shape
    hyper_pars = {"mu_tilde_l": 0., "alpha_tilde_l": 5., "beta_tilde_l": 0.1, "mu_tilde_sigma": 0.,
                  "alpha_tilde_sigma": 5., "beta_tilde_sigma": 0.1, "a": 1., "b": 1., "c": 10.}

    # import pdb
    # pdb.set_trace()
    # train non-stationary model
    train(x, Y, N_opt=n_opt, N_hmc=n_hmc, do_MAP=do_MAP, do_HMC=do_HMC, save_dir=save_dir, folder_name=folder_name, folder_name_stationary=folder_name_stationary, subfolder_name=subfolder_name, hyper_pars=hyper_pars)

    if do_map_analysis:
        # Load MAP result
        with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err = posterior_analysis.vec2pars_est(estPars, N, M)

        x_array = x.numpy()
        order = np.argsort(x_array)
        x_array = x_array[order]
        # Plot est_tilde_l
        fig = plt.figure()
        plt.plot(x_array, est_tilde_l[order])
        plt.savefig(save_dir + folder_name + subfolder_name + "est_tilde_l_map.png")
        # Plot est_tilde_sigma
        fig = plt.figure()
        plt.plot(x_array, est_tilde_sigma[order])
        plt.savefig(save_dir + folder_name + subfolder_name + "est_tilde_sigma_map.png")

    if do_pos_analysis:
        # Posterior analysis
        with open(save_dir + folder_name + subfolder_name + "HMC_sample.pickle", "rb") as res:
            sample = pickle.load(res)
        tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist = posterior_analysis.vec2pars(sample, N,
                                                                                                        M)  # array
        posterior_analysis.visualization_pos(x.numpy(), tilde_l_hist, tilde_sigma_hist)

    grids = np.linspace(0., 1., 201)
    grids = torch.from_numpy(grids).type(settings.torchType)

    if do_bayes_pred:
        if do_pred_inf:
            # Predictive inference
            tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist = prediction.vec2pars(
                torch.from_numpy(sample).type(settings.torchType), N, M)  # tensor
            sampled_y_hist = prediction.pointwise_predsample(tilde_l_hist, tilde_sigma_hist, L_vec_hist,
                                                             tilde_sigma2_err_hist, Y, x, grids, **hyper_pars)
            print(sampled_y_hist.size())
            with open(save_dir + folder_name + subfolder_name + "pred_res.pickle", "wb") as res:
                pickle.dump(sampled_y_hist, res)

        if do_pred_test:
            # Predict testing data
            tilde_l_hist, tilde_sigma_hist, L_vec_hist, tilde_sigma2_err_hist = prediction.vec2pars(
                torch.from_numpy(sample).type(settings.torchType), N, M)  # tensor
            sampled_test_hist = prediction.test_predsample(tilde_l_hist, tilde_sigma_hist, L_vec_hist,
                                                           tilde_sigma2_err_hist, Y, x, x_test, **hyper_pars)
            print(sampled_test_hist.size())
            with open(save_dir + folder_name + subfolder_name + "pred_test_res.pickle", "wb") as res:
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
        with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err = posterior_analysis.vec2pars_est(estPars, N, M)
        est_tilde_l = torch.from_numpy(est_tilde_l).type(settings.torchType)
        est_tilde_sigma = torch.from_numpy(est_tilde_sigma).type(settings.torchType)
        est_L_vec = torch.from_numpy(est_L_vec).type(settings.torchType)
        est_tilde_sigma2_err = torch.FloatTensor([est_tilde_sigma2_err]).type(settings.torchType)
        if do_pred_inf:
            # Predictive inference
            gridy_quantiles = prediction.pointwise_predmap(est_tilde_l, est_tilde_sigma, est_L_vec,
                                                           est_tilde_sigma2_err, Y, x, grids, **hyper_pars)
            # print(gridy_quantiles.size())
            with open(save_dir + folder_name + subfolder_name + "pred_resmap.pickle", "wb") as res:
                pickle.dump(gridy_quantiles, res)

        if do_pred_test:
            # Predict testing data
            testy_quantiles = prediction.test_predmap(est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err, Y,
                                                      x, x_test, **hyper_pars)
            # print(testy_quantiles.size())
            with open(save_dir + folder_name + subfolder_name + "pred_test_resmap.pickle", "wb") as res:
                pickle.dump(testy_quantiles, res)

    if do_vis_bayes:
        print("HMC result:")
        # Visualization
        with open(save_dir + folder_name + subfolder_name + "pred_res.pickle", "rb") as res:
            sampled_y_hist = pickle.load(res)
        with open(save_dir + folder_name + subfolder_name + "pred_test_res.pickle", "rb") as res:
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
        pred_std_orig = pred_std.data.numpy() * scale
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        visualization.Plot_posterior_trainandtest(x.numpy() * x_scale, Y_orig, grids.numpy() * x_scale,
                                                  sampled_y_quantile_orig, x_test=x_test.numpy() * x_scale,
                                                  Y_test=Y_test_orig, Y_pred=pred_test_orig)
        # compute RMSE for tasks separately
        # pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig, axis=0)
        # print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))
        with open(save_dir + folder_name + subfolder_name + "bayes_res.pickle", "wb") as res:
            pickle.dump([pred_test_rmse, pred_test_lpd], res)

    if do_vis_freq:
        # print("MAP results:")
        # Visualization
        with open(save_dir + folder_name + subfolder_name + "pred_resmap.pickle", "rb") as res:
            gridy_quantile = pickle.load(res)
        with open(save_dir + folder_name + subfolder_name + "pred_test_resmap.pickle", "rb") as res:
            testy_quantile = pickle.load(res)
        gridy_quantile = gridy_quantile.data.numpy()
        # print(sampled_y_quantile.shape)
        # visualization.Plot_posterior(x.data.numpy(), Y.data.numpy(), grids.data.numpy(), sampled_y_quantile)

        pred_test = testy_quantile[:, 1, :]
        pred_std = (testy_quantile[:, 1, :] - testy_quantile[:, 0, :]) / 1.96

        # convert back to original data
        gridy_quantile_orig = preprocess_realdata.adj2orig(gridy_quantile, trend, scale)
        # print(sampled_y_quantile_orig.shape)
        Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
        pred_test_orig = preprocess_realdata.adj2orig(pred_test.data.numpy(), trend, scale)
        pred_std_orig = pred_std.data.numpy() * scale
        Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
        visualization.Plot_posterior_trainandtest(x.numpy() * x_scale, Y_orig, grids.numpy() * x_scale,
                                                  np.transpose(gridy_quantile_orig, axes=(1, 0, 2)),
                                                  x_test=x_test.numpy() * x_scale, Y_test=Y_test_orig,
                                                  Y_pred=pred_test_orig, save_dir=save_dir, folder_name=folder_name,
                                                  subfolder_name=subfolder_name, attributes=attributes)
        # compute RMSE for tasks separately
        # pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig, axis=0)
        # print("RMSE_tasks = {}".format(pred_test_rmse))
        # compute RMSE, LPD for all tasks
        pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
        pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)
        # print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))
        with open(save_dir + folder_name + subfolder_name + "freq_res.pickle", "wb") as res:
            pickle.dump([pred_test_rmse, pred_test_lpd], res)


def load_data(rank, extrapolation=False):
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
    # Split data for training and testing
    if extrapolation:
        x_train, x_test, Y_train, Y_test = utils.data_split_extrapolation(x, Y)
    else:
        x_train, x_test, Y_train, Y_test = utils.data_split(x, Y, random_state=22)
    return x_train, x_test, Y_train, Y_test, trend, scale, x_scale, attributes


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
        rank2index = np.arange(size)
    save_dir = "../res/"
    # folder_name_stationary = "kaiser_stationary_distributed/"
    # folder_name = "kaiser_separable_distributed/"
    if do_small:
        folder_name_stationary = "kaiser_stationary_distributed_extrapolation/"
        folder_name = "kaiser_separable_distributed_extrapolation/"
    else:
        folder_name_stationary = "kaiser_stationary_distributed_extrapolation0/"
        folder_name = "kaiser_separable_distributed_extrapolation0/"

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

    do_MAP = True
    do_HMC = False
    do_map_analysis = True
    do_pos_analysis = False
    do_bayes_pred = False
    do_freq_pred = True
    do_pred_inf = True
    do_pred_test = True
    do_vis_bayes = False
    do_vis_freq = True

    x_train, x_test, Y_train, Y_test, trend, scale, x_scale, attributes = load_data(ID2index[ID], extrapolation=True)

    ts = time.time()
    separable_model(x_train, x_test, Y_train, Y_test, trend, scale, x_scale, n_opt=1000)
    print("ID_{} training costs {}s".format(ID, time.time() - ts))

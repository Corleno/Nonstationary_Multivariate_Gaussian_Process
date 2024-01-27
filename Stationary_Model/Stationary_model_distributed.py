# import standard libraries
import os
import time
import matplotlib
matplotlib.use('Agg')

# import private libraries
from Stationary_model import *

# import parallelism libraries
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def stationary_model(x, x_test, Y, Y_test, trend, scale, x_scale, do_MAP=False, n_iter=1000, verbose=False):
    # convert numpy to torch
    x = torch.from_numpy(x).type(settings.torchType)
    Y = torch.from_numpy(Y).type(settings.torchType)
    x_test = torch.from_numpy(x_test).type(settings.torchType)
    Y_test = torch.from_numpy(Y_test).type(settings.torchType)

    est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err = train(x, Y, N_opt=n_iter, do_MAP=do_MAP, save_dir=save_dir, folder_name=folder_name, subfolder_name=subfolder_name, verbose=verbose)

    # Predictive inference
    grids = np.linspace(0., 1., 201)
    grids = torch.from_numpy(grids).type(settings.torchType)
    pred_grids_percentiles = prediction.pointwise_predmap_S(est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err,
                                            Y, x, grids)
    pred_grids_percentiles = torch.transpose(pred_grids_percentiles, 0, 1)
    # visualization.Plot_posterior(x.numpy(), Y.numpy(), grids.numpy(), pred_y_percentiles.data.numpy())

    # Predict testing data
    pred_testdata, pred_teststd = prediction.test_predmap_S(est_tilde_l, est_tilde_sigma, est_L_vec, est_tilde_sigma2_err, Y, x, x_test)

    # convert back to original data
    pred_grids_quantile_orig = preprocess_realdata.adj2orig(pred_grids_percentiles.data.numpy(), trend, scale)
    Y_orig = preprocess_realdata.adj2orig(Y.data.numpy(), trend, scale)
    pred_test_orig = preprocess_realdata.adj2orig(pred_testdata.data.numpy(), trend, scale)
    pred_std_orig = pred_teststd.data.numpy() * scale
    Y_test_orig = preprocess_realdata.adj2orig(Y_test.data.numpy(), trend, scale)
    # visualization.Plot_posterior(x.data.numpy()*x_scale, Y_orig, grids.data.numpy()*x_scale, pred_grids_quantile_orig)
    visualization.Plot_posterior_trainandtest(x.numpy() * x_scale, Y_orig, grids.numpy() * x_scale,
                                              pred_grids_quantile_orig, x_test=x_test.numpy() * x_scale,
                                              Y_test=Y_test_orig, Y_pred=pred_test_orig, save_dir=save_dir,
                                              folder_name=folder_name, subfolder_name=subfolder_name,
                                              attributes=attributes)
    # compute RMSE for tasks separately
    # pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig, axis=0)
    # print("RMSE_tasks = {}".format(pred_test_rmse))
    # compute RMSE, LPD for all tasks
    pred_test_rmse = utils.RMSE(pred_test_orig, Y_test_orig)
    pred_test_lpd = utils.LPD(pred_test_orig, pred_std_orig, Y_test_orig)
    if verbose:
        print("RMSE = {}, LPD = {}".format(pred_test_rmse, pred_test_lpd))
    return pred_test_rmse, pred_test_lpd


def load_data(rank, extrapolation=False):
    # Load Kaiser data
    if do_small:
        with open("../data/kaiser_distributed_small.pickle", "rb") as res:
            data = pickle.load(res)
    else:
        with open("../data/kaiser_distributed.pickle", "rb") as res:
            data = pickle.load(res)
    # import pdb
    # pdb.set_trace()
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
    with open(save_dir + folder_name + subfolder_name + "freq_res.pickle", "wb") as res:
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
    # folder_name = "kaiser_stationary_distributed/"
    # folder_name = "kaiser_stationary_distributed_extrapolation/"
    # folder_name = "kaiser_stationary_distributed_extrapolation0/"
    folder_name = "kaiser_stationary_distributed_interpolation"
    do_MAP = True 

    #Load ID_dict
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

    x_train, x_test, Y_train, Y_test, trend, scale, x_scale, attributes = load_data(ID2index[ID], extrapolation=True)
    print("N_train={}, M={}".format(Y_train.shape[0], Y_train.shape[1]))
    # import pdb
    # pdb.set_trace()

    ts = time.time()
    pred_test_rmse, pred_test_lpd = stationary_model(x_train, x_test, Y_train, Y_test, trend, scale, x_scale, do_MAP=do_MAP, n_iter=1000, verbose=False)
    print("ID_{} training costs {}s".format(ID, time.time()-ts))

    save_res(pred_test_rmse, pred_test_lpd)



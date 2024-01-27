import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import private libraries
import preprocess_realdata
import utils

def explore_kaiser_res(ID, save_dir=None, folder_name=None, folder_name_stationary=None, folder_name_separable=None):
    # load kaiser data
    # origx, origY, attributes = data[rank]
    # Y, trend, scale = preprocess_realdata.orig2adj(origx, origY)
    # x_train, x_test, Y_train, Y_test = utils.data_split_extrapolation(x, Y)
    # with open(save_dir + folder_name_stationary + subfolder_name + "pred_grid_map.pickle", "rb") as res:
    #     gridy_quantiles = pickle.load(res)
    # with open(save_dir + folder_name_stationary + subfolder_name + "pred_test_map.pickle", "rb") as res:
    #     predy_quantiles = pickle.load(res)
    
    with open(save_dir + folder_name_stationary + "ID_{}/freq_res.pickle".format(ID), "rb") as res:
        rmse, lpd = pickle.load(res)    
    print("stationary model:")
    print("rmse: {}, lpd: {}".format(rmse, lpd))
    with open(save_dir + folder_name_separable + "ID_{}/freq_res.pickle".format(ID), "rb") as res:
        rmse, lpd = pickle.load(res)  
    print("separable model:")
    print("rmse: {}, lpd: {}".format(rmse, lpd))
    with open(save_dir + folder_name + "ID_{}/freq_res.pickle".format(ID), "rb") as res:
        rmse, lpd = pickle.load(res)    
    print("nonseparable model:")
    print("rmse: {}, lpd: {}".format(rmse, lpd))
 

if __name__ == "__main__":
    save_dir = "../res/"
    folder_name_stationary = "kaiser_stationary_distributed_extrapolation/"
    folder_name_separable = "kaiser_separable_distributed_extrapolation/"
    folder_name = "kaiser_nonseparable_distributed_extrapolation/"
    # Load ID_dict
    with open("../data/IDs_small.pickle", "rb") as res:
        ID_dict = pickle.load(res)
    ID2index = {ID: index for index, ID in enumerate(ID_dict)}
    # Load raw data
    with open("../data/kaiser_distributed_small.pickle", "rb") as res:
        data = pickle.load(res)
    ID_targets = [41168468]
    for ID in ID_targets:
        rank = ID2index[ID]
        explore_kaiser_res(ID, save_dir=save_dir, folder_name=folder_name, folder_name_stationary=folder_name_stationary, folder_name_separable=folder_name_separable) 

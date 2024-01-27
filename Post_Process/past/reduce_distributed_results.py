import numpy as np
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="three models are available: stationary, separable and nonseparable", type=str)
    parser.add_argument('--data', help="prediction, extrapolation", default="prediction")
    parser.add_argument('--assign_IDs', help="kaiser_stationary_distributed/, kaiser_separable_distributed/ or "
                                             "kaiser_nonseparable_distributed/", type=str, default=None)
    args = parser.parse_args()

    save_dir = "../res/"
    do_small = False

    # Load ID_dict
    if args.assign_IDs is None:
        if do_small:
            with open("../data/IDs_small.pickle", "rb") as res:
                ID_dict = pickle.load(res)
        else:
            with open("../data/IDs.pickle", "rb") as res:
                ID_dict = pickle.load(res)
    else:
        with open(save_dir + args.assign_IDs + 'valid_IDs.pickle', "rb") as res:
            ID_dict = pickle.load(res)
    ID2index = {ID: index for index, ID in enumerate(ID_dict)}

    if args.data == "prediction":
        if args.model == "stationary":
            folder_name = "kaiser_stationary_distributed/"
        if args.model == "separable":
            folder_name = "kaiser_separable_distributed/"
        if args.model == "nonseparable":
            folder_name = "kaiser_nonseparable_distributed/"
    if args.data == "extrapolation":
        if do_small:
            if args.model == "stationary":
                folder_name = "kaiser_stationary_distributed_extrapolation/"
            if args.model == "separable":
                folder_name = "kaiser_separable_distributed_extrapolation/"
            if args.model == "nonseparable":
                folder_name = "kaiser_nonseparable_distributed_extrapolation/"
        else:
            if args.model == "stationary":
                folder_name = "kaiser_stationary_distributed_extrapolation0/"
            if args.model == "separable":
                folder_name = "kaiser_separable_distributed_extrapolation0/"
            if args.model == "nonseparable":
                folder_name = "kaiser_nonseparable_distributed_extrapolation0/"
 

    valid_num = 0
    rmse_list = []
    lpd_list = []
    valid_IDs = []
    for ID in ID_dict:
        try:
            with open(save_dir + folder_name + "ID_{}/freq_res.pickle".format(ID), "rb") as res:
                rmse, lpd = pickle.load(res)
            valid_num += 1
            rmse_list.append(rmse)
            lpd_list.append(lpd)
            valid_IDs.append(ID)
        except:
            pass
            print("ID {} is not available".format(ID))
    print("validate number: {}".format(valid_num))
    rmses = np.stack(rmse_list)
    lpds = np.stack(lpd_list)
    print(rmses)
    # print(lpds)
    print("rmse: median {}, mean {}, std {}, range {}-{} ".format(np.median(rmses), np.mean(rmses), np.std(rmses), np.min(rmses), np. max(rmses)))
    print("lpd: median{},  mean {}, std {}, range {}-{}".format(np.median(lpds), np.mean(lpds), np.std(lpds), np.min(lpds), np.max(lpds)))
    if args.assign_IDs is None:
        with open(save_dir + folder_name + "valid_IDs.pickle", "wb") as res:
            pickle.dump(valid_IDs, res)

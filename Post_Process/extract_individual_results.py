import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc 
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import argparse

import sys
sys.path.append("..")
import os
from KAISER_code.vital_features import extract_LAPS
from Utility import posterior_analysis


def load_kaiser_data(data_dir="../data/KAISER/", ID=None, group="sepsis"):
    with open(data_dir + "{}/ID_{}.pickle".format(group, ID), "rb")as res:
        origx, origY, attributes = pickle.load(res)
    return origx, origY, attributes

def standarized(x):
    new_x = x - np.mean(x)
    new_x = new_x / np.std(new_x)
    return new_x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default="sepsis")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ID", type=str, default='33285179')
    # sepsis 16746487
    # nonsepsis 13197304
    args = parser.parse_args()
    save_dir = "../res/"
    folder_name = "kaiser_nonseparable/"

    if args.ID == "random":
        np.random.seed(22)
        if args.seed is None:
            print("start")
            seed_set = [0, 1]
            ID_sepsis_list = []
            ID_nonsepsis_list = []
            for seed in seed_set:
                print(seed)
                ID_dir = "../data/KAISER/IDs_sampled_seed{}.pickle".format(seed)
                with open(ID_dir, "rb") as res:
                    ID_sepsis, ID_nonsepsis = pickle.load(res)
                ID_sepsis_list.append(ID_sepsis)
                ID_nonsepsis_list.append(ID_nonsepsis)
            ID_sepsis = np.concatenate(ID_sepsis_list)
            ID_nonsepsis = np.concatenate(ID_nonsepsis_list)
            print("ID_sepsis size: {}".format(ID_sepsis.shape))
            print("ID_nonsepsis size: {}".format(ID_nonsepsis.shape))
        else:
            ID_dir = "../data/KAISER/IDs_sampled_seed{}.pickle".format(args.seed)
            with open(ID_dir, "rb") as res:
                ID_sepsis, ID_nonsepsis = pickle.load(res)
        if args.group == "sepsis":
            ID = np.random.choice(ID_sepsis)
        else:
            ID = np.random.choice(ID_nonsepsis)

    if not os.path.exists("../res/kaiser_nonseparable/{}/ID_{}/pred_smoothness_grid_map.pickle".format(args.group, args.ID)):
        print("ID is not available")     

    df = pd.read_csv("../data/KAISER/LLNL_HOURLY_LAPS2.CSV")

    if not os.path.exists(save_dir + folder_name + "illustration/ID_{}".format(args.ID)):
        os.mkdir(save_dir + folder_name + "illustration/ID_{}".format(args.ID))

    # generate the observation plot
    x, y, attributes = load_kaiser_data(ID=args.ID, group=args.group)
    fig = plt.figure()
    # plt.plot(x, y[:, 0], label=attributes[0])
    # plt.plot(x, y[:, 1], label=attributes[1])
    
    plt.plot(x, y[:, 0], label=r"$\textbf{BPDIA}$")
    plt.plot(x, y[:, 1], label=r"$\textbf{BPSYS}$")
    # plt.plot(x, y[:, 3], label=attributes[3])
    plt.xlabel("Hour", fontsize=18)
    plt.ylabel("Vitals", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_dir + folder_name + "illustration/ID_{}/obs.pdf".format(args.ID))

    # generate the LAPS2 plot
    hour, laps2 = extract_LAPS.extract_LAPS(int(args.ID), df)
    fig = plt.figure()
    plt.plot(hour, laps2)
    plt.xlabel("Hour", fontsize=18)
    plt.ylabel("LAPS2", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir + folder_name + "illustration/ID_{}/laps2.pdf".format(args.ID))

    # generate latent process
    # load inferred result
    with open("../res/kaiser_nonseparable/{}/ID_{}/pred_smoothness_grid_map.pickle".format(args.group, args.ID), "rb") as res:
        sampled_tilde_ls = pickle.load(res)
    with open("../res/kaiser_nonseparable/{}/ID_{}/pred_cov_grid_map.pickle".format(args.group, args.ID), "rb") as res:
        sampled_L_fs = pickle.load(res)
    sampled_B_fs = np.stack([np.stack([np.matmul(L_f, L_f.T) for L_f in sampled_L_f]) for sampled_L_f in sampled_L_fs])
    sampled_R_fs = np.stack([np.stack([posterior_analysis.cov2cor(B_f) for B_f in sampled_B_f]) for sampled_B_f in sampled_B_fs])
    est_tilde_ls = np.mean(sampled_tilde_ls, axis = 1)
    est_R_fs = np.mean(sampled_R_fs, axis = 1)
    
    laps2_std = standarized(laps2)

    fig = plt.figure()
    plt.plot(hour, est_tilde_ls)
    plt.xlabel("Hour", fontsize=18)
    plt.ylabel("Smoothness", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir + folder_name + "illustration/ID_{}/smoothness_process.pdf".format(args.ID))
    # fig = plt.figure()
    # plt.scatter(standarized(est_tilde_ls), laps2_std)
    # plt.xlabel("standard smoothness")
    # plt.ylabel("standard laps2")
    # plt.tight_layout()
    # plt.savefig(save_dir + folder_name + "illustration/ID_{}/smoothness_vs_laps2.pdf".format(args.ID))
    
    fig = plt.figure()
    plt.plot(hour, est_R_fs[:, 0, 1])
    plt.xlabel("Hour", fontsize=18)
    plt.ylabel("Correlation", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir + folder_name + "illustration/ID_{}/corr_process_{}_vs_{}.pdf".format(args.ID, attributes[0], attributes[1]))
    # fig = plt.figure()
    # plt.scatter(standarized(est_R_fs[:, 0 ,1]), laps2_std)
    # plt.xlabel("standard correlation")
    # plt.ylabel("standard laps2")
    # plt.tight_layout()
    # plt.savefig(save_dir + folder_name + "illustration/ID_{}/correlation_{}_vs_{}_vs_laps2.pdf".format(args.ID, attributes[0], attributes[1]))

    # fig = plt.figure()
    # plt.plot(hour, est_R_fs[:, 0, 3])
    # plt.xlabel("hour")
    # plt.ylabel("correlation")
    # plt.tight_layout()
    # plt.savefig(save_dir + folder_name + "illustration/ID_{}/corr_process_{}_vs_{}.pdf".format(args.ID, attributes[0], attributes[3]))
    # fig = plt.figure()
    # plt.scatter(standarized(est_R_fs[:, 0 ,3]), laps2_std)
    # plt.xlabel("standard correlation")
    # plt.ylabel("standard laps2")
    # plt.tight_layout()
    # plt.savefig(save_dir + folder_name + "illustration/ID_{}/correlation_{}_vs_{}_vs_laps2.pdf".format(args.ID, attributes[0], attributes[3]))

    # fig = plt.figure()
    # plt.plot(hour, est_R_fs[:, 1, 3])
    # plt.xlabel("hour")
    # plt.ylabel("correlation")
    # plt.tight_layout()
    # plt.savefig(save_dir + folder_name + "illustration/ID_{}/corr_process_{}_vs_{}.pdf".format(args.ID, attributes[1], attributes[3]))
    # fig = plt.figure()
    # plt.scatter(standarized(est_R_fs[:, 1 ,3]), laps2_std)
    # plt.xlabel("standard correlation")
    # plt.ylabel("standard laps2")
    # plt.tight_layout()
    # plt.savefig(save_dir + folder_name + "illustration/ID_{}/correlation_{}_vs_{}_vs_laps2.pdf".format(args.ID, attributes[1], attributes[3]))

    if not os.path.exists(save_dir + folder_name + "illustration/ID_{}/corrmap".format(args.ID)):
        os.mkdir(save_dir + folder_name + "illustration/ID_{}/corrmap".format(args.ID))

    # attributes = [r"$\textbf{BPDIA}$", r"$\textbf{BPSYS}$", r"$\textbf{HRTRT}$", r"$\textbf{O2SAT}$", r"$\textbf{PP}$"]
    attributes = [r"$\textbf{BPDIA}$", r"$\textbf{BPSYS}$", r"$\textbf{HRTRT}$", r"$\textbf{O2SAT}$"]
    posterior_analysis.visualization_pos_map_heatmap_withR_s(hour, R_fs=est_R_fs, save_dir=save_dir,
                                        folder_name=folder_name, subfolder_name="illustration/ID_{}/".format(args.ID), attributes=attributes)

    


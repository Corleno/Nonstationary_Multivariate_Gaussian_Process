import pickle
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# matplotlib.use('TKAgg', warn=False, force=True)
import argparse

import sys
sys.path.append("..")
import os
from KAISER_code.vital_features import extract_LAPS
from Utility import posterior_analysis
import scipy.stats as stats


def t_test(x):
    # H0: mu = 0
    n = x.shape[0]
    hat_mu = np.mean(x)
    hat_std = np.std(x)
    t_value = (hat_mu - 0)/(hat_std/np.sqrt(n))
    p_value = 2*(1 - stats.t.cdf(np.abs(t_value), df = n-1))
    return t_value, p_value


def sign_test(x):
    # H0: a nonparamtric hypothesis test: median is 0
    x_pos = np.sum(x > 0)
    x_neg = np.sum(x < 0)
    # import pdb
    # pdb.set_trace()
    p_value = 2*stats.binom.cdf(k=np.min([x_pos, x_neg]), n=x_pos+x_neg, p=0.5)
    return p_value


def z_test(x):
    # H0: a paramtric hypothesis test: p = 1/2 where N+ - Binomial(N, 0.5):
    n = x.shape[0]
    hat_p = np.sum(x>0)/n
    z_value = (hat_p-0.5)/np.sqrt(0.5*(1-0.5)/n)
    p_value = 2*(1 - stats.norm.cdf(np.abs(z_value)))
    return z_value, p_value


def compute_coverage_rate(X, credible_quantile=0.95):
    # compute the probability of credible interval covering the 0 point
    left , mid, right = 0., 0., 0.
    n_individual, n_sample = X.shape
    for x in X:
        # compute the credible interval
        c_min, c_max = np.percentile(x, q = np.array([(1 - credible_quantile)/2, 1 - (1 - credible_quantile)/2])*100)
        # import pdb
        # pdb.set_trace()
        # print(c_max, c_max)
        if c_max < 0:
            left += 1
        if (c_min < 0) & (c_max > 0):
            mid += 1
        if c_min > 0:
            right += 1
    return left/n_individual, mid/n_individual, right/n_individual


def analysis_datasize(verbose=False):
    # analysis of the data size of complete records
    with open("../data/KAISER/ID2N_R/ID2N_R.pickle", "rb") as res:
        ID2N_R = pickle.load(res)
    with open("../data/KAISER/ID2N_R/ID2TS.pickle", "rb") as res:
        ID2TS = pickle.load(res)

    N_R_sepsis = list()
    N_R_nonsepsis = list()
    MTS_sepsis = list()
    MTS_nonsepsis = list()
    ID_sepsis_updated = list()
    ID_nonsepsis_updated = list()

    print("start to compute N_R_sepsis.")
    for ID in ID_sepsis:
        try:
            N_R_sepsis.append(ID2N_R[ID])
            if ID2N_R[ID] == 0:
                MTS_sepsis.append(0)
            else:
                MTS_sepsis.append(np.max(ID2TS[ID]))
            ID_sepsis_updated.append(ID)
        except:
            print("ID {} is not avaiable.".format(ID))
    print("start to compute N_R_nonsepsis.")
    for ID in ID_nonsepsis:
        try:
            N_R_nonsepsis.append(ID2N_R[ID])
            if ID2N_R[ID] == 0:
                MTS_nonsepsis.append(0)
            else:
                MTS_nonsepsis.append(np.max(ID2TS[ID]))
            ID_nonsepsis_updated.append(ID)
        except:
            print("ID {} is not avaiable.".format(ID))
    N_R_sepsis = np.array(N_R_sepsis)
    N_R_nonsepsis = np.array(N_R_nonsepsis)
    MTS_sepsis = np.array(MTS_sepsis)
    MTS_nonsepsis = np.array(MTS_nonsepsis)
    ID_sepsis_updated = np.array(ID_sepsis_updated)
    ID_nonsepsis_updated = np.array(ID_nonsepsis_updated)
    print("{} patients have sepsis.".format(N_R_sepsis.shape[0]))
    print("{} patients have nonsepsis.".format(N_R_nonsepsis.shape[0]))
    N_Rs = np.concatenate([N_R_sepsis, N_R_nonsepsis])
    print("range of the number of records: ({}, {})".format(np.min(N_Rs), np.max(N_Rs)))

    if verbose:
        fig = plt.figure()
        plt.hist(N_R_sepsis, bins=100)
        plt.title("Histogram of complete records for patient with sepsis")
        plt.show()
        plt.close(fig)
        fig = plt.figure()
        plt.hist(N_R_nonsepsis, bins=100)
        plt.title("Histogram of complete records for patient without sepsis")
        plt.show()
        plt.close(fig)
        # Plot histogram of records
        fig = plt.figure()
        plt.hist(N_Rs, bins=100)
        plt.title("Histogram of complete records for the whole dataset")
        plt.show()
        plt.close(fig)

    # generate_target_IDs()

    return ID2N_R, N_R_sepsis, N_R_nonsepsis, MTS_sepsis, MTS_nonsepsis, ID_sepsis_updated, ID_nonsepsis_updated


def analysis_waitingtime_vs_corr(corr_sepsis, corr_nonsepsis, MTS_sepsis, MTS_nonsepsis, verbose=True):
    print("MTS(sepsis): min {}, max {}".format(np.min(MTS_sepsis), np.max(MTS_sepsis)))
    print("MTS(nonsepsis): min {}, max {}".format(np.min(MTS_nonsepsis), np.max(MTS_nonsepsis)))
    sepsis_days = np.arange(5, int(np.floor(np.max(MTS_sepsis)/24)))
    nonsepsis_days = np.arange(5, int(np.floor(np.max(MTS_nonsepsis)/24)))
    mean_list = list()
    std_list = list()
    for day in sepsis_days:
        mean_list.append(np.mean(corr_sepsis[MTS_sepsis < day * 24]))
        std_list.append(np.std(corr_sepsis[MTS_sepsis < day * 24]))
    mean_sepsis = np.array(mean_list)
    std_sepsis = np.array(std_list)
    mean_list = list()
    std_list = list()
    for day in nonsepsis_days:
        mean_list.append(np.mean(corr_nonsepsis[MTS_nonsepsis < day * 24]))
        std_list.append(np.std(corr_nonsepsis[MTS_nonsepsis < day * 24]))
    mean_nonsepsis = np.array(mean_list)
    std_nonsepsis = np.array(std_list)
    if verbose:
        fig = plt.figure()
        plt.plot(sepsis_days, mean_sepsis, 'b')
        plt.plot(sepsis_days, mean_sepsis - std_sepsis, 'b--')
        plt.plot(sepsis_days, mean_sepsis + std_sepsis, 'b--')
        plt.plot(nonsepsis_days, mean_nonsepsis, 'r')
        plt.plot(nonsepsis_days, mean_nonsepsis - std_nonsepsis, 'r--')
        plt.plot(nonsepsis_days, mean_nonsepsis + std_nonsepsis, 'r--')
        plt.xlabel("Day")
        plt.ylabel("Correlation")
        plt.tight_layout()
        plt.savefig("corr.png")
        plt.close(fig)
    return mean_sepsis, std_sepsis, mean_nonsepsis, std_nonsepsis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    save_dir = "../res/"
    folder_name = "kaiser_nonseparable/"

    do_compute_corr = False
    do_coverage_rate = False
    do_plot_hist = False
    do_t_test = False
    do_sign_test = False
    do_z_test = True
    do_combine = False
    do_analysis = True

    # ID_dir = "../data/KAISER/IDs_sampled_seed.pickle".format(args.seed)
    if args.seed is None:
        print("start")
        seed_set = [7]
        # seed_set = [0, 1]
        ID_sepsis_list = []
        ID_nonsepsis_list = []
        for seed in seed_set:
            print(seed)
            ID_dir = "../data/KAISER/ID_samples/IDs_sampled_seed{}_updated.pickle".format(seed)
            with open(ID_dir, "rb") as res:
                ID_sepsis, ID_nonsepsis = pickle.load(res)
            ID_sepsis_list.append(ID_sepsis)
            ID_nonsepsis_list.append(ID_nonsepsis)
        ID_sepsis = np.concatenate(ID_sepsis_list)
        ID_nonsepsis = np.concatenate(ID_nonsepsis_list)
        print("ID_sepsis size: {}".format(ID_sepsis.shape))
        print("ID_nonsepsis size: {}".format(ID_nonsepsis.shape))
    else:
        ID_dir = "../data/KAISER/ID_samples/IDs_sampled_seed{}_updated.pickle".format(args.seed)
        with open(ID_dir, "rb") as res:
            ID_sepsis, ID_nonsepsis = pickle.load(res)
        print("ID_sepsis size: {}".format(ID_sepsis.shape))
        print("ID_nonsepsis size: {}".format(ID_nonsepsis.shape))
    df = pd.read_csv("../data/KAISER/LLNL_HOURLY_LAPS2.CSV")
    corrcoef_dict = dict()

    missing_smoothness_sepsis_list = []
    missing_smoothness_nonsepsis_list = []
    missing_cov_sepsis_list = []
    missing_cov_nonsepsis_list = []

    # Check all results
    for ID in ID_sepsis:
        if not os.path.exists("../res/kaiser_nonseparable/sepsis/ID_{}/pred_smoothness_grid_map.pickle".format(ID)):
            # print("D_{} of sepsis pred_smoothness_grid_map is missing".format(ID))
            missing_smoothness_sepsis_list.append(ID)
        if not os.path.exists("../res/kaiser_nonseparable/sepsis/ID_{}/pred_cov_grid_map.pickle".format(ID)):
            # print("D_{} of sepsis pred_cov_grid_map is missing".format(ID))
            missing_cov_sepsis_list.append(ID)

    for ID in ID_nonsepsis:
        if not os.path.exists("../res/kaiser_nonseparable/nonsepsis/ID_{}/pred_smoothness_grid_map.pickle".format(ID)):
            # print("D_{} of nonsepsis pred_smoothness_grid_map is missing".format(ID))
            missing_smoothness_nonsepsis_list.append(ID)
        if not os.path.exists("../res/kaiser_nonseparable/nonsepsis/ID_{}/pred_cov_grid_map.pickle".format(ID)):
            # print("D_{} of nonsepsis pred_cov_grid_map is missing".format(ID))
            missing_cov_nonsepsis_list.append(ID)

    attributes = ['BPDIA', 'BPSYS', 'HRTRT', 'O2SAT']
    # summarize sepsis

    # import pdb
    # pdb.set_trace()

    if do_compute_corr:
        # sepsis
        sampled_corr_ls = list()
        sampled_corr_Rs = list()
        est_corr_ls = list()
        est_corr_Rs = list()
        num = 0
        for ID in ID_sepsis:
            num += 1
            print("sepsis: ", num)
            # print("ID {}:".format(ID))
            subfolder_name = "ID_{}/".format(ID)
            # load LASP2 data
            hour, laps2 = extract_LAPS.extract_LAPS(ID, df)
            N = len(hour)
            M = len(attributes)
            # load inferred result
            with open("../res/kaiser_nonseparable/sepsis/ID_{}/pred_smoothness_grid_map.pickle".format(ID), "rb") as res:
                sampled_tilde_ls = pickle.load(res)
            with open("../res/kaiser_nonseparable/sepsis/ID_{}/pred_cov_grid_map.pickle".format(ID), "rb") as res:
                sampled_L_fs = pickle.load(res)
            sampled_B_fs = np.stack([np.stack([np.matmul(L_f, L_f.T) for L_f in sampled_L_f]) for sampled_L_f in sampled_L_fs])
            sampled_R_fs = np.stack([np.stack([posterior_analysis.cov2cor(B_f) for B_f in sampled_B_f]) for sampled_B_f in sampled_B_fs])
            
            est_tilde_ls = np.mean(sampled_tilde_ls, axis = 1)
            est_R_fs = np.mean(sampled_R_fs, axis = 1)
            
            n_sample = sampled_tilde_ls.shape[1]
            sampled_corr_l = np.array([np.corrcoef(sampled_tilde_ls[:, s], laps2)[0,1] for s in range(n_sample)])
            sampled_corr_R = np.zeros([n_sample, M, M])
            for s in range(n_sample):
                for i in range(M):
                    for j in range(i+1, M):
                        sampled_corr_R[s ,i, j] = np.corrcoef(sampled_R_fs[:, s, i, j], laps2)[0,1]

            est_corr_l = np.corrcoef(est_tilde_ls, laps2)[0,1]
            est_corr_R = np.zeros([M, M])
            
            for i in range(M):
                for j in range(i+1, M):
                    est_corr_R[i,j] = np.corrcoef(est_R_fs[:,i,j], laps2)[0,1]

            sampled_corr_ls.append(sampled_corr_l)
            sampled_corr_Rs.append(sampled_corr_R)
            est_corr_ls.append(est_corr_l)
            est_corr_Rs.append(est_corr_R)
        sampled_corr_ls_sepsis = np.stack(sampled_corr_ls)
        sampled_corr_Rs_sepsis = np.stack(sampled_corr_Rs)
        est_corr_ls_sepsis = np.stack(est_corr_ls)
        est_corr_Rs_sepsis = np.stack(est_corr_Rs)
        if args.seed is None:
            with open("../res/kaiser_nonseparable/sepsis/corr.pickle", "wb") as res:
                pickle.dump([sampled_corr_ls_sepsis, sampled_corr_Rs_sepsis, est_corr_ls_sepsis, est_corr_Rs_sepsis], res)
        else:
            with open("../res/kaiser_nonseparable/sepsis/corr_seed{}.pickle".format(args.seed), "wb") as res:
                pickle.dump([sampled_corr_ls_sepsis, sampled_corr_Rs_sepsis, est_corr_ls_sepsis, est_corr_Rs_sepsis], res)

        # nonsepsis
        sampled_corr_ls = list()
        sampled_corr_Rs = list()
        est_corr_ls = list()
        est_corr_Rs = list()
        num = 0
        for ID in ID_nonsepsis:
            num += 1
            print("nonsepsis: ", num)
            # print("ID {}:".format(ID))
            subfolder_name = "ID_{}/".format(ID)
            # load LASP2 data
            hour, laps2 = extract_LAPS.extract_LAPS(ID, df)
            N = len(hour)
            M = len(attributes)
            # load inferred result
            with open("../res/kaiser_nonseparable/nonsepsis/ID_{}/pred_smoothness_grid_map.pickle".format(ID), "rb") as res:
                sampled_tilde_ls = pickle.load(res)
            with open("../res/kaiser_nonseparable/nonsepsis/ID_{}/pred_cov_grid_map.pickle".format(ID), "rb") as res:
                sampled_L_fs = pickle.load(res)
            sampled_B_fs = np.stack([np.stack([np.matmul(L_f, L_f.T) for L_f in sampled_L_f]) for sampled_L_f in sampled_L_fs])
            sampled_R_fs = np.stack([np.stack([posterior_analysis.cov2cor(B_f) for B_f in sampled_B_f]) for sampled_B_f in sampled_B_fs])
            
            est_tilde_ls = np.mean(sampled_tilde_ls, axis = 1)
            est_R_fs = np.mean(sampled_R_fs, axis = 1)

            n_sample = sampled_tilde_ls.shape[1]
            sampled_corr_l = np.array([np.corrcoef(sampled_tilde_ls[:, s], laps2)[0,1] for s in range(n_sample)])
            sampled_corr_R = np.zeros([n_sample, M, M])
            for s in range(n_sample):
                for i in range(M):
                    for j in range(i+1, M):
                        sampled_corr_R[s ,i, j] = np.corrcoef(sampled_R_fs[:, s, i, j], laps2)[0,1]

            est_corr_l = np.corrcoef(est_tilde_ls, laps2)[0,1]
            est_corr_R = np.zeros([M, M])
            for i in range(M):
                for j in range(i+1, M):
                    est_corr_R[i,j] = np.corrcoef(est_R_fs[:,i,j], laps2)[0,1]

            sampled_corr_ls.append(sampled_corr_l)
            sampled_corr_Rs.append(sampled_corr_R)
            est_corr_ls.append(est_corr_l)
            est_corr_Rs.append(est_corr_R)
            
        sampled_corr_ls_nonsepsis = np.stack(sampled_corr_ls)
        sampled_corr_Rs_nonsepsis = np.stack(sampled_corr_Rs)
        est_corr_ls_nonsepsis = np.stack(est_corr_ls)
        est_corr_Rs_nonsepsis = np.stack(est_corr_Rs)

        if args.seed is None:
            with open("../res/kaiser_nonseparable/nonsepsis/corr.pickle", "wb") as res:
                pickle.dump([sampled_corr_ls_nonsepsis, sampled_corr_Rs_nonsepsis, est_corr_ls_nonsepsis, est_corr_Rs_nonsepsis], res)
        else:
            with open("../res/kaiser_nonseparable/nonsepsis/corr_seed{}.pickle".format(args.seed), "wb") as res:
                pickle.dump([sampled_corr_ls_nonsepsis, sampled_corr_Rs_nonsepsis, est_corr_ls_nonsepsis, est_corr_Rs_nonsepsis], res)
    else:
        # with open("../res/kaiser_nonseparable/sepsis/corr.pickle", "rb") as res:
        #     sampled_corr_ls_sepsis, sampled_corr_Rs_sepsis, est_corr_ls_sepsis, est_corr_Rs_sepsis = pickle.load(res)
        # with open("../res/kaiser_nonseparable/nonsepsis/corr.pickle", "rb") as res:
        #     sampled_corr_ls_nonsepsis, sampled_corr_Rs_nonsepsis, est_corr_ls_nonsepsis, est_corr_Rs_nonsepsis = pickle.load(res)
        if args.seed is None:
            with open("../res/kaiser_nonseparable/sepsis/corr_cohort2000.pickle".format(args.seed), "rb") as res:
                sampled_corr_ls_sepsis, sampled_corr_Rs_sepsis, est_corr_ls_sepsis, est_corr_Rs_sepsis = pickle.load(res)
            with open("../res/kaiser_nonseparable/nonsepsis/corr_cohort2000.pickle".format(args.seed), "rb") as res:
                sampled_corr_ls_nonsepsis, sampled_corr_Rs_nonsepsis, est_corr_ls_nonsepsis, est_corr_Rs_nonsepsis = pickle.load(res)
        else:
            with open("../res/kaiser_nonseparable/sepsis/corr_seed{}.pickle".format(args.seed), "rb") as res:
                sampled_corr_ls_sepsis, sampled_corr_Rs_sepsis, est_corr_ls_sepsis, est_corr_Rs_sepsis = pickle.load(res)
            with open("../res/kaiser_nonseparable/nonsepsis/corr_seed{}.pickle".format(args.seed), "rb") as res:
                sampled_corr_ls_nonsepsis, sampled_corr_Rs_nonsepsis, est_corr_ls_nonsepsis, est_corr_Rs_nonsepsis = pickle.load(res)

    if do_coverage_rate:
        M = len(attributes)
        coverage_p_smoothness_sepsis = compute_coverage_rate(sampled_corr_ls_sepsis, credible_quantile=0.95)
        print("corr_smoothness_sepsis coverage: {}.".format(coverage_p_smoothness_sepsis))
        coverage_p_smoothness_nonsepsis = compute_coverage_rate(sampled_corr_ls_nonsepsis, credible_quantile=0.95)
        print("corr_smoothness_nonsepsis coverage: {}.".format(coverage_p_smoothness_nonsepsis))
        for i in range(M):
            for j in range(i+1, M):
                coverage = compute_coverage_rate(sampled_corr_Rs_sepsis[:, :, i, j], credible_quantile=0.95)
                print("corr_{}_vs_{}_sepsis coverage: {}.".format(attributes[i], attributes[j], coverage))
                coverage = compute_coverage_rate(sampled_corr_Rs_nonsepsis[:, :, i, j], credible_quantile=0.95)
                print("corr_{}_vs_{}_nonsepsis coverage: {}.".format(attributes[i], attributes[j], coverage))  
    
    if do_plot_hist:
        # plot the histogram of correlation
        M = len(attributes)
        fig = plt.figure()
        plt.hist(est_corr_ls_sepsis, bins = 50)
        plt.savefig("est_corr_ls_sepsis_hist.png")
        plt.close()
        fig = plt.figure()
        plt.hist(est_corr_ls_nonsepsis, bins = 50)
        plt.savefig("est_corr_ls_nonsepsis_hist.png")
        plt.close()
        for i in range(M):
            for j in range(i+1, M):
                fig = plt.figure()
                plt.hist(est_corr_Rs_sepsis[:, i, j], bins = 50)
                plt.savefig("est_corr_{}_vs_{}_sepsis_hist.png".format(attributes[i], attributes[j]))
                plt.close()
                fig = plt.figure()
                plt.hist(est_corr_Rs_nonsepsis[:, i, j], bins = 50)
                plt.savefig("est_corr_{}_vs_{}_nonsepsis_hist.png".format(attributes[i], attributes[j]))
                plt.close()

    if do_t_test:
        M = len(attributes)
        if do_combine:
            est_corr_ls = np.concatenate([est_corr_ls_sepsis, est_corr_ls_nonsepsis])
            # compute t-value and p-value:
            t_value, p_value = t_test(est_corr_ls)
            print("smoothness, t = {}, p = {}".format(t_value, p_value))
            for i in range(M):
                for j in range(i+1, M):
                    est_corr_Rs = np.concatenate([est_corr_Rs_sepsis[:, i, j], est_corr_Rs_nonsepsis[:, i, j]])
                    t_value, p_value = t_test(est_corr_Rs)
                    print("{} vs {}: t = {}, p = {}".format(attributes[i], attributes[j], t_value, p_value)) 
        else:
            # compute t-value and p-value:
            t_value, p_value = t_test(est_corr_ls_sepsis)
            print("smoothness of sepsis, t = {}, p = {}".format(t_value, p_value))
            t_value, p_value = t_test(est_corr_ls_nonsepsis)
            print("smoothness of nonsepsis, t = {}, p = {}".format(t_value, p_value))
            for i in range(M):
                for j in range(i+1, M):
                    t_value, p_value = t_test(est_corr_Rs_sepsis[:, i, j])
                    print("sepsis, {} vs {}: t = {}, p = {}".format(attributes[i], attributes[j], t_value, p_value)) 
                    t_value, p_value = t_test(est_corr_Rs_nonsepsis[:, i, j])
                    print("nonsepsis, {} vs {}: t = {}, p = {}".format(attributes[i], attributes[j], t_value, p_value)) 
                    
    if do_sign_test:
        M = len(attributes)
        if do_combine:
            est_corr_ls = np.concatenate([est_corr_ls_sepsis, est_corr_ls_nonsepsis])
            # compute p-value:
            p_value = sign_test(est_corr_ls)
            print("smoothness, p = {}".format(p_value))
            for i in range(M):
                for j in range(i+1, M):
                    est_corr_Rs = np.concatenate([est_corr_Rs_sepsis[:, i, j], est_corr_Rs_nonsepsis[:, i, j]])
                    p_value = sign_test(est_corr_Rs)
                    print("{} vs {}: p = {}".format(attributes[i], attributes[j], p_value)) 
        else:
            # compute p-value:
            p_value = sign_test(est_corr_ls_sepsis)
            print("smoothness of sepsis, p = {}".format(p_value))
            p_value = sign_test(est_corr_ls_nonsepsis)
            print("smoothness of nonsepsis, p = {}".format(p_value))
            for i in range(M):
                for j in range(i+1, M):
                    p_value = sign_test(est_corr_Rs_sepsis[:, i, j])
                    print("sepsis, {} vs {}: p = {}".format(attributes[i], attributes[j], p_value)) 
                    p_value = sign_test(est_corr_Rs_nonsepsis[:, i, j])
                    print("nonsepsis, {} vs {}: p = {}".format(attributes[i], attributes[j], p_value)) 
                    
    if do_z_test:
        M = len(attributes)
        if do_combine:
            est_corr_ls = np.concatenate([est_corr_ls_sepsis, est_corr_ls_nonsepsis])
            # compute t-value and p-value:
            z_value, p_value = z_test(est_corr_ls)
            print("smoothness, z = {}, p = {}".format(z_value, p_value))
            for i in range(M):
                for j in range(i+1, M):
                    est_corr_Rs = np.concatenate([est_corr_Rs_sepsis[:, i, j], est_corr_Rs_nonsepsis[:, i, j]])
                    z_value, p_value = z_test(est_corr_Rs)
                    print("{} vs {}: z = {}, p = {}".format(attributes[i], attributes[j], z_value, p_value)) 
        else:
            # compute t-value and p-value:
            z_value, p_value = z_test(est_corr_ls_sepsis)
            print("smoothness of sepsis, z = {}, p = {}".format(z_value, p_value))
            z_value, p_value = z_test(est_corr_ls_nonsepsis)
            print("smoothness of nonsepsis, z = {}, p = {}".format(z_value, p_value))
            for i in range(M):
                for j in range(i+1, M):
                    z_value, p_value = z_test(est_corr_Rs_sepsis[:, i, j])
                    print("sepsis, {} vs {}: z = {}, p = {}".format(attributes[i], attributes[j], z_value, p_value)) 
                    z_value, p_value = z_test(est_corr_Rs_nonsepsis[:, i, j])
                    print("nonsepsis, {} vs {}: z = {}, p = {}".format(attributes[i], attributes[j], z_value, p_value))

    np.random.seed(22)
    print(np.random.choice(ID_sepsis, 1))
    print(np.random.choice(ID_nonsepsis, 1))

    if do_analysis:
        ID2N_R, N_R_sepsis, N_R_nonsepsis, MTS_sepsis, MTS_nonsepsis, ID_sepsis_updated, ID_nonsepsis_updated = analysis_datasize()
        # est_corr_ls_sepsis, est_corr_Rs_sepsis
        # est_corr_ls_nonsepsis, est_corr_Rs_nonsepsis
        # print(np.mean(est_corr_ls_sepsis[N_R_sepsis < 100]), np.std(est_corr_ls_sepsis[N_R_sepsis < 100]))
        # print(np.mean(est_corr_ls_sepsis[N_R_sepsis >= 100]), np.std(est_corr_ls_sepsis[N_R_sepsis >= 100]))
        # print(np.mean(est_corr_ls_nonsepsis[N_R_nonsepsis < 100]), np.std(est_corr_ls_nonsepsis[N_R_nonsepsis < 100]))
        # print(np.mean(est_corr_ls_nonsepsis[N_R_nonsepsis >= 100]), np.std(est_corr_ls_nonsepsis[N_R_nonsepsis >= 100]))


        # print(np.mean(est_corr_ls_sepsis[MTS_sepsis < 5*24]), np.std(est_corr_ls_sepsis[MTS_sepsis < 5*24]))
        # print(np.mean(est_corr_ls_sepsis[MTS_sepsis >= 5*24]), np.std(est_corr_ls_sepsis[MTS_sepsis >= 5*24]))
        # print(np.mean(est_corr_ls_nonsepsis[MTS_nonsepsis < 5*24]), np.std(est_corr_ls_nonsepsis[MTS_nonsepsis < 5*24]))
        # print(np.mean(est_corr_ls_nonsepsis[MTS_nonsepsis >= 5*24]), np.std(est_corr_ls_nonsepsis[MTS_nonsepsis >= 5*24]))

        # res_l = analysis_waitingtime_vs_corr(est_corr_ls_sepsis, est_corr_ls_nonsepsis, MTS_sepsis, MTS_nonsepsis)
        # i = 0; j = 1
        # res_corr = analysis_waitingtime_vs_corr(est_corr_Rs_sepsis[:, i, j], est_corr_Rs_nonsepsis[:, i, j], MTS_sepsis, MTS_nonsepsis)

        # select a patient
        temp_IDs = ID_sepsis_updated[np.argsort(est_corr_ls_sepsis)]
        target_ID = temp_IDs[np.argmax(est_corr_Rs_sepsis[:, 0, 1][np.argsort(est_corr_ls_sepsis)[:10]])]

        import pdb; pdb.set_trace()


    import pdb; pdb.set_trace()
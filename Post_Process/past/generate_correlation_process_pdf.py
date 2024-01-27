import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
matplotlib.rcParams.update({'font.size': 11})
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'figure.autolayout': True})
from scipy import stats


import extract_LAPS
import posterior_analysis
import utils

if __name__ == "__main__":
    save_dir = "../res/"
    folder_name = "kaiser_nonseparable_distributed/"
    do_summary = False 
    do_all = False 
    # Load ID_dict
    with open("../data/IDs.pickle", "rb") as res:
        ID_dict = pickle.load(res)	
    # import pdb
    # pdb.set_trace()
    ID2index = {ID: index for index, ID in enumerate(ID_dict)} 
    with open("../data/kaiser_distributed.pickle", "rb") as res:
        data = pickle.load(res)
    df = pd.read_csv("../data/LLNL_HOURLY_LAPS2.CSV")
    corrcoef_dict = dict()
    pdf = matplotlib.backends.backend_pdf.PdfPages('correlation_coefficient_histogram.pdf')
    # ID_dict = [42935736]
    if do_summary:
        for ID in ID_dict:
            print("ID_{}".format(ID))
            subfolder_name = "ID_{}/".format(ID)
            rank = ID2index[ID]
            origx, origy, attributes = data[rank]
            N, M = origy.shape
            hour, laps2 = extract_LAPS.extract_LAPS(ID, df)
            # load prediction results

            if do_all:
                with open(save_dir + folder_name + subfolder_name + "pred_grid_map.pickle", "rb") as res:
                    grid_quantiles, grid_L_vecs = pickle.load(res)
                grid_L_vecs = grid_L_vecs.data.numpy()
                L_vec_list = [L_vec for L_vec in grid_L_vecs]
                L_f_list = np.stack([utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list])
                B_fs = np.stack([np.matmul(L_f, L_f.T) for L_f in L_f_list])
                R_fs = np.stack([posterior_analysis.cov2cor(np.matmul(L_f, L_f.T)) for L_f in L_f_list]) # size N, M, M
                for i in range(M):
                    for j in range(i+1, M):
                        corrcoef = np.corrcoef(R_fs[:, i, j], laps2)[0,1]
                        # import pdb
                        # pdb.set_trace()
                        corr_name = attributes[i] + "_" + attributes[j]
                        if not (corr_name in corrcoef_dict):
                            corrcoef_dict[corr_name] = list()
                        corrcoef_dict[corr_name].append(corrcoef)
            else:
                try:
                    with open(save_dir + folder_name + subfolder_name + "pred_grid_map.pickle", "rb") as res:
                       grid_quantiles, grid_L_vecs = pickle.load(res)
                    # import pdb
                    # pdb.set_trace()
                    grid_L_vecs = grid_L_vecs.data.numpy()
                    L_vec_list = [L_vec for L_vec in grid_L_vecs]
                    L_f_list = np.stack([utils.vec2lowtriangle(L_vec, M) for L_vec in L_vec_list])
                    B_fs = np.stack([np.matmul(L_f, L_f.T) for L_f in L_f_list])
                    R_fs = np.stack([posterior_analysis.cov2cor(np.matmul(L_f, L_f.T)) for L_f in L_f_list]) # size N, M, M
                    for i in range(M):
                        for j in range(i+1, M):
                            corrcoef = np.corrcoef(R_fs[:, i, j], laps2)[0,1]
                            corr_name = attributes[i] + "_" + attributes[j]
                            if not (corr_name in corrcoef_dict):
                                corrcoef_dict[corr_name] = list()
                            corrcoef_dict[corr_name].append(corrcoef) 
                except:
                    print("ID {} is not available".format(ID))
        for corr_name, corr_values in corrcoef_dict.items():
            corr_values = np.array(corr_values)
            print(corr_name, corr_values)
            print(corr_name, np.mean(corr_values), np.std(corr_values))
        with open(save_dir + folder_name + "corrcoef.pickle", "wb") as res:
            pickle.dump(corrcoef_dict, res)
    else:
        with open(save_dir + folder_name + "corrcoef.pickle", "rb") as res:
            corrcoef_dict = pickle.load(res)
        T_statistics = []
        p_values = []
        for corr_name, corr_values in corrcoef_dict.items(): 
            corr_values = np.array(corr_values)
            mean = np.mean(corr_values)
            std = np.std(corr_values)
            t = (mean - 0)/std*np.sqrt(corr_values.shape[0])
            p = 2 * stats.t.sf(np.abs(t), corr_values.shape[0]-1)
            # Two side hypho
            print("{} vs LAPS: number of data {}, mean value {}, std {}, t value {}, p value {}.".format(corr_name, len(corr_values) , mean, std, t, p))
            fig = plt.figure()
            plt.hist(corr_values, bins = 50)
            plt.title(corr_name)
            plt.xlabel("correlation coefficient")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.clf()
    pdf.close()

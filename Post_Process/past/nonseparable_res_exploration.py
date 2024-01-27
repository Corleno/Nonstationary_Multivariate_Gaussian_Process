"""
Nonseparable model exploration
"""
import pickle

# import private libraries
import posterior_analysis


def private_explore(ID, save_dir=None, folder_name=None):
    subfolder_name = "ID_{}/".format(ID)
    # load private data
    rank = ID2index[ID]
    origx, origY, attributes = data[rank]
    N, M = origY.shape
    # load private MAP result
    with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
        estPars = pickle.load(res)
    est_tilde_l, est_L_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
    posterior_analysis.visualization_pos_map_heatmap(origx, L_vecs=est_L_vecs, N=N, M=M, save_dir=save_dir,
                                        folder_name=folder_name, subfolder_name=subfolder_name, attributes=attributes)



if __name__ == "__main__":
    save_dir = "../res/"
    folder_name = "kaiser_nonseparable_distributed/"
    # load ID_dict
    with open("../data/IDs_small.pickle", "rb") as res:
        ID_dict = pickle.load(res)
    ID2index = {ID: index for index, ID in enumerate(ID_dict)}
    # load raw data
    with open("../data/kaiser_distributed_small.pickle", "rb") as res:
        data = pickle.load(res)
    n = 0
    ID_targets = [13296382, 41168468, 12986958, 12978238]
    N = len(ID_targets)
    for ID in ID_targets:
        n = n + 1
        print("{}/{} complete".format(n, N))
        private_explore(ID, save_dir=save_dir, folder_name=folder_name)


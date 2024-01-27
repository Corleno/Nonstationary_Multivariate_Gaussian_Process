import numpy as np
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, default="stationary")
    parser.add_argument('--model2', type=str, default="nonseparable")
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()
    save_dir = "../res/"
    folder_name_dict = {"stationary_extrapolation": "kaiser_stationary_distributed_extrapolation0/", "separable_extrapolation": "kaiser_separable_distributed_extrapolation0/", "nonseparable_extrapolation": "kaiser_nonseparable_distributed_extrapolation0/","stationary": "kaiser_stationary_distributed/", "separable": "kaiser_separable_distributed/", "nonseparable": "kaiser_nonseparable_distributed/"}
    folder_name1 = folder_name_dict[args.model1]
    folder_name2 = folder_name_dict[args.model2]	
    with open (save_dir + folder_name1 + "valid_IDs.pickle", "rb") as res:
        IDs1 = pickle.load(res)
    with open (save_dir + folder_name2 + "valid_IDs.pickle", "rb") as res:
        IDs2 = pickle.load(res)
    IDs = list(set(IDs1) & set(IDs2))
    scores = []
    for ID in IDs:
        with open(save_dir + folder_name1 + "ID_{}/freq_res.pickle".format(ID), "rb") as res:
            rmse1, _ = pickle.load(res)
        with open(save_dir + folder_name2 + "ID_{}/freq_res.pickle".format(ID), "rb") as res:
            rmse2, _ = pickle.load(res)
        score = (rmse1 - rmse2)/rmse1
        scores.append(score)
    scores = np.array(scores)
    order = np.argsort(scores)
    ordered_IDs = np.array(IDs)[order]
    ordered_scores = scores[order]
    best_IDs = ordered_IDs[-args.n:]
    best_scores = ordered_scores[-args.n:]
    print(best_IDs, best_scores)
    print(np.sum(scores>=0), scores.shape[0])
    print(np.sum(scores>=0).astype(np.float)/scores.shape[0]) 

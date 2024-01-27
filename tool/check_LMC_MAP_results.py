import pickle
import os

if __name__ == "__main__":
    # load IDs
    ID_dir = "../data/KAISER/ID_samples/IDs_sampled_seed7_updated.pickle"
    with open(ID_dir, "rb") as res:
        ID_sepsis, ID_nonsepsis = pickle.load(res)
    # check results
    save_dir = "../res/"
    group = "sepsis"
    folder_name = "kaiser_stationary/" + group + "/"
    for ID in ID_sepsis:
        subfolder_name = subfolder_name = "ID_{}/".format(ID)
        if not os.path.exists(save_dir + folder_name + subfolder_name + "MAP.dat"):
            print("sepsis: ID_{} does not exits.".format(ID))
    group = "nonsepsis"
    folder_name = "kaiser_stationary/" + group + "/"
    for ID in ID_nonsepsis:
        subfolder_name = subfolder_name = "ID_{}/".format(ID)
        if not os.path.exists(save_dir + folder_name + subfolder_name + "MAP.dat"):
            print("nonsepsis: ID_{} does not exits.".format(ID))

import numpy as np 
import pickle
import matplotlib.pyplot as plt

import statsmodels.api as sm
#import statsmodels.graphics.functional.fboxplot as fboxplot

import sys
sys.path.append("..")
from Utility import posterior_analysis
from Utility import utils

from matplotlib import rc 
import matplotlib.font_manager as font_manager
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
legend_properties = dict(weight='bold', size=16)

from mpl_toolkits.axes_grid.inset_locator import inset_axes

def plot_mean_and_CI_true(x, mean, lb, ub, true_y, color_mean=None, color_shading=None, color_true=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(x, ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(x, mean, color_mean)
    plt.plot(x, true_y, color_true)

def load_syndata(directory="../data/sim/sim_MNTS.pickle"):
    # Load synthetic data
    with open(directory, "rb") as res:
        x, true_l, true_L_vecs, true_sigma2_err, Y = pickle.load(res)
    return x, true_l, true_L_vecs, true_sigma2_err, Y

def summary_eval(folder_name):
    Gs = list()
    Ps = list()
    Ds = list()
    for i in range(100):
        if folder_name == "sim_stationary/":
            with open(save_dir + folder_name + "{}/".format(i) + "model_evaluation_map.pickle", "rb") as res:
                repy_mean, repy_std, G, P, D = pickle.load(res)
        else:
            with open(save_dir + folder_name + "{}/".format(i) + "model_evaluation_map.pickle", "rb") as res:
                repy_quantiles, repy_mean, repy_std, G, P, D = pickle.load(res)
        Gs.append(G)
        Ps.append(P)
        Ds.append(D)
    Gs = np.array(Gs)
    Ps = np.array(Ps)
    Ds = np.array(Ds)
    fig, ax = plt.subplots(nrows = 1, ncols=3)
    ax[0].set_title('Gs')
    ax[0].boxplot(Gs, sym='')
    ax[1].set_title('Ps')
    ax[1].boxplot(Ps, sym='')
    ax[2].set_title('Ds')
    ax[2].boxplot(Ds, sym='')
    fig.savefig("measure_eval_{}.png".format(folder_name[:-1]))
    # print(Gs)
    print(np.mean(Gs), np.std(Gs))
    # print(Ps)
    print(np.mean(Ps), np.std(Ps))
    # print(Ds)
    print(np.mean(Ds), np.std(Ds))
    return Gs, Ps, Ds

def summary_pred(folder_name):
    PMSEs = list()
    for i in range(100):
        if folder_name == "sim_stationary/":
            with open(save_dir + folder_name + "{}/".format(i) + "pred_test_map.pickle", "rb") as res:
                repy_mean, repy_std, PMSE = pickle.load(res)
        else:
            with open(save_dir + folder_name + "{}/".format(i) + "pred_test_map.pickle", "rb") as res:
                repy_mean, repy_std, PMSE = pickle.load(res)
        PMSEs.append(PMSE)
    PMSEs = np.array(PMSEs)
    fig = plt.figure()
    plt.boxplot(PMSEs)
    fig.savefig("measure_{}.png".format(folder_name[:-1]))
    print("PMSE: ", np.mean(PMSEs), np.std(PMSEs))
    return PMSEs

def summary_latent_processes(folder_name):
    est_tilde_l_list = list()
    est_stds_list = list()
    est_R01s_list = list()
    for i in range(100):
        # Load MAP results
        subfolder_name = "{}/".format(i)
        with open(save_dir + folder_name + subfolder_name + "MAP.dat", "rb") as res:
            estPars = pickle.load(res)
        est_tilde_l, est_uL_vecs, est_tilde_sigma2_err = posterior_analysis.vec2pars_est_SVC(estPars, N)
        est_L_vecs = utils.uLvecs2Lvecs(est_uL_vecs, N, M)
        est_L_vec_list = [est_L_vecs[n * int(M * (M + 1) / 2):(n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
        est_L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in est_L_vec_list]
        est_B_list = [np.matmul(L_f, L_f.T) for L_f in est_L_f_list]
        est_R_list = [posterior_analysis.cov2cor(B) for B in est_B_list]
        est_R_01 = [R[0,1] for R in est_R_list]
        est_stds = np.stack([np.sqrt(np.diag(B)) for B in est_B_list])

        est_tilde_l_list.append(est_tilde_l)
        est_stds_list.append(est_stds)
        est_R01s_list.append(np.stack(est_R_01))
    est_tilde_ls = np.stack(est_tilde_l_list)
    est_stds = np.stack(est_stds_list)
    est_R01s = np.stack(est_R01s_list)
    #print(est_tilde_ls)
    #print(x)

    # plot estimated log length-scale process
    fig = plt.figure()
    # lb, ub = np.percentile(est_tilde_ls, q=(2.5, 97.5), axis = 0)
    # plot_mean_and_CI_true(x, np.mean(est_tilde_ls, axis = 0), lb = lb, ub = ub, true_y=true_tilde_l, color_mean='k', color_shading='k', color_true='r')
    # plt.plot(x, est_tilde_ls.T)
    res = sm.graphics.fboxplot(data=est_tilde_ls, xdata=x, wfactor = 10)
    plt.plot(x, true_tilde_l, label=r"$\textbf{true log lengthscale process}$", color='r', linestyle="--")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="upper right", prop=legend_properties)
    plt.tight_layout()
    plt.savefig("simlog_l.png")
    plt.close(fig)    

    for m in range(M):
        fig = plt.figure()
        # lb, ub = np.percentile(est_stds[:, :, m], q = (2.5,97.5), axis = 0)
        # plot_mean_and_CI_true(x, np.mean(est_stds[:,:,m], axis=0), lb=lb, ub=ub, true_y=true_stds[:, m], color_mean='k', color_shading='k', color_true='r')
        res = sm.graphics.fboxplot(data=est_stds[:, :, m], xdata=x, wfactor = 10)
        # plt.plot(x, true_stds[:, m], label="true standard deviation process on dimension {}".format(m+1), color='r', linestyle="--")
        # plt.plot(x, true_stds[:, m], label=r"$\textbf{true standard deviation process on dimension} {{}}$".format(m+1), color='r', linestyle="--"))
        plt.plot(x, true_stds[:, m], label=r"$\textbf{{true standard deviation process on dimension}} \ {}$".format(m+1), color='r', linestyle="--")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(loc="upper right", prop=legend_properties)
        plt.tight_layout()
        plt.savefig("sim_stds_dim{}.png".format(m))
        plt.close(fig)


    fig = plt.figure()
    # lb, ub = np.percentile(est_R01s, q=(2.5, 97.5), axis = 0)
    # plot_mean_and_CI_true(x, np.mean(est_R01s, axis = 0), lb = lb, ub = ub, true_y=true_R_01, color_mean='k', color_shading='k', color_true='r')
    res = sm.graphics.fboxplot(data=est_R01s, xdata=x, wfactor = 10)
    plt.plot(x, true_R_01, label=r"$\textbf{true correlation process}$", color='r', linestyle="--")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="upper right", prop=legend_properties)
    plt.tight_layout()
    plt.savefig("simR.png")
    plt.close(fig)    
    # import pdb
    # pdb.set_trace()

        # plot estimated log length-scale process
        # fig = plt.figure()
        # plt.plot(x.numpy(), est_tilde_l)
        # plt.savefig(save_dir + folder_name + subfolder_name + "est_log_l.png")
        # plt.close(fig)
        
        # plot estimated std process
        # fig = plt.figure()
        # for m in range(M):
        #     plt.plot(x.numpy(), est_stds[:, m], label="Dim {}".format(m+1))
        # plt.legend()
        # plt.savefig(save_dir + folder_name + subfolder_name + "est_std.png")
        # plt.close((fig))
        
        # plot correlation process
        # for i in range(M):
        #     for j in range(i+1, M):
        #         fig = plt.figure()
        #         R_ij = np.stack([R_f[i, j] for R_f in est_R_list])
        #         plt.plot(x.numpy(), R_ij)
        #         plt.savefig(save_dir + folder_name + subfolder_name + "est_log_R_{}{}.png".format(i, j))
        #         plt.close(fig)


if __name__ == "__main__":
    # save_dir = "../res/sim_recovery_empirical/"
    # save_dir = "../res/sim100_strong_train_test/"
    save_dir = "../res/sim100_strong_full/"
    # save_dir = "../res/sim100_S/"
    # save_dir = "../res/"
    folder_name_stationary = "sim_stationary/"
    folder_name_separable = "sim_separable/"
    folder_name_nonseparable = "sim_nonseparable/"
    do_GPD = False
    do_lp = True
    do_GPD_S = False
    
    x, true_l, true_L_vecs, true_sigma2_err, Y = load_syndata(directory= save_dir + "simulation/sim_MNTS.pickle")
    # x, true_l, true_L_vecs, true_sigma2_err, Y = load_syndata(directory= save_dir + "simulation/sim_MNTS_S.pickle")
    

    true_tilde_l = np.log(true_l)
    N, M = Y.shape
    true_L_vec_list = [true_L_vecs[n * int(M * (M + 1) / 2):(n + 1) * int(M * (M + 1) / 2)] for n in range(N)]
    true_L_f_list = [utils.vec2lowtriangle(L_vec, M) for L_vec in true_L_vec_list]
    true_B_list = [np.matmul(L_f, L_f.T) for L_f in true_L_f_list]
    true_R_list = [posterior_analysis.cov2cor(B) for B in true_B_list]
    true_stds = np.stack([np.sqrt(np.diag(B)) for B in true_B_list])
    true_R_01 = np.cos(x * np.pi)

    if do_GPD:
        Gs_s, Ps_s, Ds_s = summary_eval(folder_name_stationary)
        Gs_se, Ps_se, Ds_se = summary_eval(folder_name_separable)
        Gs_nse, Ps_nse, Ds_nse = summary_eval(folder_name_nonseparable)
    
        Gs = [Gs_s, Gs_se, Gs_nse]
        Ps = [Ps_s, Ps_se, Ps_nse]
        Ds = [Ds_s, Ds_se, Ds_nse]

        fig, ax= plt.subplots()
        plt.boxplot(Gs, labels=['LMC', 'SNMGP', 'GNMGP'], sym='')
        fig.savefig("measure_G_eval.png")
        fig, ax = plt.subplots()
        plt.boxplot([Gs_se, Gs_nse], labels=['SNMGP', 'GNMGP'], sym='')
        fig.savefig("measure_G_eval0.png")
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.boxplot(Gs, labels=['LMC', 'SNMGP', 'GNMGP'], sym='')
        inset_axes(ax, width="50%", height="50%", loc = 1)
        plt.boxplot([Gs_se, Gs_nse], labels=['SNMGP', 'GNMGP'], sym='')
        fig.savefig("measure_G_eval1.png")
        fig, ax = plt.subplots()
        plt.boxplot(Ps, labels=['LMC', 'SNMGP', 'GNMGP'], sym='')
        fig.savefig("measure_P_eval.png")
        fig, ax = plt.subplots()
        plt.boxplot([Ps_se, Ps_nse], labels=['SNMGP', 'GNMGP'], sym='')
        fig.savefig("measure_P_eval0.png")
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.boxplot(Ps, labels=['LMC', 'SNMGP', 'GNMGP'], sym='')
        inset_axes(ax, width="50%", height="50%", loc = 1)
        plt.boxplot([Ps_se, Ps_nse], labels=['SNMGP', 'GNMGP'], sym='')
        fig.savefig("measure_P_eval1.png")
        fig, ax = plt.subplots()
        plt.boxplot(Ds, labels=['LMC', 'SNMGP', 'GNMGP'], sym='')
        fig.savefig("measure_D_eval.png")
        fig, ax = plt.subplots()
        plt.boxplot([Ds_se, Ds_nse], labels=['SNMGP', 'GNMGP'], sym='')
        fig.savefig("measure_D_eval0.png")
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.boxplot(Ds, labels=['LMC', 'SNMGP', 'GNMGP'], sym='')
        inset_axes(ax, width="50%", height="50%", loc = 1)
        plt.boxplot([Ds_se, Ds_nse], labels=['SNMGP', 'GNMGP'], sym='')
        fig.savefig("measure_D_eval1.png")

        PMSEs_s = summary_pred(folder_name_stationary)
        PMSEs_se = summary_pred(folder_name_separable)
        PMSEs_nse = summary_pred(folder_name_nonseparable)
    
        PMSEs = [PMSEs_s, PMSEs_se, PMSEs_nse]

        fig = plt.figure()
        plt.boxplot(PMSEs, labels=['LMC', 'SNMGP', 'GNMGP'], sym='')
        fig.savefig("measure_PMSE.png")

        import pdb
        pdb.set_trace()
    
    if do_GPD_S:
        Gs_s, Ps_s, Ds_s = summary_eval(folder_name_stationary)
        Gs_se, Ps_se, Ds_se = summary_eval(folder_name_separable)

        Gs = [Gs_s, Gs_se]
        Ps = [Ps_s, Ps_se]
        Ds = [Ds_s, Ds_se]

        fig, ax = plt.subplots()
        plt.boxplot(Gs, labels=['LMC', 'SNMGP'], sym='')
        fig.savefig("measure_G_eval.png")
        fig, ax = plt.subplots()
        plt.boxplot(Ps, labels=['LMC', 'SNMGP'], sym='')
        fig.savefig("measure_P_eval.png")
        fig, ax = plt.subplots()
        plt.boxplot(Ds, labels=['LMC', 'SNMGP'], sym='')
        fig.savefig("measure_D_eval.png")

        PMSEs_s = summary_pred(folder_name_stationary)
        PMSEs_se = summary_pred(folder_name_separable)

        PMSEs = [PMSEs_s, PMSEs_se]

        fig = plt.figure()
        plt.boxplot(PMSEs, labels=['LMC', 'SNMGP'], sym='')
        fig.savefig("measure_PMSE.png")
    
    if do_lp:
        summary_latent_processes(folder_name_nonseparable)
    
    import pdb
    pdb.set_trace()
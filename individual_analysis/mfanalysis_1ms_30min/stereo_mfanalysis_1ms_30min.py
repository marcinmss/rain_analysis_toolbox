import numpy as np
import os
import subprocess as subprocess
import shutil as shutil

from lab_mf_toolbox.Additional_tools_v2 import *
from lab_mf_toolbox.Multifractal_tools_box_Python_HMCo_ENPC_v_0_93 import *
from lab_mf_toolbox.Tools_data_base_use_v3 import *

from stereo.read_write import stereo_read_from_pickle
from pathlib import Path
from matplotlib import pyplot as plt
from collections import namedtuple
from multifractal_analysis.general import closest_smaller_power_of_2
from numpy import load, ndarray

MFAnalysis = namedtuple("MFAnalysis", ["df", "sa", "tm", "dtm"])

path_data_direct_field = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/stereo_ensemble_direct_field_1ms_30min.npy"
)
path_data_fluctuations = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/stereo_ensemble_fluctuations_1ms_30min.npy"
)
OUTPUTDIRECTFIELD = Path(__file__).parent / "direct_field/"
OUTPUTFLUCTUATIONS = Path(__file__).parent / "fluctuations/"


# Define the scalling regime
minimum_resolution = 0.001
size_array = closest_smaller_power_of_2(int(30 * 60 / minimum_resolution))


def main(data: ndarray, output_folder: Path | str):
    os.chdir(output_folder)
    ########
    ####UM Analysis
    ########
    N = size_array
    data_name = "Scale"

    print("\n")
    print("UM analysis ", data_name)
    N_sample = N

    data_file_name = ""
    dim = 1
    l_range = [(1, 2**7), (2**7, 2**13), (2**13, N)]
    file_index = 0
    file_name = "DF_test_truc.npy"
    plot_index = 1
    plt.figure(150)
    D1, D2, D3 = fractal_dimension(
        data, data_file_name, dim, l_range, file_index, file_name, plot_index
    )
    plt.close()
    print("Fractal dimension")
    print("D1,D2,D3", D1, D2, D3)

    ##TM analysis
    dim = 1
    file_index = 0
    q_values = np.array([-1])
    file_name = "TM_test_truc.npy"
    plot_index = 5
    Kq_1, Kq_2, Kq_3, r2_1, r2_2, r2_3 = TM(
        data, q_values, data_file_name, dim, l_range, file_index, file_name, plot_index
    )
    plt.figure(plot_index)
    # imgTM1 = (
    #     path_output
    #     + dw
    #     + "_"
    #     + str_evt
    #     + "_reso"
    #     + str(reso)
    #     + "_"
    #     + data_name
    #     + "_N_sample_"
    #     + str(N_sample)
    #     + scaling_regime
    #     + "_"
    #     + dof
    #     + "_TM_1.png"
    # )
    # if img_sav == 1:
    #     plt.title("TM " + data_name)
    #     plt.savefig(imgTM1, bbox_inches="tight", dpi=200)
    plt.close()
    plt.figure(plot_index + 1)
    plt.close()

    q_values = np.concatenate(
        (
            np.arange(0.05, 1, 0.05),
            np.array([1.01]),
            np.array([1.05]),
            np.arange(1.1, 3.1, 0.1),
        ),
        axis=0,
    )

    ##first scaling regime
    R2_TM = r2_1[25]
    C1_TM = (Kq_1[21] - Kq_1[17]) / 0.2
    alpha_TM = (Kq_1[21] + Kq_1[17]) / (0.1 * 0.1 * C1_TM)
    # q_values at 1.1 and 0.1
    print("Trace moment analysis")
    print("alpha = ", alpha_TM)
    print("C1 = ", C1_TM)
    print("R2 for q=1.5  = ", R2_TM)

    ##second scaling regime
    R2_TM_2 = r2_2[25]
    C1_TM_2 = (Kq_2[21] - Kq_2[17]) / 0.2
    alpha_TM_2 = (Kq_2[21] + Kq_2[17]) / (0.1 * 0.1 * C1_TM_2)

    print("-Second scaling regime-")
    print("alpha_2 = ", alpha_TM_2)
    print("C1_2 = ", C1_TM_2)
    print("R2_2 for q=1.5  = ", R2_TM_2)

    #######
    ## Step 3 : Double Trace moment analysis
    #######

    DTM_index = 1
    q_values = np.array([1.5])
    plot_index = 30

    plt.figure(plot_index + 3)
    UM_par_1, UM_par_2, UM_par_3 = DTM(
        data,
        q_values,
        data_file_name,
        dim,
        l_range,
        DTM_index,
        file_index,
        file_name,
        plot_index,
    )

    # first scaling regime
    alpha_DTM = UM_par_1[0]
    C1_DTM = UM_par_1[1]

    ##second scaling regime
    alpha_DTM_2 = UM_par_2[0]
    C1_DTM_2 = UM_par_2[1]

    print("Double trace moment analysis" + "_" + data_name)
    print("alpha = ", alpha_DTM)
    print("C1 = ", C1_DTM)

    print("-Second scaling regime-")
    print("alpha_DTM_2 = ", alpha_DTM_2)
    print("C1_DTM_2 = ", C1_DTM_2)

    # Compring empirical and theoretical curves for K(q)
    # Choice of the array with the values of q.
    q_values = np.concatenate(
        (
            np.arange(0.05, 1, 0.05),
            np.array([1.01]),
            np.array([1.05]),
            np.arange(1.1, 3.1, 0.1),
        ),
        axis=0,
    )
    # Computation of the K(q) for theoretical formula and UM parameter estimates
    Kq_TM = K_q(q_values, alpha_TM, C1_TM, 0)
    Kq_DTM = K_q(q_values, alpha_DTM, C1_DTM, 0)

    # second scaling regime
    Kq_TM_2 = K_q(q_values, alpha_TM_2, C1_TM_2, 0)
    Kq_DTM_2 = K_q(q_values, alpha_DTM_2, C1_DTM_2, 0)

    plt.figure(9)
    plt.plot(q_values, Kq_1, color="k", label="Empirical")
    # plt.plot(q_values,Kq_TM,color='g',lw='2', ls='--', label = 'Theoretical with UM par. via TM')
    plt.plot(
        q_values,
        Kq_DTM,
        color="b",
        lw="2",
        ls="--",
        label="Theoretical with UM par. via DTM first scale",
    )
    plt.plot(
        q_values,
        Kq_DTM_2,
        color="r",
        lw="2",
        ls="--",
        label="Theoretical with UM par. via DTM second scale",
    )
    plt.legend(loc="upper right", frameon=False)
    plt.xlabel(r"$q$", fontsize=20, color="k")
    plt.ylabel(r"$K(q)$", fontsize=20, color="k")
    plt.close()

    #######
    ## Step 4 : Spectral analysis
    #######

    plot_index = 15
    plt.figure(250)
    k_range = [(1, 2**7), (2**7, 2**13), (2**13, np.intp(N / 2) - 1)]
    bet1, bet2, bet3, x_spec, y_spec = spectral_analysis(
        data, data_file_name, k_range, dim, plot_index
    )

    # H with fluctuations
    k_2 = K_q(np.array([2]), alpha_DTM, C1_DTM, 0)
    H_1 = 0.5 * (bet1 - 1 + k_2)
    print("H_1 for 1 to N", H_1)
    # second scaling regime
    H_2 = 0.5 * (bet2 - 1 + k_2)

    print("Spectral analysis" + "_" + data_name)
    print("bet1,bet2,bet3", bet1, bet2, bet3)
    plt.close("all")


if __name__ == "__main__":
    print("Running analysis for direct fields")
    data = load(path_data_direct_field)
    main(data, OUTPUTDIRECTFIELD)
    del data
    print("    done.")

    print("Running analysis for direct fields")
    data = load(path_data_fluctuations)
    main(data, OUTPUTFLUCTUATIONS)
    del data
    print("    done.")

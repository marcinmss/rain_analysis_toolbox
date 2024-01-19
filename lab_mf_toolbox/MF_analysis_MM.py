from parsivel.read_write import parsivel_read_from_pickle
from multifractal_analysis.cascade_simulations import discreat_um_sym


import math
import numpy as np

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os as os
import subprocess as subprocess
import shutil as shutil

# from lab_mf_toolbox.Tools_data_base_use_v3 import *
from lab_mf_toolbox.Additional_tools_v2 import *
from lab_mf_toolbox.Multifractal_tools_box_Python_HMCo_ENPC_v_0_93 import *
from lab_mf_toolbox.Tools_data_base_use_v3 import *

pars_event = parsivel_read_from_pickle(
    "/home/marcio/stage_project/data/saved_events/pasivel01.obj"
)


##function for finding nearest power of 2
def floor_log(num, base):
    if num < 0:
        raise ValueError("Non-negative number only.")
    if num == 0:
        return 0
    return base ** int(math.log(num, base))


##loading data (change as you require)
# R_data_all=exporting_R(start_evt,end_evt,disdro_name,path_outputs,path_daily_data_python) # The R series is also given as output of the function

# n = 13
# alpha = 1.08
# c1 = 0.18
# R_data_all = discreat_um_sym(n, alpha, c1)
R_data_all = pars_event.files_rain_rate
# R_data_all = fluctuations(R_data_all, 1)


N_org = len(R_data_all)
edata = R_data_all
data_name = "first"
N = floor_log(len(edata), 2)

edata = np.nan_to_num(R_data_all)

# N fitting with data - N with max cumulative value
##script that reshapes data to power of 2 based on maximum cumulative rainfall rate
E_minus_N = len(edata) - N
if E_minus_N == 0:
    data = edata[0:N,].reshape((N, 1))
else:
    N_trial = []
    for EmN in range(E_minus_N):
        N_trial.append(np.sum(edata[EmN : N + EmN]))
    fit_s = N_trial.index(max(N_trial))
    data = edata[fit_s : N + fit_s,].reshape((N, 1))


###changing nan (not a number) to zero
data = np.nan_to_num(data)

##Normailizing data
data = data / np.nanmean(data)


########
####UM Analysis
########

print("\n")
print("UM analysis ", data_name)
N_sample = N

data_file_name = ""
dim = 1
l_range = [(1, N)]
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
k_range = []
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

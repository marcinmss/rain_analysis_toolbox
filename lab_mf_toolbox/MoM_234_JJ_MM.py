import math
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import os as os
from scipy.stats import uniform
import subprocess as subprocess
import shutil as shutil
from datetime import datetime, timedelta, date
from time import sleep
from numpy import genfromtxt

from Tools_data_base_use_v3 import *
from Multifractal_tools_box_Python_HMCo_ENPC_v_0_93 import *
from lab_mf_toolbox.Additional_tools_v2 import *
from lab_mf_toolbox.Tools_data_base_use_v3 import *

plt.rcParams.update({"figure.max_open_warning": 0})


###flatten_array_of_arrays_start
def flatten(lst):
    for v in lst:
        if isinstance(v, list):
            yield from flatten(v)
        else:
            yield v


###flatten_array_of_arrays_end


def mass_w(t, p, rh):
    ##t in K
    ##p in Pa
    ##rh in %

    Univ_gas_const = 8.3  # J/Kmol
    Mol_wt_water = 18  # g/mol
    Vol = 1  # m3
    Univ_gas_const = 8.3145  # J/K/mol
    # T = 273 + t (loaded data already in K)
    T_c = abs(t - 273)
    # P = p*100 # (loaded pressure already in Pa)
    # saturated vapour pressure from Magnus formula
    P_w = 610.94 * np.exp((17.625 * T_c) / (T_c + 243.04))

    RH_w = rh / 100
    T_K = t  ##temperature in Kelvin
    ##mvs(mass of ater vapour in air unsaturated)
    mvs = Mol_wt_water * P_w * Vol / (Univ_gas_const * T_K)
    ##mv (mass of water vapour unsaturated)
    mass_w = RH_w * mvs
    return mass_w


start_list = ["2022/04/02"]
end_list = ["2022/04/02"]


img_sav = 1  # 0

loc = 1
result = []

disdro_name_list = ["Pars1", "Pars2"]

# loop for devices
for i in range(1):
    disdro_name = disdro_name_list[i]

    start = start_list[i]
    end = end_list[i]
    start_evt = datetime.strptime(start + " 00:00:00", "%Y/%m/%d %H:%M:%S")
    end_evt = datetime.strptime(end + " 23:59:30", "%Y/%m/%d %H:%M:%S")
    str_evt1 = (
        start_evt.strftime("%Y_%m_%d_%H_%M_%S")
        + "__"
        + end_evt.strftime("%Y_%m_%d_%H_%M_%S")
    )

    # path_data_base = '../Data_base_rw_turb/'
    path_data_base = "../"
    data_folder = "Daily_power_production_Boralex/"
    list_path = "output/disdro/"

    ##load list of events
    evt_list = pd.read_csv(
        list_path + "list_" + dw + "_abs_" + disdro_name_file + "_" + str_evt1 + ".csv",
        delimiter=";",
        header=None,
    )
    evt_list.columns = ["start", "end"]
    start_evt_list = evt_list.start
    end_evt_list = evt_list.end

    first = 0
    last = len(start_evt_list)

    D_0_list = []
    mu_g_list = []
    N_0_list = []
    D_m_list = []
    Lambda_list = []
    a_DSD_list = []
    b_DSD_list = []
    r_squared_dsd_list = []
    r_squared_g_list = []
    RMSE_list = []
    len_evt = []
    len_evt_N = []

    evno = 0
    m234_result = []
    nan_evt_append = []

    path_outputs = "output/MoM/"

    # first = 1
    # last = 20

    for j in range(first, last):
        ###adding extra timesteps if length of evens > 80% of closest power of 2
        start_evt = datetime.strptime(start_evt_list[j], "%Y_%m_%d_%H_%M_%S")
        end_evt = datetime.strptime(end_evt_list[j], "%Y_%m_%d_%H_%M_%S")
        str_evt_N = (
            start_evt.strftime("%Y_%m_%d_%H_%M_%S")
            + "__"
            + end_evt.strftime("%Y_%m_%d_%H_%M_%S")
        )

        print(str_evt)
        c = np.random.rand(
            3,
        )
        evno = evno + 1
        events = "event" + str(evno)

        print("current_event: " + str_evt + "__" + str(evno))

        ####
        ##getting rain rate
        ####

        ##for disdrometers on mast and radar tower
        parsivel_name = disdro_name
        path_data_base = "../"
        Pars_one_event = extracting_one_event_Parsivel(
            start_evt, end_evt, path_data_base, parsivel_name
        )

        # Definition of the data for the PWS
        T_Pars1 = Pars_one_event[0]
        R_disdro_Pars1 = Pars_one_event[1]
        R_emp_1_Pars1 = Pars_one_event[2]
        R_emp_2_Pars1 = Pars_one_event[3]
        Nb_all_drops_Pars1 = Pars_one_event[4]
        N_D_emp_Pars1 = Pars_one_event[5]
        rho_Pars1 = Pars_one_event[6]
        Nt_Pars1 = Pars_one_event[7]
        Dm_Pars1 = Pars_one_event[8]

        # For the Pars1
        V_Pars1 = np.array(
            (
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
                0.55,
                0.65,
                0.75,
                0.85,
                0.95,
                1.1,
                1.3,
                1.5,
                1.7,
                1.9,
                2.2,
                2.6,
                3,
                3.4,
                3.8,
                4.4,
                5.2,
                6,
                6.8,
                7.6,
                8.8,
                10.4,
                12,
                13.6,
                15.2,
                17.6,
                20.8,
            )
        )
        V_width_Pars1 = np.array(
            (
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.4,
                0.4,
                0.4,
                0.4,
                0.4,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                1.6,
                1.6,
                1.6,
                1.6,
                1.6,
                3.2,
                3.2,
            )
        )
        D_Pars1 = np.array(
            [
                0.062,
                0.187,
                0.312,
                0.437,
                0.562,
                0.687,
                0.812,
                0.937,
                1.062,
                1.187,
                1.375,
                1.625,
                1.875,
                2.125,
                2.375,
                2.750,
                3.250,
                3.750,
                4.250,
                4.750,
                5.5,
                6.5,
                7.5,
                8.5,
                9.5,
                11,
                13,
                15,
                17,
                19,
                21.5,
                24.5,
            ]
        )
        D_width_Pars1 = np.array(
            [
                0.125,
                0.125,
                0.125,
                0.125,
                0.125,
                0.125,
                0.125,
                0.125,
                0.125,
                0.125,
                0.250,
                0.250,
                0.250,
                0.250,
                0.250,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
            ]
        )
        V_Pars1 = V_Pars1
        V_width_Pars1 = V_width_Pars1
        D_Pars1 = D_Pars1
        D_width_Pars1 = D_width_Pars1
        V_theo_Pars1 = V_D_Lhermitte_1988(D_Pars1)
        S_eff_Pars1 = 10 ** (-6) * 180 * (30 - D_Pars1 / 2)

        R_data_all = R_emp_2_Pars1
        Nb_time_steps = R_data_all.shape[0]

        if (np.count_nonzero(R_data_all) / R_data_all.shape[0]) * 100 < 30:
            result.append(
                (
                    events,
                    disdro_name,
                    np.nan,
                    np.nan,
                    "70%+_zero",
                    np.nan,
                    str_evt,
                    np.nan,
                    np.nan,
                    evt_len,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            )
            if dw == "rain":
                continue

        ##KE calculation
        disdro_name = parsivel_name
        path_daily_data_python = "../Daily_data_python_disdrometer/"
        KE_data_all = exporting_KE(
            start_evt, end_evt, disdro_name, path_outputs, path_daily_data_python
        )  # The KE series is also given as output of the function
        # print('KE_data_'+disdro_name+start_evt.strftime('_%Y_%m_%d_%H_%M_%S')+'__'+end_evt.strftime('%Y_%m_%d_%H_%M_%S')+'_generated')

        KE_max = np.nanmax(KE_data_all)
        KE_mean = np.nanmean(KE_data_all)
        KE_std = np.nanstd(KE_data_all)

        LWC = rho_Pars1
        LWC_max = np.nanmax(LWC)
        LWC_mean = np.nanmean(LWC)
        LWC_std = np.nanstd(LWC)

        if R_data_all.shape[0] < 20:
            R_10min = np.nanmean(R_data_all)

        # 5min moving average
        R_5min = []
        ma = 0
        while ma <= (R_data_all.shape[0] - 10):
            mov_avg = np.mean(R_data_all[ma : ma + 10])
            ma = ma + 1
            R_5min.append(mov_avg)

        # 10min moving average
        R_10min = []
        ma = 0
        while ma <= (R_data_all.shape[0] - 20):
            mov_avg = np.mean(R_data_all[ma : ma + 20])
            ma = ma + 1
            R_10min.append(mov_avg)

        # if event less than moving average range, replace with mean
        if R_10min == [] or R_5min == []:
            R_5min = np.nanmean(R_data_all)
            R_10min = np.nanmean(R_data_all)

        # average rain rate criteria (Tokay and Short)
        avg_rain_rate = np.nanmean(R_data_all)
        std_dev_rr = np.std(R_data_all)

        buffer = np.copy(R_data_all)
        buffer[buffer == 0] = np.nan
        avg_rain_rate_rdp = np.nanmean(buffer)  # rainy data points only
        rain_type_list = [
            "very light",
            "light",
            "moderate",
            "heavy",
            "very heavy",
            "extreme",
        ]
        event_length = R_data_all.shape[0]
        # buffer[buffer == np.nan] = 0
        buffer = np.nan_to_num(buffer)
        rainy_data_points = len(np.nonzero(buffer)[0])
        # avg_rain_rate_5min = np.nanmean(R_data_all) #5 min average rain rate
        max_R_5min = np.nanmax(R_5min)  # max of moving average 5 min
        max_R_10min = np.nanmax(R_10min)  # max of moving average 10 min
        cumul_depth = []

        # total average
        if avg_rain_rate <= 1:
            rain_type = rain_type_list[0]
        if 1 <= avg_rain_rate < 2:
            rain_type = rain_type_list[1]
        if 2 <= avg_rain_rate < 5:
            rain_type = rain_type_list[2]
        if 5 <= avg_rain_rate < 10:
            rain_type = rain_type_list[3]
        if 10 <= avg_rain_rate < 20:
            rain_type = rain_type_list[4]
        if avg_rain_rate > 20:
            rain_type = rain_type_list[5]

        # rainy data points average
        if avg_rain_rate_rdp <= 1:
            rain_type_rdp = rain_type_list[0]
        if 1 <= avg_rain_rate_rdp < 2:
            rain_type_rdp = rain_type_list[1]
        if 2 <= avg_rain_rate_rdp < 5:
            rain_type_rdp = rain_type_list[2]
        if 5 <= avg_rain_rate_rdp < 10:
            rain_type_rdp = rain_type_list[3]
        if 10 <= avg_rain_rate_rdp < 20:
            rain_type_rdp = rain_type_list[4]
        if avg_rain_rate_rdp > 20:
            rain_type_rdp = rain_type_list[5]

        # 5min moving average
        if max_R_5min <= 1:
            rain_type_5min = rain_type_list[0]
        if 1 <= max_R_5min < 2:
            rain_type_5min = rain_type_list[1]
        if 2 <= max_R_5min < 5:
            rain_type_5min = rain_type_list[2]
        if 5 <= max_R_5min < 10:
            rain_type_5min = rain_type_list[3]
        if 10 <= max_R_5min < 20:
            rain_type_5min = rain_type_list[4]
        if max_R_5min > 20:
            rain_type_5min = rain_type_list[5]

        # 10min moving average
        if max_R_10min <= 1:
            rain_type_10min = rain_type_list[0]
        if 1 <= max_R_10min < 2:
            rain_type_10min = rain_type_list[1]
        if 2 <= max_R_10min < 5:
            rain_type_10min = rain_type_list[2]
        if 5 <= max_R_10min < 10:
            rain_type_10min = rain_type_list[3]
        if 10 <= max_R_10min < 20:
            rain_type_10min = rain_type_list[4]
        if max_R_10min > 20:
            rain_type_10min = rain_type_list[5]

        # R and KE max values
        R_max = np.nanmax(R_data_all)
        R_mean = np.nanmean(R_data_all)
        R_std = np.nanstd(R_data_all)

        # DSD condition definition for disdrometers
        if disdro_name == "Pars_1":
            N_D = N_D_emp_Pars1
            dD = D_width_Pars1
            NDdD = np.sum(N_D * dD, axis=0)
            ND = np.nanmean(N_D, axis=0)  # mean over whole time steps

            D_i = D_Pars1
            delta_D = D_width_Pars1

            temp = []
            for i in range(len(NDdD)):
                tp = np.sum(NDdD[0:i]) - (np.sum(NDdD) / 2)
                temp.append(tp)
            D_0 = D_i[np.where(np.sign(temp[:-1]) != np.sign(temp[1:]))[0]]

        #        if disdro_name == 'Pars_RW_turb_2':
        #            N_D=N_D_emp_Pars2
        #            dD=D_width_Pars2
        #            NDdD= np.sum(N_D*dD,axis=0)
        #            ND=np.nanmean(N_D,axis=0) #mean over whole time steps
        #
        #            D_i = D_Pars2
        #            delta_D = D_width_Pars2
        #
        #            temp=[]
        #            for i in range (len(NDdD)):
        #               tp = np.sum(NDdD[0:i])-(np.sum(NDdD)/2)
        #               temp.append(tp)
        #            D_0=D_i[np.where(np.sign(temp[:-1]) != np.sign(temp[1:]))[0]]

        #        #display DSD
        #        plt.figure(10)
        #        plt.plot(D_i[0:29],ND[0:29],color='r',lw=2)
        #        plt.title(disdro_name+'_'+str_evt)
        #        plt.ylabel(r'$N(D)_{emp}\/\/(m^{-3}.m^{-1})$',fontsize=20,color='k')
        #        plt.xlabel(r'$\mathit{D}\  (mm)$',fontsize=20,color='k')
        #        plt.legend()
        #        plt.show()

        from scipy.special import gamma, factorial
        from scipy.special import factorial

        # Smith M234 2005 medium range moments; normalized DSD
        ##
        # Brawn&Upton 2008, Cao&Zhang 2008 MoM234; non normalized
        # empirical DSD
        N_D_m = np.nanmean(N_D, axis=0)  # mean over all time steps

        # average moments for the events
        # moments 2nd, 3rd and 4th
        ##M_n = np.sum((D_i**n)*N_D_i*delta_D)
        M_2 = np.sum((D_i**2) * N_D_m * delta_D)
        M_3 = np.sum((D_i**3) * N_D_m * delta_D)
        M_4 = np.sum((D_i**4) * N_D_m * delta_D)
        M_6 = np.sum((D_i**6) * N_D_m * delta_D)

        D_m = M_4 / M_3

        ##DSD parameters with average Moments M234
        eta = (M_3**2) / (M_2 * M_4)
        mu_g = (1 / (1 - eta)) - 4
        lbda = (M_2 / M_3) * (mu_g + 3)
        N_0 = M_2 * (lbda ** (mu_g + 3)) / (gamma(mu_g + 3))

        #        ##DSD parameters with average Moments M346
        #        eta = (M_4**3)/((M_3**2)*M_6)
        #        mu_g = ((8-11*eta)-(((eta**2)+(8*eta))**0.5))/(2*(eta-1))
        #        lbda = (M_3/M_4)*(mu_g+4)
        #        N_0 = (M_3*(lbda**(mu_g+4)))/(gamma(mu_g+4))

        ##DSD a & b using theoretical formulae with average moments
        c_ms = 3.78
        a_DSD = (6.01 + mu_g) / (4.67 + mu_g)
        t1 = (6 * 3.14 * c_ms * N_0 * 10 ** (-4)) ** (1 - a_DSD)
        t2 = gamma(6.01 + mu_g) / ((gamma(4.67 + mu_g)) ** a_DSD)
        b_DSD = 5 * 10 ** (-4) * 1000 * (c_ms**2) * t1 * t2

        ##induvidual time step moments
        ##M_n = np.sum((D_i**n)*N_D_i*delta_D)
        R_DSD = []
        mu_g_im = []
        N_0_im = []
        lbda_im = []
        M_2_im = []
        M_3_im = []
        M_4_im = []
        M_6_im = []
        ND_Ulbrich_im = []
        a_DSD_im = []
        b_DSD_im = []
        eta_im = []
        stra_con_im = []
        s_c_N0_i = []
        s_c_lbda_i = []
        R_cs_i = []
        for ind in range(N_D.shape[0]):
            N_D_i = N_D[ind]  # sep for each time steps
            M_2_i = np.sum((D_i**2) * N_D_i * delta_D)
            M_3_i = np.sum((D_i**3) * N_D_i * delta_D)
            M_4_i = np.sum((D_i**4) * N_D_i * delta_D)
            M_6_i = np.sum((D_i**6) * N_D_i * delta_D)
            D_m_i = M_4 / M_3

            ##DSD parameters with ind Moments M234
            eta_i = (M_3_i**2) / (M_2_i * M_4_i)
            mu_g_i = (1 / (1 - eta_i)) - 4
            lbda_i = (M_2_i / M_3_i) * (mu_g_i + 3)
            N_0_i = M_2_i * (lbda_i ** (mu_g_i + 3)) / (gamma(mu_g_i + 3))

            ##strat_conv_N0
            rr_ind = R_data_all[ind]
            N_0_i_line = 4 * (10**9) * rr_ind ** (-4.3)
            if N_0_i > N_0_i_line:
                s_c_N0_i.append("conv")
            else:
                s_c_N0_i.append("strat")

            ##strat_conv_lambda
            rr_ind = R_data_all[ind]
            lbda_i_line = 17 * (rr_ind ** (-0.37))
            if lbda_i > lbda_i_line:
                s_c_lbda_i.append("conv")
            else:
                s_c_lbda_i.append("strat")

            ##strat_conv_Marzano2010
            if ind < 5 or ind > (N_D.shape[0] - 5):
                R_cs_i.append("unknown")
            elif ind >= 5 and ind <= (N_D.shape[0] - 5):
                if np.all(R_data_all[ind - 5 : ind + 5] > 10):
                    R_cs_i.append("conv")
                elif (
                    np.all(R_data_all[ind - 5 : ind + 5] < 10)
                    and np.std(R_data_all[ind - 5 : ind + 5]) < 1.5
                ):
                    R_cs_i.append("strat")
                else:
                    R_cs_i.append("mixed")

            #            ##DSD parameters with ind Moments M346
            #            eta_i = (M_4_i**3)/((M_3_i**2)*M_6_i)
            #            mu_g_i = ((8-(11*eta_i))-(((eta_i**2)+(8*eta_i))**0.5))/(2*(eta_i-1))
            #            lbda_i = (M_3_i/M_4_i)*(mu_g_i+4)
            #            N_0_i = (M_3_i*(lbda_i**(mu_g_i+4)))/(gamma(mu_g_i+4))

            # gamma DSD
            ND_Ulbrich_i = N_0_i * (D_i**mu_g_i) * np.exp(-lbda_i * D_i)

            ##Rain rate from fitted DSD
            R_DSD_i = (
                6
                * (10 ** (-4))
                * 3.14
                * 3.78
                * N_0_i
                * gamma(4 + 0.67 + mu_g_i)
                / (lbda_i ** (4 + 0.67 + mu_g_i))
            )

            c_ms = 3.78
            a_DSD_i = (6.01 + mu_g_i) / (4.67 + mu_g_i)
            t1 = (6 * 3.14 * c_ms * N_0_i * 10 ** (-4)) ** (1 - a_DSD_i)
            t2 = gamma(6.01 + mu_g_i) / ((gamma(4.67 + mu_g_i)) ** a_DSD_i)
            b_DSD_i = 5 * 10 ** (-4) * 1000 * (c_ms**2) * t1 * t2

            R_DSD.append(R_DSD_i)
            mu_g_im.append(mu_g_i)
            N_0_im.append(N_0_i)
            lbda_im.append(lbda_i)
            M_2_im.append(M_2_i)
            M_3_im.append(M_3_i)
            M_4_im.append(M_4_i)
            M_6_im.append(M_6_i)
            ND_Ulbrich_im.append(ND_Ulbrich_i)
            a_DSD_im.append(a_DSD_i)
            b_DSD_im.append(b_DSD_i)
            eta_im.append(eta_i)
        R_DSD = np.array(R_DSD)
        ND_Ulbrich_imean = np.stack(ND_Ulbrich_im, axis=0)
        ND_Ulbrich_imean[~np.isfinite(ND_Ulbrich_imean)] = 0

        s_c_N0_i = np.array(s_c_N0_i)
        perc_con_N0_i = (np.where(s_c_N0_i == "conv")[0].shape[0] / len(s_c_N0_i)) * 100
        perc_strat_N0_i = (
            np.where(s_c_N0_i == "strat")[0].shape[0] / len(s_c_N0_i)
        ) * 100
        s_c_lbda_i = np.array(s_c_lbda_i)
        perc_con_lbda_i = (
            np.where(s_c_lbda_i == "conv")[0].shape[0] / len(s_c_lbda_i)
        ) * 100
        perc_strat_lbda_i = (
            np.where(s_c_lbda_i == "strat")[0].shape[0] / len(s_c_lbda_i)
        ) * 100
        R_cs_i = np.array(R_cs_i)
        perc_con_R_cs_i = (
            np.where(R_cs_i == "conv")[0].shape[0] / len(s_c_lbda_i)
        ) * 100
        perc_strat_R_cs_i = (
            np.where(R_cs_i == "strat")[0].shape[0] / len(s_c_lbda_i)
        ) * 100
        perc_unkn_R_cs_i = (
            np.where(R_cs_i == "unknown")[0].shape[0] / len(s_c_lbda_i)
        ) * 100
        perc_mixd_R_cs_i = (
            np.where(R_cs_i == "mixed")[0].shape[0] / len(s_c_lbda_i)
        ) * 100

        R_30s = np.copy(R_data_all)

        # M=4
        # ##resolution changeto 2min
        # R_2m=np.zeros((sp.int0(R_30s.shape[0]/M,)))
        # for n in range(R_2m.shape[0]):
        #     R_2m[n]=np.mean(R_30s[n*M:(n+1)*M])

        # gamma DSD ind moments
        ND_Ulbrich_m_ind = np.nanmean(ND_Ulbrich_imean, axis=0)
        res = ND - ND_Ulbrich_m_ind
        ss_res = np.nansum(res**2)
        ss_tot = np.nansum((ND - np.nanmean(ND)) ** 2)
        r_squared_g_ind = 1 - (ss_res / ss_tot)
        RMSE_ind = np.sqrt(ss_res / ND.shape[0])

        # gamma DSD mean moments
        ND_Ulbrich_m = N_0 * (D_i**mu_g) * np.exp(-lbda * D_i)
        res = ND - ND_Ulbrich_m
        ss_res = np.nansum(res**2)
        ss_tot = np.nansum((ND - np.nanmean(ND)) ** 2)
        r_squared_g = 1 - (ss_res / ss_tot)
        RMSE = np.sqrt(ss_res / ND.shape[0])

        if img_sav == 1:
            plt.figure(40)
            plt.plot(
                D_i[0:25],
                ND_Ulbrich_m[0:25],
                color="b",
                lw=2,
                label="RMSE = %.3f" % RMSE + " ;  " + r"$r^2$ = %5.3f" % r_squared_g,
            )
            # plt.plot(D_i[0:29],ND_Ulbrich_m_ind[0:29],color='g',lw=2, label = '$RMSE \ ND_{ind}$ = %.3f' % RMSE_ind + ' ;  ' + r'$r^2$ = %5.3f' %r_squared_g_ind)
            plt.plot(D_i[0:25], ND[0:25], color="r", lw=2)
            plt.title(disdro_name + "_" + str_evt)
            plt.ylabel(r"$N(D)\/\/(m^{-3}.mm^{-1})$", fontsize=20, color="k")
            plt.xlabel(r"$\mathit{D}\  (mm)$", fontsize=20, color="k")
            plt.legend()
            if img_sav == 1:
                plt.savefig(
                    path_outputs
                    + "RMSE_ND_r_sq_dsd/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_ND_Ulbrich.png",
                    bbox_inches="tight",
                    dpi=200,
                )
            plt.show()
            plt.close()

        ###for displaying dsd
        if disdro_name == "PWS":
            slc = 10
            slc2 = 5
        else:
            slc = 8
            slc2 = 4

        ##limiting length of DSD diameter range
        # gamma DSD ind moments 1mm
        ND_Ulbrich_m_ind_1mm = np.nanmean(ND_Ulbrich_imean, axis=0)
        ND_Ulbrich_m_ind_1mm = ND_Ulbrich_m_ind_1mm[
            slc:
        ]  # slc corresponds to 1.1mm dia
        ND_1mm = ND[slc:]
        res = ND_1mm - ND_Ulbrich_m_ind_1mm
        ss_res = np.nansum(res**2)
        ss_tot = np.nansum((ND - np.nanmean(ND)) ** 2)
        r_squared_g_ind_1mm = 1 - (ss_res / ss_tot)
        RMSE_ind_1mm = np.sqrt(ss_res / ND.shape[0])

        # gamma DSD mean moments 1mm
        ND_Ulbrich_m_1mm = N_0 * (D_i**mu_g) * np.exp(-lbda * D_i)
        ND_Ulbrich_m_1mm = ND_Ulbrich_m_1mm[slc:]  # slc corresponds to 1.1mm dia
        ND_1mm = ND[slc:]
        res = ND_1mm - ND_Ulbrich_m_1mm
        ss_res = np.nansum(res**2)
        ss_tot = np.nansum((ND - np.nanmean(ND)) ** 2)
        r_squared_g_1mm = 1 - (ss_res / ss_tot)
        RMSE_1mm = np.sqrt(ss_res / ND.shape[0])

        # DSD1mm
        if img_sav == 1:
            plt.figure(41)
            # plt.plot(D_i[0:29],ND_Ulbrich_m[0:29],color='b',lw=2, label = '$RMSE \ ND_{avg}$ = %.3f' % RMSE + ' ;  ' + r'$r^2$ = %5.3f' %r_squared_g)
            plt.plot(
                D_i[slc:25],
                ND_Ulbrich_m[slc:25],
                color="b",
                lw=2,
                label="RMSE = %.3f" % RMSE_1mm
                + " ;  "
                + r"$r^2$ = %5.3f" % r_squared_g_1mm,
            )
            # plt.plot(D_i[0:29],ND_Ulbrich_m_ind[0:29],color='g',lw=2, label = '$RMSE \ ND_{ind}$ = %.3f' % RMSE_ind + ' ;  ' + r'$r^2$ = %5.3f' %r_squared_g_ind)
            plt.plot(D_i[slc:25], ND[slc:25], color="r", lw=2)
            plt.title(disdro_name + "_" + str_evt + "_ND_1mm")
            plt.ylabel(r"$N(D)\/\/(m^{-3}.mm^{-1})$", fontsize=20, color="k")
            plt.xlabel(r"$\mathit{D}\  (mm)$", fontsize=20, color="k")
            plt.legend()
            if img_sav == 1:
                # plt.savefig(path_outputs+disdro_name+'_'+str_evt+'_ND_1mm.png',bbox_inches='tight', dpi=200)
                plt.savefig(
                    path_outputs
                    + "RMSE_ND_r_sq_dsd/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_ND_1mm.png",
                    bbox_inches="tight",
                    dpi=200,
                )
            plt.show()
            plt.close()

        # DSD0.5mm
        # gamma DSD ind moments 0.5mm
        ND_Ulbrich_m_ind_05 = np.nanmean(ND_Ulbrich_imean, axis=0)
        ND_Ulbrich_m_ind_05 = ND_Ulbrich_m_ind_05[slc2:]  # slc corresponds to 1.05 dia
        ND_05 = ND[slc2:]
        res = ND_05 - ND_Ulbrich_m_ind_05
        ss_res = np.nansum(res**2)
        ss_tot = np.nansum((ND - np.nanmean(ND)) ** 2)
        r_squared_g_ind_05 = 1 - (ss_res / ss_tot)
        RMSE_ind_05mm = np.sqrt(ss_res / ND.shape[0])

        # gamma DSD mean moments 0.5mm
        ND_Ulbrich_m_05 = N_0 * (D_i**mu_g) * np.exp(-lbda * D_i)
        ND_Ulbrich_m_05 = ND_Ulbrich_m_05[slc2:]  # slc corresponds to 1.05 dia
        ND_05 = ND[slc2:]
        res = ND_05 - ND_Ulbrich_m_05
        ss_res = np.nansum(res**2)
        ss_tot = np.nansum((ND - np.nanmean(ND)) ** 2)
        r_squared_g_05 = 1 - (ss_res / ss_tot)
        RMSE_05mm = np.sqrt(ss_res / ND.shape[0])

        # DSD0.5mm
        if img_sav == 1:
            plt.figure(42)
            # plt.plot(D_i[0:29],ND_Ulbrich_m[0:29],color='b',lw=2, label = '$RMSE \ ND_{avg}$ = %.3f' % RMSE + ' ;  ' + r'$r^2$ = %5.3f' %r_squared_g)
            plt.plot(
                D_i[slc2:25],
                ND_Ulbrich_m[slc2:25],
                color="b",
                lw=2,
                label="RMSE = %.3f" % RMSE_05mm
                + " ;  "
                + r"$r^2$ = %5.3f" % r_squared_g_05,
            )
            # plt.plot(D_i[0:29],ND_Ulbrich_m_ind[0:29],color='g',lw=2, label = '$RMSE \ ND_{ind}$ = %.3f' % RMSE_ind + ' ;  ' + r'$r^2$ = %5.3f' %r_squared_g_ind)
            plt.plot(D_i[slc2:25], ND[slc2:25], color="r", lw=2)
            plt.title(disdro_name + "_" + str_evt + "_ND_0.5mm")
            plt.ylabel(r"$N(D)\/\/(m^{-3}.mm^{-1})$", fontsize=20, color="k")
            plt.xlabel(r"$\mathit{D}\  (mm)$", fontsize=20, color="k")
            plt.legend()
            if img_sav == 1:
                # plt.savefig(path_outputs+disdro_name+'_'+str_evt+'_ND_05.png',bbox_inches='tight', dpi=200)
                img05 = (
                    path_outputs
                    + "RMSE_ND_r_sq_dsd/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_ND_0.5mm.png"
                )
                plt.savefig(img05, bbox_inches="tight", dpi=200)
            # plt.savefig(path_output+'temp',bbox_inches='tight', dpi=200)
            plt.show()
            plt.close()

        peak = D_i[np.where(ND == np.nanmax(ND))[0][0]]

        x = [0, max(np.nanmax(R_DSD), np.nanmax(R_data_all))]
        y = [0, max(np.nanmax(R_DSD), np.nanmax(R_data_all))]

        if img_sav == 1:
            # R_vs_R_DSD
            plt.figure(45)
            plt.scatter(R_data_all, R_DSD)
            plt.plot(x, y, ":", color="k", lw="1")
            plt.title(disdro_name + "_" + str_evt)
            plt.xlabel(r"Observed rainfall rate $(mmh^{-1})$", fontsize=10, color="k")
            plt.ylabel(
                r"Moment calculated rainfall rate $(mmh^{-1})$", fontsize=10, color="k"
            )
            plt.legend()
            if img_sav == 1:
                img1 = (
                    path_outputs
                    + "dsd_par/par/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_R_vs_R_DSD.png"
                )
                plt.savefig(img1, bbox_inches="tight", dpi=200)
            plt.show()
            plt.close()
            # R_R_DSD
            plt.figure(46)
            time = np.arange(0, R_data_all.shape[0])
            plt.plot(time, R_data_all, label="R emp")
            plt.plot(time, R_DSD, color="orange", label="R DSD")
            plt.title(disdro_name + "_" + str_evt)
            plt.ylabel(r"Observed rainfall rate $(mmh^{-1})$", fontsize=10, color="k")
            plt.xlabel(r"time $(s)$", fontsize=10, color="k")
            plt.legend()
            if img_sav == 1:
                img2 = (
                    path_outputs
                    + "dsd_par/R_R_dsd/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_R_R_DSD.png"
                )
                plt.savefig(img2, bbox_inches="tight", dpi=200)
            plt.show()
            plt.close()
            # Distribution plots
            import seaborn as sns

            # mu
            plt.figure(47)
            A = np.array(mu_g_im)
            Anan = A[np.isfinite(A)]
            Amean = Anan.mean()
            sns.distplot(Anan, hist=True, label="mean = " + "%.2f" % Amean)
            plt.title(disdro_name + "_" + str_evt)
            plt.xlabel("Shape parameter (mu)", fontsize=10, color="k")
            plt.ylabel("Counts", fontsize=10, color="k")
            plt.legend()
            if img_sav == 1:
                img3 = (
                    path_outputs
                    + "dsd_par/par/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_mu.png"
                )
                plt.savefig(img3, bbox_inches="tight", dpi=200)
            plt.show()
            plt.close()
            # N_0
            plt.figure(48)
            B = np.array(N_0_im)
            Bnan = B[np.isfinite(B)]
            Bmean = Bnan.mean()
            sns.distplot(Bnan, hist=True, label="mean = " + "%.1E" % Bmean)
            plt.title(disdro_name + "_" + str_evt)
            plt.xlabel("Intercept parameter (N_0)", fontsize=10, color="k")
            plt.ylabel("Counts", fontsize=10, color="k")
            plt.legend()
            if img_sav == 1:
                img4 = (
                    path_outputs
                    + "dsd_par/par/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_N_0.png"
                )
                plt.savefig(img4, bbox_inches="tight", dpi=200)
            plt.show()
            plt.close()
            # lbda_im
            plt.figure(49)
            C = np.array(lbda_im)
            Cnan = C[np.isfinite(C)]
            Cmean = Cnan.mean()
            sns.distplot(Cnan, hist=True, label="mean = " + "%.2f" % Cmean)
            plt.title(disdro_name + "_" + str_evt)
            plt.xlabel("Slope parameter " + r"$\Lambda$", fontsize=10, color="k")
            plt.ylabel("Counts", fontsize=10, color="k")
            plt.legend()
            if img_sav == 1:
                img5 = (
                    path_outputs
                    + "dsd_par/par/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_lbda.png"
                )
                plt.savefig(img5, bbox_inches="tight", dpi=200)
            plt.show()
            plt.close()
            # a_DSD_ind_b_DSD_ind
            # a_time_series
            plt.figure(52)
            a_DSD_ind = np.array(a_DSD_im)
            a_DSD_indm = np.nanmean(a_DSD_ind)
            plt.plot(time, a_DSD_ind, label="mean = " + "%.2f" % a_DSD_indm)
            plt.title(disdro_name + "_" + str_evt)
            plt.ylabel("exponent " + r"$a$", fontsize=10, color="k")
            plt.xlabel("time $(s)$", fontsize=10, color="k")
            plt.legend()
            if img_sav == 1:
                img6 = (
                    path_outputs
                    + "dsd_par/par/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_a_ts.png"
                )
                plt.savefig(img6, bbox_inches="tight", dpi=200)
            plt.show()
            plt.close()
            # b_time_series
            plt.figure(53)
            b_DSD_ind = np.array(b_DSD_im)
            b_DSD_indm = np.nanmean(b_DSD_ind)
            plt.plot(time, b_DSD_ind, label="mean = " + "%.2f" % b_DSD_indm)
            plt.title(disdro_name + "_" + str_evt)
            plt.ylabel("exponent " + r"$b$", fontsize=10, color="k")
            plt.xlabel("time $(s)$", fontsize=10, color="k")
            plt.legend()
            if img_sav == 1:
                img7 = (
                    path_outputs
                    + "dsd_par/par/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_b_ts.png"
                )
                plt.savefig(img7, bbox_inches="tight", dpi=200)
            plt.show()
            plt.close()
            # a_N_0_mu
            data_N0 = np.log(B)
            data_mu = A
            a_DSD_indn = a_DSD_ind  # [np.isfinite(a_DSD_ind)]
            b_DSD_indn = b_DSD_ind  # [np.isfinite(b_DSD_ind)]
            fig, ax1 = plt.subplots()
            color = "tab:red"
            ax1.set_xlabel("exponent " + r"$a$", fontsize=12)
            ax1.set_ylabel(r"log $N_0$", color=color, fontsize=12)
            ax1.scatter(
                a_DSD_indn,
                data_N0,
                color=color,
                label="mean a = " + "%.2f" % b_DSD_indm,
            )
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.set_title(disdro_name + "_" + str_evt)
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = "tab:blue"
            ax2.set_ylabel(
                r"$\mu$", color=color, fontsize=12
            )  # we already handled the x-label with ax1
            ax2.scatter(a_DSD_indn, data_mu, color=color)
            ax2.tick_params(axis="y", labelcolor=color)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.annotate(
                "mean a = " + "%.2f" % a_DSD_indm,
                xy=(0.65, 0.93),
                xycoords="axes fraction",
            )
            plt.annotate(
                "mean $N_0$ = " + "%.1E" % Bmean,
                xy=(0.65, 0.87),
                xycoords="axes fraction",
            )
            plt.annotate(
                "mean $\mu$ = " + "%.2f" % Amean,
                xy=(0.65, 0.82),
                xycoords="axes fraction",
            )
            plt.annotate(
                "a evt = " + "%.2f" % a_DSD, xy=(0.65, 0.73), xycoords="axes fraction"
            )
            plt.annotate(
                "$N_0$ evt = " + "%.1E" % N_0, xy=(0.65, 0.67), xycoords="axes fraction"
            )
            plt.annotate(
                "$\mu$ evt = " + "%.2f" % mu_g,
                xy=(0.65, 0.62),
                xycoords="axes fraction",
            )
            if img_sav == 1:
                img8 = (
                    path_outputs
                    + "dsd_par/par/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_a_N_0_mu.png"
                )
                fig.savefig(img8, bbox_inches="tight", dpi=200)
            plt.show()
            plt.close()
            # b_N_0_mu
            fig, ax1 = plt.subplots()
            color = "tab:red"
            ax1.set_xlabel("exponent " + r"$b$", fontsize=12)
            ax1.set_ylabel(r"log $N_0$", color=color, fontsize=12)
            ax1.scatter(b_DSD_indn, data_N0, color=color)
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.set_title(disdro_name + "_" + str_evt)
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = "tab:blue"
            ax2.set_ylabel(
                r"$\mu$", color=color, fontsize=12
            )  # we already handled the x-label with ax1
            ax2.scatter(b_DSD_indn, data_mu, color=color)
            ax2.tick_params(axis="y", labelcolor=color)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.annotate(
                "mean b = " + "%.2f" % b_DSD_indm,
                xy=(0.65, 0.93),
                xycoords="axes fraction",
            )
            plt.annotate(
                "mean $N_0$ = " + "%.1E" % Bmean,
                xy=(0.65, 0.87),
                xycoords="axes fraction",
            )
            plt.annotate(
                "mean $\mu$ = " + "%.2f" % Amean,
                xy=(0.65, 0.82),
                xycoords="axes fraction",
            )
            plt.annotate(
                "b evt = " + "%.2f" % b_DSD, xy=(0.65, 0.73), xycoords="axes fraction"
            )
            plt.annotate(
                "$N_0$ evt = " + "%.1E" % N_0, xy=(0.65, 0.67), xycoords="axes fraction"
            )
            plt.annotate(
                "$\mu$ evt = " + "%.2f" % mu_g,
                xy=(0.65, 0.62),
                xycoords="axes fraction",
            )
            if img_sav == 1:
                img9 = (
                    path_outputs
                    + "dsd_par/par/"
                    + disdro_name
                    + "_"
                    + str_evt
                    + "_b_N_0_mu.png"
                )
                fig.savefig(img9, bbox_inches="tight", dpi=200)
            plt.show()
            plt.close()

        if img_sav == 1:
            import PIL
            from PIL import Image

            imgs1 = [
                PIL.Image.open(img1),
                PIL.Image.open(img3),
                PIL.Image.open(img6),
                PIL.Image.open(img8),
            ]
            imgs2 = [
                PIL.Image.open(img4),
                PIL.Image.open(img5),
                PIL.Image.open(img7),
                PIL.Image.open(img9),
            ]
            imgs_strcon = [
                PIL.Image.open(img10),
                PIL.Image.open(img11),
                PIL.Image.open(img12),
                PIL.Image.open(img13),
            ]
            # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
            min_shape = sorted([(np.sum(i.size), i.size) for i in imgs1])[0][1]
            imgs_comb1 = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs1))
            imgs_comb2 = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs2))
            imgs_comb3 = np.hstack(
                (np.asarray(i.resize(min_shape)) for i in imgs_strcon)
            )
            imgs3 = [imgs_comb1, imgs_comb2, imgs_comb3]
            imgs_comb = np.vstack((np.asarray(i) for i in imgs3))
            # save that beautiful picture
            imgs_comb = PIL.Image.fromarray(imgs_comb)
            imgs_comb.save(
                path_outputs
                + "dsd_par/par/"
                + disdro_name
                + "_"
                + str_evt
                + "_DSD_params.png"
            )

            file_list = [
                img1,
                img3,
                img4,
                img5,
                img6,
                img7,
                img8,
                img9,
                img10,
                img11,
                img12,
                img13,
            ]
            for d in range(len(file_list)):
                file = file_list[d]
                os.remove(file)

            # combining images 2
            imgx = (
                path_outputs
                + "RMSE_ND_r_sq_dsd/"
                + disdro_name
                + "_"
                + str_evt
                + "_ND_Ulbrich.png"
            )
            imgy = img2
            imgz = (
                path_outputs
                + "RMSE_ND_r_sq_dsd/"
                + disdro_name
                + "_"
                + str_evt
                + "_ND_1mm.png"
            )
            imgsxy = [PIL.Image.open(imgx), PIL.Image.open(imgy)]
            imgsv2 = [PIL.Image.open(img05), PIL.Image.open(imgz)]
            # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
            min_shape = sorted([(np.sum(i.size), i.size) for i in imgsxy])[0][1]
            imgs_comb_h1 = np.hstack((np.asarray(i.resize(min_shape)) for i in imgsxy))

            imgs_comb_h2 = np.hstack((np.asarray(i.resize(min_shape)) for i in imgsv2))

            imgs4 = [imgs_comb_h1, imgs_comb_h2]
            imgs_comb = np.vstack((np.asarray(i) for i in imgs4))

            # save that beautiful picture
            imgs_comb = PIL.Image.fromarray(imgs_comb)
            imgs_comb.save(
                path_outputs
                + "/RMSE_ND_r_sq_dsd/"
                + disdro_name
                + "_"
                + str_evt
                + "RMSE_ND_r_sq_dsd_b_fit.png"
            )
            # deleting individual image
            os.remove(imgx)
            # os.remove(imgy)
            os.remove(imgz)
            os.remove(img05)

        m234_result.append(
            (
                evno,
                str_evt,
                event_length,
                rainy_data_points,
                R_max,
                R_mean,
                R_std,
                KE_max,
                KE_mean,
                KE_std,
                avg_rain_rate,
                rain_type,
                avg_rain_rate_rdp,
                rain_type_rdp,
                max_R_5min,
                rain_type_5min,
                max_R_10min,
                rain_type_10min,
                RMSE,
                r_squared_g,
                RMSE_1mm,
                RMSE_05mm,
                peak,
                D_0[0],
                mu_g,
                N_0,
                D_m,
                lbda,
                strat_conv,
                s_c_N0,
                s_c_lbda,
                perc_con,
                perc_strat,
                perc_none,
                perc_con_N0_i,
                perc_strat_N0_i,
                perc_con_lbda_i,
                perc_strat_lbda_i,
            )
        )

    # writing results
    # df = pd.DataFrame(m234_result)
    index = [
        "#event",
        "event",
        "event_length",
        "rainy_data_points",
        "R_max",
        "R_mean",
        "R_std",
        "KE_max",
        "KE_mean",
        "KE_std",
        "avg_rain_rate",
        "Rain_type",
        "avg_rain_rate_rdp",
        "Rain_type_rdp",
        "R_5min_mov_avg",
        "Rain_type_5min_mov_avg",
        "max_R_10min",
        "Rain_type_10min_mov_avg",
        "RMSE_ND",
        "r_squared_g_ND",
        "RMSE_1mm",
        "RMSE_05mm",
        "peak",
        "D_0",
        "mu_g",
        "N_0",
        "D_m",
        "lambda",
        "strat_conv_BR03",
        "strat_conv_N_0",
        "strat_conv_lbda",
        "perc_con_Marz",
        "perc_strat_Marz",
        "perc_none_Marz",
        "perc_con_N0_i",
        "perc_strat_N0_i",
        "perc_con_lbda_i",
        "perc_strat_lbda_i",
    ]

    df_dsd = pd.DataFrame(m234_result, columns=index, dtype="float")
    df_dsd.to_excel(
        path_outputs
        + dw
        + "_dsd_"
        + str_evt1
        + "_loc"
        + str(loc)
        + "_rw_turb_M234"
        + ".xlsx"
    )

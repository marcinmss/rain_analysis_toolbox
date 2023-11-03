# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:29:30 2013

@author: gires
updated TM_corr for 2/3 scaling regimes
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


#####################################################################
#####################################################################
##  This script was used for the preparation of the following paper :
##  “Approximate multifractal correlation and products of universal multifractal fields,
##  with application to rainfall data” by Auguste Gires, Ioulia Tchiguirinskaia, and
##  Daniel Schertzer which has been published in 2020 in Nonlin. Processes Geophys.
##  (https://www.nonlinear-processes-in-geophysics.net/).
##  It should be cited if used.
##
##  It contains few additional multifractal tools that were developed for this paper, and notable TM_corr funtion which computes r(q,h) (Eq. 4)
#####################################################################
#####################################################################


# This script allows to perform a correlated TM analysis for two fields (see paper)
# It currently works with 1D fields with a unique scaling regime (which is what is used in the paper l_range=[])
# It should be updated for the other cases.


def TM_corr(
    data_q,
    data_h,
    q_values,
    h_values,
    data_file_name,
    dim,
    l_range,
    file_index,
    file_name,
    plot_index,
):
    # Inputs :
    # - data_h or data_q : data is a numpy matrix that contain the data to be analysed.
    #          Its meaning depends on dim
    #      - dim=1 : 1D analysis are performed
    #                data is a 2D matrix, each column is a sample
    #                size(data)=[2^power,2^power,n], where n is number of samples
    #                if one column, then a sample analysis is performed)
    #      - dim=2 : 2D analysis are performed
    #                data is a 3D matrix, each layer (i.e. data[:,:,s]) is a sample
    #                size(data)=[2^power,2^power,n], where n is number of samples
    #                (if one layer, then a sample analysis is performed)
    # - q_values : a numpy vector containing the different q for which r(q,h) is evaluated
    #          if q_values=np.array([-1]), then a default value is set np.concatenate((np.arange(0.05,1,0.05),np.array([1.01]),np.array([1.05]),np.arange(1.1,3.1,0.1)),axis=0)
    # - h_values : a numpy vector containing the different q for which r(q,h) is evaluated
    #          if h_values=np.array([-1]), then a default value is set np.concatenate((np.arange(0.05,1,0.05),np.array([1.01]),np.array([1.05]),np.arange(1.1,3.1,0.1)),axis=0)
    #               (it is advised to use this option)
    # - data_file_name : - if data_file_name =='' then "data" is used for the analysis
    #                    - otherwise : data_file_name is list of .csv file name
    #                                  the 1D or 2D field contained in each file is considered as a independant sample
    #                                  (used for large files)
    # - l_range is a list of list. Each element contains two elements (reso_min and reso_max)
    #                and a fractal dimension is evaluated between reso_min and reso_max
    #                if l_range=[] then a single scaling regime is considered and the all the available resolution are used
    #                the max length of l_range is 3; ie no more than three scaling regime can be studied
    # - file_index : the computation of the moments might be quite long, and therefore can be recorded in a file
    #                that can be used later
    #                file_index=0 --> "moments" is recorded
    #                file_index=1 --> "moments" is retrieved from an existing file
    # - file_name : the name of the file recorded or retrieved
    #               WARNING the file should have a .npy extension !!
    # - plot_index : the number of the first figure opened for graph display
    #
    # Outputs :
    #  - rqh_1 : the K(q) function of the scaling regime associated with l_range[0] for the values in q_values
    #  - rqh_2 : the K(q) function of the scaling regime associated with l_range[1] for the values in q_values
    #  - rqh_3 : the K(q) function of the scaling regime associated with l_range[2] for the values in q_values
    #  - r2_1 : the values of r2 for each q obtained in the linear regression leading to K(q) for the 1st scaling regime
    #  - r2_2 : the values of r2 for each q obtained in the linear regression leading to K(q) for the 2nd scaling regime
    #  - r2_3 : the values of r2 for each q obtained in the linear regression leading to K(q) for the 3rd scaling regime
    #
    # Note 1 : the output of "unused" scaling regime(s) are returned as "nan"
    # Note 2 : a ration of 2 is used in the upscaling proces

    # Evaluating l_max (maximum resolution) and nb_s (number of samples)
    if dim == 1:
        if data_file_name == "":
            l_max = data_q.shape[0]
            if len(data_q.shape) == 1:
                nb_s = 1
            else:
                nb_s = data_q.shape[1]
        else:
            nb_s = len(data_file_name)
            l_max = np.loadtxt(data_file_name[0]).shape[0]
    elif dim == 2:
        if data_file_name == "":
            l_max = data_q.shape[0]
            if len(data_q.shape) == 2:
                nb_s = 1
            else:
                nb_s = data_q.shape[2]
        else:
            nb_s = len(data_file_name)
            l_max = np.loadtxt(data_file_name[0], delimiter=";").shape[0]
    else:
        print("Error in TM, wrong dim")

    # TO BE DONE : write an error message of the size is not a power of 2
    n_max = np.intp(sp.log(l_max) / sp.log(2))
    l_list = np.zeros((n_max + 1))
    for n in range(np.intp(n_max) + 1):
        l_list[n] = sp.power(2, n)

    # Affecting the value of q_values if the default option has been selected
    # The initial value is stored in q_values_bis (which not is a numpy array for simplicity)
    if q_values.shape[0] == 1 and q_values[0] == -1:
        q_values = np.concatenate(
            (
                np.arange(0.05, 1, 0.05),
                np.array([1.01]),
                np.array([1.05]),
                np.arange(1.1, 3.1, 0.1),
            ),
            axis=0,
        )
        q_values_bis = -1
    else:
        q_values_bis = 1

    # Affecting the value of q_values if the default option has been selected
    # The initial value is stored in q_values_bis (which not is a numpy array for simplicity)
    if h_values.shape[0] == 1 and h_values[0] == -1:
        h_values = np.concatenate(
            (
                np.arange(0.05, 1, 0.05),
                np.array([1.01]),
                np.array([1.05]),
                np.arange(1.1, 3.1, 0.1),
            ),
            axis=0,
        )
        h_values_bis = -1
    else:
        h_values_bis = 1

    nb_q = q_values.shape[0]
    nb_h = h_values.shape[0]

    rqh_1 = np.zeros((nb_q, nb_h))
    rqh_2 = np.zeros((nb_q, nb_h))
    rqh_3 = np.zeros((nb_q, nb_h))
    r2_1 = np.zeros((nb_q, nb_h))
    r2_2 = np.zeros((nb_q, nb_h))
    r2_3 = np.zeros((nb_q, nb_h))

    moments_q = np.zeros((n_max + 1, nb_q))
    moments_h = np.zeros((n_max + 1, nb_h))
    moments_q_h = np.zeros((n_max + 1, nb_q, nb_h))

    # Step 1 : Evaluate the moments <data_q^q  data_h^h>
    #          Results are stored in a numpy array moments
    #                               (1st index --> resolution,)
    #                               (2nd index --> value of q, same order as in q_values )

    if file_index == 0:  # moments is computed
        if dim == 1:
            if data_file_name == "":
                for n_q in range(nb_q):
                    moments_q[n_max, n_q] = np.mean(np.power(data_q, q_values[n_q]))
                for n_h in range(nb_h):
                    moments_h[n_max, n_h] = np.mean(np.power(data_h, h_values[n_h]))

                for n_q in range(nb_q):
                    for n_h in range(nb_h):
                        moments_q_h[n_max, n_q, n_h] = np.mean(
                            np.power(data_q, q_values[n_q])
                            * np.power(data_h, h_values[n_h])
                        )

                for n_l in range(np.intp(n_max - 1), -1, -1):
                    nb_el = data_q.shape[0]
                    data_q = (
                        data_q[0:nb_el:2, :] + data_q[1 : nb_el + 1 : 2, :]
                    ) / 2  # Upscaling of the field
                    data_h = (
                        data_h[0:nb_el:2, :] + data_h[1 : nb_el + 1 : 2, :]
                    ) / 2  # Upscaling of the field
                    for n_q in range(nb_q):
                        moments_q[n_l, n_q] = np.mean(np.power(data_q, q_values[n_q]))
                    for n_h in range(nb_h):
                        moments_h[n_l, n_h] = np.mean(np.power(data_h, h_values[n_h]))
                    for n_q in range(nb_q):
                        for n_h in range(nb_h):
                            moments_q_h[n_l, n_q, n_h] = np.mean(
                                np.power(data_q, q_values[n_q])
                                * np.power(data_h, h_values[n_h])
                            )

    elif file_index == 1:  # nb_ones is simply loaded
        print("Error in TM_corr, file_index equal to 1 is not an option")
        moments = np.load(file_name)

    else:
        print("Error in TM, file_index not equal to 0 or 1")

    # Step 2 : Evaluate r_qh for is the case where a unique scaling regime is considered (defined in l_range)

    x = sp.log(l_list)

    if plot_index > 0:
        plt.figure(plot_index)
        # if the default value of q_values was used, the evaluation of r(q,h) is shown only for pre-defined values
        # to obtain a "good looking graph. Otherwise curves are plotted for all the values in q_values (in that
        # case, less information is displayed in the legend, any all the r2 are outputted of the function)
        if q_values_bis == -1:
            (p1,) = plt.plot(
                x,
                sp.log(moments[:, 1]),
                ls="None",
                marker="+",
                ms=5,
                mew=2,
                mfc="k",
                mec="k",
            )
            (p2,) = plt.plot(
                x,
                sp.log(moments[:, 9]),
                ls="None",
                marker="x",
                ms=5,
                mew=2,
                mfc="k",
                mec="k",
            )
            (p3,) = plt.plot(
                x,
                sp.log(moments[:, 15]),
                ls="None",
                marker="o",
                ms=5,
                mew=2,
                mfc="k",
                mec="k",
            )
            (p4,) = plt.plot(
                x,
                sp.log(moments[:, 19]),
                ls="None",
                marker="d",
                ms=5,
                mew=2,
                mfc="k",
                mec="k",
            )
            (p5,) = plt.plot(
                x,
                sp.log(moments[:, 25]),
                ls="None",
                marker="s",
                ms=5,
                mew=2,
                mfc="k",
                mec="k",
            )
            (p6,) = plt.plot(
                x,
                sp.log(moments[:, 30]),
                ls="None",
                marker="^",
                ms=5,
                mew=2,
                mfc="k",
                mec="k",
            )
            (p7,) = plt.plot(
                x,
                sp.log(moments[:, 35]),
                ls="None",
                marker="v",
                ms=5,
                mew=2,
                mfc="k",
                mec="k",
            )
        label_all = list()

    nb_scale_reg = len(l_range)  # Evaluation of the number of scaling regime

    if nb_scale_reg == 0:
        # A single scaling regime, all the resolutions are considered
        # The only case available for now
        for n_q in range(nb_q):
            for n_h in range(nb_h):
                y = sp.log(
                    moments_q_h[:, n_q, n_h] / (moments_q[:, n_q] * moments_h[:, n_h])
                )
                # print(n_q,n_h)
                # print(y)
                a = sp.polyfit(x, y, 1)
                reg_lin = sp.poly1d(a)
                rqh_1[n_q, n_h] = a[0]
                r2_1[n_q, n_h] = sp.corrcoef(x, y)[0, 1] ** 2
                if plot_index > 0:
                    if q_values_bis == -1:
                        if (
                            n_q == 1
                            or n_q == 9
                            or n_q == 15
                            or n_q == 19
                            or n_q == 25
                            or n_q == 30
                            or n_q == 35
                        ):
                            label_all.append(
                                r"$\mathit{q}\ =\ $"
                                + str(sp.floor(q_values[n_q] * 100) / 100)
                                + " ;  "
                                + r"$\mathit{r}^2\ =\ $"
                                + str(sp.floor(r2_1[n_q] * 100) / 100)
                            )
                            plt.plot(
                                [x[0], x[-1]],
                                [reg_lin(x[0]), reg_lin(x[-1])],
                                lw=2,
                                color="k",
                            )
                        plt.legend(
                            [p1, p2, p3, p4, p5, p6, p7],
                            label_all,
                            loc=2,
                            fontsize=12,
                            frameon=False,
                        )
                    elif q_values_bis == 1:
                        plt.plot(
                            x, y, ls="None", marker="+", ms=5, mew=2, mfc="k", mec="k"
                        )
                        plt.plot(
                            [x[0], x[-1]],
                            [reg_lin(x[0]), reg_lin(x[-1])],
                            lw=2,
                            color="k",
                        )
                rqh_2[n_q, n_h] = np.nan
                r2_2[n_q, n_h] = np.nan
                rqh_3[n_q, n_h] = np.nan
                r2_3[n_q, n_h] = np.nan

    elif nb_scale_reg == 1:
        # A single scaling regime (not yet available)
        i_l_min = np.where(l_list == l_range[0][0])[0][0]
        i_l_max = np.where(l_list == l_range[0][1])[0][0]
        for n_q in range(nb_q):
            # y = sp.log(moments[:,n_q])
            y = sp.log(
                moments_q_h[:, n_q, n_h] / (moments_q[:, n_q] * moments_h[:, n_h])
            )
            a = sp.polyfit(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1], 1)
            reg_lin = sp.poly1d(a)
            # Kq_1[n_q]=a[0]
            rqh_1[n_q, n_h] = a[0]
            r2_1[n_q] = (
                sp.corrcoef(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1])[0, 1]
                ** 2
            )
            if plot_index > 0:
                if q_values_bis == -1:
                    if (
                        n_q == 1
                        or n_q == 9
                        or n_q == 15
                        or n_q == 19
                        or n_q == 25
                        or n_q == 30
                        or n_q == 35
                    ):
                        label_all.append(
                            r"$\mathit{q}\ =\ $"
                            + str(sp.floor(q_values[n_q] * 100) / 100)
                            + " ;  "
                            + r"$\mathit{r}^2\ =\ $"
                            + str(sp.floor(r2_1[n_q] * 100) / 100)
                        )
                        plt.plot(
                            [x[i_l_min], x[i_l_max]],
                            [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                            lw=2,
                            color="k",
                        )
                    plt.legend(
                        [p1, p2, p3, p4, p5, p6, p7],
                        label_all,
                        loc=2,
                        fontsize=12,
                        frameon=False,
                    )
                elif q_values_bis == 1:
                    plt.plot(x, y, ls="None", marker="+", ms=5, mew=2, mfc="k", mec="k")
                    plt.plot(
                        [x[i_l_min], x[i_l_max]],
                        [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                        lw=2,
                        color="k",
                    )

            # For the 2nd scaling regime
            # Kq_2[n_q]=np.nan
            r2_2[n_q] = np.nan
            # For the 3rd scaling regime
            # Kq_3[n_q]=np.nan
            r2_3[n_q] = np.nan

    elif nb_scale_reg == 2:
        # (not yet available)
        for n_q in range(nb_q):
            # y = sp.log(moments[:,n_q])
            y = sp.log(
                moments_q_h[:, n_q, n_h] / (moments_q[:, n_q] * moments_h[:, n_h])
            )
            # For the 1st scaling regime
            i_l_min = np.where(l_list == l_range[0][0])[0][0]
            i_l_max = np.where(l_list == l_range[0][1])[0][0]
            a = sp.polyfit(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1], 1)
            reg_lin = sp.poly1d(a)
            # Kq_1[n_q]=a[0]
            rqh_1[n_q, n_h] = a[0]
            r2_1[n_q] = (
                sp.corrcoef(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1])[0, 1]
                ** 2
            )
            if plot_index > 0:
                if q_values_bis == -1:
                    if (
                        n_q == 1
                        or n_q == 9
                        or n_q == 15
                        or n_q == 19
                        or n_q == 25
                        or n_q == 30
                        or n_q == 35
                    ):
                        label_all.append(
                            r"$\mathit{q}\ =\ $"
                            + str(sp.floor(q_values[n_q] * 100) / 100)
                        )
                        plt.plot(
                            [x[i_l_min], x[i_l_max]],
                            [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                            lw=2,
                            color="b",
                        )
                    plt.legend(
                        [p1, p2, p3, p4, p5, p6, p7],
                        label_all,
                        loc=2,
                        fontsize=12,
                        frameon=False,
                    )
                elif q_values_bis == 1:
                    plt.plot(x, y, ls="None", marker="+", ms=5, mew=2, mfc="k", mec="k")
                    plt.plot(
                        [x[i_l_min], x[i_l_max]],
                        [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                        lw=2,
                        color="b",
                    )

            # For the 2nd scaling regime
            i_l_min = np.where(l_list == l_range[1][0])[0][0]
            i_l_max = np.where(l_list == l_range[1][1])[0][0]
            a = sp.polyfit(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1], 1)
            reg_lin = sp.poly1d(a)
            # Kq_2[n_q]=a[0]
            rqh_2[n_q, n_h] = a[0]
            r2_2[n_q] = (
                sp.corrcoef(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1])[0, 1]
                ** 2
            )
            if plot_index > 0:
                if q_values_bis == -1:
                    if (
                        n_q == 1
                        or n_q == 9
                        or n_q == 15
                        or n_q == 19
                        or n_q == 25
                        or n_q == 30
                        or n_q == 35
                    ):
                        plt.plot(
                            [x[i_l_min], x[i_l_max]],
                            [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                            lw=2,
                            color="r",
                        )
                elif q_values_bis == 1:
                    plt.plot(
                        [x[i_l_min], x[i_l_max]],
                        [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                        lw=2,
                        color="r",
                    )

            # For the 3rd scaling regime
            # Kq_3[n_q]=np.nan
            r2_3[n_q] = np.nan

    elif nb_scale_reg == 3:
        # (not yet available)
        for n_q in range(nb_q):
            # y = sp.log(moments[:,n_q])
            y = sp.log(
                moments_q_h[:, n_q, n_h] / (moments_q[:, n_q] * moments_h[:, n_h])
            )
            # For the 1st scaling regime
            i_l_min = np.where(l_list == l_range[0][0])[0][0]
            i_l_max = np.where(l_list == l_range[0][1])[0][0]
            a = sp.polyfit(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1], 1)
            reg_lin = sp.poly1d(a)
            # Kq_1[n_q]=a[0]
            rqh_1[n_q, n_h] = a[0]
            r2_1[n_q] = (
                sp.corrcoef(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1])[0, 1]
                ** 2
            )
            if plot_index > 0:
                if q_values_bis == -1:
                    if (
                        n_q == 1
                        or n_q == 9
                        or n_q == 15
                        or n_q == 19
                        or n_q == 25
                        or n_q == 30
                        or n_q == 35
                    ):
                        label_all.append(
                            r"$\mathit{q}\ =\ $"
                            + str(sp.floor(q_values[n_q] * 100) / 100)
                        )
                        plt.plot(
                            [x[i_l_min], x[i_l_max]],
                            [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                            lw=2,
                            color="b",
                        )
                    plt.legend(
                        [p1, p2, p3, p4, p5, p6, p7],
                        label_all,
                        loc=2,
                        fontsize=12,
                        frameon=False,
                    )
                elif q_values_bis == 1:
                    plt.plot(x, y, ls="None", marker="+", ms=5, mew=2, mfc="k", mec="k")
                    plt.plot(
                        [x[i_l_min], x[i_l_max]],
                        [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                        lw=2,
                        color="b",
                    )

            # For the 2nd scaling regime
            i_l_min = np.where(l_list == l_range[1][0])[0][0]
            i_l_max = np.where(l_list == l_range[1][1])[0][0]
            a = sp.polyfit(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1], 1)
            reg_lin = sp.poly1d(a)
            # Kq_2[n_q]=a[0]
            rqh_2[n_q, n_h] = a[0]
            r2_2[n_q] = (
                sp.corrcoef(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1])[0, 1]
                ** 2
            )
            if plot_index > 0:
                if q_values_bis == -1:
                    if (
                        n_q == 1
                        or n_q == 9
                        or n_q == 15
                        or n_q == 19
                        or n_q == 25
                        or n_q == 30
                        or n_q == 35
                    ):
                        plt.plot(
                            [x[i_l_min], x[i_l_max]],
                            [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                            lw=2,
                            color="r",
                        )
                elif q_values_bis == 1:
                    plt.plot(
                        [x[i_l_min], x[i_l_max]],
                        [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                        lw=2,
                        color="r",
                    )

            # For the 3rd scaling regime
            i_l_min = np.where(l_list == l_range[2][0])[0][0]
            i_l_max = np.where(l_list == l_range[2][1])[0][0]
            a = sp.polyfit(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1], 1)
            reg_lin = sp.poly1d(a)
            # Kq_3[n_q]=a[0]
            rqh_3[n_q, n_h] = a[0]
            r2_3[n_q] = (
                sp.corrcoef(x[i_l_min : i_l_max + 1], y[i_l_min : i_l_max + 1])[0, 1]
                ** 2
            )
            if plot_index > 0:
                if q_values_bis == -1:
                    if (
                        n_q == 1
                        or n_q == 9
                        or n_q == 15
                        or n_q == 19
                        or n_q == 25
                        or n_q == 30
                        or n_q == 35
                    ):
                        plt.plot(
                            [x[i_l_min], x[i_l_max]],
                            [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                            lw=2,
                            color="g",
                        )
                elif q_values_bis == 1:
                    plt.plot(
                        [x[i_l_min], x[i_l_max]],
                        [reg_lin(x[i_l_min]), reg_lin(x[i_l_max])],
                        lw=2,
                        color="g",
                    )

    else:
        print("Error in TM, l_range has wrong size")

    if plot_index > 0:
        plt.xlabel(r"$\log(\lambda)$", fontsize=20, color="k")
        plt.ylabel(r"$\log(JTM_\lambda)$", fontsize=20, color="k")
        plt.title("TM Analysis", fontsize=20, color="k")
        ax = plt.gca()
        for xtick in ax.get_xticklabels():
            plt.setp(xtick, fontsize=14)
        for ytick in ax.get_yticklabels():
            plt.setp(ytick, fontsize=14)
        plt.savefig("evaluation_K_q_in_TM.png")

    return rqh_1, rqh_2, rqh_3, r2_1, r2_2, r2_3


def levy_speed(alpha, size):
    # It returns an array of size "size" with Levy variables
    # (works in 1, 2 and 3D)

    from scipy.stats import uniform, expon

    phi = uniform.rvs(loc=-sp.pi / 2, scale=sp.pi, size=size)
    W = expon.rvs(size=size)
    # L=np.zeros((N,1))
    if alpha != 1:
        phi0 = (sp.pi / 2) * (1 - np.abs(1 - alpha)) / alpha
        L = (
            sp.sign(1 - alpha)
            * (sp.sin(alpha * (phi - phi0)))
            * (((sp.cos(phi - alpha * (phi - phi0))) / W) ** ((1 - alpha) / alpha))
            / ((sp.cos(phi)) ** (1 / alpha))
        )
    else:
        print("Error : alpha = " + str(alpha) + " in Levy")

    return L


def generation_list_increments_sharp_standard(n, alpha, C1, dim):
    list_all_increments_sharp_standard = list()
    if dim == 1:
        for t in range(n):
            size = 2 ** (t + 1)
            L = levy_speed(alpha, (size,))
            mueps_all_t = sp.exp(
                L * (C1 * sp.log(2) / abs(alpha - 1)) ** (1 / alpha)
            ) / (2 ** (C1 / (alpha - 1)))
            list_all_increments_sharp_standard.append(mueps_all_t)
    elif dim == 2:
        for t in range(n):
            size = 2 ** (t + 1)
            L = levy_speed(alpha, (size, size))
            mueps_all_t = sp.exp(
                L * (C1 * sp.log(2) / abs(alpha - 1)) ** (1 / alpha)
            ) / (2 ** (C1 / (alpha - 1)))
            list_all_increments_sharp_standard.append(mueps_all_t)
    else:
        print('Wong dim in "generation_list_increments_sharp"')
    return list_all_increments_sharp_standard


def generation_list_gamma_incr_large(n, list_all_increments, dim):
    list_gamma_incr_large = list()
    if dim == 1:
        for t in range(n):
            # print(list_all_increments[t].shape)
            N = np.intp(2**n / list_all_increments[t].shape[0])
            # print(list_all_increments[t].shape,N)
            buf = np.repeat(np.log(list_all_increments[t]) / np.log(2), N, axis=0)
            buf[buf < -100] = -100
            list_gamma_incr_large.append(buf)

    elif dim == 2:
        for t in range(n):
            N = np.intp(2**n / list_all_increments[t].shape[0])
            list_gamma_incr_large.append(
                np.repeat(
                    np.repeat(np.log(list_all_increments[t]) / np.log(2), N, axis=0),
                    N,
                    axis=1,
                )
            )
    else:
        print("wrong dim")

    return list_gamma_incr_large

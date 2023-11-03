# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:02:20 2013

@author: gires
"""



############################################################################################
############################################################################################
##
## Author : Auguste GIRES (2017)      (auguste.gires@enpc.fr)
##
## This script is part of the toolbox developed for handling the data obtained with the help of 3 disdrometers which are in
## the multi-scale rainfall observatory currenlty under developement at Ecole des Ponts ParisTech.
##
## A data paper, by Gires A., Tchiguirinskaia I. and Schertzer D. entitled "Two months of disdrometer data in the Paris area" 
## presenting in details this data base was submitted to https://www.earth-system-science-data.net. User should cite this 
## paper as well as this data base.
##
############################################################################################
############################################################################################

import numpy as np
import scipy as sp  
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle 
#import matplotlib.colors as colors
#import matplotlib.cm as cm
from scipy.stats import uniform
#from datetime import datetime, timedelta

# Functions for generating all the samples
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches

import os as os
import subprocess as subprocess
import shutil as shutil

from datetime import datetime, timedelta, date

from time import sleep

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import matplotlib.image as mpimg




def V_D_Lhermitte_1988 (D):
    D=D*10**(-3);    
    V=9.25-9.25*np.exp(-(68000*D**2+488*D))
    return V



def Quicklook_and_R_series_generation_Carnot_1(start_evt,end_evt,path_daily_data_python,path_outputs):
############################################################################################
# Aim : generating a quicklook image and the corresponding rainfall 30s and 5min rain rate time series for 
#            a given rainfall event for the Carnot_1 measurement campaign
# Inputs: 
#    - The start and end date of the studied event (datetime objects; should be with 0s and 30s)
#    - path_daily_data_python (a string): Path to the folder where inputs are stored (assumed to contain folders Pars1, Pars2 and PWS 
#      containing the data for the disdrometers taking part to the this campaign)
#    - path_outputs (a string): path the folder where outputs (.png image, 30 s time series and 5 min time series will be stored)
#         The names of the three output files include explicitely the strat and and end of event
#                  Quicklook_Carnot_1_2017_05_18_16_00_00__2017_05_19_01_00_00.png
#                  R_5_min_Carnot_1_2017_05_18_16_00_00__2017_05_19_01_00_00.csv
#                        (1 line per time step, missing data are noted as "nan")
#                  R_30_sec_Carnot_1_2017_05_18_16_00_00__2017_05_19_01_00_00.csv
#                        (1 line per time step, missing data are noted as "nan")
##############################################################################################    

    # String defining the events for file names
    str_evt=start_evt.strftime('%Y_%m_%d_%H_%M_%S')+'__'+end_evt.strftime('%Y_%m_%d_%H_%M_%S')
    print(str_evt)
    
    time_step=timedelta(0,30)
    
    t=start_evt
    N=0
    i_correction_PWS=1   # This enables the correction of the data from the PWS to better take into account the drop oblateness

    
    while t<=end_evt:
        #print(str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second), N)
        t=t+time_step
        N=N+1
    
    Nb_time_steps=N
    print('Number of time steps = ',Nb_time_steps)
    
    
    #
    ##########
    ## Retrieving the data
    ##########
    #
    
    # Using the python function "extracting_one_event"
    PWS_one_event,Pars1_one_event,Pars2_one_event = extracting_one_event_Carnot_1(start_evt,end_evt,path_daily_data_python)
    
    # Definition of the data for the PWS
    T_PWS = PWS_one_event[0]
    R_disdro_PWS = PWS_one_event[1]
    R_emp_1_PWS = PWS_one_event[2]
    R_emp_2_PWS = PWS_one_event[3]
    Nb_all_drops_PWS = PWS_one_event[4]
    N_D_emp_PWS = PWS_one_event[5]
    rho_PWS = PWS_one_event[6]
    Nt_PWS = PWS_one_event[7]
    Dm_PWS = PWS_one_event[8]
    Zh_PWS = PWS_one_event[9]
    Zv_PWS = PWS_one_event[10]
    Kdp_PWS = PWS_one_event[11]
    Rela_humi_PWS = PWS_one_event[12]
    
    # Definition of the data for the PWS
    T_Pars1 = Pars1_one_event[0]
    R_disdro_Pars1 = Pars1_one_event[1]
    R_emp_1_Pars1 = Pars1_one_event[2]
    R_emp_2_Pars1 = Pars1_one_event[3]
    Nb_all_drops_Pars1 = Pars1_one_event[4]
    N_D_emp_Pars1 = Pars1_one_event[5]
    rho_Pars1 = Pars1_one_event[6]
    Nt_Pars1 = Pars1_one_event[7]
    Dm_Pars1 = Pars1_one_event[8]
    Zh_Pars1 = Pars1_one_event[9]
    Zv_Pars1 = Pars1_one_event[10]
    Kdp_Pars1 = Pars1_one_event[11]
    
    
    print('Nb_all_drops_Pars1.shape = ',Nb_all_drops_Pars1.shape)    
    
    # Definition of the data for the PWS
    T_Pars2 = Pars2_one_event[0]
    R_disdro_Pars2 = Pars2_one_event[1]
    R_emp_1_Pars2 = Pars2_one_event[2]
    R_emp_2_Pars2 = Pars2_one_event[3]
    Nb_all_drops_Pars2 = Pars2_one_event[4]
    N_D_emp_Pars2 = Pars2_one_event[5]
    rho_Pars2 = Pars2_one_event[6]
    Nt_Pars2 = Pars2_one_event[7]
    Dm_Pars2 = Pars2_one_event[8]
    Zh_Pars2 = Pars2_one_event[9]
    Zv_Pars2 = Pars2_one_event[10]
    Kdp_Pars2 = Pars2_one_event[11]
    
    
    # For the PWS
    V_PWS = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12.0,13.6,15.2,17.6,20.8,24.0,27.2))
    V_width_PWS=np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,3.2,3.2))
    D_PWS = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12.0,13.6,15.2,17.6,20.8,24.0,27.2))
    D_width_PWS = np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,3.2,3.2))
    V_theo_PWS = V_D_Lhermitte_1988 (D_PWS)
    S_eff_PWS = 0.004*np.ones(D_PWS.shape)

    # If a the correction for oblateness is implemented, D_PWS and D_width_PWS are modified
    if i_correction_PWS==1:
        D_PWS_buf=np.zeros(D_PWS.shape)
        D_width_PWS_buf=np.zeros(D_PWS.shape)

        coeff=np.array((-0.01361193,0.12089066,0.87497141,0.00989984)) # With Pars and n=1.33
        #coeff=np.array((-0.03302464,0.28672044,0.43507802,0.33835173)) # With Ands and n=1.33
        #coeff=np.array((-0.02259775,0.19820372,0.6685606,0.17391049)) # With Beard and n=1.33
    
        for i in range(34):
            if D_PWS[i]>1:
                D_meas=D_PWS[i]**(1/1.25)
                D_PWS_buf[i]=coeff[0]*D_meas**3+coeff[1]*D_meas**2+coeff[2]*D_meas+coeff[3]
            else : 
                D_PWS_buf[i]=D_PWS[i]

        for i in range(34):
            D_min=D_PWS[i]-D_width_PWS[i]/2
            if D_min > 1:
                D_min=D_min**(1/1.25)
                D_min_buf=coeff[0]*D_min**3+coeff[1]*D_min**2+coeff[2]*D_min+coeff[3]
            else:
                D_min_buf=D_min
            D_max=D_PWS[i]+D_width_PWS[i]/2
            if D_max > 1:
                D_max=D_max**(1/1.25)
                D_max_buf=coeff[0]*D_max**3+coeff[1]*D_max**2+coeff[2]*D_max+coeff[3]
            else:
                D_max_buf=D_max
            D_width_PWS_buf[i]=D_max_buf-D_min_buf
      
        
        D_PWS=D_PWS_buf
        D_width_PWS=D_width_PWS_buf
        V_theo_PWS = V_D_Lhermitte_1988 (D_PWS)
    
    # For the Pars1
    V_Pars1 = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8))
    V_width_Pars1=np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2))  
    D_Pars1=np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.750,3.250,3.750,4.250,4.750,5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5])
    D_width_Pars1=np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.250,0.250,0.250,0.250,0.250,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,2,3,3])
    V_Pars1 = V_Pars1
    V_width_Pars1 = V_width_Pars1
    D_Pars1 = D_Pars1
    D_width_Pars1 = D_width_Pars1
    V_theo_Pars1 = V_D_Lhermitte_1988 (D_Pars1)
    S_eff_Pars1 = 10**(-6)*180*(30-D_Pars1/2)
    
    # For the Pars2
    V_Pars2 = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8))
    V_width_Pars2=np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2))  
    D_Pars2=np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.750,3.250,3.750,4.250,4.750,5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5])
    D_width_Pars2=np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.250,0.250,0.250,0.250,0.250,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,2,3,3])
    V_Pars2 = V_Pars2
    V_width_Pars2 = V_width_Pars2
    D_Pars2 = D_Pars2
    D_width_Pars2 = D_width_Pars2
    V_theo_Pars2 = V_D_Lhermitte_1988 (D_Pars2)
    S_eff_Pars2 = 10**(-6)*180*(30-D_Pars2/2)
    
    
    ######
    # Displaying the data
    ######
    
    time = np.array(range(N))*30/3600
    
    ####
    # Plot with some comparison with the three disdrometers
    ####
    
    ## Missing data

    plt.figure(0)
    plt.clf()
    
    ax_PWS = plt.axes([0.1,0.44,0.8,0.1])    
    #plt.ylabel(r'$No\/Data$',fontsize=20,color='k')
    plt.title('PWS 100 : '+str(np.count_nonzero(np.isnan(R_emp_2_PWS))) + ' missing time steps',fontsize=20,color='k')  
      
    No_data_PWS=np.ones((Nb_time_steps))
    for i in range(Nb_time_steps):
        if R_emp_2_PWS[i]>=0 :
            No_data_PWS[i]=0
    plt.bar(time,No_data_PWS,width=30./3600,color='r',edgecolor='r')    
      
    ax_PWS.set_xlim(0,time[-1])
    ax_PWS.set_ylim(0,1)
    for xtick in ax_PWS.get_xticklabels():
        plt.setp(xtick,fontsize=0)
    for ytick in ax_PWS.get_yticklabels():
        plt.setp(ytick,fontsize=0)  
    
    
    ax_Pars1 = plt.axes([0.1,0.27,0.8,0.1])#fig.add_subplot(312)
    plt.ylabel(r'$No\/Data$',fontsize=20,color='k')
    plt.title('Parsivel #1 :'+str(np.count_nonzero(np.isnan(R_emp_2_Pars1))) + ' missing time steps',fontsize=20,color='k')  

    No_data_Pars1=np.ones((Nb_time_steps))
    for i in range(Nb_time_steps):
        if R_emp_2_Pars1[i]>=0 :
            No_data_Pars1[i]=0
    plt.bar(time,No_data_Pars1,width=30./3600,color='b',edgecolor='b')  
  
    ax_Pars1.set_xlim(0,time[-1])
    ax_Pars1.set_ylim(0,1)
    for xtick in ax_Pars1.get_xticklabels():
        plt.setp(xtick,fontsize=0)
    for ytick in ax_Pars1.get_yticklabels():
        plt.setp(ytick,fontsize=0)  
    
    
    ax_Pars2 = plt.axes([0.1,0.1,0.8,0.1])#fig.add_subplot(312)
    plt.xlabel(r'$Time\/(h)$',fontsize=20,color='k')
    #plt.ylabel(r'$No\/Data$',fontsize=20,color='k')
    plt.title('Parsivel #2: '+str(np.count_nonzero(np.isnan(R_emp_2_Pars2))) + ' missing time steps',fontsize=20,color='k')  
 
    No_data_Pars2=np.ones((Nb_time_steps))
    for i in range(Nb_time_steps):
        if R_emp_2_Pars2[i]>=0 :
            No_data_Pars2[i]=0
    plt.bar(time,No_data_Pars2,width=30./3600,color='g',edgecolor='g')  
  
    ax_Pars2.set_xlim(0,time[-1])
    ax_Pars2.set_ylim(0,1)
    for xtick in ax_Pars2.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax_Pars2.get_yticklabels():
        plt.setp(ytick,fontsize=0)  

    plt.savefig(path_outputs+'no_data_'+str_evt+'.png',dpi=70,bbox_inches='tight',format='png')
    plt.clf()
    plt.close()
    img_no_data=mpimg.imread(path_outputs+'no_data_'+str_evt+'.png')



    ## Rain rate
    plt.figure(1)
    plt.clf()
    p1,=plt.plot(time,R_emp_2_PWS,color='r',lw=2)   # WARNING : check whether there is a need for a correction as for the PWS
    p2,=plt.plot(time,R_emp_2_Pars1,color='b',lw=2)   # R_disdro_pars1 = R_emp_2_Pars1
    p3,=plt.plot(time,R_emp_2_Pars2,color='g',lw=2)   # R_disdro_pars2 = R_emp_2_Pars2
    plt.legend([p1,p2,p3],['PWS','Parsivel #1','Parsivel #2'],loc='upper right',frameon=False)
    plt.xlabel(r'$Time\/(h)$',fontsize=20,color='k')
    plt.ylabel(r'$\mathit{R}\  (mm.h^{-1})$',fontsize=20,color='k')
    plt.title('',fontsize=20,color='k')  
    plt.xlim(0,time[-1])
    ax=plt.gca()
    for xtick in ax.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax.get_yticklabels():
        plt.setp(ytick,fontsize=14)  
    plt.savefig(path_outputs+'temporal_rain_rate_'+str_evt+'.png',dpi=70,bbox_inches='tight',format='png')
    plt.close()
    img_R=mpimg.imread(path_outputs+'temporal_rain_rate_'+str_evt+'.png')
    
    
    ## Cumulative rain rate
    
    Cumul_PWS=np.zeros(R_emp_2_PWS.shape)
    Cumul_Pars1=np.zeros(R_emp_2_Pars1.shape)
    Cumul_Pars2=np.zeros(R_emp_2_Pars2.shape)    

    for t in range(0,Cumul_PWS.shape[0]):
        Cumul_PWS[t]=np.nansum(R_emp_2_PWS[0:(t+1)])/120
        Cumul_Pars1[t]=np.nansum(R_emp_2_Pars1[0:(t+1)])/120
        Cumul_Pars2[t]=np.nansum(R_emp_2_Pars2[0:(t+1)])/120

    
    plt.figure(2)
    plt.clf()
    p1,=plt.plot(time,Cumul_PWS,color='r',lw=2,label='PWS')   # WARNING : check whether there is a need for a correction as for the PWS
    p2,=plt.plot(time,Cumul_Pars1,color='b',lw=2,label='Parsivel #1')   # R_disdro_pars1 = R_emp_2_Pars1
    p3,=plt.plot(time,Cumul_Pars2,color='g',lw=2,label='Parsivel #2')   # R_disdro_pars2 = R_emp_2_Pars2
    
    plt.legend(loc='upper left',frameon=False)
    plt.xlabel(r'$Time\/(h)$',fontsize=20,color='k')
    plt.ylabel(r'$\mathit{Cumulative\/rainfall\/depth}\  (mm)$',fontsize=20,color='k')
    plt.title('',fontsize=20,color='k')  
    plt.xlim(0,time[-1])
    ax=plt.gca()
    for xtick in ax.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax.get_yticklabels():
        plt.setp(ytick,fontsize=14)  
    plt.savefig(path_outputs+'temporal_cumul_depth_'+str_evt+'.png',dpi=70,bbox_inches='tight',format='png')
    plt.clf()
    plt.close()
    img_C=mpimg.imread(path_outputs+'temporal_cumul_depth_'+str_evt+'.png')
       
    
    ## Temperature
    plt.figure(3)
    plt.clf()
    p1,=plt.plot(time,T_PWS,color='r',lw=2)   # WARNING : check whether there is a need for a correction as for the PWS
    p2,=plt.plot(time,T_Pars1,color='b',lw=2)   # R_disdro_pars1 = R_emp_2_Pars1
    p3,=plt.plot(time,T_Pars2,color='g',lw=2)   # R_disdro_pars2 = R_emp_2_Pars2
    plt.legend([p1,p2,p3],['PWS','Parsivel #1','Parsivel #2'],loc='upper right',frameon=False)    
    plt.xlabel(r'$Time\/(h)$',fontsize=20,color='k')
    plt.ylabel(r'$T \/ (^{\circ}C)$',fontsize=20,color='k')
    plt.title('',fontsize=20,color='k')  
    plt.xlim(0,time[-1])
    ax=plt.gca()
    for xtick in ax.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax.get_yticklabels():
        plt.setp(ytick,fontsize=14)  
    plt.savefig(path_outputs+'temporal_temperature_'+str_evt+'.png',dpi=70,bbox_inches='tight',format='png')  
    plt.close()
    
    img_T=mpimg.imread(path_outputs+'temporal_temperature_'+str_evt+'.png')    
    
    
    ## Rho rate
    plt.figure(4)
    plt.clf()
    p1,=plt.plot(time,rho_PWS,color='r',lw=2)   # WARNING : check whether there is a need for a correction as for the PWS
    p2,=plt.plot(time,rho_Pars1,color='b',lw=2)   # R_disdro_pars1 = R_emp_2_Pars1
    p3,=plt.plot(time,rho_Pars2,color='g',lw=2)   # R_disdro_pars1 = R_emp_2_Pars1    
    plt.legend([p1,p2,p3],['PWS','Parsivel #1','Parsivel #2'],loc='upper right',frameon=False)
    plt.xlabel(r'$Time\/(h)$',fontsize=20,color='k')
    plt.ylabel(r'$\rho\/\/(g.m^{-3})$',fontsize=20,color='k')
    plt.title('',fontsize=20,color='k')  
    plt.xlim(0,time[-1])
    ax=plt.gca()
    for xtick in ax.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax.get_yticklabels():
        plt.setp(ytick,fontsize=14)  
    plt.savefig(path_outputs+'temporal_evo_rho_'+str_evt+'.png',dpi=70,bbox_inches='tight',format='png')    
    plt.clf()
    plt.close()
    
    #
    # N_D_emp
    #
    
    plt.figure(5)
    plt.clf()
    p1,=plt.plot(D_PWS[0:29],np.nanmean(N_D_emp_PWS,axis=0)[0:29],color='r',lw=2)
    p2,=plt.plot(D_Pars1[0:26],np.nanmean(N_D_emp_Pars1,axis=0)[0:26],color='b',lw=2)
    p3,=plt.plot(D_Pars2[0:26],np.nanmean(N_D_emp_Pars2,axis=0)[0:26],color='g',lw=2)
    plt.legend([p1,p2,p3],['PWS','Parsivel #1','Parsivel #2'],loc='upper right',frameon=False)
    plt.xlabel(r'$\mathit{D}\/\/(mm)$',fontsize=20,color='k')
    plt.ylabel(r'$\mathit{N(D)}\  (m^{-3}.m^{-1})$',fontsize=20,color='k')
    plt.title('',fontsize=20,color='k')  
    ax=plt.gca()
    for xtick in ax.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax.get_yticklabels():
        plt.setp(ytick,fontsize=14)  
    ax.set_xlim(0,6)
    plt.savefig(path_outputs+'N_D_'+str_evt+'.png',dpi=70,bbox_inches='tight',format='png')
    plt.clf()
    plt.close()
    
    #
    # N_D_emp *Dh3
    #
    
    plt.figure(6)
    plt.clf()
    p1,=plt.plot(D_PWS[0:29],np.nanmean(N_D_emp_PWS,axis=0)[0:29]*D_PWS[0:29]**3,color='r',lw=2)
    p2,=plt.plot(D_Pars1[0:26],np.nanmean(N_D_emp_Pars1,axis=0)[0:26]*D_Pars1[0:26]**3,color='b',lw=2)
    p3,=plt.plot(D_Pars2[0:26],np.nanmean(N_D_emp_Pars2,axis=0)[0:26]*D_Pars2[0:26]**3,color='g',lw=2)
    plt.legend([p1,p2,p3],['PWS','Parsivel #1','Parsivel #2'],loc='upper right',frameon=False)
    plt.xlabel(r'$\mathit{D}\/\/(mm)$',fontsize=20,color='k')
    plt.ylabel(r'$\mathit{N(D)}\/\mathit{D}^3$',fontsize=20,color='k')
    plt.title('',fontsize=20,color='k')  
    ax=plt.gca()
    for xtick in ax.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax.get_yticklabels():
        plt.setp(ytick,fontsize=14)  
    ax.set_xlim(0,9)
    plt.savefig(path_outputs+'N_D_D3_'+str_evt+'.png',dpi=70,bbox_inches='tight',format='png')    
    plt.clf()
    plt.close()
    img_ND3=mpimg.imread(path_outputs+'N_D_D3_'+str_evt+'.png')

    #
    # Temporal evolution of N(D)
    #

    N=32
    list_colors=[]
    jet =  plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=N-1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    
    for i in range(0,N):
        list_colors.append(scalarMap.to_rgba(i))
        #print(colorVal)

    max_log10_N_D_emp_PWS=np.log10(np.nanmax(N_D_emp_PWS))
    max_log10_N_D_emp_Pars1=np.log10(np.nanmax(N_D_emp_Pars1))
    max_log10_N_D_emp_Pars2=np.log10(np.nanmax(N_D_emp_Pars2))
    max_value=np.nanmax([max_log10_N_D_emp_PWS,max_log10_N_D_emp_Pars1,max_log10_N_D_emp_Pars2])

    
    fig = plt.figure(7)
    plt.clf()
    
    ax_PWS = plt.axes([0.1,0.7,0.72,0.23])    
    
    try:
        codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]                
        for t in range(Nb_time_steps-1) : # A rojouter après 
            for i in range(29) :  #
                if N_D_emp_PWS[t,i]>0 :
                    verts = [(time[t], D_PWS[i]), (time[t],D_PWS[i+1]), (time[t+1], D_PWS[i+1]), (time[t+1], D_PWS[i]), (time[t], D_PWS[i]),]           
                    ind_color=np.floor(32*(np.log10(N_D_emp_PWS[t,i]-0.0001))/max_value)   
                    ind_color=ind_color.astype(int)           
                    if ind_color==-1:
                        ind_color=0
                    path = Path(verts, codes)
                    patch = patches.PathPatch(path, facecolor=list_colors[ind_color], lw=0)
                    ax_PWS.add_patch(patch)
    except: 
        print('souci')
    #plt.xlabel('Time (h)',fontsize=20,color='k')
   
    plt.ylabel(r'$\mathit{D}\/\/(mm)$',fontsize=20,color='k')
    plt.title('PWS 100',fontsize=20,color='k')  
    #ax=plt.gca()
    plt.xlim(0,time[-1]) 
    for xtick in ax_PWS.get_xticklabels():
        plt.setp(xtick,fontsize=0)
    for ytick in ax_PWS.get_yticklabels():
        plt.setp(ytick,fontsize=12)  
    ax_PWS.set_xlim(0,time[-1])
    ax_PWS.set_ylim(0,9)
    
    ax_Pars1 = plt.axes([0.1,0.4,0.72,0.23])#fig.add_subplot(312)
    try:    
        codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]            
        for t in range(Nb_time_steps-1) : # A rojouter après 
            for i in range(26) :  #
                if N_D_emp_Pars1[t,i]>0 :
                    verts = [(time[t], D_Pars1[i]), (time[t],D_Pars1[i+1]), (time[t+1], D_Pars1[i+1]), (time[t+1], D_Pars1[i]), (time[t], D_Pars1[i]),]           
                    ind_color=np.floor(32*(np.log10(N_D_emp_Pars1[t,i]-0.0001))/max_value)   
                    ind_color=ind_color.astype(int)          
                    if ind_color==-1:
                        ind_color=0
                    path = Path(verts, codes)
                    patch = patches.PathPatch(path, facecolor=list_colors[ind_color], lw=0)
                    ax_Pars1.add_patch(patch)
    except: 
        print('souci')
    #plt.xlabel(r'$Time\/(h)$',fontsize=20,color='k')
    plt.ylabel(r'$\mathit{D}\/\/(mm)$',fontsize=20,color='k')
    plt.title('Parsivel #1',fontsize=20,color='k')  
    plt.xlim(0,time[-1])
    for xtick in ax_Pars1.get_xticklabels():
        plt.setp(xtick,fontsize=0)
    for ytick in ax_Pars1.get_yticklabels():
        plt.setp(ytick,fontsize=12)  
    ax_Pars1.set_xlim(0,time[-1])
    ax_Pars1.set_ylim(0,9)

    ax_Pars2 = plt.axes([0.1,0.1,0.72,0.23])#fig.add_subplot(312)
    try:
        codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]            
        for t in range(Nb_time_steps-1) : # A rojouter après 
            for i in range(26) :  #
                if N_D_emp_Pars2[t,i]>0 :
                    verts = [(time[t], D_Pars2[i]), (time[t],D_Pars2[i+1]), (time[t+1], D_Pars2[i+1]), (time[t+1], D_Pars2[i]), (time[t], D_Pars2[i]),]           
                    ind_color=np.floor(32*(np.log10(N_D_emp_Pars2[t,i]-0.0001))/max_value)   
                    ind_color=ind_color.astype(int)          
                    if ind_color==-1:
                        ind_color=0
                    path = Path(verts, codes)
                    patch = patches.PathPatch(path, facecolor=list_colors[ind_color], lw=0)
                    ax_Pars2.add_patch(patch)
    except: 
        print('souci')
        
    plt.xlabel(r'$Time\/(h)$',fontsize=20,color='k')
    plt.ylabel(r'$\mathit{D}\/\/(mm)$',fontsize=20,color='k')
    plt.title('Parsivel #2',fontsize=20,color='k')  
    plt.xlim(0,time[-1])
    for xtick in ax_Pars2.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax_Pars2.get_yticklabels():
        plt.setp(ytick,fontsize=12)  
    ax_Pars2.set_xlim(0,time[-1])
    ax_Pars2.set_ylim(0,9)

    ax_colorbar = plt.axes([0.83,0.1,0.05,0.83]) 
    cmap = colors.ListedColormap(list_colors)
    #cmap.set_over((1., 0., 0.))
    #cmap.set_under((0., 0., 1.))
    try:
        bounds = range(33)*max_value/32 # [-1., -.5, 0., .5, 1.]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        #cax, kw = colorbar.make_axes(ax_Pars1)
        cb3 = colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm,boundaries=bounds, extendfrac='auto', ticks=bounds[0:-1:2], spacing='uniform', orientation='vertical')
        cb3.set_label(r'$log_{10}\/\mathit{N(D)}$',fontsize=20)
        cb3.set_ticklabels(np.floor(bounds[0:-1:2]*100)/100)
    except: 
        print('souci')
        
    plt.savefig(path_outputs+'temporal_evo_N_D_'+str_evt+'.png',dpi=70,bbox_inches='tight',format='png')
    #plt.savefig(path_outputs+'Temporal_evo_N_D_'+str_evt+'.png',dpi=70,bbox_inches='tight',format='png')
    plt.clf()
    plt.close()    
    
    img_DSD=mpimg.imread(path_outputs+'temporal_evo_N_D_'+str_evt+'.png')
   
   
    #####
    # Displaying the size / velocity map
    #####

    plt.figure(8)
    plt.clf()


    log10_all_drops_2_PWS=np.zeros((34,34))
    for i in range(29):
        for j in range(29):
            log10_all_drops_2_PWS[i,j]=np.log10(Nb_all_drops_PWS[i,j]/(D_width_PWS[i]*V_width_PWS[j]))

    log10_all_drops_2_Pars1=np.zeros((32,32))
    log10_all_drops_2_Pars2=np.zeros((32,32))
    for i in range(26):
        for j in range(30):
            log10_all_drops_2_Pars1[i,j]=np.log10(Nb_all_drops_Pars1[i,j]/(D_width_Pars1[i]*V_width_Pars1[j]))
            log10_all_drops_2_Pars2[i,j]=np.log10(Nb_all_drops_Pars2[i,j]/(D_width_Pars2[i]*V_width_Pars2[j]))
    max_log10_all_drops_2_PWS=np.max(log10_all_drops_2_PWS)
    max_log10_all_drops_2_Pars1=np.max(log10_all_drops_2_Pars1)
    max_log10_all_drops_2_Pars2=np.max(log10_all_drops_2_Pars2)
    max_value=np.max([max_log10_all_drops_2_PWS,max_log10_all_drops_2_Pars1,max_log10_all_drops_2_Pars2])    
    
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]            
    
    ax_PWS = plt.axes([0.11,0.1,0.18,0.24])    
    for i in range(29) : # A rojouter après
        for j in range(29) :
            if log10_all_drops_2_PWS[i,j]>0 :
                verts = [(D_PWS[i], V_PWS[j]), (D_PWS[i], V_PWS[j+1]), (D_PWS[i+1], V_PWS[j+1]), (D_PWS[i+1], V_PWS[j]), (D_PWS[i], V_PWS[j]),]
                ind_color=np.floor(32*(log10_all_drops_2_PWS[i,j]-0.00001)/max_value)    
                ind_color=ind_color.astype(int)
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor=list_colors[ind_color], lw=0)
                ax_PWS.add_patch(patch)
    plt.plot(D_PWS[0:29],V_theo_PWS[0:29],color='k')
    plt.xlabel(r'$\mathit{D}\/\/(mm)$',fontsize=14,color='k')
    plt.ylabel(r'$\mathit{V}\/\/(m.s^{-1})$',fontsize=14,color='k')
    plt.title('PWS',fontsize=14,color='k')  
    #ax=plt.gca()
    for xtick in ax_PWS.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax_PWS.get_yticklabels():
        plt.setp(ytick,fontsize=14)  
    ax_PWS.set_xlim(0,9)
    ax_PWS.set_ylim(0,12)
        
    ax_Pars1 = plt.axes([0.34,0.1,0.18,0.24])  
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]            
    for i in range(26) : # A rojouter après
        for j in range(30) :
            if log10_all_drops_2_Pars1[i,j]>0 :
                verts = [(D_Pars1[i], V_Pars1[j]), (D_Pars1[i], V_Pars1[j+1]), (D_Pars1[i+1], V_Pars1[j+1]), (D_Pars1[i+1], V_Pars1[j]), (D_Pars1[i], V_Pars1[j]),]
                ind_color=np.floor(32*(log10_all_drops_2_Pars1[i,j]-0.00001)/max_value)    
                ind_color=ind_color.astype(int)
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor=list_colors[ind_color], lw=0)
                ax_Pars1.add_patch(patch)
    plt.plot(D_Pars1,V_theo_Pars1,color='k')
    plt.xlabel(r'$\mathit{D}\/\/(mm)$',fontsize=14,color='k')
    #plt.ylabel(r'$\mathit{V}\/\/(m.s^{-1})$',fontsize=20,color='k')
    plt.title('Parsivel #1',fontsize=14,color='k')  
    for xtick in ax_Pars1.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax_Pars1.get_yticklabels():
        plt.setp(ytick,fontsize=14)  
    ax_Pars1.set_xlim(0,9)
    ax_Pars1.set_ylim(0,12)

    ax_Pars2 = plt.axes([0.57,0.1,0.18,0.24])  
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]            
    for i in range(26) : # A rojouter après
        for j in range(30) :
            if log10_all_drops_2_Pars2[i,j]>0 :
                verts = [(D_Pars2[i], V_Pars2[j]), (D_Pars2[i], V_Pars2[j+1]), (D_Pars2[i+1], V_Pars2[j+1]), (D_Pars2[i+1], V_Pars2[j]), (D_Pars2[i], V_Pars2[j]),]
                ind_color=np.floor(32*(log10_all_drops_2_Pars2[i,j]-0.00001)/max_value)    
                ind_color=ind_color.astype(int)
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor=list_colors[ind_color], lw=0)
                ax_Pars2.add_patch(patch)
    plt.plot(D_Pars2,V_theo_Pars2,color='k')
    plt.xlabel(r'$\mathit{D}\/\/(mm)$',fontsize=14,color='k')
    #plt.ylabel(r'$\mathit{V}\/\/(m.s^{-1})$',fontsize=20,color='k')
    plt.title('Parsivel #2',fontsize=14,color='k')  
    for xtick in ax_Pars2.get_xticklabels():
        plt.setp(xtick,fontsize=14)
    for ytick in ax_Pars2.get_yticklabels():
        plt.setp(ytick,fontsize=14)  
    ax_Pars2.set_xlim(0,9)
    ax_Pars2.set_ylim(0,12)

    
    ax_colorbar = plt.axes([0.8,0.1,0.04,0.24])
    cmap = colors.ListedColormap(list_colors)
    try : 
        bounds = range(33)*max_value/32 # [-1., -.5, 0., .5, 1.]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        #    cax, kw = colorbar.make_axes(ax)
        cb3 = colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm, boundaries=bounds, extendfrac='auto', ticks=bounds[0:-1:3], spacing='uniform', orientation='vertical')
        cb3.set_ticklabels(np.floor(bounds[0:-1:3]*100)/100)
        #cb3.set_label('log10(all_drops_2)')
    except: 
        print('souci')
        
    plt.savefig(path_outputs+'size_velocity_map_'+str_evt+'.png',dpi=70,bbox_inches='tight',format='png')
    plt.clf()
    plt.close()
    
    img_DV=mpimg.imread(path_outputs+'size_velocity_map_'+str_evt+'.png')
    
    
    # Generating a patchwork image (the Quicklook) containing all the desired pictures  
    
    i1=np.max([img_R.shape[0],img_C.shape[0]])
    i2=np.max([img_DSD.shape[0],img_no_data.shape[0] + img_DV.shape[0]])
    i3=np.max([img_ND3.shape[0],img_T.shape[0]])
    
    
    j1=np.max([img_R.shape[1],img_DSD.shape[1],img_ND3.shape[1]])
    j2=np.max([img_C.shape[1],img_DV.shape[1],img_T.shape[1],img_no_data.shape[1]])     
    
    img_quicklook=np.zeros((i1+i2+i3,j1+j2,4))+1    
    
    img_quicklook[0:img_R.shape[0],0:img_R.shape[1],:]=img_R
    img_quicklook[0:img_C.shape[0],j1:j1+img_C.shape[1],:]=img_C    
    img_quicklook[i1:i1+img_DSD.shape[0],0:img_DSD.shape[1],:]=img_DSD   

    img_quicklook[i1:i1+img_no_data.shape[0],j1:j1+img_no_data.shape[1],:]=img_no_data  
    img_quicklook[i1+img_no_data.shape[0]:i1+img_no_data.shape[0]+img_DV.shape[0],j1:j1+img_DV.shape[1],:]=img_DV

    img_quicklook[i1+i2:i1+i2+img_ND3.shape[0],0:img_ND3.shape[1],:]=img_ND3  
    img_quicklook[i1+i2:i1+i2+img_T.shape[0],j1:j1+img_T.shape[1],:]=img_T      
    plt.figure(17)
    plt.imshow(img_quicklook)    
    mpimg.imsave(path_outputs+'Quicklook_Carnot_1_'+str_evt+'.png',img_quicklook,format='png')

    subprocess.call('rm -r '+path_outputs+'no_data_'+str_evt+'.png',shell=True)        
    subprocess.call('rm -r '+path_outputs+'temporal_rain_rate_'+str_evt+'.png',shell=True)
    subprocess.call('rm -r '+path_outputs+'temporal_cumul_depth_'+str_evt+'.png',shell=True)
    subprocess.call('rm -r '+path_outputs+'temporal_temperature_'+str_evt+'.png',shell=True)
    subprocess.call('rm -r '+path_outputs+'temporal_evo_rho_'+str_evt+'.png',shell=True)
    subprocess.call('rm -r '+path_outputs+'N_D_'+str_evt+'.png',shell=True)
    subprocess.call('rm -r '+path_outputs+'N_D_D3_'+str_evt+'.png',shell=True)
    subprocess.call('rm -r '+path_outputs+'temporal_evo_N_D_'+str_evt+'.png',shell=True)
    subprocess.call('rm -r '+path_outputs+'size_velocity_map_'+str_evt+'.png',shell=True)
    
    
    
    # Saving the data with 30 s time steps 
    R_30s=np.zeros((R_emp_2_PWS.shape[0],3))
    R_30s[:,0]=R_emp_2_PWS
    R_30s[:,1]=R_emp_2_Pars1
    R_30s[:,2]=R_emp_2_Pars2
    np.savetxt(path_outputs+'R_30_sec_Carnot_1_'+str_evt+'.csv',R_30s,delimiter=';')
    
    # Computing and saving the date with 5 min time steps
    NB_5min=np.intp(np.floor(R_emp_2_PWS.shape[0]/10))
    
    R_5min=np.zeros((NB_5min,3))
    for i in range(NB_5min):
        R_5min[i,0]=np.mean(R_emp_2_PWS[i*10:((i+1)*10)])
        R_5min[i,1]=np.mean(R_emp_2_Pars1[i*10:((i+1)*10)])
        R_5min[i,2]=np.mean(R_emp_2_Pars2[i*10:((i+1)*10)])
    
    np.savetxt(path_outputs+'R_5_min_Carnot_1_'+str_evt+'.csv',R_5min,delimiter=';')
    
    return 2





def extracting_one_event_Carnot_1(start_evt,end_evt,path_daily_data_python):
############################################################################################
# Aim : reading daily.npy files and genrating three lists (one for each disdrometer) containing all the data that can be analyzed
# 
# Inputs: 
#    - The start and end date of the studied event (datetime objects; should be with 0s and 30s)
#    - path_daily_data_python (a string): Path to the folder where inputs are stored (assumed to contain folders Pars1, Pars2 and PWS 
#      containing the data for the disdrometers taking part to the this campaign)
#
# Outputs: three lists (one for each disdrometers) containing time series or matrix with the data for the studied event.
#       The order in a given list is: 
#             - T : temperature (time series)
#             - R_disdro : rain rate (mm/h) computed by the disdrometer (time series)
#             - R_emp_1 : rain rate (mm/h) computed with reconstructed N(D) and V(D) theoretical (Lhermitte et al., 1988) (time series)
#             - R_emp_2 : rain rate (mm/h) computed with the drops fallen during the time step (Recommended to use this one) (time series)
#             - Nb_all_drops : a matrix (26*26) for all disdro containing the number of drop within each class during the whole event
#             - N_D_emp : empirical binned drop size distribution (m-3.mm-1) (matrix, (Nb_time_steps,26))
#             - rho : liquid water content (g/m3)  (time series)
#             - Nt : total drop concentration (m-3)  (time series)
#             - Dm : mass-weighted diameter (mm)  (time series)
#             - Zh : horizontal reflectvity (mm6.m-3)  (time series)
#             - Zv : Vertical reflectivity (mm6.m-3)  (time series)
#             - Kdp : specific differetnial phase (°/km)  (time series)
#
# Note : 
#    - the 10/9 correction for PWS (measuring only 9/10 of time) is taken into account
#    - data is filtered as in Jaffrain and Berne 2012 (i.e. if velocities should be bounded for a given diameter); see Gires et al. 2015 for more details.
#    - missing data are marked as 0 and not "nan". The number of missing time steps for each disdrometer is printed.
############################################################################################

    i_correction_PWS=1 # To either implement or not the correction (warning if not, the files for radar parameter should be changed)
    time_step=timedelta(0,30) # Setting the time step as the recording one
 
    # Computing the number of time steps
    t=start_evt
    N=0    
    while t<=end_evt:
        #print(str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second), N)
        t=t+time_step
        N=N+1    
    Nb_time_steps=N
    
    #print(start_evt,end_evt)    
    
    ##########
    ## Retrieving the data
    ##########
        
    # Definition of the data for the PWS
    T_PWS = np.zeros(Nb_time_steps)
    R_disdro_PWS = np.zeros(Nb_time_steps)
    R_emp_1_PWS = np.zeros(Nb_time_steps)
    R_emp_2_PWS = np.zeros(Nb_time_steps)
    Nb_all_drops_PWS = np.zeros((34,34))
    N_D_emp_PWS = np.zeros((Nb_time_steps,34))
    Rela_humi_PWS = np.zeros(Nb_time_steps)
    rho_PWS = np.zeros(Nb_time_steps)
    Nt_PWS = np.zeros(Nb_time_steps)
    Dm_PWS = np.zeros(Nb_time_steps)
    Zh_PWS = np.zeros(Nb_time_steps)
    Zv_PWS = np.zeros(Nb_time_steps)
    Kdp_PWS = np.zeros(Nb_time_steps)
    
    # Definition of the data for the Pars1
    T_Pars1 = np.zeros(Nb_time_steps)
    R_disdro_Pars1 = np.zeros(Nb_time_steps)
    R_emp_1_Pars1 = np.zeros(Nb_time_steps)
    R_emp_2_Pars1 = np.zeros(Nb_time_steps)
    Nb_all_drops_Pars1 = np.zeros((32,32))
    N_D_emp_Pars1 = np.zeros((Nb_time_steps,32))
    rho_Pars1 = np.zeros(Nb_time_steps)
    Nt_Pars1 = np.zeros(Nb_time_steps)
    Dm_Pars1 = np.zeros(Nb_time_steps)
    Zh_Pars1 = np.zeros(Nb_time_steps)
    Zv_Pars1 = np.zeros(Nb_time_steps)
    Kdp_Pars1 = np.zeros(Nb_time_steps)

    # Definition of the data for the Pars1
    T_Pars2 = np.zeros(Nb_time_steps)
    R_disdro_Pars2 = np.zeros(Nb_time_steps)
    R_emp_1_Pars2 = np.zeros(Nb_time_steps)
    R_emp_2_Pars2 = np.zeros(Nb_time_steps)
    Nb_all_drops_Pars2 = np.zeros((32,32))
    N_D_emp_Pars2 = np.zeros((Nb_time_steps,32))
    rho_Pars2 = np.zeros(Nb_time_steps)
    Nt_Pars2 = np.zeros(Nb_time_steps)
    Dm_Pars2 = np.zeros(Nb_time_steps)
    Zh_Pars2 = np.zeros(Nb_time_steps)
    Zv_Pars2 = np.zeros(Nb_time_steps)
    Kdp_Pars2 = np.zeros(Nb_time_steps)
    
    # For the PWS
    V_PWS = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12.0,13.6,15.2,17.6,20.8,24.0,27.2))
    V_width_PWS=np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,3.2,3.2))
    D_PWS = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12.0,13.6,15.2,17.6,20.8,24.0,27.2))
    D_width_PWS = np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,3.2,3.2))
    V_theo_PWS = V_D_Lhermitte_1988 (D_PWS)
    S_eff_PWS = 0.004*np.ones(D_PWS.shape)

    # If a the correction for oblateness is implemented, D_PWS and D_width_PWS are modified
    if i_correction_PWS==1:
        D_PWS_buf=np.zeros(D_PWS.shape)
        D_width_PWS_buf=np.zeros(D_PWS.shape)

        coeff=np.array((-0.01361193,0.12089066,0.87497141,0.00989984)) # With Pars and n=1.33
        #coeff=np.array((-0.03302464,0.28672044,0.43507802,0.33835173)) # With Ands and n=1.33
        #coeff=np.array((-0.02259775,0.19820372,0.6685606,0.17391049)) # With Beard and n=1.33
    
        for i in range(34):
            if D_PWS[i]>1:
                D_meas=D_PWS[i]**(1/1.25)
                D_PWS_buf[i]=coeff[0]*D_meas**3+coeff[1]*D_meas**2+coeff[2]*D_meas+coeff[3]
            else : 
                D_PWS_buf[i]=D_PWS[i]

        for i in range(34):
            D_min=D_PWS[i]-D_width_PWS[i]/2
            if D_min > 1:
                D_min=D_min**(1/1.25)
                D_min_buf=coeff[0]*D_min**3+coeff[1]*D_min**2+coeff[2]*D_min+coeff[3]
            else:
                D_min_buf=D_min
            D_max=D_PWS[i]+D_width_PWS[i]/2
            if D_max > 1:
                D_max=D_max**(1/1.25)
                D_max_buf=coeff[0]*D_max**3+coeff[1]*D_max**2+coeff[2]*D_max+coeff[3]
            else:
                D_max_buf=D_max
            D_width_PWS_buf[i]=D_max_buf-D_min_buf
      
        
        D_PWS=D_PWS_buf
        D_width_PWS=D_width_PWS_buf
        V_theo_PWS = V_D_Lhermitte_1988 (D_PWS)


    # For the Pars1
    V_Pars1 = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8))
    V_width_Pars1=np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2))    
    D_Pars1=np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.750,3.250,3.750,4.250,4.750,5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5])
    D_width_Pars1=np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.250,0.250,0.250,0.250,0.250,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,2,3,3])
    V_Pars1 = V_Pars1
    V_width_Pars1 = V_width_Pars1
    D_Pars1 = D_Pars1
    D_width_Pars1 = D_width_Pars1
    V_theo_Pars1 = V_D_Lhermitte_1988 (D_Pars1)
    S_eff_Pars1 = 10**(-6)*180*(30-D_Pars1/2)
    
    # For the Pars2
    V_Pars2 = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8))
    V_width_Pars2=np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2))
    D_Pars2=np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.750,3.250,3.750,4.250,4.750,5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5])
    D_width_Pars2=np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.250,0.250,0.250,0.250,0.250,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,2,3,3])
    V_Pars2 = V_Pars2
    V_width_Pars2 = V_width_Pars2
    D_Pars2 = D_Pars2
    D_width_Pars2 = D_width_Pars2
    V_theo_Pars2 = V_D_Lhermitte_1988 (D_Pars2)
    S_eff_Pars2 = 10**(-6)*180*(30-D_Pars2/2)
    
    # Definition of indxes for missing data
    missing_PWS_data=0
    missing_Pars1_data=0
    missing_Pars2_data=0


    ##############################
    ## WARNING FOR THE RADAR PARAMETERS IT ONLY WORKS IF THE CORRECTION IS IMPLEMENTED, OTHERWISE I SHOULD CHANGE 
    ## THE NAMES HERE
    ##############################
    
    # Retrieving the radar parameters
    sigma_b_hor_Pars=np.loadtxt('test_sigma_hor_Pars.csv',delimiter=';')
    sigma_b_vert_Pars=np.loadtxt('test_sigma_vert_Pars.csv',delimiter=';')
    truc_kdp_Pars=np.loadtxt('test_truc_kdp_Pars.csv',delimiter=';')

    sigma_b_hor_PWS=np.loadtxt('test_sigma_hor_PWS.csv',delimiter=';')
    sigma_b_vert_PWS=np.loadtxt('test_sigma_vert_PWS.csv',delimiter=';')
    truc_kdp_PWS=np.loadtxt('test_truc_kdp_PWS.csv',delimiter=';')

    #m_wat=complex(8.208,1.886)    #refractive.m_w_20C for X_band
    m_wat=complex(8.633,1.289) # for C band
    #wl =33.3   # Expressed in mm for X_band
    wl =53.5   # Expressed in mm for C_band
    kw=0.927840502631 #for C band at 20C 
#    print('info radar')
#    print('sigma_b_hor_PWS = ')
#    print(sigma_b_hor_PWS)
#    print('wl = ',wl)
#    print('k = ',np.abs((m_wat*m_wat-1)/(m_wat*m_wat+2))**2)

          
    # Definition of the filter based on speed (as in Jaffrain et al.)
    filter_vel_PWS=np.zeros((34,34))
    for i in range(30): # Class of size
        for j in range(34): # Class of speed
            if np.abs(V_PWS[j]-V_theo_PWS[i])<=0.6*V_theo_PWS[i]:
                filter_vel_PWS[i,j]=1
            else:
                filter_vel_PWS[i,j]=0

    filter_vel_Pars1=np.ones((32,32))
    filter_vel_Pars2=np.ones((32,32))
    for i in range(32): # Class of size
        for j in range(32): # Class of speed
            if np.abs(V_Pars1[j]-V_theo_Pars1[i])<=0.6*V_theo_Pars1[i]:
                filter_vel_Pars1[i,j]=1
            else:
                filter_vel_Pars1[i,j]=0            
            if np.abs(V_Pars2[j]-V_theo_Pars2[i])<=0.6*V_theo_Pars2[i]:
                filter_vel_Pars2[i,j]=1
            else:
                filter_vel_Pars2[i,j]=0            

    t=start_evt
    N=0
    
    # Extracting the data from the data 
    while t<=end_evt:
        # Retrieving the name of the time step
        time_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second)
        #print(time_str)
        day_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)
        day_str=t.strftime('%Y%m%d')
        if t.second==0:
            n_time=120*t.hour+2*t.minute-1
        elif t.second==30:
            n_time=120*t.hour+2*t.minute
        else :
            print('Issue on t.second in extracting_data_one_event')
        if n_time==-1: # handling the case 00:00:00
            n_time=2879

        if t==start_evt: # Opending the .npy files
            try :
                day_str_file_PWS=day_str
                f=open(path_daily_data_python+'PWS/PWS_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro_PWS = pickle.load(f) 
                f.close()
            except:
                print('File : '+path_daily_data_python+'PWS/PWS_raw_data_'+day_str+'.npy is missing')

            try :
                day_str_file_Pars1=day_str
                f=open(path_daily_data_python+'Pars1/Pars1_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro_Pars1 = pickle.load(f) 
                f.close()
            except:
                print('File : '+path_daily_data_python+'/Pars1/Pars1_raw_data_'+day_str+'.npy is missing')
                
            try :
                day_str_file_Pars2=day_str
                f=open(path_daily_data_python+'Pars2/Pars2_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro_Pars2 = pickle.load(f) 
                f.close()
            except:
                print('File : '+path_daily_data_python+'Pars2/Pars2_raw_data_'+day_str+'.npy is missing')


        try :
            if day_str != day_str_file_PWS: # Opending a new .npy file if there is a change of day 
                day_str_file_PWS=day_str
                try :
                    f=open(path_daily_data_python+'PWS/PWS_raw_data_'+day_str+'.npy', 'rb') 
                    day_all_disdro_PWS = pickle.load(f) 
                    f.close()
                except:
                    day_all_disdro_PWS=list()
                    print('File : '+path_daily_data_python+'PWS/PWS_raw_data_'+day_str+'.npy is missing')          
            # Computing the data for the PWS 
            T_PWS[N] = day_all_disdro_PWS[n_time][7]
            Rela_humi_PWS[N] = day_all_disdro_PWS[n_time][8]
            R_disdro_PWS[N] = day_all_disdro_PWS[n_time][12]
            full_map = day_all_disdro_PWS[n_time][18][0:34,0:34]*filter_vel_PWS
            #np.savetxt('PWS_'+str(N+1)+'.csv',full_map,delimiter=' ')            
            Nb_all_drops_PWS = Nb_all_drops_PWS + full_map            
            for i in range(29) : # Loop on the sizes
                for j in range(29) : # Loop on the velocity
                    N_D_emp_PWS[N,i] = N_D_emp_PWS[N,i] + full_map[i,j]/(S_eff_PWS[i]*D_width_PWS[i]*V_PWS[j]*30)
                    R_emp_2_PWS[N] = R_emp_2_PWS[N] + 3.14159*120*full_map[i,j]*D_PWS[i]**3/(6*S_eff_PWS[i]*10**6)
            R_emp_2_PWS[N]=R_emp_2_PWS[N]*10/9
            N_D_emp_PWS[N,:]=N_D_emp_PWS[N,:]*10/9
            a=0
            b=0
            for i in range(29) : 
                R_emp_1_PWS[N] = R_emp_1_PWS[N] + 6*3.14159*10**(-4)*N_D_emp_PWS[N,i]*V_theo_PWS[i]*D_PWS[i]**(3)*D_width_PWS[i]
                rho_PWS[N] = rho_PWS[N] + 3.14159*N_D_emp_PWS[N,i]*D_PWS[i]**3*D_width_PWS[i]/(6*10**3) # in g/m3
                Nt_PWS[N]=Nt_PWS[N]+N_D_emp_PWS[N,i]*D_width_PWS[i]
                a=a+N_D_emp_PWS[N,i]*D_PWS[i]**(4)*D_width_PWS[i]
                b=b+N_D_emp_PWS[N,i]*D_PWS[i]**(3)*D_width_PWS[i]
                #Zh_PWS[N] = Zh_PWS[N] + wl**4/((np.pi**5)*kw)*N_D_emp_PWS[N,i]*sigma_b_hor_PWS[i]*D_width_PWS[i]
                #Zv_PWS[N] = Zv_PWS[N] + wl**4/((np.pi**5)*kw)*N_D_emp_PWS[N,i]*sigma_b_vert_PWS[i]*D_width_PWS[i]
                #Kdp_PWS[N] = Kdp_PWS[N] + 0.001*(180.0/np.pi)*wl*N_D_emp_PWS[N,i]*truc_kdp_PWS[i]*D_width_PWS[i]
            Dm_PWS[N]=a/b
            #print(N,Dm_PWS[N],a,b)
        except:
            missing_PWS_data=missing_PWS_data+1
            T_PWS[N] = np.nan
            Rela_humi_PWS[N] = np.nan
            R_disdro_PWS[N] = np.nan
            for i in range(29):
                N_D_emp_PWS[N,i] = np.nan
            R_emp_2_PWS[N]=np.nan
            R_emp_1_PWS[N] = np.nan 
            rho_PWS[N] = np.nan 
            Nt_PWS[N]=np.nan 
            Zh_PWS[N] = np.nan 
            Zv_PWS[N] = np.nan 
            Kdp_PWS[N] =np.nan
            Dm_PWS[N]=np.nan   
         
                
        # Computing the data for the Pars1    
        try :    
            if day_str != day_str_file_Pars1: # Opending a new .npy file if there is a change of day 
                day_str_file_Pars1=day_str
                try :
                    f=open(path_daily_data_python+'Pars1/Pars1_raw_data_'+day_str+'.npy', 'rb') 
                    day_all_disdro_Pars1 = pickle.load(f) 
                    f.close()
                except:
                    day_all_disdro_Pars1=list()
                    print('File : '+path_daily_data_python+'/Pars1/Pars1_raw_data_'+day_str+'.npy is missing')          

            T_Pars1[N] = day_all_disdro_Pars1[n_time][2]
            R_disdro_Pars1[N] = day_all_disdro_Pars1[n_time][1]
            #full_map = day_all_disdro_Pars1[n_time][3][0:32,0:32]
            full_map = day_all_disdro_Pars1[n_time][3][0:32,0:32]*filter_vel_Pars1
            #np.savetxt('Pars_'+str(N+1)+'.csv',full_map,delimiter=' ')
            Nb_all_drops_Pars1 = Nb_all_drops_Pars1 + full_map      
            for i in range(26) : # Loop on the sizes
                for j in range(30) : # Loop on the velocity
                    N_D_emp_Pars1[N,i] = N_D_emp_Pars1[N,i] + full_map[i,j]/(S_eff_Pars1[i]*D_width_Pars1[i]*V_Pars1[j]*30)
                    R_emp_2_Pars1[N] = R_emp_2_Pars1[N] + 3.14159*120*full_map[i,j]*D_Pars1[i]**3/(6*S_eff_Pars1[i]*10**6)    
            a=0
            b=0
            for i in range(26) : 
                R_emp_1_Pars1[N] = R_emp_1_Pars1[N] + 6*3.14159*10**(-4)*N_D_emp_Pars1[N,i]*V_theo_Pars1[i]*D_Pars1[i]**(3)*D_width_Pars1[i]
                rho_Pars1[N] = rho_Pars1[N] + 3.14159*N_D_emp_Pars1[N,i]*D_Pars1[i]**3*D_width_Pars1[i]/(6*10**3) # in g/m3
                Nt_Pars1[N]=Nt_Pars1[N]+N_D_emp_Pars1[N,i]*D_width_Pars1[i]
                a=a+N_D_emp_Pars1[N,i]*D_Pars1[i]**(4)*D_width_Pars1[i]
                b=b+N_D_emp_Pars1[N,i]*D_Pars1[i]**(3)*D_width_Pars1[i]
                ##Zh_Pars1[N] = Zh_Pars1[N] + (wl**4/((np.pi**5)*(np.abs((m_wat*m_wat-1)/(m_wat*m_wat+2)))**2))*N_D_emp_Pars1[N,i]*sigma_b_hor_Pars[i]*D_width_Pars1[i]
                Zh_Pars1[N] = Zh_Pars1[N] + wl**4/((np.pi**5)*kw)*N_D_emp_Pars1[N,i]*sigma_b_hor_Pars[i]*D_width_Pars1[i]
                Zv_Pars1[N] = Zv_Pars1[N] + wl**4/((np.pi**5)*kw)*N_D_emp_Pars1[N,i]*sigma_b_vert_Pars[i]*D_width_Pars1[i]
                Kdp_Pars1[N] = Kdp_Pars1[N] + 0.001*180*wl/np.pi*N_D_emp_Pars1[N,i]*truc_kdp_Pars[i]*D_width_Pars1[i]
            Dm_Pars1[N]=a/b
        except:
            missing_Pars1_data=missing_Pars1_data+1
            T_Pars1[N] = np.nan
            R_disdro_Pars1[N] =np.nan
            for i in range(32) : # Loop on the sizes
                N_D_emp_Pars1[N,i] = np.nan
            R_emp_2_Pars1[N] = np.nan
            R_emp_1_Pars1[N] = np.nan 
            rho_Pars1[N] = np.nan
            Nt_Pars1[N]=np.nan
            Zh_Pars1[N] = np.nan
            Zv_Pars1[N] = np.nan
            Kdp_Pars1[N] = np.nan
            Dm_Pars1[N] = np.nan
    
        # Computing the data for the Pars2    
        try :    
            if day_str != day_str_file_Pars2: # Opending a new .npy file if there is a change of day 
                day_str_file_Pars2=day_str
                try :
                    f=open(path_daily_data_python+'Pars2/Pars2_raw_data_'+day_str+'.npy', 'rb') 
                    day_all_disdro_Pars2 = pickle.load(f) 
                    f.close()
                except:
                    day_all_disdro_Pars2=list()
                    print('File : '+path_daily_data_python+'Pars2/Pars2_raw_data_'+day_str+'.npy is missing')          

            T_Pars2[N] = day_all_disdro_Pars2[n_time][2]
            R_disdro_Pars2[N] = day_all_disdro_Pars2[n_time][1]
            #full_map = day_all_disdro_Pars2[n_time][3][0:32,0:32]
            full_map = day_all_disdro_Pars2[n_time][3][0:32,0:32]*filter_vel_Pars2
            Nb_all_drops_Pars2 = Nb_all_drops_Pars2 + full_map    
            for i in range(26) : # Loop on the sizes
                for j in range(30) : # Loop on the velocity
                    N_D_emp_Pars2[N,i] = N_D_emp_Pars2[N,i] + full_map[i,j]/(S_eff_Pars2[i]*D_width_Pars2[i]*V_Pars2[j]*30)
                    R_emp_2_Pars2[N] = R_emp_2_Pars2[N] + 3.14159*120*full_map[i,j]*D_Pars2[i]**3/(6*S_eff_Pars2[i]*10**6)    
            a=0
            b=0
            for i in range(26) : 
                R_emp_1_Pars2[N] = R_emp_1_Pars2[N] + 6*3.14159*10**(-4)*N_D_emp_Pars2[N,i]*V_theo_Pars2[i]*D_Pars2[i]**(3)*D_width_Pars2[i]
                rho_Pars2[N] = rho_Pars2[N] + 3.14159*N_D_emp_Pars2[N,i]*D_Pars2[i]**3*D_width_Pars2[i]/(6*10**3) # in g/m3
                Nt_Pars2[N]=Nt_Pars2[N]+N_D_emp_Pars2[N,i]*D_width_Pars2[i]
                a=a+N_D_emp_Pars2[N,i]*D_Pars2[i]**(4)*D_width_Pars2[i]
                b=b+N_D_emp_Pars2[N,i]*D_Pars2[i]**(3)*D_width_Pars2[i]
                Zh_Pars2[N] = Zh_Pars2[N] + wl**4/((np.pi**5)*kw)*N_D_emp_Pars2[N,i]*sigma_b_hor_Pars[i]*D_width_Pars2[i]
                Zv_Pars2[N] = Zv_Pars2[N] + wl**4/((np.pi**5)*kw)*N_D_emp_Pars2[N,i]*sigma_b_vert_Pars[i]*D_width_Pars2[i]
                Kdp_Pars2[N] = Kdp_Pars2[N] + 0.001*180*wl/np.pi*N_D_emp_Pars2[N,i]*truc_kdp_Pars[i]*D_width_Pars2[i]
            Dm_Pars2[N]=a/b
        except:
            missing_Pars2_data=missing_Pars2_data+1
            T_Pars2[N] = np.nan
            R_disdro_Pars2[N] =np.nan
            for i in range(32) : # Loop on the sizes
                N_D_emp_Pars2[N,i] = np.nan
            R_emp_2_Pars2[N] = np.nan 
            R_emp_1_Pars2[N] = np.nan 
            rho_Pars2[N] = np.nan 
            Nt_Pars2[N]=np.nan 
            Zh_Pars2[N] = np.nan
            Zv_Pars2[N] = np.nan
            Kdp_Pars2[N] = np.nan
            Dm_Pars2[N] = np.nan
    
        
        # Incrementation of time 
        t=t+time_step
        N=N+1
        
    # Out the loop for retrieving the data
    
    # Displaying when missing data are noticed
    print('Nb of missing time steps for PWS100 = ',missing_PWS_data)
    print('Nb of missing time steps for Pars1 = ',missing_Pars1_data)
    print('Nb of missing time steps for Pars2 = ',missing_Pars2_data)
    
    # Retrieving the data for the ENSG rain gauge if available


    # Definition of the data for the PWS
    PWS_one_event=list()
    PWS_one_event.append(T_PWS)
    PWS_one_event.append(R_disdro_PWS)
    PWS_one_event.append(R_emp_1_PWS)
    PWS_one_event.append(R_emp_2_PWS)
    PWS_one_event.append(Nb_all_drops_PWS)
    PWS_one_event.append(N_D_emp_PWS)
    PWS_one_event.append(rho_PWS)
    PWS_one_event.append(Nt_PWS)
    PWS_one_event.append(Dm_PWS)
    PWS_one_event.append(Zh_PWS)
    PWS_one_event.append(Zv_PWS)
    PWS_one_event.append(Kdp_PWS)
    PWS_one_event.append(Rela_humi_PWS)
    
    # Definition of the data for the Pars1
    Pars1_one_event=list()
    Pars1_one_event.append(T_Pars1)
    Pars1_one_event.append(R_disdro_Pars1)
    Pars1_one_event.append(R_emp_1_Pars1)
    Pars1_one_event.append(R_emp_2_Pars1)
    Pars1_one_event.append(Nb_all_drops_Pars1)
    Pars1_one_event.append(N_D_emp_Pars1)
    Pars1_one_event.append(rho_Pars1)
    Pars1_one_event.append(Nt_Pars1)
    Pars1_one_event.append(Dm_Pars1)
    Pars1_one_event.append(Zh_Pars1)
    Pars1_one_event.append(Zv_Pars1)
    Pars1_one_event.append(Kdp_Pars1)

    # Definition of the data for the Pars2
    Pars2_one_event=list()
    Pars2_one_event.append(T_Pars2)
    Pars2_one_event.append(R_disdro_Pars2)
    Pars2_one_event.append(R_emp_1_Pars2)
    Pars2_one_event.append(R_emp_2_Pars2)
    Pars2_one_event.append(Nb_all_drops_Pars2)
    Pars2_one_event.append(N_D_emp_Pars2)
    Pars2_one_event.append(rho_Pars2)
    Pars2_one_event.append(Nt_Pars2)
    Pars2_one_event.append(Dm_Pars2)
    Pars2_one_event.append(Zh_Pars2)
    Pars2_one_event.append(Zv_Pars2)
    Pars2_one_event.append(Kdp_Pars2)

    return PWS_one_event,Pars1_one_event,Pars2_one_event







def exporting_full_matrix(start_evt,end_evt,disdro_name,path_outputs,path_daily_data_python):
############################################################################################
# Aim : reading daily.npy files and exporting full matrix in .csv files
# 
# Inputs: 
#    - The start and end date of the studied event (datetime objects; should be with 0s and 30s)
#    - disdro_name (a string) : the name of the disdrometer (either 'Pars1', 'Pars2', 'Pars_Rad' or 'PWS')
#    - path_outputs (a string) : path to the folder where outputs will be written
#    - path_daily_data_python (a string): Path to the folder where inputs are stored (assumed to contain folders Pars1, Pars2, Pars_Rad and PWS 
#      containing the data for the disdrometers taking part to the this campaign)
#
# Outputs: 
#     - a file saved : 'Disdro_name'_full_matrix_2017_05_18_16_00_00__2017_05_19_01_00_00.csv
#              the start and end of the event is stored in file name
#        One line per time step
#        Date; number of drops per class of velocity and size (1st size class - 1st velocity class,1st size class - 2nd velocity class, 1st size class - 2nd velocity class, ... , 2nd size class - 1st velocity class...) separated by comas
#        34 * 34 classes for PWS data
#        32 * 32 classes for Parsivel data
#        WARNING : data is measured only 9/10 of the time for PWS, hence to have comparable values, one should multiply by 10/9 the number of drops in each class for the PWS.
#        ex : 2017-05-18 16:00:00;0.0,0.0,0.0,0.0,0.0,0.0,......
#        missing data is noted as np.nan
############################################################################################

    str_evt=start_evt.strftime('%Y_%m_%d_%H_%M_%S')+'__'+end_evt.strftime('%Y_%m_%d_%H_%M_%S')
    print(str_evt)

    time_step=timedelta(0,30)
    t=start_evt

    f_output = open(path_outputs+disdro_name+'_full_matrix_'+str_evt+'.csv', 'w')

    # Loop on time steps
    while t<=end_evt:
        # Retrieving the name of the time step
        time_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second)
        #print(time_str)
        day_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)
        day_str=t.strftime('%Y%m%d')
        if t.second==0:
            n_time=120*t.hour+2*t.minute-1
        elif t.second==30:
            n_time=120*t.hour+2*t.minute
        else :
            print('Issue on t.second in extracting_data_one_event')
        if n_time==-1: # handling the case 00:00:00
            n_time=2879

        # Selecting the correct daily file to look for the disdrometer data
        if t==start_evt:
            try :
                day_str_file=day_str
                f=open(path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro = pickle.load(f) 
                f.close()
            except:
                print('File : '+path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy is missing')

           
        if day_str != day_str_file:
            day_str_file=day_str
            try :
                f=open(path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro = pickle.load(f) 
                f.close()
            except:
                day_all_disdro=list()
                print('File : '+path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy is missing')          
                # Reading the data from the Pars1    

        # Writing the full matrix in the line corresponding got the time step in the file
        if disdro_name=='Pars1':
            line_str=t.strftime('%Y-%m-%d %H:%M:%S')+';'
            try:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(day_all_disdro[n_time][3][k,l])+','
            except:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(np.nan)+','
            f_output.write(line_str+'\n')                   

        elif disdro_name=='Pars2':
            line_str=t.strftime('%Y-%m-%d %H:%M:%S')+';'
            try:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(day_all_disdro[n_time][3][k,l])+','
            except:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(np.nan)+','
            f_output.write(line_str+'\n')                   

        elif disdro_name=='Pars_Rad':
            line_str=t.strftime('%Y-%m-%d %H:%M:%S')+';'
            try:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(day_all_disdro[n_time][3][k,l])+','
            except:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(np.nan)+','
            f_output.write(line_str+'\n')                   
        elif disdro_name=='PWS':
            line_str=t.strftime('%Y-%m-%d %H:%M:%S')+';'
            try:
                for k in range(34):   # size class
                    for l in range(34):  # velocity class
                        line_str=line_str+str(day_all_disdro[n_time][18][k,l])+','
            except:
                for k in range(34):   # size class
                    for l in range(34):  # velocity class
                        line_str=line_str+str(np.nan)+','
            f_output.write(line_str+'\n')    
                
        # Incrementation of time         
        t=t+time_step
        
        
    # Colsing the file
    f_output.close() 


def exporting_full_matrix_and_T(start_evt,end_evt,disdro_name,path_outputs,path_daily_data_python):
############################################################################################
# Aim : reading daily.npy files and exporting full matrix in .csv files
# 
# Inputs: 
#    - The start and end date of the studied event (datetime objects; should be with 0s and 30s)
#    - disdro_name (a string) : the name of the disdrometer (either 'Pars1', 'Pars2', 'Pars_Rad' or 'PWS')
#    - path_outputs (a string) : path to the folder where outputs will be written
#    - path_daily_data_python (a string): Path to the folder where inputs are stored (assumed to contain folders Pars1, Pars2, Pars_Rad and PWS 
#      containing the data for the disdrometers taking part to the this campaign)
#
# Outputs: 
#     - a file saved : 'Disdro_name'_daily_data_2017_05_18_16_00_00__2017_05_19_01_00_00.csv
#              the start and end of the event is stored in file name
#        One line per time step
#        Date; number of drops per class of velocity and size (1st size class - 1st velocity class,1st size class - 2nd velocity class, 1st size class - 2nd velocity class, ... , 
#            2nd size class - 1st velocity class...) separated by comas; Temperature (°C)
#        34 * 34 classes for PWS data
#        32 * 32 classes for Parsivel data
#        WARNING : data is measured only 9/10 of the time for PWS, hence to have comparable values, one should multiply by 10/9 the number of drops in each class for the PWS.
#        ex : 2017-05-18 16:00:00;0.0,0.0,0.0,0.0,0.0,0.0,......0.0;7
#        missing data is noted as np.nan
############################################################################################

    str_evt=start_evt.strftime('%Y_%m_%d_%H_%M_%S')+'__'+end_evt.strftime('%Y_%m_%d_%H_%M_%S')
    print(str_evt)

    time_step=timedelta(0,30)
    t=start_evt

    f_output = open(path_outputs+disdro_name+'_daily_data_'+str_evt+'.csv', 'w')
    #f_output = open(path_outputs+disdro_name+'_daily_data_'+start_evt.strftime('%Y%m%d')+'.csv', 'w')


    # Loop on time steps
    while t<=end_evt:
        # Retrieving the name of the time step
        time_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second)
        #print(time_str)
        day_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)
        day_str=t.strftime('%Y%m%d')
        if t.second==0:
            n_time=120*t.hour+2*t.minute-1
        elif t.second==30:
            n_time=120*t.hour+2*t.minute
        else :
            print('Issue on t.second in extracting_data_one_event')
        if n_time==-1: # handling the case 00:00:00
            n_time=2879

        # Selecting the correct daily file to look for the disdrometer data
        if t==start_evt:
            try :
                day_str_file=day_str
                f=open(path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro = pickle.load(f) 
                f.close()
            except:
                print('File : '+path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy is missing')

           
        if day_str != day_str_file:
            day_str_file=day_str
            try :
                f=open(path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro = pickle.load(f) 
                f.close()
            except:
                day_all_disdro=list()
                print('File : '+path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy is missing')          
                # Reading the data from the Pars1    

        # Writing the full matrix in the line corresponding got the time step in the file
        if disdro_name=='Pars1':
            line_str=t.strftime('%Y-%m-%d %H:%M:%S')+';'
            try:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(day_all_disdro[n_time][3][k,l])+','
                line_str=line_str+';'+str(day_all_disdro[n_time][2])
            except:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(np.nan)+','
                line_str=line_str+';'+str(np.nan)
            f_output.write(line_str+'\n')                   

        elif disdro_name=='Pars2':
            line_str=t.strftime('%Y-%m-%d %H:%M:%S')+';'
            try:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(day_all_disdro[n_time][3][k,l])+','
                line_str=line_str+';'+str(day_all_disdro[n_time][2])
            except:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(np.nan)+','
                line_str=line_str+';'+str(np.nan)
            f_output.write(line_str+'\n')                   

        elif disdro_name=='Pars_Rad':
            line_str=t.strftime('%Y-%m-%d %H:%M:%S')+';'
            try:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(day_all_disdro[n_time][3][k,l])+','
                line_str=line_str+';'+str(day_all_disdro[n_time][2])
            except:
                for k in range(32):   # size class
                    for l in range(32):  # velocity class
                        line_str=line_str+str(np.nan)+','
                line_str=line_str+';'+str(np.nan)
            f_output.write(line_str+'\n')                   
        elif disdro_name=='PWS':
            line_str=t.strftime('%Y-%m-%d %H:%M:%S')+';'
            try:
                for k in range(34):   # size class
                    for l in range(34):  # velocity class
                        line_str=line_str+str(day_all_disdro[n_time][18][k,l])+','
                line_str=line_str+';'+str(day_all_disdro[n_time][7])
            except:
                for k in range(34):   # size class
                    for l in range(34):  # velocity class
                        line_str=line_str+str(np.nan)+','
                line_str=line_str+';'+str(np.nan)
            f_output.write(line_str+'\n')    
                
        # Incrementation of time         
        t=t+time_step
        
        
    # Colsing the file
    f_output.close() 




def exporting_R(start_evt,end_evt,disdro_name,path_outputs,path_daily_data_python):
############################################################################################
# Aim : reading daily.npy files and exporting full matrix in .csv files
# 
# Inputs: 
#    - The start and end date of the studied event (datetime objects; should be with 0s and 30s)
#    - disdro_name (a string) : the name of the disdrometer (either 'Pars1', 'Pars2', 'Pars_Rad' or 'PWS')
#    - path_outputs (a string) : path to the folder where outputs will be written
#    - path_daily_data_python (a string): Path to the folder where inputs are stored (assumed to contain folders Pars1, Pars2, Pars_Rad and PWS 
#      containing the data for the disdrometers taking part to the this campaign)
#
# Outputs: 
#     - a file saved : 'Disdro_name'_R_2017_05_18_16_00_00__2017_05_19_01_00_00.csv
#              the start and end of the event is stored in file name
#         One line per time step
#        Date; R in mm/h
#        ex : 2017-05-18 16:00:00;0.0
#        missing data is noted as np.nan
#
# Note : 
#    - data is filtered as in Jaffrain and Berne 2012 (i.e. if velocities should be bounded for a given diameter); see Gires et al. 2015 for more details.
############################################################################################

    str_evt=start_evt.strftime('%Y_%m_%d_%H_%M_%S')+'__'+end_evt.strftime('%Y_%m_%d_%H_%M_%S')
    print(str_evt)

    time_step=timedelta(0,30)

    # Computation of the number of time steps and creating a list with all of them     
    t=start_evt
    N=0
    all_time_steps=list()
    all_time_steps.append(t)
    while t<=end_evt:
        #print(str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second), N)
        t=t+time_step
        all_time_steps.append(t)
        N=N+1
    
    Nb_time_steps=N

    t=start_evt
    N=0 # Index in the time series
    
    R=np.zeros((Nb_time_steps,))    
    
    # Settng the parameters for the various classes of the disdrometers

    # For the PWS
    V_PWS = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12.0,13.6,15.2,17.6,20.8,24.0,27.2))
    V_width_PWS=np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,3.2,3.2))
    D_PWS = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12.0,13.6,15.2,17.6,20.8,24.0,27.2))
    D_width_PWS = np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,3.2,3.2))
    V_theo_PWS = V_D_Lhermitte_1988 (D_PWS)
    S_eff_PWS = 0.004*np.ones(D_PWS.shape)

    # If a the correction for oblateness is implemented, D_PWS and D_width_PWS are modified
    i_correction_PWS=1
    if i_correction_PWS==1:
        D_PWS_buf=np.zeros(D_PWS.shape)
        D_width_PWS_buf=np.zeros(D_PWS.shape)

        coeff=np.array((-0.01361193,0.12089066,0.87497141,0.00989984)) # With Pars and n=1.33
        #coeff=np.array((-0.03302464,0.28672044,0.43507802,0.33835173)) # With Ands and n=1.33
        #coeff=np.array((-0.02259775,0.19820372,0.6685606,0.17391049)) # With Beard and n=1.33
    
        for i in range(34):
            if D_PWS[i]>1:
                D_meas=D_PWS[i]**(1/1.25)
                D_PWS_buf[i]=coeff[0]*D_meas**3+coeff[1]*D_meas**2+coeff[2]*D_meas+coeff[3]
            else : 
                D_PWS_buf[i]=D_PWS[i]

        for i in range(34):
            D_min=D_PWS[i]-D_width_PWS[i]/2
            if D_min > 1:
                D_min=D_min**(1/1.25)
                D_min_buf=coeff[0]*D_min**3+coeff[1]*D_min**2+coeff[2]*D_min+coeff[3]
            else:
                D_min_buf=D_min
            D_max=D_PWS[i]+D_width_PWS[i]/2
            if D_max > 1:
                D_max=D_max**(1/1.25)
                D_max_buf=coeff[0]*D_max**3+coeff[1]*D_max**2+coeff[2]*D_max+coeff[3]
            else:
                D_max_buf=D_max
            D_width_PWS_buf[i]=D_max_buf-D_min_buf
      
        
        D_PWS=D_PWS_buf
        D_width_PWS=D_width_PWS_buf
        V_theo_PWS = V_D_Lhermitte_1988 (D_PWS)



    # For the Pars
    V_Pars = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8))
    V_width_Pars=np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2))    
    D_Pars=np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.750,3.250,3.750,4.250,4.750,5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5])
    D_width_Pars=np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.250,0.250,0.250,0.250,0.250,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,2,3,3])
    V_theo_Pars = V_D_Lhermitte_1988 (D_Pars)
    S_eff_Pars = 10**(-6)*180*(30-D_Pars/2)
    

    # Definition of the filter based on speed (as in Jaffrain et al.)
    filter_vel_PWS=np.zeros((34,34))
    for i in range(30): # Class of size
        for j in range(34): # Class of speed
            if np.abs(V_PWS[j]-V_theo_PWS[i])<=0.6*V_theo_PWS[i]:
                filter_vel_PWS[i,j]=1
            else:
                filter_vel_PWS[i,j]=0

    filter_vel_Pars=np.ones((32,32))
    for i in range(32): # Class of size
        for j in range(32): # Class of speed
            if np.abs(V_Pars[j]-V_theo_Pars[i])<=0.6*V_theo_Pars[i]:
                filter_vel_Pars[i,j]=1
            else:
                filter_vel_Pars[i,j]=0            

    # Loop on time steps
    while t<=end_evt:
        # Retrieving the name of the time step
        time_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second)
        #print(time_str)
        day_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)
        day_str=t.strftime('%Y%m%d')
        if t.second==0:
            n_time=120*t.hour+2*t.minute-1
        elif t.second==30:
            n_time=120*t.hour+2*t.minute
        else :
            print('Issue on t.second in extracting_data_one_event')
        if n_time==-1: # handling the case 00:00:00
            n_time=2879

        # Selecting the correct daily file to look for the disdrometer data
        if t==start_evt:
            try :
                day_str_file=day_str
                f=open(path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro = pickle.load(f) 
                f.close()
            except:
                print('File : '+path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy is missing')

           
        if day_str != day_str_file:
            day_str_file=day_str
            try :
                f=open(path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro = pickle.load(f) 
                f.close()
            except:
                day_all_disdro=list()
                print('File : '+path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy is missing')          

        # Computing the rain rate for the corresponding time step
        if disdro_name=='Pars1':
            try:
                full_map = day_all_disdro[n_time][3][0:32,0:32]*filter_vel_Pars
                for i in range(26) : # Loop on the sizes
                    for j in range(30) : # Loop on the velocity
                        R[N] = R[N] + 3.14159*120*full_map[i,j]*D_Pars[i]**3/(6*S_eff_Pars[i]*10**6)    
            except:
                R[N]=np.nan

        elif disdro_name=='Pars2':
            try:
                full_map = day_all_disdro[n_time][3][0:32,0:32]*filter_vel_Pars
                for i in range(26) : # Loop on the sizes
                    for j in range(30) : # Loop on the velocity
                        R[N] = R[N] + 3.14159*120*full_map[i,j]*D_Pars[i]**3/(6*S_eff_Pars[i]*10**6)    
            except:
                R[N]=np.nan

        elif disdro_name=='Pars_Rad':
            try:
                full_map = day_all_disdro[n_time][3][0:32,0:32]*filter_vel_Pars
                for i in range(26) : # Loop on the sizes
                    for j in range(30) : # Loop on the velocity
                        R[N] = R[N] + 3.14159*120*full_map[i,j]*D_Pars[i]**3/(6*S_eff_Pars[i]*10**6)    
            except:
                R[N]=np.nan
               
        elif disdro_name=='PWS':
            try:
                full_map = day_all_disdro[n_time][18][0:34,0:34]*filter_vel_PWS*10/9 # The 10/9 is for the correction ...
                for i in range(29) : # Loop on the sizes
                    for j in range(29) : # Loop on the velocity
                        R[N] = R[N] + 3.14159*120*full_map[i,j]*D_PWS[i]**3/(6*S_eff_PWS[i]*10**6)
            except:
                R[N]=np.nan
        
        
        # Incrementation of time         
        t=t+time_step
        N=N+1
        

        
    # Writing the file
    f = open(path_outputs+disdro_name+'_R_30_sec_'+start_evt.strftime('%Y_%m_%d_%H_%M_%S')+'__'+end_evt.strftime('%Y_%m_%d_%H_%M_%S')+'.csv', 'w')
    for N in range(Nb_time_steps):
        f.write(all_time_steps[N].strftime('%Y-%m-%d %H:%M:%S')+';'+str(R[N])+'\n')
    f.close()
    
    
    return R



def exporting_T(start_evt,end_evt,disdro_name,path_outputs,path_daily_data_python):
############################################################################################
# Aim : reading daily.npy files and exporting full matrix in .csv files
# 
# Inputs: 
#    - The start and end date of the studied event (datetime objects; should be with 0s and 30s)
#    - disdro_name (a string) : the name of the disdrometer (either 'Pars1', 'Pars2', 'Pars_Rad' or 'PWS')
#    - path_outputs (a string) : path to the folder where outputs will be written
#    - path_daily_data_python (a string): Path to the folder where inputs are stored (assumed to contain folders Pars1, Pars2, Pars_Rad and PWS 
#      containing the data for the disdrometers taking part to the this campaign)
#
# Outputs: 
#     - a file saved : 'Disdro_name'_R_2017_05_18_16_00_00__2017_05_19_01_00_00.csv
#              the start and end of the event is stored in file name
#         One line per time step
#        Date; T (in °C)
#        ex : 2017-05-18 16:00:00;27
#        missing data is noted as np.nan
#
# Note : 
#    - data is filtered as in Jaffrain and Berne 2012 (i.e. if velocities should be bounded for a given diameter); see Gires et al. 2015 for more details.
############################################################################################

    str_evt=start_evt.strftime('%Y_%m_%d_%H_%M_%S')+'__'+end_evt.strftime('%Y_%m_%d_%H_%M_%S')
    print(str_evt)

    time_step=timedelta(0,30)

    # Computation of the number of time steps and creating a list with all of them     
    t=start_evt
    N=0
    all_time_steps=list()
    all_time_steps.append(t)
    while t<=end_evt:
        #print(str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second), N)
        t=t+time_step
        all_time_steps.append(t)
        N=N+1
    
    Nb_time_steps=N

    t=start_evt
    N=0 # Index in the time series
    
    T=np.zeros((Nb_time_steps,))    
    

    # Loop on time steps
    while t<=end_evt:
        # Retrieving the name of the time step
        time_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second)
        #print(time_str)
        day_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)
        day_str=t.strftime('%Y%m%d')
        if t.second==0:
            n_time=120*t.hour+2*t.minute-1
        elif t.second==30:
            n_time=120*t.hour+2*t.minute
        else :
            print('Issue on t.second in extracting_data_one_event')
        if n_time==-1: # handling the case 00:00:00
            n_time=2879

        # Selecting the correct daily file to look for the disdrometer data
        if t==start_evt:
            try :
                day_str_file=day_str
                f=open(path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro = pickle.load(f) 
                f.close()
            except:
                print('File : '+path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy is missing')

           
        if day_str != day_str_file:
            day_str_file=day_str
            try :
                f=open(path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro = pickle.load(f) 
                f.close()
            except:
                day_all_disdro=list()
                print('File : '+path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy is missing')          

        # Computing the rain rate for the corresponding time step
        if disdro_name=='Pars1':
            try:
                T[N]=day_all_disdro[n_time][2]            
            except:
                T[N]=np.nan

        elif disdro_name=='Pars2':
            try:
                T[N]=day_all_disdro[n_time][2]            
            except:
                T[N]=np.nan

        elif disdro_name=='Pars_Rad':
            try:
                T[N]=day_all_disdro[n_time][2]            
            except:
                T[N]=np.nan    
                
        elif disdro_name=='PWS':
            try:
                T[N]=day_all_disdro[n_time][7]            
            except:
                T[N]=np.nan        
        
        # Incrementation of time         
        t=t+time_step
        N=N+1
        

        
    # Writing the file
    f = open(path_outputs+disdro_name+'_T_30_sec_'+start_evt.strftime('%Y_%m_%d_%H_%M_%S')+'__'+end_evt.strftime('%Y_%m_%d_%H_%M_%S')+'.csv', 'w')
    for N in range(Nb_time_steps):
        f.write(all_time_steps[N].strftime('%Y-%m-%d %H:%M:%S')+';'+str(T[N])+'\n')
    f.close()
    
    
    return T





def Generation_daily_data_python_Carnot_1(day_str,path_inputs,path_outputs):
############################################################################################
# Aim : reading raw data (i.e. .txt file generated by each disdrometer for each 30s time step) and generating
#       one .npy file per day and disdrometers used in the Carnot_1 campaign containing all the retrieved data
#
# Inputs : 
#    - day_str (a string '%Y%m%d' ex : 20160110) corresponding to the day for which the data should be generated
#    - path_inputs (a string) : Path to the input files folder (where daily .zip files are stored)
#      This folder contains at least these three folders; one for each disdrometers
#          - PWS
#          - Pars1
#          - Pars2
#           In each folder a .zip file (Raw_pars2_20160110.zip) for each day containing the .txt file (one each 30s) corresponding to raw data
#    - path_outputs (a string) : Path to the output files folder (where daily python files are stored)
#      This folder contains at least these three folders; one for each disdrometer
#          - PWS
#          - Pars1
#          - Pars2
##############################################################################################

    list_missing_time_step_PWS=list()
    list_missing_time_step_Pars1=list()
    list_missing_time_step_Pars2=list()
    
    time_step=timedelta(0,30)

#    print('From Generation_daily_data_python_HMCO_SIRTA')
#    print(day_str)
#    print(path_inputs)
#    print(path_outputs)

    ################
    # For PWS
    ################
    print('For PWS')
    
    day_all_disdro_PWS=list()    
    # Unzipping the raw data for the studied day
    try:
        subprocess.call('cp '+path_inputs+'PWS/Raw_PWS_'+day_str+'.zip .',shell=True)
        subprocess.call('unzip -q Raw_PWS_'+day_str+'.zip ',shell=True)
    except:
        print('Error in copying the unzipping Raw_PWS_'+day_str+'.zip file')
    
    # Generating the .npy file
    t_start=datetime.strptime(day_str+'_0_0_0','%Y%m%d_%H_%M_%S')
    for n_time in range(2880): # Loop number on the 30 s time step in a day
        t=t_start+n_time*time_step
        #print(t)
        time_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second)
        # Reading the data from the PWS100    
        PWS=list()  # This list contains all the data from a time step
        try:    
            f = open('Raw_PWS_'+time_str+'.txt','r')
            txt = f.readline()
            txt.replace(' 00 ',' 0 0 ')
            txt.replace('  ',' 0 ')        
            txt=txt[1:(len(txt)-7)]
            buf=txt.split(sep=' ')
            PWS.append('PWS')                             # PWS[0] = Sensor ID
            PWS.append(eval(buf[2]))                      # PWS[1] = Average visibility (m) (meesage field 20)
            PWS.append(eval(buf[3]))                      # PWS[2] = Present Weather Code (WMO)  (meesage field 21)
            PWS.append(buf[4])                            # PWS[3] = Present Weather Code (METAR)  (meesage field 21)
            PWS.append(buf[5])                            # PWS[4] = Present Weather Code (NWS)  (meesage field 23)
            PWS.append([float(i) for i in buf[6:22]])     # PWS[5] = Alarms (message field (24))
            PWS.append(buf[22])                           # PWS[6] = Fault status of PWS100 (message field 25)
            PWS.append(buf[23])                           # PWS[7] = Temperature (°C) (message field 30)
            PWS.append(buf[24])                           # PWS[8] = Sampled relative humidity (%) (message field 30)
            PWS.append(buf[25])                           # PWS[9] = Average wetbulb temperature (°C)(message field 30)
            PWS.append(buf[26])                           # PWS[10] = Maximum temperature (°C)(message field 31)
            PWS.append(buf[27])                           # PWS[11] = Minimum temperature (°C)(message field 31)
            PWS.append(buf[28])                           # PWS[12] = Precipitation rate (mm/h)(message field 40)                    
            PWS.append(buf[29])                           # PWS[13] = Precipitation accumulation (mm/h)(message field 40)
            PWS.append(np.array([float(i) for i in buf[30:330]])) # PWS[14] = Drop size distribution (message field 42)
            PWS.append(buf[330])                           # PWS[15] = Average velocity (mm/s)(message field 43)
            PWS.append(buf[331])                           # PWS[16] = Average size (mm/h)(message field 43)
            PWS.append(np.array([float(i) for i in buf[332:343]])) # PWS[17] = Type distribution (message field 44)
         
            map_list=[float(i) for i in buf[343:1499]]
            map_PWS=np.zeros((34,34))
            for k in range(34):   # size class
                for l in range(34):  # velocity class
                    map_PWS[k,l]=map_list[k*34+l]
            PWS.append(map_PWS)                             # PWS[18] = Campbell Scientifific standard size/velocity map (34*34) (message field 47)
            
            PWS.append(np.array([float(i) for i in buf[1499:1549]])) # PWS[19] = Peak to pedestal ratio distribution histogram (message field 48)
            
            f.close()

        except:
            PWS=list()
            list_missing_time_step_PWS.append(t.strftime('%Y-%m-%d %H:%M:%S'))
        
        day_all_disdro_PWS.append(PWS)    
        

    # Removing the .txt file corresponding to each 30 s time step
    try:
        subprocess.call('rm -f Raw_PWS_*',shell=True)
    except:
        print('Error in removing Raw_PWS_* txt file')

    # Saving the lists
    f =open(path_outputs+'/PWS/PWS_raw_data_'+day_str+'.npy', 'wb') 
    pickle.dump(day_all_disdro_PWS, f) 
    f.close()


    ################
    # For Pars1
    ################
    print('For Pars1')

    day_all_disdro_Pars1=list()

    # Unzipping the raw data for the studied day
    try:
        subprocess.call('cp '+path_inputs+'Pars1/Raw_pars1_'+day_str+'.zip .',shell=True)
        subprocess.call('unzip -q Raw_pars1_'+day_str+'.zip ',shell=True)
    except:
        print('Error in copying the unzipping Raw_pars1_'+day_str+'.zip file')
        
    # Generating the .npy file
    t_start=datetime.strptime(day_str+'_0_0_0','%Y%m%d_%H_%M_%S')
    for n_time in range(2880): # Loop number on the 30 s time step in a day
        t=t_start+n_time*time_step
        #print(t)
        time_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second)
                
        # Reading the data from the Parsivel_1    
        Pars1=list()  # This list contains all the data from a time step
        try:
            f = open('Raw_pars1_'+time_str+'.txt','r')    
            list_txt = f.readlines()
            Pars1.append('Pars1')                           # Pars1[0] = Sensor ID
            Pars1.append(eval(list_txt[3][3:-1]))           # Pars1[1] = Precipitation rate (mm/h) (message field 01)    
            Pars1.append(float(list_txt[25][3:-1]))         # Pars1[2] = Temperature in the sensor (°C) (message field 12)    
        

        
            buf=list_txt[81][3:-2].split(sep=';')
            map_list=[float(i) for i in buf]
            map_pars=np.zeros((32,32))
            for k in range(32):   # size class
                for l in range(32):  # velocity class
                    map_pars[k,l]=map_list[l*32+k]
            Pars1.append(map_pars)                         # PWS[3] = OTT standard size/velocity map (34*34) (message field 93)
        
            f.close()
        except: 
            Pars1=list()
            list_missing_time_step_Pars1.append(t.strftime('%Y-%m-%d %H:%M:%S'))
            
        day_all_disdro_Pars1.append(Pars1) 
     
    # Removing the .txt file corresponding to each 30 s time step
    try:
        subprocess.call('rm -f Raw_pars1_*',shell=True)
    except:
        print('Error in removing Raw_pars1_* txt file')

    # Saving the lists 

    f =open(path_outputs+'/Pars1/Pars1_raw_data_'+day_str+'.npy', 'wb') 
    pickle.dump(day_all_disdro_Pars1, f) 
    f.close()



    ################
    # For Pars2
    ################
    print('For Pars2')

    # Unzipping the raw data for the studied day
    try:
        subprocess.call('cp '+path_inputs+'Pars2/Raw_pars2_'+day_str+'.zip .',shell=True)
        subprocess.call('unzip -q Raw_pars2_'+day_str+'.zip ',shell=True)
    except:
        print('Error in copying the unzipping Raw_pars2_'+day_str+'.zip file')
        
    # Generating the .npy file
    day_all_disdro_Pars2=list()
    t_start=datetime.strptime(day_str+'_0_0_0','%Y%m%d_%H_%M_%S')
    for n_time in range(2880): # Loop number on the 30 s time step in a day
        t=t_start+n_time*time_step
        #print(t)
        time_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second)

 
        # Reading the data from the Parsivel_2    
        Pars2=list() # This list contains all the data from a time step
        try:
            f = open('Raw_pars2_'+time_str+'.txt','r')    
            list_txt = f.readlines()
            Pars2.append('Pars2')                           # Pars2[0] = Sensor ID
            Pars2.append(eval(list_txt[3][3:-1]))           # Pars2[1] = Precipitation rate (mm/h) (message field 01)    
            Pars2.append(float(list_txt[25][3:-1]))         # Pars2[2] = Temperature in the sensor (°C) (message field 12)    
                
            buf=list_txt[81][3:-2].split(sep=';')
            map_list=[float(i) for i in buf]
            map_pars=np.zeros((32,32))
            for k in range(32):   # size class
                for l in range(32):  # velocity class
                    map_pars[k,l]=map_list[l*32+k]
            Pars2.append(map_pars)                         # PWS[3] = OTT standard size/velocity map (34*34) (message field 93)
        
            f.close()
        except: 
            Pars2=list()
            list_missing_time_step_Pars2.append(t.strftime('%Y-%m-%d %H:%M:%S'))
            
        day_all_disdro_Pars2.append(Pars2) 
   
    
    # Removing the .txt file corresponding to each 30 s time step
    try:
        subprocess.call('rm -f Raw_pars2_*',shell=True)
    except:
        print('Error in removing Raw_pars2_* txt file')

    # Saving the lists 

    f =open(path_outputs+'/Pars2/Pars2_raw_data_'+day_str+'.npy', 'wb') 
    pickle.dump(day_all_disdro_Pars2, f) 
    f.close()


    return list_missing_time_step_PWS, list_missing_time_step_Pars1, list_missing_time_step_Pars2



def exporting_KE(start_evt,end_evt,disdro_name,path_outputs,path_daily_data_python):
############################################################################################
# Aim : reading daily.npy files and exporting full matrix in .csv files
# 
# Inputs: 
#    - The start and end date of the studied event (datetime objects; should be with 0s and 30s)
#    - disdro_name (a string) : the name of the disdrometer (either 'Pars1', 'Pars2', 'Pars_Rad' or 'PWS')
#    - path_outputs (a string) : path to the folder where outputs will be written
#    - path_daily_data_python (a string): Path to the folder where inputs are stored (assumed to contain folders Pars1, Pars2, Pars_Rad and PWS 
#      containing the data for the disdrometers taking part to the this campaign)
#
# Outputs: 
#     - a file saved : 'Disdro_name'_R_2017_05_18_16_00_00__2017_05_19_01_00_00.csv
#              the start and end of the event is stored in file name
#         One line per time step
#        Date; R in mm/h
#        ex : 2017-05-18 16:00:00;0.0
#        missing data is noted as np.nan
#
# Note : 
#    - data is filtered as in Jaffrain and Berne 2012 (i.e. if velocities should be bounded for a given diameter); see Gires et al. 2015 for more details.
############################################################################################

    str_evt=start_evt.strftime('%Y_%m_%d_%H_%M_%S')+'__'+end_evt.strftime('%Y_%m_%d_%H_%M_%S')
    print(str_evt)

    time_step=timedelta(0,30)

    # Computation of the number of time steps and creating a list with all of them     
    t=start_evt
    N=0
    all_time_steps=list()
    all_time_steps.append(t)
    while t<=end_evt:
        #print(str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second), N)
        t=t+time_step
        all_time_steps.append(t)
        N=N+1
    
    Nb_time_steps=N

    t=start_evt
    N=0 # Index in the time series
    
    KE=np.zeros((Nb_time_steps,))    
    
    # Settng the parameters for the various classes of the disdrometers

    # For the PWS
    V_PWS = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12.0,13.6,15.2,17.6,20.8,24.0,27.2))
    V_width_PWS=np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,3.2,3.2))
    D_PWS = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12.0,13.6,15.2,17.6,20.8,24.0,27.2))
    D_width_PWS = np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,3.2,3.2))
    V_theo_PWS = V_D_Lhermitte_1988 (D_PWS)
    S_eff_PWS = 0.004*np.ones(D_PWS.shape)

    # If a the correction for oblateness is implemented, D_PWS and D_width_PWS are modified
    i_correction_PWS=1
    if i_correction_PWS==1:
        D_PWS_buf=np.zeros(D_PWS.shape)
        D_width_PWS_buf=np.zeros(D_PWS.shape)

        coeff=np.array((-0.01361193,0.12089066,0.87497141,0.00989984)) # With Pars and n=1.33
        #coeff=np.array((-0.03302464,0.28672044,0.43507802,0.33835173)) # With Ands and n=1.33
        #coeff=np.array((-0.02259775,0.19820372,0.6685606,0.17391049)) # With Beard and n=1.33
    
        for i in range(34):
            if D_PWS[i]>1:
                D_meas=D_PWS[i]**(1/1.25)
                D_PWS_buf[i]=coeff[0]*D_meas**3+coeff[1]*D_meas**2+coeff[2]*D_meas+coeff[3]
            else : 
                D_PWS_buf[i]=D_PWS[i]

        for i in range(34):
            D_min=D_PWS[i]-D_width_PWS[i]/2
            if D_min > 1:
                D_min=D_min**(1/1.25)
                D_min_buf=coeff[0]*D_min**3+coeff[1]*D_min**2+coeff[2]*D_min+coeff[3]
            else:
                D_min_buf=D_min
            D_max=D_PWS[i]+D_width_PWS[i]/2
            if D_max > 1:
                D_max=D_max**(1/1.25)
                D_max_buf=coeff[0]*D_max**3+coeff[1]*D_max**2+coeff[2]*D_max+coeff[3]
            else:
                D_max_buf=D_max
            D_width_PWS_buf[i]=D_max_buf-D_min_buf
      
        
        D_PWS=D_PWS_buf
        D_width_PWS=D_width_PWS_buf
        V_theo_PWS = V_D_Lhermitte_1988 (D_PWS)



    # For the Pars
    V_Pars = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8))
    V_width_Pars=np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2))    
    D_Pars=np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.750,3.250,3.750,4.250,4.750,5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5])
    D_width_Pars=np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.250,0.250,0.250,0.250,0.250,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,2,3,3])
    V_theo_Pars = V_D_Lhermitte_1988 (D_Pars)
    S_eff_Pars = 10**(-6)*180*(30-D_Pars/2)
    

#    # Definition of the filter based on speed (as in Jaffrain et al.)
#    filter_vel_PWS=np.zeros((34,34))
#    for i in range(30): # Class of size
#        for j in range(34): # Class of speed
#            if np.abs(V_PWS[j]-V_theo_PWS[i])<=0.6*V_theo_PWS[i]:
#                filter_vel_PWS[i,j]=1
#            else:
#                filter_vel_PWS[i,j]=0
#
#    filter_vel_Pars=np.ones((32,32))
#    for i in range(32): # Class of size
#        for j in range(32): # Class of speed
#            if np.abs(V_Pars[j]-V_theo_Pars[i])<=0.6*V_theo_Pars[i]:
#                filter_vel_Pars[i,j]=1
#            else:
#                filter_vel_Pars[i,j]=0            

    # No filtera are implemented in this case
    filter_vel_PWS=np.zeros((34,34))
    filter_vel_Pars=np.ones((32,32))

    rho_wat = 1000   #mass in Kg per unit volume of water  (kg/m3)
    
    # Loop on time steps
    while t<=end_evt:
        # Retrieving the name of the time step
        time_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)+'_'+str(t.hour)+'_'+str(t.minute)+'_'+str(t.second)
        #print(time_str)
        day_str=str(t.year)+'_'+str(t.month)+'_'+str(t.day)
        day_str=t.strftime('%Y%m%d')
        if t.second==0:
            n_time=120*t.hour+2*t.minute-1
        elif t.second==30:
            n_time=120*t.hour+2*t.minute
        else :
            print('Issue on t.second in extracting_data_one_event')
        if n_time==-1: # handling the case 00:00:00
            n_time=2879

        # Selecting the correct daily file to look for the disdrometer data
        if t==start_evt:
            try :
                day_str_file=day_str
                f=open(path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro = pickle.load(f) 
                f.close()
            except:
                print('File : '+path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy is missing')

           
        if day_str != day_str_file:
            day_str_file=day_str
            try :
                f=open(path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy', 'rb') 
                day_all_disdro = pickle.load(f) 
                f.close()
            except:
                day_all_disdro=list()
                print('File : '+path_daily_data_python+disdro_name+'/'+disdro_name+'_raw_data_'+day_str+'.npy is missing')          

        # Computing the rain rate for the corresponding time step
        if disdro_name=='Pars1' or disdro_name == 'Pars_RW_turb_1':
            try:
                full_map = day_all_disdro[n_time][3][0:32,0:32]*filter_vel_Pars
                for i in range(26) : # Loop on the sizes
                    for j in range(30) : # Loop on the velocity
                        KE[N] = KE[N] + full_map[i,j] * 0.5 * 4*np.pi/3 *(D_Pars[i]/2)**3*V_Pars[j]**2*rho_wat*10**(-9)/ (S_eff_Pars[i]) *120
            except:
                KE[N]=np.nan

        elif disdro_name=='Pars2' or disdro_name == 'Pars_RW_turb_2':
            try:
                full_map = day_all_disdro[n_time][3][0:32,0:32]*filter_vel_Pars
                for i in range(26) : # Loop on the sizes
                    for j in range(30) : # Loop on the velocity
                        KE[N] = KE[N] + full_map[i,j] * 0.5 * 4*np.pi/3 *(D_Pars[i]/2)**3*V_Pars[j]**2*rho_wat*10**(-9)/ (S_eff_Pars[i]) *120    
            except:
                KE[N]=np.nan

        elif disdro_name=='Pars_Rad':
            try:
                full_map = day_all_disdro[n_time][3][0:32,0:32]*filter_vel_Pars
                for i in range(26) : # Loop on the sizes
                    for j in range(30) : # Loop on the velocity
                        KE[N] = KE[N] + full_map[i,j] * 0.5 * 4*np.pi/3 *(D_Pars[i]/2)**3*V_Pars[j]**2*rho_wat*10**(-9)/ (S_eff_Pars[i]) *120 
            except:
                KE[N]=np.nan
               
        elif disdro_name=='PWS':
            try:
                full_map = day_all_disdro[n_time][18][0:34,0:34]*filter_vel_PWS*10/9 # The 10/9 is for the correction ...
                for i in range(29) : # Loop on the sizes
                    for j in range(29) : # Loop on the velocity
                        KE[N] = KE[N] + full_map[i,j] * 0.5 * 4*np.pi/3 *(D_PWS[i]/2)**3*V_PWS[j]**2*rho_wat*10**(-9)/ (S_eff_PWS[i]) *120
            except:
                KE[N]=np.nan
        
        
        # Incrementation of time         
        t=t+time_step
        N=N+1
        

        
    # Writing the file
    f = open(path_outputs+disdro_name+'_KE_30_sec_'+start_evt.strftime('%Y_%m_%d_%H_%M_%S')+'__'+end_evt.strftime('%Y_%m_%d_%H_%M_%S')+'.csv', 'w')
    for N in range(Nb_time_steps):
        f.write(all_time_steps[N].strftime('%Y-%m-%d %H:%M:%S')+';'+str(KE[N])+'\n')
    f.close()
    
    
    return KE





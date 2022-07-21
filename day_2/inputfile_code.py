#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:51:38 2022

@author: kdoubles
"""
import argparse as ap
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

def input_file_name():
    """
    A parser that allows the user to give a file name and plot that file as
    an outfile, which you can also name. 
    """
 
    parser = ap.ArgumentParser(description = 'Cosine Approximation using Taylor Series')
    parser.add_argument('-infile',\
                      help = 'Enter the file name that is to be graphed. \
                          String type.', type = str)   
    parser.add_argument('-outfile',\
                        help = 'Enter how you want to name the .png. \
                            String type.', type=str)
                        
    args = parser.parse_args()
    return args

def read_ascii_file_data(filename,index=-1):
        """
        Args:
            filename(str):
                name of the file to be called.
            index(int):
                The integer value associated with the variable. Deafult is -1.
        
        Returns:
            year(array):
                Array of year values retrieved from "filename".
            day(array):
                Array of day of the year from values retrieved from "filename".
            hour(array):
                Array of hour from values retrieved from "filename".
            minute(array):
                Array from minute values retrieved from "filename".
            symh(array):
                Array of the SYM-H values retrieved from "filename".
            
        Example:
            read_ascii_file_data("omni_test.lst")
            read_ascii_file_data("omni_test.lst",index=4)

        """
        with open(filename) as f:

            data_dict = {"time": [],
                         "year": [],
                         "day": [],
                         "hour": [],
                         "minute": [],
                         "symh": []} 
          
            for line in f:
                tmp = line.split()
                data_dict["year"].append(int(tmp[0]))
                data_dict["day"].append(int(tmp[1]))
                data_dict["hour"].append(int(tmp[2]))
                data_dict["minute"].append(int(tmp[3]))
                
                #create datetime in each line
                time0 = dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0)\
                    + dt.timedelta(days=int(tmp[1])-1)
                data_dict["time"].append(time0)
                
                data_dict["symh"].append(int(tmp[index]))
                
        return data_dict
   
args = input_file_name()
print(args)

filein = args.infile

file_data = read_ascii_file_data(filein)

time = np.array(file_data['time'])
data = np.array(file_data['symh'])
compare = data < -100
max_SYMH = np.max(data)
#print(max_SYMH)
min_SYMH = np.min(data)
#print(min_SYMH)

max_symh_value = np.where(max_SYMH)
min_symh_value = np.where(min_SYMH)


fig,ax = plt.subplots()
ax.plot(time,data,marker='.',c='gray',
       label = 'All Events',alpha = 0.5)
ax.plot(time[compare],data[compare],marker='+',
        linestyle='',
        c = 'tab:orange',
        label = '< -100 nT',
        alpha = 0.6)
#ax.set_title('SYM-H of March 2013 Storm')
ax.set_xlabel('Year of 2013')
ax.set_ylabel('SYM-H (nT)')
ax.grid(True)
ax.legend()

outfile = args.outfile
print('Writing file:'+outfile)
plt.savefig(outfile)
plt.close()

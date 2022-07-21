#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:47:38 2022

Introduction to opening anf manipulating files using Python

__author__ = 'Kaitlin Doublestein'
__email__ = 'kdoubles@umich.edu'
"""

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

f = open("omni_min_test.lst")
f.close()

"""
with open("omni_min_test.lst") as f:
    #can do action to file here
    example: print(list(f))
        prints the file
"""

""" 
This gives the default encoded

#import locale
#locale.getpreferredencoding(False)

"""

#Read each line of the file
#with open("omni_min_test.lst") as f:
    #line1 = f.readline()
    #line2 = f.readline()
    #line3 = f.readline()
    #print(line1,line2,line3)
    
#prints number of lines selected by nLines value    
#    nLines = 3    
#    for iLine in range(nLines):
#        temp = f.readline()
#        print(temp)

#prints all lines in file
#with open("omni_min_test.lst") as f:
#    for line in f:
#        print(line)
"""
with open("omni_min_test.lst") as f:  
    nLines = 3
    for iLine in range(nLines):
        tmp = f.readline()
    
        
    header=f.readline()
    vars=header.split()
    
    year = []
    day = []
    hour = []
    minute = []
    symh = []
    for line in f:
        tmp = line.split()
        year.append(int(tmp[0]))
        day.append(int(tmp[1]))
        hour.append(int(tmp[2]))
        minute.append(int(tmp[3]))
        symh.append(int(tmp[4]))
    
"""
"""
Can split the values of the lines so that we keep the headings of the variables
but we can get rid of the header.

Use: line.split()
"""
"""
#This can move the str type to a list type!
with open("omni_min_def_data.lst") as f:
    year = []
    day = []
    hour = []
    minute = []
    symh = []
    for line in f:
        tmp = line.split()
        year.append(int(tmp[0]))
        day.append(int(tmp[1]))
        hour.append(int(tmp[2]))
        minute.append(int(tmp[3]))
        symh.append(int(tmp[4]))
        
"""       


#We can make a defintion to assign the filename and index as inputs
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
        #year = []
        #day = []
        #hour = []
        #minute = []
        
       # time = []
        #data = []
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
    
    
file_data = read_ascii_file_data("omni_min_def_data.lst")
#print(file_data['time'])


#plotting
time = np.array(file_data['time'])
data = np.array(file_data['symh'])


max_SYMH = np.argmax(data)
print(max_SYMH)
min_SYMH = np.argmin(data)
print(min_SYMH)

max_symh_value = np.where(max_SYMH)
min_symh_value = np.where(min_SYMH)


compare = data < -100
#print(compare)

fig,ax = plt.subplots()
ax.plot(time,data,marker='.',c='gray',
       label = 'All Events',alpha = 0.5)
ax.plot(time[compare],data[compare],marker='+',
        linestyle='',
        c = 'tab:orange',
        label = '< -100 nT',
        alpha = 0.6)
ax.axvlines(x=max_symh_value)
ax.set_title('SYM-H of March 2013 Storm')
ax.set_xlabel('Year of 2013')
ax.set_ylabel('SYM-H (nT)')
ax.grid(True)
ax.legend()

plt.show()



#    print(datetime_array.isoformat())

#lean_datetime(file_data['time'])
"""
Learning to use the datetime function in Python

Show using the dt.datetime and dt.timedelta that the first date equals the 
second date when two days are added.
"""


#Create with Year, Month, day, hour, minute, second
time1 = dt.datetime(2013,1,3,10,12,30)
time2 = dt.datetime(2013,1,1,10,12,30) + dt.timedelta(days = 2)
#print(time1.date())

lp1 = time1 == time2
lp2 = time1>dt.datetime(2013,1,5,0,0,0)

#print(lp1,lp2)
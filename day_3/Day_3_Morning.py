#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welcome to Space Weather Simulation Summer School Day 3

Today, we will be working with various file types, doing some simple data 
manipulation and data visualization

We will be using a lot of things that have been covered over the last two days 
with minor vairation.

Goal: Getting comfortable with reading and writing data, doing simple data 
manipulation, and visualizing data.

Task: Fill in the cells with the correct codes

@author: Peng Mun Siew
"""

#%% 
"""
This is a code cell that we can use to partition our data. (Similar to Matlab cell)
We hace the options to run the codes cell by cell by using the "Run current cell" button on the top.
"""
#print ("Hello World")

#%%
"""
Writing and reading numpy file
"""
# Importing the required packages
import numpy as np

# Generate a random array of dimension 10 by 5
data_arr = np.random.randn(10,5)
#print(data_arr)

# Save the data_arr variable into a .npy file
np.save('test_np_save.npy', data_arr)
data_arr_loaded = np.load('test_np_save.npy')
#print(np.equal(data_arr,data_arr_loaded))

#%%
"""
Writing and reading numpy zip archive/file
"""
# Generate a second random array of dimension 8 by 1
data_arr2 = np.random.randn(8,1)
#print(data_arr2)

# Save the data_arr and data_arr2 variables into a .npz file
np.savez('test_savez.npz',data_arr,data_arr2)

#Load
npzfile = np.load('test_savez.npz')

#print(npzfile)

#print('Variable names within this file :' , sorted(npzfile.files))

#print(npzfile['arr_0'])
#print((data_arr==npzfile['arr_0']).all())
#print((data_arr2==npzfile['arr_1']).all())
#.any() works for if any of the the values are true, outputs true
#%%
"""
Error and exception
"""
#np.equal(data_arr,npzfile)

# Exception handling, can be use with assertion as well
#try:
    # Python will try to execute any code here, and if there is an exception skip to below 
    #print(np.equal(data_arr,npzfile).all())
#except:
    # Execute this code when there is an exception
    #print("The provided variable is a npz object.")
    #print(np.equal(data_arr,npzfile['arr_0']).all())


#%%
"""
Loading data from Matlab
"""

# Import required packages
import numpy as np
from scipy.io import loadmat

dir_density_Jb2008 = '/Users/kdoubles/Data/JB2008/2002_JB2008_density.mat'

# Load Density Data
try:
    loaded_data = loadmat(dir_density_Jb2008)
    #print (loaded_data)
except:
    print("File not found. Please check your directory")

# Uses key to extract our data of interest
JB2008_dens = loaded_data['densityData']

# The shape command now works
#print(JB2008_dens.shape)

#%%
"""
Data visualization I

Let's visualize the density field for 400 KM at different time.
"""

# Import required packages
import matplotlib.pyplot as plt

# Before we can visualize our density data, we first need to generate the discretization grid of the density data in 3D space. We will be using np.linspace to create evenly sapce data between the limits.

localSolarTimes_JB2008 = np.linspace(0,24,24)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0]
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
nofLat_JB2008 = latitudes_JB2008.shape[0]

# We can also impose additional constratints such as forcing the values to be integers.
time_array_JB2008 = np.linspace(0,8759,20, dtype = int)

# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008,8760), order='F') # Fortran-like index order

alt = 400

hi = np.where(altitudes_JB2008==alt)

fig, axs = plt.subplots(20, figsize=(40, 10*5), sharex=True)
for ik in range(20):
    cs = axs[ik].contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_array_JB2008[ik]].squeeze().T)
    axs[ik].set_title('JB2008 density at 400 km, t = {} hrs'.format(time_array_JB2008[ik]), fontsize=18)
    axs[ik].set_ylabel("Latitudes", fontsize=18)
    axs[ik].tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs,ax=axs[ik])
    cbar.ax.set_ylabel('Density')

axs[ik].set_xlabel("Local Solar Time", fontsize=18)

#print(JB2008_dens_reshaped[:,:,hi,time_array_JB2008[ik]].shape)

#%%
"""
Assignment 1

Can you plot the mean density for each altitude at February 1st, 2002?
"""

# First identify the time index that corresponds to  February 1st, 2002. Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = JB2008_dens_reshaped[:,:,:,time_index]

alt = altitudes_JB2008

mean_dens = np.mean(np.mean(dens_data_feb1, axis=0),axis=0)
#print(mean_dens.shape)

#print(mean_dens)
"""
plt.plot(alt,mean_dens)
plt.title('Mean Density vs Altitude')
plt.yscale('log')
plt.xlabel('Altitude [km]')
plt.ylabel('Density []')
plt.grid()

"""
#%%
"""
Data Visualization II

Now, let's us work with density data from TIE-GCM instead, and plot the density field at 310km
"""
# Import required packages
import h5py
loaded_data = h5py.File('/Users/kdoubles/Data/TIEGCM/2002_TIEGCM_density.mat')

#%%
#print('Key within database:', list(loaded_data.keys()))

tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T
altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten()
latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
nofAlt_tiegcm = altitudes_tiegcm.shape[0]
nofLat_tiegcm = latitudes_tiegcm.shape[0]
nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]

tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')
time_array_tiegcm = np.linspace(0, 8759,20,dtype=int)

alt_tiegcm = 310
hi = np.where(alt_tiegcm==altitudes_tiegcm)
"""
fig, axs = plt.subplots(20, figsize=(40, 10*5), sharex=True)
for ik in range(20):
    ts = axs[ik].contourf(localSolarTimes_tiegcm, latitudes_tiegcm, tiegcm_dens_reshaped[:,:,hi,time_array_JB2008[ik]].squeeze().T)
    axs[ik].set_title('TIEGCM density at 310 km, t = {} hrs'.format(time_array_tiegcm[ik]), fontsize=18)
    axs[ik].set_ylabel("Latitudes", fontsize=18)
    axs[ik].tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs,ax=axs[ik])
    cbar.ax.set_ylabel('Density')
 """   
#%%

time_index = 31*24
dens_data_feb1_tiegcm = tiegcm_dens_reshaped[:,:,:,time_index]

alt_tiegcm = altitudes_tiegcm

mean_dens_tiegcm = np.mean(np.mean(dens_data_feb1_tiegcm, axis=0),axis=0)
#print(mean_dens_tiegcm.shape)

#print(mean_dens)
"""
plt.plot(alt_tiegcm,mean_dens_tiegcm)
plt.title('Mean Density vs Altitude (TIEGCM)')
plt.yscale('log')
plt.xlabel('Altitude [km]')
plt.ylabel('Density []')
plt.grid()
plt.show()
"""
#%%

"""
plt.plot(alt_tiegcm,mean_dens_tiegcm, 'b-',label='TIE-GCM')
plt.plot(alt,mean_dens,'r--',label='JB2008')
plt.title('Mean Density vs Altitude (TIE-GCM)')
plt.yscale('log')
plt.xlabel('Altitude [km]')
plt.ylabel('Density []')
plt.legend()
plt.grid()
plt.show()
"""

#%%
"""
Data Interpolation (1D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy import interpolate

# Let's first create some data for interpolation
x = np.arange(0, 10)
y = np.exp(-x/3.0)

interp_func_1D = interpolate.interp1d(x,y)

xnew = np.arange(0,9,0.1)
ynew = interp_func_1D(xnew)
"""
plt.subplots(1,figsize=(10,6))
plt.plot(x,y,'o',xnew,ynew,'*',linewidth = 2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('1D Interpolation',fontsize=18)
plt.legend(['Intial Points','Interpolated Points'])
plt.grid()
"""
#%%

interp_func_1D_2 = interpolate.interp1d(x,y,kind='cubic')
interp_func_1D_3 = interpolate.interp1d(x,y,kind='quadratic')


xnew2 = np.arange(0,9,0.1)
ynew2 = interp_func_1D_2(xnew2)

xnew3 = np.arange(0,9,0.1)
ynew3 = interp_func_1D_3(xnew3)
"""
plt.subplots(1,figsize=(10,6))
plt.plot(x,y,'o',xnew,ynew,'*',xnew2,ynew2,'.',xnew3,ynew3,'-.',linewidth = 2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('1D Interpolation',fontsize=18)
plt.legend(['Intial Points','Interpolated Points-linear','Interpolated Points-cubic',\
            'Interpolated Points-quadratic'])
plt.grid()
"""

#%%
"""
Data Interpolation (3D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator

# First create a set of sample data that we will be using 3D interpolation on
def function_1(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

sample_data = function_1(xg, yg, zg)
interpolated_function_1 = RegularGridInterpolator((x, y, z), sample_data)
pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
print(interpolated_function_1(pts))
print(function_1(pts[:,0],pts[:,1],pts[:,2]))


#%%
"""
Saving mat file

Now, let's us look at how to we can save our data into a mat file
"""
# Import required packages
from scipy.io import savemat

a = np.arange(20)
mdic = {"a": a, "label": "experiment","v":1} # Using dictionary to store multiple variables
savemat("matlab_matrix.mat", mdic)

#%%
"""
Assignment 2 (a)

The two data that we have been working on today have different discretization grid.
Use 3D interpolation to evaluate the TIE-GCM density field at 400 KM on February 1st, 2002, with the discretized grid used for the JB2008 ((nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008).
"""

import numpy as np

x_TIE= localSolarTimes_tiegcm
y_TIE = latitudes_tiegcm
z_TIE = altitudes_tiegcm
xyz = np.meshgrid(x_TIE, y_TIE, z_TIE)

interpolated_TIE_func = RegularGridInterpolator((x_TIE, y_TIE, z_TIE), dens_data_feb1_tiegcm, bounds_error=False,fill_value=None)

#%%
"""
Assignment 2 (b)

Now, let's find the difference between both density models and plot out this difference in a contour plot.
"""

dens_field = np.zeros((24,20))

for lst_i in range(24):
    for lat_i in range(20):
        dens_field[lst_i,lat_i]=interpolated_TIE_func((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],400))

fig, axs = plt.subplots(2, figsize=(20,30), sharex=True)
    
axs[0].contourf(localSolarTimes_JB2008, latitudes_JB2008, dens_field.T)
axs[0].set_title('JB2008 density at 400 km, t = {} hrs'.format(time_array_JB2008[0]), fontsize=18)
axs[0].set_ylabel("Latitudes", fontsize=18)
axs[0].tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig.colorbar(cs,ax=axs[0])
cbar.ax.set_ylabel('Density')

axs[0].set_xlabel("Local Solar Time", fontsize=18)

axs[1].contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_array_JB2008[1]].squeeze().T)
axs[1].set_title('JB2008 density at 400 km, t = {} hrs'.format(time_array_JB2008[1]), fontsize=18)
axs[1].set_ylabel("Latitudes", fontsize=18)
axs[1].tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig.colorbar(cs,ax=axs[1])
cbar.ax.set_ylabel('Density')

axs[1].set_xlabel("Local Solar Time", fontsize=18)


#%%
import argparse as ap

def input_altitude():
    """
    A parser that allows the user to give a file name and plot that file as
    an outfile, which you can also name. 
    """
 
    parser = ap.ArgumentParser(description = 'Input the altitude to get the TIE-GCM density contour plot.')
    parser.add_argument('-altitude',\
                      help = 'Enter the altitude that you want the TIE-GCM density contour plot of. \
                          Deafult is 400km. Float type.', type = float,default = 400)  
    parser.add_argument('-day',\
                        help = 'Enter the day of the year that you would like to view. \
                            String type.', type=int)
    parser.add_argument('-outfile',\
                        help = 'Enter how you want to name the .png. \
                            String type.', type=str)
                            
    args = parser.parse_args()
    return args


dens_field_alt_select = np.zeros((24,20))
            
args = input_altitude()
print(args)

day_of_year = args.day

outfile_name = args.outfile
altitude_tiegcm_selected = args.altitude


x_TIE= localSolarTimes_tiegcm
y_TIE = latitudes_tiegcm
z_TIE = altitudes_tiegcm
xyz = np.meshgrid(x_TIE, y_TIE, z_TIE)

interpolated_TIE_func = RegularGridInterpolator((x_TIE, y_TIE, z_TIE), dens_data_feb1_tiegcm, bounds_error=False,fill_value=None)

dens_field_alt_select = np.zeros((24,20))
for lst_i in range(24):
    for lat_i in range(20):
        dens_field_alt_select[lst_i,lat_i]=interpolated_TIE_func((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],altitude_tiegcm_selected))

print(dens_field_alt_select.shape)

fig, axs = plt.subplots(figsize=(30,20), sharex=True)
    
vs = axs.contourf(localSolarTimes_JB2008, latitudes_JB2008, dens_field_alt_select.T)
axs.set_title('TIE-GCM density contour at selected altitude', fontsize=18)
axs.set_ylabel("Latitudes", fontsize=18)
axs.tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig.colorbar(vs,ax=axs)
cbar.ax.set_ylabel('Density')

axs.set_xlabel("Local Solar Time", fontsize=18)

print('Writing file:'+ outfile_name)
plt.savefig(outfile_name)
plt.close()

#%%
"""
Assignment 2 (c)

In the scientific field, it is sometime more useful to plot the differences in terms of mean absolute percentage difference/error (MAPE). Let's plot the MAPE for this scenario.
"""






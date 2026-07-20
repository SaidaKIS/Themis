from cProfile import label

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import glob
import sys
import os

#Compare both calculations

hdul1 = fits.open("/run/media/sdiaz/usb/THEMIS/260522/Stokes/t014_b0606_sp_20260522_080309_s4.fts")
header = hdul1[0].header
stk_data1 = hdul1[0].data

hdul2 = fits.open('stokes_output_s4.fits')
header = hdul2[0].header
stk_data2 = hdul2[0].data


fig, ax = plt.subplots(nrows=2, ncols=4)
axf = ax.flatten()
for i in range(8):
    if i < 4:
        im1=axf[i].imshow(stk_data2[0, i, :,:], cmap='gray', origin='lower')
        fig.colorbar(im1, ax=axf[i])
    else:
        im2=axf[i].imshow(stk_data1[i-4, 0, :,:], cmap='gray', origin='lower')
        fig.colorbar(im2, ax=axf[i])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=4, ncols=1)
stk_data1_ft_I = stk_data1[0,:,:,:].flatten()
stk_data2_ft_I = stk_data2[:,0,:,:].flatten()
ax[0].hist(stk_data1_ft_I, bins=200, color='blue', alpha=0.5, label='M')
ax[0].hist(stk_data2_ft_I, bins=200, color='red', alpha=0.5, label='Py')
stk_data1_ft_Q = stk_data1[1,:,:,:].flatten()
stk_data2_ft_Q = stk_data2[:,1,:,:].flatten()
ax[1].hist(stk_data1_ft_Q, bins=200, color='blue', alpha=0.5, label='M')
ax[1].hist(stk_data2_ft_Q, bins=200, color='red', alpha=0.5, label='Py')
stk_data1_ft_U = stk_data1[2,:,:,:].flatten()
stk_data2_ft_U = stk_data2[:,2,:,:].flatten()
ax[2].hist(stk_data1_ft_U, bins=200, color='blue', alpha=0.5, label='M')
ax[2].hist(stk_data2_ft_U, bins=200, color='red', alpha=0.5, label='Py')
stk_data1_ft_V = stk_data1[3,:,:,:].flatten()
stk_data2_ft_V = stk_data2[:,3,:,:].flatten()
ax[3].hist(stk_data1_ft_V, bins=200, color='blue', alpha=0.5, label='M')
ax[3].hist(stk_data2_ft_V, bins=200, color='red', alpha=0.5, label='Py')
plt.legend()
plt.show()


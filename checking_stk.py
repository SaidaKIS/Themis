import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.io import fits
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from copy import copy
import matplotlib.gridspec as gridspec

from matplotlib.widgets import Slider, Cursor
import matplotlib.gridspec as gridspec

c=2.99e8 #speed of light in m/s

plt.rcParams.update({'font.size': 8})

def wl2v(lam, b_w):
    b_loc_wl_m=b_w*1.0E-10 #m
    lam_m=lam*1.0E-10 #m
    a=-(lam_m-b_loc_wl_m)
    b=a/lam_m
    v = (b*c)/1000 #km/s
    return v

def v2wl(lam, v):
    lam_m=lam*1.0E-10
    b=(v*1000)/c
    a = b * lam_m
    b_loc_wl_m = a + lam_m
    b_w = b_loc_wl_m/1.0E-10
    return b_w

class spect_check():
    """Tool for spectral inspection
    Parameters:
    1. cube_obs: Spectropolarimetric observations (Stokes vector, H, W, Wavelenght)
    """
    def __init__(self, cube_obs, contour_image=None, levels=None):

        self.cube_obs = np.array(cube_obs)
        self.cube_shape = self.cube_obs.shape
        print("Cube shape:", self.cube_shape)
        self.stokes_fov_avg = np.nanmean(self.cube_obs, axis=(1,2))
        self.bg_map = self.cube_obs[0,:,:,0]
        
    def set_parameters(self, scan_pos, lam=None, cont_value=1, pix_size=1):
        self.scan_pos = scan_pos
        self.lam = lam
        self.pix_size = pix_size
        self.cont = cont_value

        self.wl=np.array(self.scan_pos)

        #self.vel_scan_pos=np.round(wl2v(self.lam, self.wl),1)
        self.arcsec_x=np.linspace(start=0, stop=self.cube_shape[2], num=8)
        self.arcsec_y=np.linspace(start=0, stop=self.cube_shape[1], num=8)
        self.pix2arc_x=np.around(self.arcsec_y*self.pix_size,2)
        self.pix2arc_y=np.around(self.arcsec_x*self.pix_size,2)
    
    def inspect(self, cmap_context='gray'):
        self.cmap_bg = cmap_context

        if self.cube_shape[0] == 4:
            self.fig = plt.figure(figsize=(15,10))
            spec = gridspec.GridSpec(ncols=2, nrows=4, figure=self.fig)
            self.ax0 = self.fig.add_subplot(spec[0:4, 0:1])
            self.ax1 = self.fig.add_subplot(spec[0, 1])
            self.ax2 = self.fig.add_subplot(spec[1, 1])
            self.ax3 = self.fig.add_subplot(spec[2, 1])
            self.ax4 = self.fig.add_subplot(spec[3, 1])

            self.p1 = self.ax0.imshow(self.bg_map, origin='lower', cmap=self.cmap_bg, aspect='auto')
            divider = make_axes_locatable(self.ax0)
            cax1 = divider.append_axes('right', size='5%', pad="1%")
            self.cb1=self.fig.colorbar(self.p1, cax=cax1, orientation='vertical')

            if isinstance(self.contour_image, bool) == False:
                self.X = np.arange(0,self.cube_shape[2],1)
                self.Y = np.arange(0,self.cube_shape[1],1)
                self.p2 = self.ax0.contour(self.X,self.Y,self.contour_image, levels=self.levels)
                cax2 = divider.append_axes('top', size='5%', pad="1%")
                self.cb2=self.fig.colorbar(self.p2, cax=cax2, orientation='horizontal')
                cax2.xaxis.set_ticks_position("top")

            self.ax1.plot(self.wl, self.stokes_fov_avg[0]/self.cont, color='orange', label='Mean')
            self.ax2.plot(self.wl, self.stokes_fov_avg[1]/self.cont, color='orange', label='Mean')
            self.ax3.plot(self.wl, self.stokes_fov_avg[2]/self.cont, color='orange', label='Mean')
            self.ax4.plot(self.wl, self.stokes_fov_avg[3]/self.cont, color='orange', label='Mean')

            self.line_obs1, = self.ax1.plot([], [], color='black', label='Observed')
            self.line_obs2, = self.ax2.plot([], [], color='black', label='Observed')
            self.line_obs3, = self.ax3.plot([], [], color='black', label='Observed')
            self.line_obs4, = self.ax4.plot([], [], color='black', label='Observed')

            self.ax2.hlines(3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            self.ax3.hlines(3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            self.ax4.hlines(3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            self.ax2.hlines(-3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            self.ax3.hlines(-3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            self.ax4.hlines(-3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')

            self.ax1.set_ylabel(r'$I/I_c$')
            self.ax2.set_ylabel(r'$U/I_c$')
            self.ax3.set_ylabel(r'$Q/I_c$')
            self.ax4.set_ylabel(r'$V/I_c$')
            self.ax4.set_xlabel('Wavelength [Å]')
            self.ax1.legend()
            self.ax2.legend()
            self.ax3.legend()
            self.ax4.legend()
            self.ax1.grid()
            self.ax2.grid()
            self.ax3.grid()
            self.ax4.grid()

            self.ax1.set_ylim([0.0,1.1])
            self.ax2.set_ylim([-0.05,0.05])
            self.ax3.set_ylim([-0.05,0.05])
            self.ax4.set_ylim([-0.3,0.3])

            self.ax1.set_xlim([self.wl[0],self.wl[-1]])
            self.ax2.set_xlim([self.wl[0],self.wl[-1]])
            self.ax3.set_xlim([self.wl[0],self.wl[-1]])
            self.ax4.set_xlim([self.wl[0],self.wl[-1]])

            self.cursor1 = Cursor(self.ax0, horizOn = True, vertOn = True, color = 'red',linewidth = '0.5')
            self.cid1= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing1)

            self.cursor2 = Cursor(self.ax1, horizOn = False, vertOn = True, color = 'blue',linewidth = '0.5')
            self.cid2= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing2)

            self.cursor3 = Cursor(self.ax2, horizOn = False, vertOn = True, color = 'blue',linewidth = '0.5')
            self.cid3= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing3)

            self.cursor4 = Cursor(self.ax3, horizOn = False, vertOn = True, color = 'blue',linewidth = '0.5')
            self.cid4= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing4)

            self.cursor5 = Cursor(self.ax4, horizOn = False, vertOn = True, color = 'blue',linewidth = '0.5')
            self.cid5= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing5)

            plt.show()

        elif self.cube_shape[0] == 1:
            self.fig = plt.figure(figsize=(15,10))
            spec = gridspec.GridSpec(ncols=2, nrows=1, figure=self.fig)
            self.ax0 = self.fig.add_subplot(spec[0, 0])
            self.ax1 = self.fig.add_subplot(spec[0, 1])

            self.p1 = self.ax0.imshow(self.bg_map, origin='lower', cmap=self.cmap_bg, aspect='auto')
            divider = make_axes_locatable(self.ax0)
            cax1 = divider.append_axes('right', size='5%', pad="1%")
            self.cb1=self.fig.colorbar(self.p1, cax=cax1, orientation='vertical')

            if isinstance(self.contour_image, bool) == False:
                self.X = np.arange(0,self.cube_shape[2],1)
                self.Y = np.arange(0,self.cube_shape[1],1)
                self.p2 = self.ax0.contour(self.X,self.Y,self.contour_image, levels=self.levels)
                cax2 = divider.append_axes('top', size='5%', pad="1%")
                self.cb2=self.fig.colorbar(self.p2, cax=cax2, orientation='horizontal')
                cax2.xaxis.set_ticks_position("top")

            self.ax1.plot(self.wl, self.stokes_fov_avg/self.cont, color='orange', label='Mean')

            self.line_obs1, = self.ax1.plot([], [], color='black', label='Observed')

            self.ax1.set_ylabel(r'$I/I_c$')
            self.ax1.set_xlabel(r'Wavelength [Å]')

            self.ax1.legend()
                        
            self.ax1.set_ylim([0.0,1.5])
                        
            self.ax1.set_xlim([self.wl[0],self.wl[-1]])
                        
            self.cursor1 = Cursor(self.ax0, horizOn = True, vertOn = True, color = 'red',linewidth = '0.5')
            self.cid1= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing1)
            
            self.cursor2 = Cursor(self.ax1, horizOn = False, vertOn = True, color = 'blue',linewidth = '0.5')
            self.cid2= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing2)

            plt.show()

    def pointing1(self, event):
        if event.inaxes == self.ax0 and event.xdata is not None:
            x1, y1 = int(event.xdata), int(event.ydata)
        
            # Array bound safety
            if 0 <= y1 < self.cube_shape[1] and 0 <= x1 < self.cube_shape[2]:
                # Overwrites line data instantly without touching axes formatting or cursors
                self.line_obs1.set_data(self.wl, self.cube_obs[0, y1, x1, :])
                self.line_obs2.set_data(self.wl, self.cube_obs[1, y1, x1, :])
                self.line_obs3.set_data(self.wl, self.cube_obs[2, y1, x1, :])
                self.line_obs4.set_data(self.wl, self.cube_obs[3, y1, x1, :])

        self.fig.canvas.draw_idle()

    def pointing2(self, event):
        if event.inaxes == self.ax1 and event.xdata is not None:
            wl = event.xdata
            wl_ind = np.argmin(np.abs(self.wl - wl))
            img_data = self.cube_obs[0, :, :, wl_ind]    
            self.p1.set_data(img_data)
            vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
            self.p1.set_clim(vmin=vmin, vmax=vmax)

        self.fig.canvas.draw_idle()

    def pointing3(self, event):
        if event.inaxes == self.ax2 and event.xdata is not None:
            wl = event.xdata
            wl_ind = np.argmin(np.abs(self.wl - wl))
            img_data = self.cube_obs[1, :, :, wl_ind]    
            self.p1.set_data(img_data)
            vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
            self.p1.set_clim(vmin=vmin, vmax=vmax)

        self.fig.canvas.draw_idle()

    def pointing4(self, event):
        if event.inaxes == self.ax3 and event.xdata is not None:
            wl = event.xdata
            wl_ind = np.argmin(np.abs(self.wl - wl))
            img_data = self.cube_obs[2, :, :, wl_ind]    
            self.p1.set_data(img_data)
            vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
            self.p1.set_clim(vmin=vmin, vmax=vmax)

        self.fig.canvas.draw_idle()

    def pointing5(self, event):
        if event.inaxes == self.ax4 and event.xdata is not None:
            wl = event.xdata
            wl_ind = np.argmin(np.abs(self.wl - wl))
            img_data = self.cube_obs[3, :, :, wl_ind]    
            self.p1.set_data(img_data)
            vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
            self.p1.set_clim(vmin=vmin, vmax=vmax)

        self.fig.canvas.draw_idle()

class spect_inv_check():
        """Tool for spectral inspection including the pyMilne inversion results
        Parameters:
        1. cube_obs: Spectropolarimetric observations (Stokes vector, H, W, Wavelenght)
        2. cube_inv: Profile fits from inversion results (Stokes vector, H, W, Wavelenght)
        
        """
        def __init__(self, cube_obs, cube_inv=False, inv_models=None, contour_image=None, levels=None):
    
            self.cube_obs = np.array(cube_obs)
            self.stokes_fov_avg = np.nanmean(self.cube_obs, axis=(1,2))
            self.cube_shape = self.cube_obs.shape
            self.bg_map = self.cube_obs[0,:,:,0] 
    
            if isinstance(cube_inv, bool) == False:
                self.cube_inv = np.transpose(cube_inv[0], (2, 0, 1, 3))
            else:
                self.cube_inv = False

            if isinstance(contour_image, bool) == False:
                self.contour_image = np.array(contour_image)
                self.levels = np.array(levels)
                    
            else:
                self.contour_image = False

            if isinstance(inv_models, bool) == False:
                self.inv_models = np.array(inv_models)
        
        def set_parameters(self, scan_pos, lam=None, cont_value=1, pix_size=1):
            self.scan_pos = scan_pos
            self.lam = lam
            self.pix_size = pix_size
            self.cont = cont_value
    
            self.wl=np.array(self.scan_pos)
    
            #self.vel_scan_pos=np.round(wl2v(self.lam, self.wl),1)
            self.arcsec_x=np.linspace(start=0, stop=self.cube_shape[2], num=8)
            self.arcsec_y=np.linspace(start=0, stop=self.cube_shape[1], num=8)
            self.pix2arc_x=np.around(self.arcsec_y*self.pix_size,2)
            self.pix2arc_y=np.around(self.arcsec_x*self.pix_size,2)
        
        def inspect(self, chi=False, cmap_context='gray'):
            self.cmap_bg = cmap_context
            
            self.fig = plt.figure(figsize=(15,10))
            spec = gridspec.GridSpec(ncols=2, nrows=4, figure=self.fig)
            self.ax0 = self.fig.add_subplot(spec[0:4, 0:1])
            self.ax1 = self.fig.add_subplot(spec[0, 1])
            self.ax2 = self.fig.add_subplot(spec[1, 1])
            self.ax3 = self.fig.add_subplot(spec[2, 1])
            self.ax4 = self.fig.add_subplot(spec[3, 1])

            if chi != True: 
                self.p1 = self.ax0.imshow(self.bg_map, origin='lower', cmap=self.cmap_bg, aspect='auto')
                divider = make_axes_locatable(self.ax0)
                cax1 = divider.append_axes('right', size='5%', pad="1%")
                self.cb1=self.fig.colorbar(self.p1, cax=cax1, orientation='vertical')

            else:
                self.chi_calculation_full()
                self.p1 = self.ax0.imshow(self.chi_cube_full, origin='lower', aspect='auto')
                divider = make_axes_locatable(self.ax0)
                cax1 = divider.append_axes('right', size='5%', pad="1%")
                self.cb1=self.fig.colorbar(self.p1, cax=cax1, orientation='vertical')
            
            #if isinstance(self.contour_image, bool) == False:
            #    self.X = np.arange(0,self.cube_shape[2],1)
            #    self.Y = np.arange(0,self.cube_shape[1],1)
            #    self.p2 = self.ax0.contour(self.X,self.Y,self.contour_image, levels=self.levels)
            #    cax2 = divider.append_axes('top', size='5%', pad="1%")
            #    self.cb2=self.fig.colorbar(self.p2, cax=cax2, orientation='horizontal')
            #    cax2.xaxis.set_ticks_position("top")
    
            self.ax1.plot(self.wl, self.stokes_fov_avg[0]/self.cont, color='orange', label='Mean')
            self.ax2.plot(self.wl, self.stokes_fov_avg[1]/self.cont, color='orange', label='Mean')
            self.ax3.plot(self.wl, self.stokes_fov_avg[2]/self.cont, color='orange', label='Mean')
            self.ax4.plot(self.wl, self.stokes_fov_avg[3]/self.cont, color='orange', label='Mean')
    
            self.line_obs1, = self.ax1.plot([], [], color='black', label='Observed')
            self.line_obs2, = self.ax2.plot([], [], color='black', label='Observed')
            self.line_obs3, = self.ax3.plot([], [], color='black', label='Observed')
            self.line_obs4, = self.ax4.plot([], [], color='black', label='Observed')

            self.line_syn1, = self.ax1.plot([], [], color='blue', label='Syntetic - Inv')
            self.line_syn2, = self.ax2.plot([], [], color='blue', label='Syntetic - Inv')
            self.line_syn3, = self.ax3.plot([], [], color='blue', label='Syntetic - Inv')
            self.line_syn4, = self.ax4.plot([], [], color='blue', label='Syntetic - Inv')

            self.ax2.hlines(3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            self.ax3.hlines(3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            self.ax4.hlines(3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            self.ax2.hlines(-3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            self.ax3.hlines(-3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            self.ax4.hlines(-3e-3, self.wl[0], self.wl[-1], colors='red', linestyles='dashed')
            
            self.ax1.set_ylabel(r'$I/I_c$')
            self.ax2.set_ylabel(r'$U/I_c$')
            self.ax3.set_ylabel(r'$Q/I_c$')
            self.ax4.set_ylabel(r'$V/I_c$')
            self.ax4.set_xlabel('Wavelength [Å]')
            self.ax1.legend()
            self.ax2.legend()
            self.ax3.legend()
            self.ax4.legend()
            self.ax1.grid()
            self.ax2.grid()
            self.ax3.grid()
            self.ax4.grid()
    
            self.ax1.set_ylim([0.0,1.1])
            self.ax2.set_ylim([-0.05,0.05])
            self.ax3.set_ylim([-0.05,0.05])
            self.ax4.set_ylim([-0.3,0.3])
    
            self.ax1.set_xlim([self.wl[0],self.wl[-1]])
            self.ax2.set_xlim([self.wl[0],self.wl[-1]])
            self.ax3.set_xlim([self.wl[0],self.wl[-1]])
            self.ax4.set_xlim([self.wl[0],self.wl[-1]])
    
            self.cursor1 = Cursor(self.ax0, horizOn = True, vertOn = True, color = 'red',linewidth = '0.5')
            self.cid1= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing1)

            if chi != True:
                self.cursor2 = Cursor(self.ax1, horizOn = False, vertOn = True, color = 'blue',linewidth = '0.5')
                self.cid2= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing2)

                self.cursor3 = Cursor(self.ax2, horizOn = False, vertOn = True, color = 'blue',linewidth = '0.5')
                self.cid3= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing3)

                self.cursor4 = Cursor(self.ax3, horizOn = False, vertOn = True, color = 'blue',linewidth = '0.5')
                self.cid4= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing4)

                self.cursor5 = Cursor(self.ax4, horizOn = False, vertOn = True, color = 'blue',linewidth = '0.5')
                self.cid5= self.fig.canvas.mpl_connect("motion_notify_event", self.pointing5)
    
            plt.show()
    
        def pointing1(self, event):
            if event.inaxes == self.ax0 and event.xdata is not None:
                x1, y1 = int(event.xdata), int(event.ydata)
            
                # Array bound safety
                if 0 <= y1 < self.cube_shape[1] and 0 <= x1 < self.cube_shape[2]:
                    # Overwrites line data instantly without touching axes formatting or cursors
                    self.line_obs1.set_data(self.wl, self.cube_obs[0, y1, x1, :])
                    self.line_obs2.set_data(self.wl, self.cube_obs[1, y1, x1, :])
                    self.line_obs3.set_data(self.wl, self.cube_obs[2, y1, x1, :])
                    self.line_obs4.set_data(self.wl, self.cube_obs[3, y1, x1, :])

                    self.line_syn1.set_data(self.wl, self.cube_inv[0, y1, x1, :])
                    self.line_syn2.set_data(self.wl, self.cube_inv[1, y1, x1, :])
                    self.line_syn3.set_data(self.wl, self.cube_inv[2, y1, x1, :])
                    self.line_syn4.set_data(self.wl, self.cube_inv[3, y1, x1, :])
    
            self.fig.canvas.draw_idle()
    
        def pointing2(self, event):
            if event.inaxes == self.ax1 and event.xdata is not None:
                wl = event.xdata
                wl_ind = np.argmin(np.abs(self.wl - wl))
                img_data = self.cube_obs[0, :, :, wl_ind]    
                self.p1.set_data(img_data)
                vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
                self.p1.set_clim(vmin=vmin, vmax=vmax)
    
            self.fig.canvas.draw_idle()
    
        def pointing3(self, event):
            if event.inaxes == self.ax2 and event.xdata is not None:
                wl = event.xdata
                wl_ind = np.argmin(np.abs(self.wl - wl))
                img_data = self.cube_obs[1, :, :, wl_ind]    
                self.p1.set_data(img_data)
                vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
                self.p1.set_clim(vmin=vmin, vmax=vmax)
    
            self.fig.canvas.draw_idle()
    
        def pointing4(self, event):
            if event.inaxes == self.ax3 and event.xdata is not None:
                wl = event.xdata
                wl_ind = np.argmin(np.abs(self.wl - wl))
                img_data = self.cube_obs[2, :, :, wl_ind]    
                self.p1.set_data(img_data)
                vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
                self.p1.set_clim(vmin=vmin, vmax=vmax)
    
            self.fig.canvas.draw_idle()
    
        def pointing5(self, event):
            if event.inaxes == self.ax4 and event.xdata is not None:
                wl = event.xdata
                wl_ind = np.argmin(np.abs(self.wl - wl))
                img_data = self.cube_obs[3, :, :, wl_ind]    
                self.p1.set_data(img_data)
                vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
                self.p1.set_clim(vmin=vmin, vmax=vmax)
    
            self.fig.canvas.draw_idle()

        def chi_calculation_full(self, weights=[1, 1, 1, 1], error=[3e-3, 3e-3, 3e-3, 3e-3]):
            self.weights = weights
            self.error = error
            #calculate chi-square cube considering the weights and errors per stokes parameter
            self.chi_cube_full = np.zeros((self.cube_obs.shape[1], self.cube_obs.shape[2]))
            lm = 4*self.wl.shape[0] - 9 # Constant - degree freedom VFISV case
            for stk in range(4):
                for i in range(self.cube_obs.shape[1]):
                    for j in range(self.cube_obs.shape[2]):
                        if np.isnan(self.cube_obs[stk, i, j, :]).all() or np.isnan(self.cube_inv[stk, i, j, :]).all():
                            self.chi_cube_full[i, j] = np.nan
                        else:
                            self.chi_cube_full[i, j] += np.nansum(((self.cube_obs[stk, i, j, :] - self.cube_inv[stk, i, j, :])**2) * (self.weights[stk] / self.error[stk])**2)
            self.chi_cube_full /= lm    

        def chi_calculation_specific(self, line, width=1, weights=[1, 1, 1, 1], error=[3e-3, 3e-3, 3e-3, 3e-3]):
            self.line = line
            self.width = width
            self.line_0 = line - width / 2
            self.line_1 = line + width / 2
            self.wvl_range = np.where((self.wl >= self.line_0) & (self.wl <= self.line_1))[0]

            lm = 4*self.wvl_range.shape[0] - 9 # Constant - degree freedom VFISV case

            self.cube_obs_partial = self.cube_obs[:, :, :, self.wvl_range]
            self.cube_inv_partial = self.cube_inv[:, :, :, self.wvl_range]

            #Ask the user if they want to plot the observed and syntetic profiles for a specific pixel
            plot_profiles = input("Do you want to plot the observed and syntetic profiles for a specific pixel? (y/n): ")
            if plot_profiles.lower() == 'y':    

                plt.figure(figsize=(10, 8))
                plt.subplot(2, 2, 1)
                plt.plot(self.wl[self.wvl_range], self.cube_obs_partial[0, 50, 50, :], color='black', label='Observed')
                plt.plot(self.wl[self.wvl_range], self.cube_inv_partial[0, 50, 50, :], color='blue', label='Syntetic - Inv')
                plt.title('Stokes I')
                plt.xlabel('Wavelength [Å]')
                plt.ylabel('Intensity')
                plt.legend()
                plt.subplot(2, 2, 2)
                plt.plot(self.wl[self.wvl_range], self.cube_obs_partial[1, 50, 50, :], color='black', label='Observed')
                plt.plot(self.wl[self.wvl_range], self.cube_inv_partial[1, 50, 50, :], color='blue', label='Syntetic - Inv')
                plt.title('Stokes Q')
                plt.xlabel('Wavelength [Å]')
                plt.ylabel('Intensity')
                plt.legend()
                plt.subplot(2, 2, 3)
                plt.plot(self.wl[self.wvl_range], self.cube_obs_partial[2, 50, 50, :], color='black', label='Observed')
                plt.plot(self.wl[self.wvl_range], self.cube_inv_partial[2, 50, 50, :], color='blue', label='Syntetic - Inv')
                plt.title('Stokes U')
                plt.xlabel('Wavelength [Å]')
                plt.ylabel('Intensity')
                plt.legend()
                plt.subplot(2, 2, 4)
                plt.plot(self.wl[self.wvl_range], self.cube_obs_partial[3, 50, 50, :], color='black', label='Observed')
                plt.plot(self.wl[self.wvl_range], self.cube_inv_partial[3, 50, 50, :], color='blue', label='Syntetic - Inv')
                plt.title('Stokes V')
                plt.xlabel('Wavelength [Å]')
                plt.ylabel('Intensity')
                plt.legend()
                plt.tight_layout()
                plt.show()          


            #calculate chi-square cube considering the weights and errors per stokes parameter
            self.chi_cube = np.zeros((self.cube_obs_partial.shape[1], self.cube_obs_partial.shape[2]))
            for stk in range(4):
                for i in range(self.cube_obs_partial.shape[1]):
                    for j in range(self.cube_obs_partial.shape[2]):
                        if np.isnan(self.cube_obs_partial[stk, i, j, :]).all() or np.isnan(self.cube_inv_partial[stk, i, j, :]).all():
                            self.chi_cube[i, j] = np.nan
                        else:
                            self.chi_cube[i, j] += np.nansum(((self.cube_obs_partial[stk, i, j, :] - self.cube_inv_partial[stk, i, j, :])**2) * (weights[stk] / error[stk])**2)

            self.chi_cube = self.chi_cube/lm
            #plot the chi-square cube
            plt.figure(figsize=(10, 8))
            plt.imshow(self.chi_cube, origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=50)
            plt.colorbar(label='Chi-square')
            plt.title(f'Chi-square map for line {self.line} Å')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()


        def check_infered_model(self):
            #Plot Longitudinal Magnetic Field, Transversal Magnetic Field and LOS Velocity from the inversion results
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            B_long = self.inv_models[0, :, :, 0]*np.cos(self.inv_models[0, :, :, 1])
            B_tran = self.inv_models[0, :, :, 0]*np.sin(self.inv_models[0, :, :, 1])

            plt.imshow(B_long, origin='lower', cmap='RdBu_r', aspect='auto', vmin=-2000, vmax=2000)
            plt.colorbar(label='B_long [G]')
            plt.title('Longitudinal Magnetic Field')
            plt.subplot(1, 3, 2)
            plt.imshow(B_tran, origin='lower', cmap='gray', aspect='auto')
            plt.colorbar(label='B_trans [G]')
            plt.title('Transversal Magnetic Field')
            plt.subplot(1, 3, 3)
            plt.imshow(self.inv_models[0, :, :, 3], origin='lower', cmap='seismic', aspect='auto')
            plt.colorbar(label='LOS Velocity [km/s]')
            plt.title('LOS Velocity')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    file_name_obs = input("Enter the name of the calibrated observed Stokes data file:")
    file_name_inv = input("Enter the name of the syntetic Stokes data file, enter None if not want to include it:") 

    hdul_obs = fits.open(file_name_obs) 
    stk_data_obs = hdul_obs[0].data
    stk_data_wl_obs = hdul_obs[1].data

    if file_name_inv != None:
        hdul_inv = fits.open(file_name_inv)
        stk_data_inv = hdul_inv[1].data
        checking_obs_inv=spect_inv_check(stk_data_obs, stk_data_inv, inv_models=hdul_inv[0].data)
        checking_obs_inv.set_parameters(stk_data_wl_obs*10)
        checking_obs_inv.inspect(chi=True)
        checking_obs_inv.chi_calculation_specific(line=6302.5, width=0.4)
        #checking_obs_inv.check_infered_model()

    else:
        checking_obs=spect_check(stk_data_obs)
        checking_obs.set_parameters(stk_data_wl_obs*10)
        checking_obs.inspect()






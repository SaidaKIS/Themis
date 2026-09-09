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
        self.stokes_fov_avg = np.nanmean(self.cube_obs, axis=(1,2))
        print(self.stokes_fov_avg.shape)
        self.cube_shape = self.cube_obs.shape
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
        def __init__(self, cube_obs, cube_inv=False, contour_image=None, levels=None):
    
            self.cube_obs = np.array(cube_obs)
            self.stokes_fov_avg = np.nanmean(self.cube_obs, axis=(1,2))
            print(self.stokes_fov_avg.shape)
            self.cube_shape = self.cube_obs.shape
            self.bg_map = self.cube_obs[0,:,:,0] 
    
            if isinstance(cube_inv, bool) == False:
                self.cube_inv = cube_inv
            else:
                self.cube_inv = False

            if isinstance(contour_image, bool) == False:
                self.contour_image = np.array(contour_image)
                self.levels = np.array(levels)
                    
            else:
                self.contour_image = False
        
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


if __name__ == "__main__":
    file_name_obs = input("Enter the name of the calibrated observed Stokes data file:")
    file_name_inv = input("Enter the name of the syntetic Stokes data file, enter None if not want to include it:") 

    hdul_obs = fits.open(file_name_obs)
    stk_data_obs = hdul_obs[0].data
    stk_data_wl_obs = hdul_obs[1].data

    hdul_inv = fits.open(file_name_inv)
    stk_data_inv = hdul_inv[1].data

    if file_name_inv != None:
        checking_obs_inv=spect_inv_check(stk_data_obs, stk_data_inv)
        checking_obs_inv.set_parameters(stk_data_wl_obs*10)
        checking_obs_inv.inspect()

    else:
        checking_obs=spect_check(stk_data_obs)
        checking_obs.set_parameters(stk_data_wl_obs*10)
        checking_obs.inspect()





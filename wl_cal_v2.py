import numpy as np
from astropy.io import fits
import astropy.units as u #Operations between units and constants
import matplotlib.pyplot as plt
import glob
import sys
from astropy.wcs import WCS
import os
from matplotlib.widgets import SpanSelector
from scipy.interpolate import interp1d
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
os.environ["QT_LOGGING_RULES"] = "qt.qpa.wayland*=false"

def lc_core(spect,si,sf,num):
    """
    Finding the line core center
    input: 
        spect: spectral values array
        si: initial position in the array to consider
        sf: final position in the array to consider
        num: length of the array to consider - to fit a parbola

    output:
        array (x, y) consider the full spectral array (position, intensity) 
    """
    numh=(num-1)/2
    x=np.arange(num)
    p=spect[si:sf]
    cent=np.where(p == min(p))[0]
    cent1=int(cent[0]+si)
    l=int(cent1-numh)
    u=int(cent1+numh+1)
    d=spect[l:u]
    coeff=np.polyfit(x,d,2)
    cent= -coeff[1]/(2.*coeff[0])
    io = coeff[2] + coeff[1]*cent + coeff[0]*cent**2
    cent = cent + cent1 - numh
    out=[cent,io]
    return out

def widen_multiple_lines(wavelength, flux, line_centers, stretch_factor, window_size=5.0):
    """
    Widens multiple spectral lines individually by the same stretch_factor
    without altering their central wavelengths or maximum depths.
    
    Parameters:
    - wavelength, flux: Input spectrum arrays (continuum assumed at 1.0).
    - line_centers: List or array of central wavelengths for each line.
    - stretch_factor: How much to widen the lines (e.g., 1.5).
    - window_size: The local wavelength width around each center to apply the stretch.
    """
    # Start with a copy of the original depth profile
    total_depth = 1.0 - flux
    new_total_depth = np.zeros_like(total_depth)
    
    # Process each line independently
    for center in line_centers:
        # Create a mask to isolate the local window around the current line
        mask = (wavelength >= (center - window_size)) & (wavelength <= (center + window_size))
        
        if not np.any(mask):
            continue
            
        wl_local = wavelength[mask]
        depth_local = total_depth[mask]
        
        # Stretch local coordinates relative to this specific line's center
        stretched_wl_local = center + stretch_factor * (wl_local - center)
        
        # Interpolate the stretched local line back onto the original local grid
        interp_func = interp1d(stretched_wl_local, depth_local, kind='cubic', 
                               bounds_error=False, fill_value=0.0)
        
        # Add the modified line depth to our clean output array
        new_total_depth[mask] += interp_func(wl_local)
        
    # Convert the combined stretched depths back to normalized flux
    new_flux = 1.0 - new_total_depth
    return new_flux

class rectangle_selector:
    def __init__(self, img):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.imshow(img, origin='lower', aspect='auto', cmap='gray')
        self.ax.set_title("Select rectangle region by dragging the mouse.\nExplicit red rectangle marks boundaries. Escape to reset. Close to finish.")
        self.rect = None
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.x0 = event.xdata
        self.y0 = event.ydata
        if self.rect is not None:
            self.rect.remove()
        self.rect = self.ax.add_patch(plt.Rectangle((self.x0, self.y0), 0, 0, edgecolor="red", facecolor="none", lw=2))
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.xmin = min(self.x0, self.x1)
        self.xmax = max(self.x0, self.x1)
        self.ymin = min(self.y0, self.y1)
        self.ymax = max(self.y0, self.y1)
        self.rect.set_bounds(self.xmin, self.ymin, self.xmax - self.xmin, self.ymax - self.ymin)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'escape':
            self.xmin = self.xmax = self.ymin = self.ymax = None
            if self.rect is not None:
                self.rect.remove()
                self.rect = None
            self.ax.set_title("Select rectangle region by dragging the mouse.\nExplicit red rectangle marks boundaries. Escape to reset. Close to finish.")
            self.fig.canvas.draw_idle()

class wl_range_selector_atlas:
    def __init__(self, s_wl, s_int, t_wl, t_int, type='telluric'):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.plot(t_wl, t_int, color="tab:blue", lw=2, label="Telluric Reference Atlas")
        self.ax.plot(s_wl, s_int, color="tab:green", lw=2, label="Solar Reference Atlas")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Intensity")
        self.ax.legend()
        self.type = type
        self.ax.set_title(f"Step 1: Select first {self.type} line by dragging the mouse.\nExplicit red vertical lines mark boundaries. Escape to reset.")
    
        # 1. Create vertical lines initialized out of view
        self.line1_min = self.ax.axvline(x=np.nan, color="tab:orange", lw=2, linestyle="--")
        self.line1_max = self.ax.axvline(x=np.nan, color="tab:orange", lw=2, linestyle="--")
        
        self.line2_min = self.ax.axvline(x=np.nan, color="tab:green", lw=2, linestyle="--")
        self.line2_max = self.ax.axvline(x=np.nan, color="tab:green", lw=2, linestyle="--")

        self.xmin1, self.xmax1 = None, None
        self.xmin2, self.xmax2 = None, None

        self.span1 = SpanSelector(
            self.ax,
            self.on_select1,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.15, facecolor="tab:orange"),
            interactive=True,
            drag_from_anywhere=True,
            ignore_event_outside=True,
            )

        self.span2 = SpanSelector(
            self.ax,
            self.on_select2,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.15, facecolor="tab:green"),
            interactive=True,
            drag_from_anywhere=True,
            ignore_event_outside=True,
            )

        self.span2.set_active(False)

    def on_select1(self, xmin, xmax):
        self.xmin1 = xmin
        self.xmax1 = xmax
        self.line1_min.set_xdata([xmin, xmin])
        self.line1_max.set_xdata([xmax, xmax])
        self.fig.canvas.draw_idle()

        # Deactivate span 1 and activate span 2
        self.span1.set_active(False)
        self.span2.set_active(True)
        
        self.ax.set_title(f"Step 2: Select second line. Close window when finished.")
        self.fig.canvas.draw_idle()

    def on_select2(self, xmin, xmax):
            self.xmin2 = xmin
            self.xmax2 = xmax
            self.line2_min.set_xdata([xmin, xmin])
            self.line2_max.set_xdata([xmax, xmax])
            self.fig.canvas.draw_idle()

class wl_range_selector_observed:
    def __init__(self, obs, type='Observed'):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.plot(obs, color="tab:blue", lw=2, label="Observed data")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Intensity")
        self.ax.legend()
        self.type = type
        self.ax.set_title(f"Step 1: Select first {self.type} line by dragging the mouse.\nExplicit red vertical lines mark boundaries. Escape to reset.")
    
        # 1. Create vertical lines initialized out of view
        self.line1_min = self.ax.axvline(x=np.nan, color="tab:orange", lw=2, linestyle="--")
        self.line1_max = self.ax.axvline(x=np.nan, color="tab:orange", lw=2, linestyle="--")
        
        self.line2_min = self.ax.axvline(x=np.nan, color="tab:green", lw=2, linestyle="--")
        self.line2_max = self.ax.axvline(x=np.nan, color="tab:green", lw=2, linestyle="--")

        self.xmin1, self.xmax1 = None, None
        self.xmin2, self.xmax2 = None, None

        self.span1 = SpanSelector(
            self.ax,
            self.on_select1,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.15, facecolor="tab:orange"),
            interactive=True,
            drag_from_anywhere=True,
            ignore_event_outside=True,
            )

        self.span2 = SpanSelector(
            self.ax,
            self.on_select2,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.15, facecolor="tab:green"),
            interactive=True,
            drag_from_anywhere=True,
            ignore_event_outside=True,
            )

        self.span2.set_active(False)

    def on_select1(self, xmin, xmax):
        self.xmin1 = xmin
        self.xmax1 = xmax
        self.line1_min.set_xdata([xmin, xmin])
        self.line1_max.set_xdata([xmax, xmax])
        self.fig.canvas.draw_idle()

        # Deactivate span 1 and activate span 2
        self.span1.set_active(False)
        self.span2.set_active(True)
        
        self.ax.set_title(f"Step 2: Select second line. Close window when finished.")
        self.fig.canvas.draw_idle()

    def on_select2(self, xmin, xmax):
            self.xmin2 = xmin
            self.xmax2 = xmax
            self.line2_min.set_xdata([xmin, xmin])
            self.line2_max.set_xdata([xmax, xmax])
            self.fig.canvas.draw_idle()

def wavelength_range_selection_atlas(solar_region_wavelengths, solar_region_intensities, 
                               telluric_region_wavelengths, telluric_region_intensities, 
                               num_telluric_lines):

    if num_telluric_lines == 0:
        #Use two know spectral lines of the solar reference atlas to calibrate the wavelength axis of the observed data.
        #The user will select the range of the two known spectral lines in the solar reference atlas
        selector_lines = wl_range_selector_atlas(solar_region_wavelengths, solar_region_intensities, telluric_region_wavelengths, telluric_region_intensities, type=f'Solar')
        plt.show()
        mask_line1 = (solar_region_wavelengths >= selector_lines.xmin1) & (solar_region_wavelengths <= selector_lines.xmax1)
        #use lc_core to find the central wavelength more accurately
        #min_intensity_index_line1 = np.argmin(solar_region_intensities[mask_line1])
        #central_wavelength_line1 = solar_region_wavelengths[mask_line1][min_intensity_index_line1]
        lc_line1 = lc_core(solar_region_intensities, np.where(mask_line1)[0][0], np.where(mask_line1)[0][-1]+1, 10)
        central_wavelength_line1 = solar_region_wavelengths[int(lc_line1[0])]
        mask_line2 = (solar_region_wavelengths >= selector_lines.xmin2) & (solar_region_wavelengths <= selector_lines.xmax2)
        #use lc_core to find the central wavelength more accurately
        #min_intensity_index_line2 = np.argmin(solar_region_intensities[mask_line2])
        #central_wavelength_line2 = solar_region_wavelengths[mask_line2][min_intensity_index_line2]
        lc_line2 = lc_core(solar_region_intensities, np.where(mask_line2)[0][0], np.where(mask_line2)[0][-1]+1, 5)
        central_wavelength_line2 = solar_region_wavelengths[int(lc_line2[0])]

        central_wavelengths = (central_wavelength_line1, central_wavelength_line2)

    if num_telluric_lines == 1:
        #If only one telluric line is found in the observed data range, we ask the user to select interactivelly 
        #the region of the tellluric line to get its central wavelength.
        selector_lines = wl_range_selector_atlas(solar_region_wavelengths, solar_region_intensities, telluric_region_wavelengths, telluric_region_intensities, type=f'Telluric/Solar')
        plt.show()
        mask_line1 = (solar_region_wavelengths >= selector_lines.xmin1) & (solar_region_wavelengths <= selector_lines.xmax1)
        #use lc_core to find the central wavelength more accurately
        #min_intensity_index_line1 = np.argmin(solar_region_intensities[mask_line1])
        #central_wavelength_line1 = solar_region_wavelengths[mask_line1][min_intensity_index_line1]
        lc_line1 = lc_core(solar_region_intensities, np.where(mask_line1)[0][0], np.where(mask_line1)[0][-1]+1, 10)
        central_wavelength_line1 = solar_region_wavelengths[int(lc_line1[0])]
        mask_line2 = (solar_region_wavelengths >= selector_lines.xmin2) & (solar_region_wavelengths <= selector_lines.xmax2)
        #use lc_core to find the central wavelength more accurately
        #min_intensity_index_line2 = np.argmin(solar_region_intensities[mask_line2])
        #central_wavelength_line2 = solar_region_wavelengths[mask_line2][min_intensity_index_line2]
        lc_line2 = lc_core(solar_region_intensities, np.where(mask_line2)[0][0], np.where(mask_line2)[0][-1]+1, 5)
        central_wavelength_line2 = solar_region_wavelengths[int(lc_line2[0])]

        central_wavelengths = (central_wavelength_line1, central_wavelength_line2)   

    if num_telluric_lines >= 2:
        #If 2 or more telluric lines are found in the observed data range, we ask the user to select interactivelly 
        #the region of two tellluric lines to get its central wavelength.
        selector_lines = wl_range_selector_atlas(solar_region_wavelengths, solar_region_intensities, telluric_region_wavelengths, telluric_region_intensities, type=f'Telluric')
        plt.show()
        mask_line1 = (solar_region_wavelengths >= selector_lines.xmin1) & (solar_region_wavelengths <= selector_lines.xmax1)
        #use lc_core to find the central wavelength more accurately
        #min_intensity_index_line1 = np.argmin(solar_region_intensities[mask_line1])
        #central_wavelength_line1 = solar_region_wavelengths[mask_line1][min_intensity_index_line1]
        lc_line1 = lc_core(solar_region_intensities, np.where(mask_line1)[0][0], np.where(mask_line1)[0][-1]+1, 10)
        central_wavelength_line1 = solar_region_wavelengths[int(lc_line1[0])]
        mask_line2 = (solar_region_wavelengths >= selector_lines.xmin2) & (solar_region_wavelengths <= selector_lines.xmax2)
        #use lc_core to find the central wavelength more accurately
        #min_intensity_index_line2 = np.argmin(solar_region_intensities[mask_line2])
        #central_wavelength_line2 = solar_region_wavelengths[mask_line2][min_intensity_index_line2]
        lc_line2 = lc_core(solar_region_intensities, np.where(mask_line2)[0][0], np.where(mask_line2)[0][-1]+1, 5)
        central_wavelength_line2 = solar_region_wavelengths[int(lc_line2[0])]

        central_wavelengths = (central_wavelength_line1, central_wavelength_line2) 

    return central_wavelengths

def wavelength_range_selection_observed(observed):
    #Ask the user to select the range of two lines in the observed data to be calibrated and get the central wavelength of the observed data range.
    selector_observed_lines = wl_range_selector_observed(observed)
    rest_spectrum_array = np.arange(observed.shape[0])
    mask_observed_line1 = (rest_spectrum_array >= selector_observed_lines.xmin1) & (rest_spectrum_array <= selector_observed_lines.xmax1)
    #use lc_core to find the central wavelength more accurately
    lc_observed_line1 = lc_core(observed, np.where(mask_observed_line1)[0][0], np.where(mask_observed_line1)[0][-1]+1, 10)
    central_wavelength_observed_line1 = rest_spectrum_array[int(lc_observed_line1[0])]
    mask_observed_line2 = (rest_spectrum_array >= selector_observed_lines.xmin2) & (rest_spectrum_array <= selector_observed_lines.xmax2)
    #use lc_core to find the central wavelength more accurately
    lc_observed_line2 = lc_core(observed, np.where(mask_observed_line2)[0][0], np.where(mask_observed_line2)[0][-1]+1, 10)
    central_wavelength_observed_line2 = rest_spectrum_array[int(lc_observed_line2[0])]
    central_wavelength_observed = (central_wavelength_observed_line1, central_wavelength_observed_line2)
    
    return central_wavelength_observed

def main():
    #Add the description of the code and the steps to follow.
    #Printed at the beginning of the code to inform the user about the steps to follow
    print("This code performs wavelength calibration of observed Stokes data using solar and telluric reference atlases.")
    print("Steps to follow:")
    print("1. Load the observed Stokes data file to be calibrated.")
    print("2. Compute the rest spectrum as the average of the observed spectra and normalize it.")
    print("3. Define the central wavelength of the observed data range to be calibrated and the number of telluric lines in the observed data range.")
    print("4. Select the range of the observed data to be calibrated and get the central wavelength of the observed data range.")
    print("5. Compute the linear transformation to calibrate the wavelength axis of the observed data using the central wavelengths of the selected spectral lines in the reference atlas and the central wavelengths of the selected spectral lines in the observed data.")
    print("6. Apply the linear transformation to the wavelength axis of the observed data.")
    print("7. Create a full cube of the wavelength-calibrated observed data and save it as a new fits file.")   

    #Load the solar and telluric reference atlas .npy files
    # From https://zenodo.org/records/14674504 Solar and Telluric spectra for wavelength calibration
    # Authors/Creators : National Solar Observatory (USA) V2
    solar_atlas = np.load("solar_reference_atlas.npy")
    telluric_atlas = np.load("telluric_reference_atlas.npy")

    #checking the telluric hitran data - check later
    #telluric_cal_hitran(wl_range=[630.0, 631.0])

    solar_atlas_v2_pd = pd.read_csv("solar_reference_atlas_heliospectrotron.csv", names=['wavelength[A]', 'y_final_norm'])
    solar_atlas_v2 = np.array([solar_atlas_v2_pd['wavelength[A]'].values*0.1, solar_atlas_v2_pd['y_final_norm'].values])

    #load the name of the observed Stokes data file to be calibrated
    observed_stokes_file = input("Enter the name of the observed Stokes data file to be calibrated (e.g., stokes_cam6.fits): ")
    if not os.path.isfile(observed_stokes_file):
        print(f"Error: The file '{observed_stokes_file}' does not exist.")
        sys.exit(1)


    #Load the observed Stokes data and compute the rest spectrum as the average of the observed spectra and normalize it to the maximum value of the rest spectrum.
    #Load measured stokes and compute a rest spectra as the average of the observed spectra
    stokes_cube = fits.open(observed_stokes_file)
    #Typically: [scans, stokes, y, x] y-> along slit, x-> along wavelength
    #Calculating the rest spectrum as the average of the observed spectra and normalize it to the maximum value of the rest spectrum.
    #the wavelength axis is inverted, so we need to flip it to match the reference atlas
    #Not all field of view should be use to calculate the rest spectrum, 
    # we should select a custom region visualy with the rectangle selector.
    img = stokes_cube[0].data[:, 0, :, 50]
    #flat = stokes_cube[1].data
    #flat_spect = np.mean(flat, axis=0)
    #flat_spect = flat_spect / np.max(flat_spect)  # Normalize the flat field spectrum
    #flat_spect = flat_spect[::-1]  # Flip the wavelength axis to match the reference atlas
    
    selector = rectangle_selector(img)
    plt.show()
    ymin, ymax = int(selector.ymin), int(selector.ymax)
    xmin, xmax = int(selector.xmin), int(selector.xmax)
    print(f"Selected region: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    rest_spectrum = np.mean(stokes_cube[0].data[ymin:ymax, 0, xmin:xmax, :], axis=(0,1))
    rest_spectrum = rest_spectrum[::-1]  # Flip the wavelength axis to match the reference atlas
    rest_spectrum_max = np.max(rest_spectrum)
    rest_spectrum = rest_spectrum / rest_spectrum_max
    plt.close()

    #Third, we ask the user to define the central wavelength of the observed data range to be calibrated and the number of telluric lines in the observed data range.
    central_wavelength = float(input("Enter the central wavelength of the observed data range to be calibrated (in nm): "))
    num_telluric_lines = int(input("Enter the number of telluric lines in the observed data range: "))

    #with the central wavelenght, we define the range of the observed data to be calibrated around 0.6 nm of the central wavelength
    observed_data_range = (central_wavelength - 0.35, central_wavelength + 0.35)
    solar_region_mask = (solar_atlas[0, :] >= observed_data_range[0]) & (solar_atlas[0, :] <= observed_data_range[1])
    solar_region_wavelengths = solar_atlas[0, solar_region_mask]
    solar_region_intensities = solar_atlas[1, solar_region_mask]

    #check with the new solar_atlas_v2
    print(f"Observed data range: {observed_data_range}")
    solar_region_mask_v2 = (solar_atlas_v2[0, :] >= observed_data_range[0]) & (solar_atlas_v2[0, :] <= observed_data_range[1])
    solar_region_wavelengths_v2 = solar_atlas_v2[0, solar_region_mask_v2]
    solar_region_intensities_v2 = solar_atlas_v2[1, solar_region_mask_v2]

    #plt.figure()
    #plt.plot(solar_region_wavelengths, solar_region_intensities, label='Solar Atlas')
    #plt.plot(solar_region_wavelengths_v2, solar_region_intensities_v2, label='Solar Atlas v2')
    #plt.xlabel('Wavelength (nm)')
    #plt.ylabel('Intensity')
    #plt.title('Solar Atlas Comparison')
    #plt.legend()
    #plt.show()

    #for the telluric reference atlas, we will use the wavelength range of the observed data range.
    telluric_region_mask = (telluric_atlas[0, :] >= observed_data_range[0]) & (telluric_atlas[0, :] <= observed_data_range[1])
    telluric_region_wavelengths = telluric_atlas[0, telluric_region_mask]
    telluric_region_intensities = telluric_atlas[1, telluric_region_mask]

    #Fourth, we call the wavelength_range_selection function to get the central wavelengths of the selected spectral lines.
    central_wavelengths_atlas = wavelength_range_selection_atlas(solar_region_wavelengths, solar_region_intensities, telluric_region_wavelengths, telluric_region_intensities, num_telluric_lines)

    #Fifth, we ask the user to select the range of the observed data to be calibrated and get the central wavelength of the observed data range.
    central_wavelengths_observed = wavelength_range_selection_observed(rest_spectrum)

    #Sixth, we compute the linear transformation to calibrate the wavelength axis of the observed data using the central wavelengths of the selected spectral lines in the reference atlas and the central wavelengths of the selected spectral lines in the observed data.
    #We will use the two central wavelengths of the selected spectral lines in the reference atlas and the two central wavelengths of the selected spectral lines in the observed data to compute the linear transformation.
    #The linear transformation is defined as: wavelength_calibrated = a * wavelength_observed + b, where a and b are the coefficients of the linear transformation.
    #We will use the two central wavelengths of the selected spectral lines in the reference atlas and the two central wavelengths of the selected spectral lines in the observed data to compute the coefficients a and b of the linear transformation.
    a = (central_wavelengths_atlas[1] - central_wavelengths_atlas[0]) / (central_wavelengths_observed[1] - central_wavelengths_observed[0])
    b = central_wavelengths_atlas[0] - a * central_wavelengths_observed[0]

    #Seventh, we apply the linear transformation to the wavelength axis of the observed data.
    rest_spectrum_array = np.arange(rest_spectrum.shape[0])
    wavelength_calibrated = a * rest_spectrum_array + b

    spectral_resolution_observed_binned = 10000 * ((wavelength_calibrated[1] - wavelength_calibrated[0]))
    print("Spectral resolution of the original wavelength-calibrated observed data:", spectral_resolution_observed_binned, "mA")  

    #Eighth, make two axis in a figure sharing the x axis comparing 
    # the wavelength-calibrated observed data with the solar and telluric reference atlas.
    fig, ax = plt.subplots(nrows= 1, ncols=1, sharex=True, figsize=(10, 8))
    ax.plot(wavelength_calibrated, rest_spectrum, color='red', label='Wavelength-Calibrated Observed Data')
    #ax.plot(wavelength_calibrated, flat_spect, color='black', linestyle='--', label='Wavelength-Calibrated Grand Flat Data')
    ax.plot(solar_region_wavelengths, solar_region_intensities, color='blue', linestyle='--', label='Solar Reference Atlas')
    ax.plot(telluric_region_wavelengths, telluric_region_intensities, color='green', linestyle='--', label='Telluric Reference Atlas')
    ax.set_title("Wavelength-Calibrated Observed Data - Comparison with Solar and Telluric Reference Atlas")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.legend()
    plt.tight_layout()
    plt.show()


    #Test to rid out of the telluric lines from the wavelength-calibrated observed data.
    #First, we need to equaly sample the telluric reference atlas to match the wavelength grid of the observed data.
    
    #telluric_interpolator = interp1d(telluric_region_wavelengths, telluric_region_intensities, kind='linear', bounds_error=False, fill_value="extrapolate")
    #telluric_region_intensities_resampled = telluric_interpolator(wavelength_calibrated)
    #
    ##use the widen_multiple_lines function to stretch the telluric lines
    #telluric_region_intensities_stretched = widen_multiple_lines(wavelength_calibrated, telluric_region_intensities_resampled, central_wavelengths_atlas, stretch_factor=1.65, window_size=0.03)
#
    #rest_spectrum_corrected = rest_spectrum - (telluric_region_intensities_stretched - 1)
    ##plot the corrected spectrum to check the result
    #fig, ax = plt.subplots(nrows= 1, ncols=1, sharex=True, figsize=(10, 8))
    #ax.plot(wavelength_calibrated, rest_spectrum_corrected, color='red', label='Telluric-Corrected Observed Data')
    #ax.set_title("Telluric-Corrected Observed Data")
    #ax.set_xlabel("Wavelength (nm)")
    #ax.set_ylabel("Intensity")
    #ax.legend()
    #plt.tight_layout()
    #plt.show()



    #the final step is to create a full cube of the wavelength-calibrated observed data and save it as a new fits file. 
    #The new fits file will have a shape of [stokes, y (spatial dimension), x(scans), wavelength] including the new scale of calibrated wavelengths and it will have the same header as the original observed data file, but with the updated wavelength axis.
    #Remember: [scans, stokes, y, x]
    #We also add a binning of the wavelenght to avoid oversampling and reduce the data size.
    scans, stokes, y, wavelenght = stokes_cube[0].data.shape
    print(f"Original Stokes cube shape: {stokes_cube[0].data.shape}")

    if wavelenght % 2 != 0:
        print("Odd number of wavelength pixels, reducing by 1.")
        wavelenght -= 1
        wavelength_calibrated = wavelength_calibrated[:-1]

    binned_cond = input("Do you want binning of the spectral sampling? (yes/no)").strip().lower()
    
    if binned_cond == 'yes':
        wavelenght_binned = wavelenght // 2  # update the wavelength dimension after binning
        wavelength_calibrated = wavelength_calibrated.reshape(wavelenght_binned, 2).mean(axis=1) 
    elif binned_cond == 'no':
        wavelenght_binned = wavelenght

    wavelength_calibrated_cube = np.zeros_like(np.zeros([stokes, y, scans, wavelenght_binned]), dtype=np.float32)
    for i in range(scans):
        for j in range(stokes):
            #normalize the data to the maximum value of the rest spectrum
            stokes_cube[0].data[i, j, :, :] /= rest_spectrum_max
            #apply the binning to the wavelength axis with mean as operation
            if binned_cond == 'yes':
                wavelength_calibrated_cube[j, :, i, :] = stokes_cube[0].data[i, j, :, :wavelenght].reshape(y, wavelenght_binned, 2).mean(axis=2) 
            elif binned_cond == 'no':
                wavelength_calibrated_cube[j, :, i, :] = stokes_cube[0].data[i, j, :, :wavelenght]

    #Check the new spectral resolution of the wavelength-calibrated observed data.
    spectral_resolution_observed_binned = 10000 * ((wavelength_calibrated[1] - wavelength_calibrated[0]))
    print("Spectral resolution of the binned wavelength-calibrated observed data:", spectral_resolution_observed_binned, "mA")  

    print('Final wavelength-calibrated cube shape:', wavelength_calibrated_cube.shape)

    #Invert the wavelength axis to ensure increasing order
    wavelength_calibrated_cube = wavelength_calibrated_cube[:, :, :, ::-1]

    ##Ask the user to select the wavelength for a plot of the wavelength-calibrated cube to check the results.
    #wavelength_position = float(input("Enter the wavelength (in nm) to plot the wavelength-calibrated cube: "))
    #wavelength_index = np.argmin(np.abs(wavelength_calibrated - wavelength_position))
    #print(f"Wavelength index for {wavelength_position} nm:", wavelength_index)
    #plt.figure(figsize=(10, 5))
    #plt.imshow(wavelength_calibrated_cube[0, :, :, wavelength_index], aspect='auto', cmap='gray')
    #plt.title(f"Wavelength-Calibrated Cube - First Stokes Parameter at {wavelength_position} nm")
    #plt.xlabel("X (scans)")
    #plt.ylabel("Y (spatial dimension)")
    #plt.colorbar(label="Intensity")
    #plt.show()
#
    #if wavelength_calibrated_cube.shape[0] > 1:
    #    plt.figure(figsize=(10, 5))
    #    plt.imshow(wavelength_calibrated_cube[3, :, :, wavelength_index], aspect='auto', cmap='gray')
    #    plt.title(f"Wavelength-Calibrated Cube -  Stokes V Parameter at {wavelength_position} nm")
    #    plt.xlabel("X (scans)")
    #    plt.ylabel("Y (spatial dimension)")
    #    plt.colorbar(label="Intensity")
    #    plt.show()
#
    #Create a new fits file with the wavelength-calibrated observed data and save it with out replacing the original observed data file.
    #Include as a extra hdu the new wavelength axis and the original header of the observed data file.
    if ".fts" in observed_stokes_file:
        new_fits_file = observed_stokes_file.replace(".fts", "_norm_wavelength_calibrated.fits")
    else:
        new_fits_file = observed_stokes_file.replace(".fts", "_norm_wavelength_calibrated.fits")
    hdu = fits.PrimaryHDU(data=wavelength_calibrated_cube, header=stokes_cube[0].header)
    hdul = fits.HDUList([hdu])
    wavelength_hdu = fits.ImageHDU(data=wavelength_calibrated, header=stokes_cube[0].header)
    hdul.append(wavelength_hdu)
    hdul.writeto(new_fits_file, overwrite=True)



if __name__ == "__main__":
    main()

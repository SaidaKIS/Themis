import numpy as np
from astropy.io import fits
import astropy.units as u #Operations between units and constants
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
from scipy.optimize import minimize_scalar
from scipy.ndimage import map_coordinates
from interative_get_roi import get_roi
from scipy.optimize import curve_fit
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import glob
plt.rcParams.update({'figure.dpi': 200})

#------Basic procedure: Raw Data -> Dark Subtraction -> 
#      De-curving (Straightening) -> Flat-Field Division -> Affine Beam Alignment 
#      -> Stokes Demodulation

# Conclusion based on the analysis of the flat-field images: 
# The dual-beam MTR2 system has a slight misalignment between the two beams, 
# which can be corrected using an affine transformation. Additionally, 
# the spectral lines exhibit curvature that can be straightened using polynomial fitting. 
# measuring the diference between the flat images in both beams the distribution of the residuals is centered around zero (62) 
# with a standard deviation of ~700 counts,
# The gain factor 's' between the two beams is also determined to ensure accurate Stokes parameter extraction.

#Example:
# Camera 606 central wavelenght 6301 and 6302
# "260522_observation_test/t012_b0606_sp_20260522_073915_y3.fts"
# "260522_observation_test/t013_b0606_sp_20260522_073955_x3.fts"
# "260522_observation_test/t014_b0606_sp_20260522_080309_b3.fts"

# Camera 505 central wavelenght 5247 and 5250
# "260522_observation_test/t012_b0505_sp_20260522_073915_y3.fts"
# "260522_observation_test/t013_b0505_sp_20260522_073955_x3.fts"
# "260522_observation_test/t014_b0505_sp_20260522_080309_b3.fts"

# Third update: Calibration to generate only Stokes I - no polarization, and to save the Stokes cube in a FITS file with the original header from the science data.

# Fourth update: Calibrate several files in a batch mode - modification of the code structure to handle multiple science and flat files efficiently.
# use class objets as flats, darks and science files to streamline the calibration process

def straighten_spectral_lines(image, line_center_col, ymin, ymax, search_window=20):
    """
    Tracks a curved spectral line along the row axis (Y), fits a 2nd-order 
    polynomial, and straightens the entire image based on that curve.
    """
    height, width = image.shape
    y_indices = np.arange(ymin, ymax)
    detected_x = []
    
    # Step 1: Trace the center (minima/core) of the spectral line for each row
    for y in range(ymin, ymax):
        # Isolate a small horizontal window around the expected line position
        start_win = max(0, line_center_col - search_window)
        end_win = min(width, line_center_col + search_window)
        row_segment = image[y, start_win:end_win]
        
        # Find the pixel column where the absorption line is deepest (minimum intensity)
        local_min_idx = np.argmin(row_segment)
        actual_x = start_win + local_min_idx
        detected_x.append(actual_x)
        
    # Step 2: Fit a 2nd-degree polynomial (x = a*y^2 + b*y + c) to the curve
    poly_coefficients = np.polyfit(y_indices, detected_x, deg=2)

    # Step 3: Create a destination coordinate map for the un-warping process
    # We want to shift every pixel horizontally so the line becomes perfectly vertical

    straightened_image, fitted_line = maping_coord(image, poly_coefficients, line_center_col, order=1, mode='nearest')
    
    return straightened_image, poly_coefficients, fitted_line

def remove_spectral_lines(straightened_flat):
    """
    Removes the vertical spectral line profiles to isolate pure pixel-to-pixel spatial gain variations.
    """
    # Collapse along the Y-axis (spatial) to find the average spectral line profile
    spectral_profile = np.mean(straightened_flat, axis=0, keepdims=True)
    
    # Avoid division by zero
    spectral_profile[spectral_profile == 0] = 1.0
    
    # Divide out the spectral profile to leave ONLY the spatial sensor artifacts
    pure_spatial_flat = straightened_flat / spectral_profile
    
    return pure_spatial_flat

def maping_coord(image, poly_coefficients, fixed_x_target, order=1, mode='nearest'):

    height, width = image.shape
    straightened_image = np.zeros_like(image)
    line_x = []

    for y in range(height):
        # Find out where the curve thinks the line core is at this specific row
        current_line_x = np.polyval(poly_coefficients, y)
        line_x.append(current_line_x)
        
        # Calculate the sub-pixel shift required to pull it to the target column
        dx = current_line_x - fixed_x_target
        
        # Create an interpolated coordinate map for just this row
        # (Original column coordinates + the horizontal shift delta)
        shifted_cols = np.arange(width) + dx
        
        # We clamp the coordinates to ensure they don't break the boundaries
        shifted_cols = np.clip(shifted_cols, 0, width - 1)
        
        # Build coordinate grid for this specific row line
        coords = np.vstack((np.full(width, y), shifted_cols))
        
        # Re-sample the image line smoothly. 
        # FIX: Changed mode='edge' to mode='nearest' for scipy compatibility
        straightened_image[y, :] = map_coordinates(
            image, 
            coords, 
            order=order, 
            mode=mode
        )


    return straightened_image, np.array(line_x)

class MTR2_flat:
    def __init__(self, filepath, mode):
        self.filepath = filepath
        self.mode = mode
        timestamp = self.filepath.split('/')[-1].split('_')[3] + self.filepath.split('/')[-1].split('_')[4]
        timestamp_dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        self.timestamp = timestamp_dt

    def compute_calibration(self, dark_filepath=None, config_roi=None, plot_check=False):
        self.config_roi = config_roi
        if self.mode == 'non-pol':
            """
            Mode of observations with no polarization.
            Reads flat-field, applies dark subtraction if available, computes sub-pixel alignment and crop into the interesting region of the image. 
            This version is for Stokes I only (no polarization).
            """
            # Load the flat-field FITS image
            with fits.open(self.filepath) as hdul:
                self.data_raw = hdul[0].data
                self.flat_header = hdul[0].header
                # Convert raw data to signed integers and then to float for processing - Correct way for reading MATLAB fits
                data_sign = np.asanyarray(self.data_raw).view(np.int16).astype(np.float64)
                self.flat_data = data_sign + self.flat_header.get('BZERO', 0)  # Apply BZERO offset if present
                
            if self.flat_data.ndim == 3:
                self.flat_data = np.mean(self.flat_data, axis=0) # Average frames if it's a cube
                    
            # --- CRITICAL STEP: DARK CORRECTION (Performed first) ---
            
            if dark_filepath:
                print(f"Applying dark frame correction to flat-field using: {dark_filepath}")
                with fits.open(dark_filepath) as hdul:
                    data_raw = hdul[0].data
                    dark_header = hdul[0].header
                    # Convert raw data to signed integers and then to float for processing - Correct way for reading MATLAB fits
                    data_sign = np.asanyarray(data_raw).view(np.int16).astype(np.float64)
                    self.dark_data = data_sign + dark_header.get('BZERO', 0)  # Apply BZERO offset if present
                if self.dark_data.ndim == 3:
                    self.dark_data = np.mean(self.dark_data, axis=0)
                
                # Additive dark noise must be removed before dealing with multiplicative flats
                self.flat_data = self.flat_data - self.dark_data

        
            #Get the config file of the parameters interativelly
            if self.config_roi is None:
                self.config_roi = get_roi(self.flat_data, no_pol=True)
        
            self.b_flat = self.flat_data[: , :]

            self.height, self.width = self.flat_data.shape
        
            # Compute the the polynomial De-curving debugging
            self.b_flat_straight, self.poly_coeffs_b, self.fitted_line_b = straighten_spectral_lines(self.b_flat, line_center_col=self.config_roi['line_center_b1'], ymin=self.config_roi['lrg1'][0], ymax=self.config_roi['lrg1'][1])   
            b_spatial_flat = remove_spectral_lines(self.b_flat_straight)
        
            self.b_flat_norm = b_spatial_flat / np.mean(b_spatial_flat)
        
            if plot_check == True:
                plt.ioff()
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), sharex=True, sharey=True)
                p=ax.imshow(self.b_flat_norm, cmap='gray', origin='lower')
                ax.set_title(f"Normalized Flat")
                divider = make_axes_locatable(ax)
                cax1 = divider.append_axes('right', size='5%', pad="1%")
                cb1 = fig.colorbar(p, cax=cax1, orientation='vertical')
                            
                plt.tight_layout()
                plt.show()

            self.tform = None
            self.s_factor = None
            self.poly_coeffs_b1 = None
            self.poly_coeffs_b2 = None 
            self.b1_flat = None
            self.b2_flat = None
            self.b1_flat_norm = None
            self.b2_flat_norm = None
            self.half_y = None
            self.b_flat_mean = self.b_flat_norm

        elif self.mode == 'pol':
            """
            Reads flat-field, applies dark subtraction if available, splits the dual 
            beams, computes sub-pixel alignment, and finds the intensity scale factor 's'.
            """
            # Load the flat-field FITS image
            with fits.open(self.filepath) as hdul:
                self.data_raw = hdul[0].data
                self.flat_header = hdul[0].header
                # Convert raw data to signed integers and then to float for processing - Correct way for reading MATLAB fits
                data_sign = np.asanyarray(self.data_raw).view(np.int16).astype(np.float64)
                self.flat_data = data_sign + self.flat_header.get('BZERO', 0)  # Apply BZERO offset if present
            
            if self.flat_data.ndim == 3:
                self.flat_data = np.mean(self.flat_data, axis=0) # Average frames if it's a cube
                    
            # --- CRITICAL STEP: DARK CORRECTION (Performed first) ---
        
            if dark_filepath:
                print(f"Applying dark frame correction to flat-field using: {dark_filepath}")
                with fits.open(dark_filepath) as hdul:
                    data_raw = hdul[0].data
                    dark_header = hdul[0].header
                    # Convert raw data to signed integers and then to float for processing - Correct way for reading MATLAB fits
                    data_sign = np.asanyarray(data_raw).view(np.int16).astype(np.float64)
                    self.dark_data = data_sign + dark_header.get('BZERO', 0)  # Apply BZERO offset if present
                if self.dark_data.ndim == 3:
                    self.dark_data = np.mean(self.dark_data, axis=0)
                
                # Additive dark noise must be removed before dealing with multiplicative flats
                self.flat_data = self.flat_data - self.dark_data
        
            #Get the config file of the parameters interativelly
            if self.config_roi is None:
                self.config_roi = get_roi(self.flat_data)    
        
            self.height, self.width = self.flat_data.shape
            self.half_y = self.height // 2
            
            # Split into Top Beam (b1) and Bottom Beam (b2)
            self.b1_flat = self.flat_data[0:self.half_y, :]
            self.b2_flat = self.flat_data[self.half_y:, :]
        
            # Compute the the polynomial De-curving debugging
            self.b1_flat_straight, self.poly_coeffs_b1, self.fitted_line_b1 = straighten_spectral_lines(self.b1_flat, line_center_col=self.config_roi['line_center_b1'], ymin=self.config_roi['lrg1'][0], ymax=self.config_roi['lrg1'][1])   
            self.b2_flat_straight, self.poly_coeffs_b2, self.fitted_line_b2 = straighten_spectral_lines(self.b2_flat, line_center_col=self.config_roi['line_center_b2'], ymin=self.config_roi['lrg2'][0], ymax=self.config_roi['lrg2'][1])   
        
            self.b1_spatial_flat = remove_spectral_lines(self.b1_flat_straight)
            self.b2_spatial_flat = remove_spectral_lines(self.b2_flat_straight)
        
            if plot_check == True:
        
                # Checking the differnce between b1_flat and b1_flat_straight and plot it 
                diff_b1 = self.b1_flat - self.b1_flat_straight
                diff_b2 = self.b2_flat - self.b2_flat_straight
            
                plt.ioff()
                fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 5), sharex=True, sharey=True)
                ax[0][0].tick_params(bottom=True, top=True, left=True, right=False)
                ax[0][1].tick_params(bottom=True, top=True, left=True, right=False)
                ax[0][2].tick_params(bottom=True, top=True, left=True, right=False)
                ax[0][0].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
                ax[0][1].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
                ax[0][2].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
                ax[1][0].tick_params(bottom=True, top=True, left=True, right=False)
                ax[1][1].tick_params(bottom=True, top=True, left=True, right=False)
                ax[1][2].tick_params(bottom=True, top=True, left=True, right=False)
                ax[1][0].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
                ax[1][1].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
                ax[1][2].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
                ax[0][0].imshow(self.b1_flat, cmap='gray', origin='lower')
                ax[0][0].plot(self.fitted_line_b1, np.arange(self.half_y), color='red', linewidth=1.5, label='Fitted Curve')
                ax[0][0].set_title("Beam 1 Flat")
                ax[0][1].imshow(self.b1_flat_straight, cmap='gray', origin='lower')
                ax[0][1].set_title("Beam 1 Straightened")
                im = ax[0][2].imshow(self.b1_flat - self.b1_flat_straight, cmap='bwr', origin='lower')
                ax[0][2].set_title("Residual Difference (b1 - b1_straight)")
                plt.colorbar(im, ax=ax[0][2], fraction=0.046, pad=0.04)
        
                ax[1][0].imshow(self.b2_flat, cmap='gray', origin='lower')
                ax[1][0].plot(self.fitted_line_b2, np.arange(self.half_y), color='red', linewidth=1.5, label='Fitted Curve')
                ax[1][0].set_title("Beam 2 Flat")
                ax[1][1].imshow(self.b2_flat_straight, cmap='gray', origin='lower')
                ax[1][1].set_title("Beam 2 Straightened")
                im = ax[1][2].imshow(self.b2_flat - self.b2_flat_straight, cmap='bwr', origin='lower')
                ax[1][2].set_title("Residual Difference (b2 - b2_straight)")
                plt.colorbar(im, ax=ax[1][2], fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.show() 
        
            print(f"--- Beam 1 Straightening ---")
            print(f"Polynomial Coefficients for Beam 1: {self.poly_coeffs_b1}\n")    
            print(f"--- Beam 2 Straightening ---")
            print(f"Polynomial Coefficients for Beam 2: {self.poly_coeffs_b2}\n")    
        
            #This measures the exact shift needed to register b2 (moving) onto b1 (reference).
            # upsample_factor=100 enables ultra-precise 0.01 sub-pixel registration.
            shift, error, diffphase = phase_cross_correlation(
                self.b2_flat_straight, 
                self.b1_flat_straight, 
                upsample_factor=100
            )
            shift_y, shift_x = shift[0], shift[1]
        
            self.tform = AffineTransform(translation=(shift_x, shift_y))
        
            self.b2_flat_reg = warp(self.b2_flat_straight, self.tform, order=1)
            
            # Find scale factor 's' by minimizing intensity variance between channels
            def loss_function(s):
                diff = self.b1_flat_straight - s * self.b2_flat_reg
                return np.std(diff)
        
            # from scal=fminbnd(@(s) std(mean(f1R(:,c1wrg)-s*f2R(:,c1wrg),1)),0.5,1.5); in get_tform.m
            res = minimize_scalar(loss_function, bounds=(0.5, 1.5), method='bounded')
            self.s_factor = res.x
            
            print(f"--- Calibration Computed ---")
            print(f"Measured Alignment Shift -> X: {shift_x:.3f}px, Y: {shift_y:.3f}px")
            print(f"Calculated Gain Scale Factor (s): {self.s_factor:.4f}\n")
        
            # Normalize flats by their mean values so they only correct spatial response, not global intensity
            self.b1_flat_norm = self.b1_spatial_flat / np.mean(self.b1_spatial_flat)
            self.b2_flat_norm = self.b2_spatial_flat / np.mean(self.b2_spatial_flat)
        
            #Check the b1_flat_norm and b2_flat_norm after dark subtraction, straightening, and flat-fielding
            if plot_check == True:
                plt.ioff()
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)
                ax[0].imshow(self.b1_flat_norm, cmap='gray', origin='lower')
                ax[0].set_title(f"Beam 1 Normalized Flat")
                ax[1].imshow(self.b2_flat_norm, cmap='gray', origin='lower')
                ax[1].set_title(f"Beam 2 Normalized Flat")
                plt.tight_layout()
                plt.show()
        
            self.poly_coeffs_b = None
            self.b_flat_norm = None           

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return self.config_roi

    def processed_flats_evaluation(self, plot_check=False):
        # Apply polynomial straightening if coefficients are provided
        if self.poly_coeffs_b1 is not None and self.poly_coeffs_b2 is not None and self.config_roi is not None:
            b1_flat_eval,_ = maping_coord(self.b1_flat, self.poly_coeffs_b1, self.config_roi['line_center_b1'], order=1, mode='nearest')
            b2_flat_eval,_ = maping_coord(self.b2_flat, self.poly_coeffs_b2, self.config_roi['line_center_b2'], order=1, mode='nearest')
            
        if plot_check:
            plt.figure()
            plt.imshow(b1_flat_eval, aspect='auto', cmap='gray', origin='lower')
            plt.title("Beam 1 Flat After Straightening")
            plt.show()
            plt.figure()
            plt.imshow(b2_flat_eval, aspect='auto', cmap='gray', origin='lower')
            plt.title("Beam 2 Flat After Straightening")
            plt.show()
        
        # Warp beam 2 coordinates onto beam 1 coordinates
        b2_flat_reg = warp(b2_flat_eval, self.tform, order=1)
        
            # Compute the residual difference after alignment and scaling
        self.residual_diff = b1_flat_eval - self.s_factor * b2_flat_reg
        
        self.x_start, self.x_end = self.config_roi.get('crg1', [0, self.width]) if self.config_roi else [0, self.width]
        self.y_start, self.y_end = self.config_roi.get('lrg1', [0, self.half_y]) if self.config_roi else [0, self.half_y]
                
        self.b_flat_mean = (b1_flat_eval[self.y_start:self.y_end, self.x_start:self.x_end]+ self.s_factor * b2_flat_reg[self.y_start:self.y_end, self.x_start:self.x_end])/2
        
        if plot_check:
            plt.ioff()
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            ax[0].imshow(b1_flat_eval, cmap='gray', origin='lower')
            ax[0].set_title("Beam 1 Flat")
    
            ax[1].imshow(b2_flat_reg, cmap='gray', origin='lower')
            ax[1].set_title("Beam 2 Flat (Registered)")
    
            im = ax[2].imshow(self.residual_diff, cmap='bwr', origin='lower', vmin=-10000, vmax=10000)
            ax[2].set_title("Residual Difference (b1 - s*b2)")
            plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()    
    
            n, bins, patches = plt.hist(self.residual_diff.flatten(), bins=200, color='gray', alpha=0.7)
    
            bin_centers = (bins[:-1] + bins[1:]) / 2
    
            def gaussian_funtion(x, amp, mean, sigma):
                return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))
    
            p0 = [max(n), np.mean(self.residual_diff.flatten()), np.std(self.residual_diff.flatten())]
    
            popt, pcov = curve_fit(gaussian_funtion, bin_centers, n, p0=p0)
    
            x_curva = np.linspace(min(bins), max(bins), 100)
            y_curva = gaussian_funtion(x_curva, *popt)
    
            # Print gaussian fit parameters
            print(f"Gaussian Fit Parameters:")
            print(f"Amplitude: {popt[0]:.2f}")
            print(f"Mean: {popt[1]:.2f}")
            print(f"Sigma: {popt[2]:.2f}") 
    
            plt.plot(x_curva, y_curva, 'r-', linewidth=2, label=f'Gaussian Fit\n$mu={popt[1]:.2f}, sigma={popt[2]:.2f}$')
    
            plt.title('Histogram of Residual Differences and Gaussian Fit')
            plt.xlabel("Residual Intensity")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show() 
            
class MTR2_sci:
    def __init__(self, filepath, mode):
        self.filepath = filepath
        self.mode = mode
        timestamp = self.filepath.split('/')[-1].split('_')[3] + self.filepath.split('/')[-1].split('_')[4]
        timestamp_dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        self.timestamp = timestamp_dt
        
    def process_data(self, tform=None, s_factor=None, poly_coeffs_b=None, poly_coeffs_b1=None, poly_coeffs_b2=None,
                     b_flat=None, b1_flat=None, b2_flat=None, config_roi=None, dark_filepath=None, plot_check=False):

        self.tform = tform
        self.s_factor = s_factor
        self.poly_coeffs_b = poly_coeffs_b
        self.poly_coeffs_b1 = poly_coeffs_b1
        self.poly_coeffs_b2 = poly_coeffs_b2
        self.b_flat = b_flat
        self.b1_flat = b1_flat
        self.b2_flat = b2_flat
        self.config_roi = config_roi
        self.dark_filepath = dark_filepath
        self.plot_check = plot_check    

        if self.mode == 'non-pol':
            """
            Applies initial dark subtraction, maps alignment, normalizes gain, 
            and resolves the Stokes I component from the modulation sequence.
            This version is for Stokes I only (no polarization).
            """
            print("\nStarting the calibration and processing steps for Stokes I only...\n")
            # 1. Open with memmap=True (Do not use 'with' so the file stays mapped)
            hdul = fits.open(self.filepath)
            self.header = hdul[0].header
            
            # Keep it as an int16 view to save massive RAM
            raw_data_mapped = np.asanyarray(hdul[0].data).view(np.int16)
            self.bzero = self.header.get('BZERO', 0)
                
            print("Dark correction and Stokes I extraction will be applied to the science data...")
                    
            # --- CRITICAL STEP: DARK CORRECTION (Performed first) ---
            if self.dark_filepath:
                print(f"Applying dark frame correction to science data...")
                with fits.open(self.dark_filepath) as hdul:
                    data_raw_dark = hdul[0].data
                    dark_header = hdul[0].header
                    dark_raw = np.asanyarray(data_raw_dark).view(np.int16).astype(np.float32)
                    dark_bzero = dark_header.get('BZERO', 0)
                    if dark_bzero != 0:
                        dark_raw += dark_bzero
        
                self.dark_data = np.mean(dark_raw, axis=0) if dark_raw.ndim == 3 else dark_raw
                del dark_raw # Immediately free memory
                self.dark_data = self.dark_data.astype(np.float32)
            
            self.frames, self.height, self.width = raw_data_mapped.shape
            self.scans = self.header["NAXIS3"]

            seq_stk = self.header["SEQ_STOK"].split(" ")
            nsew_stk = len(seq_stk)
            if nsew_stk == 6:    
                self.scans = self.header["NAXIS3"]//nsew_stk
            else:
                self.scans = self.header["NAXIS3"]

            self.x_start, self.x_end = self.config_roi.get('crg1', [0, self.width]) if self.config_roi else [0, self.width]
            self.y_start, self.y_end = self.config_roi.get('lrg1', [0, self.height]) if self.config_roi else [0, self.height]
                
            print("Analyzing frame intensities within clean ROI...")
            avb = np.zeros(self.frames, dtype=np.float32)
            avb1 = np.zeros(self.frames, dtype=np.float32)
            
            for idx in tqdm(range(self.frames), desc="Profiling Frames", unit="frame"):
                # 1. Get raw frame stats
                    raw_b = raw_data_mapped[idx, :, :].astype(np.float32) + self.bzero
                    
                    avb[idx] = np.mean(raw_b[self.y_start:self.y_end, self.x_start:self.x_end]) # Combined global mean equivalent
            
                    # 2. Process frame just enough to grab the mean, then discard it from RAM
                    if self.dark_data is not None:
                        raw_b -= self.dark_data[:, :]
            
                    if self.poly_coeffs_b is not None and self.config_roi is not None:
                        raw_b, _ = maping_coord(raw_b, self.poly_coeffs_b, self.config_roi['line_center_b1'], order=1, mode='nearest')
        
                    if self.b_flat is not None:
                        raw_b /= self.b_flat
                
                    # Store only the 1D scalar vector tracks[cite: 5]
                    avb1[idx] = np.mean(raw_b[self.y_start:self.y_end, self.x_start:self.x_end])
            
                # Compute global scalars
            min_avb, max_avb = np.min(avb), np.max(avb)
            min_avb1, max_avb1 = np.min(avb1), np.max(avb1)
            
            # Pre-allocate ONLY the final output array
            self.scan_stokes = np.zeros((self.scans, 1, self.height, self.width), dtype=np.float32)
            
            print("Processing Stokes I ...")
                
            for s in tqdm(range(self.scans), desc="Processing Scans", unit="scan"):
                # Temporary small storage just for the current scan's modulation frames

                if nsew_stk == 6:
                    frame_idx = (s*nsew_stk)+1
                else:
                    frame_idx = s
            
                b_frame = raw_data_mapped[frame_idx, :, :].astype(np.float32) + self.bzero
            
                if self.dark_data is not None:
                    b_frame -= self.dark_data[:, :]
            
                if self.poly_coeffs_b is not None and self.config_roi is not None:
                    b_frame, _ = maping_coord(b_frame, self.poly_coeffs_b, self.config_roi['line_center_b1'], order=1, mode='nearest')
            
                if self.b_flat is not None:
                    b_frame /= self.b_flat
            
                self.scan_stokes[s, 0] = min_avb + ((b_frame - min_avb1) * (max_avb - min_avb)) / (max_avb1 - min_avb1)
            
            hdul.close()
            
            self.scan_stokes = self.scan_stokes[:,:,self.y_start:self.y_end, self.x_start:self.x_end]

        elif self.mode == 'pol':
            """
            Applies initial dark subtraction, maps alignment, normalizes gain, 
            and resolves the 4 Stokes components from the modulation sequence.
            """
            # 1. Open with memmap=True (Do not use 'with' so the file stays mapped)
            hdul = fits.open(self.filepath)
            self.header = hdul[0].header
        
            # Keep it as an int16 view to save massive RAM
            raw_data_mapped = np.asanyarray(hdul[0].data).view(np.int16)
            self.bzero = self.header.get('BZERO', 0)
        
            # For validation of the procedure
            self.input_local='no'
            
            print("Dark correction and Stokes demodulation will be applied to the science data...")
                
            # --- CRITICAL STEP: DARK CORRECTION (Performed first) ---
            if dark_filepath:
                print(f"Applying dark frame correction to science data...")
                with fits.open(dark_filepath) as hdul:
                    data_raw_dark = hdul[0].data
                    dark_header = hdul[0].header
                    dark_raw = np.asanyarray(data_raw_dark).view(np.int16).astype(np.float32)
                    dark_bzero = dark_header.get('BZERO', 0)
                    if dark_bzero != 0:
                        dark_raw += dark_bzero
        
                self.dark_data = np.mean(dark_raw, axis=0) if dark_raw.ndim == 3 else dark_raw
                del dark_raw # Immediately free memory
                self.dark_data = self.dark_data.astype(np.float32)
        
            self.frames, height, width = raw_data_mapped.shape
            half_y = height // 2
        
            self.seq_stk = self.header["SEQ_STOK"].split(" ")
            self.nsew_stk = len(self.seq_stk)
            self.scans = self.header["NAXIS3"]//self.nsew_stk
        
            x_start, x_end = self.config_roi.get('crg1', [0, width]) if self.config_roi else [0, width]
            y_start, y_end = self.config_roi.get('lrg1', [0, half_y]) if self.config_roi else [0, half_y]
            
            print("Analyzing frame intensities to calculate scaling profiles within clean ROI...")
            avb = np.zeros(self.frames, dtype=np.float32)
            avb1 = np.zeros(self.frames, dtype=np.float32)
            avb2 = np.zeros(self.frames, dtype=np.float32)
        
            for idx in tqdm(range(self.frames), desc="Profiling Frames", unit="frame"):
                # 1. Get raw frame stats
                raw_b1 = raw_data_mapped[idx, 0:half_y, :].astype(np.float32) + self.bzero
                raw_b2 = raw_data_mapped[idx, half_y:, :].astype(np.float32) + self.bzero
        
                avb[idx] = np.mean(raw_b1[y_start:y_end, x_start:x_end] + 
                                   raw_b2[y_start:y_end, x_start:x_end]) / 2.0 # Combined global mean equivalent
        
                # 2. Process frame just enough to grab the mean, then discard it from RAM
                if self.dark_data is not None:
                    raw_b1 -= self.dark_data[0:half_y, :]
                    raw_b2 -= self.dark_data[half_y:, :]
        
                if self.poly_coeffs_b1 is not None and self.poly_coeffs_b2 is not None and self.config_roi is not None:
                    raw_b1, _ = maping_coord(raw_b1, self.poly_coeffs_b1, self.config_roi['line_center_b1'], order=1, mode='nearest')
                    raw_b2, _ = maping_coord(raw_b2, self.poly_coeffs_b2, self.config_roi['line_center_b2'], order=1, mode='nearest')
        
                if self.b1_flat is not None:
                    raw_b1 /= self.b1_flat
                if self.b2_flat is not None:
                    raw_b2 /= self.b2_flat
        
                raw_b2_reg = warp(raw_b2, self.tform, order=1)
        
                # Store only the 1D scalar vector tracks[cite: 5]
                avb1[idx] = np.mean(raw_b1[y_start:y_end, x_start:x_end])
                avb2[idx] = np.mean(raw_b2_reg[y_start:y_end, x_start:x_end])
        
            # Compute global scalars
            min_avb, max_avb = np.min(avb), np.max(avb)
            min_avb1, max_avb1 = np.min(avb1), np.max(avb1)
            min_avb2, max_avb2 = np.min(avb2), np.max(avb2)
        
            # Pre-allocate ONLY the final output array
            self.scan_stokes = np.zeros((self.scans, 4, half_y, width), dtype=np.float32)
        
            print("Processing and demodulating Stokes parameters...")
            
            for s in tqdm(range(self.scans), desc="Processing Scans", unit="scan"):
                # Temporary small storage just for the current scan's modulation frames
                b1_chunk = np.zeros((self.nsew_stk, half_y, width), dtype=np.float32)
                b2_chunk = np.zeros((self.nsew_stk, half_y, width), dtype=np.float32)
        
                for k in range(self.nsew_stk):
                    frame_idx = s * self.nsew_stk + k
        
                    # Load individual 2D frame slice
                    b1_frame = raw_data_mapped[frame_idx, 0:half_y, :].astype(np.float32) + self.bzero
                    b2_frame = raw_data_mapped[frame_idx, half_y:, :].astype(np.float32) + self.bzero
        
                    if self.dark_data is not None:
                        b1_frame -= self.dark_data[0:half_y, :]
                        b2_frame -= self.dark_data[half_y:, :]
        
                    if self.poly_coeffs_b1 is not None and self.poly_coeffs_b2 is not None and self.config_roi is not None:
                        b1_frame, _ = maping_coord(b1_frame, self.poly_coeffs_b1, self.config_roi['line_center_b1'], order=1, mode='nearest')
                        b2_frame, _ = maping_coord(b2_frame, self.poly_coeffs_b2, self.config_roi['line_center_b2'], order=1, mode='nearest')
        
                    if self.b1_flat is not None:
                        b1_frame /= self.b1_flat
                    if self.b2_flat is not None:
                        b2_frame /= self.b2_flat
        
                    b2_frame_reg = warp(b2_frame, self.tform, order=1)
        
                    b1_chunk[k] = min_avb + ((b1_frame - min_avb1) * (max_avb - min_avb)) / (max_avb1 - min_avb1)
                    b2_chunk[k] = min_avb + ((b2_frame_reg - min_avb2) * (max_avb - min_avb)) / (max_avb2 - min_avb2)
        
                # Calculate differences and sums on the chunk
                bs = (b1_chunk - self.s_factor * b2_chunk) / 2.0
                is_map = (self.s_factor * (b1_chunk + b2_chunk)) / 2.0
            
                # Save straight into the final pre-allocated matrix
                self.scan_stokes[s, 0] = np.mean(is_map, axis=0)            
                self.scan_stokes[s, 1] = (bs[0] - bs[1]) / 2.0     
                self.scan_stokes[s, 2] = (bs[2] - bs[3]) / 2.0     
                self.scan_stokes[s, 3] = (bs[4] - bs[5]) / 2.0
        
            hdul.close()

            self.scan_stokes=self.scan_stokes[:,:,y_start:y_end, x_start:x_end]

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


if __name__ == "__main__":
    #Add the description of the code and the steps to follow in the README.md file
    #Printed at the beginning of the code to inform the user about the steps to follow
    print("Welcome to the MTR2 Data Processing Pipeline!")
    print("This script will guide you through the steps to initially process your MTR2 data.")
    print("Please ensure you have the following files ready:")
    print("1. Flat-field FITS file or files (flats). For several use '*' as a regular expression.")
    print("2. Dark-frame FITS file (darks) - Optional.")
    print("3. Science FITS file (science data). For several use '*' as a regular expression.")
    print("The script will perform the following steps:")
    print("1. Compute alignment and scale factor using the flat-field data.")
    print("2. Evaluate the calibration visually (optional).")
    print("3. Extract the final Stokes matrix from the science data.")
    print("Please follow the prompts to provide the necessary file paths.")

    checking_calibration = 0

    no_pol = input("Do you want to process the data with or without polarization? (with/without): ").strip().lower()
    if no_pol == "without":
        mode = 'non-pol'
    elif no_pol == "with":
        mode = 'pol'
    else:
        raise ValueError("Invalid input. Please enter 'with' or 'without'.")
    
    filename_dark = input("Enter the path and file of the dark (x3) (or leave blank if not available):")
    if filename_dark.strip() == "":
        filename_dark = None
    filenames_flat = input("Enter the path and file or files (* as regular expresion) of the flats (y3):")
    filenames_sci = input("Enter the path and file or files (* as regular expresion) of the science data (b3):")
    results_save_folder = input('Enter the path of the folder to save the stokes files:')
    list_raw_file_flats = glob.glob(filenames_flat)
    list_raw_file_sci = glob.glob(filenames_sci)
    list_raw_file_flats.sort()
    list_raw_file_sci.sort()
    list_flats=[]

    print("\nStarting the calibration and processing steps...\n")
    config_roi = None
    for ff in list_raw_file_flats:
        print('Processing '+ff+ " ...")
        raw_file_flat = MTR2_flat(ff, mode=mode)
        config_roi = raw_file_flat.compute_calibration(dark_filepath=filename_dark, config_roi=config_roi, plot_check=False)
        if mode == 'pol':
            print("Would you like to evaluate the flat-field calibration visually? (yes/no)")
            evaluate_flats = input().strip().lower()
            if evaluate_flats == "yes":
                raw_file_flat.processed_flats_evaluation(plot_check=True)
            else:
                raw_file_flat.processed_flats_evaluation(plot_check=False)           

        list_flats.append(raw_file_flat)

    for fc in list_raw_file_sci:
        print('Processing '+fc+ " ...")
        raw_file_sci = MTR2_sci(fc, mode=mode)
        #Check the closest flat for this science file using timestamps
        flat_ts_list = [flat.timestamp for flat in list_flats]
        closest_flat_index = min(range(len(flat_ts_list)), key=lambda i: abs(flat_ts_list[i] - raw_file_sci.timestamp))
        closest_flat = list_flats[closest_flat_index]

        raw_file_sci.process_data(tform=closest_flat.tform, s_factor=closest_flat.s_factor,
                                  poly_coeffs_b=closest_flat.poly_coeffs_b, poly_coeffs_b1=closest_flat.poly_coeffs_b1,
                                  poly_coeffs_b2=closest_flat.poly_coeffs_b2, b_flat=closest_flat.b_flat_norm,
                                  b1_flat=closest_flat.b1_flat_norm, b2_flat=closest_flat.b2_flat_norm, 
                                  config_roi=closest_flat.config_roi,
                                  dark_filepath=filename_dark)


        print(f"Processing of science file {fc} completed.\n")
        print(f"Final Stokes Cube Shape: {raw_file_sci.scan_stokes.shape}")

        if checking_calibration == 0:
            if mode == "non-pol":
                plt.ioff()
                fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
                stokes_labels = ['I']
                ax.imshow(raw_file_sci.scan_stokes[0, 0, :,:], cmap='gray', origin='lower')
                ax.set_title(f"Stokes {stokes_labels[0]}")
                plt.tight_layout()
                plt.show()
            else:
                plt.ioff()
                fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
                stokes_labels = ['I', 'Q', 'U', 'V']
                for i in range(4):
                    ax[i//2, i%2].imshow(raw_file_sci.scan_stokes[0, i, :,:], cmap='gray', origin='lower')
                    ax[i//2, i%2].set_title(f"Stokes {stokes_labels[i]}")
                plt.tight_layout()
                plt.show()

            print("Do you agree to proceed with the calibration of the science files? (yes/no)")
            proceed_stokes = input().strip().lower()
            if proceed_stokes != 'yes':
                print("Process aborted by user. Exiting.")
                exit(0)
            else:
                checking_calibration = 1
            
        file_name = 'stk_' + raw_file_sci.filepath.split('/')[-1]
        sci_file_hdu = fits.open(raw_file_sci.filepath)
        fits_header = sci_file_hdu[0].header.copy()  # Copy the original header to preserve metadata
        sci_file_hdu.close()
        #add some keywords from the original header to the new header
        fits_header['HISTORY'] = 'Processed with MTR2 pipeline'
        fits_header['DATE'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fits_header['NAXIS'] = 4
        fits_header['NAXIS1'] = raw_file_sci.scan_stokes.shape[3]  #Width
        fits_header['NAXIS2'] = raw_file_sci.scan_stokes.shape[2]  #Height
        fits_header['NAXIS3'] = raw_file_sci.scan_stokes.shape[1]  #Stokes parameters
        fits_header['NAXIS4'] = raw_file_sci.scan_stokes.shape[0]  #Scans
        fits_header['CDELT2'] = config_roi['arcperpix']
        if mode == "non-pol":
            fits_header['NOPOL'] = 'yes'
            fits_header['SCALE'] = None
            fits_header['STOKES'] = 'I'
            fits_header["CRPIX1"] = None  # X translation/offset
            fits_header["CRPIX2"] = None  # Y translation/offset
            fits_header["CD1_1"] = None  # X scaling/rotation
            fits_header["CD1_2"] = None
            fits_header["CD2_1"] = None
            fits_header["CD2_2"] = None
            fits_header['POLYB1'] = None
            fits_header['POLYB2'] = None
            flat_mean = None
        else:
            fits_header['NOPOL'] = 'no'
            fits_header['SCALE'] = raw_file_sci.s_factor
            matrix = raw_file_sci.tform.params
            fits_header["CRPIX1"] = float(matrix[0][2])  # X translation/offset
            fits_header["CRPIX2"] = float(matrix[1][2])  # Y translation/offset
            fits_header["CD1_1"] = float(matrix[0][0])  # X scaling/rotation
            fits_header["CD1_2"] = float(matrix[0][1])
            fits_header["CD2_1"] = float(matrix[1][0])
            fits_header["CD2_2"] = float(matrix[1][1])
            fits_header['POLYB1'] = str(raw_file_sci.poly_coeffs_b1)
            fits_header['POLYB2'] = str(raw_file_sci.poly_coeffs_b2)
            fits_header['STOKES'] = 'IQUV'
            flat_mean = closest_flat.b_flat_mean

        hdu = fits.PrimaryHDU(raw_file_sci.scan_stokes, header=fits_header)
        hdul = fits.HDUList([hdu])
        flat_hdu = fits.ImageHDU(data=flat_mean, header=fits_header)
        hdul.append(flat_hdu)
        hdul.writeto(results_save_folder+file_name, overwrite=True)
        print(f"Successfully saved clean Stokes parameters to '{file_name}'")





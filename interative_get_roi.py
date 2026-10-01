from configparser import NoOptionError

import numpy as np
import matplotlib
# Force a GUI backend so interactive windows pop up cleanly
matplotlib.use('TkAgg') # Can also change to 'TkAgg' depending on system setup
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import filters
from scipy.signal import find_peaks
import sys

def border_calc(flat_data, no_pol=False, slit_size=55):
    height, width = flat_data.shape    
    if no_pol:
        f = flat_data[: , :]
        edges = filters.sobel(f)
        v_strip = edges[:,width//2]
        peaks, _ = find_peaks(v_strip, distance=width//4)
        if len(peaks) > 2:
            r_border = [x for x in peaks if 10 < x < height - 10]
            if len(r_border) == 2:
                diff = np.abs(r_border[1] - r_border[0])
                arcpix = np.round(slit_size/diff, 4)
            else:
                r_border = [x for x in peaks if 20 < x < height - 20]
                if len(r_border) == 2:
                    diff = np.abs(r_border[1] - r_border[0])
                    arcpix = np.round(slit_size/diff, 4)
                else:
                    raise ValueError("No borders found")

        elif len(peaks) == 2:
            r_border = peaks
            diff = np.abs(r_border[1] - r_border[0])
            arcpix = np.round(slit_size/diff, 4)

        else:
            raise ValueError("No borders found")
        
    else:
        half_y = height // 2
        f = flat_data[0:half_y, :]
        edges = filters.sobel(f)
        v_strip = edges[:,width//2]
        peaks, _ = find_peaks(v_strip, distance=width//4)
        if len(peaks) > 2:
            r_border = [x for x in peaks if 10 < x < height - 10]
            if len(r_border) == 2:
                diff = np.abs(r_border[1] - r_border[0])
                arcpix = np.round(slit_size/diff, 4)
            else:
                r_border = [x for x in peaks if 20 < x < height - 20]
                if len(r_border) == 2:
                    diff = np.abs(r_border[1] - r_border[0])
                    arcpix = np.round(slit_size/diff, 4)
                else:
                    raise ValueError("No borders found")
        elif len(peaks) == 2:
            r_border = peaks
            diff = np.abs(r_border[1] - r_border[0])
            arcpix = np.round(slit_size/diff, 4)

        else:
            raise ValueError("No borders found")

    return arcpix  

def get_roi(flat_data, no_pol=False):
    """
    Complete interactive coordinate picker replicating the exact ginput workflow 
    found in MATLAB's get_tform.m.

    Parameters:
    flat_data: 2D numpy array of the flat-field image (shape: (height, width))
    no_pol: Boolean flag indicating if the data is non-polarimetric (default: False) so no need to split into two beams.
    Returns:
    config: Dictionary containing the selected coordinates for both beams:
        {
            "l1wrg": [bottom_row, top_row],  # Beam 1 line core
            "line_center_b1": int,           # Beam 1 line center (global coordinates)
            "lrg1": [top_row, bottom_row],   # Beam 1 final crop
            "crg1": [left_col, right_col],   # Beam 1 final crop
            "l2wrg": [bottom_row, top_row],  # Beam 2 line core
            "line_center_b2": int,           # Beam 2 line center (global coordinates)
            "lrg2": [top_row, bottom_row],   # Beam 2 final crop
            "crg2": [left_col, right_col]    # Beam 2 final crop
            "arcperpix": Measures the distance between the edges of the beams and give a value of arcsec per pixel
        }

    """
    if no_pol:
        # If no polarization, we can treat the entire flat_data as a single beam
        height, width = flat_data.shape
        arcperpix = border_calc(flat_data, no_pol=no_pol)

        f1 = flat_data[: , :]

        
        plt.ion()

        # =========================================================================
        # STEP 2: Beam (Top Field) Line Core Selection (c1wrg, l1wrg)
        # =========================================================================
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(f1, cmap='gray', origin='lower')
        ax.set_title("BEAM: Click LEFT then RIGHT of the line core to warp")
        plt.draw()
        print(">> Beam: Click LEFT then RIGHT edges of the spectral line core...")
        clicks_c = plt.ginput(n=2, timeout=0)
        c_wrg = sorted([int(clicks_c[0][0]), int(clicks_c[1][0])])

        line_center_b1 = int(np.mean(c_wrg))

        ax.set_title("BEAM: Click BOTTOM then TOP limits of the line to warp")
        plt.draw()
        print(">> Beam 1: Click BOTTOM then TOP vertical limits of the line core...")
        clicks_l1 = plt.ginput(n=2, timeout=0)
        l1wrg = sorted([int(clicks_l1[0][1]), int(clicks_l1[1][1])])
        plt.close(fig)
        
                
        #=========================================================================
        #STEP 4: Define Final Spatial ROI Crop Box 1 (crg1, lrg1)
        #=========================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(f1, cmap='gray', origin='lower')
        ax.set_title("4. FINAL CROP: Click TOP-LEFT then BOTTOM-RIGHT corners for clean ROI")
        plt.draw()
        print(">> Final Crop: Click TOP-LEFT then BOTTOM-RIGHT corners for spatial ROI...")
        
        clicks_roi = plt.ginput(n=2, timeout=0)
        x_roi = sorted([clicks_roi[0][0], clicks_roi[1][0]])
        y_roi = sorted([clicks_roi[0][1], clicks_roi[1][1]])
        crg1 = (int(x_roi[0]), int(x_roi[1]))
        lrg1 = (int(y_roi[0]), int(y_roi[1]))
        plt.close(fig)

        # Package all metadata into a clean dictionary configuration matrix
        config = {
            "l1wrg" : l1wrg,                     # [bottom_row, top_row] Beam 1 line core
            "line_center_b1": line_center_b1,       # Beam 1 line center (global coordinates)
            "lrg1": lrg1,                     # [top_row, bottom_row] Beam 1 final crop
            "crg1": crg1,                     # [left_col, right_col] Beam 1 final crop
            "l2wrg" : None,                     # [bottom_row, top_row] Beam 2 line core
            "line_center_b2": None,       # Beam 2 line center
            "lrg2": None,                     # [top_row, bottom_row] Beam 2 final crop
            "crg2": None,                      # [left_col, right_col] Beam 2 final crop
            "arcperpix" : arcperpix
        }

    else:
        height, width = flat_data.shape
        half_y = height // 2
        border_calc(flat_data, no_pol=no_pol)

        # Slice the flat data into Top and Bottom fields 
        f1 = flat_data[0:half_y, :]
        f2 = flat_data[half_y:, :]

        # Enable interactive plotting mode
        plt.ion()

        # =========================================================================
        # STEP 2: Beam 1 (Top Field) Line Core Selection (c1wrg, l1wrg)
        # =========================================================================
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(f1, cmap='gray', origin='lower')
        ax.set_title("2. BEAM 1: Click LEFT then RIGHT of the line core to warp")
        plt.draw()
        print(">> Beam 1: Click LEFT then RIGHT edges of the spectral line core...")
        clicks_c1 = plt.ginput(n=2, timeout=0)
        c1wrg = sorted([int(clicks_c1[0][0]), int(clicks_c1[1][0])])

        ax.set_title("2. BEAM 1: Click BOTTOM then TOP limits of the line to warp")
        plt.draw()
        print(">> Beam 1: Click BOTTOM then TOP vertical limits of the line core...")
        clicks_l1 = plt.ginput(n=2, timeout=0)
        l1wrg = sorted([int(clicks_l1[0][1]), int(clicks_l1[1][1])])
        plt.close(fig)

        # Calculate the exact horizontal line center for Beam 1 (offset back to global coordinates)
        line_center_b1 = int(np.mean(c1wrg))
#   
        # =========================================================================
        # STEP 3: Beam 2 (Bottom Field) Line Core Selection (c2wrg, l2wrg)
        # =========================================================================
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(f2, cmap='gray', origin='lower')
        ax.set_title("3. BEAM 2: Click LEFT then RIGHT of the line core to warp")
        plt.draw()
        print(">> Beam 2: Click LEFT then RIGHT edges of the spectral line core...")
        clicks_c2 = plt.ginput(n=2, timeout=0)
        c2wrg = sorted([int(clicks_c2[0][0]), int(clicks_c2[1][0])])

        ax.set_title("3. BEAM 2: Click BOTTOM then TOP limits of the line to warp")
        plt.draw()
        print(">> Beam 2: Click BOTTOM then TOP vertical limits of the line core...")
        clicks_l2 = plt.ginput(n=2, timeout=0)
        l2wrg = sorted([int(clicks_l2[0][1]), int(clicks_l2[1][1])])
        plt.close(fig)

        # Calculate the exact horizontal line center for Beam 2 (offset back to global coordinates)
        line_center_b2 = int(np.mean(c2wrg))

        #=========================================================================
        #STEP 4: Define Final Spatial ROI Crop Box 1 (crg1, lrg1)
        #=========================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(f1, cmap='gray', origin='lower')
        ax.set_title("4. FINAL CROP: Click TOP-LEFT then BOTTOM-RIGHT corners for clean ROI")
        plt.draw()
        print(">> Final Crop: Click TOP-LEFT then BOTTOM-RIGHT corners for spatial ROI...")

        clicks_roi = plt.ginput(n=2, timeout=0)
        x_roi = sorted([clicks_roi[0][0], clicks_roi[1][0]])
        y_roi = sorted([clicks_roi[0][1], clicks_roi[1][1]])
        crg1 = (int(x_roi[0]), int(x_roi[1]))
        lrg1 = (int(y_roi[0]), int(y_roi[1]))
        plt.close(fig)
#   
        ## =========================================================================
        ## STEP 5: Define Final Spatial ROI Crop Box 2 (crg2, lrg2)
        ## =========================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(f2, cmap='gray', origin='lower')
        ax.set_title("5. FINAL CROP: Click TOP-LEFT then BOTTOM-RIGHT corners for clean ROI")
        plt.draw()
        print(">> Final Crop: Click TOP-LEFT then BOTTOM-RIGHT corners for spatial ROI...")

        clicks_roi = plt.ginput(n=2, timeout=0)
        x_roi = sorted([clicks_roi[0][0], clicks_roi[1][0]])
        y_roi = sorted([clicks_roi[0][1], clicks_roi[1][1]])
        crg2 = (int(x_roi[0]), int(x_roi[1]))
        lrg2 = (int(y_roi[0]), int(y_roi[1]))
        plt.close(fig)

        #correction to have the same dimensions for both beams after cropping
        l1 = lrg1[1] - lrg1[0]
        l2 = lrg2[1] - lrg2[0]
        c1 = crg1[1] - crg1[0]
        c2 = crg2[1] - crg2[0]
        if l1 > l2:
            lrg1 = (lrg1[0], lrg1[0] + l2)
        elif l2 > l1:
            lrg2 = (lrg2[0], lrg2[0] + l1)
        if c1 > c2:     
            crg1 = (crg1[0], crg1[0] + c2)
        elif c2 > c1:
            crg2 = (crg2[0], crg2[0] + c1)


        # Package all metadata into a clean dictionary configuration matrix
        config = {
            "l1wrg" : l1wrg,                     # [bottom_row, top_row] Beam 1 line core
            "line_center_b1": line_center_b1,       # Beam 1 line center (global coordinates)
            "lrg1": lrg1,                     # [top_row, bottom_row] Beam 1 final crop
            "crg1": crg1,                     # [left_col, right_col] Beam 1 final crop
            "l2wrg" : l2wrg,                     # [bottom_row, top_row] Beam 2 line core
            "line_center_b2": line_center_b2,       # Beam 2 line center
            "lrg2": lrg2,                     # [top_row, bottom_row] Beam 2 final crop
            "crg2": crg2,                      # [left_col, right_col] Beam 2 final crop
            "arcperpix" : arcperpix
        }

    print("\n--- Interactive Selection Complete ---")
    
    return config


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
from matplotlib.widgets import SpanSelector


def border_calc(flat_data, no_pol=False, slit_size=55, plot_test=False):
    height, width = flat_data.shape    
    if no_pol:
        f = flat_data[: , :]
        edges = filters.sobel(f)
        if plot_test == True:
            fig, ax = plt.subplots(nrows=1,ncols=1)
            ax.imshow(edges, origin='lower', cmap='gray')
            plt.show()
        v_strip = edges[:,width//2]
        peaks, _ = find_peaks(v_strip, distance=width//4)
        if len(peaks) > 2:
            r_border = [x for x in peaks if width*0.05 < x < height - width*0.05]
            if len(r_border) == 2:
                diff = np.abs(r_border[1] - r_border[0])
                arcpix = np.round(slit_size/diff, 4)
            else:
                r_border = [x for x in peaks if width*0.1 < x < height - width*0.1]
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

def line_core_selector(f):
    r = {'c_wrg': None, 'line_center_b1': None, 'l1wrg': None}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(f, cmap='gray', origin='lower')
    ax.set_title("BEAM straighten spectral lines:\nClick left button to range HORIZONAL edges of an spectral line core.\nClick button to range VERTICAL edges of an spectral line core\n Escape to reset. Close to finish.")
    plt.draw()
    print(">> Beam: Click left button to range HORIZONAL edges and Click button to range VERTICAL edges of an spectral line core")

    def on_select_horizontal(xmin, xmax):
        r['c_wrg'] = [int(xmin), int(xmax)]
        r['line_center_b1'] = int(np.mean(r['c_wrg']))
        print(f">> Line core position saved: {r['c_wrg']}")

    def on_select_vertical(ymin, ymax):
        r['l1wrg'] = [int(ymin), int(ymax)]
        print(f">> Line heigth saved: {r['l1wrg']}")

    def on_key_press(event):
        if event.key == 'escape':
            r['c_wrg'] = None
            r['line_center_b1'] = None
            r['l1wrg'] = None
            
            # 2. Borrar los rectángulos sombreados de la pantalla
            span_x.clear()
            span_y.clear()
            
            # 3. Restaurar el título y refrescar el lienzo gráfico
            ax.set_title("BEAM straighten spectral lines: Click left button to range HORIZONAL edges of an spectral line core.\nClick button to range VERTICAL edges of an spectral line core\nEscape to reset.")
            fig.canvas.draw_idle()
            print("\n>> [RESET] Selection erased. Try again ...")

    span_x = SpanSelector(ax, on_select_horizontal, 'horizontal', button=1, useblit=True, props=dict(alpha=0.2, facecolor='green'))
    span_y = SpanSelector(ax, on_select_vertical, 'vertical', button=3, useblit=True, props=dict(alpha=0.2, facecolor='blue'))

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # Bloquea la ejecución aquí hasta que el usuario termine y la ventana se cierre
    plt.show(block=True) 
    
    # 4. Retornamos las variables locales limpiamente al final de la función
    return r['c_wrg'], r['line_center_b1'], r['l1wrg']

class rectangle_selector:
    def __init__(self, img):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(img, origin='lower', aspect='auto', cmap='gray')
        self.title = "FINAL CROP: Click TOP-LEFT then BOTTOM-RIGHT corners for clean ROI\nSelect rectangle region by dragging the mouse.\nExplicit red rectangle marks boundaries. Escape to reset. Close to finish."
        self.ax.set_title(self.title)
        self.rect = None
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.x0 = event.xdata
        self.y0 = event.ydata
        if self.rect is not None:
            self.rect.remove()
        self.rect = self.ax.add_patch(plt.Rectangle((self.x0, self.y0), 0, 0, edgecolor="red", facecolor="none", lw=1))
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
            self.ax.set_title(self.title)
            self.fig.canvas.draw_idle()
            print("\n>> [RESET] Selection erased. Try again ...")

    def show_plot(self):
        plt.show(block=True)
        return (int(self.xmin), int(self.xmax)), (int(self.ymin), int(self.ymax))

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
        c_wrg, line_center_b1, l1wrg = line_core_selector(f1)
        #=========================================================================
        #STEP 4: Define Final Spatial ROI Crop Box 1 (crg1, lrg1)
        #=========================================================================
        selector = rectangle_selector(f1)
        crg1, lrg1 = selector.show_plot()

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
        arcperpix=border_calc(flat_data, no_pol=no_pol)

        # Slice the flat data into Top and Bottom fields 
        f1 = flat_data[0:half_y, :]
        f2 = flat_data[half_y:, :]

        # Enable interactive plotting mode
        plt.ion()

        # =========================================================================
        # STEP 2: Beam 1 (Top Field) Line Core Selection (c1wrg, l1wrg)
        # =========================================================================
        c1wrg, line_center_b1, l1wrg = line_core_selector(f1)
#   
        # =========================================================================
        # STEP 3: Beam 2 (Bottom Field) Line Core Selection (c2wrg, l2wrg)
        # =========================================================================
        c2wrg, line_center_b2, l2wrg = line_core_selector(f2)

        #=========================================================================
        #STEP 4: Define Final Spatial ROI Crop Box 1 (crg1, lrg1)
        #=========================================================================
        selector1 = rectangle_selector(f1)
        crg1, lrg1 = selector1.show_plot()
#   
        ## =========================================================================
        ## STEP 5: Define Final Spatial ROI Crop Box 2 (crg2, lrg2)
        ## =========================================================================
        selector2 = rectangle_selector(f2)
        crg2, lrg2 = selector2.show_plot()

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


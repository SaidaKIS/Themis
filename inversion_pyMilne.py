import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import sys
import os
sys.path.append("../pyMilne")
import MilneEddington as ME
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure path to pyMilne repository
sys.path.append("../pyMilne")
import MilneEddington as ME


# 1. Update the Worker Initialization
def initialize_worker(regions_config, line_ids, precision='float32'):
    """
    Initializes the C++ backend once per worker process.
    regions_config: [[wavelength_array, psf_array], ...]
    """
    global me_inverter
    me_inverter = ME.MilneEddington(
        regions=regions_config,
        lines=line_ids,
        nthreads=1,          # ProcessPoolExecutor handles multi-core batching
        precision=precision
    )

# 2. Update the Worker Inversion Step
def invert_spatial_batch(batch_indices, stokes_batch, initial_guess_1d):
    """
    stokes_batch shape: (N_batch, 4, N_wvl)
    initial_guess_1d shape: (9,)
    """
    try:
        n_batch, n_stokes, n_wvl = stokes_batch.shape

        # 1. Reshape Stokes to 4D C-contiguous array: (ny=N_batch, nx=1, 4, N_wvl)
        stokes_4d = np.ascontiguousarray(
            stokes_batch.reshape((n_batch, 1, n_stokes, n_wvl)), 
            dtype=np.float32
        )

        # 2. Tile initial guess into a 3D C-contiguous array: (ny=N_batch, nx=1, 9)
        model_3d = np.ascontiguousarray(
            np.tile(initial_guess_1d, (n_batch, 1, 1)), 
            dtype=np.float32
        )

        # 3. Call invert(model, obs) -> returns (inverted_model, synthetic_spectra)
        inverted_model, _ , _ = me_inverter.invert(model_3d, stokes_4d)

        # 4. Extract (ny, nx, 9) and flatten back to 2D for batch assignment: (N_batch, 9)
        results_2d = inverted_model.reshape((n_batch, 9)).astype(np.float32)

        return batch_indices, results_2d

    except Exception as e:
        # Prints which batch of indices encountered an issue without killing execution silently
        print(f"Error processing batch starting at index {batch_indices[0]}: {str(e)}")
        raise e


# 3. Process a FITS data cube with parallelized Milne-Eddington inversion
def process_fits_cube(fits_path, out_path, wavelength, lines, initial_guess=None, psf=None, N_WORKERS=4, BATCH_SIZE=500):
    with fits.open(fits_path, memmap=True) as hdul:
        data_cube = hdul[0].data  # Shape: (4, Ny, Nx, N_wvl)
        
        if data_cube.ndim == 4:
            stokes_dim, ny, nx, n_wvl = data_cube.shape
            n_pixels = ny * nx
            reshaped_stokes = np.transpose(data_cube, (1, 2, 0, 3)).reshape((n_pixels, 4, n_wvl))
        else:
            raise ValueError(f"Expected 4D array (Stokes, Y, X, Lambda), got {data_cube.ndim}D")

        # Memory-mapped output storage on disk
        output_shape = (ny, nx, 9)
        out_memmap = np.memmap(
            out_path, 
            dtype='float32', 
            mode='w+', 
            shape=output_shape
        )

        if initial_guess is None:
            initial_guess = np.array([500., 1.0, 0.39, 0.1, 0.02, 30., 0.1, 0.2, 0.8], dtype=np.float32)
        else:
            initial_guess = np.array(initial_guess, dtype=np.float32)

        # Region format required by pyMilne: [[wav_array, psf_array]]
        regions_config = [[wavelength.astype(np.float32), psf]]

        print(f"Memory-mapped {fits_path}: {n_pixels} spatial pixels.")
        print(f"Parallelizing with {N_WORKERS} workers...")

        indices = np.arange(n_pixels)
        batch_ranges = [
            indices[i:i + BATCH_SIZE] 
            for i in range(0, n_pixels, BATCH_SIZE)
        ]

        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=initialize_worker,
            initargs=(regions_config, lines, 'float32')
        ) as executor:
            
            # Submit ALL batch tasks cleanly to the pool
            futures = [
                executor.submit(
                    invert_spatial_batch, 
                    idxs, 
                    reshaped_stokes[idxs], 
                    initial_guess
                )
                for idxs in batch_ranges
            ]

            completed = 0
            # Track as futures finish
            for future in as_completed(futures):
                idxs, results = future.result()
                
                # Assign to output array
                y_coords = idxs // nx
                x_coords = idxs % nx
                out_memmap[y_coords, x_coords, :] = results
                out_memmap.flush()

                completed += len(idxs)
                print(f"Progress: {completed}/{n_pixels} pixels ({(completed/n_pixels)*100:.1f}%)")

        # Save out to standard FITS format
        hdu = fits.PrimaryHDU(data=np.array(out_memmap))
        hdu.header['COMMENT'] = "pyMilne Milne-Eddington inversion parameters"
        hdu.writeto(out_path, overwrite=True)
        
        del out_memmap
        print("Inversion finished successfully for all pixels.")


if __name__ == "__main__":
    INPUT_FITS = input("Enter path to input FITS file: ")
    OUTPUT_FITS = input("Enter path to output FITS file: ")
    LINES_IN = input("Enter spectral line labels (e.g., 6301, 6302, 5247, 5250): ")
    LINES = [int(line.strip()) for line in LINES_IN.split(',')]

    # Calibrated wavelength axis in Ångströms
    WVL_nm = fits.open(INPUT_FITS)[1].data
    WVL_A = (WVL_nm * 10).astype(np.float32)
    
    N_WORKERS = max(1, os.cpu_count() - 2)
    BATCH_SIZE = 500

    #[|B| [G], inc [rad], azi [rad], vlos [km/s], vDop [\AA], eta_l, damp, S0, S1]
    m_in = np.array([500., 1.0, 0.39, 0.1, 0.02, 30., 0.1, 0.2, 0.8], dtype=np.float32)

    process_fits_cube(
        INPUT_FITS, 
        OUTPUT_FITS, 
        WVL_A, 
        LINES, 
        initial_guess=m_in, 
        N_WORKERS=N_WORKERS, 
        BATCH_SIZE=BATCH_SIZE
    )

    #Check inversion results by loading the output FITS file and inspecting its shape
    # and ploting
    inv_results = fits.open(OUTPUT_FITS)[0].data
    print(f"Inversion results shape: {inv_results.shape}")  

    plt.figure(figsize=(10, 5))
    plt.imshow(inv_results[:, :, 0]*np.s, aspect='auto', cmap='seismic',
                vmin=-np.mean(inv_results[:, :, 0]), vmax=np.mean(inv_results[:, :, 0]))
    plt.title("Inversion Results - Vertical magnetic field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Gauss")
    plt.show()























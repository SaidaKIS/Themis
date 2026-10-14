## Calibration procedure: MulTi-Ray spectrograph (MTR2) ##

This tutorial introduce to the procedure to calibrate L0 data to generate L1 and L1.5 data products of the MTR2 instrument. The scripts here included run in a Python 3 enviorment with the following requirements:

astropy==8.0.1
matplotlib==3.11.2
numpy==2.5.4
pandas==3.0.6
scipy==1.18.1
skimage==0.0
sunpy==8.0.0
tqdm==4.68.4

If you have access to THEMIS computers, you can activate the specific enviorment build using in a Terminal:

conda activate pythemis

1. Dual-beam MTR2 calibration:
As a first step, run python main_prep_MTR2_v2.py and follow the prompts to provide the necessary file paths and validation requests. Initially, indicate if you are processing full polarimetric data (Stokes I, Q, U, V) or spectroscopic data (Stokes I). This script processes the raw science using the flat-field and dark-frame files. The dual-beam MTR2 system has a slight misalignment between the two beams, which the script corrects using an affine transformation. Additionally, the spectral lines exhibit curvature, which the script corrects by straightening them using polynomial fitting. For this procedure, manually select a spectral line (x (green strip) and y (blue strip) range) to align and de-curve for both beams (polarimetry mode) or one beam (spectroscopy mode). Then select a clean region of interest, which will be the final spectral and spatial (along the slit) region considered for calibration.  

In the polarimetry mode, the script computes the alignment and scale factor of both beams and extracts the final Stokes matrix result of the beam combination per scan. In the spectroscopy mode, the script computes the alignment and extract a final Stokes I profile per scan.

A new fits file will be created, namely stk_(name of the original file).fits which should be used for the next calibration step.

You can calibrate several files at once if you provide the files path using '*' as a regular expression, for example: /home/MTR2/t0*_b0505_*_sp_b3.fts it will process all sequences acording to the expresion.

2. Wavelength calibration using reference atlas of solar and telluric spectral lines:
As the second step, run python wl_cal.py and follow the prompts to provide the fits file created previously and validation requests. You will be initially asked to provide a region on the FOV where to calculate the rest spectrum, consider a quite Sun region. Then, indicate the central spectral region of the data in nm, for example: 630.15 for the case of scans of the pair iron lines at 630.1 and 630.2 nm. Following, indicate the number of telluric lines to calibrate (max 2). If there is no telluric lines in the spectral range, consider any 2 solar lines to calibrate. Then select manually the spectral range per line and per window first in the   

Then Using the rest spectrum of the observed data, we consider several options to find the best matching region of the reference atlas with the observed data. 
#The options are defined with the number of telluric in the wavelength range of the observed data. 

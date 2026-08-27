import numpy as np
from astropy.io import fits
import astropy.units as u #Operations between units and constants
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import glob
import sys
from astropy.wcs import WCS
import os

#load solar and telluric reference atlas .npy files
solar_atlas = np.load("solar_reference_atlas.npy")
telluric_atlas = np.load("telluric_reference_atlas.npy")

#check structure of the reference atlas
print("Solar Reference Atlas shape:", solar_atlas.shape)
print("Telluric Reference Atlas shape:", telluric_atlas.shape)

#check the wavelength range of the reference atlas
solar_wavelength_range = (np.min(solar_atlas[0, :]), np.max(solar_atlas[0, :]))
telluric_wavelength_range = (np.min(telluric_atlas[0, :]), np.max(telluric_atlas[0, :]))
print("Solar Reference Atlas Wavelength Range (nm):", solar_wavelength_range)
print("Telluric Reference Atlas Wavelength Range (nm):", telluric_wavelength_range)

#plot a specific region of the solar and telluric reference atlas for comparison in the same figure
plt.figure(figsize=(10, 5))
solar_region_mask = (solar_atlas[0, :] >= 630.0) & (solar_atlas[0, :] <= 630.5)
plt.plot(solar_atlas[0, solar_region_mask], solar_atlas[1, solar_region_mask], color='blue')
plt.title("Solar and Telluric Reference Atlas (630-630.5 nm)")
telluric_region_mask = (telluric_atlas[0, :] >= 630.0) & (telluric_atlas[0, :] <= 630.5)
plt.plot(telluric_atlas[0, telluric_region_mask], telluric_atlas[1, telluric_region_mask], color='green')
plt.xlabel("Wavelength (nm)")
plt.tight_layout()

#plot a specific region of the solar and telluric reference atlas for comparison in the same figure
plt.figure(figsize=(10, 5))
solar_region_mask = (solar_atlas[0, :] >= 524.3) & (solar_atlas[0, :] <= 525.3)
plt.plot(solar_atlas[0, solar_region_mask], solar_atlas[1, solar_region_mask], color='blue')
plt.title("Solar and Telluric Reference Atlas (524.3-525.3 nm)")
telluric_region_mask = (telluric_atlas[0, :] >= 524.3) & (telluric_atlas[0, :] <= 525.3)
plt.plot(telluric_atlas[0, telluric_region_mask], telluric_atlas[1, telluric_region_mask], color='green')
plt.xlabel("Wavelength (nm)")
plt.tight_layout()
plt.show()

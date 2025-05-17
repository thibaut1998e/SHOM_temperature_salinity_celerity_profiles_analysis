# HydrologyMap

## Project Sponsor: SHOM

This project was developed in collaboration with **SHOM** (Service Hydrographique et Océanographique de la Marine), the French national hydrographic and oceanographic service. SHOM is responsible for collecting, analyzing, and disseminating maritime and oceanographic data to support navigation safety, maritime defense, and environmental monitoring. The organization plays a crucial role in understanding and mapping the marine environment, particularly in regions such as the Mediterranean Sea.

## Dataset Description

The dataset consists of **temperature and salinity profiles** measured at various geographic locations across the Mediterranean Sea. Each profile represents the **evolution of either temperature or salinity as a function of depth**. These profiles are associated with specific **latitude and longitude coordinates**, providing a 3D perspective (longitude, latitude, depth) on the physical properties of the sea water.

## Importance of Sound Speed Profiles for SHOM

From the collected **temperature and salinity profiles**, it is possible to compute the **speed of sound in seawater**. The sound speed is a critical variable for SHOM because it directly influences **underwater acoustic propagation**, which is essential for:

- **Sonar operations**
- **Submarine navigation**
- **Bathymetric mapping using echo sounding**
- **Marine geophysics and underwater communication**

Accurate knowledge of the sound speed profile enables SHOM to make precise measurements and ensure reliable acoustic data interpretation.

## Work Performed

The project involved multiple steps of data processing and analysis:

### 1. Interpolation of Temperature and Salinity

To create a continuous map of the Mediterranean, **interpolation techniques** were applied to the sparse measurement points. This allows estimation of temperature and salinity values at any location across the sea, forming a spatially continuous representation of the variables.

### 2. Clustering of Profiles

**Unsupervised classification methods** were used to group similar profiles together. The following clustering algorithms were implemented:

- **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)
- **KMeans** (Partitioning the dataset into k clusters based on similarity)

This step helps identify regions with similar hydrological behavior, which can be crucial for regional acoustic modeling and marine zoning.

### 3. Kalman Filter Implementation on Profiles

A **Kalman filter** was implemented from scratch (without using pre-built libraries) to smooth and predict the temperature and salinity profiles. The Kalman filter is a **recursive estimation algorithm** used to infer the hidden state of a system from noisy observations. In this context, it helps denoise the vertical profiles and improve the reliability of derived variables such as the sound speed.

### 4. Spatial Variogram Analysis

To better understand the spatial correlation structure of the physical properties of seawater (temperature and salinity), a variogram analysis was conducted. This technique, commonly used in geostatistics, quantifies how the similarity between values decreases as the spatial separation between them increases.



---

This project provides tools for visualizing and analyzing oceanographic profiles, which are essential for acoustic modeling in naval and marine research operations conducted by SHOM.

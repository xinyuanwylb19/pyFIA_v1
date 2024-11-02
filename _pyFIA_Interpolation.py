# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:17:48 2024

@author: xinyuan.wei
"""
#########################################################################
### Offline PyFIA Interplation State Biomass Map ###
#########################################################################
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import _pyFIA_State_Stats as ss
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Point
from scipy.interpolate import griddata

def bMap_Interpolate(tree_data, plot_data, coty_data,
                     cond_data, popp_data, pops_data, 
                     inv_yr, tree_size, n_years, 
                     basemap, FIPS, resolution, interp_method, savefile):
       
    # Calculate plot biomass for the state
    plot_biomass = ss.biomass(tree_data, plot_data, coty_data, 
                              cond_data, popp_data, pops_data, 
                              inv_yr, tree_size, n_years)[0]

    # Filter the study area using FIPS code
    study_area = basemap[basemap['STATEFP'] == FIPS]

    # Create a grid with the specified resolution
    lon_min, lat_min, lon_max, lat_max = study_area.total_bounds
    grid_lon, grid_lat = np.mgrid[lon_min:lon_max:resolution, lat_min:lat_max:resolution]

    # Interpolate biomass data using griddata
    points = plot_biomass[['LON', 'LAT']].values
    values = plot_biomass['PlotBM'].values
    grid_values = griddata(points, values, (grid_lon, grid_lat), method=interp_method)

    # Create a mask based on the study area shapefile
    mask = np.zeros_like(grid_values, dtype=bool)
    study_area_polygons = MultiPolygon([geom for geom in study_area.geometry if geom.is_valid])
    for i in range(grid_values.shape[0]):
        for j in range(grid_values.shape[1]):
            point = Point(grid_lon[i, j], grid_lat[i, j])
            mask[i, j] = not study_area_polygons.contains(point)

    # Apply the mask to the interpolated values
    grid_values[mask] = np.nan

    # Plot the interpolated biomass data
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid_values.T, extent=(lon_min, lon_max, lat_min, lat_max), 
                   origin='lower', cmap='viridis')
    plt.colorbar(im, label='Biomass density (kg/m²)')
    study_area.boundary.plot(ax=ax, color='black')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Interpolated biomass map')
    plt.show()

    # Save the interpolated biomass map to a GeoTIFF
    transform = from_origin(grid_lon[0, 0], grid_lat[-1, 0], resolution, resolution)
    
    with rasterio.open(savefile, 'w', driver='GTiff', height=grid_values.shape[1], 
                       width=grid_values.shape[0], count=1, dtype=grid_values.dtype,
                       crs=study_area.crs, transform=transform) as dst:
        dst.write(grid_values.T, 1)
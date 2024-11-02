# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:03:55 2024

@author: xinyuan.wei
"""
#########################################################################
### Offline Book Keeping Biomass Function ###
#########################################################################
'''
Kriging Interpolation:
Data is split into training and testing sets for model validation.
A Kriging model is trained using cross-validation to find the best parameters.
Model performance is evaluated using the test set.
'''
import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from pykrige.uk import UniversalKriging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from pyproj import Proj, transform
from scipy.spatial import cKDTree

#########################################################################
### Input information and load data ###
#########################################################################

def bookkeeping()

os.makedirs('Results_Bookkeeping', exist_ok=True)
biomass_plot = 'Results_Bookkeeping/Biomass_Plot.csv'

# Load the Tree, Condition, Plot, County, Pop_stratum, Pop_Stratum_Assgn
tree_data = pd.read_csv('ME_FIA_Data/ME_TREE.csv', low_memory=False)
cond_data = pd.read_csv('ME_FIA_Data/ME_COND.csv')
plot_data = pd.read_csv('ME_FIA_Data/ME_PLOT.csv')

# Open the forest data
forest_map_path = 'NLCD 2019_Maine/NLCD_2019_Maine.tif'
'''
with rasterio.open(forest_map_path) as src:
    # Display the TIFF file
    plt.figure(figsize=(10, 10))
    show(src)
'''
with rasterio.open(forest_map_path) as src:
    forest_map = src.read(1)  # Read the first band
    forest_meta = src.meta
''''  
#########################################################################
### Function to calculate the plot biomass ###
#########################################################################
def Plot_Biomass(inv_yr, size, tree_data, cond_data, plot_data):  
    # Filter the condition data with the inventory year and CONDID (CONDID = 1: forested)
    cond_req = ['INVYR', 'PLOT', 'CONDPROP_UNADJ']
    cnf_df = cond_data.loc[(cond_data['INVYR'] == inv_yr) & 
                           (cond_data['CONDID'] == 1), cond_req].copy()
    #cnf_df.to_csv('Plot_Condition.csv', index=False)
    
    # Filter the tree data with the inventory year
    tree_req = ['INVYR', 'PLOT', 'SUBP', 'DIA', 'CARBON_AG', 'CARBON_BG', 'TPA_UNADJ']
    tree_data = tree_data.loc[(tree_data['INVYR']==inv_yr)&
                              (tree_data['DIA']>size),tree_req].copy()
    #tree_data.to_csv('Filtered_Tree.csv', index=False)
    
    # Calculate the aboveground, belowground, and total biomass
    tree_data['treeAG'] = tree_data['CARBON_AG'] * tree_data['TPA_UNADJ']
    tree_data['treeBG'] = tree_data['CARBON_BG'] * tree_data['TPA_UNADJ']
    tree_data['treeBM'] = tree_data['treeAG'] + tree_data['treeBG']
    
    # Filter empty records
    tree_data.dropna(subset=['CARBON_AG'], inplace=True)
        
    # Sum biomass for each plot (a plot has four subplots)
    plot_sum = tree_data.groupby('PLOT')[['treeAG','treeBG','treeBM']].sum().reset_index()
    biomass_plot = plot_sum[['PLOT','treeAG','treeBG','treeBM']].copy()
    
    # Rename the columns
    biomass_plot = biomass_plot.rename(columns={'treeAG': 'PlotAG', 
                                                'treeBG': 'PlotBG', 
                                                'treeBM': 'PlotBM'})
     
    # Add the condition data on 'PLOT' field
    biomass_plot = pd.merge(biomass_plot, cnf_df, on='PLOT')
    
    # Add the latitude and lonitude data on 'PLOT' field     
    plot_location = plot_data
    plot_location = plot_location.drop_duplicates(subset=['PLOT'])
    biomass_plot = pd.merge(biomass_plot, plot_location[['PLOT','LAT']], on='PLOT')
    biomass_plot = pd.merge(biomass_plot, plot_location[['PLOT','LON']], on='PLOT')
    
    # Calculate the plot biomass density with condition propotion factor
    # pakgm: pounds per acre to kg per m2
    pakgm = 0.000112085
    
    condf = biomass_plot['CONDPROP_UNADJ'] 
    biomass_plot['PlotAG'] = biomass_plot['PlotAG'] * pakgm * condf 
    biomass_plot['PlotBG'] = biomass_plot['PlotBG'] * pakgm * condf 
    biomass_plot['PlotBM'] = biomass_plot['PlotBM'] * pakgm * condf 
    biomass_plot.to_csv('biomass_plot.csv', index=False)
    
    return (biomass_plot)
    
Plot_Biomass(inv_yr, 0, tree_data, cond_data, plot_data)
'''
#########################################################################
### Bookkeeping function to derive the state biomass map ###
#########################################################################
'''
# Define the projection information
proj_geo = Proj(init='epsg:4326')  # WGS84
proj_map = Proj(proj='aea', lat_1=29.5, lat_2=45.5, lat_0=23.0, lon_0=-96.0, x_0=0, y_0=0, datum='WGS84', units='m')

# Define forested pixel values
forested_values = [41, 42, 43, 90, 95]

# Create a mask for forested areas
forested_mask = np.isin(forest_map, forested_values)

# Load biomass plot data
biomass_data = pd.read_csv(biomass_plot)

# Extract coordinates and biomass values
coords_geo = biomass_data[['LON', 'LAT']].values
biomass_values = biomass_data['PlotAG'].values

# Convert geographic coordinates to map coordinates
coords_map = np.array([transform(proj_geo, proj_map, lon, lat) for lon, lat in coords_geo])

# Convert map coordinates to pixel coordinates
transform = from_origin(forest_meta['transform'][2], forest_meta['transform'][5], 30, 30)
inv_transform = ~transform

def coord_to_pixel(x, y, inv_transform):
    px, py = inv_transform * (x, y)
    return int(px), int(py)

# Get pixel coordinates for biomass plots
pixel_coords = np.array([coord_to_pixel(x, y, inv_transform) for x, y in coords_map])

# Visualize the forest map and biomass plot pixels
plt.figure(figsize=(10, 10))
plt.imshow(forest_map, cmap='gray', vmin=0, vmax=255)
plt.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=biomass_values, cmap='viridis', edgecolor='red')
plt.colorbar(label='Biomass (kgC/m2)')
plt.title('Forest Map with Biomass Plot Pixels')
plt.show()

# Interpolate biomass values to the entire map
forest_pixels = np.column_stack(np.where(forested_mask))
biomass_map = np.zeros_like(forest_map, dtype=float)

try:
    # Interpolation
    biomass_map[forested_mask] = griddata(pixel_coords, biomass_values, forest_pixels, method='linear', fill_value=0)
except ValueError as e:
    print(f"Linear interpolation failed: {e}")
    # Use nearest neighbor interpolation as a fallback
    biomass_map[forested_mask] = griddata(pixel_coords, biomass_values, forest_pixels, method='nearest', fill_value=0)

# Copy the forest map metadata and update data type and count
biomass_meta = forest_meta.copy()
biomass_meta.update(dtype=rasterio.float32, count=1)

# Save the biomass map
output_path = 'biomass_map.tif'
with rasterio.open(output_path, 'w', **biomass_meta) as dst:
    dst.write(biomass_map, 1)

print(f"Biomass map saved to {output_path}")
'''
# Define the projection information
proj_geo = Proj(init='epsg:4326')  # WGS84
proj_map = Proj(proj='aea', lat_1=29.5, lat_2=45.5, lat_0=23.0, lon_0=-96.0, x_0=0, y_0=0, datum='WGS84', units='m')

# Define forested pixel values
forested_values = [41, 42, 43, 90, 95]

# Create a mask for forested areas
forested_mask = np.isin(forest_map, forested_values)

# Load biomass plot data
biomass_data = pd.read_csv(biomass_plot)

# Extract coordinates and biomass values
coords_geo = biomass_data[['LON', 'LAT']].values
biomass_values = biomass_data['PlotAG'].values

# Convert geographic coordinates to map coordinates
coords_map = np.array([transform(proj_geo, proj_map, lon, lat) for lon, lat in coords_geo])

# Convert map coordinates to pixel coordinates
transform = from_origin(forest_meta['transform'][2], forest_meta['transform'][5], 30, 30)
inv_transform = ~transform

def coord_to_pixel(x, y, inv_transform):
    px, py = inv_transform * (x, y)
    return int(px), int(py)

# Get pixel coordinates for biomass plots
pixel_coords = np.array([coord_to_pixel(x, y, inv_transform) for x, y in coords_map])

# Interpolate biomass values to the entire map
forest_pixels = np.column_stack(np.where(forested_mask))

# Use cKDTree for faster interpolation
tree = cKDTree(pixel_coords)
distances, indices = tree.query(forest_pixels, k=1)
biomass_map = np.zeros_like(forest_map, dtype=float)
biomass_map[forested_mask] = biomass_values[indices]

# Visualize the forest map and biomass plot pixels
plt.figure(figsize=(10, 10))
plt.imshow(forest_map, cmap='gray', vmin=0, vmax=255)
plt.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=biomass_values, cmap='viridis', edgecolor='red')
plt.colorbar(label='Biomass (kgC/m2)')
plt.title('Forest Map with Biomass Plot Pixels')
plt.show()

# Copy the forest map metadata and update data type and count
biomass_meta = forest_meta.copy()
biomass_meta.update(dtype=rasterio.float32, count=1)

# Save the biomass map
output_path = 'biomass_map.tif'
with rasterio.open(output_path, 'w', **biomass_meta) as dst:
    dst.write(biomass_map, 1)

print(f"Biomass map saved to {output_path}")
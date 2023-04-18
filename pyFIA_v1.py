# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:40:21 2023
@author: xinyuan.wei
Updated 4/17/2023
"""

#########################################################################
### pyFIA Biomass ###
#########################################################################
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPolygon
from scipy.interpolate import griddata
from rasterio.transform import from_origin
#########################################################################
### pyFIA Functions and calculation units ###
#########################################################################
'''
Current functions:
    1. Calculate the state biomass for any inventory period.
    2. Calculate the state biomass for any size class.
    3. Plot all inventory plot for any state.
    4. Plot the county-level biomass density.
    5. Interpolate the plot biomass density to the entire state.
The output state-level biomass unit is TgC/year.
The population tree biomass unit is kg.
The carbon density unit is kgC/m2.
'''
#########################################################################
### Calculation performed ###
#########################################################################
# Plot all FIA plots?
plt_FIA = 'N'    # 'Y' or 'N'

# State biomass for any given inventory period (the last year).
inv_yr = 2021    # e.g., 1995 ...
size_state = 0   # The tree with a DBH larger than a specified value (inch).

# County level biomass density for any given inventory year (the last year).
inv_yc = 2016   # e.g., 1995 ...
size_county = 0 # The tree with a DBH larger than a specified value (inch).

# Interpolate the plot biomass density to a map (the last year).
inv_ym = 0   # e.g., 1995 ...

#########################################################################
### Load a state FIA data and US basemap ###
#########################################################################
# Load the Tree, Condition, Plot, County, Pop_stratum, Pop_Stratum_Assgn
tree_data = pd.read_csv('ME_CSV/ME_Tree.csv', low_memory=False)
cond_data = pd.read_csv('ME_CSV/ME_COND.csv')
plot_data = pd.read_csv('ME_CSV/ME_PLOT.csv')
coty_data = pd.read_csv('ME_CSV/ME_COUNTY.csv')
pops_data = pd.read_csv('ME_CSV/ME_POP_STRATUM.csv')
popp_data = pd.read_csv('ME_CSV/ME_POP_PLOT_STRATUM_ASSGN.csv', low_memory=False)

# Years have imventories
#inv_years = set(tree_data['INVYR'].unique())
#print(inv_years)
inv_years = {1995, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
             2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021}

# Load and read the US basemap
basemap = gpd.read_file('us_county/us_county.shp')

# State FIPS code (Maine: 23)
FIPS = '23'
#########################################################################
### Function to plot the FIA plots on a map ###
#########################################################################
def FIA_Plot(plot_data, basemap, FIPS):
    
    # Filter the shapefile data to only include target state
    state_map = basemap[basemap['STATEFP'] == FIPS]
    
    # Create a geometry column for the plot data
    geometry = [Point(xy) for xy in zip(plot_data['LON'], plot_data['LAT'])]
    plot_gdf = gpd.GeoDataFrame(plot_data, crs='EPSG:4269', geometry=geometry)

    # Plot the basemap and plot locations on a map
    fig, ax = plt.subplots(figsize=(12, 8))
    state_map.plot(ax=ax, alpha=0.5)
    plot_gdf.plot(ax=ax, markersize=1, color='black')
    ax.set_title('FIA Plot Locations')
    plt.show()

if plt_FIA == 'Y':
    FIA_Plot(plot_data, basemap, FIPS)

#########################################################################
### Function to calculate biomass with one year inventory ###
#########################################################################
def Biomass(inv_yr, size, tree_data, coty_data, 
            cond_data, popp_data, pops_data):
    
    # Filter the stratum data with the inventory year and EVALID
    # The last 4 digits of EVALID == year + 03
    # ADJ_FACTOR_SUBP, ADJ_FACTOR_MICR, and EXPNS
    popp_data = popp_data[popp_data['INVYR'] == inv_yr]
    lfd = inv_yr % 100 * 100 + 3
    popp_data = popp_data[popp_data['EVALID'] % 10000 == lfd]

    for index, row in popp_data.iterrows():
        stratum_cn = row['STRATUM_CN']
        matching_row = pops_data.loc[pops_data['CN'] == stratum_cn]
        
        # Get the values of EXPNS, ADJ_FACTOR_SUBP, and ADJ_FACTOR_MICR
        expns = matching_row.iloc[0]['EXPNS']
        asubp = matching_row.iloc[0]['ADJ_FACTOR_SUBP']
        amicr = matching_row.iloc[0]['ADJ_FACTOR_MICR']
        
        # Add the values to the popp_data DataFrame
        popp_data.at[index, 'EXPNS'] = expns
        popp_data.at[index, 'ADJ_FACTOR_SUBP'] = asubp
        popp_data.at[index, 'ADJ_FACTOR_MICR'] = amicr        
    #popp_data.to_csv('Plot_Expansion.csv', index=False)
    
    # Filter the condition data with the inventory year and CONDID 
    # The fraction of forested area for each plot (CONDID = 1: forested area)
    cond_req = ['INVYR', 'PLOT', 'CONDPROP_UNADJ']
    cnf_df = cond_data.loc[(cond_data['INVYR'] == inv_yr) & 
                           (cond_data['CONDID'] == 1), cond_req].copy()
    #cnf_df.to_csv('Plot_Condition.csv', index=False)
    
    # Add the county data on 'COUNTYCD' field
    tree_data = pd.merge(tree_data, coty_data, on='COUNTYCD')
    
    # Filter the tree data with the inventory year
    tree_req = ['INVYR', 'PLOT', 'SUBP', 'DIA', 'CARBON_AG', 'CARBON_BG',
                'TPA_UNADJ', 'COUNTYCD','COUNTYNM']
    tree_data = tree_data.loc[(tree_data['INVYR']==inv_yr)&
                              (tree_data['DIA']>size),tree_req].copy()
    #trf_df.to_csv('Tree_filtered.csv', index=False)
    
    # Calculate the aboveground, below, and total biomass for each tree class
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
    
    # Add the stratum data on 'PLOT' field
    biomass_plot = biomass_plot.merge(popp_data[['PLOT','EXPNS']], on='PLOT')
    biomass_plot = biomass_plot.merge(popp_data[['PLOT','ADJ_FACTOR_SUBP']], on='PLOT')
    biomass_plot = biomass_plot.merge(popp_data[['PLOT','ADJ_FACTOR_MICR']], on='PLOT')
    
    # Extend the plot tree biomass to the stratum biomass
    biomass_stra = biomass_plot
    fEXPN = biomass_stra['EXPNS']
    fSUBP = biomass_stra['ADJ_FACTOR_SUBP'] 
    fMICR = biomass_stra['ADJ_FACTOR_MICR']
    
    biomass_stra['StraAG'] = biomass_stra['PlotAG'] * fEXPN * fSUBP * fMICR
    biomass_stra['StraBG'] = biomass_stra['PlotBG'] * fEXPN * fSUBP * fMICR
    biomass_stra['StraBM'] = biomass_stra['PlotBM'] * fEXPN * fSUBP * fMICR
    #biomass_stra.to_csv('Biomass_Stratum.csv', index=False)
    
    # Calculate the county biomass density
    # Add the condition data on 'PLOT' field
    biomass_stra = pd.merge(biomass_stra, cnf_df, on='PLOT')
    
    # Calculate the plot biomass density with condition propotion factor
    #  pakgm: pounds per acre to kg per m2
    pakgm = 0.000112085
    condf = biomass_stra['CONDPROP_UNADJ']
    biomass_stra['pPlotAG'] = biomass_stra['PlotAG'] * condf * pakgm
    biomass_stra['pPlotBG'] = biomass_stra['PlotBG'] * condf * pakgm
    biomass_stra['pPlotBM'] = biomass_stra['PlotBM'] * condf * pakgm
    
    # Add the county information on 'PLOT' field
    tree_county = tree_data.drop_duplicates(subset=['PLOT'])
    biomass_stra = biomass_stra.merge(tree_county[['PLOT','COUNTYNM']], on='PLOT')
    
    # Calculate the county average biomass density
    biomass_coty = biomass_stra.groupby('COUNTYNM')[['pPlotAG','pPlotBG',
                                                     'pPlotBM']].mean().reset_index()

    return (biomass_stra, biomass_coty)
#########################################################################
### Function to plot the county biomass density ###
#########################################################################
def Biomass_Plot_County(basemap, FIPS, county_biomass):
    # Filter the counties to only include target state
    counties = basemap[basemap['STATEFP'] == FIPS]

    # Join the county shapefile with the biomass data
    biomass_geo = counties.merge(county_biomass, left_on='NAME', right_on='COUNTYNM')

    # Plot the biomass density map
    fig, ax = plt.subplots(figsize=(12,8))
    biomass_geo.plot(column='pPlotBM', cmap='YlGnBu', linewidth=0.5, 
                     edgecolor='gray', legend=True, ax=ax,
                     legend_kwds={'label': 'Biomass density (kgC/m2)'})

    ax.set_title('Biomass density (county level)')
    ax.set_axis_off()
    plt.show()
#########################################################################
### Function to interplate plot data to a map ###
#########################################################################
def Map_Interpolate(dataframe, shapefile, resolution, output, interp_method):
    # Load the data and shapefile
    study_data = dataframe
    study_area = shapefile

    # Create a grid with the specified resolution
    lon_min, lat_min, lon_max, lat_max = study_area.total_bounds
    grid_lon, grid_lat = np.mgrid[lon_min:lon_max:resolution, lat_min:lat_max:resolution]

    # Interpolate the plot biomass using griddata
    points = study_data[['LON', 'LAT']].values
    values = study_data['pPlotBM'].values
    grid_values = griddata(points, values, (grid_lon, grid_lat), method=interp_method)

    # Create a mask based on the study area shapefile
    mask = np.zeros_like(grid_values, dtype=bool)
    study_area_polygons = MultiPolygon([geom for geom in study_area.geometry])
    for i in range(grid_values.shape[0]):
        for j in range(grid_values.shape[1]):
            mask[i, j] = not study_area_polygons.contains(Point(grid_lon[i, j],
                                                                grid_lat[i, j]))

    # Apply the mask to the interpolated values
    grid_values[mask] = np.nan

    # Plot the interpolated biomass data
    fig, ax = plt.subplots()
    plt.imshow(grid_values.T, extent=(lon_min, lon_max, lat_min, lat_max), 
               origin='lower', cmap='viridis')
    plt.colorbar(label='Biomass density (kg/m2)')
    study_area.boundary.plot(ax=ax, color='black')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Interpolated bimass map')
    plt.show()

    # Save the interpolated biomass map to a GeoTIFF
    transform = from_origin(grid_lon[0, 0], grid_lat[-1, 0], resolution, resolution)
    
    with rasterio.open(output, 'w', driver='GTiff', height=grid_values.shape[1], 
                       width=grid_values.shape[0], count=1, dtype=grid_values.dtype,
                       crs=study_area.crs, transform=transform) as dst:
        dst.write(grid_values.T, 1)
#########################################################################
### Biomass for any given inventory // Main code ###
#########################################################################
if type(inv_yr) in [int, float, complex] and inv_yr in inv_years:
    # The five-year inventories used to calculate the biomass
    yr_list = [inv_yr, inv_yr - 1, inv_yr - 2, inv_yr - 3, inv_yr - 4]
        
    biomass_state = pd.DataFrame()  
    for inv_yr in yr_list: 
        #print('Processing for the data measured in year ', str(inv_yr))
        biomass_temp = Biomass(inv_yr, size_state, tree_data, coty_data, 
                               cond_data, popp_data, pops_data) [0]
            
        biomass_state = pd.concat([biomass_state, biomass_temp], ignore_index=True)
            
    biomass_state.to_csv('Biomass_State.csv', index=False)
        
    # Calculate the statewide biomass
    st_AG = round(biomass_state[['StraAG']].sum() * 4.53592e-10)
    st_BG = round(biomass_state[['StraBG']].sum() * 4.53592e-10)
    st_BM = round(biomass_state[['StraBM']].sum() * 4.53592e-10)
        
    print('The biomass estimated with inventories during', 
          str(yr_list[4]), '-', str(yr_list[0]), ':')
        
    print('Aboveground biomass: ', str(st_AG[0]), 'TgC.')
    print('Belowground biomass: ', str(st_BG[0]), 'TgC.')
    print('Total biomass: ', str(st_BM[0]), 'TgC.')
        
else:
    print('Check the inventory year for state biomass calculation!!!')
 
#########################################################################
### County biomass density for any given inventory // Main code ###
#########################################################################
if type(inv_yc) in [int, float, complex] and inv_yc in inv_years:
    biomass_county = Biomass(inv_yc, size_county, tree_data, coty_data, 
                             cond_data, popp_data, pops_data) [1]
            
    Biomass_Plot_County(basemap, FIPS, biomass_county)
    biomass_county.to_csv('Biomass_County.csv', index=False)
            
else:
    print('Check the inventory year for county-level biomass density plot!!!')
        
#########################################################################
### Interpolate the state biomass map // Main code ###
#########################################################################
'''
Interpolation method:
    'nearest': Nearest-neighbor interpolation
    'linear': Linear interpolation
    'cubic': Cubic interpolation
'''                                  
if type(inv_ym) in [int, float, complex] and inv_ym in inv_years:
    # The five-year inventories used to calculate the biomass
    yr_list = [inv_ym, inv_ym - 1, inv_ym - 2, inv_ym - 3, inv_ym - 4]
        
    biomass_map = pd.DataFrame()     
    for inv_ym in yr_list:
        biomass_temp = Biomass(inv_ym, size_state, tree_data, coty_data, 
                                   cond_data, popp_data, pops_data) [0]
        biomass_map = pd.concat([biomass_map, biomass_temp], ignore_index=True)
        
    # Add the latitude and lonitude data on 'PLOT' field     
    plot_location = plot_data
    plot_location = plot_location.drop_duplicates(subset=['PLOT'])
    biomass_map = pd.merge(biomass_map, plot_location[['PLOT','LAT']], on='PLOT')
    biomass_map = pd.merge(biomass_map, plot_location[['PLOT','LON']], on='PLOT')

    # Extract the state map.
    state_map = basemap[basemap['STATEFP'] == FIPS]  
    resolution = 0.1  
    output_file = 'Interpolated_Biomass_Density.tif'
    interp_method = 'nearest'
    
    Map_Interpolate(biomass_map, state_map, resolution, output_file, interp_method) 
    
else:
    print('Check the inventory year for interpolation!!!') 
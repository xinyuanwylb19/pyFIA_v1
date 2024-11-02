# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:13:12 2024

@author: xinyuan.wei
"""
#########################################################################
### Offline PyFIA Plot Functions ###
#########################################################################
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import matplotlib.colors as mcolors
import _pyFIA_State_Stats as ss
import _pyFIA_County_Stats as cs
from shapely.geometry import Point
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer

# Show all the inventory plots
def FIA_Plot(plot_data, basemap, FIPS, savefile):
    
    # Filter the basemap to include only the target state
    state_map = basemap[basemap['STATEFP'] == FIPS]

    # Create a geometry column for the plot data using longitude and latitude
    geometry = [Point(xy) for xy in zip(plot_data['LON'], plot_data['LAT'])]
    plot_gdf = gpd.GeoDataFrame(plot_data, crs='EPSG:4269', geometry=geometry)

    # Reproject both plot data and state map to EPSG:3857 for contextily
    state_map = state_map.to_crs(epsg=3857)
    plot_gdf = plot_gdf.to_crs(epsg=3857)

    # Plot the basemap and plot locations
    fig, ax = plt.subplots(figsize=(12, 8))
    state_map.plot(ax=ax, alpha=0.5, edgecolor='black', facecolor='none')
    plot_gdf.plot(ax=ax, markersize=2, color='black', marker='o', label='FIA Plots')

    # Add a basemap layer (e.g., OpenStreetMap)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Set title and labels
    ax.set_title('Locations of FIA inventory plots')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()

    # Convert x and y axis labels back to latitude and longitude
    def to_latlon(x, pos):
        point = gpd.GeoSeries([Point(x, 0)], crs="EPSG:3857").to_crs("EPSG:4326")
        return f"{point.x[0]:.2f}"

    def to_lat(x, pos):
        point = gpd.GeoSeries([Point(0, x)], crs="EPSG:3857").to_crs("EPSG:4326")
        return f"{point.y[0]:.2f}"

    ax.xaxis.set_major_formatter(FuncFormatter(to_latlon))
    ax.yaxis.set_major_formatter(FuncFormatter(to_lat))

    # Show the plot
    plt.show()
    
    # Save the figure
    fig.savefig(savefile, dpi=1200, bbox_inches='tight')

# Show state plot-level biomass density
def State_Biomass_Plot(tree_data, plot_data, coty_data, 
                       cond_data, popp_data, pops_data, 
                       tree_size, inv_yr, n_years, 
                       basemap, FIPS, savefile):
    # Calculate state biomass
    state_biomass_data = ss.biomass(tree_data, plot_data, coty_data, 
                                    cond_data, popp_data, pops_data, 
                                    inv_yr, tree_size, n_years)
    
    # Convert to DataFrame
    state_biomass = state_biomass_data[0]
    state_map = basemap[basemap['STATEFP'] == FIPS]

    # Create a GeoDataFrame from the biomass data with geometry
    geometry = gpd.points_from_xy(state_biomass['LON'], state_biomass['LAT'])
    biomass_geo = gpd.GeoDataFrame(state_biomass, crs='EPSG:4326', geometry=geometry)

    # Reproject to Web Mercator
    state_map_web = state_map.to_crs('EPSG:3857')
    biomass_geo_web = biomass_geo.to_crs('EPSG:3857')

    # Create color map from yellow to blue
    cmap = mcolors.LinearSegmentedColormap.from_list('yellow_to_blue', ['yellow', 'blue'])

    # Plot the basemap and biomass density
    fig, ax = plt.subplots(figsize=(12, 8))
    state_map_web.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
    biomass_geo_web.plot(column='PlotBM', ax=ax, markersize=10, 
                         legend=True, cmap=cmap, alpha=0.8, 
                         scheme='NaturalBreaks', k=7,)

    # Add a basemap layer with a specified zoom level
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10)

    ax.text(0.99, 0.27, 'Biomass density (kg/m²)', transform=ax.transAxes, ha='right',
            fontsize=10)
    
    # Initialize Transformer
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # Define custom formatter functions
    def format_lon(x, pos):
        lon, _ = transformer.transform(x, 0)
        return f'{lon:.2f}°'

    def format_lat(y, pos):
        _, lat = transformer.transform(0, y)
        return f'{lat:.2f}°'

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    # Set axis labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Plot-level biomass density')

    # Save the plot with tight bounding box
    plt.savefig(savefile, dpi=1200, bbox_inches='tight')

    # Show the plot
    plt.show()

# Show state plot-level biodiversity
def State_Biodiversity_Plot(tree_data, plot_data, inv_yr, n_years, 
                            basemap, FIPS, savefile):
    # Calculate state biomass
    state_biodiversity = ss.shannon(tree_data, plot_data, inv_yr, n_years)
    
    # Create a basemap
    state_map = basemap[basemap['STATEFP'] == FIPS]

    # Create a GeoDataFrame from the biomass data with geometry
    geometry = gpd.points_from_xy(state_biodiversity['LON'], state_biodiversity['LAT'])
    biomass_geo = gpd.GeoDataFrame(state_biodiversity, crs='EPSG:4326', geometry=geometry)

    # Reproject to Web Mercator
    state_map_web = state_map.to_crs('EPSG:3857')
    biomass_geo_web = biomass_geo.to_crs('EPSG:3857')

    # Create color map from yellow to blue
    cmap = mcolors.LinearSegmentedColormap.from_list('yellow_to_blue', ['yellow', 'blue'])

    # Plot the basemap and diodiversity
    fig, ax = plt.subplots(figsize=(12, 8))
    state_map_web.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
    biomass_geo_web.plot(column='shannon', ax=ax, markersize=10, 
                         legend=True, cmap=cmap, alpha=0.8, 
                         scheme='NaturalBreaks', k=7,)

    # Add a basemap layer with a specified zoom level
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10)

    ax.text(0.99, 0.27, 'Shannon index', transform=ax.transAxes, ha='right',
            fontsize=10)
    
    # Initialize Transformer
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # Define custom formatter functions
    def format_lon(x, pos):
        lon, _ = transformer.transform(x, 0)
        return f'{lon:.2f}°'

    def format_lat(y, pos):
        _, lat = transformer.transform(0, y)
        return f'{lat:.2f}°'

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    # Set axis labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Plot-level shannon index')

    # Save the plot with tight bounding box
    plt.savefig(savefile, dpi=1200, bbox_inches='tight')

    # Show the plot
    plt.show()

# Show county biomass density
def County_Biomass_Plot(tree_data, plot_data, coty_data, 
                        cond_data, popp_data, pops_data, 
                        inv_yr, tree_size, n_years, 
                        basemap, FIPS, savefile):
    # Calculate the county biomass density
    County_Biomass = cs.biomass(tree_data, plot_data, coty_data,
                                cond_data, popp_data, pops_data, 
                                inv_yr, tree_size, n_years)
    
    # Filter the counties in the shapefile to only include target state
    counties = basemap[basemap['STATEFP'] == FIPS]

    # Join the county shapefile with the biomass data
    biomass_geo = counties.merge(County_Biomass, left_on='NAME', right_on='COUNTYNM')

    # Plot the biomass density map
    fig, ax = plt.subplots(figsize=(12,8))
    biomass_geo.plot(column='PlotBM', cmap='YlGnBu', linewidth=0.1, 
                     edgecolor='black', legend=True, ax=ax,
                     legend_kwds={'label': 'Biomass density (kg/m²)'})
    
    ax.set_title('County biomass density')
    ax.set_axis_off()
    
    # Save the plot as a file
    plt.savefig(savefile, dpi=1200, bbox_inches='tight')
        
    # Show the plot    
    plt.show()


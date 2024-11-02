# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 08:34:18 2024

@author: xinyuan.wei
"""
import pandas as pd
import geopandas as gpd
import _pyFIA_State_Stats as ss
import _pyFIA_County_Stats as cs
import _pyFIA_Map_Plots as plts
import _pyFIA_Interpolation as inpl

# Load the FIA Tree, Condition, Plot, County, Pop_stratum, Pop_Stratum_Assgn data
tree_data = pd.read_csv('_ME_FIA_Data/ME_TREE.csv', low_memory=False)
cond_data = pd.read_csv('_ME_FIA_Data/ME_COND.csv')
plot_data = pd.read_csv('_ME_FIA_Data/ME_PLOT.csv')
coty_data = pd.read_csv('_ME_FIA_Data/ME_COUNTY.csv')
pops_data = pd.read_csv('_ME_FIA_Data/ME_POP_STRATUM.csv')
popp_data = pd.read_csv('_ME_FIA_Data/ME_POP_PLOT_STRATUM_ASSGN.csv', low_memory=False)

# Load the US basemap (shapefile)
basemap = gpd.read_file('_US_County/us_county.shp')

# State FIPS code (Maine: 23)
FIPS = '23'

# Summarize the state forest biomass/carbon
State_Biomass = ss.biomass(tree_data=tree_data,
                           plot_data=plot_data,
                           coty_data=coty_data,
                           cond_data=cond_data,
                           popp_data=popp_data,
                           pops_data=pops_data,
                           inv_yr=2021,            # The ending inventory year.
                           tree_size=0,            # Minimum tree diameter (inch)
                           n_years=5)              # Inventory cycle (years)
savefile = '_Results/State_Plot_Biomass.csv'
State_Biomass[0].to_csv(savefile, index=False)   
print('State Forest Biomass/Carbon Summary:')
for key, value in State_Biomass[1].items():
    print(f"{key}: {value}")

# Summarize the county-level forest biomass/carbon within a state
County_Biomass = cs.biomass(tree_data=tree_data,
                            plot_data=plot_data,
                            coty_data=coty_data,
                            cond_data=cond_data,
                            popp_data=popp_data,
                            pops_data=pops_data,
                            inv_yr=2021,            # The ending inventory year. 
                            tree_size=0,            # Minimum tree diameter (inch)
                            n_years=5)              # Inventory cycle (years)
savefile = '_Results/County_Biomass.csv'
County_Biomass.to_csv(savefile, index=False)   
print('County Average Biomass/Carbon:')
print(County_Biomass)

# Calculate the biodiversity of a state
State_Biodiversity = ss.shannon(tree_data=tree_data,
                                plot_data=plot_data,
                                inv_yr=2021,            # The ending inventory year.
                                n_years=5)              # Inventory cycle (years)
savefile = '_Results/State_Biodiversity.csv'
State_Biodiversity.to_csv(savefile, index=False) 

# Show all the invotory plots in a state
savefile = '_Results/Plot_Location.png'
plts.FIA_Plot(plot_data, basemap, FIPS, savefile)

# Show state plot-level biomass density
savefile = '_Results/State_plot_biomass.png'
plts.State_Biomass_Plot(tree_data=tree_data,
                        plot_data=plot_data,
                        coty_data=coty_data,
                        cond_data=cond_data,
                        popp_data=popp_data,
                        pops_data=pops_data,
                        inv_yr=2021,           # The ending inventory year. 
                        tree_size=0,           # Minimum tree diameter (inch)
                        n_years=5,             # Inventory cycle (years)
                        basemap=basemap,
                        FIPS=FIPS,
                        savefile=savefile) 
 
# Show state plot-level biomass density
savefile = '_Results/State_plot_biodiversity.png'
plts.State_Biodiversity_Plot(tree_data=tree_data,
                             plot_data=plot_data,
                             inv_yr=2021,           # The ending inventory year. 
                             n_years=5,             # Inventory cycle (years)
                             basemap=basemap,
                             FIPS=FIPS,
                             savefile=savefile)  

# Show county level biomass density
savefile = '_Results/County_Biomass_Density.png'
plts.County_Biomass_Plot(tree_data=tree_data,
                         plot_data=plot_data,
                         coty_data=coty_data,
                         cond_data=cond_data,
                         popp_data=popp_data,
                         pops_data=pops_data,
                         inv_yr=2021,           # The ending inventory year. 
                         tree_size=0,           # Minimum tree diameter (inch)
                         n_years=5,             # Inventory cycle (years)
                         basemap=basemap,
                         FIPS=FIPS,
                         savefile=savefile)               

# Interplate the state biomass map
savefile = '_Results/Interplation_Biomass_Map.png'
inpl.bMap_Interpolate(tree_data=tree_data,
                      plot_data=plot_data,
                      coty_data=coty_data,
                      cond_data=cond_data,
                      popp_data=popp_data,
                      pops_data=pops_data,
                      inv_yr=2021,           # The ending inventory year. 
                      tree_size=0,           # Minimum tree diameter (inch)
                      n_years=5,             # Inventory cycle (years) 
                      basemap=basemap,
                      FIPS=FIPS,
                      resolution=0.25,
                      interp_method='nearest',
                      savefile=savefile)         



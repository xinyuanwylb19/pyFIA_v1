# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:03:55 2024

@author: xinyuan.wei
"""
#########################################################################
### Offline PyFIA County Biomass Function ###
#########################################################################
import _pyFIA_State_Stats as ss

def biomass(tree_data, plot_data, coty_data, 
            cond_data, popp_data, pops_data,
            inv_yr, tree_size, n_years):
    
    # Call the state_biomass function to get the plot-level biomass data
    state_plot_biomass = ss.biomass(tree_data, plot_data, coty_data, 
                                          cond_data, popp_data, pops_data, 
                                          inv_yr, tree_size, n_years)[0]

    # Calculate the average biomass for each county
    county_biomass_df = state_plot_biomass.groupby('COUNTYNM')[['PlotAG', 'PlotBG', 'PlotBM']].mean().reset_index()

    return county_biomass_df

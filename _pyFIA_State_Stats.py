# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:40:21 2023
@author: xinyuan.wei
Updated 5/01/2023
"""
#########################################################################
### Offline PyFIA State Biomass Function ###
#########################################################################
import pandas as pd
import numpy as np

def biomass(tree_data, plot_data, coty_data, 
            cond_data, popp_data, pops_data,
            inv_yr, tree_size, n_years):

    # Create a list of inventory years (an inventory cycle)
    inv_yrs = [inv_yr - i for i in range(n_years)]

    # Initialize an empty DataFrame to collect plot biomass data
    state_plot_biomass = pd.DataFrame()

    for year in inv_yrs:
        # Filter the population plot data for the current inventory year
        popp_filtered = popp_data[popp_data['INVYR'] == year].copy()
        lfd = year % 100 * 100 + 3
        popp_filtered = popp_filtered[popp_filtered['EVALID'] % 10000 == lfd]

        # Merge with population stratum data to get expansion factors
        popp_merged = popp_filtered.merge(
            pops_data[['CN', 'EXPNS', 'ADJ_FACTOR_SUBP', 'ADJ_FACTOR_MICR']],
            left_on='STRATUM_CN', right_on='CN', how='left'
        )

        # Filter condition data for the current inventory year
        cond_filtered = cond_data[cond_data['INVYR'] == year][['PLOT', 'CONDPROP_UNADJ']].copy()

        # Merge tree data with county data to get county names
        tree_merged = tree_data.merge(
            coty_data[['COUNTYCD', 'COUNTYNM']], on='COUNTYCD', how='left'
        )

        # Filter tree data for the current inventory year and diameter size
        tree_filtered = tree_merged[
            (tree_merged['INVYR'] == year) & (tree_merged['DIA'] > tree_size)
        ].copy()

        # Remove records with missing carbon data
        tree_filtered.dropna(subset=['CARBON_AG', 'CARBON_BG', 'TPA_UNADJ'], inplace=True)

        # Calculate tree-level aboveground, belowground, and total biomass
        tree_filtered['treeAG'] = tree_filtered['CARBON_AG'] * tree_filtered['TPA_UNADJ']
        tree_filtered['treeBG'] = tree_filtered['CARBON_BG'] * tree_filtered['TPA_UNADJ']
        tree_filtered['treeBM'] = tree_filtered['treeAG'] + tree_filtered['treeBG']

        # Sum biomass at the plot level
        plot_biomass = tree_filtered.groupby('PLOT')[['treeAG', 'treeBG', 'treeBM']].sum().reset_index()
        plot_biomass.rename(columns={'treeAG': 'PlotAG', 'treeBG': 'PlotBG', 'treeBM': 'PlotBM'}, inplace=True)

        # Merge plot biomass with population data to get expansion factors
        biomass_plot = plot_biomass.merge(
            popp_merged[['PLOT', 'EXPNS', 'ADJ_FACTOR_SUBP', 'ADJ_FACTOR_MICR']],
            on='PLOT', how='left'
        )

        # Merge with condition data to get condition proportion
        biomass_plot = biomass_plot.merge(cond_filtered, on='PLOT', how='left')

        # Merge with plot data to get latitude and longitude
        plot_location = plot_data[['PLOT', 'LAT', 'LON']].drop_duplicates(subset=['PLOT'])
        biomass_plot = biomass_plot.merge(plot_location, on='PLOT', how='left')

        # Fill missing condition proportions with 1 (assumes full plot is forested)
        biomass_plot['CONDPROP_UNADJ'] = biomass_plot['CONDPROP_UNADJ'].fillna(1)

        # Convert plot biomass to per unit area and adjust with condition proportion
        pakgm = 0.000112085  # Conversion factor from pounds per acre to kg per m^2
        biomass_plot['PlotAG'] *= pakgm * biomass_plot['CONDPROP_UNADJ']
        biomass_plot['PlotBG'] *= pakgm * biomass_plot['CONDPROP_UNADJ']
        biomass_plot['PlotBM'] *= pakgm * biomass_plot['CONDPROP_UNADJ']

        # Calculate stratum-level biomass using expansion factors
        m2a = 4046.8564224  # Conversion factor from acres to m^2
        kgTg = 1e9          # Conversion factor from kg to Tg
        factors = biomass_plot['EXPNS'] * biomass_plot['ADJ_FACTOR_SUBP'] * biomass_plot['ADJ_FACTOR_MICR']

        biomass_plot['StraAG'] = (biomass_plot['PlotAG'] * factors * m2a) / kgTg
        biomass_plot['StraBG'] = (biomass_plot['PlotBG'] * factors * m2a) / kgTg
        biomass_plot['StraBM'] = (biomass_plot['PlotBM'] * factors * m2a) / kgTg

        # Merge with tree data to get county names
        tree_plots = tree_filtered[['PLOT', 'COUNTYNM']].drop_duplicates(subset=['PLOT'])
        biomass_plot = biomass_plot.merge(tree_plots, on='PLOT', how='left')

        # Append the results to the state_plot_biomass DataFrame
        state_plot_biomass = pd.concat([state_plot_biomass, biomass_plot], ignore_index=True)

    # Calculate total biomass across all years
    summary_biomass = {
        'Aboveground Biomass (TgC)': round(state_plot_biomass['StraAG'].sum(), 2),
        'Belowground Biomass (TgC)': round(state_plot_biomass['StraBG'].sum(), 2),
        'Total Biomass (TgC)': round(state_plot_biomass['StraBM'].sum(), 2)
    }

    return state_plot_biomass, summary_biomass

def shannon(tree_data, plot_data, inv_yr, n_years):
    # Create a list of inventory years (an inventory cycle)
    inv_yrs = [inv_yr - i for i in range(n_years)]
    
    # Filter the data for the inventory period
    tree_data = tree_data[tree_data['INVYR'].isin(inv_yrs)]
    
    # Define and retain necessary columns
    columns_keep = ['INVYR', 'STATECD', 'COUNTYCD', 'PLOT', 'SUBP', 'TREE', 
                    'CONDID', 'STATUSCD', 'SPCD', 'DIA', 'HT', 'TPA_UNADJ']
    tree_data = tree_data[columns_keep]
    
    # Filter and drop duplicate plots in plot_data
    plot_data = plot_data[plot_data['INVYR'].isin(inv_yrs)]
    plot_data = plot_data.drop_duplicates(subset=['PLOT', 'INVYR'])
    
    # Add location information for each tree record by merging with plot_data
    tree_data = pd.merge(tree_data, plot_data[['PLOT', 'INVYR', 'LAT', 'LON']], 
                         on=['PLOT', 'INVYR'], how='left')
    
    # Calculate the Shannon index for each plot and inventory year
    sindex = tree_data.groupby(['PLOT', 'INVYR']).apply(lambda df: 
        -np.sum((df['SPCD'].value_counts(normalize=True) * 
                 np.log(df['SPCD'].value_counts(normalize=True))))
    )
    
    # Reset index to turn the Series into a DataFrame
    sindex_df = sindex.reset_index()
    sindex_df.columns = ['PLOT', 'INVYR', 'shannon']
    
    # Get unique 'LAT' and 'LON' for each 'PLOT' and 'INVYR'
    plot_location = tree_data[['PLOT', 'INVYR', 'LAT', 'LON']].drop_duplicates(subset=['PLOT', 'INVYR'])
    
    # Merge the Shannon index with plot location
    sindex_df = pd.merge(sindex_df, plot_location, on=['PLOT', 'INVYR'], how='left')
    
    return sindex_df
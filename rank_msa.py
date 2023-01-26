"""
This script will rank the MSAs based on
your own criteria.
"""

# Import libraries

from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression

### Read in helper datasets

# Read in state FIPS codes
state_fips = pd.read_csv(
    "datasets/helper_datasets/state_FIPS_codes.csv",
    dtype={'state_code':str}
)

# Read in MSA state codes
msa_state_fips = pd.read_csv(
    "datasets/helper_datasets/msa_and_state_codes.csv",
    dtype={'FIPS State Code':str, 'CBSA Code':str}
)

# Get only necessary columns
msa_state_fips = msa_state_fips[['CBSA Code','CBSA Title','FIPS State Code']]

# Rename column
msa_state_fips.rename(columns={'FIPS State Code':'state_code'}, inplace=True)


### Define helper functions

def turn_df_into_datetime(dataframe):
    """
    Turns a dataframe created by the API functions
    into a tidy datetime format.
    """
    # Make a copy
    df = dataframe.copy()
    
    # Set index
    df = df.set_index(['msa_name','msa_code'])
    
    # Stack
    df = df.stack()
    
    # Turn into dataframe
    df = pd.DataFrame(df).reset_index()
    
    # Rename the post-stacked columns
    df.rename(columns={'level_2':'year', 0:'value'}, inplace=True)
    
    # Make year column integer
    df['year'] = df['year'].astype(int)
    
    # Make datetime column
    df['date'] = pd.to_datetime(df['year'], format='%Y')
    
    return df


def prep_census_datasets(dataframe, msa_state_fips=msa_state_fips):
    """
    This function preps the census datasets into
    the same format as the BLS job datasets
    for future city comparisons using both
    Census and BLS datasets.
    """
    # Make copy
    df = dataframe.copy()
    
    # Turn into tidy datetime
    df = turn_df_into_datetime(df)

    # Add MSA state code
    df = df.merge(msa_state_fips, 
                    how='left', 
                    left_on=['msa_name','msa_code'],
                    right_on=['CBSA Title','CBSA Code'])

    # Drop unnecessary columns
    df.drop(columns=['CBSA Title','CBSA Code'], inplace=True)

    # Replace NECTA Division
    df['msa_name'] = df['msa_name'].apply(lambda x: x.replace(" NECTA Division",""))
    df['msa_name'] = df['msa_name'].apply(lambda x: x.replace(" NECTA",""))
    
    return df


def normalize_column(
    series, mean_standardize=False, 
    min_max_standardized=False):
    """
    Normalizes a column's values.
    
    Arguments
    -----------
        series (Series): A pandas Series, which can
            simply be passed as a column of a
            DataFrame.
            
    Returns
    -----------
        series (Series): A normalized Series, which can
            be set to a column in a DataFrame.
    """
    # Make a copy
    sr = series.copy()
    
    # Standardize around the mean or by min-max
    if mean_standardize:
        # Make normalized column
        sr = (sr - sr.mean())/sr.std()
    elif min_max_standardized:
        # Make normalized column
        sr = (sr - sr.min())/(sr.max()-sr.min())
    else:
        raise ValueError("Please specify how to normalize.")
    
    return sr


def run_lr(df, column):
    """
    Run linear regression on time-series data 
    and return the coefficient and intercept.
    
    Arguments
    -----------
        df (DataFrame): A dataframe that contains the
            target column and an 'ordinal_date' column
            that was created by a time-series column in 
            the format of "%Y-%m-%d" and making it ordinal,
            such as running the code below in some other 
            step. 
            
            EXAMPLE...
            # Create ordinal column
            df['ordinal_date'] = df['date'].map(
                datetime.toordinal)
                
        column (str): The name of the target column.
            
    Returns
    -----------
        coef (float): The coefficient of the linear
            equation calculated.
        
        intercept (float): The y-intercept of the linear
            equation calculated.
    
    """
    # Run linear regression
    normal_lr = LinearRegression()
    X = df[['ordinal_date']]
    y = df[column]
    normal_lr.fit(X, y)
    coef = normal_lr.coef_[0]
    intercept = normal_lr.intercept_

    # Return lr coefficient
    return coef, intercept


# Loop through all cities, sort by coefficient, plot top 10
def plot_top_10_cities(
    ranked_cities,
    plot_jobs=False,
    plot_rent=False,
    plot_income=False,
    plot_price=False,
    plot_units=False,
    plot_rent_to_price=False,
    plot_jobs_per_unit=False,
    begin_year_1=2013,
    plot_all=False,
):
    """
    Plots the top cities for a given demographic. Top cities
    are chosen based on their trend. This function can also 
    find the top cities based on multiple datasets.
    
    Arguments
    ----------
        ranked_cities (DataFrame): A dataframe of cities
            already ranked by various demographics. The
            dataframe returned by the "make_ranking()"
            function is the ideal dataframe to pass
            to this function.
            
        plot_jobs (True/False): If Ture, plot jobs. Only one 
            demographic can be plotted at a time, so if you'd 
            like to plot a different demographic, this must 
            be set to False.
        
        plot_rent (True/False): If Ture, plot rent. Only one 
            demographic can be plotted at a time, so if you'd 
            like to plot a different demographic, this must 
            be set to False.
        
        plot_income (True/False): If Ture, plot income. Only one 
            demographic can be plotted at a time, so if you'd 
            like to plot a different demographic, this must 
            be set to False.
        
        plot_price (True/False): If Ture, plot price. Only one 
            demographic can be plotted at a time, so if you'd 
            like to plot a different demographic, this must 
            be set to False.
        
        plot_units (True/False): If Ture, plot units. Only one 
            demographic can be plotted at a time, so if you'd 
            like to plot a different demographic, this must 
            be set to False.
        
        plot_rent_to_price (True/False): If Ture, plot rent-ro-price
            ratio. Only one demographic can be plotted at a time, 
            so if you'd like to plot a different demographic, 
            this must be set to False.
        
        plot_jobs_per_unit (True/False): If Ture, plot jobs-per-unit. 
            Only one demographic can be plotted at a time, so if 
            you'd like to plot a different demographic, this must 
            be set to False.
            
        begin_year_1 (int): The year you'd like the
            analysis to start.
            
        plot_all (True/False): True if you want to plot every
            city in the dataframe (slow). False if you only want
            to plot the top 5 (fast).
    
    Returns
    ----------
        coef_df (DataFrame): A dataframe with the rankings
            of each city, from "best to worst."
    
    """
    # Make a copy of the ranked cities
    ranked = ranked_cities.copy()
    
    # If not plotting all cities (and just top 10),
    # keep only the top 10 cities in the dataframe
    if not plot_all:
        ranked = ranked.head(10)
    
    ### Call in the dataset we will be graphing from
    
    # If plotting job growth
    if plot_jobs:
        
        # Set demographic title for graphs
        demographic_1="Job"
        
        # Read in most recent job data
        dataframe_1 = pd.read_csv('datasets/bls/raw/most_recent_bls_data.csv',
                           dtype={'msa_code':str, 'state_code':str})

        # Make sure the date column is in datetime format
        dataframe_1['date'] = pd.to_datetime(jobs['date'])

        # Replace NECTA Division
        dataframe_1['msa_name'] = dataframe_1['msa_name'].apply(lambda x: x.replace(" NECTA Division",""))
        dataframe_1['msa_name'] = dataframe_1['msa_name'].apply(lambda x: x.replace(" NECTA",""))
    
    # If plotting rent growth
    elif plot_rent:
        
        # Set demographic title for graphs
        demographic_1="Median Rent"
        
        # Read in data
        dataframe_1 = pd.read_csv(
            "datasets/cleaned_census_api_files/msa_data/median_rent_msa.csv",
            dtype={'msa_code':str})

        # Run prep function to get into correct format
        dataframe_1 = prep_census_datasets(dataframe_1)
        
    # If plotting income growth
    elif plot_income:
        
        # Set demographic title for graphs
        demographic_1="Median Income"
        
        # Read in data
        dataframe_1 = pd.read_csv(
            "datasets/cleaned_census_api_files/msa_data/median_income_msa.csv",
            dtype={'msa_code':str})

        # Run prep function to get into correct format
        dataframe_1 = prep_census_datasets(dataframe_1)
        
    # If plotting price growth
    elif plot_price:
        
        # Set demographic title for graphs
        demographic_1="Median Price"
        
        # Read in data
        dataframe_1 = pd.read_csv(
            "datasets/cleaned_census_api_files/msa_data/median_price_msa.csv",
            dtype={'msa_code':str})

        # Run prep function to get into correct format
        dataframe_1 = prep_census_datasets(dataframe_1)
        
    # If plotting unit growth
    elif plot_units:
        
        # Set demographic title for graphs
        demographic_1="Total Units"
        
        # Read in data
        dataframe_1 = pd.read_csv(
            "datasets/cleaned_census_api_files/msa_data/total_units_msa.csv",
            dtype={'msa_code':str})

        # Run prep function to get into correct format
        dataframe_1 = prep_census_datasets(dataframe_1)
        
    # If plotting rent-to-price
    elif plot_rent_to_price:
        
        # Set demographic title for graphs
        demographic_1="Rent-to-Price"
        
        # Read in data
        dataframe_1 = pd.read_csv(
            "datasets/cleaned_census_api_files/msa_data/rent_to_price_ratio_msa.csv",
            dtype={'msa_code':str})

        # Run prep function to get into correct format
        dataframe_1 = prep_census_datasets(dataframe_1)
        
    # If plotting jobs-per-unit
    elif plot_jobs_per_unit:
        
        # Set demographic title for graphs
        demographic_1="Jobs per Unit"
        
        # Read in data
        dataframe_1 = pd.read_csv(
            "datasets/cleaned_census_api_files/msa_data/jobs_per_unit_msa.csv",
            dtype={'msa_code':str})

        # Run prep function to get into correct format
        dataframe_1 = prep_census_datasets(dataframe_1)
    
    # Make copy
    main_df = dataframe_1.copy()
            
    # Create main variable to use for the rest of the script
    column = 'value'

    # Create dictionary to store filtered dataframes
    filtered_dict = {}

    # Set y_lim list to find max and min
    y_lim_list_trend = []
    y_min_list_trend = [0]
    y_lim_list_pct = []
    y_min_list_pct = [0]
    
    # Loop through all cities in the ranked dataframe
    for city in ranked['msa_name'].dropna().unique():

        # Isolate just that city
        df = main_df[main_df['msa_name']==city].copy()
        
        # Sort by date
        df = df.sort_values('date')
                
        # Create difference column
        df['value_change'] = df['value'].diff()
        
        # Create pct_change column
        df['percent_change'] = df['value'].pct_change()
        
        # Filter by beginning year
        df = df[df['year']>=begin_year_1].reset_index(drop=True)
                
        # If an MSA's most recent year is after the beginning
        # year, remove it from the graphs. For example, if we want to
        # view the growth of all cities since 2016, but Prescott Valley
        # only has data starting at 2019, this may skew the data.
        if df['year'].iloc[0] != begin_year_1:
            print(f"Dropping {city}, it's dataframe has a smaller window.")
            continue
                
        # Remove NaN values
        df = df[df['percent_change'].notna()]
        
        # Isolate date and value columns
        df = df[[
            'date', 'value', 'value_change',
            'percent_change']].reset_index(drop=True)
        
        # Add this dataframe's y_lim to list
        y_lim_list_trend.append(df['value'].max())
        y_min_list_trend.append(df['value'].min())
        y_lim_list_pct.append(df['percent_change'].max())
        y_min_list_pct.append(df['percent_change'].min())
        
        # get next months's datetime
        next_year = df['date'].iloc[-1] + relativedelta(months=1)
        
        # Create ordinal column
        df['ordinal_date'] = df['date'].map(datetime.toordinal)
        
        # Run linear regression and get the trend's coefficient
        coef_value, intercept_value = run_lr(df, column='value')
        coef_pct, intercept_pct = run_lr(df, column='percent_change')
        
        # Create next year's date
        df.loc[len(df.index)] = [
            next_year, np.nan, np.nan, 
            np.nan, datetime.toordinal(next_year)]
        
        # Create averages column
        the_average_pct = df['percent_change'].mean()
        df['average_pct'] = the_average_pct
        the_average_value = df['value_change'].mean()
        df['average_value'] = the_average_value

        # Fill in with linear regression values.
        # Also add highest trend value to lim_list.
        df['value_trend'] = df['ordinal_date']*coef_value + intercept_value
        df['percent_change_trend'] = df['ordinal_date']*coef_pct + intercept_pct

        # Also add highest trend value to lim_list
        y_lim_list_trend.append(df['value_trend'].max())
        y_lim_list_pct.append(df['percent_change_trend'].max())

        # Get the y_lim
        y_lim_trend = max(y_lim_list_trend) * 1.1
        y_min_trend = min(y_min_list_trend)
        y_lim_pct = max(y_lim_list_pct) * 1.1
        y_min_pct = min(y_min_list_pct)
            
        # Save filtered data to dictionary
        filtered_dict[city] = df
    
    # Loop through each city in the ranked df
    for city_name in ranked['msa_name']:

        # Get the job data
        df = filtered_dict[city_name]    
            
        # Make a grid to plot 2 graphs on
        fig = plt.figure(figsize=(12,3), dpi=300)
        gs = GridSpec(nrows=1, ncols=2)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax_list = [ax1, ax2]

        # Set title
        fig.suptitle(f"{city_name}\n\n\n", 
             fontweight="bold")

        # Plot first graph
        ax1 = df.plot(x='date',y='value', ax=ax1)
        ax1 = df.plot(x='date',y='value_trend', ax=ax1, linestyle="--")

        # Plot second graph
        ax2 = df.plot(x='date',y='percent_change', ax=ax2)
        ax2 = df.plot(x='date',y='percent_change_trend', ax=ax2, linestyle="--")

        # Second graph's zero line
        df['zero'] = 0
        ax2 = df.plot(x='date', y='zero', ax=ax2, color="grey")

        # Also plot the average line
        ax2 = df.plot(x='date', y='average_pct', 
                      ax=ax2, color="black", linestyle="-")

        # Set title's for both graphs
        ax1.set_title(f"{demographic_1} Growth")
        ax2.set_title(f"Percent Change in {demographic_1}")

        # Set y lims and y ticks
        ax1.set_ylim([y_min_trend, y_lim_trend])
        ax2.set_ylim([y_min_pct, y_lim_pct])

        # Set y limits
        y_tick_list_trend = [
            y_lim_trend*0.25, y_lim_trend*0.5, 
            y_lim_trend*0.75, y_lim_trend]
        y_tick_list_pct = [
            y_min_pct, y_min_pct*0.5, 0, 
            y_lim_pct*0.5, y_lim_pct]

        # Set y_ticks
        ax1.yaxis.set_major_locator(
            mticker.FixedLocator(y_tick_list_trend))
        ax2.yaxis.set_major_locator(
            mticker.FixedLocator(y_tick_list_pct))

        # Set y-tick labels
        ax1.set_yticklabels(
            ['{:,}'.format(round(float(x), 3)) for x in y_tick_list_trend])
        ax2.set_yticklabels(
            ['{:,}'.format(round(float(x), 3)) for x in y_tick_list_pct])

        # Show plot
        plt.show()


### Read in and clean Jobs data

# Read in most recent job data
jobs = pd.read_csv('datasets/bls/raw/most_recent_bls_data.csv',
                   dtype={'msa_code':str, 'state_code':str})

# Make sure the date column is in datetime format
jobs['date'] = pd.to_datetime(jobs['date'])

# Replace NECTA Division
jobs['msa_name'] = jobs['msa_name'].apply(lambda x: x.replace(" NECTA Division",""))
jobs['msa_name'] = jobs['msa_name'].apply(lambda x: x.replace(" NECTA",""))

### Read in and clean Median Rent data

# Read in median rent
median_rent = pd.read_csv(
    "datasets/cleaned_census_api_files/msa_data/median_rent_msa.csv",
    dtype={'msa_code':str}
)

# Run prep function to get into correct format
median_rent = prep_census_datasets(median_rent)

### Read in and clean Median Price data

# Read in median price
median_price = pd.read_csv(
    "datasets/cleaned_census_api_files/msa_data/median_price_msa.csv",
    dtype={'msa_code':str}
)

# Run prep function to get into correct format
median_price = prep_census_datasets(median_price)

### Read in and clean Median Income data

# Read in median income
median_income = pd.read_csv(
    "datasets/cleaned_census_api_files/msa_data/median_income_msa.csv",
    dtype={'msa_code':str}
)

# Run prep function to get into correct format
median_income = prep_census_datasets(median_income)

### Read in and clean Rent-to-Price data

# Read in rent_to_price
rent_to_price = pd.read_csv(
    "datasets/cleaned_census_api_files/msa_data/rent_to_price_ratio_msa.csv",
    dtype={'msa_code':str}
)

# Run prep function to get into correct format
rent_to_price = prep_census_datasets(rent_to_price)


### MAKE FUNCTION THAT MAKES A TOTAL RANK 
### BASED ON MULTIPLE DEMOGRAPHICS
def make_ranking(
    df_dict,
    max_price=False,
    min_rent_to_price=False,
    use_total_trend=True,
    use_average_percent=True,
    total_trend_weight_dict={},
    average_percent_weight_dict={},
    plot_graphs=False
):
    """
    This function ranks the invest-ability of every
    city based on the demographics passed. It
    analyzes the total average growth per year, as 
    well as the relative average growth per year
    (measured as the average percent growth per year).
    
    Arguments
    -----------
        df_dict (dict): A dictionary to be used if you
            want to combine multiple dataframes for analysis
            and plotting. If using this, the key should be
            a string with the demographic name, and the value
            should be a list containing the dataframe in position
            0, and the beginning year in position 1. See below
            for two examples...
            
            Example 1, One Extra Dataframe
            {"Median Rent": [median_rent_df, 2013]}
            
            Example 2, Multiple Extra Dataframes
            {"Median Rent": [median_rent_df, 2013],
            "Population" : [population_df, 2013]}
            
        max_price (int): If you only want to measure and
            compare MSAs up to a certain median price,
            enter the max median price as an integer.
            
        min_rent_to_price (float): If you only want to measure and
            compare MSAs up to a certain rent-to-price ratio
            (based on median rent and median price values),
            enter the minimum rent-to-price ratio as a float.
            
        use_total_trend (True/False): Set to True if you'd like
            to include the total trend weights in the ranking
            of MSAs. Use False if not.
        
        use_average_percent (True/False): Set to True if you'd like
            to include the average percent weights in the ranking
            of MSAs. Use False if not.
        
        total_trend_weight_dict (dict): A dictionary to set the
            weights of each demo. For example, if you'd like to
            multiply the "Median Rent" weights by 2, giving a bigger
            weight to the "Median Rent" demographic, all you need
            is to make the key "Median Rent" set to a value of 2.
            This dictionary is specifically for total trend weights.
            See the example below.
            
            EXAMPLE...
            total_trend_weight_dict={
                "Jobs":1,
                "Median Rent":1}
        
        average_percent_weight_dict (dict): A dictionary to set the
            weights of each demo. For example, if you'd like to
            multiply the "Median Rent" weights by 2, giving a bigger
            weight to the "Median Rent" demographic, all you need
            is to make the key "Median Rent" set to a value of 2.
            This dictionary is specifically for average percent weights.
            See the example below.
            
            EXAMPLE...
            average_percent_weight_dict={
                "Jobs":3,
                "Median Rent":3}
                
        plot_graphs (True/False): If True, ask for user inputs
            and run the plot_top_10_cities() function.
            
    Returns
    -----------
        final_df (DataFrame): A dataframe with each city
            sorted by total rank.
    """
    # Make a list to add each dataframe to
    df_list = []
    
    # Rename all columns by appending the demo name,
    # except for the MSA name and date, which we will 
    # use as the key to merge all dataframes.
    for demo in df_dict:
        
        # Get dataframe
        df = df_dict[demo][0].copy()
        
        # Rename every column except for msa_name
        for col in df.columns:
            if (col != 'msa_name') & (col != 'date'):
                df.rename(
                    columns={col:f'{col}_{demo}'}, 
                    inplace=True)
                
        # Add dataframe to list
        df_list.append(df)
                
    # Merge all dataframes
    merged_df = reduce(lambda left, right: 
                       pd.merge(left, right, 
                                left_on=['msa_name','date'], 
                                right_on=['msa_name','date'],
                                suffixes=(None, "_y"),
                                how="outer"), df_list)
    
    # Only keep necessary columns
    for demo in df_dict:
        merged_df.drop(columns=[
            f"msa_code_{demo}",
            f"state_code_{demo}",
        ], inplace=True)
        
    # Loop through columns and clean out the rest
    for col in merged_df.columns:
        if ('month_' in col) | ('series_id_' in col):
            merged_df.drop(columns=[col], inplace=True)
    
    # Create new df to store coefficients
    coef_df = pd.DataFrame(
        data=None, columns=['msa_name'])
    
    # Add columns for every demo
    for demo in df_dict:
        coef_df[f'trend_coef_{demo}'] = None
        coef_df[f'average_value_{demo}'] = None
        coef_df[f'average_pct_{demo}'] = None
        
    # Make temporary coef_df to use later
    temp_coef_1 = coef_df.copy()
        
    # Create ordinal column
    merged_df['ordinal_date'] = merged_df['date'].map(datetime.toordinal)
    
    # Loop through all cities
    for city in merged_df['msa_name'].dropna().unique():

        # Isolate just that city
        df = merged_df[merged_df['msa_name']==city].copy()
        
        # Sort by date
        df = df.sort_values('date')
        
        # Make duplicate
        temp_coef_2 = temp_coef_1.copy()
        
        # Set temp coef_df
        temp_coef_2.loc[len(temp_coef_2.index)] = np.nan
        temp_coef_2['msa_name'] = city
        
        # Loop through each demo
        for demo in df_dict:
            
            # Test to see if there's data for the demo
            if df[df[f'year_{demo}'].notna()].shape[0] > 0:
                
                # Make copy
                df_temp = df.copy()
                
                # Get beginning year
                begin_year = df_dict[demo][1]
                
                # Filter by beginning year minus 1
                df_temp = df[df[f'year_{demo}']>=begin_year-1].reset_index(drop=True)
                        
                # Create difference column
                df_temp[f'value_change_{demo}'] = df_temp[f'value_{demo}'].diff()

                # Create pct_change column
                df_temp[f'percent_change_{demo}'] = df_temp[f'value_{demo}'].pct_change()

                # Filter by beginning year
                df_temp = df_temp[
                    df_temp[f'year_{demo}']>=begin_year].reset_index(drop=True)

                # If an MSA's most recent year is after the beginning
                # year, remove it from the graphs. For example, if we want to
                # view the growth of all cities since 2016, but Prescott Valley
                # only has data starting at 2019, this may skew the data.
                if df_temp[f'year_{demo}'].iloc[0] != begin_year:
                    continue

                # Remove NaN values
                df_temp = df_temp[df_temp[f'percent_change_{demo}'].notna()]

                # Run linear regression
                coef_value, intercept_value = run_lr(df_temp, column=f'value_{demo}')
                coef_pct, intercept_pct = run_lr(df_temp, column=f'percent_change_{demo}')

                # Create trend columns
                df_temp[f'value_trend_{demo}'] = df_temp['ordinal_date']*coef_value + intercept_value
                df_temp[f'percent_change_trend_{demo}'] = df_temp['ordinal_date']*coef_pct + intercept_pct

                # Create averages column
                the_average_pct = df_temp[f'percent_change_{demo}'].mean()
                df_temp[f'average_pct_{demo}'] = the_average_pct
                the_average_value = df_temp[f'value_change_{demo}'].mean()
                df_temp[f'average_value_{demo}'] = the_average_value
                
                # Update temp coef
                temp_coef_2[f'trend_coef_{demo}'] = coef_value
                temp_coef_2[f'average_value_{demo}'] = the_average_value
                temp_coef_2[f'average_pct_{demo}'] = the_average_pct
            
        # Append temp coef to dataframe
        coef_df = pd.concat([coef_df, temp_coef_2])
            
    # Drop duplicates
    coef_df = coef_df.drop_duplicates().reset_index(drop=True)
    
    # Drop MSAs that have missing values (they will have missing
    # values if we couldn't join Census MSAs with BLS MSAs which
    # only occurs for a few specific MSAs)
    bad_msa = set()
    
    for demo in df_dict:
        
        # Filter by nulls
        coef_temp = coef_df[coef_df[f'trend_coef_{demo}'].isnull()]
        
        # Get list of MSAs
        bad_msa.update(coef_temp['msa_name'].unique())
        
    # Remove these cities Print helpful message
    if len(bad_msa) > 0:
        
        # Remove cities in bad_msa
        coef_df = coef_df[~coef_df['msa_name'].isin(bad_msa)].reset_index(drop=True)
        
    # Create the rankings for each demographic
    for demo in df_dict:
        
        # Calculate rankings for both, then sort by the total
        # ranking. For example, if a city has the highest average
        # percent change, it will get a ranking of "1" for average_pct,
        # and if it has the 8th highest trend coefficient, it will
        # get a ranking of "8" for trend_coef. When we add those two
        # rankings together, the city will have a total ranking
        # of "9". In this case, the lower the ranking, the better,
        # and we will sort total rankings from lowest to highest.
        
        # Normalize total trend column
        coef_df[f'normalized_trend_coef_{demo}'] = normalize_column(
            coef_df[f'trend_coef_{demo}'], min_max_standardized=True)
        
        # Normalize avg pct column
        coef_df[f'normalized_average_pct_{demo}'] = normalize_column(
            coef_df[f'average_pct_{demo}'], min_max_standardized=True)
        
        # Check to see if there are weights, and if not,
        # set each weight to 1
        if demo in total_trend_weight_dict.keys():
            trend_weight = total_trend_weight_dict[demo]
        else:
            trend_weight = 1
            
        # Check pct weight dict
        if demo in average_percent_weight_dict.keys():
            pct_weight = average_percent_weight_dict[demo]
        else:
            pct_weight = 1
            
        # Re-adjust weights based on whether we are using
        # only total trend, only percent, or both. As an example, 
        # if we aren't using percent, we set the weight to 0, that
        # way the percent weight isn't used when totalling the
        # demographic's weight.
        if use_total_trend == False:
            trend_weight = 0
        if use_average_percent == False:
            pct_weight = 0
        
        # Create weights
        coef_df[f'{demo}_weight'] = (
            (coef_df[f'normalized_trend_coef_{demo}'] * trend_weight) 
            + (coef_df[f'normalized_average_pct_{demo}'] * pct_weight)
        )

    # Make final total rank column by adding up
    # all demo total rankings
    coef_df['total_weight'] = 0
    for demo in df_dict:
        coef_df['total_weight'] += coef_df[f'{demo}_weight']

    # Sort by total weight, highest to lowest
    final_df = coef_df.sort_values(
        'total_weight', ascending=False).reset_index(drop=True)
    
    # Merge median rent, price, and income to final df
    for demo in ["median_rent","median_price",
                 "median_income","rent_to_price_ratio",
                 "total_units"]:
        
        # Call in demographic dataset
        demo_df = pd.read_csv(
            f"datasets/cleaned_census_api_files/msa_data/{demo}_msa.csv",
            dtype={'msa_code':str}
        )

        # Run prep function to get into correct format
        demo_df = prep_census_datasets(demo_df)

        # Get most recent year for median price
        recent_year = demo_df['year'].max()

        # Filter by recent_year
        recent_year_df = demo_df[demo_df['year']==recent_year].copy()        
        
        # Only keep certain columns
        recent_year_df = recent_year_df[['msa_name','value']]
        
        # Rename column
        recent_year_df.rename(columns={'value':f'{demo}'}, inplace=True)
        
        # Merge to final_df
        final_df = final_df.merge(recent_year_df, how='left', on='msa_name')
        
    # Merge jobs and create jobs-per-unit
    jobs = pd.read_csv('datasets/bls/raw/most_recent_bls_data.csv',
                   dtype={'msa_code':str, 'state_code':str})

    # Make sure the date column is in datetime format
    jobs['date'] = pd.to_datetime(jobs['date'])

    # Replace NECTA Division
    jobs['msa_name'] = jobs['msa_name'].apply(lambda x: x.replace(" NECTA Division",""))
    jobs['msa_name'] = jobs['msa_name'].apply(lambda x: x.replace(" NECTA",""))

    # Get most recent job date
    recent_date = jobs['date'].max()

    # Filter jobs so it's the most recent date
    new_jobs = jobs[jobs['date']==recent_date].copy().reset_index(drop=True)

    # Only keep certain columns
    new_jobs = new_jobs[['msa_name','value']]
    
    # Rename column
    new_jobs.rename(columns={'value':f'jobs'}, inplace=True)

    # Merge to final_df
    final_df = final_df.merge(new_jobs, how='left', on='msa_name')
    
    # Create jobs-per-units column
    final_df['jobs_per_unit'] = final_df['jobs']/final_df['total_units']
    
    # If max price, filter it
    if max_price:
        final_df = final_df[
            final_df['median_price']<=max_price].reset_index(drop=True)
        
    # If min rent-price ratio, filter
    if min_rent_to_price:
        final_df = final_df[
            final_df['rent_to_price_ratio']>=min_rent_to_price].reset_index(drop=True)
    
    # If plot_grpahs = True, ask for user input to then
    # pass as arguments to the plot_top_10_cities() function
    if plot_graphs:
        
        # Define argument dictionary
        plot_arg_dict = {}
        
        # 'ranked_cities' is the final_df
        plot_arg_dict['ranked_cities'] = final_df
        
        # Define loop to ensure the right arguments
        # are passed
        num_input = 0
        while num_input not in [i for i in range(1,8)]:
        
            # Ask user what demographic they'd like
            print("""Please enter a number (1-7) according to which
                    demographic you'd like to graph:
                    1) Jobs
                    2) Median Rent
                    3) Median Income
                    4) Median Price
                    5) Total Units
                    6) Rent-to-Price Ratio
                    7) Jobs-per-Unit Ratio
            """)
            num_input = int(input())

            # Fill dictionary based on input
            if num_input == 1:
                plot_arg_dict['plot_jobs'] = True
                print("Plotting Job Growth.")
            elif num_input == 2:
                plot_arg_dict['plot_rent'] = True
                print("Plotting Median Rent Growth.")
            elif num_input == 3:
                plot_arg_dict['plot_income'] = True
                print("Plotting Median Income Growth.")
            elif num_input == 4:
                plot_arg_dict['plot_price'] = True
                print("Plotting Median Price Growth.")
            elif num_input == 5:
                plot_arg_dict['plot_units'] = True
                print("Plotting Total Unit Growth.")
            elif num_input == 6:
                plot_arg_dict['plot_rent_to_price'] = True
                print("Plotting Rent to Price Growth.")
            elif num_input == 7:
                plot_arg_dict['plot_jobs_per_unit'] = True
                print("Plotting Jobs per Unit Growth.")
            else:
                print("Invalid entry, please try again.")
                
        # Ask for the beginning year
        print("""Input the year to begin the graph. If plotting jobs,
                it's recommended to input the most recent year, as the
                job data is organized monthly. Otherwise, 2013 is always
                a good starting date, as some MSAs don't have data before
                then.
        """)
        plot_arg_dict['begin_year_1'] = int(input())
        
        # Ask if plotting all graphs or only top 10
        print("""Input 'all' to graph every MSA, otherwise enter anything
                else to only graph the top 10 MSAs.
        """)
        all_or_10 = input()
        if all_or_10 == 'all':
            plot_arg_dict['plot_all'] = True
            
        # Run the plotting function
        plot_top_10_cities(**plot_arg_dict)
            
    return final_df




def ask_ranking_function_args(
    jobs=jobs,
    median_rent=median_rent,
    median_price=median_price,
    median_income=median_income,
):
    """
    This function asks for user input
    and turns them into inputs for the
    ranking function.
    """
    # Define main argument dictionary
    function_args = {}
    
    # Ask for which dataframes to make the
    # ranking based off of
    
    # Create the df_dict to pass as an argument
    df_dict = {}
    
    # Ask for Jobs dataframe
    use_jobs = False
    print("""Want to include the Jobs dataframe in the ranking?""")
    y_input = input("Type 'y' if yes, or anything else if not: ")
    if y_input == 'y':
        use_jobs = True
        jobs_begin_year = int(input("What's the year you want to start analysis on? For jobs, its recommended to start with the most recent year: "))
        df_dict['Jobs'] = [jobs, jobs_begin_year]
    
    # Ask for Rent dataframe
    use_rent = False
    print("""Want to include the Median Rent dataframe in the ranking?""")
    y_input = input("Type 'y' if yes, or anything else if not: ")
    if y_input == 'y':
        use_rent = True
        rent_begin_year = int(input("What's the year you want to start analysis on? For rent, its recommended to start with the year 2013: "))
        df_dict['Median Rent'] = [median_rent, rent_begin_year]
        
    # Ask for Price dataframe
    use_price = False
    print("""Want to include the Median Price dataframe in the ranking?""")
    y_input = input("Type 'y' if yes, or anything else if not: ")
    if y_input == 'y':
        use_price = True
        price_begin_year = int(input("What's the year you want to start analysis on? For price, its recommended to start with the year 2013: "))
        df_dict['Median Price'] = [median_price, price_begin_year]
        
    # Ask for Income dataframe
    use_income = False
    print("""Want to include the Median Income dataframe in the ranking?""")
    y_input = input("Type 'y' if yes, or anything else if not: ")
    if y_input == 'y':
        use_income = True
        income_begin_year = int(input("What's the year you want to start analysis on? For income, its recommended to start with the year 2013: "))
        df_dict['Median Income'] = [median_income, income_begin_year]
            
    # Ask about max price filtering
    max_price_number = False
    use_max_price = input("""Type 'y' to filter ranking by a maximum median price.
    Type anything else to not filter by a maximum purchase price.""")
    if use_max_price == 'y':
        max_price_number = int(input("Enter the maximum median price (in integer form, no commas): "))
        
    # Ask about rent-to-price filtering
    min_rent_price = False
    use_rent_price = input("""Type 'y' to filter ranking by a minimum rent-to-price ratio.
    Type anything else to not filter by a rent-to-price ratio.""")
    if use_rent_price == 'y':
        min_rent_price = float(input("Enter the minimum rent-to-price ratio (in decimal form, not as percentage, such as '0.002' for a 0.2% rent-to-price ratio): "))
        
    # Ask about using total trend
    use_total_trend = False
    total_trend_ask = input("""Type 'y' to include the total growth trend into the ranking weights.
    Type anything else to not use it.""")
    if total_trend_ask == 'y':
        use_total_trend = True
        
    # Ask about using average percent
    use_avg_percent = False
    percent_ask = input("""Type 'y' to include the average percent growth into the ranking weights.
    Type anything else to not use it.""")
    if percent_ask == 'y':
        use_avg_percent = True
        
    # Define the total trend weights dictionary.
    # If using total trend, add weights
    total_trend_weight_dict = {}
    if use_total_trend:
        
        # Loop through the keys in df_dict
        for demo in df_dict:
            print(f"What weight will you assign to {demo}'s total trend?")
            temp_weight = int(input("For total trend, the recommended weight for everything is 1: "))
            total_trend_weight_dict[demo] = temp_weight
            
    # Define the avg percent weights dictionary.
    # If using avg percent, add weights
    average_percent_weight_dict = {}
    if use_avg_percent:
        
        # Loop through the keys in df_dict
        for demo in df_dict:
            print(f"What weight will you assign to {demo}'s average percent growth?")
            temp_weight = int(input("Input your weight multiplier as an integer: "))
            average_percent_weight_dict[demo] = temp_weight
            
    # Ask if they want to plot immediately after
    plot_graphs = False
    ask_plot = input("Do you wish to plot the rankings immediately after creation? Type 'y' if yes, or anything else if no: ")
    if ask_plot == 'y':
        plot_graphs = True
        
    # Finalize the function_args dictionary
    function_args = {
    'df_dict': df_dict,
    'max_price':max_price_number,
    'min_rent_to_price':min_rent_price,
    'use_total_trend':use_total_trend,
    'use_average_percent':use_avg_percent,
    'total_trend_weight_dict':total_trend_weight_dict,
    'average_percent_weight_dict':average_percent_weight_dict,
    'plot_graphs':plot_graphs}
    
    return function_args

if __name__ == "__main__":

    # Get the arguments to use in the ranking function
    function_args = ask_ranking_function_args()

    # Run the ranking function
    ranked_df = make_ranking(**function_args)
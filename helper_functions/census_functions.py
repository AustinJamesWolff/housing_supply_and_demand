"""
Helper functions to grab and organize data from
the U.S. Census Bureau and other related datasets.
"""
# Import libraries

import sys
import re
import requests
import asyncio
import nest_asyncio
import aiohttp
import time
import warnings

# Suppress unnecessary Shapely warning
warnings.filterwarnings('ignore',
                        '.*Shapely GEOS version.*')

from aiohttp import ClientSession
from requests import request, Session
from itertools import product, repeat
from dotenv import load_dotenv
import os
from os import getenv
from threading import Thread
import time
import inspect
import pandas as pd
import geopandas as gp
import shapely
import pygeos
from functools import reduce
from pandas.plotting import lag_plot
import pickle
import numpy as np
import seaborn as sns
import datetime as dt
import copy
import math
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
import plotly.express as px


# Set up Pandas defaults
pd.options.display.float_format = '{:.2f}'.format
pd.set_option("display.max_columns", None)

# Create the fifty states variable
first_ten_states = ['01','02','03','04','05','06','07','08','09']
forty_states = [str(i) for i in range(10, 52)]
fifty_states = first_ten_states + forty_states + ['53','54','55','56','72']
fifty_states_list = []
fifty_states_list.append(','.join(fifty_states[:26]))
fifty_states_list.append(','.join(fifty_states[26:]))

# Define helper function to create directory
def create_folder(the_path):
    if not os.path.isdir(the_path):
        os.mkdir(the_path)


# Call in block_groups_1020
block_groups_1020 = pd.read_csv(
    'datasets/cleaned_census_api_files/block_group_relationships/bg_2010_to_2020.csv',
    encoding='utf-8',
    dtype={'BG10':str, 'BG20':str, 'TRACT20':str, 'TRACT10':str})


def get_block_group_data(year, sequence_num='0003', 
                         start_position=130, data_name='population'):
    """
    Get and save the dataset per block group for a
    given year (only for years 2010-2012).
    
    Instructions:
    ---------------
    In order to get population or total units or any other 
    dataset you want in the future on a per block group, 
    you'll need to first open the 
    'datasets/ipums_data/Sequence_Number_and_Table_Number_Lookup.csv' 
    dataset. Then you'll need to search for the name of your 
    dataset (using CMD+F or CTRL+F). Beware, there are cases 
    where there may be multiple tables with the name you've 
    searched for, please try and be specific and look through 
    your options carefully. The table name will be IN ALL CAPS. 
    As an example, if you search for "Total Population" you'll 
    get 48 results, but only one of those results is the table 
    we want (it's the only one in ALL CAPS: "TOTAL POPULATION"). 
    Once you've found the row, look at the values in the columns 
    "Start Position" and "Sequence Number." The Sequence Number 
    is what you'll need to find the appropriate .txt in the 
    Block_201X folder. For example, TOTAL POPULATION has a 
    Sequence Number of 3, so the correct file for the state of 
    Alaska for the 2012 5-Year ACS will have a name of 
    "e20125ak0003000.txt" where the Sequence Number is typically 
    followed by "000.txt". Also take note of the "Start Position". 
    This will tell you which column in the .txt dataset has the 
    number you're looking for. For example, the 
    "e20125ak0003000.txt" has 130 columns. The first 6 columns 
    will always be needed and are unique identifiers. But after 
    that, the Start Position tells us that column 130 contains 
    the "TOTAL POPULATION" value.
    
    Parameters:
            year (str): The year to get.
            sequence_num (str): The sequence number that tells 
                us which table to grab. Try to pass it in the
                following format -- "000X" or "00XX" or "0XXX"
                where "X" is the sequence number.
            start_position (int): The "Start Position" of the
                data we want as specified by the 
                Sequence_Number_and_Table_Number_Lookup.csv file.
            data_name (str): The name of the dataset we are getting.
           
    Returns: 
            dataframe (DataFrame): A dataframe of all values for
                all block groups for all states in a given year.
                
    This function also automatically saves the final dataframe
    to a csv.
    
    """
    year = str(year)
    
    # Check for correct sequence number
    if type(sequence_num) != str:
        sequence_num = str(sequence_num)
    elif len(sequence_num) != 4:
        num_zeros = 4 - len(sequence_num)
        sequence_num = ("0" * num_zeros) + sequence_num
    
    # Subtract 1 for zero indexing
    start_position = int(start_position) - 1
    
    # Get filepaths to iterate through
    block_filepath = f'datasets/ipums_data/Block_{year}'

    # Call in Geography Column Names
    summary_filepath = f'datasets/ipums_data/{year}_5yr_Summary_FileTemplates'
    geonames_filepath = os.path.join(summary_filepath, f'{year}_SFGeoFileTemplate.xls')
    geonames = pd.read_excel(geonames_filepath)
    geoname_columns = geonames.columns.tolist()

    # Iterate through Block folder and add
    # population files to a list
    pop_files = []
    sequence_regex = r'e' + re.escape(str(year)) + r'5[a-z]{2}' + re.escape(sequence_num) + r'000\.txt'

    # Encode filepath
    directory = os.fsencode(block_filepath)

    # Iterate through directory
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if re.match(sequence_regex, filename):
            pop_files.append(filename)

    # Loop through pop_files and save final results
    joined_dataframes = []
    geo_filepath = f'datasets/ipums_data/{year}_ACS_Geography_Files'
    for file in pop_files:
        state = file[6:8]
        pop_filepath = os.path.join(block_filepath, file)
        pop = pd.read_csv(pop_filepath,
                            header=None,
                            usecols=[0, 1, 2, 3, 4, 5, start_position],
                            names=['fileid','filetype','state','character',
                                  'sequence','record',f'{data_name}'],
                            dtype={'fileid':str, 'filetype':str, 'state':str,
                                  'character':str, 'sequence':str, 'record':str,
                                  f'{data_name}':float}
                            )

        # Call in that state's geography
        state_geo_filepath = os.path.join(geo_filepath, f'g{year}5{state}.csv')
        geo = pd.read_csv(state_geo_filepath,
                            header=None,
                            names=geoname_columns,
                             dtype=str,
                          encoding='latin-1'
                            )

        # Keep only rows where Name contains "Block Group"
        blocks = geo[geo['NAME'].str.contains('Block Group')].copy()
        blocks.dropna(how='all', axis=1, inplace=True)
        blocks.reset_index(inplace=True, drop=True)
        blocks['TRACT'] = blocks['GEOID'].apply(lambda x: str(x)[-12:-1])

        # Join population based on blocks
        join = pd.merge(blocks, pop, how='inner', 
                           left_on='LOGRECNO', right_on='record')
        join['BLOCK'] = join['GEOID'].apply(lambda x: str(x)[-12:])
        join = join[['TRACT','BLOCK','GEOID','NAME','state','sequence','record',
                          f'{data_name}']]

        # Save dataframe into a list to be concatenated
        joined_dataframes.append(join)

    # Concat all joined dataframes
    all_states = pd.concat(joined_dataframes)
    all_states = all_states.reset_index(drop=True)
    
    # Create folder to save data
    try:
        os.mkdir(f"datasets/ipums_data/block_{data_name}/")
        print(f"Created datasets/ipums_data/block_{data_name}/ folder!")
    except FileExistsError:
        print(f"/block_{data_name}/ folder already exists, do nothing.")
    
    # Save data
    all_states.to_csv(f'datasets/ipums_data/block_{data_name}/block_group_{data_name}_{year}.csv',
                          encoding='utf-8',
                          index=False)
    
    print(f"Successfully saved block_group_{data_name}_{year}.csv!")
    
    return all_states


def get_statistics(dataframe, begin_year, end_year):
    """
    Get ratio for 2020 value to 2019 value, and get the
    z_score for the new ratio column.

    Parameters:
            dataframe (DataFrame): A standardized dataframe.
           
    Returns: 
            df (DataFrame): A dataframe with two new columns,
                one for the ratio between 2020 and 2019 values,
                and one for the z_scores of the ratio column. 
    """
    df = dataframe.copy()
    
    # If 2020 is 0, make all other years 0
    years = [str(i) for i in range(begin_year, end_year+1)]
    for year in years:
        df[year] = np.where(df['2020'] == 0,
                             0,
                             df[year])

    # Calculate 2020 to 2019 ratio
    df['ratio'] = np.where(df['2020'] != 0, 
                            (df['2020'] - df['2019'])/(df['2020']),
                            0)

    # Look for outliers by calculating z_score
    df['z_score'] = (df['ratio'] - df['ratio'].mean())/df['ratio'].std()

    return df


def plot_zscores(series, data_name):
    """Plot distribution of z_scores"""
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    ax = series.plot(kind='kde', stacked=False)
    ax.set_xlim([-4, 4])
    ax.set_xlabel('z_score')
    ax.set_ylabel('Distribution')
    plt.title(f'Distribution of {data_name} z-scores')
    plt.show()
    
    return None


def get_census_6(string):
    """
    Get census name in the same format as it
    appears in the software.
    """
    str_list = list(string)
    str_list.insert(-2, '.')
    new_string = ''.join(str_list)

    return new_string


load_dotenv()
census_key = os.environ["CENSUS_KEY"]
async def url_to_dataframe_async_owners(begin_year,
                            end_year,
                            fifty_states_list,
                            census_code_dict,
                            df_list,
                            census_code_meaning,
                            get_blocks=False,
                            census_key=census_key,
                            census_table='profile',
                            multi_code=False,
                            code_name_dict=None
                          ):
    """
    Convert API url request to dataframe
    """
    session = Session()

    for year in range(begin_year, end_year + 1):
        
        # Create temporary list to help combine dataframes with
        # the same year together
        temp_list = []

        if multi_code == True:
            renamed_code_columns = {}
        
        for states in fifty_states_list:
            print("Working on year:", year,"and states:\n", states)
            
            if get_blocks == False:
                # URL to connect to the API
                url = f"""https://api.census.gov/data/{year}/acs/acs5/{census_table}?get=NAME,GEO_ID,{census_code_dict[year]}&for=tract:*&in=state:{states}&in=county:*&key={census_key}"""
            elif (get_blocks == True) & (multi_code == False):
                url = f"""https://api.census.gov/data/{year}/acs/acs5?get=NAME,GEO_ID,{census_code_dict[year]}&for=block%20group:*&in=state:{states}&in=county:*&in=tract:*&key={census_key}"""
            elif (get_blocks == True) & (multi_code == True):
                code_string = ""
                for i in range(len(census_code_dict[year])):
                    if census_code_dict[year][i] != census_code_dict[year][-1]:
                        code_string  = code_string + census_code_dict[year][i] + ","
                    elif census_code_dict[year][i] == census_code_dict[year][-1]:
                        code_string  = code_string + census_code_dict[year][i]
                url = f"""https://api.census.gov/data/{year}/acs/acs5?get=NAME,GEO_ID,{code_string}&for=block%20group:*&in=state:{states}&in=county:*&in=tract:*&key={census_key}"""

            # Double-check the URL is correct by printing it
            print("URL:", url)
            
            # Simple function we can recursively recall if
            # the connection times out. However, this may
            # cause an infinite loop if the connection times
            # out due to any external reason, such as the server
            # being down. Please keep an eye as this function runs.
            def get_response(url):
                try:
                    # GET request the API
                    response = session.get(url, timeout=10)
                except (requests.exceptions.ReadTimeout, 
                        requests.exceptions.ConnectionError):
                    print("Request timed out, trying again.\n")
                    response = get_response(url)
                return response

            response = get_response(url)

            # Parse the response
            df = pd.DataFrame(response.json()[1:], columns=response.json()[0])

            # Turn year into an indexable column
            year_col = str(year)
            
            # Rename column
            if multi_code == False:
                df.rename(columns={census_code_dict[year]:year_col}, inplace=True)

                # Make sure our new column is float64
                df[year_col] = df[year_col].astype('float64')

            else:
                for code in census_code_dict[year]:
                    rename_var = code_name_dict[code]
                    renamed_code_columns[code] = f'{year_col}_{rename_var}'
                    # renamed_code_columns.append(f'{year_col}_{rename_var}')
                    df.rename(columns={code:f'{year_col}_{rename_var}'}, inplace=True)
                    
                    # Make sure our new column is float64
                    df[f'{year_col}_{rename_var}'] = df[f'{year_col}_{rename_var}'].astype('float64')
            
            # Make columns lowercase
            df.columns = df.columns.str.lower()

            # Append our dataframe to the temporary list to later combine
            temp_list.append(df)
            
        # Concat our two year-based dataframes together
        new_df = pd.concat(temp_list).reset_index().drop(columns=['index'])
        
        # create a block group column if get_blocks=True
        if get_blocks == True:
            new_df['block'] = new_df['geo_id'].apply(lambda x: x[-12:])
        
        # Set geo_id as the index for joining all years
        # together at the end 
        new_df = new_df.set_index('geo_id')
        
        display(new_df)
                        
        # If this is our first dataframe, include every column
        if len(df_list) == 0:
            # Append dataframe to list
            display(new_df.head())
            df_list.append(new_df)
        # Otherwise, only include the specific data column
        # to save memory
        else:
            # Only keep our main column
            if multi_code == False:
                new_df = new_df[year_col]
            else:
                renamed_list = []
                for code in renamed_code_columns:
                    renamed_list.append(renamed_code_columns[code])
                new_df = new_df[renamed_list]

            # Append dataframe to list
            display(new_df.head())
            df_list.append(new_df)
        
    return


# Define function to download census data at the MSA level
async def url_to_dataframe_async_acs1(begin_year,
                            end_year,
                            census_code_dict,
                            df_list,
                            census_code_meaning,
                            get_msa=False,
                            census_key=census_key,
                            census_table='profile',
                            multi_code=False,
                            code_name_dict=None
                          ):
    """
    Convert API url request to dataframe
    """
    session = Session()

    for year in range(begin_year, end_year + 1):
        
        # The ACS did not collect data for 2020 due to COVID-19
        if year != 2020:
        
            # Create temporary list to help combine dataframes with
            # the same year together
            temp_list = []

            if multi_code == True:
                renamed_code_columns = {}

            # Get the correct URL based on city or MSA level
            if get_msa:
                url = f"""https://api.census.gov/data/{year}/acs/acs1?get=NAME,{census_code_dict[year]}&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*&key={census_key}"""

            # Double-check the URL is correct by printing it
            print("URL:", url)

            # Simple function we can recursively recall if
            # the connection times out. However, this may
            # cause an infinite loop if the connection times
            # out due to any external reason, such as the server
            # being down. Please keep an eye as this function runs.
            def get_response(url):
                try:
                    # GET request the API
                    response = session.get(url, timeout=10)
                except (requests.exceptions.ReadTimeout, 
                        requests.exceptions.ConnectionError):
                    print("Request timed out, trying again.\n")
                    response = get_response(url)
                return response

            response = get_response(url)

            # Parse the response
            df = pd.DataFrame(response.json()[1:], columns=response.json()[0])

            # Turn year into an indexable column
            year_col = str(year)

            # Rename column
            if multi_code == False:
                df.rename(columns={census_code_dict[year]:year_col}, inplace=True)

                # Make sure our new column is float64
                df[year_col] = df[year_col].astype('float64')

            else:
                for code in census_code_dict[year]:
                    rename_var = code_name_dict[code]
                    renamed_code_columns[code] = f'{year_col}_{rename_var}'
                    # renamed_code_columns.append(f'{year_col}_{rename_var}')
                    df.rename(columns={code:f'{year_col}_{rename_var}'}, inplace=True)

                    # Make sure our new column is float64
                    df[f'{year_col}_{rename_var}'] = df[f'{year_col}_{rename_var}'].astype('float64')

            # Make columns lowercase
            df.columns = df.columns.str.lower()
            
            # If msa, use msa code provided
            if get_msa:
                df = df.rename(columns={
                    'metropolitan statistical area/micropolitan statistical area':'geo_id'
                })

            # Append our dataframe to the temporary list to later combine
            temp_list.append(df)

            # Concat our two year-based dataframes together
            new_df = pd.concat(temp_list).reset_index().drop(columns=['index'])

            # Set geo_id as the index for joining all years
            # together at the end 
            new_df = new_df.set_index('geo_id')

            display(new_df)

            # If this is our first dataframe, include every column
            if len(df_list) == 0:
                # Append dataframe to list
                display(new_df.head())
                df_list.append(new_df)
            # Otherwise, only include the specific data column
            # to save memory
            else:
                # Only keep our main column
                if multi_code == False:
                    new_df = new_df[year_col]
                else:
                    renamed_list = []
                    for code in renamed_code_columns:
                        renamed_list.append(renamed_code_columns[code])
                    new_df = new_df[renamed_list]

                # Append dataframe to list
                display(new_df.head())
                df_list.append(new_df)

    return



def final_data_prep(df_list, name, 
                    tracts=False, 
                    blocks=False, 
                    msa=False,
                    end_year=2021):
    """
    Merge the dataframes and clean them.
    """
    # Concat our three dataframes
    df = reduce(lambda left, right: 
               pd.merge(left, right, 
                        left_index=True, 
                        right_index=True,
                        how="outer"), df_list)
    
    # Prepare data for export
    df = df.reset_index()

    # Double-clean state, county, and tract columns with geo_id
    if tracts:
        # For tract-based data
        df['state'] = df['geo_id'].apply(lambda x: str(x)[-11:-9])
        df['county'] = df['geo_id'].apply(lambda x: str(x)[-9:-6])
        df['tract'] = df['geo_id'].apply(lambda x: str(x)[-11:])
        df['geo_id'] = df['geo_id'].astype(str)
        
        # Save to csv in raw folder
        df.to_csv(f'datasets/cleaned_census_api_files/raw/{name}_raw.csv', 
              encoding='utf-8', index=False)
        
    elif blocks:
        # For block-based data
        df['state'] = df['geo_id'].apply(lambda x: str(x)[-12:-10])
        df['county'] = df['geo_id'].apply(lambda x: str(x)[-10:-7])
        df['tract'] = df['geo_id'].apply(lambda x: str(x)[-12:-1])
        df['block'] = df['geo_id'].apply(lambda x: str(x)[-12:])
        df['geo_id'] = df['geo_id'].astype(str)
        
        # Save to csv in raw folder
        df.to_csv(f'datasets/cleaned_census_api_files/raw/{name}_raw.csv', 
              encoding='utf-8', index=False)
        
    elif msa:
        # For msa-based data
        df['msa_code'] = df['geo_id'].apply(lambda x: str(x)[-5:])
        
        # Drop geo_id, there is no need for it
        df.drop(columns=['geo_id'], inplace=True)
        
        # Since 2020 ACS 1-Year data is not available, I am
        # making the decision to interpolate the value (using
        # the 2019 and 2021 values)
        
        # First create the column
        df['2020'] = np.nan
        
        # Now reorganize the column
        df = df[['name','msa_code'] + [str(i) for i in range(2010, end_year + 1)]]
        
        # Now interpolate 2020 values by taking 
        # the average of 2019 and 2021
        df['2020'] = np.where(
            (df['2019'].notna() & df['2021'].notna()),
            (df['2019'] + df['2021'])/2,
            df['2020']
        )
        
        # create msa folder if nonexistent
        create_folder("datasets/cleaned_census_api_files/msa_data")
        
        # Save to csv in msa folder
        df.to_csv(f'datasets/cleaned_census_api_files/msa_data/{name}.csv', 
              encoding='utf-8', index=False)

    
    print("Completed data prep.")
    
    return df


def make_geodataframe(dataframe, census_geo_2010, census_geo_2019, census_geo_2020):
    """Add geometry columns and turn dataframe into a geodataframe"""
    
    df = copy.deepcopy(dataframe)
    
    # Create column to merge on
    df['geoid_11'] = df['geo_id'].apply(lambda x: str(x)[-11:])
    
    # Merge 2010 geodataframe and original dataframe
    geo_df = df.merge(census_geo_2010, left_on='geoid_11', right_on='GEOID10', how='left')
    geo_df.rename(columns={'geometry':'geometry_2010'}, inplace=True)

    # Merge 2019 geodataframe
    geo_df = geo_df.merge(census_geo_2019, left_on='geoid_11', right_on='GEOID', how='left')
    geo_df.rename(columns={'geometry':'geometry_2019'}, inplace=True)
    
    # Merge 2020 geodataframe
    geo_df = geo_df.merge(census_geo_2020, left_on='geoid_11', right_on='GEOID', how='left')
    geo_df.rename(columns={'geometry':'geometry_2020'}, inplace=True)
    
    # Create graphable census column
    geo_df['census'] = geo_df['geo_id'].apply(lambda x: str(x)[-6:]).apply(get_census_6)
        
    gp_df = gp.GeoDataFrame(geo_df, geometry='geometry_2010', crs='esri:102003')
    gp_df = gp_df.to_crs('esri:102003')
    gp_df['geometry_2019'] = gp_df['geometry_2019'].to_crs('esri:102003')
    gp_df['geometry_2020'] = gp_df['geometry_2020'].to_crs('esri:102003')
    gp_df['area_2010'] = gp_df['geometry_2010'].area
    gp_df['area_2019'] = gp_df['geometry_2019'].area
    gp_df['area_2020'] = gp_df['geometry_2020'].area
    
    gp_df = gp_df[['geo_id','geoid_11','census','state','county',
                    '2010','2011','2012','2013','2014','2015',
                    '2016','2017','2018','2019','2020',
                     'area_2010','area_2019','area_2020',
                     'geometry_2010','geometry_2019','geometry_2020']]
    
    print("Function make_geodataframe()... Done!")
    
    return gp_df


def merge_2010_2020(dataframe, relationship):
    """Merge 2010 census tracts with 2020"""
    
    rel = relationship[['GEOID_TRACT_20','GEOID_TRACT_10']]

    rel_2019 = pd.read_csv('datasets/cleaned_census_api_files/census_tract_relationships/relationship_2019_2010.csv',
                          encoding='utf-8',
                          dtype={'GEOID_TRACT_19':str, 'GEOID_TRACT_10':str})
    
    df = copy.deepcopy(dataframe)
    
    # Create geo_id_11 column to match with relationship data
    df['geo_id_11'] = df['geo_id'].apply(lambda x: str(x)[-11:])
    
    # Save years 2010-2019 for column indexing below
    year_2010_to_2019 = [str(i) for i in range(2010, 2020)]
    
    used_tracts = []
    
    df_merge_1 = pd.merge(df, rel, 
                        left_on='geo_id_11', 
                        right_on='GEOID_TRACT_10',
                        how="left"
                       )
    
    df_merge_2 = pd.merge(df, rel,
                         left_on='geo_id_11',
                         right_on='GEOID_TRACT_20',
                        how="left"
                         )
    
    df_merge_2019_2010 = pd.merge(df, rel_2019,
                             left_on='geo_id_11',
                             right_on='GEOID_TRACT_10',
                            how='left')
    
    df_merge_2010_2019 = pd.merge(df, rel_2019,
                             left_on='geo_id_11',
                             right_on='GEOID_TRACT_19',
                            how='left')
    
    df_merge_3 = pd.concat([df_merge_1, df_merge_2,
                            df_merge_2019_2010,
                            df_merge_2010_2019
                           ])
    df_merge_3 = df_merge_3.drop_duplicates(
        subset=['geo_id_11','GEOID_TRACT_10','GEOID_TRACT_20'])
    
    df_merge_3 = df_merge_3.reset_index(drop=True)
    
    print("Function merge_2010_2020()... Done!")
    
    return df_merge_3


def clean_census_rows(dataframe):
    """Match 2010-2019 values to 2020 census tract values"""
    df = copy.deepcopy(dataframe)
    year_2010_to_2019 = [str(i) for i in range(2010, 2020)]
    
    # Step 1: Get all GEOID_19 tract
    tracts_2019 = df['GEOID_TRACT_19'].value_counts().index.tolist()

    # Step 2: Iterate through them to clean GEOID_19 tracts
    for tract in tracts_2019:

        test_df = df[df['GEOID_TRACT_19']==tract].copy()

        x_vals = test_df[year_2010_to_2019].iloc[0].dropna().to_dict()
        try:
            x_area = test_df['area_2010'].dropna().iloc[0]
            x_geom = test_df['geometry_2010'].dropna().iloc[0]
            test_df['area_2010'] = x_area
            test_df['geometry_2010'] = x_geom
        except IndexError:
            x_area = test_df['area_2019'].dropna().iloc[0]
            x_geom = test_df['geometry_2019'].dropna().iloc[0]
            test_df['area_2019'] = x_area
            test_df['geometry_2019'] = x_geom
        
        tract_2010 = test_df['GEOID_TRACT_10'].iloc[0]

        for key in x_vals:
            df[key] = np.where(df['GEOID_TRACT_19']==tract, x_vals[key], df[key])
        new_vals = df[df['GEOID_TRACT_19']==tract][year_2010_to_2019].iloc[-1].dropna().to_dict()
        for key in new_vals:
            df[key] = np.where(df['GEOID_TRACT_10']==tract_2010, new_vals[key], df[key])
        
    # Step 3: Clean all other tracts
    
    def transform_prev_values(df):
                
        year_2010_to_2019 = [str(i) for i in range(2010, 2020)]
        
        # Transfer census values
        x_vals = df[year_2010_to_2019].iloc[0].dropna().to_dict()
    
        try:
            x_area = df['area_2010'].dropna().iloc[0]
            x_geom = df['geometry_2010'].dropna().iloc[0]
            df['area_2010'] = x_area
            df['geometry_2010'] = x_geom
        except IndexError:
            x_area = df['area_2019'].dropna().iloc[0]
            x_geom = df['geometry_2019'].dropna().iloc[0]
            df['area_2019'] = x_area
            df['geometry_2019'] = x_geom
            

        for key in x_vals:
            df[key] = x_vals[key]
        
        prop_sum = df['2020'].sum()
        df['2020_proportion'] = df['2020']/prop_sum

        return df
    
    # All 2020 census tracts should have the same 2010-2019 values
    df['2020_proportion'] = 0
    new_df = df.groupby(['GEOID_TRACT_10']).apply(transform_prev_values)
    na_df = df[df['GEOID_TRACT_10'].isnull()]
    new_df = pd.concat([new_df, na_df])
    new_df = new_df.reset_index(drop=True)
    
    print("Function clean_census_rows()... Done!")
    
    return new_df


def drop_non_2020_tracts(dataframe):
    """Only keep tracts included in the 2020 decennial census"""
    
    df = copy.deepcopy(dataframe)
    df = df[df['geo_id_11'] == df['GEOID_TRACT_20']]
    
    # Sort by area_2010 in descending order allows us to easily
    # keep the 2010 tracts that are most likely to represent
    # the actual 2020 tracts. The relationship file keeps all
    # 2010 tracts related to a 2020 tract that have had their
    # geometries moved, even if just a little. That means some
    # (if not most) 2020 tracts have multiple 2010 tracts related
    # to them, and if we don't sort by highest area, it could mean
    # mistakenly assigning one of these smaller tracks as the
    # original, giving the appearance that the 2020 tract grew
    # in size, when in most of the time the 2020 tract should
    # actually shrink.

    
    # Resort
    df.sort_values('state', ascending=True, inplace=True)
    
    # Create area_percentage column
    conditions = [df['area_2010'].notna(), 
                 (df['area_2010'].isnull()) & (df['area_2019'].notna()),
                 (df['area_2010'].isnull()) & (df['area_2019'].isnull())]
    
    choices = [df['area_2020']/df['area_2010'],
               df['area_2020']/df['area_2019'],
               df['area_2020']]
    
    df['area_percent'] = np.select(conditions, choices, default=np.nan)
    
    df = df.reset_index(drop=True)
    
    # Keep only the columns we want
    df = df[['geo_id','geo_id_11','census','state','county',
            '2010','2011','2012','2013','2014','2015',
            '2016','2017','2018','2019','2020',
             'area_2010','area_2020',
             'area_percent', 'GEOID_TRACT_10', 'GEOID_TRACT_20',
            '2020_proportion']]

    
    print("Function drop_non_2020_tracts()... Done!")
    
    return df


def old_standardize(dataframe, standardize=False):
    """
    Geographically standardize census values.
    
    Args
    ------
    dataframe (DataFrame): A dataframe.
    standardize (Boolean): True/False. Set
    to False by default.
    
    Returns
    ------
    If standardize=True (DataFrame): A dataframe with its 
    census values from 2010-2019 standardized 
    according to the percentage of area it
    occupies in 2020 with respect to its 2010 area.
    
    If stanrdize=False (None): Returns nothing.
    """
    if standardize == False:
        print('"standardize=" set to False. Skipping standardization.')
        return dataframe
    else:
        df = copy.deepcopy(dataframe)
        year_2010_to_2019 = [str(i) for i in range(2010, 2020)]
        
        # Census tracts that are bigger in 2020 occur less than
        # 2% of the time, and for simplicity, we will not
        # adjust their 2010-2019 values, as there shouldn't be
        # a need to geo-standardize them. Technically, if the 
        # 'area_percent' is greater than 1, we may want to instead
        # multiply the 'area_percent' by the value in 2020, rather
        # than the values in 2010-2019. If the area_percent is 
        # more than double in the year 2020, in some cases this means
        # we may want to multiply the 2020 value by the same
        # amount (more than double). This should be discussed.
        
        for year in year_2010_to_2019:
            df[year] = np.where(df['area_percent'] < 1.1, 
                                df[year]*df['area_percent'],
                                df[year])
        
        print("Function standardize()... Done!")
        
        return df  
        

def old_proportion_standardize(dataframe, standardize=False):
    """
    Standardize census values NOT geographically but instead
    by the proportional value of each new tract. For example,
    if tract 300.00 was split into 300.01 and 300.02,
    and 300.01 has a geometry that's 10% of the original
    geometry, and 300.02 has 90% of the original geometry,
    if the 2010-2019 values were "geo-standardized" by 
    geographical area, the values for 2010-2019 for tract
    300.01 would be multiplied by 0.10. It is looking like
    this may be the incorrect way to standardize the datasets,
    as for some tracts, even if a new tract like 300.01 has
    10% of the original area, it could have 50% or more
    of the original tract's value. So instead of multiplying
    its 2010-2019 values by 10%, it might make more sense
    to multiply those values by the proportion its 2020
    value has with respect to the other tract it was split 
    with. As an example, if 300.01 has 40% of the combined value 
    between 300.01 and 300.02, this function will multiply
    300.01's 2010-2019 values by 0.40 and multiply 300.02's
    2010-2019 values by 0.60. Thus, standardizing by their 2020
    census variable value proportions rather than their 2020
    geometry proportions.
    
    Args
    ------
    dataframe (DataFrame): A dataframe.
    standardize (Boolean): True/False. Set
    to False by default.
    
    Returns
    ------
    If standardize=True (DataFrame): A dataframe with its 
    census values from 2010-2019 standardized 
    according to the percentage its 2020 value has
    with respect to total 2020 values of all tracts
    that were split.
    
    If stanrdize=False (None): Returns nothing.
    """
    if standardize == False:
        print('"standardize=" set to False. Skipping standardization.')
        return dataframe
    else:
        df = copy.deepcopy(dataframe)
        year_2010_to_2019 = [str(i) for i in range(2010, 2020)]
        
        for year in year_2010_to_2019:
            df[year] = df[year] * df['2020_proportion']
        
        print("Function proportion_standardize()... Done!")
        
        return df  


def spot_check(df, 
                first_year, 
                last_year,
                post_std=False):
    """
    Run a spot check on the dataframe to look
    for anomalies.

    Parameters:
        df (DataFrame): The DataFrame you want statistics on.
        first_year (int): The year to start with.
        last_year (int): The year to end with.
        post_std (Boolean): Set this to True if you're analyzing
            a dataset post-standardization.
    """
    n = last_year + 1
    
    # Check for all 50 states, DC, and Puerto Rico
    print("How many states:", len(df['state'].value_counts()), "\n")

    if post_std == False:
        # Check for null values
        for i in range(first_year, n):
            print(f"Null values in {i}:", df[df[f'{i}'].isnull()].shape[0])
        print("\n")
        
        # Check for null values in multiple years
        for i in range(first_year, last_year):
            print(f"{last_year} and {i} null values:", df[(df[f'{last_year}'].isnull()) & (df[f'{i}'].isnull())].shape[0])
        print("\n")

        # Check stats
        for i in range(first_year, n):
            print(f"Stats for year {i}:\n", df[f'{i}'].describe(), "\n")
    else:
        # Check for null values in block groups
        print('BLOCK GROUPS:\n')
        for i in range(first_year, n):
            print(f"Null values in {i}_block:", df[df[f'{i}_block'].isnull()].shape[0])
        print("\n")
        
        # Check for null values in multiple years
        for i in range(first_year, last_year):
            print(f"{last_year} and {i} block null values:", df[(df[f'{last_year}_block'].isnull()) & (df[f'{i}_block'].isnull())].shape[0])
        print("\n")

        # Check stats
        for i in range(first_year, n):
            print(f"Stats for year {i}_block:\n", df[f'{i}_block'].describe(), "\n")
        
        # Check for null values in tracts
        print('TRACTS:\n')
        for i in range(first_year, n):
            print(f"Null values in {i}_tract:", df[df[f'{i}_tract'].isnull()].shape[0])
        print("\n")
        
        # Check for null values in multiple years
        for i in range(first_year, last_year):
            print(f"{last_year} and {i} tract null values:", df[(df[f'{last_year}_tract'].isnull()) & (df[f'{i}_tract'].isnull())].shape[0])
        print("\n")

        # Check stats
        for i in range(first_year, n):
            print(f"Stats for year {i}_tract:\n", df[f'{i}_tract'].describe(), "\n")
        
    return


def merge_2010_2012_to_2013_2020(data_name, dataframe):
    """
    Merge the 2010-2012 block datasets to the
    2013-2020 dataset.
    """
    df = copy.deepcopy(dataframe)
    
    # READ IN 2010-2012 POPULATION
    data_2010 = pd.read_csv(f'datasets/ipums_data/block_{data_name}/block_group_{data_name}_2010.csv',
                            encoding='utf-8',
                            dtype={'TRACT':str, 'BLOCK':str})
    data_2011 = pd.read_csv(f'datasets/ipums_data/block_{data_name}/block_group_{data_name}_2011.csv',
                                encoding='utf-8',
                                dtype={'TRACT':str, 'BLOCK':str})
    data_2012 = pd.read_csv(f'datasets/ipums_data/block_{data_name}/block_group_{data_name}_2012.csv',
                                encoding='utf-8',
                                dtype={'TRACT':str, 'BLOCK':str})
    
    # Merge the 2010-2012 and 2013-2020 datasets
    new_2012 = pd.merge(df, 
                        data_2012[['BLOCK',f'{data_name}']],
                        left_on="block", 
                        right_on="BLOCK",
                        how="left").rename(columns={f'{data_name}':'2012'})

    new_2011 = pd.merge(new_2012, 
                        data_2011[['BLOCK',f'{data_name}']],
                        left_on="block", 
                        right_on="BLOCK",
                        how="left").rename(columns={f'{data_name}':'2011'})

    new_2010 = pd.merge(new_2011, 
                        data_2010[['BLOCK',f'{data_name}']],
                        left_on="block", 
                        right_on="BLOCK",
                        how="left").rename(columns={f'{data_name}':'2010'})

    new_2010['block'] = new_2010['geo_id'].apply(lambda x: str(x)[-12:])
    new_2010['tract'] = new_2010['block'].apply(lambda x: str(x)[:-1])

    new_df = new_2010[['geo_id','block','tract'] + 
                     [str(i) for i in range(2010, 2021)]]
    
    new_df.to_csv(f'datasets/cleaned_census_api_files/raw/{data_name}_blocks_raw.csv', encoding='utf-8', index=False)

    print("Completed 2010-2012 and 2013-2020 block data merge.")
    
    return new_df


def merge_with_crosswalk(dataframe, block_groups_1020=block_groups_1020):
    """Merge dataframe with block crosswalk file."""
    
    df = dataframe.copy()
    
    crosswalk_10 = pd.merge(df, 
                            block_groups_1020,
                            left_on="block",
                            right_on="BG10",
                            how="left")

    crosswalk_20 = pd.merge(crosswalk_10,
                           block_groups_1020,
                           left_on="block",
                           right_on="BG20",
                           how="left")

    crosswalk_20['BG10'] = np.where(crosswalk_20['BG10_x'].notna(), 
                                    crosswalk_20['BG10_x'],
                                    crosswalk_20['BG10_y'])

    crosswalk_20['BG20'] = np.where(crosswalk_20['BG20_x'].notna(), 
                                    crosswalk_20['BG20_x'],
                                    crosswalk_20['BG20_y'])

    crosswalk_20['wt_pop'] = np.where(crosswalk_20['wt_pop_x'].notna(), 
                                    crosswalk_20['wt_pop_x'],
                                    crosswalk_20['wt_pop_y'])

    crosswalk_20['wt_hu'] = np.where(crosswalk_20['wt_hu_x'].notna(), 
                                    crosswalk_20['wt_hu_x'],
                                    crosswalk_20['wt_hu_y'])

    crosswalk_20['wt_adult'] = np.where(crosswalk_20['wt_adult_x'].notna(), 
                                    crosswalk_20['wt_adult_x'],
                                    crosswalk_20['wt_adult_y'])

    crosswalk_20['wt_fam'] = np.where(crosswalk_20['wt_fam_x'].notna(), 
                                    crosswalk_20['wt_fam_x'],
                                    crosswalk_20['wt_fam_y'])

    crosswalk_20['wt_hh'] = np.where(crosswalk_20['wt_hh_x'].notna(), 
                                    crosswalk_20['wt_hh_x'],
                                    crosswalk_20['wt_hh_y'])

    crosswalk_20['parea'] = np.where(crosswalk_20['parea_x'].notna(), 
                                    crosswalk_20['parea_x'],
                                    crosswalk_20['parea_y'])

    crosswalk_20['TRACT20'] = np.where(crosswalk_20['TRACT20_x'].notna(), 
                                    crosswalk_20['TRACT20_x'],
                                    crosswalk_20['TRACT20_y'])

    crosswalk_20['TRACT10'] = np.where(crosswalk_20['TRACT10_x'].notna(), 
                                    crosswalk_20['TRACT10_x'],
                                    crosswalk_20['TRACT10_y'])

    crosswalk_20.drop(columns=crosswalk_20.filter(regex='_x|_y').columns, inplace=True)
    
    crosswalk_20 = crosswalk_20[(crosswalk_20['BG20'].notna())
                         ].reset_index(drop=True).copy()
    
    return crosswalk_20


def block_standardize(block, 
                      pop_dict,
                      og_df,
                      year_end):
    """
    WARNING: This function alone takes a few seconds to complete
    per block group, but to standard all 242,333 block groups
    can take many, many hours to run.
    It would be wise to run this function on any type of
    parrallel processing, such as using Dask, or a GPU,
    or parrallelized cloud computing, as there is no
    serialization (the block groups can be standardized
    in no particular order).
    
    This function standardizes all block group rows. It 
    should be called in a loop or vectorized if possible,
    such as the example below. (Note, the example below
    may not be the most efficient way to loop through
    or vectorize the block groups.)
    
    ```
    # Loop through all population block groups
    # and standardize them
    pop_dictionary = {}
    array2 = pop_pre_st['BG20'].unique()
    [block_standardize(
            x, 
            pop_dict=pop_dictionary, 
            og_df=pop_pre_st) 
        for x in array2]
    ```
    
    Parameters:
        block (str): the block_group to group by.
        pop_dict (dict): The dictionary where the
            function should append results to.
        og_df (DataFrame): The dataframe we are 
            standardizing from.
    
    Returns:
        None. However, it appends the standardized values
            per block group to a pre-defined dictionary.
    """
    
    years_13_19 = [str(i) for i in range(2013, 2020)]
        
    # Step 1: Get a dataframe grouped by BG20
    df = og_df.copy()
    bg20_df = df[df['BG20']==block].drop_duplicates()
    bg20_df = bg20_df.fillna(0)
        
    # Step 2: Get dot product of 2013-2019 values with wt_pop values
    array_13_19 = bg20_df[years_13_19].to_numpy().T
    wt_array = bg20_df['wt_pop'].to_numpy()
    dots = array_13_19.dot(wt_array)
    
    # Step 3: Append standardized 2010-2019 and 
    # the block's 2020+ values to new dictionary
    filtered = df[df['block']==block].drop_duplicates()
    val_20s = filtered[[
        str(i) for i in range(2020, year_end + 1)]].iloc[0].to_numpy()

    # If the 2020 value is 0, then all years
    # before then should be 0 also
    if val_20s[0] == 0:
        dots = dots * 0

    # Finalize the append
    dots = np.append(dots, val_20s)

    # Add to dictionary
    pop_dict[block] = dots
    
    return


def block_standardize_tuple(tuple):
    """
    WARNING: This function alone takes a few seconds to complete
    per block group, but to standard all 242,333 block groups
    can take many, many hours to run.
    It would be wise to run this function on any type of
    parrallel processing, such as using Dask, or a GPU,
    or parrallelized cloud computing, as there is no
    serialization (the block groups can be standardized
    in no particular order).
    
    This function standardizes all block group rows. It 
    should be called in a loop or vectorized if possible,
    such as the example below. (Note, the example below
    may not be the most efficient way to loop through
    or vectorize the block groups.)
    
    ```
    # Loop through all population block groups
    # and standardize them
    pop_dictionary = {}
    array2 = pop_pre_st['BG20'].unique()
    [block_standardize(
            x, 
            pop_dict=pop_dictionary, 
            og_df=pop_pre_st) 
        for x in array2]
    ```
    
    Parameters:
        tuple (tuple): A tuple containing the below.
            block (str): The block_group to group by.
            og_df (DataFrame): The dataframe we are 
                standardizing from.
            year_start (int): Which year to start from.
            year_end (int): Which year to end from.
            weight (str): Which weight to use (such as 
                'wt_pop' pr 'wt_hh').
    
    Returns:
        None. However, it appends the standardized values
            per block group to a pre-defined dictionary.
    """

    block = tuple[0]
    og_df = tuple[1]
    year_start = tuple[2]
    year_end = tuple[3]
    weight = tuple[4]
    
    years_10_19 = [str(i) for i in range(year_start, year_end)]
        
    # Step 1: Get a dataframe grouped by BG20
    df = og_df.copy()
    bg20_df = df[df['BG20']==block].drop_duplicates()
    bg20_df = bg20_df.fillna(0)
        
    # Step 2: Get dot product of 2010-2019 values with the target weight values values
    array_10_19 = bg20_df[years_10_19].to_numpy().T
    wt_array = bg20_df[weight].to_numpy()
    dots = array_10_19.dot(wt_array)
    
    # Step 3: Append standardized 2010-2019 and 
    # the block's 2020 value to new dictionary
    filtered = df[df['block']==block]
    val_20 = filtered['2020'].iloc[0]
    dots = np.append(dots, val_20)
    
    # If the 2020 value is 0, then all years
    # before then should be 0 also
    if val_20 == 0:
        dots = dots * 0
        
    return {block : dots} 


def block_standardize_tuple_medians(tuple):
    """
    WARNING: This function alone takes a few seconds to complete
    per block group, but to standard all 242,333 block groups
    can take many, many hours to run.
    It would be wise to run this function on any type of
    parrallel processing, such as using Dask, or a GPU,
    or parrallelized cloud computing, as there is no
    serialization (the block groups can be standardized
    in no particular order).
    
    This function standardizes all block group rows. It 
    should be called in a loop or vectorized if possible,
    such as the example below. (Note, the example below
    may not be the most efficient way to loop through
    or vectorize the block groups.)
    
    ```
    # Loop through all population block groups
    # and standardize them
    pop_dictionary = {}
    array2 = pop_pre_st['BG20'].unique()
    [block_standardize(
            x, 
            pop_dict=pop_dictionary, 
            og_df=pop_pre_st) 
        for x in array2]
    ```
    
    Parameters:
        tuple (tuple): A tuple containing the below.
            block (str): The block_group to group by.
            og_df (DataFrame): The dataframe we are 
                standardizing from.
            year_start (int): Which year to start from.
            year_end (int): Which year to end from.
            weight (str): Which weight to use (such as 
                'wt_pop' pr 'wt_hh').
    
    Returns:
        None. However, it appends the standardized values
            per block group to a pre-defined dictionary.
    """

    block = tuple[0]
    og_df = tuple[1]
    year_start = tuple[2]
    year_end = tuple[3]
    weight = tuple[4]  
    code_name_dict_2013_2014 = tuple[5]  
    code_name_dict_2015_2020 = tuple[6]

        
    # Step 1: Get a dataframe grouped by BG20
    df = og_df.copy()
    bg20_df = df[df['BG20']==block].drop_duplicates()
    bg20_df = bg20_df.fillna(0)
    
    return_array = np.array([]) # make sure the final is in the form of {block : dots}
    
    # Step 2: Loop through the code names that 2013 and 2014 are guaranteed to have
    for code in code_name_dict_2013_2014:
        
        rent_category = code_name_dict_2013_2014[code]
        years_10_19 = [f"{i}_{rent_category}" for i in range(year_start, year_end)]
        
        # Step 3: Get dot product of 2010-2019 values with the target weight values values
        array_10_19 = bg20_df[years_10_19].to_numpy().T
        wt_array = bg20_df[weight].to_numpy()
        dots = array_10_19.dot(wt_array)
    
        # Step 3: Append standardized 2010-2019 and 
        # the block's 2020 value to new dictionary
        filtered = df[df['block']==block]
        val_20 = filtered[f'2020_{rent_category}'].iloc[0]
        dots = np.append(dots, val_20)

        # If the 2020 value is 0, then all years
        # before then should be 0 also
        if val_20 == 0:
            dots = dots * 0
            
        set_dots = dots.copy()
            
        # Update return_dictionary
        return_array = np.append(return_array, set_dots)
        
    # Step 2: Loop through the code names that 2013 and 2014 won't have
    for code in code_name_dict_2015_2020:
        
        rent_category = code_name_dict_2015_2020[code]
        years_10_19 = [f"{i}_{rent_category}" for i in range(2015, year_end)]
        
        # Step 3: Get dot product of 2010-2019 values with the target weight values values
        array_10_19 = bg20_df[years_10_19].to_numpy().T
        wt_array = bg20_df[weight].to_numpy()
        dots = array_10_19.dot(wt_array)
    
        # Step 3: Append standardized 2010-2019 and 
        # the block's 2020 value to new dictionary
        filtered = df[df['block']==block]
        val_20 = filtered[f'2020_{rent_category}'].iloc[0]
        dots = np.append(dots, val_20)

        # If the 2020 value is 0, then all years
        # before then should be 0 also
        if val_20 == 0:
            dots = dots * 0
            
        set_dots = dots.copy()
            
        # Update return_dictionary
        return_array = np.append(return_array, set_dots)
        
    return {block : return_array}


def block_standardize_medians(block,
                            og_df,
                            year_start,
                            year_end,
                            weight,
                            code_name_dict_2013_2014,
                            code_name_dict_2015_2020):

    """
    WARNING: This function alone takes a few seconds to complete
    per block group, but to standard all 242,333 block groups
    can take many, many hours to run.
    It would be wise to run this function on any type of
    parrallel processing, such as using Dask, or a GPU,
    or parrallelized cloud computing, as there is no
    serialization (the block groups can be standardized
    in no particular order).
    
    This function standardizes all block group rows. It 
    should be called in a loop or vectorized if possible,
    such as the example below. (Note, the example below
    may not be the most efficient way to loop through
    or vectorize the block groups.)
    
    ```
    # Loop through all population block groups
    # and standardize them
    pop_dictionary = {}
    array2 = pop_pre_st['BG20'].unique()
    [block_standardize(
            x, 
            pop_dict=pop_dictionary, 
            og_df=pop_pre_st) 
        for x in array2]
    ```
    
    Parameters:
        tuple (tuple): A tuple containing the below.
            block (str): The block_group to group by.
            og_df (DataFrame): The dataframe we are 
                standardizing from.
            year_start (int): Which year to start from.
            year_end (int): Which year to end from.
            weight (str): Which weight to use (such as 
                'wt_pop' pr 'wt_hh').
    
    Returns:
        None. However, it appends the standardized values
            per block group to a pre-defined dictionary.
    """

        
    # Step 1: Get a dataframe grouped by BG20
    df = og_df.copy()
    bg20_df = df[df['BG20']==block].drop_duplicates()
    bg20_df = bg20_df.fillna(0)
    # filtered = df[df['block']==block]
    
    return_array = np.array([]) # make sure the final is in the form of {block : dots}
    
    # Step 2: Loop through the code names that 2013 and 2014 are guaranteed to have
    for code in code_name_dict_2013_2014:
        
        rent_category = code_name_dict_2013_2014[code]
        years_10_19 = [f"{i}_{rent_category}" for i in range(year_start, year_end)]
        
        # Step 3: Get dot product of 2010-2019 values with the target weight values values
        array_10_19 = bg20_df[years_10_19].to_numpy().T
        wt_array = bg20_df[weight].to_numpy()
        dots = array_10_19.dot(wt_array)
    
        # Step 3: Append standardized 2010-2019 and 
        # the block's 2020 value to new dictionary
        val_20 = bg20_df[f'2020_{rent_category}'].iloc[0]
        dots = np.append(dots, val_20)

        # If the 2020 value is 0, then all years
        # before then should be 0 also
        if val_20 == 0:
            dots = dots * 0
            
        set_dots = dots.copy()
            
        # Update return_dictionary
        return_array = np.append(return_array, set_dots)
        
    # Step 2: Loop through the code names that 2013 and 2014 won't have
    for code in code_name_dict_2015_2020:
        
        rent_category = code_name_dict_2015_2020[code]
        years_10_19 = [f"{i}_{rent_category}" for i in range(2015, year_end)]
        
        # Step 3: Get dot product of 2010-2019 values with the target weight values values
        array_10_19 = bg20_df[years_10_19].to_numpy().T
        wt_array = bg20_df[weight].to_numpy()
        dots = array_10_19.dot(wt_array)
    
        # Step 3: Append standardized 2010-2019 and 
        # the block's 2020 value to new dictionary
        val_20 = bg20_df[f'2020_{rent_category}'].iloc[0]
        dots = np.append(dots, val_20)

        # If the 2020 value is 0, then all years
        # before then should be 0 also
        if val_20 == 0:
            dots = dots * 0
            
        set_dots = dots.copy()
            
        # Update return_dictionary
        return_array = np.append(return_array, set_dots)
        
    return {block : return_array}


def specify_geographies(dataframe, start_year, end_year):
    """
    Add columns for state, county, and tract, 
    and make sums for census tracts.
    """
    df = dataframe.copy()
    df['geoid_tract'] = df['geoid_block'].apply(lambda x: str(x)[:-1])
    df['state'] = df['geoid_block'].apply(lambda x: str(x)[0:2])
    df['county'] = df['geoid_block'].apply(lambda x: str(x)[2:5])
    df['tract'] = df['geoid_block'].apply(lambda x: str(x)[5:11])

    n = end_year + 1
    
    years = [str(i) for i in range(start_year, n)]
    
    # Create summations per tract
    for year in years:
        df.rename(columns={f'{year}':f'{year}_block'}, inplace=True)
        df[f'{year}_tract'] = df.groupby(['geoid_tract'])[f'{year}_block'].transform('sum')
    
    # Reorder the columns -- If I have to change this again,
    # just reorder the columns by naming every columns to keep
    df = df.iloc[:, [0, len(years) + 3, len(years) + 4, 
                    len(years) + 5, len(years) + 6] + 
                    [i for i in range(1, len(years) + 3)] + 
                    [i for i in range(len(years) + 7, 2*len(years) + 7)]]
    
    return df


def create_and_save_geo_files(dataframe, 
                              name,
                              keyword='',
                              begin_year=2010, 
                              end_year=2020):
    """
    Create and save a census tract-based shapefile.
    """
    df = dataframe.copy()
    census20_geo = gp.read_file('datasets/census_original_files/census_tracts_2020/all_tracts_2020/all_tracts_2020.shp')

    census20_geo = census20_geo[['GEOID','geometry']]
    
    standardized_tracts_geo = census20_geo.merge(df,
                                                 how='inner',
                                                 left_on='GEOID',
                                                 right_on='geoid_tract')

    if keyword != '':
        keyword = '_' + keyword
    
    year_tract_list = [f'{year}_tract{keyword}' for year in range(begin_year, end_year+1)]
        
    standardized_tracts_geo = standardized_tracts_geo[
        ['geoid_tract','state',
        'county','tract'] + year_tract_list +
        ['geometry']].drop_duplicates().reset_index(drop=True)
    
    print('Standardized tracts:')
    display(standardized_tracts_geo)

    # Create folder if nonexistent
    create_folder("datasets/for_software")
    create_folder(f"datasets/for_software/{name}")
    create_folder(f"datasets/for_software/{name}/{name}_tract_geo/")

    # Save the files
    standardized_tracts_geo.to_file(f'datasets/for_software/{name}/{name}_tract_geo/{name}_tract_geo.gpkg', driver="GPKG")
    print(f'Successfully saved {name}_tract_geo.gpkg')
    
    return standardized_tracts_geo
    


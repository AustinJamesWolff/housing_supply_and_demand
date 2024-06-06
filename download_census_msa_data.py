"""
This module downloads the most recent census data from the
US Census at the metropolitan level.

You can simply run "python download_census_msa_data.py" in
the terminal to begin the download. Make sure your
virtual environment is activated.
"""
# Import libraries
import pandas as pd

# All helper functions are in this module:
from helper_functions.census_functions import *

# Set up Pandas defaults
pd.options.display.float_format = '{:.4f}'.format
pd.set_option("display.max_columns", None)

# Create directories for file saving
create_folder("datasets/cleaned_census_api_files/")
create_folder("datasets/cleaned_census_api_files/graphable/")
create_folder("datasets/cleaned_census_api_files/raw/")
create_folder("datasets/cleaned_census_api_files/standardized/")

# Define beginning year
begin_year = 2010

# Ask for end year
end_year = int(input("What is the last year to download data from? "))

# Run the API to download population
population_msa = download_and_format_msa_census_data(
    census_code="B01003_001E",
    census_code_meaning="population_msa",
    end_year=end_year)

# Run the API to download median income
median_income_msa = download_and_format_msa_census_data(
    census_code="B19013_001E",
    census_code_meaning="median_income_msa",
    end_year=end_year)

# Run the API to download median price
median_price_msa = download_and_format_msa_census_data(
    census_code="B25077_001E",
    census_code_meaning="median_price_msa",
    end_year=end_year)

# Run the API to download median rent
median_rent_msa = download_and_format_msa_census_data(
    census_code="B25058_001E",
    census_code_meaning="median_rent_msa",
    end_year=end_year)

# Run the API to download total units
total_units_msa = download_and_format_msa_census_data(
    census_code="B25001_001E",
    census_code_meaning="total_units_msa",
    end_year=end_year)

# Run the API to download vacant units
vacant_units_msa = download_and_format_msa_census_data(
    census_code="B25002_003E",
    census_code_meaning="vacant_units_msa",
    end_year=end_year)


# ## Create Rent-to-Price dataset

# Rename columns
for i in range(2010, end_year + 1):
    median_rent_msa.rename(columns={f"{i}":f"{i}_rent"}, inplace=True)

# Rename columns
for i in range(2010, end_year + 1):
    median_price_msa.rename(columns={f"{i}":f"{i}_price"}, inplace=True)

# Merge price data
rent_to_price = median_rent_msa.merge(
    median_price_msa, how='inner', 
    on=['msa_code','msa_name'])

# Loop through columns and divide rent by price per year
for i in range(2010, end_year + 1):
    rent_to_price[f'{i}'] = rent_to_price[f"{i}_rent"]/rent_to_price[f"{i}_price"]
    
    # Drop rent and price columns
    rent_to_price.drop(columns=[f'{i}_rent',f'{i}_price'], inplace=True)

# Save dataset
rent_to_price.to_csv(
    "datasets/cleaned_census_api_files/msa_data/rent_to_price_ratio_msa.csv", 
    index=False)

### Create Jobs per Unit dataset

# Read in jobs
jobs = pd.read_csv('datasets/bls/raw/most_recent_bls_data.csv',
                   dtype={'msa_code':str, 'state_code':str})

# Make sure the date column is in datetime format
jobs['date'] = pd.to_datetime(jobs['date'])

# Replace NECTA Division
jobs['msa_name'] = jobs['msa_name'].apply(lambda x: x.replace(" NECTA Division",""))
jobs['msa_name'] = jobs['msa_name'].apply(lambda x: x.replace(" NECTA",""))

# Keep only december months
new_jobs = jobs[jobs['month']=='December'].reset_index(drop=True)

# Get earliest year
earliest_year = new_jobs['year'].min()

# Get latest year
latest_year = new_jobs['year'].max()

# Only keep certain columns
new_jobs = new_jobs[['msa_name','year','value']]

# Rename column
new_jobs.rename(columns={'value':f'jobs'}, inplace=True)

# Stack and unstack
new_jobs = new_jobs.set_index(['msa_name','year'])
new_jobs = new_jobs.unstack('year')

# Reset index
new_jobs = new_jobs.reset_index()

# Rename jobs columns
new_jobs.columns = ['msa_name'] + [
    f'{i}_jobs' for i in range(earliest_year, latest_year + 1)]

# Read in total units and rename columns
total_units_for_jobs = pd.read_csv(
    "datasets/cleaned_census_api_files/msa_data/total_units_msa.csv")
for i in range(earliest_year, latest_year + 1):
    total_units_for_jobs.rename(columns={f"{i}":f"{i}_units"}, inplace=True)

# Merge data
jobs_per_unit = new_jobs.merge(
    total_units_for_jobs, how='inner', 
    on=['msa_name'])

# Loop through columns and divide rent by price per year
for i in range(earliest_year, end_year + 1):
    jobs_per_unit[f'{i}'] = jobs_per_unit[f"{i}_jobs"]/jobs_per_unit[f"{i}_units"]

# Only keep main columns
jobs_per_unit = jobs_per_unit[['msa_name','msa_code'] +
    [f'{i}' for i in range(earliest_year, end_year + 1)]]

# Save dataset
jobs_per_unit.to_csv(
    "datasets/cleaned_census_api_files/msa_data/jobs_per_unit_msa.csv", 
    index=False)


### CREATE POPULATION PER UNIT

# Rename columns
for i in range(2010, end_year + 1):
    population_msa.rename(columns={f"{i}":f"{i}_population"}, inplace=True)
for i in range(2010, end_year + 1):
    total_units_msa.rename(columns={f"{i}":f"{i}_units"}, inplace=True)

# Enforce dtype
population_msa['msa_code'] = population_msa['msa_code'].astype(str)
total_units_msa['msa_code'] = total_units_msa['msa_code'].astype(str)

# Merge price data
pop_per_units = population_msa.merge(
    total_units_msa, how='inner', 
    on=['msa_code','msa_name'])

# Loop through columns and divide rent by price per year
for i in range(2010, end_year + 1):
    pop_per_units[f'{i}'] = pop_per_units[f"{i}_population"]/pop_per_units[f"{i}_units"]
    
    # Drop rent and price columns
    pop_per_units.drop(columns=[f'{i}_population',f'{i}_units'], inplace=True)

# Save dataset
pop_per_units.to_csv(
    "datasets/cleaned_census_api_files/msa_data/population_per_unit_msa.csv", 
    index=False)


### CREATE VACANCY RATE
# Rename columns
for i in range(2010, end_year + 1):
    vacant_units_msa.rename(columns={f"{i}":f"{i}_vacant"}, inplace=True)

# Enforce dtype
vacant_units_msa['msa_code'] = vacant_units_msa['msa_code'].astype(str)

# Merge price data
vacancy_rate = vacant_units_msa.merge(
    total_units_msa, how='inner', 
    on=['msa_code','msa_name'])

# Loop through columns and divide rent by price per year
for i in range(2010, end_year + 1):
    vacancy_rate[f'{i}'] = vacancy_rate[f"{i}_vacant"]/vacancy_rate[f"{i}_units"]
    
    # Drop rent and price columns
    vacancy_rate.drop(columns=[f'{i}_vacant',f'{i}_units'], inplace=True)

# Save dataset
vacancy_rate.to_csv(
    "datasets/cleaned_census_api_files/msa_data/vacancy_rate_msa.csv", 
    index=False)

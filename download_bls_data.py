"""
This module downloads the most recent job data from the
BLS at the metropolitan level for the past 10 years.

You can simply run "python download_bls_data.py" in
the terminal to begin the download. Make sure your
virtual environment is activated.
"""

# Import libraries
import requests
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np

# Set up Pandas defaults
pd.options.display.float_format = '{:.2f}'.format
pd.set_option("display.max_columns", None)

# Test getting BLS key
load_dotenv()
bls_key = os.environ["BLS_KEY"]

# Define helper function to create directory
def create_folder(the_path):
    "Create directory if nonexistent."
    if not os.path.isdir(the_path):
        os.mkdir(the_path)

# Step 1: Prepare the variables needed for the API pull

# Read in the MSA Codes
msa_codes = pd.read_csv(
    "datasets/helper_datasets/msa_and_state_codes_bls_api.csv",
    dtype={'CBSA Code':str, 'FIPS State Code':str, 'state_and_msa_code':str}
)

# The seriesID code needed for the pull is in this format:
# "SMS" + "state_and_msa_code" + "0000000001"

# Create a seriesID column for employment
msa_codes['employment_seriesID'] = "SMS" + msa_codes['state_and_msa_code'] + "0000000001"

# The BLS API can only take up to 50 seriesIDs at a time,
# so we will split up the dataframe into 50-row chunks.
# The number of BLS MSAs we will download is 390, 
# so we will have 8 chunks.

# Create main dictionary
msa_code_dict = {}

# Create 50-row chunks
for i in range(8):
    start_index = i * 50
    
    if i != 7:
        end_index = (i + 1) * 50
        temp_df = msa_codes[start_index:end_index].copy()
    else:
        temp_df = msa_codes[start_index:].copy()
        
    # Create dictionary to match seriesID to metro name
    temp_dict = pd.Series(temp_df['CBSA Title'].values, 
                              index=temp_df['employment_seriesID']).to_dict()
    
    # Update main dictionary
    msa_code_dict[i] = temp_dict

# Step 2: Get the most recent BLS data at the city level

# Get data and save it into a 
# dictionary to later turn into a dataframe
new_dict = {
    'msa_code': [],
    'msa_name': [],
    'state_code': [],
    'year': [],
    'month': [],
    'date': [],
    'value': [],
    'series_id': []
}

# Print a helpful statement
print("Beginning the download process. This usually takes between 5-10 minutes.")

# Loop through the seriesID dictionary
for num in msa_code_dict:
    
    # Get the dictionary
    series_dict = msa_code_dict[num]

    # Get the series IDs
    series_ids = list(series_dict.keys())

    # Get current year
    end_year = datetime.now().year

    # Get start year
    start_year = end_year - 9

    # Get the data from the API
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": series_ids,"startyear":str(start_year), 
                       "endyear":str(end_year), 'registrationkey' : bls_key,
                       "calculations":True})
    p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', 
                      data=data, headers=headers)
    json_data = json.loads(p.text)
    
    # Loop through every Series
    for series in json_data['Results']['series']:

        # Get the seriesID
        seriesID = series['seriesID']

        for row in series['data']:

            # Get the data from the JSON format
            year = row['year']
            month = row['periodName']
            value = row['value']

            # Value is in thousands, so convert to proper number
            value = float(value) * 1000

            # Create a string we cna convert to datetime
            date_string = month + " 1, " + year

            # Add data to dictionary
            new_dict['year'].append(year)
            new_dict['month'].append(month)
            new_dict['date'].append(date_string)
            new_dict['value'].append(value)

            # Now get the msa code
            msa_code = seriesID[5:10]
            
            # Get the state code
            state_code = seriesID[3:5]
            
            # Get msa_name
            msa_name = series_dict[seriesID]
            
            # Append these remaining values
            new_dict['msa_code'].append(msa_code)
            new_dict['msa_name'].append(msa_name)
            new_dict['state_code'].append(state_code)
            new_dict['series_id'].append(seriesID)
            

# Turn dictionary into a dataframe
new_df = pd.DataFrame(new_dict)

### Save raw data
create_folder("datasets/bls/raw")
new_df.to_csv("datasets/bls/raw/most_recent_bls_data.csv", index=False)
print("Newest BLS data saved to 'datasets/bls/raw/most_recent_bls_data.csv'.")

### Now work on smoothing out the COVID 
### time-series anomaly via linear interpolation

print("""\nNow working on cleaning dataset and smoothing out COVID anomaly via linear interpolation.""")

# Create helpful cleaning functions
def clean_BLS_msa_names(dataframe):
    """
    This functions standardizes the MSA names
    between the BLS dataset and the Zillow
    datasets by only taking the first city
    within a BLS-MSA with a "-" hyphenate, 
    and the first state with a "-" hyphenate.
    
    For example: 'Houston-The Woodlands-Sugar Land, TX'
    will be turned into just 'Houston, TX' and
    'Cincinnati, OH-KY-IN' will be turned into
    just 'Cincinnati, OH'.
    """
    
    df = dataframe.copy()
    
    # Get the state column
    df['state'] = df['msa_name'].str.split(', ').str[1].str.strip()
    df['state'] = df['state'].str.split('-').str[0].str.strip()
    
    # Get the first city name
    df['city'] = df['msa_name'].str.split(', ').str[0].str.strip()
    df['city'] = df['city'].str.split('-').str[0].str.strip()
    
    # Get msa name
    df['msa_name'] = df['city'] + ", " + df['state']
    
    return df

# Create interpolation function
def interpolate_smoothen(
    dataframe, 
    msa_name,
    end_date='2021-04-01',
    start_date='2020-04-01'
):
    """
    This function adds an "interpolated" column in which
    the job numbers between the start_date and end_date
    are linearly interpolated. The purpose of this function
    is to give the user a column in which the impact of
    COVID on job deceleration and acceleration has been
    smoothed.
    """
    
    smooth_jobs = dataframe.copy()
    smooth_jobs = smooth_jobs[['msa_name','date','value']]
    msa = smooth_jobs[smooth_jobs['msa_name']==msa_name].copy()

    # Create dummy column
    msa['dummy'] = msa['value']

    # Filter and replace with NaNs
    mask = (msa['date']<=end_date) & (msa['date']>=start_date)
    msa.loc[mask, 'dummy'] = np.nan

    # Use interpolation now
    msa['interpolated'] = msa['dummy'].interpolate(method='linear')

    # Drop dummy column
    msa.drop(columns=['dummy'], inplace=True)
    
    return msa

# Create main function to interpolate the dataset
def interpolate_all_msa_job_data():
    
    # Call in the original job dataset
    jobs = pd.read_csv('datasets/bls/raw/most_recent_bls_data.csv',
                   dtype={'msa_code':str, 'state_code':str})

    # Make sure the date column is in datetime format
    jobs['date'] = pd.to_datetime(jobs['date'])

    # Replace NECTA Division
    jobs['msa_name'] = jobs['msa_name'].apply(lambda x: x.replace(" NECTA Division",""))
    jobs['msa_name'] = jobs['msa_name'].apply(lambda x: x.replace(" NECTA",""))

    # Standardize MSA names
    jobs = clean_BLS_msa_names(jobs)
    
    # Add new column
    jobs['interpolated'] = np.nan
    
    # Loop through all MSAs
    for msa in jobs['msa_name'].unique():
    
        # Smoothen the data at the MSA level
        smooth_msa = interpolate_smoothen(
            dataframe=jobs, 
            msa_name=msa,
            end_date='2022-04-01',
            start_date='2020-04-01')
        
        # Create a mask
        mask = jobs['msa_name'] == msa

        # Add the interpolated data to the original jobs dataframe
        jobs.loc[mask, 'interpolated'] = smooth_msa['interpolated']
    
    # Create folder to save covid-smoothed dataset
    folder_name = 'datasets/bls/smoothed'
    create_folder(folder_name)
    
    # Create filename
    filename = f"{folder_name}/most_recent_bls_covid_smoothed.csv"
    
    # Save file
    jobs.to_csv(filename, index=False)

# Now run the interpolation function
interpolate_all_msa_job_data()

print("\nInterpolated BLS data saved to 'datasets/bls/smoothed/most_recent_bls_covid_smoothed.csv'.")


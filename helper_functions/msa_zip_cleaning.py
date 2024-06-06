import pandas as pd
import os
from sklearn.linear_model import LinearRegression

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


def clean_zillow_dataset(dataframe):

    # Make a copy
    df = dataframe.copy()

    # Drop unnecessary columns
    df = df.drop(columns=['RegionID','SizeRank','RegionType','StateName'])

    # Set index
    df = df.set_index(['RegionName'])

    # Stack
    df = df.stack()

    # Turn into dataframe
    df = pd.DataFrame(df).reset_index()

    # Rename the post-stacked columns
    df.rename(columns={
        'RegionName':'msa_name',
        'level_1':'date', 
        0:'value'
    }, inplace=True)

    # Make date column datetime
    df['date'] = pd.to_datetime(df['date'])

    # Make it beginning of month
    df['date'] = df['date'].astype("datetime64[M]")
    
    # Create year column
    df['year'] = df['date'].dt.year
    
    return df


# Define helper function to create directory
def create_folder(the_path):
    if not os.path.isdir(the_path):
        os.mkdir(the_path)


# Turn into datetime format
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


# Turn into datetime format for monthly datasets
def turn_df_into_datetime_monthly(dataframe, 
                          columns_as_index: list):
    """
    Turns a dataframe created by the API functions
    into a tidy datetime format.
    """
    # Make a copy
    df = dataframe.copy()
    
    # Set index
    df = df.set_index(columns_as_index)
    
    # Stack
    df = df.stack()
    
    # Turn into dataframe
    df = pd.DataFrame(df).reset_index()
    
    # Rename the post-stacked columns
    df.rename(columns={'level_2':'year', 0:'value'}, inplace=True)
    
    # Make date column datetime
    df['year'] = pd.to_datetime(df['year'])
    
    # Make it beginning of month
    df['year'] = df['year'].astype("datetime64[M]")
    
    return df


### Define normalizing function
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


# Define helper function that runs linear regression
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

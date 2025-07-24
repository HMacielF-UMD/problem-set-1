'''
PART 1: ETL the two datasets and save each in `data/` as .csv's
'''
# Import necessary libraries
import pandas as pd
import os

# Define the ETL function
def run_etl():
    ''' Extract, Transform, Load (ETL) the datasets and save them as .csv files
    This function loads the raw datasets from provided URLs, processes them,
    and saves them in the `data/` directory.

    Parameters:
        None
    
    Returns:
        None
    '''

    
    """    
    # Question by HMF:
    # Was I expected to transform the data in any way?
    """

    # Ensure the 'data' directory exists
    os.makedirs('data', exist_ok=True)

    # Load the datasets from the provided URLs
    pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
    arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')

    # Convert 'filing_date' to datetime and create 'arrest_date' columns
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw.filing_date)
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw.filing_date)

    # Drop the 'filing_date' column from both dataframes
    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)

    # Ensure the 'data' directory exists
    os.makedirs('data', exist_ok=True)

    # Save both data frames to `data/` -> 'pred_universe_raw.csv', 'arrest_events_raw.csv'
    pred_universe_raw.to_csv('data/pred_universe_raw.csv', index=False)
    arrest_events_raw.to_csv('data/arrest_events_raw.csv', index=False)

# Print confirmation of successful ETL
if __name__ == "__main__":
    run_etl()
    print("ETL process completed. Data saved in 'data/' directory.")
    print("Files saved: 'pred_universe_raw.csv' and 'arrest_events_raw.csv'.")

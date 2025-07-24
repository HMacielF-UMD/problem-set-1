'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- 1. Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- 2. Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- 3. Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - 3.1. Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- 4 Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - 4.1 Use a print statment to print this question and its answer: What share of current charges are felonies?
- 5. Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - 5.1 Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- 6. Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- 7. Print pred_universe.head()
- 8. Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# Import necessary libraries
import pandas as pd

def run_preprocessing():
    ''' Pre-process the data for further analysis
    This function loads the raw datasets, performs necessary transformations,
    and creates the target and feature variables needed for modeling.

    Parameters:
        None
    
    Returns:
        pd.DataFrame: A DataFrame containing the processed arrest data with target and features.
    '''
    
    # 1. Load the datasets
    pred_universe = pd.read_csv('data/pred_universe_raw.csv')
    arrest_events = pd.read_csv('data/arrest_events_raw.csv')

    # Convert arrest_date columns to datetime
    pred_universe['arrest_date_univ'] = pd.to_datetime(pred_universe['arrest_date_univ'])
    arrest_events['arrest_date_event'] = pd.to_datetime(arrest_events['arrest_date_event'])

    # 2. Full outer join on 'person_id'
    df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer')

    # Create target variable `y`
    def check_rearrest(row):
        arrest_date = row['arrest_date_event']
        person_id = row['person_id']
        
        if pd.isnull(arrest_date):
            return 0
        
        rearrests = arrest_events[
            (arrest_events['person_id'] == person_id) &
            (arrest_events['arrest_date_event'] > arrest_date) &
            (arrest_events['arrest_date_event'] <= arrest_date + pd.Timedelta(days=365)) &
            (arrest_events['charge_degree'] == 'felony')
        ]
        
        return int(not rearrests.empty)

    # 3 Apply the function to create the 'y' column
    df_arrests['y'] = df_arrests.apply(check_rearrest, axis=1)

    # Calculate and print share of rearrested individuals
    share_rearrested = df_arrests['y'].mean()

    # 3.1 Print the share of arrestees rearrested for a felony crime in the next year
    print(f"What share of arrestees were rearrested for a felony crime in the next year? {share_rearrested:.2%}")

    # 4. Create predictive feature: is current charge a felony?
    df_arrests['current_charge_felony'] = df_arrests['charge_degree'].apply(lambda x: 1 if x == 'felony' else 0)
    
    # Print share of current charges that are felonies
    share_felony_charges = df_arrests['current_charge_felony'].mean()
    
    # 4.1 Print the share of current charges that are felonies
    print(f"What share of current charges are felonies? {share_felony_charges:.2%}")

    # 5. Create predictive feature: number of felony arrests in the year before current arrest
    def count_felony_arrests_last_year(row):
        arrest_date = row['arrest_date_event']
        person_id = row['person_id']
        
        if pd.isnull(arrest_date):
            return 0 
        
        felony_arrests = arrest_events[
            (arrest_events['person_id'] == person_id) &
            (arrest_events['arrest_date_event'] < arrest_date) &
            (arrest_events['arrest_date_event'] >= arrest_date - pd.Timedelta(days=365)) &
            (arrest_events['charge_degree'] == 'felony')
        ]
        
        return len(felony_arrests)

    # 5. Apply the function to create the 'num_fel_arrests_last_year' column
    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(count_felony_arrests_last_year, axis=1)

    # 5.1 Print average number of felony arrests in the last year
    average_felony_arrests_last_year = df_arrests['num_fel_arrests_last_year'].mean()
    print(f"What is the average number of felony arrests in the last year? {average_felony_arrests_last_year:.2f}")

    """
    # 6. I am not sure why we would call pred_universe['num_fel_arrests_last_year'].mean()
    # pred_universe does not have 'num_fel_arrests_last_year' column 
    # 7. Did actually mean df_arrests.head() instead of pred_universe.head()?
    """

    # 7. Show first few rows
    print(pred_universe.head())

    # 8. Save for use in main.py
    df_arrests.to_csv('data/df_arrests.csv', index=False)

# Run the preprocessing function if this script is executed directly
if __name__ == "__main__":
    df = run_preprocessing()
    print("Preprocessing completed and data saved to 'data/df_arrests.csv'.")
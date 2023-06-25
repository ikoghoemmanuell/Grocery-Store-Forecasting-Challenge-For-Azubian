import numpy as np

# Define the getDateFeatures() function
def getDateFeatures(df):
    df['holiday_type'] = 'Workday'
    df['is_holiday'] = False
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofmonth'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.weekofyear

    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
    # Extract the 'year' and 'weekofyear' components from the 'date' column
    df['year_weekofyear'] = df['date'].dt.year * 100 + df['date'].dt.weekofyear

    # create new coolumns to represent the cyclic nature of a year
    df['dayofyear'] = df['date'].dt.dayofyear
    df["sin(dayofyear)"] = np.sin(df["dayofyear"])
    df["cos(dayofyear)"] = np.cos(df["dayofyear"])

    df["is_weekend"] = np.where(df['dayofweek'] > 4, 1, 0)

    # Define the criteria for each season
    seasons = {'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Autumn': [9, 10, 11]}

    # Create the 'season' column based on the 'date' column
    df['season'] = df['month'].map({month: season for season, months in seasons.items() for month in months})
    

    return df
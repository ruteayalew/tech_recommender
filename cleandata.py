# CLEAN DATA MODULE **********************************************************************
import os
import numpy as np
import pandas as pd

def clean_data(df, y_train, time_stamp_present, time_stamp_col, target):

    dfcopy = df
    dfcopy[target] = y_train
    if time_stamp_present == True:
        dfcopy = dfcopy.drop([time_stamp_col],axis=1)

    df_dropped = drop_duplicates(df=dfcopy)
    df_no_null = drop_null(df=df_dropped)
    df_cleaned = drop_out_of_domain(df=df_no_null)

    return df_cleaned

def drop_duplicates(df):
    # Drop duplicate rows
    print('\nDuplicate row removal:')
    print('Sample count before: ', len(df.index))
    df_no_duplicates = df.drop_duplicates()
    print('Sample count after: ', len(df_no_duplicates.index))

    return df_no_duplicates

def drop_null(df):
     # Drop rows with null values
    df_no_nulls = df.dropna()

    print('\nNull row removal:')
    print('Sample count before: ', len(df.index))
    print('Sample count after: ', len(df_no_nulls.index))

    return df_no_nulls

def drop_out_of_domain(df, std = 2):
    # Get only numeric data to identify rows with out-of-domain properties
    df_numeric = numeric_only(df)

    # Calculate the mean and standard deviation of all row means
    print('\nOut-of-domain row removal:')
    all_rows_mean = df_numeric.mean(axis=1)
    all_rows_mean_mean = all_rows_mean.mean()
    all_rows_mean_std = all_rows_mean.std()
    threshold_std = std
    threshold = all_rows_mean_mean + threshold_std * all_rows_mean_std
    print('Threshold =', threshold_std, ' standard deviations')

    out_of_domain_indices = []

    # Iterate over rows and check for out-of-domain properties
    for idx, row in df_numeric.iterrows():
        row_mean = row.mean()
        if row_mean > threshold:
            out_of_domain_indices.append(idx)
            #print(f"Row {idx} has out-of-domain properties.")

    df_reduced = df.drop(out_of_domain_indices)
    print('Number of rows with out-of-domain properties: ',len(out_of_domain_indices))
    print('Sample count before: ', len(df.index))
    print('Sample count after: ', len(df_reduced.index))

    return df_reduced

def numeric_only(df):
    # Select numeric columns
    numeric_columns = df.select_dtypes(include='number')

    # Create a copy with only numeric columns
    df_numeric = numeric_columns.copy()

    return df_numeric

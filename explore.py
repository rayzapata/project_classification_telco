# Z0096

import pandas as pd
import numpy as np

from acquire import get_data
from prepare import encode, impute_mean, split_df


#################### Explore telco_churn Data ####################


# assign list of all columns for DataFrame
cols = [
    # customer demographics
    'is_female',
    'is_senior',
    'has_partner',
    'has_dependent',
    # phone service status
    'has_phone',
    'one_line',
    'multiple_lines',
    # internet service status
    'has_internet',
    'dsl',
    'fiber',
    # internet options
    'streaming_tv',
    'streaming_movies',
    'online_security',
    'online_backup',
    'device_protection',
    'tech_support',
    # service charges
    'monthly_charges',
    'total_charges',
    # payment information
    'mailed_check',
    'electronic_check',
    'bank_transfer',
    'credit_card',
    'paperless_billing',
    'autopay',
    # subscription information
    'no_contract',
    'tenure'
]


def explore_data(columns=cols, cache=False):
    '''

    Create a basic DataFrame for purposes of exploration

    columns=cols default behavior, pass list of columns to specify only
    certain columns, otherwise all columns are retained

    cache=False default behavior, set true to force write new CSV
    file, otherwise cached version is used

    '''

    # read in data to DataFrame
    df = get_data(cache=cache)
    # fill missing values in total_charges
    df = impute_mean(df)
    # set boolean values for true/false columns
    df['one_line'] = np.where(df.multiple_lines == 'No', 1, 0)
    df['dsl'] = np.where(df.internet_service_type_id == 1, 1, 0)
    df['mailed_check'] = np.where(df.payment_type_id == 2, 1, 0)
    df['no_contract'] = np.where(df.contract_type_id == 1, 1, 0)
    df = encode(df)
    # set desired or default DataFrame columns
    df = pd.concat((df[columns], df['churn']), axis=1)
    # obtain training dataset for exploration
    subset, _, _, = split_df(df)

    print(f'''
             Data Processing Complete
    +----------------------------------------+
    |   Source DataFrame Shape : {df.shape[0]} x {df.shape[1]:<5}|
    |   Subset DataFrame Shape : {subset.shape[0]} x {subset.shape[1]:<5}|
    |     Data Percentage Used : {subset.shape[0] / df.shape[0]:<12.2%}|
    +----------------------------------------+   
        ''')

    return subset

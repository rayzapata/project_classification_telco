# Z0096

import pandas as pd
import numpy as np

from acquire import get_data
from prepare import impute_mean


#################### Explore telco_churn Data ####################


# set list of columns for base DataFrame with all data
cols = [

    # customer demographics
    'gender',
    'is_senior',
    'has_partner',
    'has_dependent',

    # service types
    'phone_service_type',
    'internet_service_type',

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
    'payment_type',
    'paperless_billing',
    'autopay',

    # subscription information
    'contract_type',
    'tenure',
    'churn'

]


def encode(df):
    '''

    Set yes/no columns to hold boolean values

    Used in conjunction with explore_df function

    '''

    # assign boolean values to yes/no columns
    df['has_partner'] = np.where(df.partner == 'Yes', 1, 0)
    df['has_dependent'] = np.where(df.dependents == 'Yes', 1, 0)
    df['streaming_tv'] = np.where(df.streaming_tv == 'Yes', 1, 0)
    df['streaming_movies'] = np.where(df.streaming_movies == 'Yes', 1, 0)
    df['online_security'] = np.where(df.online_security == 'Yes', 1, 0)
    df['online_backup'] = np.where(df.online_backup == 'Yes', 1, 0)
    df['device_protection'] = np.where(df.device_protection == 'Yes', 1, 0)
    df['tech_support'] = np.where(df.tech_support == 'Yes', 1, 0)
    df['paperless_billing'] = np.where(df.paperless_billing == 1, 0, 1)
    df['autopay'] = np.where(df.payment_type.str.contains('auto') == True, 1, 0)
    df['churn'] = np.where(df.churn == 'Yes', 1, 0)

    return df


def rename_cols(df):
    '''

    Rename columns for appropriate data context and clarity of data
    contained in columns

    Used in conjunction with explore_df function

    '''

    # rename columns to match data context
    df = df.rename(columns={'senior_citizen':'is_senior',
        'multiple_lines':'phone_service_type'})

    # rename values for service types for clarity
    df['phone_service_type'] = df.phone_service_type.replace(
        ['Yes', 'No', 'No phone service'],
        ['Multiple Lines', 'Single Line', 'None'])
    df['internet_service_type'] = df.internet_service_type.replace(
        ['Fiber optic'], ['Fiber'])

    return df


def explore_df(columns=cols, cache=False):
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
    df = encode(df)
    # rename columns
    df = rename_cols(df)
    # set desired or default DataFrame columns
    df = df[columns]

    return df

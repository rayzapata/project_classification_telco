# Z0096

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from acquire import get_data


#################### Prep telco_churn Data ####################


# set list of columns for base DataFrame with all data
cols = [

    # customer demographics
    'is_female',
    'is_senior',
    'has_partner',
    'has_dependent',

    # phone service status
    'has_phone',
    'multiple_lines',

    # internet service status
    'has_internet',
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
    'electronic_check',
    'bank_transfer',
    'credit_card',
    'paperless_billing',

    # subscription information
    'one_year_contract',
    'two_year_contract',
    'tenure',
    'churn'
]


def encode(df):
    '''

    Set yes/no columns to hold boolean values and create new columns to
    hold encoded data as boolean values

    Used in conjunction with base_prep function

    '''

    df['is_female'] = np.where(df.gender == 'Female', 1, 0)
    df['has_partner'] = np.where(df.partner == 'Yes', 1, 0)
    df['has_dependent'] = np.where(df.dependents == 'Yes', 1, 0)

    df['has_phone'] = np.where(df.phone_service == 'Yes', 1, 0)
    df['multiple_lines'] = np.where(df.multiple_lines == 'Yes', 1, 0)

    df['has_internet'] = np.where(df.internet_service_type_id == 3, 0, 1)
    df['fiber'] = np.where(df.internet_service_type_id == 2, 1, 0)

    df['streaming_tv'] = np.where(df.streaming_tv == 'Yes', 1, 0)
    df['streaming_movies'] = np.where(df.streaming_movies == 'Yes', 1, 0)
    df['online_security'] = np.where(df.online_security == 'Yes', 1, 0)
    df['online_backup'] = np.where(df.online_backup == 'Yes', 1, 0)
    df['device_protection'] = np.where(df.device_protection == 'Yes', 1, 0)
    df['tech_support'] = np.where(df.tech_support == 'Yes', 1, 0)

    df['one_year_contract'] = np.where(df.contract_type_id == 1, 0, 1)
    df['two_year_contract'] = np.where(df.contract_type_id == 3, 0, 1)

    df['electronic_check'] = np.where(df.payment_type_id == 1, 1, 0)
    df['bank_transfer'] = np.where(df.payment_type_id == 3, 1, 0)
    df['credit_card'] = np.where(df.payment_type_id == 4, 1, 0)
    df['paperless_billing'] = np.where(df.paperless_billing == 1, 0, 1)

    df['churn'] = np.where(df.churn == 'Yes', 1, 0)

    return df


def impute_mean(df):
    '''
    
    Fill in missing values with mean from total_charges column with
    imputer

    Used in conjunction with base_prep function

    '''

    # set options to output on two decimal places
    pd.options.display.float_format = '{:.2f}'.format

    # replace whitespace strings with NaNs
    df['total_charges'] = df.total_charges.replace(' ', np.nan)

    # use SimpleImputer to fill empty values with mean of total_charges
    imp_mean = SimpleImputer(strategy='mean')
    df[['total_charges']] = imp_mean.fit_transform(df[['total_charges']])

    return df


def base_prep(cache=False):
    '''

    Create a basic prepped DataFrame that holds all data before
    splitting into filtered DataFrames for specific questions

    cache=False default behavior, set to true to force write new CSV file
    
    '''

    # read in data to DataFrame
    df = get_data(cache=cache)

    # fill missing values in total_charges
    df = impute_mean(df)

    # encode values to binary columns
    df = encode(df)

    # rename columns and drop unneccesary columns
    df = df.rename(columns={'senior_citizen':'is_senior'})
    df = df[cols]

    return df


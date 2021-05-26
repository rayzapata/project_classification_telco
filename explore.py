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

    # subscription information
    'contract_type',
    'tenure',
    'churn'

]


def base_explore(cache=False):
    '''

    Create a basic  DataFrame that holds all data before for exploration
    purposes

    cache=False default behavior, set to true to force write new CSV file

    '''

    # read in data to DataFrame
    df = get_data(cache=cache)

    # fill missing values in total_charges
    df = impute_mean(df)

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
    df['churn'] = np.where(df.churn == 'Yes', 1, 0)

    # rename columns to match data context
    df = df.rename(columns={'senior_citizen':'is_senior',
        'partner':'has_partner', 'dependents':'has_dependent',
        'multiple_lines':'phone_service_type'})

    # rename values for service types for clarity
    df['phone_service_type'] = df.phone_service_type.replace(
        ['Yes', 'No', 'No phone service'],
        ['Multiple Lines', 'Single Line', 'None'])
    df['internet_service_type'] = df.internet_service_type.replace(
        ['Fiber optic'], ['Fiber'])

    # drop unneccesary columns
    df = df[cols]

    return df



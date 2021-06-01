# Z0096

import pandas as pd
from os.path import isfile
from env import db_connect


#################### Acquire telco_churn Data ####################


def new_data():
    '''

    Uses login credentials and db_connect function from env.py to query
    Codeup database servers for all data contained in telco_churn
    database tables

    To be used in conjunction with get_data function defined in
    acquire.py

    '''

    # MySQL Query for all data
    query = '''
    SELECT *
    FROM customers
    JOIN contract_types USING(contract_type_id)
    JOIN internet_service_types USING(internet_service_type_id)
    JOIN payment_types USING(payment_type_id);'''
    # Use pandas to read into DataFrame
    df = pd.read_sql(query, db_connect('telco_churn'))

    return df


def get_data(cache=False):
    '''

    Obtains data from telco_churn database on Codeup server and checks
    if CSV cached version is stored for offline and quicker access, if
    not it creates one then reads into DataFrame

    Used in conjunction with db_connect function defined in env.py
    and new_data function defined in acquire.py

    cache=False default behavior, set to true to force write new CSV
    file

    '''

    # check if cached CSV file already exists or if forced cache true
    if cache == True or isfile('telco_churn.csv') == False:
        # read in new data into DataFrame and output to CSV file
        df = new_data()
        df.to_csv('telco_churn.csv')
    else:
        df = df = pd.read_csv('telco_churn.csv')

    return df

# Z0096

import pandas as pd
import numpy as np
from os.path import isfile

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from acquire import get_data


#################### Prepare telco_churn Data ####################


# assign list of all columns for DataFrame
cols = [

    'customer_id',
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
    'autopay',
    # subscription information
    'no_contract',
    'tenure'
]


def encode(df):
    '''

    Set yes/no columns to hold boolean values and create new columns to
    hold encoded data as boolean values

    Used in conjunction with prep_df function

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
    df['no_contract'] = np.where(df.contract_type_id == 1, 1, 0)
    df['electronic_check'] = np.where(df.payment_type_id == 1, 1, 0)
    df['bank_transfer'] = np.where(df.payment_type_id == 3, 1, 0)
    df['credit_card'] = np.where(df.payment_type_id == 4, 1, 0)
    df['paperless_billing'] = np.where(df.paperless_billing == 'Yes', 1, 0)
    df['autopay'] = np.where(df.payment_type.str.contains('auto') == True, 1, 0)
    df['churn'] = np.where(df.churn == 'Yes', 1, 0)
    # rename senior_citizen for clarity of value context
    df = df.rename(columns={'senior_citizen':'is_senior'})

    return df


def impute_mean(df):
    '''
    
    Fill in missing values with mean from total_charges column with
    imputer

    Used in conjunction with prep_df function

    '''

    # set options to output on two decimal places
    pd.options.display.float_format = '{:.2f}'.format
    # replace whitespace strings with NaNs
    df['total_charges'] = df.total_charges.replace(' ', np.nan)
    # use SimpleImputer to fill empty values with mean of total_charges
    imp_mean = SimpleImputer(strategy='mean')
    df[['total_charges']] = imp_mean.fit_transform(df[['total_charges']])

    return df


def split_df(df):
    '''

    Splits DataFrame into train, validate, and test DataFrames for 
    model creation and validation

    Uses approximately 60% of data for training, 15% to validate, and
    25% for an adequate size test dataset

    Used in conjunction with prep_df function
    '''

    # split data into train, validate, and test DataFrames
    train_validate, test = train_test_split(df, test_size=0.2,
        random_state=19, stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=0.25,
        random_state=19, stratify=train_validate.churn)

    return train, validate, test


def separate_x(train, validate, test):
    '''

    Separates train, validate, and test into DataFrames
    containing all but the last column, churn

    Used in conjunction with prep_df function
    
    '''

    X_train = train[train.columns[0:-1]]
    X_validate = validate[validate.columns[0:-1]]
    X_test = test[test.columns[0:-1]]

    return X_train, X_validate, X_test


def separate_y(train, validate, test):
    '''

    Separates train, validate, and test into series
    containing only the last column, churn

    Used in conjunction with prep_df function
    
    '''

    y_train = train[train.columns[-1]]
    y_validate = validate[validate.columns[-1]]
    y_test = test[test.columns[-1]]

    return y_train, y_validate, y_test


def prep_data(columns=cols, cache=False):
    '''

    Creates three each of pandas DataFrames and series from the
    telco_churn data for the purpose of predictive model creation and
    validation

    Returns values X_train, y_train, X_validate, y_validate, X_test,
    and y_test

    columns=cols default behavior, pass list of columns to specify only
    certain columns, otherwise all columns are retained

    cache=False default behavior, set true to force write new CSV
    file, otherwise cached version is used
    
    '''

    # read in data to DataFrame
    df = get_data(cache=cache)
    # fill missing values in total_charges
    df = impute_mean(df)
    # encode values to binary columns
    df = encode(df)
    # set desire or default DataFrame columns
    df = pd.concat((df[columns], df['churn']), axis=1)
    # split data into three sets for train, validate, test
    train, validate, test = split_df(df)
    # separate train, validate, test into X_variable DataFrames
    X_train, X_validate, X_test = separate_x(train, validate, test)
    # separate train, validate, test into y_variable series
    y_train, y_validate, y_test = separate_y(train, validate, test)

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def pred_proba(model, X):
    '''

    Creates a DataFrame containing prediction probability for passed
    model and returns only the column for positive case churn

    Used in conjunction with get_final_report

    '''

    # convert predict_proba array into DataFrame
    proba_df = pd.DataFrame(model.predict_proba(X), columns=['retain', 'churn'])

    return proba_df.churn


def get_final_report(model, features, cache=False):
    '''

    Generates a CSV and reads into a DataFrame the passed model
    predictive probabilities, predicitons, and customer_id
    
    '''
    # check if cached CSV file already exists or if forced cache=true
    if cache == True or isfile('final_report.csv') == False:

        # read in new data into DataFrame and output to CSV file
        df = get_data(cache=cache)
        df = encode(df)
        df['probability_of_churn'] = pred_proba(model, df[features])
        df['prediction_of_churn'] = model.predict(df[features])
        # reduce DataFrame to deliverable product
        df = df[['customer_id', 'probability_of_churn', 'prediction_of_churn']]
        df.to_csv('final_report.csv', index=False)

    else:
        df = pd.read_csv('final_report.csv')

    return df

# Z0096

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import ceil


#################### Visualize telco_churn Data ####################


def big_heat(df):
    
    '''

    Use seaborn to create heatmap with coeffecient annotations to
    visualize correlation between all chosen variables


    '''
    n_vars = len(df.columns.to_list())
    # Set up large figure size for easy legibility
    plt.figure(figsize=(n_vars + 5, n_vars + 1))
    # assign pd.corr() output to variable and create a mask to remove
    # redundancy from graphic
    corr = df.corr()
    mask = np.triu(corr, k=0)
    # define custom cmap for heatmap where the darker the reds the more
    # positive and vice versa for blues
    cmap = sns.diverging_palette(h_neg=220, h_pos=13, sep=25, as_cmap=True)
    # create graphic with zero centered cmap and annotations set to one
    # significant figure
    sns.heatmap(corr, cmap=cmap, center=0, annot=True, fmt=".1g", square=True,
                mask=mask, cbar_kws={
                                     'shrink':0.5,
                                     'aspect':50,
                                     'use_gridspec':False,
                                     'anchor':(-0.75,0.75)
                                      })
    # format xticks for improved legibility and clarity
    plt.xticks(ha='right', va='top', rotation=35, rotation_mode='anchor')
    plt.title('Correlation Heatmap of Selected Variables')
    plt.show()


def heater(df):
    '''

    Creates heatmap with annotated coefficients of all current 
    DataFrame variables relative to target variable 'churn'

    Darker Reds indicate stronger positive
    Darker Blues indicate stronger negative

    '''

    # define variable for corr matrix
    heat_churn = df.corr()['churn'][:-1]
    # set figure size
    fig, ax = plt.subplots(figsize=(30, 1))
    # define cmap for chosen color palette
    cmap = sns.diverging_palette(h_neg=220, h_pos=13, sep=25, as_cmap=True)
    # plot matrix turned to DataFrame
    sns.heatmap(heat_churn.to_frame().T, cmap=cmap, center=0,
                annot=True, fmt=".1g", cbar=False, square=True)
    #  improve readability of xticks, remove churn ytick
    plt.xticks(ha='right', va='top', rotation=35, rotation_mode='anchor')
    plt.yticks(ticks=[])
    # set title and print graphic
    plt.title('Correlation to Churn\n')
    plt.show()


def hist_vars(df):
    '''

    Creates figure and subplots of seaborn histplots for each variable in
    DataFrame with a hue for churn

    Figure size and rows automatically set based on number of variables chosen

    '''

    # set number of n_rows, n_cols, and n_plot
    n_cols = 4
    n_rows = ceil(len(df.columns[:-1].to_list()) / n_cols)
    n_plot = 0
    # set figure size based on number of plots
    plt.figure(figsize=((n_cols * 7), (n_rows * 6)))
    # loop for each column in DataFrame to create histplot
    for col in df.columns[:-1]:
        n_plot = n_plot + 1
        plt.subplot(n_rows, n_cols, n_plot)
        if len(df[col].value_counts()) == 2:
            sns.histplot(data=df, x=df[col], hue=df.churn)
            plt.xticks(ticks=[0,1], labels=[False, True])
        else:
            sns.histplot(data=df, x=df[col], hue=df.churn)
        plt.legend(['Churn', ' Retain'], bbox_to_anchor=(.7,1))
    plt.show()


def internet_violin(df):
    '''
    '''

    first_year = df[df.tenure <= 12]
    first_year_count = first_year.shape[0]
    
    post_year = df[df.tenure > 12]
    post_year_count = post_year.shape[0]

    first_year_net = first_year[first_year.has_internet == True]
    post_year_net = post_year[post_year.has_internet == True]

    plt.figure(figsize=(28, 14))

    plt.subplot(2, 1, 1)
    sns.violinplot(x=first_year_net['fiber'],
        y=first_year_net['monthly_charges'], hue=first_year_net['churn'],
        linewidth=5)
    
    plt.xticks(ticks=[])
    plt.xlabel('')
    plt.ylabel('\nMonthly Charges $(USD)$\n')
    plt.ylim((0,125))
    plt.title(f'First Year Customers : {first_year_count}\n')
    
    plt.subplot(2, 1, 2)
    sns.violinplot(x=post_year_net['fiber'],
        y=post_year_net['monthly_charges'], hue=post_year_net['churn'],
        linewidth=5)
    
    plt.xticks(ticks=[0,1], labels=['DSL', 'Fiber'])
    plt.xlabel('Internet Service Type')
    plt.ylabel('\nMonthly Charges $(USD)$\n')
    plt.title(f'Post Year Customers : {post_year_count}\n')
    plt.legend([])
    plt.ylim((0,125))
    
    plt.suptitle('               Monthly Charges by Internet Service Type\n')
    plt.tight_layout()
    plt.show()


def internet_breakdown(df):
    '''
    '''

    first_year = df[df.tenure <= 12]
    first_year_count = first_year.shape[0]
    
    post_year = df[df.tenure > 12]
    post_year_count = post_year.shape[0]

    first_year_net = first_year[first_year.has_internet == True]
    post_year_net = post_year[post_year.has_internet == True]

    net_cust = df[df.has_internet == 1].shape[0]
    
    year_net = first_year_net.shape[0]
    post_year = post_year_net.shape[0]
    
    year_dsl = first_year_net[first_year_net.dsl == 1].shape[0]
    year_dsl_churn = first_year_net[(first_year_net.dsl == 1) &
                                    (first_year_net.churn == 1)].shape[0]
    
    year_fiber = first_year_net[first_year_net.fiber == 1].shape[0]
    year_fiber_churn = first_year_net[(first_year_net.fiber == 1) &
                                    (first_year_net.churn == 1)].shape[0]
    
    post_dsl = post_year_net[post_year_net.dsl == 1].shape[0]
    post_dsl_churn = post_year_net[(post_year_net.dsl == 1) &
                                    (post_year_net.churn == 1)].shape[0]
    
    post_fiber = post_year_net[post_year_net.fiber == 1].shape[0]
    post_fiber_churn = post_year_net[(post_year_net.fiber == 1) &
                                    (post_year_net.churn == 1)].shape[0]
    
    print(f'''
      Total Internet Customers: {net_cust}

+ ------------------------------------------ +
|                                            |
|   First Year Total Customers: {year_net:<13}|
|                                            |
|                DSL Customers: {year_dsl:<13}|
|                      Churned: {year_dsl_churn:<13}|
|                      Percent: {(year_dsl_churn / year_dsl):<13.2%}|
|                                            |
|              Fiber Customers: {year_fiber:<13}|
|                      Churned: {year_fiber_churn:<13}|
|                      Percent: {(year_fiber_churn / year_fiber):<13.2%}|
|                                            |
|                                            |
|    Post Year Total Customers: {post_year:<13}|
|                                            |
|                DSL Customers: {post_dsl:<13}|
|                      Churned: {post_dsl_churn:<13}|
|                      Percent: {(post_dsl_churn / post_dsl):<13.2%}|
|                                            |
|              Fiber Customers: {post_fiber:<13}|
|                      Churned: {post_fiber_churn:<13}|
|                      Percent: {(post_fiber_churn / post_fiber):<13.2%}|
|                                            |
+ ------------------------------------------ +
''')


def internet_contract_compare(df):
    '''
    '''
    first_year = df[(df.fiber == 1) & (df.tenure <=12)]
    post_year = df[(df.fiber == 1) & (df.tenure > 12)]

    first_year_churn = first_year[first_year.churn == 1].no_contract.value_counts()
    x = (first_year_churn[1] / first_year_churn.sum())
    first_year_retain = first_year[first_year.churn == 0].no_contract.value_counts()
    y = (first_year_retain[1] / first_year_retain.sum())

    post_year_churn = post_year[post_year.churn == 1].no_contract.value_counts()
    x2 = (post_year_churn[1] / post_year_churn.sum())
    post_year_retain = post_year[post_year.churn == 0].no_contract.value_counts()
    y2 = (post_year_retain[1] / post_year_retain.sum())

    total_fiber = df[df.fiber == 1]
    total_contract = total_fiber[total_fiber.no_contract == 0]
    contract_percent = total_contract.shape[0] / total_fiber.shape[0]

    print(f'''
        Fiber Customer Comparisons
+----------------------------------------+
|                                        |
|     First Year Customers: {first_year.shape[0]:<13}|
|                                        |
|                  Churned: {first_year_churn.sum():<13}|
|             Mean Charges: ${first_year[first_year.churn == 1].monthly_charges.mean():<12.2f}|
|           Under Contract: {1 - x:<13.2%}|
|                                        |
|                 Retained: {first_year_retain.sum():<13}|
|             Mean Charges: ${first_year[first_year.churn == 0].monthly_charges.mean():<12.2f}|
|         Under Contracted: {1 - y:<13.2%}|
|                                        |
|      Post Year Customers: {post_year.shape[0]:<13}|
|                                        |
|                  Churned: {post_year_churn.sum():<13}|
|             Mean Charges: ${post_year[post_year.churn == 1].monthly_charges.mean():<12.2f}|
|           Under Contract: {1 - x2:<13.2%}|
|                                        |
|                 Retained: {post_year_retain.sum():<13}|
|             Mean Charges: ${post_year[post_year.churn == 0].monthly_charges.mean():<12.2f}|
|         Under Contracted: {1 - y2:<13.2%}|
|                                        |
|     Total Under Contract: {contract_percent:<13.2%}|
|                                        |
+----------------------------------------+
''')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#################### Visualize telco_churn ####################


def big_heat(df):
    
    '''

    Use seaborn to create heatmap with coeffecient annotations to
    visualize correlation between all chosen variables


    '''
    
    # Set up large figure size for easy legibility
    plt.figure(figsize=(35,30))

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

    plt.title('Correlation Heatmap of All Variables')
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

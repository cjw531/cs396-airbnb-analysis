'''
Encoding utilities for ML features
'''

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

def y_binarization(raw_arr):
    '''
    Input: t/f values in array
    Output: maped t/f -> 1/0 array
    '''
    mapped = []
    for i in raw_arr:
        if (i == 't'): mapped.append(1)
        else: mapped.append(0)
    
    return mapped

def df_binarization(df, label):
    '''
    Input: t/f values in df column
    Output: df mapped t/f -> 1/0 array
    '''
    df[label] = (df[label] == 't').astype(int)
    return df

def str_to_int(df, label):
    ''''
    Manual way of one hot encoding
    Input: df columns with 2+ unique number of strings
    Output: strings are mapped into int values
    '''
    unique = df[label].unique()
    for i in range(len(unique)):
        df.replace(unique[i], i, inplace=True)
    return df

def host_year_only(df):
    '''
    Input: dataframe
    Output: host since yyyy-mm-dd -> yyyy
    '''
    df['host_since'] = df['host_since'].str.split("-").str[0]
    return df

def host_nth_year(df):
    '''
    Input: dataframe
    Output: host since yyyy-mm-dd -> nth year being host
    '''
    df['host_since'] = df['host_since'].str.split("-").str[0]
    df['host_since'] = df['host_since'].astype(int)
    df['host_since'] = 2021 - df['host_since']
    return df

def draw_matrix(y_test, y_pred, title, labels):
    cm = confusion_matrix(y_test, y_pred)
    figure, ax = plot_confusion_matrix(conf_mat = cm,
                                    class_names = labels,
                                    show_absolute = False,
                                    show_normed = True,
                                    colorbar = True)
    plt.title(title)
    plt.rcParams.update({'font.size': 14})
    return plt
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:11:33 2020

@author: Andrew
"""
import numpy as np
import pandas as pd

def CSV2VoI(raw_file='data/recordings/test1.csv', VoI_file='VoI.txt', target_fps=25):
    """Turns a csv file of raw leap data into a pandas df containing gesture + variables of interest
    
    Attributes:
    raw_file -- str, giving the path/name of the leap motion data
    VoI_file -- str, giving the path/name of the txt file, with a variable of interest for each line
    target_fps -- int, output fps. Every nth frame is taken to acheive this, n is calculated using average fps of file

    Note:
    The VoI txt file shouldn't reference handedness for each of its chosen variables, or contain any
    variables that won't work with 'left_' or 'right_' appended to the start of them.
    Assumes that 'gesture' is a column name, indicating the gesture being performed.
    The error thrown when VoI contains an invalid name does not specify which name is invalid. This is annoying!

    """
    # get the raw leap data from a csv file
    with open(raw_file, 'r') as f:
        raw = pd.read_csv(f)

    # get the variables of interest
    with open(VoI_file, 'r') as f:
        VoI = f.read()
    VoI = VoI.split()

    # get the average frame rate of the file
    mean_fps = raw['currentFrameRate'].mean()
    # get number of frames to skip
    skip = round(mean_fps / target_fps)
    # replace raw df with skipped frame version
    raw = raw.iloc[::skip,:]

    print(f'mean fps: {mean_fps:.2f}')
    print(f'target fps: {target_fps}')
    print(f'skipping every {skip} frames')
    

    ### get df with VoI only

    # make list of variables to extract
    VoI_list = ['gesture']

    # check there is data for the left hand, before adding left hand variables to VoI_list
    left, right = False, False
    if len(raw.filter(regex='left').columns) != 0:
        left = True
        VoI_list += ['left_' + v for v in VoI]
    # likewise, for right
    if len(raw.filter(regex='right').columns) != 0:
        right = True
        VoI_list += ['right_' + v for v in VoI]

    df = raw[::][VoI_list]
    
    # add variable indicating which hands are active, using first variable
    # assumes that missing values have been filled by NAs
    if left == True:
        df['left_active'] = df['left_' + VoI[0]].isna()
    if right == True:
        df['right_active'] = df['right_' + VoI[0]].isna()
    
    print('Found left hand data: ', left)
    print('Found right hand data: ', right)

    return df


def split2examples(X, n_frames):
    """splits a list of frames up into sublist of length n_frames each
    
    Arguments:
    X -- list to be split up
    n_frames -- int, length of each training example

    Note:
    Any trailing frames not long enough for a complete example will be dropped
    
    """

    return np.array([X[i:i+n_frames] for i in range(0, len(X) - len(X) % n_frames, n_frames)])


def df2X_y(df, g2idx={'no_gesture': 0, 'so_so': 1}, hand='right', standardize=True):
    """Extracts X and y from pandas data frame, drops nan rows, and normalizes variables

    Arguments:
    df -- a dataframe of leap motion capture data
    g2idx -- dict mapping gesture names to integers
    hand -- str, left or right

    Returns:
    df.values -- np array of shape (time steps, features), predictors for every time step
    y -- np array of shape (time steps), with an int label for every time step

    Note:
    Purging na rows is a bit clumsy, it results in sudden time jumps in the input.
    Ideally a single training example shouldn't contain such a jump.

    """
    
    # drop columns for other hand, drop na rows 
    len_with_na = len(df)
    # filter to gesture + variables for hand of interest. drop hand_active variable.
    df = df.filter(regex=hand+'|gesture').drop(columns=[hand + '_active']).dropna()
    print(f'dropped {len_with_na - len(df)} of {len_with_na} rows with nans')
    # extract the gesture label after dropping nans
    y = [g2idx[i] for i in df['gesture']]
    df = df.drop(columns=['gesture'])
    # perform mean normalization and scaling for unit variance
    if standardize:
        df = (df - df.mean()) / df.std()
        # get range for each variable, to check normalization:
        # print(df.min(), df.max())
    

    return df.values, np.array(y)


def synced_shuffle(x, y):
    """shuffles two numpy arrays in sync"""
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)




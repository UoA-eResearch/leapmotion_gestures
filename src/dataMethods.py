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
    The VoI file shouldn't reference handedness for each of its chosen variables.
    Assumes that 'gesture' is a column name, indicating the gesture being performed.
    Error thrown when VoI contains invalid name does not specify which name is invalid.

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
    n_frames -- length of each training example

    Note:
    Any trailing frames not long enough for a complete example will be dropped
    
    """

    return [X[i:i+n_frames] for i in range(0, len(X) - len(X) % n_frames, n_frames)]


def df2X_y(df, hand='right', standardize=True):
    """extract X and y from pandas data frame, drop nan rows, and normalize
    
    Note: purging na rows is a bit clumsy, results in sudden time jumps in the input

    """
    y = list(df['gesture'])
    # select required columns, get rid of na rows
    df = df.filter(regex=hand).drop(columns=[hand + '_active']).dropna()
    # perform mean normalization and scaling for unit variance
    if standardize:
        df = (df - df.mean()) / df.std()
    return df.values, y






#print(raw['right_armBasis_2'].head())

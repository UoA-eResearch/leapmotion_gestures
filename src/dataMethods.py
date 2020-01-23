# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:11:33 2020

@author: Andrew
"""
import numpy as np
import pandas as pd
import json

def CSV2VoI(raw_file='data/recordings/test1.csv', VoI_file='params/VoI.txt', target_fps=25):
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


def X_y2examples(X,y=[],n_frames=30):
    """splits a contiguous list of frames and labels up into single gesture examples of length n_frames
    
    Arguments:
    X -- features to be split up
    y -- labels for each frame
    n_frames -- int, length of each training example

    Returns:
    X_final -- np array of shape (examples, frames, features)
    y_final -- np array of shape (examples), using integers to indicate gestures. I.e. not one hot.

    Note:
    Any trailing frames not long enough for a complete example will be dropped
    
    """
    # # simple test case for split2examples
    # X = [[0,1],[0,2],[0,3],[0,4],[1,1],[1,2],[1,3],[1,4],[3,1],[3,2],[3,3],[3,4],[4,1],[4,2]]
    # y = [0,0,0,0,1,1,1,1,3,3,3,3,4,4]
    
    # if there are no y labels, then just return X split up into n_frames length examples
    if len(y) == 0:
        return np.array([X[i:i+n_frames] for i in range(0, len(X) - len(X) % n_frames, n_frames)])
    

    # each sublist contains two indices, for start and end of a gesture
    Xsplit = [[0]]
    # ysplit contains a label for each sublist in Xsplit
    ysplit = [y[0]]
    for i, g in enumerate(y):
        # check if this frame is a different gestre
        if g != ysplit[-1]:
            # note down i - 1 as last index of previous gesture
            Xsplit[-1].append(i-1)
            # note down i as first index of current gesture
            Xsplit.append([i])
            ysplit.append(g)
    Xsplit[-1].append(len(X)-1)
    # part 2: split up into examples
    X_final = []
    y_final = []
    for i, g in enumerate(ysplit):
        # for j in range of number of examples in this section of X
        for j in range((Xsplit[i][1] - Xsplit[i][0] + 1)//n_frames):
            example_start = Xsplit[i][0] + j * n_frames
            example_end = example_start + n_frames
            X_final.append(X[example_start:example_end])
            y_final.append(g)
    
    return np.array(X_final), np.array(y_final)




def df2X_y(df, g2idx = {'no_gesture': 0, 'so_so': 1, 'open_close': 2, 'maybe': 3}, hand='right', standardize=True):
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
    Ideally a single training example shouldn't contain such a jump, but this is likely rare.

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
        # use the dictionaries of means and stds for each variable
        with open('params/means_dict.json', 'r') as f:
            means_dict = json.load(f)
        for col in df.columns:
            df[col] = df[col] - means_dict[col]
        with open('params/stds_dict.json', 'r') as f:
            stds_dict = json.load(f)
        for col in df.columns:
            df[col] = df[col] / stds_dict[col]
        # get range for each variable, to check normalization:
    # print(df.min(), df.max())
    # need to make sure that columns are in alphabetical order, so that model training and deployment accord with one another
    df = df.reindex(sorted(df.columns), axis=1)

    return df.values, np.array(y)


def synced_shuffle(x, y):
    '''shuffles two numpy arrays in sync'''
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)

def CSV2examples(raw_file='data/recordings/test1.csv', target_fps=25,
        g2idx={'no_gesture': 0, 'so_so': 1}, n_frames=25):
    """all of the above: gets VoI, splits a CSV to X and y"""
    df = CSV2VoI(raw_file=raw_file, VoI_file='params/VoI.txt', target_fps=target_fps)
    X_contiguous, y_contiguous = df2X_y(df, g2idx)
    X, y = X_y2examples(X_contiguous, y=y_contiguous, n_frames=n_frames)
    synced_shuffle(X, y)
    return X, y


def folder2examples(folder='data/loops/', target_fps=25,
        g2idx={'no_gesture': 0, 'so_so': 1}, n_frames=25):
    '''all of the above: gets VoI, splits a folder of CSVs to X and y'''
    # create empty data frame
    df = pd.DataFrame()
    # read in all training data from folder
    for file in os.scandir(folder):
        df2 = CSV2VoI(file, target_fps=target_fps)
        df = pd.concat([df, df2], ignore_index=True)
    X_contiguous, y_contiguous = df2X_y(df, g2idx)
    X, y = X_y2examples(X_contiguous, y=y_contiguous, n_frames=n_frames)
    synced_shuffle(X, y)
    return X, y
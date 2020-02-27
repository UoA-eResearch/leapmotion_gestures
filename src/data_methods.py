# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:11:33 2020

@author: Andrew
"""
import numpy as np
import pandas as pd
import src.features as features
import json
import os

#### methods for reading in or creating parameter files in params/

def read_ignoring_comments(filepath):
    """read in a file and return a list of lines not starting with '#'"""
    with open(filepath) as f:
        contents = f.read()
    contents = contents.split('\n')
    # filter out any blank lines or lines that are comments
    contents = [c for c in contents if c != '' and c[0] != '#']
    return contents

def get_gestures(version=2, path = 'params/'):
    """fetches gestures, and dictionaries mapping between integers and gestures"""
    gestures = read_ignoring_comments(f'{path}gesturesV{version}.txt')
    # get gesture to id dictionary
    g2idx = {g: i for i, g in enumerate(gestures)}
    # get id to gesture dictionary
    idx2g = {i: g for i, g in enumerate(gestures)}
    return gestures, g2idx, idx2g

def get_VoI(path = 'params/'):
    """fetches variables of interest from params/VoI.txt"""
    VoI = read_ignoring_comments(f'{path}VoI.txt')
    return VoI

def get_VoI_drop(path = 'params/'):
    """fetches variables to drop at prediction time from params/VoI_drop.txt"""
    VoI_drop = read_ignoring_comments(f'{path}VoI_drop.txt')
    return VoI_drop

def get_derived_feature_dict(path = 'params/'):
    """fetches two lists, one each for one and two handed features to derive"""
    feature_dict = {}
    feature_dict['one_handed'] = read_ignoring_comments(f'{path}derived_features_one_handed.txt')
    feature_dict['two_handed'] = read_ignoring_comments(f'{path}derived_features_two_handed.txt')
    return feature_dict

def create_dicts(df):
    """generates dictionaries for mean and std of a pandas df's columns, saving them to params/"""
    with open('params/means_dict.json', 'w') as f:
        json.dump(df.mean().to_dict(), f)
    with open('params/stds_dict.json', 'w') as f:
        json.dump(df.std().to_dict(), f)


#### methods for the different steps in getting training examples from a CSV file

def CSV2VoI(raw_file='data/recordings/fist_test.csv', VoI_file='params/VoI.txt', target_fps=25):
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
    VoI = get_VoI()

    # get the average frame rate of the file
    mean_fps = raw['currentFrameRate'].mean()
    # get number of frames to skip
    skip = round(mean_fps / target_fps)
    if skip == 0:
        print('WARNING: Average file frame rate is less that half the target frame rate. Taking every frame.')
        skip = 1
    # replace raw df with skipped frame version
    raw = raw.iloc[::skip,:]

    print(f'mean fps: {mean_fps:.2f}')
    print(f'target fps: {target_fps}')
    print(f'taking every {skip} frames')
    

    ### get df with VoI only

    # make list of variables to extract
    VoI_list = ['gesture']

    # check there is data for the left hand, before adding left hand variables to VoI_list
    left, right = False, False
    if len(raw.filter(regex='left').columns) != 0:
        left = True
        VoI_list += ['left_' + v for v in VoI]
        fraction_active = 1 - sum(raw['left_' + VoI[0]].isna()) / len(raw)
        print(f'{fraction_active*100:.2f}% of rows contain valid LH data')
    # likewise, for right
    if len(raw.filter(regex='right').columns) != 0:
        right = True
        VoI_list += ['right_' + v for v in VoI]
        fraction_active = 1 - sum(raw['right_' + VoI[0]].isna()) / len(raw)
        print(f'{fraction_active*100:.2f}% of rows contain valid RH data')

    df = raw[::][VoI_list]

    print('Found left hand data: ', left)
    
    print('Found right hand data: ', right)

    df.reset_index(inplace=True)

    return df


def df2X_y(df, g2idx = {'no_gesture': 0, 'so_so': 1, 'open_close': 2, 'maybe': 3}, hands=['right', 'left'],
            derive_features=True, standardize=True, dicts_gen=False, mirror=False):
    """Extracts X and y from pandas data frame, drops nan rows, and normalizes variables

    Arguments:
    df -- a dataframe of leap motion capture data
    g2idx -- dict mapping gesture names to integers
    hands -- list of hands to keep columns for
    derive_features -- bool, indicates whether or not to derive more features
    standardize -- bool, indicates whether or not to standardize and center variables
    create_dicts -- if true, new standard deviations and means dictionary are generated from the df, and saved to params/.
        needed if new features have been added to the model.
    mirror -- if true, flips data so that left hand becomes right hand and vice versa

    Returns:
    df.values -- np array of shape (time steps, features), predictors for every time step
    y -- np array of shape (time steps), with an int label for every time step

    Note:
    Purging na rows is a bit clumsy, it results in sudden time jumps in the input.
    Ideally a single training example shouldn't contain such a jump. In any case, this is likely rare.

    """
    
    # drop columns for other hand, drop na rows 
    len_with_na = len(df)
    # filter to gesture + variables for hands of interest
    df = df.filter(regex='|'.join(hands + ['gesture']))
    # filter out any rows that have a gesture not in g2idx
    allowable_gestures = [g in g2idx.keys() for g in df['gesture']]
    if False in allowable_gestures:
        n_rows = len(allowable_gestures)
        invalid_rows = n_rows - sum(allowable_gestures)
        print(f'Warning: {invalid_rows} of {n_rows} rows contain gestures not in g2idx')
        df = df[allowable_gestures]
    if len(hands) == 1:
        # if we are only interested in one hand, then at this point the df will only contain cols for that hand
        # if the other hand was active while the hand of interest wasn't, this will leave NA rows
        df.dropna(inplace=True)
    else:
        # if both hands are required, then we replace nans with the last observed valid value
        df.fillna(method='ffill', inplace=True)
        # the first rows might contain nans - remove these, then reset index
        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)

        # make sure that data for both hands is present in the dataframe
        assert df.filter(regex='left').shape[1] > 0 and df.filter(regex='right').shape[1] > 0, 'Dataframe contains columns for only one hand, but data for both is requested'
    print(f'dealt with {len_with_na - len(df)} of {len_with_na} rows with nans')

    if mirror:
        df = mirror_data(df)
        print('Data successfully mirrored')

    # at this point, we may wish to derive some more features, and drop some of the original VoI
    if derive_features:
        derived_feature_dict = get_derived_feature_dict()
        df = pd.concat([df, pd.DataFrame.from_records(df.apply(features.get_derived_features,args=(derived_feature_dict,),axis=1))], axis=1)
        for hand in hands:
            for VoI in get_VoI_drop():
                df.drop(hand + '_' + VoI, axis=1, inplace=True)
    
    # extract the gesture label after dealing with nans
    y = [g2idx[i] for i in df['gesture']]
    df = df.drop(columns=['gesture'])

    # # use create_dicts here if new derived variables have been created
    if dicts_gen:
        create_dicts(df)

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
    # make sure that columns are in alphabetical order, so that model training and deployment accord with one another
    df = df.reindex(sorted(df.columns), axis=1)
    
    return df.values, np.array(y)



def X_y2examples(X,y=[],n_frames=30, stride=None):
    """splits a contiguous list of frames and labels up into single gesture examples of length n_frames
    
    Arguments:
    X -- features to be split up
    y -- labels for each frame
    n_frames -- int, length of each training example
    stride -- int, determines how far to move along the sliding window that takes training examples, defaults to n_frames // 2

    Returns:
    X_final -- np array of shape (examples, frames, features)
    y_final -- np array of shape (examples), using integers to indicate gestures. I.e. not one hot.

    Note:
    A sliding window is used to take training examples, with stride equal to half of frame size
    Any trailing frames not long enough for a complete example will be dropped
    
    """
    # # simple test case
    # X = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[1,1],[1,2],[1,3],[1,4],[1,5],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[4,1],[4,2]]
    # y = [0,0,0,0,0,0,0,0,1,1,1,1,1,3,3,3,3,3,3,3,4,4]
    
    # if there are no y labels, then just return X split up into n_frames length examples
    if len(y) == 0:
        return np.array([X[i:i+n_frames] for i in range(0, len(X) - len(X) % n_frames, n_frames)])
    
    if stride == None:
        stride = n_frames // 2
    
    #### part 1: get the start and end indices of each gesture
    # each sublist contains two indices, for start and end
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

    #### part 2: split up into examples, using the generated indices
    X_final = []
    y_final = []
    for i, g in enumerate(ysplit):
        # iterate over what will be the end index of each training example
        # we add 2 to the upper bound because it is non inclusive (+1), but then neither is the stop index when slicing to get the example (+1 again)
        for j in range(Xsplit[i][0] + n_frames, Xsplit[i][1] + 2, stride):
            X_final.append(X[j-n_frames:j])
            y_final.append(g)

    return np.array(X_final), np.array(y_final)


def synced_shuffle(x, y):
    '''shuffles two numpy arrays in sync'''
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)


def mirror_data(df, hands=['left', 'right']):
    """takes a data frame of gesture data and generates its mirror image: RH <-> LH"""
    #### warning: this function MIGHT break if different VoI are used
    # swap around left and right variables
    df_flipped = df.copy()
    df_flipped.columns = [col.replace('left', 'lft') for col in df_flipped.columns]
    df_flipped.columns = [col.replace('right', 'left') for col in df_flipped.columns]
    df_flipped.columns = [col.replace('lft', 'right') for col in df_flipped.columns]
    df_flipped = df_flipped.apply(lambda x: -x if x.name[-1] == '0' else x, axis=0)
    return df_flipped


#### methods for combining the above together, to go straight from a CSVs to training examples

def CSV2examples(raw_file='data/recordings/test1.csv', target_fps=30,
        g2idx={'no_gesture': 0, 'so_so': 1}, hands=['left', 'right'], n_frames=25, standardize=True, dicts_gen=False, mirror=True, derive_features=True):
    """all of the above: gets VoI, and using these, splits a CSV to X and y"""
    df = CSV2VoI(raw_file=raw_file, VoI_file='params/VoI.txt', target_fps=target_fps)
    X_contiguous, y_contiguous = df2X_y(df, g2idx, hands=hands, standardize=standardize, dicts_gen=dicts_gen, mirror=False, derive_features=derive_features)
    X, y = X_y2examples(X_contiguous, y=y_contiguous, n_frames=n_frames)
    if mirror:
        X_contiguous, y_contiguous = df2X_y(df, g2idx, hands=hands, standardize=standardize, dicts_gen=dicts_gen, mirror=True, derive_features=derive_features)
        X2, y2 = X_y2examples(X_contiguous, y=y_contiguous, n_frames=n_frames)
        X = np.concatenate([X,X2])
        y = np.concatenate([y, y2])
    synced_shuffle(X, y)
    return X, y


def folder2examples(folder='data/loops/', target_fps=30,
        g2idx={'no_gesture': 0, 'so_so': 1}, hands=['left', 'right'], n_frames=25, standardize=True,
        dicts_gen=False, mirror=True, derive_features=True):
    '''all of the above: gets VoI, splits a folder of CSVs to X and y'''
    # create empty data frame
    df = pd.DataFrame()
    # read in all training data from folder
    for file in os.scandir(folder):
        print(' ')
        print(file)
        df2 = CSV2VoI(file, target_fps=target_fps)
        df = pd.concat([df, df2], ignore_index=True)
    X_contiguous, y_contiguous = df2X_y(df, g2idx, hands=hands, standardize=standardize, dicts_gen=dicts_gen, mirror=False, derive_features=derive_features)
    X, y = X_y2examples(X_contiguous, y=y_contiguous, n_frames=n_frames)
    if mirror:
        X_contiguous, y_contiguous = df2X_y(df, g2idx, hands=hands, standardize=standardize, dicts_gen=dicts_gen, mirror=True, derive_features=derive_features)
        X2, y2 = X_y2examples(X_contiguous, y=y_contiguous, n_frames=n_frames)
        X = np.concatenate([X,X2])
        y = np.concatenate([y, y2])
    synced_shuffle(X, y)
    return X, y



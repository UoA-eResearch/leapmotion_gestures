import numpy as np
import pandas as pd

def get_derived_features(frame, hands=['left', 'right']):
    """given a frame of gesture data, returns some derived features"""
    # get fingertip positions
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

    # create dictionary of the features needed for deriving new features
    features = {}
    for hand in hands:
        for i, finger in enumerate(fingers):
            features[f'{hand}_f{i+1}'] = np.array([frame[f'{hand}_{finger}_tipPosition_{j}'] for j in (0,1,2)])
    for hand in hands:
        features[f'{hand}_palm_norm'] = np.array([frame[f'{hand}_palmNormal_{i}'] for i in (0,1,2)])
        features[f'{hand}_palm_position'] = np.array([frame[f'{hand}_palmPosition_{i}'] for i in (0,1,2)])
        features[f'{hand}_wrist'] = np.array([frame[f'{hand}_wrist_{i}'] for i in (0,1,2)])
        features[f'{hand}_elbow'] = np.array([frame[f'{hand}_elbow_{i}'] for i in (0,1,2)])
        features[f'{hand}_palm_velocity'] = np.array([frame[f'{hand}_palmVelocity_{i}'] for i in (0,1,2)])

    # create dictionary of new features
    new_features = {}
    for hand in hands:
        # compute within hand features
        adjacent_finger_distances(new_features, features, hand)
        finger_palm_distances(new_features, features, hand)
        finger_palm_plain_distances(new_features, features, hand)
        wrist_angle(new_features, features, hand)
        palm_velocity(new_features, features, hand)
        
    
    if len(hands) == 2:
        # compute between hand features
        interpalm_distance(new_features, features)
        interfinger_distances(new_features, features)
        interpalm_angle(new_features, features)

    return new_features


def adjacent_finger_distances(new_features, features, hand):
    """Calculates distances between fingertips for adjacent fingers, and stores them in new_features

    Arguments:
    new_features -- dictionary to store the distances in, may contain other new features already
    fingertips -- dictionary containing fingertip locations, each location stored as a np array

    Notes:
    The expected keys for fingertip positions are f1 through to f5.
    The keys used to store fingertip distances are f1_2 through to f4_5
    
    """
    for i in range(1, 5):
        new_features[f'{hand}_f{i}_{i+1}'] = np.linalg.norm(features[f'{hand}_f{i}'] - features[f'{hand}_f{i+1}'])


def finger_palm_distances(new_features, features, hand):
    """Calculates distance to the center of the palm for each finger"""
    for i in range(1,6):
        new_features[f'{hand}_f{i}_p'] = np.linalg.norm(features[f'{hand}_palm_position'] - features[f'{hand}_f{i}'])

def finger_palm_plain_distances(new_features, features, hand):
    """Calculates distance to the palm plain for each finger"""
    for i in range(1,6):
        new_features[f'{hand}_f{i}_p_plain'] = np.dot((features[f'{hand}_f{i}'] - features[f'{hand}_palm_position']), features[f'{hand}_palm_norm'])

def wrist_angle(new_features, features, hand):
    """calculates angle of wrist flexion/extension"""
    # get vector of elbow to wrist
    wrist_direction = features[f'{hand}_wrist'] - features[f'{hand}_elbow']
    # normalize it, so it is a unit vector
    wrist_direction /= np.linalg.norm(wrist_direction)
    # take dot product with palm normal vector. These are both unit vectors, so this will give the angle between them
    new_features[f'{hand}_wrist_angle'] = np.dot(wrist_direction, features[f'{hand}_palm_norm'])

def palm_velocity(new_features, features, hand):
    """calculates the magnitude of palm velocity"""
    new_features[f'palm_velocity'] = np.linalg.norm(features[f'{hand}_palm_velocity'])

def interpalm_distance(new_features, features):
    """Calculates distance between center of each palm"""
    new_features[f'palm_distance'] = np.linalg.norm(features[f'right_palm_position'] - features[f'left_palm_position'])

def interfinger_distances(new_features, features):
    """Calculates distances between left and right hand fingers of the same type"""
    for i in range(1,6):
        new_features[f'f{i}_f{i}'] = np.linalg.norm(features[f'left_f{i}'] - features[f'right_f{i}'])

def interpalm_angle(new_features, features):
    """calculate angle between palm norms... i.e. the dot product"""
    new_features[f'palm_angle'] = np.dot(features[f'right_palm_norm'], features[f'left_palm_norm'])




    







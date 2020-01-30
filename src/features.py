import numpy as np
import pandas as pd

def get_derived_features(frame, hands=['left', 'right']):
    """gets the distance to palm for each finger tip"""
    # get fingertip positions
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    # get dictionary of fingertip coordinates (with keys as f1, f2, ..., f5, and values as np arrays)
    # create dictionary of the features needed for deriving new features
    features = {}
    for hand in hands:
        for i, finger in enumerate(fingers):
            features[f'{hand}_f{i+1}'] = np.array([frame[f'{hand}_{finger}_tipPosition_{j}'] for j in (0,1,2)])
    # old way, with list comprehension:
    # features = {f'f{i+1}': np.array([frame[f'{hand}_{finger}_tipalm_positionition_{j}'] for j in (0,1,2)]) for i, finger in enumerate(fingers)}
    for hand in hands:
        features[f'{hand}_palm_norm'] = np.array([frame[f'{hand}_palmNormal_{i}'] for i in (0,1,2)])
        features[f'{hand}_palm_position'] = np.array([frame[f'{hand}_palmPosition_{i}'] for i in (0,1,2)])

    new_features = {}
    for hand in hands:
        # compute inter-fingertip distances
        interfinger_distances(new_features, features, hand)
        # compute palm-fingertip and fingertip to palm plain distances
        # finger_palm_distances(new_features, features, hand)
        finger_palm_plain_distances(new_features, features, hand)
    
    if len(hands) == 2:
        interpalm_distance(new_features, features)

    return new_features


def interfinger_distances(new_features, features, hand):
    """Calculates interfingertip distances for adjacent fingers

    Arguments:
    new_features -- dictionary to store the distances in, may contain other new features already
    fingertips -- dictionary containing fingertip locations

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


def interpalm_distance(new_features, features):
    """Calculates distance between center of each palm"""
    new_features[f'palm_distance'] = np.linalg.norm(features[f'right_palm_position'] - features[f'left_palm_position'])


    







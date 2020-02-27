#!/usr/bin/env python3

import websocket
import config
import json
import pandas as pd
import numpy as np
from src.data_methods import get_gestures
from src.leap_methods import collect_frame
import src.features as features
import random
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", type=int, help="select which gesture file version to use", default=3)
args = parser.parse_args()

websocket_cache = {}

# get gestures
gestures, _, _ = get_gestures(version=args.version)

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# note: the modes expect no_gesture to be in first place
current_gesture = 0
change_time = time.time()
warned = False
next_gesture = 0
current_gesture = 0
# whether or not to store captured frames
record = True
# collect only every nth frame, use this for lowering frame rate
n = 4

if __name__ == "__main__":
    frames = []
    frames_total = 0
    # variables for gesture messages
    gesturing = False
    
    message = ''
    # frame at which user is notified of impending change
    notify_frame = 0
    mode = int(input('Select mode:\n1 for alternating between gestures randomly\
        \n2 for performing each gesture in succession\
        \n3 for alternating no gesture with other gestures in sequential order\
        \n4 for continuously performing a single gesture\
        \n5 for viewing variables recorded\n'))
    if mode == 1 or mode == 2 or mode == 3:
        # where are we up to in the sequence of gestures?
        seq_n = 0
        g_i = np.arange(len(gestures))
        # seconds warning to receive before performing a gesture
        warn_time = 1
        # delay between gestures
        delay = float(input('enter time length for each gesture: '))
    elif mode == 4:
        print('Available gestures:')
        for i, g in enumerate(gestures):
            print(f'{i}. {g}')
        print()
        gesture_n = int(input('Select gesture to record: '))
        gesture = gestures[gesture_n]
    elif mode == 5:
        record = False
        print_variable = input('input variable name (e.g. right_grabAngle): ')
    else:
        raise Exception('Input not a valid mode')
    try:
        while True:
            frames_total += 1

            packed_frame = collect_frame(frames_total, n, websocket_cache)

            if len(packed_frame) > 0:                
                # store variable indicating gesture
                if mode == 1 or mode == 2 or mode == 3:
                    packed_frame["gesture"] = gestures[current_gesture]
                elif mode == 4:
                    packed_frame["gesture"] = gesture
                
                if record:
                    frames.append(packed_frame)

                # change to the next gesture
                if (mode == 1 or mode == 2) and change_time < time.time():
                    current_gesture = next_gesture
                    change_time = time.time() + delay # + random.uniform(-1,1)
                    print('###### Start ' + gestures[current_gesture])
                    warned = False
                    seq_n += 1
                
                # set the next gesture, and warn user of impending change
                elif (mode == 1 or mode == 2) and change_time - 1 < time.time() and warned == False:
                    if seq_n >= len(gestures): #check that we're not out of range
                        seq_n = 0
                        if mode == 1: #randomize
                            np.random.shuffle(g_i)
                    next_gesture = g_i[seq_n]
                    print('Prepare to perform ' + gestures[next_gesture])
                    # the user has been warned
                    warned = True
                
                elif mode == 3 and change_time < time.time():
                    current_gesture = next_gesture
                    change_time = time.time() + delay # + random.uniform(-1,1) # can include a slight randomness in change time
                    print('###### Start ' + gestures[current_gesture])
                    warned = False
                    if current_gesture == 0:
                        seq_n += 1
                
                # set the next gesture, and warn user of impending change
                elif mode == 3 and change_time - warn_time < time.time() and warned == False:
                    if seq_n >= len(gestures): #check that we're not out of range
                        seq_n = 1
                    if current_gesture == 0:
                        next_gesture = g_i[seq_n]
                    else:
                        next_gesture = 0
                    print('Prepare to perform ' + gestures[next_gesture])
                    # the user has been warned
                    warned = True
                elif mode == 5:
                    if frames_total % 1 == 0:
                        try:
                            print(print_variable + ': ', packed_frame[print_variable])
                        except KeyError:
                            print(print_variable + ' not found')                    


                            
    except KeyboardInterrupt:
        if mode != 5:
            fn = input("Enter filename to save recording to: ")
            df = pd.DataFrame(frames)
            df.to_csv(f"recordings/{fn}.csv", index=False)
            print("Saved")
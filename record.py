#!/usr/bin/env python3

import websocket
import config
import json
import pandas as pd
import numpy as np
from src.dataMethods import get_gestures
import src.features as features
import random
import time

websocket_cache = {}

# get gestures
gestures, _, _ = get_gestures(version=0)

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# note: the modes expect no_gesture to be in first place
current_gesture = 0
change_time = time.time()
warned = False
next_gesture = 0
current_gesture = 0
# whether or not to store captured frames
record = True
# collect only every nth frame
n = 1

if __name__ == "__main__":
    frames = []
    frames_captured = 0
    # variables for gesture messages
    gesturing = False
    
    message = ''
    # frame at which user is notified of impending change
    notify_frame = 0
    mode = int(input('Select mode:\n1 for alternating between gestures randomly (short time per gesture)\
        \n2 for performing each gesture in succession (longer time for each gesture)\
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
        if mode == 1:
            delay = 4
        elif mode == 2:
            delay = 10
        elif mode == 3:
            delay = 2.2
    elif mode == 4:
        print('Available gestures:')
        for i, g in enumerate(gestures):
            print(f'{i}. {g}')
        print()
        gesture_n = int(input('Select gesture to record: '))
        gesture = gestures[gesture_n]
    elif mode == 5:
        record = False
    else:
        raise Exception('Input not a valid mode')
    try:
        while True:
            for i, device in enumerate(config.devices):
                frames_captured += 1
                if i not in websocket_cache:
                    ws = websocket.create_connection(device["url"])
                    if device["mode"] == "desktop":
                        ws.send(json.dumps({"optimizeHMD": False}))
                    else:
                        ws.send(json.dumps({"optimizeHMD": True}))
                    version = ws.recv()
                    print(i, version)
                    websocket_cache[i] = ws
                elif frames_captured % n != 0:
                    # collect the frame, but don't unpack it
                    resp = websocket_cache[i].recv()
                else:
                    resp = websocket_cache[i].recv()
                    if "event" in resp:
                        # connect / disconnect
                        print(i, resp)
                    else:
                        frame = json.loads(resp)
                        if frame["hands"]:
                            packed_frame = dict([(k,v) for k,v in frame.items() if type(v) in [int, float]])
                            packed_frame["device_index"] = i
                            packed_frame["device_mode"] = 0 if device["mode"] == "desktop" else 1
                            packed_frame["currentFrameRate"] /= n #adjust frame rate by the number we're actually capturing at the moment 

                            # store variable indicating gesture
                            if mode == 1 or mode == 2 or mode == 3:
                                packed_frame["gesture"] = gestures[current_gesture]
                            elif mode == 4:
                                packed_frame["gesture"] = gesture

                            for hand in frame["hands"]:
                                left_or_right = hand["type"]
                                for key, value in hand.items():
                                    if key == "type":
                                        continue
                                    if key == "armBasis":
                                        #flatten
                                        value = [item for sublist in value for item in sublist]
                                    if type(value) is list:
                                        for j, v in enumerate(value):
                                            packed_frame["_".join((left_or_right, key, str(j)))] = v
                                    else:
                                        packed_frame["_".join((left_or_right, key))] = value
                            for finger in frame["pointables"]:
                                if finger["handId"] == packed_frame.get("left_id"):
                                    left_or_right = "left"
                                elif finger["handId"] == packed_frame.get("right_id"):
                                    left_or_right = "right"
                                finger_name = FINGERS[finger["type"]]
                                for key, value in finger.items():
                                    if key == "type":
                                        continue
                                    if key == "bases":
                                        #flatten
                                        value = [item for sublist in value for subsublist in sublist for item in subsublist]
                                    if key == "extended":
                                        value = int(value)
                                    if type(value) is list:
                                        for j, v in enumerate(value):
                                            packed_frame["_".join((left_or_right, finger_name, key, str(j)))] = v
                                    else:
                                        packed_frame["_".join((left_or_right, finger_name, key))] = value
                            if record:
                                frames.append(packed_frame)

                            # if len(frames) % 300 == 0:
                            #     print(f"{len(frames)} frames captured")

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
                                if frames_captured % 80 == 0:
                                    new_features = features.get_derived_features(packed_frame, hands=['right'])
                                    new_features = {k: round(v, 1) for k, v in new_features.items()}
                                # direction = np.round(np.array([packed_frame[f'right_direction_{i}'] for i in (0,1,2)]), 1)
                                
                                    print(new_features)
                                    # print(packed_frame['right_index_tipPosition_0'] - packed_frame['right_palmPosition_0'])
                                    


                            
    except KeyboardInterrupt:
        fn = input("Enter filename to save recording to: ")
        df = pd.DataFrame(frames)
        df.to_csv(f"recordings/{fn}.csv", index=False)
        print("Saved")
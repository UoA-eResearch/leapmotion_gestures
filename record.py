#!/usr/bin/env python3

import websocket
import config
import json
import pandas as pd
import numpy as np
from src.dataMethods import get_gestures
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
        \n5 for viewing variables recorded'))
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
    try:
        while True:
            for i, device in enumerate(config.devices):
                if i not in websocket_cache:
                    ws = websocket.create_connection(device["url"])
                    if device["mode"] == "desktop":
                        ws.send(json.dumps({"optimizeHMD": False}))
                    else:
                        ws.send(json.dumps({"optimizeHMD": True}))
                    version = ws.recv()
                    print(i, version)
                    websocket_cache[i] = ws
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
                            frames_captured += 1

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
                            elif mode == 5 and frames_captured % 5 == 0:
                                lwrist0 = packed_frame['left_wrist_0']
                                lwrist1 = packed_frame['left_wrist_1']
                                lwrist2 = packed_frame['left_wrist_2']
                                rwrist0 = packed_frame['right_wrist_0']
                                rwrist1 = packed_frame['right_wrist_1']
                                rwrist2 = packed_frame['right_wrist_2']
                                dist = np.sqrt((lwrist0 - rwrist0) ** 2 + (lwrist1 - rwrist1) ** 2 + (lwrist2 - rwrist2) ** 2) 
                                print(f'left wrist: {lwrist0:.1f}, {lwrist1:.1f}, {lwrist2:.1f}, right: {rwrist0:.1f}, {rwrist1:.1f}, {rwrist2:.1f}, dist: {dist:.2f}')

                            
                                    


                            
    except KeyboardInterrupt:
        fn = input("Enter filename to save recording to: ")
        df = pd.DataFrame(frames)
        df.to_csv(f"recordings/{fn}.csv", index=False)
        print("Saved")
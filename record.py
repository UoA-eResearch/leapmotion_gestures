#!/usr/bin/env python3

import websocket
import config
import json
import pandas as pd
import numpy as np
import random
import time

websocket_cache = {}

with open('params/gesturesV2.txt') as f:
    gestures = f.read()
gestures = gestures.split()



FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# note: the modes expect no_gesture to be in first place
current_gesture = 0
change_time = time.time()
warned = False
next_gesture = 0
current_gesture = 0


if __name__ == "__main__":
    frames = []
    
    # variables for gesture messages
    gesturing = False
    
    message = ''
    # frame at which user is notified of impending change
    notify_frame = 0
    
    mode = int(input('Select mode:\n1 for alternating between gestures randomly (short time per gesture)\
        \n2 for performing each gesture in succession (longer time for each gesture)\
        \n3 for continuously performing a single gesture\n'))
    if mode == 1 or mode == 2:
        # where are we up to in the sequence of gestures?
        seq_n = 0
        g_i = np.arange(len(gestures))
        # delay between gestures
        delay = 4.5
        if mode == 2:
            delay = 15
    if mode == 3:
        print('Available gestures:')
        for i, g in enumerate(gestures):
            print(f'{i}. {g}')
        print()
        gesture_n = int(input('Select gesture to record: '))
        gesture = gestures[gesture_n]
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
                            if mode == 1 or mode == 2:
                                packed_frame["gesture"] = gestures[current_gesture]
                            else:
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
                            elif (mode == 1 or mode == 2) and change_time - 1.5 < time.time() and warned == False:
                                if seq_n >= len(gestures): #check that we're not out of range
                                    seq_n = 0
                                    if mode == 1: #randomize
                                        np.random.shuffle(g_i)
                                next_gesture = g_i[seq_n]
                                print('Prepare to perform ' + gestures[next_gesture])
                                # the user has been warned
                                warned = True
                                

                            
                                    


                            
    except KeyboardInterrupt:
        fn = input("Enter filename to save recording to: ")
        df = pd.DataFrame(frames)
        df.to_csv(f"recordings/{fn}.csv", index=False)
        print("Saved")
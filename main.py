#!/usr/bin/env python3

import websocket
import config
import json
import pandas as pd
import numpy as np
import random
import time

websocket_cache = {}

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# possibilities: 'no_gesture', 'hitchhiking', 'fistshake', 'so_so', 'open_close', 'pointing_around', 'stop', 'shuffle_over', 'come'
# note: the modes expect no_gesture to be in first place
gestures = ['no_gesture', 'so_so', 'open_close', 'shuffle_over', 'no']
current_gesture = 0
change_time = time.time()

# mapping of variable names to feature indexes
with open('VoI.txt', 'r') as f:
    VoI = f.read()
VoI = VoI.split()
VoI.sort()
v2idx = {'right_' + v: i for i, v in enumerate(VoI)}

if __name__ == "__main__":
    # predictors used
    predictors = ['left_middle_bases_12']
    # no of frames to keep stored
    keep = 100
    # initialize frame storage
    frames = np.empty((keep,len(v2idx)))
    # total number of frames received
    frames_total = 0
    # no. captured
    frames_recorded = 0
    # how often to predict (in no. of frames)
    pred_interval = 50
    
    message = ''
    # frame at which user is notified of impending change
    notify_frame = 0
    # delay between notification and change
    delay = 150
    
    mode = int(input('Enter mode:\n0 for alternating between gestures (including non gesture)\n1 for alternating between gestures (without non gesture) \n2 for single gesture\n'))
    if mode == 2:
        gesture = input('Enter gesture name: ')
    try:
        while True:
            for i, device in enumerate(config.devices):
                frames_total += 1
                if i not in websocket_cache:
                    ws = websocket.create_connection(device["url"])
                    if device["mode"] == "desktop":
                        ws.send(json.dumps({"optimizeHMD": False}))
                    else:
                        ws.send(json.dumps({"optimizeHMD": True}))
                    version = ws.recv()
                    print(i, version)
                    websocket_cache[i] = ws
                elif frames_total % 4 != 0:
                    # collect the frame, but don't unpack it
                    # not quite right...
                    resp = websocket_cache[i].recv()
                else:
                    # collect and unpack
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
                            if mode == 0 or mode == 1:
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


                            frames_recorded += 1
                            frame_index = (frames_recorded - 1) % keep
                            try:
                                # using the variable name to id mapping dict, update the np array containing the frames
                                for v, i in v2idx.items():
                                    frames[frame_index, i] = packed_frame[v]
                                # print(frames[frame_index,0])
                            except:
                                # if the above fails, then we can't record the frame
                                # the continuity of the input is broken, and we restart
                                frames_recorded = 0
                                print('bad frames')

                            # make a prediction every pred_interval number of frames
                            # but first ensure there is a complete training example's worth of consecutive frames
                            if frames_recorded >= keep and frames_recorded % pred_interval == 0:
                                example = np.concatenate((frames[frame_index:,:], frames[:frame_index,:]))
                                print('example!')

                            
    except KeyboardInterrupt:
        fn = input("Enter filename to save recording to: ")
        df = pd.DataFrame(frames)
        df.to_csv(f"recordings/{fn}.csv", index=False)
        print("Saved")
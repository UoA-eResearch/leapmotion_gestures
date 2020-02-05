#!/usr/bin/env python3

import websocket
import config
import json
import pandas as pd
import numpy as np
from src.dataMethods import *
import random
import time
import tkinter as tk
import tensorflow as tf

# dead bird from https://www.flickr.com/photos/9516941@N08/3180449008

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("GUI")
        self.gesture = tk.StringVar()
        self.gesture.set('position hand')
        self.img = tk.PhotoImage(file='data/images/dead.png')
        self.bad = tk.PhotoImage(file='data/images/dead.png')
        self.label = tk.Label(master, image=self.bad)
        self.label.pack()
        self.label2 = tk.Label(master, font=("Helvetica", 44), textvariable=self.gesture)
        self.label2.pack(side=tk.BOTTOM)
        
root = tk.Tk()
root.geometry("500x500")
gui = GUI(root)


websocket_cache = {}
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# Get the VoI that are used as predictors
VoI = get_VoI()
VoI_drop = get_VoI_drop()
VoI_predictors = [v for v in VoI if v not in VoI_drop]
# which hands will be used in predicting?
hands = ['left', 'right']
# use these to get VoI, labelled by hand
VoI_predictors = [hand + '_' + v for v in VoI_predictors for hand in hands]

# IMPORTANT: always have variables in alphabetical order
# the model expects to receive them that way
VoI.sort()
v2idx = {v: i for i, v in enumerate(VoI)}
# mapping of gestures to integers: need this for decoding model output
gestures, g2idx, idx2g = get_gestures(version=0)

# get mean and standard deviation dictionaries
# these are used for standardizing input to model
with open('params/means_dict.json', 'r') as f:
    means_dict = json.load(f)
with open('params/stds_dict.json', 'r') as f:
    stds_dict = json.load(f)

# load the prediction model
model = tf.keras.models.load_model('models/V0/40f_5hs.h5')
# no of frames to keep stored in memory for prediction
keep = model.input.shape[-2]
# how often to make a prediction (in frames)
# for now, just set to same frequency as keep
pred_interval = keep

# initialize frame storage
frames = np.empty((keep,model.input.shape[-1]))
# keep track of total no. of frames received
frames_total = 0
# no. captured continuously
# this will be reset if there is an interruption to input
frames_recorded = 0

# indicates whether or not left and right hands were present in a frame
hands_presence = {'left': False, 'right': False}
# store previous frame, just in case a hand drops out, and we need to vill in values
previous_frame = None
first_two_handed_frame = True

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

                    for hand in frame["hands"]:
                        left_or_right = hand["type"]
                        hands_presence[left_or_right] = True
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
                    # # if no hands are present, the the continuity of the input is broken, and we start again
                    # if there has been a gap in input:
                    #     frames_recorded = 0
                    #     previous_frame = None
                    #     # gui.label.configure(image=gui.bad)
                    #     # gui.gesture.set('reposition hands')
                    # if only one hand is present, and the previous frame has no data for the other hand, then we can't do anything
                    if False in hands_presence.values() and previous_frame == None:
                        print('need two hands to start')
                        gui.label.configure(image=gui.bad)
                        gui.gesture.set('position hands')
                    # if we have at least one hand, and a previous frame to supplement any missing hand data, then we can proceed
                    else:
                        # if a hand is missing, fill in the data from the previous frame
                        if False in hands_presence.values():
                            packed_frame.update(previous_frame)
                            print('Warning: a hand is missing')
                        # reset the indicators of hand presence to False
                        hands_presence = {'left': False, 'right': False}
                        previous_frame = packed_frame.copy()
                        # get the derived features
                        new_features = features.get_derived_features(packed_frame)
                        # if this is the first two handed frame, generate a sorted list of variables used for prediction, including derived features
                        if first_two_handed_frame:
                            first_two_handed_frame = False
                            all_predictors = VoI_predictors + [pred for pred in new_features.keys()]
                            all_predictors.sort()
                        
                        # add the derived features to the packed frame
                        packed_frame.update(new_features)

                        for i, p in enumerate(all_predictors):
                            frames[frame_index, i] = (packed_frame[p] - means_dict[p]) / stds_dict[p]

                        # make a prediction every pred_interval number of frames
                        # but first ensure there is a complete training example's worth of consecutive frames
                        if frames_recorded >= keep and frames_recorded % pred_interval == 0:
                            example = np.concatenate((frames[frame_index:,:], frames[:frame_index,:]))
                            # feed example into model, and get a prediction
                            pred = model.predict(np.expand_dims(example, axis=0))
                            print(pred)
                            print(idx2g[np.argmax(pred)])
                            if pred[0][np.argmax(pred)] > 0.6:
                                # gui.img = tk.PhotoImage(file=f'data/images/{idx2g[np.argmax(pred)]}.png')
                                gui.label.configure(image=gui.img)
                                gui.gesture.set(idx2g[np.argmax(pred)].replace('_', ' '))

    # update the gui
    root.update_idletasks()
    root.update()
#!/usr/bin/env python3

import websocket
import config
import json
import pandas as pd
import numpy as np
from src.dataMethods import *
from src.leapMethods import collect_frame
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
gestures, g2idx, idx2g = get_gestures(version=1)

# get mean and standard deviation dictionaries
# these are used for standardizing input to model
with open('params/means_dict.json', 'r') as f:
    means_dict = json.load(f)
with open('params/stds_dict.json', 'r') as f:
    stds_dict = json.load(f)

# load the prediction model
model = tf.keras.models.load_model('models/V1/40f_5hs.h5')
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

# store previous frame, just in case a hand drops out, and we need to vill in values
previous_frame = None
first_two_handed_frame = True

#capture every nth frame
n = 4

while True:
    frames_total += 1
    
    packed_frame = collect_frame(frames_total, n, websocket_cache)
    if len(packed_frame) > 0:
        frames_recorded += 1
        frame_index = (frames_recorded - 1) % keep
        # length of packed frame is ~350 if one hand present, ~700 for two
        if len(packed_frame) < 400 and previous_frame == None:
            print('need two hands to start')
            gui.label.configure(image=gui.bad)
            gui.gesture.set('position hands')
        # if we have at least one hand, and a previous frame to supplement any missing hand data, then we can proceed
        else:
            # if a hand is missing, fill in the data from the previous frame
            if len(packed_frame) < 400:
                packed_frame.update(previous_frame)
                print('Warning: a hand is missing')

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
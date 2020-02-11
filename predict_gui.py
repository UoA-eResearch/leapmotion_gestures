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
        self.label2 = tk.Label(master, font=("Helvetica", 36), textvariable=self.gesture)
        self.label2.pack(side=tk.BOTTOM)
        self.label_fury = tk.Label(master, foreground="#%02x%02x%02x" % (0,50,0,), font=("Helvetica", 30), text='fury')
        self.label_fury.pack(side=tk.LEFT)
        self.label_angularity = tk.Label(master, foreground="#%02x%02x%02x" % (0,50,0,), font=("Helvetica", 30), text='angularity')
        self.label_angularity.pack(side=tk.RIGHT)
        
root = tk.Tk()
root.geometry("400x500")
gui = GUI(root)


# load the prediction model
model = tf.keras.models.load_model('models/V3/40f_5hs.h5')
# mapping of gestures to integers: need this for decoding model output
gestures, g2idx, idx2g = get_gestures(version=3)
# set whether or not to derive features and drop unused VoI
derive_features = True

websocket_cache = {}
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# which hands will be used in predicting?
hands = ['left', 'right']
# Get the VoI that are used as predictors
VoI = get_VoI()
if derive_features:
    VoI_drop = get_VoI_drop()
    VoI_predictors = [v for v in VoI if v not in VoI_drop]
    # use these to get VoI, labelled by hand
    VoI_predictors = [hand + '_' + v for v in VoI_predictors for hand in hands]
else:
    VoI = [hand + '_' + v for v in VoI for hand in hands]


# IMPORTANT: always have variables in alphabetical order
# the model expects to receive them that way
# VoI.sort()
# v2idx = {v: i for i, v in enumerate(VoI)}

# get mean and standard deviation dictionaries
# these are used for standardizing input to model
with open('params/means_dict.json', 'r') as f:
    means_dict = json.load(f)
with open('params/stds_dict.json', 'r') as f:
    stds_dict = json.load(f)


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
# indicator the next successfully received frame will be the first
first_two_handed_frame = True

#capture every nth frame
n = 4

### initialize variables for fury and angularity calculations
raw_fury = 0
fury = 0
previous_fury = 0
angularity = 0
raw_angularity = 0

# amount of old value to keep when calculating moving average
beta = 0.9

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
            if derive_features:
                new_features = features.get_derived_features(packed_frame)
                packed_frame.update(new_features)
            
            # if this is the first two handed frame, generate a sorted list of variables used for prediction, including derived features
            if first_two_handed_frame:
                first_two_handed_frame = False
                # for the first frame only, set previous complete frame to the current frame
                previous_complete_frame = packed_frame.copy()
                if derive_features:
                    all_predictors = VoI_predictors + [pred for pred in new_features.keys()]
                    all_predictors.sort()
                else:
                    all_predictors = sorted(VoI)
                print(all_predictors)

            
            ### calculate fury and angularity
            # calculate raw furiosness
            raw_fury = features.get_fury2(packed_frame, previous_complete_frame)
            # update moving average
            fury = beta * fury + (1 - beta) * raw_fury

            if frames_total % 5 == 0:
                raw_angularity = features.get_angularity(raw_fury, previous_fury)
                # a sudden movement will temporarily drive up raw angularity
                # if this happens, update angularity immediately
                if raw_angularity > 0.65:
                    angularity = raw_angularity
                else:
                    angularity = beta * angularity + (1 - beta) * raw_angularity
                previous_fury = raw_fury

            if frames_total % 10 == 0:
                print(f'angularity: {angularity:.2f} fury: {fury:.2f}')
            
            # update gui for anger and fury
            gui.label_fury.configure(foreground="#%02x%02x%02x" % (int(fury * 255),int((1-fury) * 255),0,))
            gui.label_angularity.configure(foreground="#%02x%02x%02x" % (int(angularity * 255),int((1-angularity) * 255),0,))

            previous_complete_frame = packed_frame.copy()

            for i, p in enumerate(all_predictors):
                frames[frame_index, i] = (packed_frame[p] - means_dict[p]) / stds_dict[p]

            # make a prediction every pred_interval number of frames
            # but first ensure there is a complete training example's worth of consecutive frames
            if frames_recorded >= keep and frames_recorded % pred_interval == 0:
                example = np.concatenate((frames[frame_index+1:,:], frames[:frame_index+1,:]))
                # feed example into model, and get a prediction
                pred = model.predict(np.expand_dims(example, axis=0))
                print(pred)
                print(idx2g[np.argmax(pred)])
                if pred[0][np.argmax(pred)] > 0.5:
                    # gui.img = tk.PhotoImage(file=f'data/images/{idx2g[np.argmax(pred)]}.png')
                    gui.label.configure(image=gui.img)
                    gui.gesture.set(idx2g[np.argmax(pred)].replace('_', ' '))

    # update the gui
    root.update_idletasks()
    root.update()
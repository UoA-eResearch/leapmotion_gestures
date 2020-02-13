#!/usr/bin/env python3

import websocket
import config
import json
import pandas as pd
import numpy as np
from src.data_methods import *
from src.leap_methods import collect_frame
from src.classes import *
import random
import time
import tkinter as tk
import matplotlib.pyplot as plt 
import tensorflow as tf

# dead bird from https://www.flickr.com/photos/9516941@N08/3180449008


        
root = tk.Tk()
root.geometry("600x500")
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
pred_interval = 15

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

### initialize variables for fury, angularity, and pred confidence calculations
raw_fury = 0
fury = 0
previous_fury = 0
angularity = 0
raw_angularity = 0
pred_confidence = 0
raw_pred_confidence = 0

# amount of old value to keep when calculating moving average
beta_fury = 0.9
beta_angularity = 0.9
beta_confidence = 0.8

# set up storage for fury and angularity history
angularity_cb = CircularBuffer((30,))
fury_cb = CircularBuffer((30,))
# also need history of prediction confidence...
confidence_cb = CircularBuffer((30,))

# set up plotting
plt.ion()
fig, ax = plt.subplots(figsize=(6,6))
text = ax.text(29,0.5,' ', bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 5})
line_fury, = plt.plot(fury_cb.get())
line_angularity, = plt.plot(angularity_cb.get())
line_pred, = plt.plot(confidence_cb.get())
plt.legend(['angularity', 'movement', 'prediction confidence'], loc='upper left')
plt.ylim(0,1)


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
                previous_frame.update(packed_frame)
                packed_frame = previous_frame.copy()
                if frames_total % 5 == 0:
                    print('Warning: a hand is missing')
            else:
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
            fury = beta_fury * fury + (1 - beta_fury) * raw_fury

            if frames_total % 5 == 0:
                raw_angularity = features.get_angularity(raw_fury, previous_fury)
                # a sudden movement will temporarily drive up raw angularity
                # if this happens, update angularity immediately
                if raw_angularity > 0.72:
                    angularity = raw_angularity
                else:
                    angularity = beta_angularity * angularity + (1 - beta_angularity) * raw_angularity
                previous_fury = raw_fury
                fury_cb.add(fury)
                angularity_cb.add(angularity)
                pred_confidence = beta_confidence * pred_confidence + (1 - beta_confidence) * raw_pred_confidence
                confidence_cb.add(pred_confidence)
                line_angularity.set_ydata(fury_cb.get())
                line_fury.set_ydata(angularity_cb.get())
                line_pred.set_ydata(confidence_cb.get())
                text.set_text(gui.gesture.get())
                text.set_position((27, pred_confidence))

            # if frames_total % 10 == 0:
                # print(f'angularity: {angularity:.2f} fury: {fury:.2f}')
                # print(packed_frame['right_palmPosition_0'])
            
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
                # confidence of prediction is used for plotting. Rescale so 0.3 is zero.
                raw_pred_confidence = max((np.max(pred) - 0.3) / 0.7, 0)
                print(raw_pred_confidence)
                print(idx2g[np.argmax(pred)])
                if pred[0][np.argmax(pred)] > 0.5:
                    gui.img = tk.PhotoImage(file=f'data/images/{idx2g[np.argmax(pred)]}.png')
                    gui.label.configure(image=gui.img)
                    gui.gesture.set(idx2g[np.argmax(pred)].replace('_', ' '))

    # update the gui
    root.update_idletasks()
    root.update()
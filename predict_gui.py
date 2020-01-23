#!/usr/bin/env python3

import websocket
import config
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import time
import tkinter as tk
import tensorflow as tf


# possibilities: 'no_gesture', 'maybe', 'fistshake', 'so_so', 'open_close', 'pointing_around', 'stop', 'shuffle_over', 'come'
# note: the modes expect no_gesture to be in first place

# dead bird from https://www.flickr.com/photos/9516941@N08/3180449008

# root = tk.Tk()
# good = tk.PhotoImage(file='data/images/thumbs_up.png')
# bad = tk.PhotoImage(file='data/images/thumbs_down.png')
# panel = tk.Label(window, image=bad)
# panel.pack()
# window.mainloop()
# print('got here')
# panel.configure(image=bad)


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
        
        # self.label.place(relheight='0.5', relwidth='0.5')
        # self.label2.pack(side=tk.BOTTOM)
        # self.start_button = tk.Button(master, text="start", command=self.live_predict)
        # self.start_button.pack()
        # self.close_button = tk.Button(master, text="Close", command=master.quit)
        # self.close_button.pack()
                                    

root = tk.Tk()
root.geometry("500x500")
gui = GUI(root)


websocket_cache = {}
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# mapping of variable names to feature indexes
with open('params/VoI.txt', 'r') as f:
    VoI = f.read()
VoI = VoI.split()
# IMPORTANT: always have variables in alphabetical order
# the model expects to receive them that way
VoI.sort()
v2idx = {'right_' + v: i for i, v in enumerate(VoI)}
# mapping of gestures to integers: need this for decoding model output
with open('params/gesturesV2.txt') as f:
    gestures = f.read()
    gestures = gestures.split()
# gesture to id
g2idx = {g: i for i, g in enumerate(gestures)}
# id to gesture
idx2g = {i: g for i, g in enumerate(gestures)}

# get mean and standard deviation dictionaries
# these are used for standardizing input to model
with open('params/means_dict.json', 'r') as f:
    means_dict = json.load(f)
with open('params/stds_dict.json', 'r') as f:
    stds_dict = json.load(f)

# load the prediction model
model = tf.keras.models.load_model('models/V2/25f_32hs_15c_2.h5')
# no of frames to keep stored in memory for prediction
keep = model.input.shape[-2]
# how often to make a prediction (in frames)
# for now, just set to same frequency as keep
pred_interval = keep

# initialize frame storage
frames = np.empty((keep,len(v2idx)))
# keep track of total no. of frames received
frames_total = 0
# no. captured continuously
# this will be reset if there is an interruption to input
frames_recorded = 0

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
                            frames[frame_index, i] = (packed_frame[v] - means_dict[v]) / stds_dict[v]
                    # print(frames[frame_index,0])
                    except:
                        # if the above fails, then we can't record the frame
                        # the continuity of the input is broken, and we restart
                        frames_recorded = 0
                        print('bad frames')
                        gui.label.configure(image=gui.bad)
                        gui.gesture.set('reposition hand')


                    # make a prediction every pred_interval number of frames
                    # but first ensure there is a complete training example's worth of consecutive frames
                    if frames_recorded >= keep and frames_recorded % pred_interval == 0:
                        example = np.concatenate((frames[frame_index:,:], frames[:frame_index,:]))
                        # feed example into model, and get a prediction
                        pred = model.predict(np.expand_dims(example, axis=0))
                        print(pred)
                        print(idx2g[np.argmax(pred)])
                        gui.img = tk.PhotoImage(file=f'data/images/{idx2g[np.argmax(pred)]}.png')
                        gui.label.configure(image=gui.img)
                        gui.gesture.set(idx2g[np.argmax(pred)].replace('_', ' '))
                        # gui.label.image=gui.good
    root.update_idletasks()
    root.update()
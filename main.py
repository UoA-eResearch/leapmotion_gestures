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


# possibilities: 'no_gesture', 'hitchhiking', 'fistshake', 'so_so', 'open_close', 'pointing_around', 'stop', 'shuffle_over', 'come'
# note: the modes expect no_gesture to be in first place



# root = tk.Tk()
# good = tk.PhotoImage(file='data/images/thumbs_up.png')
# bad = tk.PhotoImage(file='data/images/thumbs_down.png')
# panel = tk.Label(window, image=bad)
# panel.pack()
# window.mainloop()
# print('got here')
# panel.configure(image=bad)



class predict_GUI:
    def __init__(self, master):
        self.master = master
        master.title("GUI")
        self.websocket_cache = {}
        self.FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
        # mapping of variable names to feature indexes
        with open('params/VoI.txt', 'r') as f:
            VoI = f.read()
        VoI = VoI.split()
        # IMPORTANT: always have variables in alphabetical order
        # the model expects to receive them that way
        VoI.sort()
        self.v2idx = {'right_' + v: i for i, v in enumerate(VoI)}
        # mapping of gestures to integers: need this for decoding model output
        with open('params/gesturesV1.txt') as f:
            gestures = f.read()
            gestures = gestures.split()
        # gesture to id
        self.g2idx = {g: i for i, g in enumerate(gestures)}
        # id to gesture
        self.idx2g = {i: g for i, g in enumerate(gestures)}

        # get mean and standard deviation dictionaries
        # these are used for standardizing input to model
        with open('params/means_dict.json', 'r') as f:
            self.means_dict = json.load(f)
        with open('params/stds_dict.json', 'r') as f:
            self.stds_dict = json.load(f)

        # load the prediction model
        self.model = tf.keras.models.load_model('models/20HS8C.h5')
        # no of frames to keep stored in memory for prediction
        self.keep = self.model.input.shape[-2]
        # how often to make a prediction (in frames)
        # for now, just set to same frequency as keep
        self.pred_interval = 30

        # initialize frame storage
        self.frames = np.empty((self.keep,len(self.v2idx)))
        # keep track of total no. of frames received
        self.frames_total = 0
        # no. captured continuously
        # this will be reset if there is an interruption to input
        self.frames_recorded = 0


        self.good = tk.PhotoImage(file='data/images/thumbs_up.png')
        self.bad = tk.PhotoImage(file='data/images/thumbs_down.png')
        self.label = tk.Label(master, image=self.bad)
        # self.label.bind("<Button-1>", self.cycle_label_text)
        self.label.pack()

        # self.start_button = tk.Button(master, text="start", command=self.live_predict)
        # self.start_button.pack()
        self.close_button = tk.Button(master, text="Close", command=master.quit)
        self.close_button.pack()
    
    def live_predict(self):
        for i, device in enumerate(config.devices):
            self.frames_total += 1
            if i not in self.websocket_cache:
                ws = websocket.create_connection(device["url"])
                if device["mode"] == "desktop":
                    ws.send(json.dumps({"optimizeHMD": False}))
                else:
                    ws.send(json.dumps({"optimizeHMD": True}))
                version = ws.recv()
                print(i, version)
                self.websocket_cache[i] = ws
            elif self.frames_total % 4 != 0:
                # collect the frame, but don't unpack it
                # not quite right...
                resp = self.websocket_cache[i].recv()
            else:
                # collect and unpack
                resp = self.websocket_cache[i].recv()
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
                            finger_name = self.FINGERS[finger["type"]]
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


                        self.frames_recorded += 1
                        frame_index = (self.frames_recorded - 1) % self.keep
                        try:
                        # using the variable name to id mapping dict, update the np array containing the frames
                            for v, i in self.v2idx.items():
                                self.frames[frame_index, i] = (packed_frame[v] - self.means_dict[v]) / self.stds_dict[v]
                        # print(frames[frame_index,0])
                        except:
                            # if the above fails, then we can't record the frame
                            # the continuity of the input is broken, and we restart
                            self.frames_recorded = 0
                            print('bad frames')
                            self.label.configure(image=self.bad)


                        # make a prediction every pred_interval number of frames
                        # but first ensure there is a complete training example's worth of consecutive frames
                        if self.frames_recorded >= self.keep and self.frames_recorded % self.pred_interval == 0:
                            example = np.concatenate((self.frames[frame_index:,:], self.frames[:frame_index,:]))
                            # feed example into model, and get a prediction
                            pred = self.model.predict(np.expand_dims(example, axis=0))
                            print(pred)
                            print(self.idx2g[np.argmax(pred)])
                            self.label.configure(image=self.good)
                            self.label.image=self.good
        self.master.after(5, self.live_predict)
                                    

root = tk.Tk()
gui = predict_GUI(root)
root.after(100, gui.live_predict)
root.mainloop()
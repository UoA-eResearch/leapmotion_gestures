import tkinter as tk
import numpy as np

class GUI:
    """simple tkinter gui for gesture recognition"""
    def __init__(self, master):
        # root window of gui
        self.master = master
        master.title("GUI")
        # text variable to record current gesture
        self.gesture = tk.StringVar()
        self.gesture.set('position hand')
        # image to be displayed, representing gesture: begin with image of dead parrot
        self.img = tk.PhotoImage(file='data/images/dead.png')
        # image to display when input is bad
        self.bad = tk.PhotoImage(file='data/images/dead.png')
        self.label = tk.Label(master, image=self.bad)
        self.label.pack()
        # label for gesture text
        self.label2 = tk.Label(master, font=("Helvetica", 36), textvariable=self.gesture)
        self.label2.pack(side=tk.BOTTOM)
        # label with text coloured according to level of fury
        self.label_fury = tk.Label(master, foreground="#%02x%02x%02x" % (0,50,0,), font=("Helvetica", 30), text='   movement')
        self.label_fury.pack(side=tk.LEFT)
        # label with text coloured according to level of angularity
        self.label_angularity = tk.Label(master, foreground="#%02x%02x%02x" % (0,50,0,), font=("Helvetica", 30), text='angularity   ')
        self.label_angularity.pack(side=tk.RIGHT)


class SettingsGUI:
    """simple tkinter gui for setting up variables on application start"""
    def __init__(self, master):
        # root window of gui
        self.master = master
        master.title("settings GUI")
        self.ok = False
        # setup variables for update interval, including default value
        self.update_interval = 3
        self.update_interval_strvar = tk.StringVar()
        self.update_interval_strvar.set(str(self.update_interval))
        # setup spinbox for data entry
        self.update_interval_sb = tk.Spinbox(from_=1, to=20, textvariable=self.update_interval_strvar)
        self.update_interval_sb.pack(side=tk.LEFT)

        # setup variables for prediction interval, including default value
        self.pred_interval = 15
        self.pred_interval_strvar = tk.StringVar()
        self.pred_interval_strvar.set(str(self.pred_interval))
        # setup spinbox for data entry
        self.pred_interval_sb = tk.Spinbox(from_=1, to=40, textvariable=self.pred_interval_strvar)
        self.pred_interval_sb.pack(side=tk.RIGHT)

        self.ok_button = tk.Button(text='Update Settings', command=self.update_settings)
        self.ok_button.pack()
    
    def update_settings(self):
        """returns values selected by user"""
        self.update_interval = int(self.update_interval_sb.get())
        self.pred_interval = int(self.pred_interval_sb.get())

class CircularBuffer:
    """reasonbly efficient circular buffer for storing last n frames or levels of furiosity etc."""
    def __init__(self, shape):
        # shape determines the shape of the storage used
        # The first axis represents time steps, and the pointer increments on this axis, determining what will next be overwritten
        self.mem = np.zeros(shape)
        # store the next position to write too (store % len)
        self.count = 0
        self.len = shape[0]
    
    def add(self, item):
        """add item to circular buffer to pointer location, then move pointer along by one"""
        self.mem[self.count % self.len] = item
        self.count += 1
    
    def get(self):
        """return all items in buffer, ordered from oldest to newest"""
        return np.concatenate((self.mem[self.count % self.len:], self.mem[:self.count % self.len]))



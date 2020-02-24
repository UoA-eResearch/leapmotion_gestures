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



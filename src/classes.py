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
        # dictionaries of settings and associated values
        self.settings = {}
        self.sb_dict = {}
        self.strvar_dict = {}
        self.label_dict = {}
        self.text_format = ("Helvetica", 16)
        # root window of gui
        self.master = master
        master.title("settings")

        # generate spinboxes + labels for some different settings
        self.create_setting(0, 'prediction interval', 1, 40, 15)
        self.create_setting(1, 'graph update interval', 1, 20, 3)

        self.update_button = tk.Button(self.master, text='Update Settings', command=self.update_settings, font=self.text_format)
        self.update_button.grid()
    
    def create_setting(self, row, variable, from_, to, default):
        """generate spinbox + label"""
        # setup variables for prediction interval, including default value
        self.settings[variable] = default
        self.strvar_dict[variable] = tk.StringVar()
        self.strvar_dict[variable].set(str(self.settings[variable]))
        # setup spinbox for data entry
        self.sb_dict[variable] = tk.Spinbox(self.master, from_=from_, to=to, textvariable=self.strvar_dict[variable], font=self.text_format)
        self.sb_dict[variable].grid(row=row, column=1)
        # make text label
        self.label_dict[variable] = tk.Label(self.master, text=variable, font=self.text_format)
        self.label_dict[variable].grid(row=row, column=0)

    def update_settings(self):
        """returns values selected by user"""
        for setting in self.settings.keys():
            self.settings[setting] = int(self.sb_dict[setting].get())
        # self.update_interval = int(self.update_interval_sb.get())
        # self.pred_interval = int(self.pred_interval_sb.get())

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



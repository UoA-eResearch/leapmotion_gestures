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
    """tkinter gui for changing settings during application run"""
    def __init__(self, master):
        # dictionaries of current settings values 
        self.settings = {}
        # dictionary of tk spinbox objects
        self.sb_dict = {}
        # dictionary of string variables, for setting default spinbox values
        self.strvar_dict = {}
        # dictionary of labels used to demarcate spinboxes
        self.label_dict = {}
        # set univeral text format for all spinboxes and labels
        self.text_format = ("Helvetica", 16)
        # row of next spinbox/label, += 1 every time a spinbox is created
        self.next_row = 0
        # root window of settings gui
        self.master = master
        master.title("settings")

        # generate spinboxes + labels for some different settings
        self.create_setting('prediction interval', 1, 40, 1, 15)
        self.create_setting('graph update interval', 1, 20, 1, 3)
        self.create_setting('fury beta', 0.8, 1, 0.005, 0.9)
        self.create_setting('angularity beta', 0.8, 1, 0.005, 0.975)
        self.create_setting('confidence beta', 0.8, 1, 0.005, 0.98)
        self.create_setting('x axis range', 1, 150, 1, 30)
        self.create_setting('effective confidence zero', 0.05, 0.95, 0.05, 0.7)
        self.create_setting('min conf. to change image', 0.0, 0.95, 0.05, 0.55)
        self.create_setting('every nth frame to model', 1, 10, 1, 5)

        self.update_button = tk.Button(self.master, text='Update Settings', command=self.update_settings, font=self.text_format)
        self.update_button.grid()
    
    def create_setting(self, variable, from_, to, increment, default):
        """generate spinbox + label"""
        # set up default value of spinbox
        self.settings[variable] = default
        self.strvar_dict[variable] = tk.StringVar()
        self.strvar_dict[variable].set(str(self.settings[variable]))
        # set up spinbox the spinbox
        self.sb_dict[variable] = tk.Spinbox(self.master, from_=from_, to=to, increment=increment, textvariable=self.strvar_dict[variable], font=self.text_format)
        self.sb_dict[variable].grid(row=self.next_row, column=1)
        # set up corresponding text label
        self.label_dict[variable] = tk.Label(self.master, text=variable, font=self.text_format)
        self.label_dict[variable].grid(row=self.next_row, column=0)
        # increment row number
        self.next_row += 1

    def update_settings(self):
        """update the settings dictionary to reflect spinbox values"""
        for setting in self.settings.keys():
            if '.' in self.sb_dict[setting].get():
                self.settings[setting] = float(self.sb_dict[setting].get())
            else:
                self.settings[setting] = int(self.sb_dict[setting].get())

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



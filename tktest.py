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

websocket_cache = {}

good = mpimg.imread('data/images/thumbs_up.png')
bad = mpimg.imread('data/images/thumbs_down.png')

window = tk.Tk()
# image_data = tk.Image('data/images/thumbs_up.png')
good = tk.PhotoImage(file='data/images/thumbs_up.png')
bad = tk.PhotoImage(file='data/images/thumbs_down.png')
# bad = Tkinter.Image.open('data/images/thumbs_down.png')
panel = tk.Label(window, image=bad)
panel.pack()
window.mainloop()
print(panel.image)
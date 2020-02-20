# leap-gesture-ml

The purpose of this project is to develop a gesture recognition system using leap motion devices and machine learning, in consultation with artist Lucinda Boermans. The motivation for this is to investigate the potential of such a tool to be used in the exploration of gesture and communication. Currently, this repository stands as a test of feasibility, implementing gesture recognition and the capture of 'affective dimensions' which are displayed live in a simple GUI.

The next step would be to extend this to a conversational context, in which such a digital interface is used to facilitate the interaction of two people communicating through gestures and the affectations of movement.

The project as it currently stands has two main parts:
1. Training LSTM based models for recognizing gestures using a leap motion device;
2. Using such models with a leap motion device in a live environment, such that predictions and other 'affective' information can be viewed in real time.

## Training Data Flow
It is important to understant how the data from leap motion devices is transformed and used in this project. It looks something like this:
raw data -> select variables of interest -> calculate some derived variables -> drop unwanted variables -> standardize variables -> split into examples

Much of this process is controlled by files in the `params/` folder.

A good place to start is the notebook `exploration-many2one.ipynb`, which works step by step from loading data to training a model.

### raw data -> select variables of interest
A leap motion device outputs a lof of extraneous information, much of it redundant for prediction. The file `VoI.txt` controls which variables of interest (VoI) will be selected for when using the methods responsible for collecting and processing data from the leap motion device.

During this stage, the frame rate will also be checked, and every nth frame taken to aproximate a target frame rate. The target frame rate I have used is 25fps.

### select variables of interest -> calculate some derived variables
We may want higher level variables that are more informative for our model. Examples of this include using fingertip positions to calculate their distances from the plain the palm of the hand lies on, or their distance to the palm of the hand. This sort of information is much more informative than using the x/y/z coordinates of every fingertip relative to the leap motion device. This process is controlled as follows:

* `params/derived_features_one_handed.txt` contains the list of methods to apply to the data that will generate new one handed features (e.g. distances between the thumb and index fingers on the same hand)
* `params/derived_features_two_handed.txt` contains the list of methods to apply to the data that will generate new two handed features (e.g. distances between the left and right index fingers)
* `src/features.py` contains all such possible methods that can be applied

### calculate some derived variables -> drop unwanted variables
Some of the VoI may have been included only because they were needed for calculating derived variables. `params/VoI_drop.txt` contains the list of such variables that at this point should be dropped.

### drop unwanted variables -> standardize variables
We now have only the variables that will be fed to the model for training or prediction. They just need to be centered and standardized, so that they have unit variance and mean of zero. There are dictionaries with the standard deviation and mean of each variable, found in `params/stds_dict.json` and `params/means_dict.json`, respectively.

When using new derived variables or new VoI for the first time, they won't have means and standard deviations in the relevant dictionaries. The notebook recommended above contains a code block for regenerating these.

### standardize variables -> split into examples
The final step is to split the data into examples a certain number of frames long. I have been using somewhere values between 25 to 40 frames long.


## Live Data Flow
The data flow at prediction time, when using the GUI, looks much the same as during training - the same process of selecting VoI, deriving new variables, and discarding and standardizing the remaining variables still applies. Two differences apply: this is done live, and an 'affective dimension' is captured based on simple transoformations of variables.

### Facilitating live prediction
Live prediction is achieved as follows:
1. Keep recorded the last n frames captured by the leap motion device. n is the number of frames used by the model for single prediction.
2. Every k frames, feed the last n frames. k determines how often the model makes a prediction. Predict too often, and things will begin to lag.

n frames always need to be stored - we thus need a fixed quantity of storage, and there is a CircullarBuffer class in   `src/classes.py` for implementing this efficiently. This class stores n time steps worth of information in a numpy array, always overwriting the oldest time step when given new information.

### The affective dimension
In addition to predictions, we also want information on the affective dimension - how smooth are the movements? How angry?

Some of these affections are very context dependent, e.g. the same gesture might be humourous or angry, depending on the context. Thus in this project I have focussed on those that can be easily derived:

* Furiousness: this is a loaded word, and really is just a measure of how fast the hands and fingers are moving.
* Angularity: this is how jerky gestures are, and relies on the acceleration of the hands, and acceleration of the fingers relative to one another.

## Live Data and the GUI
There are two main GUI elements that both show the same information (gesture predicted, angularity, furiousness), but in different ways:
1. A Tkinter window
2. A matplotlib graph

### The Tkinter Window
Contains:
* Text and an image representing the current prediction (provided the prediction meets a certain threshold of significane - we don't want the window being updated continually by low confidence predictions)
* text 'Fury' and 'Angularity', each coloured somewhere between green and red, depending on the levels of these recorded.

### The matplotlib Window
This shows a live graph of furiousness, angularity, and prediction confidence. Floating labels on the prediction confidence line show what gesture is being predicted. A few details to note:
* The last m values need to be stored for plotting - thus, all three of these use the CircullarBuffer class for storage.
* For making smooth lines that can be nicely plotted, a weighted running average of values is used that looks something like: `value = 0.05 * current value + 0.95 * last value`.
* Prediction confidence is rescaled for best visual effect - some models will output 0.95 when confident, 0.6 when not so confident, rarely predicting below 0.3. It therefore makes sense to map \[0.3,1\] to \[0,1\], setting any value below 0.3 to zero.
* If the newly calculated value angularity goes above a certain threshold, then the moving average is discarded, and the equation goes from `value = 0.05 * current value + 0.95 * last value` to `value = current value`. This allows for big, sudden changes.


## Areas that need work
* Furiousness/angularity are calculated only on hand speed and the speed of fingers relative to one another. Thus there are particular ways in which the hands can be move that will be missed by these metrics as they are currently calculated.
* Text labels are used on the matplotlib GUI to indicate gesture. These should be replaced by symbols representing the gestures.
* After resizing or interacting with the matplotlib GUI, old gesture labels become visible around the outside, which looks very messy. This is a consequence of using blitting.
* With blitting, frame rate is acceptable in the matplotlib GUI. But there still exist upper limits on the upper limit of timesteps that can be displayed, and frame rate. This part of the code might be a lot faster if rewritten in pyqtgraph.
* Accuracy isn't great. This might be resolved by different features, architecture, or more training data.
# leap-gesture-ml

The purpose of this project is to develop a gesture recognition system using leap motion devices and machine learning, in consultation with artist Lucinda Boermans. The motivation for this is to investigate the potential of such a tool to be used in the exploration of gesture and communication. Currently, this repository stands as a test of feasibility, implementing gesture recognition and the capture of 'affective dimensions' which are displayed live in a simple GUI.

The next step would be to extend this to a conversational context, in which such a digital interface is used to facilitate the interaction of two people communicating through gestures and the affectations of movement.

The project as it currently stands has two main parts:
1. Training LSTM based models for recognizing gestures using a leap motion device;
2. Using such models with a leap motion device in a live environment, such that predictions and other 'affective' information can be viewed in real time.

To see the program in action, check it out on youtube [here](https://youtu.be/3BBsVIJNAhM).

## Training Data Flow
It is important to understant how the data from leap motion devices is transformed and used in this project. It looks something like this:
raw data -> select variables of interest -> calculate some derived variables -> drop unwanted variables -> standardize variables -> split into examples

Much of this process is controlled by files in the `params/` folder.

A good place to start is the notebook `exploration-many2one.ipynb`, which works step by step from loading data to training a model. Data can be downloaded from [here](https://drive.google.com/drive/folders/1gx3efh6_GlQn0re9NhdmoCJ58lR3FDVp?usp=sharing).

### raw data -> select variables of interest
A leap motion device outputs a lof of extraneous information, much of it redundant for prediction. The file `VoI.txt` controls which variables of interest (VoI) will be selected for when using the methods responsible for collecting and processing data from the leap motion device.

During this stage, the frame rate will also be checked, and every nth frame taken to aproximate a target frame rate. The target frame rate I have used is 5fps. I originally used 25fps, but the higher frame rate isn't helpful.

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
The data flow at prediction time, when using the GUI, looks much the same as during training - the same process of selecting VoI, deriving new variables, and discarding and standardizing the remaining variables still applies. But rather than using the parameters stored in the parameters folder, the model, along with its parameters, are loaded from `models/prediction_model/`. That way, any model with different parameters can be substituted into the `prediction_model` folder.

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

Unlike at model training time, some model specific parameters are loaded from the `prediction_model` folder. A new model and associated parameters may be put in this folder.

### The Settings Window
Contains options for:
* prediction interval - how often should the model make a predictions, measured in frames
* graph update interval - how often should the graph refresh, measured in frames
* fury beta - how much of the old value of fury to keep in the new value that is calculated, higher values make fury smoother
* angularity beta - like fury beta, but for angularity
* confidence beta - like fury beta, but for prediction confidence
* x axis range - how many time steps to graph
* effective confidence zero - the model generally has high confidence, this value controls rescaling, so that \[effective, 1\] is mapped to \[0, 1\]
* min conf. to change image - we don't want the image changing constantly because of low confidence predictions
* every nth frame to model - a model might be trained on 5fps, but the frame rate of data used in the script is roughly 25fps. This value should be around 5 for models such as this. Each model's description.txt file will say the fps it was trained with.
* labels gather? - 1 indicates labels will be drawn to the left, 0 indicates not. WARNING: having this option on will make things become slower and slower, as these have to be redrawn over the background.

### The Tkinter Window
Contains:
* Text and an image representing the current prediction (provided the prediction meets a certain threshold of significance - we don't want the window being updated continually by low confidence predictions)
* text 'Fury' and 'Angularity', each coloured somewhere between green and red, depending on the levels of these recorded.

### The matplotlib Window
This shows a live graph of furiousness, angularity, and prediction confidence. Floating labels on the prediction confidence line show what gesture is being predicted. A few details to note:
* The last m values need to be stored for plotting - thus, all three of these use the CircullarBuffer class for storage.
* For making smooth lines that can be nicely plotted, a weighted running average of values is used that looks something like: `value = 0.05 * current value + 0.95 * last value`.
* Prediction confidence is rescaled for best visual effect - some models will output 0.95 when confident, 0.6 when not so confident, rarely predicting below 0.3. It therefore makes sense to map \[0.3,1\] to \[0,1\], setting any value below 0.3 to zero.
* If the newly calculated value angularity goes above a certain threshold, then the moving average is discarded, and the equation goes from `value = 0.05 * current value + 0.95 * last value` to `value = current value`. This allows for big, sudden changes.

## Executables
The prediction/GUI portion of this project is available as an executable file in the releases section. The executable is packaged with python and all required dependencies except the leap motion SDK.

### Using an executable
First, install the leap motion SDK, if you have not done so already:
1. Download V4 of the leap motion SDK from [here](https://developer.leapmotion.com/setup/desktop) (this may require you to make an account)
2. After installing the leap motion SDK, open the Leap Motion Control Panel, tick 'Allow Web Apps', and click Apply

To download and run the executable:
1. Find the latest release in the releases tab of this repository
2. Download `predict_gui.vX.X.zip` and unzip to a folder of your choice
3. In the unzipped folder, open the file `predict_gui.exe`, and the application will launch

To use a new/different model, copy the content of the new model's folder to models/prediction_model/.

### Building an Executable
PyInstaller was used to build executables. This needs some tweaking to work with tensorflow 2.0.0:
* `tensorflow_core` needs to be added as a hidden import; a hook for doing so is in the hooks folder.
* Importing keras in `predict_gui.py` then needs to be done by importing directly from `tensorflow_core.python`.
* The import command then looks something like this: `pyinstaller --additional-hooks-dir=some\path\to\GestRec\hooks`
* Not related to tensorflow: once the executable is built, the folders params/, data/ (only containing data/images), and models/ need to be copied into the same directory as the executable.
* If using pyinstaller to generate a folder, rather than a single file exe, then `tensorflow_core/python/_pywrap_tensorflow_internal.pyd` and `tensorflow_core/python/_pywrap_tensorflow_internal.lib` are large, and can be safely deleted.

## Areas that need work/Issues to be aware of
### GUI issues
* Furiousness/angularity are calculated only on hand speed and the speed of fingers relative to one another. Thus there are particular ways in which the hands can be move that will be missed by these metrics as they are currently calculated.
* Text labels are used on the matplotlib GUI to indicate gesture. These should be replaced by symbols representing the gestures.
* With blitting, frame rate is acceptable in the matplotlib GUI. But on my laptop there still exist upper limits on the number of timesteps that can be displayed, and frame rate. This part of the code might be a lot faster if rewritten in pyqtgraph.
* if `labels gather?` is set to 1, old labels will gather at the left, and because they need to be redrawn every time the graph is refreshed, things will run slower and slower as labels build up.
### Model issues
* If the order of gestures is changed in a gestures txt file (in the params folder), then this will change the indices of gestures, and a model will no longer predict correctly if it was trained using a file with a different order. This is mitigated somewhat by deploying every model as a folder containing the gestures text file it was trained with.
* Models for prediction are stored in a folder with a copy of the parameters used to train them. These parameter files must be copied manually. Ideally, this would be automated.
* Accuracy is pretty good, as long as someone 'knows' the gestures. The models could be trained on a wider variety of training data to make them more robust to different ways of performing gestures. 
### Recording issues
* The gesture parameter file determining what gestures to record is hard coded to version 3.

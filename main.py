from src.models import *
from src.dataMethods import *
import numpy as np

# get data frame with variables of interest
df = CSV2VoI('src/data/test1.csv')


y = df['gesture'].to_matrix()
print(y[0:5])




# n_gestures = 

# multi_model = build_multi_model(n_gestures, rnn_units)


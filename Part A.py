import keras

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statistics

dataframes = pd.read_csv("/home/pokala/coursera/concrete_data.csv")
mean_sq_er_list=[]   # list for mean squared error

def model():
    dataset= dataframes.values
    predictor = dataset[:,0:8]
    target    = dataset[:,8]

    predictor_train , predictor_test , target_train , target_test = train_test_split(predictor , target , test_size=0.30)
    n_cols= predictor.shape[1]    # the shape of input layer
    
    # model of neural network
    model = Sequential()
    model.add(Dense(10,activation="relu",input_shape=(n_cols,)))
    model.add(Dense(1))
    
    # training and testing the model 
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(predictor_train,target_train,epochs=50)
    result = model.predict(predictor_test)
    
    # for computing the mean squared error between predicted and actual value
    x=mean_squared_error(target_test,result) 
    mean_sq_er_list.append(x)                     
    
    # for repeating the step 1,2,3 50 times
for i in range(50):
    model()
    
# the final mean and standard deviation of the 50 mean squared error
mean_A= statistics.mean(mean_sq_er_list)
standard_deviation_A = statistics.stdev(mean_sq_er_list)

standard_deviation_A = 287.8232365962564
mean_A = 273.18725491088185

    

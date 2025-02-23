from re import X
import numpy as np
import torch
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from CustomDataset import CustomRawDataset
from model_dispatcher import dispatch_model
# from model_dispatcher_cnn import dispatch_model
import config
from torch import nn
import os
from glob import glob
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import date, timedelta, datetime
import time
import random
from torch.utils.tensorboard import SummaryWriter
from CustomDataset import custom_collate_fn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,TimeDistributed
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
random.seed(42)
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :12], sequences[end_ix-1, 12:]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def fit(name_of_model, path_of_saved_model, logs_path):
    
    raw_angle_files_1 = glob(os.path.join("D:\Research_Project\My_project_22\input\preprocessed_auto_encoder_modified", "*", "*.csv"))
# print(raw_angle_files_1)
    all_filenames = [i for i in raw_angle_files_1]
    df = pd.concat(map(pd.read_csv, all_filenames),ignore_index=True)
    # print(df.shape)
    # df = df.sample(frac = 1)
    # # df.iloc[1:10,:]
    data=df
    data=pd.get_dummies(data,columns=['12'])
    data=data.to_numpy()
    x,y=split_sequences(data,6)
    # x=x[None:]
    # print(x.shape)
    # print(y.shape)
    # data=data.to_numpy()
    print(data.shape)
    
    # y=data.iloc[:,26:]
    # x=data.iloc[:,:26]
    # # x=data[-1:26]
    # print(x.shape)
    # print(y.shape)
    # x=x.to_numpy()
    # y=y.to_numpy()
    x_train,x_test,y_train,y_test= train_test_split(x, y, test_size = 0.2)
    print(x_train.shape)
    print(y_train.shape)
    print(type(y_train))
 

    
    model = Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(6,12)))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(Dense(10, activation='softmax'))
  
    model.summary()

   
    # model.summary()
    # # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # # print(x.shape)
    history = model.fit(x_train, y_train,batch_size=config.BATCH_SIZE,epochs=100,validation_data=(x_test,y_test),verbose=1)
    results = model.evaluate(x_test, y_test)
    y_pred=model.predict(x_test)
    y_pred=np.argmax(y_pred,axis=1)
    y_test=np.argmax(y_test,axis=1)
    # print(y_pred)
    cf_matrix=metrics.confusion_matrix(y_test,y_pred)
    print('Confusion matrix\n',cf_matrix)
    print(metrics.classification_report(y_test,y_pred))


    return

def main():
    start_time = time.monotonic()
    name_of_model = 'CustomMotionModel'
    path_of_saved_model = os.path.join(
        config.MODELS_PATH, name_of_model,
        date.today().strftime("%d-%m-%Y"),
        datetime.now().strftime("%H-%M-%S")
    )
    logs_path = os.path.join(
        config.LOGS_PATH,
        date.today().strftime("%d-%m-%Y"),
        datetime.now().strftime("%H-%M-%S")
    )
    fit(name_of_model, path_of_saved_model, logs_path)
    end_time = time.monotonic()
    print("-" * 20)
    print("Time Elapsed {}".format(timedelta(seconds=end_time - start_time)))

if __name__ == "__main__":
    main()




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
		seq_x, seq_y = sequences[i:end_ix, :54], sequences[end_ix-1, 54:]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def fit(name_of_model, path_of_saved_model, logs_path):
    
#     raw_angle_files_1 = glob(os.path.join("D:\Research_Project\My_project_22\input\preprocessed", "*", "*.csv"))
# # print(raw_angle_files_1)
#     all_filenames = [i for i in raw_angle_files_1]
#     df = pd.concat(map(pd.read_csv, all_filenames),ignore_index=True)
   
    data=pd.read_csv(r"D:\Research_Project\My_project_22\input\final_preprocessed_merged\merged.csv")
    data=pd.get_dummies(data,columns=['54'])
    data=data.to_numpy()
    x,y=split_sequences(data,5)
    print(y.shape)
    
    # target=data['26']
    # data=data.drop(['26'],axis=1)
    x_train,x_test,y_train,y_test= train_test_split(x, y, test_size = 0.2)
   
  

    # Model building
    n_steps=5
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 54), return_sequences=True))
    model.add(LSTM(30, activation='relu'))
    model.add(Dense(10,activation='softmax'))
    # model.add(Embedding(config.EXTRACTED_FEATURES,128))
    # model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    # model.add(GlobalMaxPool1D())
    # model.add(Dropout(0.2))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(1, activation='sigmoid'))
#     # model = Sequential()
#     # model.add(Embedding(num_distinct_words, embedding_output_dims, input_length=max_sequence_length))
#     # model.add(LSTM(10))
#     # model.add(Dense(1, activation='sigmoid'))
#     # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #     model = tf.keras.Sequential([
# #     tf.keras.layers.Embedding(config.EXTRACTED_FEATURES, 64),
# #     tf.keras.layers.LSTM(64),
# #     tf.keras.layers.Dense(1, activation="sigmoid")
# # ])
#     # x_train = pad_sequences(x_train, 300)
#     # x_test = pad_sequences(x_test, 300)
    model.summary()
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(x.shape)
    history = model.fit(x_train, y_train,batch_size=config.BATCH_SIZE,epochs=100,validation_data=(x_test,y_test),verbose=1)
    results = model.evaluate(x_test, y_test)


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




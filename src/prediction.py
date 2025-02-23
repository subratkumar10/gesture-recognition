

from dis import dis
# from tkinter.tix import AUTO
from requests import head
import torch
import config
import os
from glob import glob
from shutil import copyfile
import pandas as pd
import re
import numpy as np
from math import sqrt
import math
import tensorflow as tf
from keras import regularizers
from tensorflow.keras import Model, Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from re import X
import numpy as np
import torch
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from CustomDataset import CustomRawDataset
from model_dispatcher import dispatch_model
from tensorflow import keras
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
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,TimeDistributed
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gesture_name=['both_hand_frontup','both_hand_frontup_left_leg_frontup','both_hand_frontup_left_leg_sideup','both_hand_frontup_right_leg_frontup',
    'both_hand_frontup_right_leg_sideup','both_hand_sideup','both_hand_sideup_left_leg_frontup','both_hand_sideup_left_leg_kneeup','both_hand_sideup_left_leg_sideup',
    'both_hand_sideup_right_leg_frontup','both_hand_sideup_right_leg_sideup','left_hand_frontup_right_elbowV','left_hand_sideup_right_elbowK',
    'left_hand_sideup_right_hand_frontup','right_hand_frontup','right_hand_frontup_left_elbowV','right_hand_sideup','right_hand_sideup_left_elbowK','right_hand_sideup_left_hand_frontup','right_leg_kneeup_both_hand_sideup']
dict_name={
    'both_hand_frontup_ANGLES':0,
    'both_hand_frontup_left_leg_frontup_ANGLES':1,
    'both_hand_frontup_left_leg_sideup_ANGLES':2,
    'both_hand_frontup_right_leg_frontup_ANGLES':3,
    'both_hand_frontup_right_leg_sideup_ANGLES':4,
    'both_hand_sideup_ANGLES':5,
    'both_hand_sideup_left_leg_frontup_ANGLES':6,
    'both_hand_sideup_left_leg_kneeup_ANGLES':7,
    'both_hand_sideup_left_leg_sideup_ANGLES':8,
    'both_hand_sideup_right_leg_frontup_ANGLES':9,
    'both_hand_sideup_right_leg_sideup_ANGLES':10,
    'left_hand_frontup_right_elbowV_ANGLES':11,
    'left_hand_sideup_right_elbowK_ANGLES':12,
    'left_hand_sideup_right_hand_frontup_ANGLES':13,
    'right_hand_frontup_ANGLES':14,
    'right_hand_frontup_left_elbowV_ANGLES':15,
    'right_hand_sideup_ANGLES':16,
    'right_hand_sideup_left_elbowK_ANGLES':17,
    'right_hand_sideup_left_hand_frontup_ANGLES':18,
    'right_leg_kneeup_both_hand_sideup_ANGLES':19
}
path=r"D:\Research_Project\My_project_22\kinect_test_cases\ANGLES\right_leg_kneeup_both_hand_sideup_ANGLES.csv"
x=os.path.splitext( os.path.basename( path) )[0]
path1=rf"D:\Research_Project\My_project_22\kinect_test_cases\ANGLES\{x}_{1}.csv"

def extract_angles():
    print(path1)
    df=pd.read_csv(path1)

    df=df[['right_knee_pitch','right_hip_roll','right_hip_pitch','left_knee_pitch','left_hip_roll','left_hip_pitch',
    'right_elbow_roll','right_elbow_yaw','right_shoulder_roll','right_shoulder_pitch','left_elbow_roll','left_elbow_yaw','left_shoulder_roll',
    'target']]
    # df=df[df['target']==0]
    # target=df['target']
    # df=df.iloc[-300:,:]

    
   
    # df1['target']=target
    # print(df1)
    print(df)
  
    return df


def preprocess_raw_test_angles():

    angle_name=['0','1','left_shoulder_pitch','left_shoulder_roll','left_elbow_yaw','left_elbow_roll','right_shoulder_pitch','right_shoulder_roll','right_elbow_yaw','right_elbow_roll','left_hip_pitch','left_hip_roll','left_knee_pitch','right_hip_pitch','right_hip_roll','right_knee_pitch','16']
    df=pd.read_csv(path)


    df.columns=angle_name
    print(angle_name)
    df['target']=dict_name[x]
    print(df)
    
    df.to_csv(path1,index=False)

        # df['target']=i
        # df.to_csv(all_csv_files[j],index=False) 
    

def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :13], sequences[end_ix-1, 13:]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def reshape_data(data):
 
    x=data['target'].unique()[0]
    print(x)
    data=data.drop(['target'],axis=1)
    data.head()
    for i in range(20):
        data[i]=0
    data[x]=1
    data=data.to_numpy()
    return data
def predict_gestures(data):
    x_test,y_test=split_sequences(data,6)
    print(y_test)
    new_model = tf.keras.models.load_model(r'D:\Research_Project\My_project_22\models\my_model_20')
    print(new_model.summary())
    y_pred=new_model.predict(x_test)
    y_pred1=np.argmax(y_pred,axis=1)
    count_list=[0]*20
    for i in y_pred1:
        count_list[i]+=1
        print('y_pred1',i)
    max1=0
    pred_index=0
    for i in range(len(count_list)):
        if(max1<count_list[i]):
            max1=count_list[i]
            pred_index=i
    print('The motion is ',gesture_name[pred_index])

    # y_test1=np.argmax(y_test,axis=1)
    # cf_matrix=metrics.confusion_matrix(y_test1,y_pred1)
    # print('Confusion matrix\n',cf_matrix)
    # print(metrics.classification_report(y_test1,y_pred1)) 

    

if __name__ == '__main__':

    preprocess_raw_test_angles()
    data=extract_angles()
    data=reshape_data(data)
    predict_gestures(data)
   
 
    

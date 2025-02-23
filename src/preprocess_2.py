from dis import dis
from tkinter.tix import AUTO
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

def Dispertion_Ratio(x):
    """Returns the Ratio of Arithmetic Mean and Geometric Mean. High value of this ratio indicates features with more importance"""

    AM = np.mean(x)
    GM = np.power(np.prod(x),1/len(x))
    ratio = AM/GM
    return ratio

def PCA(X , num_components):
    """Principal Component analysis reduces high dimensional data to lower dimensions while capturing maximum variability of the dataset"""
    X_meaned = X - np.mean(X , axis = 0)
    cov_mat = np.cov(X_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    eigenvector_subset = sorted_eigenvectors[:,0:num_components]

    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    return X_reduced




def return_significant_rows(data):
    """Returns a numpy representation of the DataFrame with the most Significant Rows based on some statistical measures."""

    ###### 1
    # Rows are sorted on the basis of Dispersion Ratio.
    df = pd.DataFrame(data) # Data is loaded as Pandas DataFrame
    l = df.shape[1]
    ratio_list = [] # Empty List (Used to store the calculated ratios)
    rows_dict = {k: 0 for k in range(df.shape[0])} # Rows Dictionary is created with values set as 0

    for i in range(df.shape[0]):  # Loops over each row
        ratio_list.append(Dispertion_Ratio(abs(df.iloc[i,:].values))) # Dispersion Ratio calculated for each row
        rows_dict[i] = ratio_list[i] # Mapping of Row and Ratios

    sorted_rows_dict = dict(sorted(rows_dict.items(), key=lambda item: item[1])) # Row Dictionary is sorted on the basis of dict values.

    sorted_rows_list = list(sorted_rows_dict)    # Converted to list so as to access the rows 
    rng = int(0.15 * len(sorted_rows_list)) 
    return df.iloc[sorted(sorted_rows_list[rng:])].ewm(span=5).mean().values # Return the values Dataframe with significant rows and apply Exponential moving average with a window size of 5
    # return df.iloc[sorted(sorted_rows_list[rng:])].rolling(5).mean().values # Return the values Dataframe with significant rows and apply simple moving average with a window size of 5


def return_significant_rows_with_added_features(data):
    """Return Sifnificant Dataframe rows with extra added features"""

    ###### 1
    # Rows are sorted on the basis of Dispersion Ratio.
    df = pd.DataFrame(data) # Data is loaded as Pandas DataFrame
    l = df.shape[1]
    ratio_list = [] # Empty List (Used to store the calculated ratios)
    rows_dict = {k: 0 for k in range(df.shape[0])} # Rows Dictionary is created with values set as 0

    for i in range(df.shape[0]):  # Loops over each row
        ratio_list.append(Dispertion_Ratio(abs(df.iloc[i,:].values))) # Dispersion Ratio calculated for each row
        rows_dict[i] = ratio_list[i] # Mapping of Row and Ratios

    sorted_rows_dict = dict(sorted(rows_dict.items(), key=lambda item: item[1])) # Row Dictionary is sorted on the basis of dict values.

    sorted_rows_list = list(sorted_rows_dict)    # Converted to list so as to access the rows 
    rng = int(0.15 * len(sorted_rows_list)) 
    # return df.iloc[sorted(sorted_rows_list[rng:])].ewm(span=5).mean().values # Return the values Dataframe with significant rows and apply Exponential moving average with a window size of 5
    # return df.iloc[sorted(sorted_rows_list[rng:])].rolling(5).mean().values # Return the values Dataframe with significant rows and apply simple moving average with a window size of 5


    ##### 2 (Used PCA to reduce and find most variable feature columns and added those columns to the original dataframe)
    reduced_features_df =  pd.DataFrame(PCA(df.iloc[sorted(sorted_rows_list[rng:])],num_components=6))
    for i in range(reduced_features_df.shape[1]):
        df[(l+2) + i] = reduced_features_df[i]
    # return reduced_features_df.values
    return df.ewm(span=4).mean().values

def rows_with_added_features(data):
    """Return Dataframe with extra added features"""
    df = pd.DataFrame(data) # Data is loaded as Pandas DataFrame
    l = df.shape[1]
    ##### 1 (Used PCA to reduce and find most variable feature columns and added those columns to the original dataframe)
    reduced_features_df =  pd.DataFrame(PCA(df,num_components=12))
    for i in range(reduced_features_df.shape[1]):
        df[(l+2) + i] = reduced_features_df[i]
    return reduced_features_df.values
    # return df.ewm(span=4).mean().values
    # return df.values



def AutoEncoder(x):
    # Building the Input Layer
    input_layer = Input(shape =(x.shape[1], ))
    
    # Building the Encoder network
    encoded = Dense(100, activation ='tanh',
                    activity_regularizer = regularizers.l1(10e-5))(input_layer)
    encoded = Dense(50, activation ='tanh',
                    activity_regularizer = regularizers.l1(10e-5))(encoded)
    encoded = Dense(25, activation ='tanh',
                    activity_regularizer = regularizers.l1(10e-5))(encoded)
    encoded = Dense(12, activation ='tanh',
                    activity_regularizer = regularizers.l1(10e-5))(encoded)
    encoded = Dense(6, activation ='relu')(encoded)
    
    # Building the Decoder network
    decoded = Dense(12, activation ='tanh')(encoded)
    decoded = Dense(25, activation ='tanh')(decoded)
    decoded = Dense(50, activation ='tanh')(decoded)
    decoded = Dense(100, activation ='tanh')(decoded)
    
    # Building the Output Layer
    output_layer = Dense(x.shape[1], activation ='relu')(decoded)

    # Defining the parameters of the Auto-encoder network
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer ="adadelta", loss ="mse")
    
    # Training the Auto-encoder network
    autoencoder.fit(x.values, x.values, batch_size = 8, epochs = 10, shuffle = True)

    # Take only the Encoded Part of the AutoEncoder
    hidden_representation = Sequential()
    hidden_representation.add(autoencoder.layers[0])
    hidden_representation.add(autoencoder.layers[1])
    hidden_representation.add(autoencoder.layers[2])
    hidden_representation.add(autoencoder.layers[3])
    hidden_representation.add(autoencoder.layers[4])

    # Predict the new features using the learned representation
    reduced_features = hidden_representation.predict(x)
    return pd.DataFrame(reduced_features)

def extract_participant_pose_name(x):
    x = x.split("\\")[-1]
    return x
def tree_based_classifier_feature_extraction(data,target):
    model = ExtraTreesClassifier()
    model.fit(data,target)
    print(data.head())
    print(target.head())
    # print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    print("data col",data.columns)
    feat_importances = pd.Series(model.feature_importances_, index=data.columns)
    # print(feat_importances)

    lst=list(feat_importances.nlargest(26).index)
    print(lst)
    # for i in lst:
    #     print(type(i))
    for i in range(40):
        i=str(i)
        if i not in lst:
             data=data.drop(i,axis=1)
             data.head()
    print(data.shape)
    return data,target

def create_folder_wise_classes():
    raw_angle_files_1 = glob(os.path.join(config.INPUT_DISTANCES, "*", "*.csv"))
    raw_angle_files_2 = glob(os.path.join(config.INPUT_RAW, "*","ANGLES", "*.csv"))
    total_files = len(raw_angle_files_1)
    files_done = 0
    print("\rProgress: {}/{}".format(files_done, total_files), end='')
    # looping through two folders
    for x,y in zip(raw_angle_files_1, raw_angle_files_2):
        df_distance = pd.read_csv(x) # 107 columns
        df_angle = pd.read_csv(y)
        df_angle = df_angle.iloc[:,2:-2] # 14 columns

        distance = df_distance.iloc[:,0:40] # 40 columns
        area = df_distance.iloc[:,40:57]    # 17 columns
        centroids = df_distance.iloc[:,57:107] # 50 columns
        target=df_distance['107']
        print("type of df_distance ",type(target))
        # print("df distance ",df_distance['105'])
        # print("distance shape ",distance.shape)

        ###### Feature 1 -- Joint Angles
        df1 = df_angle.values    # -> 14 columns


        ###### Feature 2 -- Joint Angles + Centroid
        df2 = pd.concat([centroids, df_angle], axis=1, ignore_index=True)
        df2.dropna(axis=0, inplace=True)  # -> 48 columns


        ###### Feature 3 -- Centroid + Triangle Area
        df3 = pd.concat([centroids, area], axis=1, ignore_index=True)
        df3.dropna(axis=0, inplace = True) # -> 51 columns


        ##### Feature 4 -- Joint Angles + Triangle Area
        df4 = pd.concat([area, df_angle], axis=1, ignore_index=True)
        df4.dropna(axis=0, inplace=True) # -> 31 columns


        ##### Feature 5 -- Joint Angles + Centroid + Area
        df5 = pd.concat([centroids, area, df_angle], axis=1, ignore_index = True)
        df5.dropna(axis=0, inplace = True) # -> 65 columns


        ##### Feature 6 -- AutoEncoder (Joint Angles)
        # print("Started Training AutoEncoder for Joint Angles")
        # df6 = AutoEncoder(df_angle)


        ##### Feature 7 -- AutoEncoder (Joint Angles + Centroid)    
        # print("Started Training AutoEncoder for Joint Angles + Centroid")
        # df7 = AutoEncoder(df2)


        ##### Feature 8 -- AutoEncoder (Centroid + Triangle Area)
        # print("Started Training AutoEncoder for Centroid + Triangle Area")
        # df8 = AutoEncoder(df3)


        ##### Feature 9 -- AutoEncoder (Area + Joint Angles)
        # print("Started Training AutoEncoder for Joint Angles + Triangle Area")
        # df9 = AutoEncoder(df4)


        ##### Feature 10 -- AutoEncoder (Centroid + Area + Joint Angles)
        # print("Started Training AutoEncoder for Centroid + Area + Joint Angles")
        # df10 = AutoEncoder(df5)


        ##### Feature 11 -- PCA (Joint Angles)
        # print("Started Training PCA for Joint Angles")
        # df11 = pd.DataFrame(PCA(df_angle, num_components=6))


        # ##### Feature 12 -- Distances + Joint Angles
        
        
        df12 = pd.concat([distance, df_angle], axis=1,ignore_index=False) 
        len1=df12.shape[1]
        l1=list(range(0,len1))
        l2=[]
        for i in l1:
            i=str(i)
            l2.append(i)
        l1=l2
            
        df12.columns=l1
      
            
        # print('df columns',type(df12.columns))
        # print(type(l1))
        # print(df12.head())
        # -> 54columns
        # print(x,"-->",y)
        df12.dropna(axis=0, inplace=True)
        df12=distance
        data=df12
    
        end_id=data.shape[0]
        target=target[:end_id]
        
       
      
        # print(data.head())
        # print(data.shape,"-->",target.shape)
        # data,target=tree_based_classifier_feature_extraction(data,target)
        # print(data.head())
        # print(data.shape,"--->",target.shape)
      

        # ##### Feature 13 -- PCA (Distances) + Joint Angles
        # distance_pca = pd.DataFrame(PCA(distance, num_components=12))
        # df13 = pd.concat([df_angle, distance_pca], axis=1, ignore_index=True) 
        # df13.dropna(axis=0, inplace=True)  
        # df13 = AutoEncoder(df13)


    #  data =pd.read_csv(raw_input_files[i],header=None)
    # data = data.iloc[:,1:]
    # folder_name = extract_participant_pose_name(raw_input_files[i])
    # distance_df = extract_distance_features(data)
    # angle_df = extract_angle_features(data)
    # angles_save_dir = save_csv_dir(raw_input_files[i],ANGLES_PATH)
    # distances_save_dir = save_csv_dir(raw_input_files[i],DISTANCES_PATH)
    # angles_save_path = os.path.join(angles_save_dir,folder_name)
    # distances_save_path = os.path.join(distances_save_dir,folder_name)
    # distance_df.to_csv(distances_save_path,index=None,header=None)
    # angle_df.to_csv(angles_save_path,index=None,header=None)
        # X =pd.DataFrame(df13)
        X=AutoEncoder(data)
        X=data
        len1=X.shape[1]
        X[len1]=target

        file_name_splits = os.path.basename(x).split('_')
        trial_string_idx = [i for i, elem in enumerate(file_name_splits) if re.search('trial*', elem)][0]
        gesture = "_".join(file_name_splits[1:trial_string_idx])
        save_dir = os.path.join(config.INPUT_PREPROCESSED, gesture)
        dest = os.path.join(save_dir,extract_participant_pose_name(x))
        # dest = os.path.join(save_dir, os.path.splitext(os.path.basename(x))[0] + '.pt')
        # dest = os.path.join(save_dir, os.path.splitext(os.path.basename(x))[0])
        # save_dir.to_csv(dest,index=None,header=None)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        X.to_csv(dest,index=None)
        # X.to_csv(dest,index=None,header=None)
        # torch.save(X, dest)
        files_done += 1
        print("\rProgress: {}/{}".format(files_done, total_files), end='')
    print("\nDONE!!!")
    return

if __name__ == "__main__":
    create_folder_wise_classes()












    # def create_folder_wise_classes():
#     raw_angle_files = glob(os.path.join(config.INPUT_RAW, "*", "*.csv"))
#     total_files = len(raw_angle_files)
#     files_done = 0
#     print("\rProgress: {}/{}".format(files_done, total_files), end='')
#     for file in raw_angle_files:
#         df = pd.read_csv(file)
#         # df = df.iloc[:, 2:-1]
#         df = df.values
#         df = return_significant_rows(df)
#         # df = return_significant_rows_with_added_features(df)
#         # df = rows_with_added_features(df)
#         X = torch.from_numpy(df).float()
#         file_name_splits = os.path.basename(file).split('_')
#         trial_string_idx = [i for i, elem in enumerate(file_name_splits) if re.search('trial*', elem)][0]
#         gesture = "_".join(file_name_splits[1:trial_string_idx])
#         save_dir = os.path.join(config.INPUT_PREPROCESSED, gesture)
#         dest = os.path.join(save_dir, os.path.splitext(os.path.basename(file))[0] + '.pt')
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         torch.save(X, dest)
#         files_done += 1
#         print("\rProgress: {}/{}".format(files_done, total_files), end='')
#     print("\nDONE!!!")
#     return

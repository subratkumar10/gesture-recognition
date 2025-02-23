import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np
from glob import glob
import os

def cubic_spline_save(file_path):
    df = pd.read_csv(file_path, header=None).drop_duplicates(subset = [0], keep = "last")
    x = df.iloc[:, 0].values
    # print(x)
    # for i in range(len(x)-1):
    #     if(x[i+1]<=x[i]):
    #         print(file_path,x[i]," ",x[i+1])
    #         break
    results = pd.DataFrame(np.arange(0, 2.04, 0.04))
    for i in range(1, df.shape[1]-1):
        y = df.iloc[:, i].values
        cs = CubicSpline(x, y)
        x_interp = np.arange(0, 2.04, 0.04)
        y_interp = cs(x_interp)
        results = pd.concat([results, pd.DataFrame(y_interp)], axis = 1)
    gesture = os.path.basename(os.path.dirname(file_path))
    # save_dir = os.path.join("22 GESTURES", gesture)
    save_path = os.path.join(r"D:\Research_Project\My_project_22\kinect_test_cases\interpolated_data", os.path.splitext(os.path.basename(file_path))[0] + "_interpolated_python.csv")
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    results.to_csv(save_path, header=None, index=None)
    return

if __name__ == "__main__":
    print(glob("D:\Research_Project\My_project_22\kinect_test_cases\*.csv"))
    for file_path in glob("D:\Research_Project\My_project_22\kinect_test_cases\*.csv"):
        cubic_spline_save(file_path)
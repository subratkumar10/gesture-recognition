import os
import config
from glob import glob
import random
from shutil import copy
import re
import numpy as np

np.random.seed(42)
def create_folder(output_path, type_of_data = "test"):
    if not os.path.exists(os.path.join(output_path, type_of_data)):
        os.makedirs((os.path.join(output_path, type_of_data)))
    return os.path.join(output_path, type_of_data)

def move_files(output_path, list_of_files, num_elements_to_pop):
    for _ in range(num_elements_to_pop):
        file = list_of_files.pop()
        if not os.path.exists(os.path.join(output_path, os.path.basename(os.path.dirname(file)))):
            os.makedirs(os.path.join(output_path, os.path.basename(os.path.dirname(file))))
        dest = os.path.join(output_path, os.path.basename(os.path.dirname(file)), os.path.basename(file))
        copy(file, dest)
    return

def move_participant_wise_file(output_path, list_of_files):
    for file in list_of_files:
        file_name_splits = os.path.basename(file).split('_')
        trial_string_idx = [i for i, elem in enumerate(file_name_splits) if re.search('trial*', elem)][0]
        gesture = "_".join(file_name_splits[1:trial_string_idx])
        save_dir = os.path.join(output_path, gesture)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dest = os.path.join(save_dir, os.path.basename(file))
        copy(file, dest)
    return

def split_train_val_test(input_path, output_path, val_ratio = 0.1, test_ratio = 0.25):
    classes = glob(os.path.join(input_path, "*"))
    for class_name in classes:
        list_of_samples = glob(os.path.join(input_path, class_name, "*"))
        total_num_files = len(list_of_samples)
        list_of_samples = sorted(list_of_samples, reverse = True)
        random.shuffle(list_of_samples)
        move_files(create_folder(output_path, "test"), list_of_samples, int(test_ratio * total_num_files))
        move_files(create_folder(output_path, "val"), list_of_samples, int(val_ratio*total_num_files))
        move_files(create_folder(output_path, "train"), list_of_samples,
                   int((1 - (test_ratio + val_ratio)) * total_num_files))
        print("{} --- Done!!!".format(class_name))
    print("Done!!")
    return

def split_train_val_test_participant_wise(input_path, output_path, val_ratio = 0.1, test_ratio = 0.25):
    list_of_participants = glob(os.path.join(input_path, "*"))
    total_participants = len(list_of_participants)

    list_of_participants = sorted(list_of_participants, reverse=True)

    num_test = int(test_ratio*total_participants)
    num_val = int(val_ratio*total_participants)

    test_set = list_of_participants[:num_test]
    test_set = [glob(os.path.join(elem, "*.pt")) for elem in test_set]
    test_set = [elem for nested_list in test_set for elem in nested_list]

    val_set = list_of_participants[num_test:num_test + num_val]
    val_set = [glob(os.path.join(elem, "*.pt")) for elem in val_set]
    val_set = [elem for nested_list in val_set for elem in nested_list]

    train_set = list_of_participants[num_test + num_val:]
    train_set = [glob(os.path.join(elem, "*.pt")) for elem in train_set]
    train_set = [elem for nested_list in train_set for elem in nested_list]

    move_participant_wise_file(create_folder(output_path, "test"), test_set)
    move_participant_wise_file(create_folder(output_path, "val"), val_set)
    move_participant_wise_file(create_folder(output_path, "train"), train_set)


    print("Done!!")
    return

def main():
    split_train_val_test(input_path=config.INPUT_PREPROCESSED, output_path=config.INPUT_FINAL)
    # split_train_val_test_participant_wise(input_path=config.INPUT_PREPROCESSED, output_path=config.INPUT_FINAL)

if __name__ == "__main__":
    main()
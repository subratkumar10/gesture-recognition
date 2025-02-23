import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
INPUT_PATH = os.path.join(ROOT_DIR, "input")
INPUT_RAW = os.path.join(INPUT_PATH, "raw")
INPUT_RAW20 = os.path.join(INPUT_PATH, "raw20")
INPUT_DISTANCES = r"D:\Research_Project\My_project_22\FEATURES_EXTRACTED\DISTANCES"  ## For accessing the Distances Feature Folder
INPUT_DISTANCES20= r"D:\Research_Project\My_project_22\FEATURES_EXTRACTED20\DISTANCES"
INPUT_PREPROCESSED = os.path.join(INPUT_PATH, "preprocessed")
INPUT_PREPROCESSED20 = os.path.join(INPUT_PATH, "preprocessed20")
INPUT_FINAL = os.path.join(INPUT_PATH, "final")
RESULTS_PATH = os.path.join(ROOT_DIR, "results")
PLOTS_PATH = os.path.join(ROOT_DIR, "plots")
MODELS_PATH = os.path.join(ROOT_DIR, "models")
LOGS_PATH = os.path.join(ROOT_DIR, "logs")
# UTILS_PATH = os.path.join(ROOT_DIR, "utils")
BATCH_SIZE = 64
NUM_EPOCHS = 200
EXTRACTED_FEATURES = 26
NUM_FRAMES = 120
CLASS_DICT = {
    'both_hand_frontup_left_leg_frontup' : 0,
    'both_hand_frontup_left_leg_sideup' : 1,
    'both_hand_frontup_right_leg_frontup' : 2,
    'both_hand_frontup_right_leg_sideup' : 3,
    'both_hand_sideup_left_leg_frontup' : 4,
    'both_hand_sideup_left_leg_sideup' : 5,
    'both_hand_sideup_right_leg_frontup' : 6,
    'both_hand_sideup_right_leg_sideup' : 7,
    'both_hand_sideup_left_leg_kneeup' : 8,
    # 'both_hand_sideup_right_leg_kneeup': 9
    'right_leg_kneeup_both_hand_sideup' : 9
}

NUM_CLASSES = len(CLASS_DICT)
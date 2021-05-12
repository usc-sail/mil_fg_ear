import os

##
## PATH-RELATED PARAMETERS
##
LOG_DIR = os.path.join('/data/rajatheb/ears/expts/FG/gap_exp1', 'logs')   # Directory in which to write log files and save model file
DATA_PATH = '/data/rajatheb/ears/splits/aging_spk'# '/enter/complete/path/to/folder/containing/train/val/test/directories'          # Directory where downloaded and unzipped tfrecord features are saved (Must contain sub-directories train, test and val
DATA_PATH_SP_TRAIN = os.path.join(DATA_PATH, 'train_speech.txt')    
DATA_PATH_NS_TRAIN = os.path.join(DATA_PATH, 'train_noise.txt')
DATA_PATH_SP_VAL = os.path.join(DATA_PATH, 'val_speech.txt')
DATA_PATH_NS_VAL = os.path.join(DATA_PATH, 'val_noise.txt')
DATA_PATH_SP_TEST = os.path.join(DATA_PATH, 'test_speech.txt')      
DATA_PATH_NS_TEST = os.path.join(DATA_PATH, 'test_noise.txt')


###
### INPUT FEATURE PARAMETERS
###
INPUT_SHAPE = (45, 256)       # Feature-dimension used in training (Time, Frequency, Channels)
NUM_VAL_SAMPLES_SP = 1668          # Number of validation feature files per class
NUM_VAL_SAMPLES_NS = 3939          # Number of validation feature files per class
NUM_TEST_SAMPLES_SP = 1668      # Number of test 'speech' feature files
NUM_TEST_SAMPLES_NS = 3940     # Number of test 'non-speech' feature files

##
## TRAINING PARAMETERS
##
LEARNING_RATE = 1e-4    # Learning rate for Adam optimizer
NUM_EPOCHS = 50         # Number of epochs during training
BATCH_SIZE = 16        # Total batch size, must be a multiple of 4 (see data_loader.py)
NUM_STEPS = int(30000/BATCH_SIZE)     # Approx 50k training samples of minority class
PATIENCE = 3            # Number of epochs to wait after validation loss stops improving
LOG_FREQ  = 100         # Frequency of batches to log training metrics during an epoch
GPU_FRAC = 0.99          # Fraction of GPU to be used

''' Training script for Multiple Instance Learning for Foreground Detection

Train a Foreground (FG) Detection model from audio.
256 dimensional SAD embeddings are used as input.

Before running training script, edit the following variables in script 
parameters.py to reflect corresponding complete paths:
LOG_DIR and DATA_PATH. Also, training parameters can be modified if 
required.
Training can then be performed by running this script as:
"python train.py"

Important Functions/Classes:
    get_session        - Start a tensorflow session.
                         Uses GPU if "CUDA_VISIBLE_DEVICES" is set to
                         a device ID other than "" on line 20.
    define_keras_model - define a Convolutional Neural Network model 
                        in keras using CNN and FC blocks defined in 
                        ConvMPBlock() and FullyConnectedLayer()
    Logger             - Class defined to implement model performance
                         logging, saving model file and stopping training
    train_model        - Train model with training parameters set in
                         SAD_parameters.py. Early stopping criterion
                         is used and defined in Logger class
    test_model         - Test keras model after training has ended
                         on external test set.

'''

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
import keras
import tensorflow as tf
import numpy as np
from keras.layers import *
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
import time
from parameters import *
from data_loader import Data
#from data_loader_emb import Data
from model_MIL import define_MIL_model

def get_session(gpu_fraction=0.333, num_cpus=8):
    if os.environ["CUDA_VISIBLE_DEVICES"] == '':
        config = tf.ConfigProto(device_count={"CPU":16})
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
    return tf.Session(config=config)

def ConvMPBlock(x, num_convs=2, fsize=32, kernel_size=3, pool_size=(2,2), strides=(2,2), BN=False, DO=False, MP=True):
    for i in range(num_convs):
       x = Conv2D(fsize, kernel_size, padding='same')(x)
       if BN:
           x = BatchNormalization()(x)
       if DO:
           x = Dropout(DO)(x)
       x = Activation('relu')(x)
    if MP:
        x = MaxPooling2D(pool_size=pool_size, strides=strides, padding='same')(x)
    return x

def FullyConnectedLayer(x, nodes=512, act='relu', BN=False, DO=False):
    x = Dense(nodes)(x)
    if BN:
        x = BatchNormalization()(x)
    if DO:
        x = Dropout(DO)(x)
    x = Activation(act)(x)
    return x

''' Define Speech activity detection model.
'''
def define_keras_model(input_shape=INPUT_SHAPE, optimizer='adam', loss='binary_crossentropy'):    
    fsize = 32
    td_dim = 1024
    inp = Input(shape=input_shape)
#    x = ConvMPBlock(inp, num_convs=2, fsize=fsize, BN=True)
#    x = ConvMPBlock(x, num_convs=2, fsize=2*fsize, BN=True)
#    x = ConvMPBlock(x, num_convs=3, fsize=4*fsize, BN=True)
    #x = Conv2D(8*fsize, 3, padding='same')(x)
#    x = Flatten()(x)
#    x = Reshape((-1, x._keras_shape[2]*x._keras_shape[3]))(x)
    x = Bidirectional(LSTM(512, return_sequences=True))(inp)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
#    x = TimeDistributed(Dense(td_dim, activation='relu'))(x)
   # x = TimeDistributed(Dense(td_dim, activation='relu'))(x)
    x = GlobalMaxPooling1D()(x)
#    x = GlobalAveragePooling1D()(x)
    #x = FullyConnectedLayer(x, 512, BN=True)
    x = FullyConnectedLayer(x, 256, BN=True)
    x = FullyConnectedLayer(x, 64, BN=True)
    x = FullyConnectedLayer(x, 2, 'softmax')
    model = Model(inp, x)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

''' Class defined to Monitor training and log metrics
    during training
'''
class Logger:
    def __init__(self, log_dir, num_epochs, num_steps):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            print("Please delete log directory manually and try again. Exiting...")
            #exit()
            os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, 'training.log')
        self.model_file = os.path.join(log_dir, 'best_model.hdf5')
        self.train_metrics = {'acc':np.empty(num_steps), 'loss':np.empty(num_steps)}
        self.val_metrics = {'acc':np.empty(num_epochs), 'loss':np.empty(num_epochs)}
        self.start_time = time.time()
        self.best_loss = 1e6
        self.best_epoch = 0

    def log_batch_metrics(self, epoch, batch):
        log_epoch = os.path.join(self.log_dir, 'epoch_{}.log'.format(epoch+1))
        with open(log_epoch,'a') as fp_log:
            fp_log.write("Avg loss : %0.4f, accuracy : %0.2f after training %d batches\n"%(
                            np.mean(self.train_metrics['loss'][:batch]),
                            np.mean(self.train_metrics['acc'][:batch]), batch))
                            
    def log_epoch_metrics(self, epoch, val_metrics):
        self.val_metrics['loss'][epoch] = val_metrics[0]
        self.val_metrics['acc'][epoch] = val_metrics[1]
        with open(self.log_file,'a') as fp_log:
            fp_log.write("""EPOCH #%d TRAINING - avg loss : %0.4f, avg acc : %0.2f,
            VALIDATION - loss : %0.4f, acc : %0.2f \t\tTIME TAKEN - %0.2f minutes\n"""%(
            epoch+1, np.mean(self.train_metrics['loss']), 
            np.mean(self.train_metrics['acc']), self.val_metrics['loss'][epoch], 
            self.val_metrics['acc'][epoch], (time.time()-self.start_time)/60.0))
        print("Validation set - loss : %0.4f, acc : %0.2f after epoch %d"%(
                self.val_metrics['loss'][epoch], self.val_metrics['acc'][epoch], epoch+1))
    
    def log_test_metrics(self, test_metrics):
        with open(self.log_file, 'a') as fp_log:
            fp_log.write("TESTING - loss : %0.4f, acc : %0.2f"%(
                test_metrics[0], test_metrics[1]))
        print("Test set - loss : %0.4f, acc : %0.2f"%(test_metrics[0], test_metrics[1]))

    def save_model(self, epoch):
        if self.val_metrics['loss'][epoch] < self.best_loss:
            self.best_loss = self.val_metrics['loss'][epoch]
            self.best_epoch = epoch
            model.save(self.model_file)
            print("Model saved after epoch %d"%(epoch+1))

    def early_stopping(self, epoch, patience):
        if epoch - self.best_epoch > patience:
            print("Early stopping criterion met, stopping training...")
            return True
        return False

''' Validate model after each epoch
'''
def validate_model(data_obj, model, sess):
    sp_data, _ = data_obj.create_TFRDataset(data_obj.data_paths_sp_val, 'val', 1)
    ns_data, _ = data_obj.create_TFRDataset(data_obj.data_paths_ns_val, 'val', 0)
    [X_val_, y_val_, _] = data_obj.generate_test_batch(sp_data, ns_data)
    val_acc = []
    val_loss = []
    while 1:
        try:
            X_val, y_val = sess.run([X_val_, y_val_])
            metrics = model.test_on_batch(X_val, y_val)
            val_loss.append(metrics[0])
            val_acc.append(metrics[1])
        except:
            break
    return np.mean(val_loss), np.mean(val_acc)

''' Test the model on external test set after training
'''
def test_model(data_obj, model, sess):
    test_acc = []
    test_loss = []
    while 1:
        try:
            X_test, y_test = sess.run([data_obj.X_test, data_obj.y_test])
            metrics = model.test_on_batch(X_test, y_test)
            test_loss.append(metrics[0])
            test_acc.append(metrics[1])
        except:
            break
    return np.mean(test_loss), np.mean(test_acc)

''' Train model with default parameters defined in
    SAD_parameters.py
'''
def train_model(model, sess, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, num_steps=NUM_STEPS, patience=PATIENCE, log_dir=LOG_DIR, log_freq=LOG_FREQ):
    data_obj = Data()
#    X_val, y_val = sess.run([data_obj.X_val, data_obj.y_val])
    print(log_dir)
    logger = Logger(log_dir=log_dir, num_epochs=num_epochs, num_steps=num_steps)

    for epoch in range(num_epochs):
        #initialize per_epoch variables
        print("\nBeginning epoch %d"%(epoch+1))
        logger.start_time = time.time()
        for batch in range(num_steps):
            X_batch, y_batch = sess.run([data_obj.X_batch, data_obj.y_batch])
            
       #     print(X_batch.shape, y_batch.shape)
            # Train on single batch
            metrics = model.train_on_batch(X_batch, y_batch)
            #print(model.predict(X_batch))
            logger.train_metrics['loss'][batch] = metrics[0]
            logger.train_metrics['acc'][batch] = metrics[1]
            
            if batch % log_freq == 0 and batch!=0:  # Log training metrics every 'log_freq' batches
                logger.log_batch_metrics(epoch, batch)
        
        val_metrics = validate_model(data_obj=data_obj, model=model, sess=sess)#model.evaluate(X_val, y_val)
#        print(val_metrics)
        logger.log_epoch_metrics(epoch, val_metrics)
        logger.save_model(epoch)
        if logger.early_stopping(epoch, patience): 
            break                           ## Stop training if loss is not decreasing

    test_metrics = test_model(data_obj=data_obj, model=model, sess=sess)    # Evaluate performance on external test set
    logger.log_test_metrics(test_metrics)
    sess.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        LOG_DIR= sys.argv[1]
    if len(sys.argv) > 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    sess = get_session(gpu_fraction=GPU_FRAC)
    set_session(sess)
    model = define_MIL_model(input_shape=INPUT_SHAPE,optimizer=keras.optimizers.Adam(lr=LEARNING_RATE))
 #   model = define_keras_model(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE))
    model.summary()
    train_model(model=model, sess=sess, log_dir=LOG_DIR)


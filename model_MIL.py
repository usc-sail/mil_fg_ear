'''
Author/year: Rajat Hebbar, 2019

    MIL model for SAD embedding inputs

Input
    1)

Output 
    1) tf.keras.Model

Usage

Example

'''


import os, sys, numpy as np
from keras.layers import *
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K
#os.environ['CUDA_VISIBLE_DEVICES']=''
#import tensorflow as tf

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



def define_MIL_model(input_shape=(45, 256), optimizer='adam', loss='binary_crossentropy'):
#    attn_inp = Input(shape=(45,))
#    attn_U = Dense(attn_dim, activation='tanh')(attn_inp)
#    attn_V = Dense(attn_dim, activation='sigmoid')(attn_inp)
#    attn_gate = Multiply()([attn_U, attn_V])
#    attn_weights = Dense(45, activation='softmax', name='attention')(attn_U)
#    attn_out_gated = Dot(axes=1, normalize=True)([attn_weights, attn_inp])
#    gated_attn_layer = Model(attn_inp, attn_out_gated)
    
#    attn_wts = Dense(45, activation='hard_sigmoid')(attn_inp)
#    attn_out = Multiply()([attn_inp, attn_wts])
#    attn_layer = Model(attn_inp, attn_out)

    
    dense_regularizer=None#regularizers.l1(0.01)
    fsize=32
    td1=256
    td2=256
    td3=1
    td4=256
    attn_dim = 256
    inp = Input(shape=input_shape)
   
    x = TimeDistributed(Dense(td1, kernel_regularizer=dense_regularizer))(inp)
    x = TimeDistributed(BatchNormalization())(x)
    xd1 = TimeDistributed(Activation('sigmoid'))(x)
 
    x = TimeDistributed(Dense(td2, kernel_regularizer=dense_regularizer))(xd1)
    x = TimeDistributed(BatchNormalization())(x)
    xd2 = TimeDistributed(Activation('sigmoid'))(x)

    x = TimeDistributed(Dense(td3, kernel_regularizer=dense_regularizer))(xd2)
    x = TimeDistributed(BatchNormalization())(x)
    xd3 = TimeDistributed(Activation('sigmoid'))(x)

    #x = TimeDistributed(Dense(td4, kernel_regularizer=dense_regularizer))(xd3)
    #x = TimeDistributed(BatchNormalization())(x)
    #xd4 = TimeDistributed(Activation('sigmoid'))(x)
    x = Reshape((input_shape[0],), name='attn_input')(xd3)
    

    attn_V = Dense(attn_dim, activation='tanh', kernel_regularizer=dense_regularizer)(x)
    attn_U = Dense(attn_dim, activation='sigmoid', kernel_regularizer=dense_regularizer)(x)
    attn_mul = Multiply()([attn_V, attn_U])
    attn_w = Dense(45, activation='sigmoid', name='attn_weights')(attn_mul)
#    attn_resh = Reshape((-1,))(attn_w)

#    attn_resh = Reshape((-1, 1))(attn_soft)
    attn_out = Multiply()([attn_w, x])
    attn_resh = Reshape((-1, 1))(attn_out)
    out = GlobalMaxPooling1D()(attn_resh)
    
    model = Model(inp, out)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    main()


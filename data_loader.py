import tensorflow as tf
import numpy as np
import glob
import os
from SAD_parameters_emb import *

class Data:
    def __init__(self, inp_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, 
                    val_size_sp=NUM_VAL_SAMPLES_SP, val_size_ns=NUM_VAL_SAMPLES_NS,
                    speech_dp_train=DATA_PATH_SP_TRAIN, non_speech_dp_train=DATA_PATH_NS_TRAIN,
                    speech_dp_val=DATA_PATH_SP_VAL, non_speech_dp_val=DATA_PATH_NS_VAL,
                    speech_dp_test=DATA_PATH_SP_TEST, non_speech_dp_test=DATA_PATH_NS_TEST):
        self.inp_shape = inp_shape
        self.batch_size = batch_size
        self.val_size_sp = val_size_sp
        self.val_size_ns = val_size_ns
        self.data_paths_sp_train = speech_dp_train
        self.data_paths_ns_train = non_speech_dp_train
        self.data_paths_sp_val = speech_dp_val
        self.data_paths_ns_val = non_speech_dp_val
        self.data_paths_sp_test = speech_dp_test
        self.data_paths_ns_test = non_speech_dp_test
        self.data = {}
        self.build_datasets()
    
    def feature_parser(self, file_id, label):
        feats = np.load(file_id)['gap_emb']
        if feats.shape[0] > self.inp_shape[0]:
            feats = feats[:self.inp_shape[0]]
        return feats.astype('float32'), label, file_id.split('/')[-1].split('.npz')[0] #np.eye(2)[label].astype('int8')
#    def feature_parser(self, utt, label):
#        context_features = {'feature_id': tf.FixedLenFeature([], tf.string)}
#        sequence_features = {'log_mels': tf.FixedLenSequenceFeature([], tf.string)}
#        [utt_id, features_raw] = tf.parse_single_sequence_example(utt,
#                context_features=context_features, 
#                sequence_features=sequence_features)
#        features = tf.reshape(tf.decode_raw(features_raw['log_mels'],tf.float32), self.feat_dim)
#        return utt_id['feature_id'], features, np.eye(2)[label]
    
    def create_TFRDataset(self, file_list, mode, label):
#        files = glob.glob(os.path.join(path_to_tfrecords, '*tfrecord'))
        file_list = [x.rstrip() for x in open(file_list, 'r').readlines()]
        #dataset = tf.data.TFRecordDataset(file_list)
        dataset = tf.data.Dataset.from_tensor_slices(file_list)
        dataset = dataset.map(lambda x: tf.py_func(self.feature_parser, (x, label), (tf.float32, tf.int32, tf.string)))
        if mode == 'train':
            dataset = dataset.shuffle(self.batch_size)
            dataset = dataset.repeat()
            dataset = dataset.padded_batch(batch_size=int(self.batch_size/2), 
                                padded_shapes=(self.inp_shape, [], []), drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=self.batch_size)
        else:
            dataset = dataset.padded_batch(batch_size=self.batch_size,
                                padded_shapes=(self.inp_shape, [], []), drop_remainder=False)
            dataset = dataset.prefetch(buffer_size=self.batch_size)
    #    elif mode == 'val':
    #        if label == 0:
    #            dataset = dataset.padded_batch(batch_size=self.batch_size,
    #                            padded_shapes=([], [3020, 64], [2]), drop_remainder=False)
    #        else:
    #            dataset = dataset.padded_batch(batch_size=self.batch_size,
    #                            padded_shapes=([], [3020, 64], [2]), drop_remainder=False)
    #    elif mode == 'test':
    #        dataset = dataset.padded_batch(batch_size=self.batch_size,
    #                            padded_shapes=([], [3020, 64], [2]), drop_remainder=True)
        dataset_iterator = dataset.make_one_shot_iterator()
        return dataset, dataset_iterator
    
    def normalize_batch(self, data):
        mean, var = tf.nn.moments(data, axes=[0,1,2])
        norm_data = (data - mean) / tf.sqrt(var + 1e-8)
        return norm_data
    
    def concatenate_and_normalize_batch(self, sp_itr, ns_itr, end_size=[]):
        if end_size==[]: end_size=self.batch_size
        X_sp, y_sp, _ = sp_itr.get_next()
        X_ns, y_ns, _ = ns_itr.get_next()
        X_batch = tf.concat((X_sp, X_ns), axis=0)
        X_batch = self.normalize_batch(X_batch)
        X_batch = tf.reshape(X_batch, np.concatenate(([-1], self.inp_shape), axis=0))
        y_batch = tf.concat((y_sp, y_ns), axis=0)
        y_batch = tf.reshape(tf.cast(y_batch, tf.int32), [-1, 1])
        return X_batch, y_batch
    
    def generate_test_batch(self, sp_ds, ns_ds):
        dataset = sp_ds.concatenate(ns_ds)
        iterator = dataset.make_one_shot_iterator()
        X_test, y_test, utt_id = iterator.get_next()
        X_test = self.normalize_batch(X_test)
        X_test = tf.reshape(X_test, np.concatenate(([-1], self.inp_shape), axis=0))
        y_test = tf.reshape(tf.cast(y_test, tf.int32), [-1, 1])
        return X_test, y_test, utt_id

    def build_datasets(self):   
        _, self.data['speech_train'] = self.create_TFRDataset(self.data_paths_sp_train, 'train', 1)
        _, self.data['non_speech_train'] = self.create_TFRDataset(self.data_paths_ns_train, 'train', 0)
#        _, self.data['speech_val'] = self.create_TFRDataset(self.data_paths_sp_val, 'val', 1)
#        _, self.data['non_speech_val'] = self.create_TFRDataset(self.data_paths_ns_val, 'val', 0)
        self.data['speech_val'], _ = self.create_TFRDataset(self.data_paths_sp_val, 'val', 1)
        self.data['non_speech_val'], _ = self.create_TFRDataset(self.data_paths_ns_val, 'val', 0)
        self.data['speech_test'], _ = self.create_TFRDataset(self.data_paths_sp_test, 'test', 1)
        self.data['non_speech_test'], _ = self.create_TFRDataset(self.data_paths_ns_test, 'test', 0)
        [self.X_batch, self.y_batch] = self.concatenate_and_normalize_batch(self.data['speech_train'], self.data['non_speech_train'])
        [self.X_val, self.y_val, self.utt_id_val] = self.generate_test_batch(self.data['speech_val'], self.data['non_speech_val'])
#        [self.X_val, self.y_val] = self.concatenate_and_normalize_batch(self.data['speech_val'], self.data['non_speech_val'], end_size=2*self.val_size_sp)#+self.val_size_ns)       
        [self.X_test, self.y_test, self.utt_id_test] = self.generate_test_batch(self.data['speech_test'], self.data['non_speech_test'])

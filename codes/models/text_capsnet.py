import models.text_layer as text_layer
import os
import numpy as np
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from models.callbacks import HyperbolicTangentLR
from keras.callbacks import ModelCheckpoint

class TextCapsnet(object):
    def __init__(self, args, 
                 seq_len=800, 
                 num_classes=10, 
                 vocab_size=30000,
                 x_train=None,
                 y_train=None, 
                 x_test=None,
                 y_test=None, 
                 pretrain_vec=None):
        
        self.args = args
        self.init_lr = args.init_lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.l2 = args.l2
        self.routing = args.routing
        self.embedding_size = args.embedding_size
        self.dropout_ratio = args.dropout_ratio
        self.num_filter = args.num_filter
        self.filter_size = args.filter_size
        
        self.num_capsule = args.num_cap
        self.len_ui = args.len_ui
        self.len_vj = args.len_vj
        
        self.num_classes = num_classes
        self.sequence_length = seq_len
        self.vocab_size = vocab_size
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        self.pretrain_vec = pretrain_vec

    def train(self):

        model = text_layer.get_model(self, summary=True)

        # callbacks
        filepath = os.path.join(self.args.save_ckpt, "{}_checkpoint.h5".format(self.args.dataset))
        
        lr_scheduler = HyperbolicTangentLR(init_lr=self.init_lr, max_epoch=self.epochs, L=-6, U=3)
        ckpt_callback = ModelCheckpoint(filepath, 
                                        monitor='val_acc', 
                                        save_best_only=True, save_weights_only=True, mode='max')

        # train
        model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                  validation_data=[self.x_test, self.y_test],
                  callbacks=[lr_scheduler, ckpt_callback])
        
        model.load_weights(filepath)
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test)
        print("TEST ACC : {:.4f}".format(test_acc))
        
        

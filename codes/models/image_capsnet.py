from keras.datasets import mnist
from keras.utils import to_categorical
import models.image_layer as image_layer
from models.callbacks import HyperbolicTangentLR
from keras.callbacks import ModelCheckpoint
from PIL import Image
import os
import numpy as np


class MnistExperiment(object):
    def __init__(self, args, image_h = 28, image_w = 28,
                 len_vj = 16, num_classes=10, routing='static',
                 init_lr=0.0001, batch_size=64, epochs=50):
        
        self.args = args
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.routing = routing
        
        self.image_h = image_h
        self.image_w = image_w
        self.len_vj = len_vj
        self.num_classes = num_classes
        
    def manipulate_latent(self, model, x_train, y_train, 
                          save_path="../image", routing='static'):
        
        if not os.path.exist(save_path):
            os.makedirs(save_path)
        
        if self.routing == 'static':
            filepath = os.path.join(self.args.save_ckpt, "{}_static_checkpoint.h5".format(self.args.dataset))
        else:
            filepath = os.path.join(self.args.save_ckpt, "{}_checkpoint.h5".format(self.args.dataset))
        
        model.load_weights(filepath)
        
        y_test = np.argmax(y_test, axis=-1)
        for digit in range(self.num_classes):
            class_images = x_train[np.where(y_train == digit)]
            class_labels = y_train[np.where(y_train == digit)]
            rand_idx = np.random.randint(len(class_labels))
            
            x_data = class_images[rand_idx]
            y_data = [class_labels[rand_idx]]

            x_data = np.reshape(x_data, [1, self.image_h, self.image_w, 1])/255.
            mask = np.tile(np.expand_dims(to_categorical(np.reshape(y_data, [1, 1]),
                                                         num_classes=self.num_classes), axis=-1), [1, 1, self.len_vj])
            
            
            noise_val = [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]
            noise = np.zeros([1, self.num_classes, self.len_vj])
            result_image = np.zeros([self.image_h*self.len_vj, self.image_w*len(nosie_val)])
            padding_image = np.zeros([self.image_h*self.len_vj, self.image_w*3])
            routing_image = np.zeros([self.image_h*self.len_vj, self.image_w*len(nosie_val)])
            
            for properties in range(self.len_vj):
                for i, value in enumerate(noise_val):
                    injection_noise = np.copy(noise)
                    injection_noise[:, digit, properties] = value
                    
                    y_pred, recons = static_model.predict([x_data, mask, injection_noise])
                    recons[np.where(recons < 0)] = 0
                    recons = np.uint8(np.reshape(recons, [self.image_h, self.image_w])*255.)
                    
                    result_image[properties*self.image_h:(1+properties)*self.image_h, 
                                 i*self.image_w:(i+1)*self.image_w] = recons

            save_full_path = os.path.join(save_path, "{}_Digit_{}_recons.png".format(self.routing, digit))
            Image.fromarray(result_image.astype(np.uint8)).save(save_full_path)
            print("Save Digit {} Reconstruction".format(digit))
  
        print("FINISH")
                                                                

    def train(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        
        # noise injection used experiment
        train_perturbation = np.zeros([len(x_train), self.num_classes, self.len_vj])
        test_perturbation = np.zeros([len(x_test), self.num_classes, self.len_vj])
        
        # simple normalization
        norm_x_train = x_train/255.
        norm_x_test = x_test/255.
        
        # convert one-hot
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        
        # reconstruction mask
        train_mask = np.tile(np.expand_dims(y_train, axis=-1), [1, 1, self.len_vj])
        test_mask = np.tile(np.expand_dims(y_test, axis=-1), [1, 1, self.len_vj])
        
        print("Num Train : {}, Test : {}".format(len(x_train), len(x_test)))
        
        if self.routing == 'static':
            routing = 0
            filepath = os.path.join(self.args.save_ckpt, "{}_static_checkpoint.h5".format(self.args.dataset))
        else:
            routing = 1
            filepath = os.path.join(self.args.save_ckpt, "{}_checkpoint.h5".format(self.args.dataset))
        
        model = image_layer.get_model(self, summary=True, routing=routing)

        # callbacks
        lr_scheduler = HyperbolicTangentLR(init_lr=self.init_lr, max_epoch=self.epochs, L=-6, U=3)
        ckpt_callback = ModelCheckpoint(filepath, 
                                        monitor='val_pred_output_acc', 
                                        save_best_only=True, save_weights_only=True, mode='max')

        # train
        model.fit([norm_x_train, train_mask, train_perturbation], 
                  [y_train, np.reshape(norm_x_train, [-1, 784])], 
                  batch_size=self.batch_size, epochs=self.epochs,
                  validation_data=[[norm_x_test, test_mask, test_perturbation], 
                                   [y_test, np.reshape(norm_x_test, [-1, 784])]],
                  callbacks=[lr_scheduler, ckpt_callback])

        print("Load trained weights")
        self.manipulate_latent(model, x_train, y_train)
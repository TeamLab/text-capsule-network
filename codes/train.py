import config
import models.text_capsnet as capsnet
import models.image_capsnet as image_capsnet
import data_load
import numpy as np
import os

if __name__ == '__main__':

    args = config.get_args()
    
    if not os.path.exists(args.save_ckpt):
        os.makedirs(args.save_ckpt)

    # gpu id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # data load
    if args.dataset != 'mnist':
        seq_len, num_classes, vocab_size, x_train, y_train, x_test, y_test, w2v, word_idx = \
            data_load.preprocessing(args.data_path, args.dataset)

        print("Number of Train {}, Validation {}".format(len(x_train), len(x_test)))

        text_capsnet = capsnet.TextCapsnet(args, 
                                           seq_len=seq_len, 
                                           num_classes=num_classes, 
                                           vocab_size=vocab_size, 
                                           x_train=x_train, 
                                           y_train=y_train,
                                           x_test=x_test,
                                           y_test=y_test, 
                                           pretrain_vec=w2v)
        
        # training
        text_capsnet.train()

    else:
        static_capsnet = image_capsnet.MnistExperiment(args, routing='static')
        static_capsnet.train()
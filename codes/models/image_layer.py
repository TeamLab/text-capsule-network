import tensorflow as tf
import numpy as np
from keras import layers
from keras.regularizers import l2
from keras import initializers
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import mean_squared_error

class CapsuleNorm(layers.Layer):
    """
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Routing(layers.Layer):

    def __init__(self, num_capsule,
                 dim_capsule,
                 routing=False,
                 num_routing=3,
                 l2_constant=0.0001,
                 kernel_initializer='glorot_uniform', **kwargs):

        super(Routing, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routing = routing
        self.num_routing = num_routing
        self.l2_constant = l2_constant
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):

        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 regularizer=l2(self.l2_constant),
                                 name='capsule_weight')
        self.built = True

    def call(self, inputs, training=True):

        inputs_expand = K.expand_dims(inputs, 1)
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # inputs_hat.shape = [None, num_capsule, input_num_capsule, upper capsule length]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # dynamic routing
        if self.routing:
            b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

            for i in range(self.num_routing):
                # c shape = [batch_size, num_capsule, input_num_capsule]
                c = tf.nn.softmax(b, dim=1)
                # outputs = [batch_size, num_classes, upper capsule length]
                outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))

                if i < self.routing - 1:
                    b += K.batch_dot(outputs, inputs_hat, [2, 3])

        # static routing
        else:
            # outputs = [batch_size, num_classes, upper capsule length]
            outputs = K.sum(inputs_hat, axis=2)
            outputs = squash(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routing': self.routing,
            'num_routing': self.num_routing,
            'l2_constant': self.l2_constant
        }
        base_config = super(Routing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


def margin_loss(y_true, y_pred):

    """
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def reconstruct_loss(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    return mean_squared_error(y_true, y_pred)
    

def get_model(self, summary=True, 
              num_capsule=32, len_ui=8, 
              len_vj=16, routing=0, init_lr=0.001,
              l2_constant=0.0, dropout_ratio=0.1, num_classes=10):

    if routing:
        use_routing = True
    else:
        use_routing = False


    input_img = layers.Input((28, 28, 1))
    input_mask = layers.Input((num_classes, len_vj))
    
    # only use experiment reconstruction
    input_permutation = layers.Input((num_classes, len_vj))
    
    conv_layer = layers.Conv2D(256, (9, 9), strides=(1, 1),
                               use_bias=True, kernel_regularizer=l2(l2_constant), activation=None)(input_img)
    conv_layer = layers.Activation('relu')(conv_layer)

    # convolutional capsule layer
    h_i = layers.Conv2D(num_capsule * len_ui,
                        kernel_size=(9, 9),
                        strides=(2, 2),
                        padding='valid',
                        use_bias=True,
                        kernel_regularizer=l2(l2_constant), activation=None)(conv_layer)
    
    h_i = layers.Reshape((K.int_shape(h_i)[1] * K.int_shape(h_i)[2] * num_capsule, len_ui))(h_i)
    h_i = layers.Activation('relu')(h_i)

    # routing algorithm
    image_caps = Routing(num_capsule=num_classes,
                         l2_constant=l2_constant,
                         dim_capsule=len_vj,
                         routing=use_routing,
                         num_routing=3)(h_i)

    output = CapsuleNorm(name='pred_output')(image_caps)
    
    # reconstruction
    
    # mask_output : [B, Num Classes, len_vj]
    mask_output = layers.Multiply()([image_caps, input_mask])
    mask_output = layers.Add()([mask_output, input_permutation])
    
    # mask_output : [B, len_vj]
    mask_output = layers.Lambda(lambda x : K.sum(mask_output, axis=1))(mask_output)

    fc = layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_constant))(mask_output)
    fc = layers.Dense(1024, activation='relu', kernel_regularizer=l2(l2_constant))(fc)
    fc = layers.Dense(784, activation='sigmoid', kernel_regularizer=l2(l2_constant), name='reconstruct')(fc)
    
    model = Model([input_img, input_mask, input_permutation], [output, fc], name='image-capsnet')

    if summary:
        model.summary()

    # compile model
    losses = {"pred_output" : margin_loss, "reconstruct": reconstruct_loss}
    loss_weights = {"pred_output": 1.0, "reconstruct" : 0.0005*784}
    metrics = {"pred_output" : 'accuracy', "reconstruct" : "mae"}
    
    model.compile(loss=losses, loss_weights=loss_weights,
                  optimizer=Adam(init_lr, beta_1=0.9, beta_2=0.999, amsgrad=True),
                  metrics=metrics)
    return model
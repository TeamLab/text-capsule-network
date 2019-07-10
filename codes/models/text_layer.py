import tensorflow as tf
import numpy as np
from keras import layers
from keras.regularizers import l2
from keras import initializers
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K


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


def get_model(self, summary=True):

    if self.routing:
        use_routing = True
    else:
        use_routing = False


    input_tokens = layers.Input((self.sequence_length,))
    embedding = layers.Embedding(self.vocab_size, self.embedding_size,
                                 dropout=self.dropout_ratio,
                                 weights=[self.pretrain_vec],
                                 trainable=True,
                                 embeddings_regularizer=l2(self.l2), mask_zero=True)(input_tokens)
    embedding = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(embedding)

    # non-linear gate layer
    elu_layer = layers.Conv2D(self.num_filter, kernel_size=(self.filter_size, self.embedding_size),
                              use_bias=False,
                              kernel_regularizer=l2(self.l2), activation=None)(embedding)
    elu_layer = layers.BatchNormalization()(elu_layer)
    elu_layer = layers.Activation('elu')(elu_layer)

    conv_layer = layers.Conv2D(self.num_filter, kernel_size=(self.filter_size, self.embedding_size),
                               use_bias=False,
                               kernel_regularizer=l2(self.l2), activation=None)(embedding)
    conv_layer = layers.BatchNormalization()(conv_layer)

    gate_layer = layers.Multiply()([elu_layer, conv_layer])

    # dropout
    gate_layer = layers.Dropout(self.dropout_ratio)(gate_layer)

    # convolutional capsule layer
    h_i = layers.Conv2D(self.num_capsule * self.len_ui,
                        kernel_size=(K.int_shape(gate_layer)[1], 1),
                        use_bias=False,
                        kernel_regularizer=l2(self.l2), activation=None)(gate_layer)
    h_i = layers.Reshape((self.num_capsule, self.len_ui))(h_i)
    h_i = layers.BatchNormalization()(h_i)

    h_i = layers.Activation('relu')(h_i)

    # dropout
    h_i = layers.Dropout(self.dropout_ratio)(h_i)

    # routing algorithm
    text_caps = Routing(num_capsule=self.num_classes,
                        l2_constant=self.l2,
                        dim_capsule=self.len_vj,
                        routing=use_routing,
                        num_routing=3)(h_i)

    output = CapsuleNorm()(text_caps)

    model = Model(input_tokens, output, name='text-capsnet')

    if summary:
        model.summary()

    # compile model
    model.compile(loss=[margin_loss], optimizer=Adam(self.init_lr, beta_1=0.7, beta_2=0.999, amsgrad=True),
                  metrics=['accuracy'])

    return model
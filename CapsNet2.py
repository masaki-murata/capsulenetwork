# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 19:25:02 2018

@author: murata
"""

# import everything
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers, losses
from keras.engine.topology import Layer
from keras.layers import Input, Dropout, Dense, Reshape, Activation, Flatten, Conv2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import metrics


# setting GPU
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="1", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))


#from keras.utils import np_utils


def squash(_s, axis=-1): #_s.shape=[:, 16]
    _s_norm = K.sum(K.square(_s), axis=axis, keepdims=True)
    _v = _s_norm*_s / ((1+_s_norm)*K.sqrt(_s_norm+K.epsilon()))
    return _v

    
class PrimaryCapsuleLayer(Layer):
    # 入力：(batch_size,1152, 8, 1, 1)：２つ目の Conv 後に Reshape した後
    # 出力：(Batch_size,1152,10, 16, 1)
    def __init__(self, 
                 #input_capsule_num, input_capsule_dim, 
                 output_capsule_num=10, output_capsule_dim=16,
                 routing_num=3, 
                 #activation, 
                 **kwargs):
        #self.input_capsule_num = 1152
        #self.input_capsule_dim = 8
        self.routing_num = routing_num
        self.output_capsule_num = output_capsule_num
        self.output_capsule_dim = output_capsule_dim
        #self.activation = activation
        super(PrimaryCapsuleLayer, self).__init__(**kwargs)
        
    def build(self, 
              input_shape,
              ):
        self.batch_size = input_shape[0]
        self.input_capsule_num = input_shape[1] # 1152
        self.input_capsule_dim = input_shape[2]  # 8
        # W.shape = [1, 1152, 8, 10, 16]
        # W_{ij} の i =1,...,1152, j =1,...10
        self.W = self.add_weight(name='kernel', 
                                      shape=[1, self.input_capsule_num, self.input_capsule_dim, self.output_capsule_num, self.output_capsule_dim],
                                      initializer='glorot_uniform',
                                     )
#         self.bias = self.add_weight(name='bias', shape=(1, self.output_dim), initializer='zeros')
        super(PrimaryCapsuleLayer, self).build(input_shape)
        
    def call(self, inputs):
        # inputs.shape = [batch_size, 1152, 8, 1, 1]
        uhat = K.sum(inputs*self.W, axis=2, keepdims=True)
        # uhat.shape = [batch_size, 1152, 1, 10, 16]
        # v = uhat
        # b.shape = [batch_size, 1152, 1, 10, 1], c.shape = [batch_size, 1152, 1, 10, 1]
        # s.shape = [batch_size, 1, 1, 10, 16]
        print("aho")
        print(self.input_capsule_num)
#        bc_shape = (None, self.input_capsule_num, 1, self.output_capsule_num, 1)
#        bc_shape = K.int_shape(uhat)[:-2]+(10,1)
#         s_shape   = (self.batch_size, 1, 1, self.output_capsule_num, self.output_capsule_dim)
#        b = K.zeros(shape=bc_shape)
        b = tf.zeros(shape=[K.shape(uhat)[0], self.input_capsule_num, 1, self.output_capsule_num, 1])
        c = tf.zeros(shape=[K.shape(uhat)[0], self.input_capsule_num, 1, self.output_capsule_num, 1])
#        print(b.shape)
#        b = K.zeros(shape=(-1, self.input_capsule_num, 1, self.output_capsule_num, 1))
        print("baka")
#        c = K.zeros(shape=bc_shape)
        s = tf.zeros(shape=[K.shape(uhat)[0], 1, 1, self.output_capsule_num, self.output_capsule_dim])
#        s = K.zeros(shape=s_shape)
        if_routing=False
        if if_routing==True:
            for i in range(self.routing_num):
                c = tf.nn.softmax(b, dim=4)#, dim=-1)
                print("c.shape = ",c.shape)
                s = K.sum(c*uhat, axis=1, keepdims=True)
                print("s.shape = ",s.shape)            
                v = squash(s, axis=-1)
                # v.shape = [batch_size, 1, 1, 10, 16]
                if i==0:
                    b = uhat * v
                else:
                    b += uhat * v     
        else:
        # uhat.shape = [batch_size, 1152, 1, 10, 16]
            v = K.sum(uhat, axis=2, keepdims=True )
        print(v.shape)
        return v
 
    def compute_output_shape(self, input_shape):
        return tuple([None, self.output_capsule_num, self.output_capsule_dim])

        
class CapsuleToPredict(Layer):

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(Layer):
#    """
#    capsule_num = 10, capsule_dim=16
#    CapsuleNetwork の出力 v に対し、一つのカプセル以外を（capsule_dim次元）0ベクトルに置き換える
#    トレーニング時と推論時で入力が異なることに注意！
#    入力：[v, y] (トレーニング時), v (推論時)
#    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
#    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
#    masked Tensor.
#    For example:
#        ```
#        v = keras.layers.Input(shape=[batch_size, capsule_num, capsule_dim])  
#        y = keras.layers.Input(shape=[batch_size, capsule_num])   # one-hot 表現（トレーニング時は教師データ)
#        # True labels. batch_num samples, capsule_num classes, one-hot coding.
#        out = SelectClass()(v)  # out.shape=[batch_size, capsule_num*capsule_dim]
#        # or
#        out2 = SelectClass()([v, y])  # out2.shape=[batch_size, capsule_num*capsule_dim]. Masked with true labels y. Of course y can also be manipulated.
#        ```
#    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # トレーニング時。正解クラス y_true が入力。y_true.shape=[batch_size, class_num] with class_num=capsule_num
            # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            lengths = K.sqrt(K.sum(K.square(inputs), -1))
            # generate t|he mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(lengths, 1), num_classes=inputs.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


# クラス分類の損失関数
def margin_loss(y_true, y_pred): # y_true.shape=y_pred.shape=(batch_size, number of class, )
    _lambda = 0.5
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + _lambda * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))
    
def CapsNet(input_shape, n_class, routing_num):
    # input_shape = [batch_size, 28, 28, 1]

    x = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # conv1.shape = (batch_size, 20, 20, 256)
    conv2 = layers.Conv2D(filters=256, kernel_size=9, strides=2, padding='valid', activation='relu', name='conv2')(conv1)
    # conv2.shape = (batch_size, 6, 6, 256)
    if_capsule=True
    if if_capsule==False:
        prediction = Flatten()(conv2)
        prediction = Dense(256, activation='relu')(prediction)
        prediction = Dense(10, activation='softmax')(prediction)
    else:        
        reshape1 = Reshape(target_shape=[-1, 8, 1, 1])(conv2)
        # reshape1.shape = (batch_size, 1152, 8, 1, 1)
        capsule = PrimaryCapsuleLayer(routing_num=routing_num)(reshape1)
        prediction = CapsuleToPredict(name='prediction')(capsule)
        prediction - Activation('softmax', name='predict_softmax')(prediction)
#     primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
#     digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)
#     out_caps = Length(name='out_caps')(digitcaps)
        y = layers.Input(shape=(n_class,))
        masked = Mask()([capsule, y])
        x_recon = Dense(512, activation='relu')(masked)
        x_recon = Dense(1024, activation='relu')(x_recon)
        x_recon = Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
        x_recon = Reshape(target_shape=input_shape, name='out_recon')(x_recon)

    return Model(x, prediction)
#     return Model([x,y], [prediction, x_recon])

def train(model, data, epoch_size=100, batch_size=128):

    (x_train, y_train), (x_test, y_test) = data

    adam = keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
#                  loss=margin_loss,
                  metrics={'predict_softmax': 'accuracy'},
#                  metrics=['accuracy'],
#                   loss=[margin_loss, 'mse'],
#                   loss_weights=[1., 0.0005],
                  #metrics={'prediction': 'accuracy'},
                  )
    
    model.fit(x_train,y_train, batch_size=batch_size, epochs=epoch_size)

#     model.fit([x_train, y_train],[y_train, x_train], batch_size=8, epochs=epoch_size)
#               validation_data=[[x_test, y_test], [y_test, x_test]],
#              )


    return model

    
def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    model = CapsNet(input_shape = (28, 28,1), n_class=10, routing_num=1)
    x_train = x_train.reshape(x_train.shape+(1,))
    x_test = x_test.reshape(x_test.shape+(1,))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(x_train.shape)
    model.summary()
    
    train(model=model, data=((x_train, y_train), (x_test, y_test)), epoch_size=4, batch_size=8)    

if __name__ == '__main__':
    main()


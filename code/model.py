import tensorflow as tf
from tensorflow.keras.layers import Flatten, Activation, Dropout, Lambda
from tensorflow.keras.layers import Dense, Input, LeakyReLU, MaxPooling2D, Conv2D, BatchNormalization, Add, Concatenate, SeparableConv2D, GlobalAveragePooling2D, DepthwiseConv2D, Maximum

from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

import random as rnd
import numpy as np
import math


def separable_conv_block(x, channels, kernel_size=3, stride=1, weight_decay=5e-4, dropout_rate=None, act='l'):

  ki = initializers.he_normal()

  kr = regularizers.l2(weight_decay)

  x = SeparableConv2D(channels, (kernel_size, kernel_size), kernel_initializer=ki,
                      strides=(stride, stride), use_bias=False, padding='same',
                      kernel_regularizer=kr)(x)

  x = BatchNormalization()(x)

  if (act == 'm'):
    x = mish(x)

  if (act == 'l'):
    x = LeakyReLU(alpha=0.2)(x)

  if dropout_rate != None and dropout_rate != 0.:
    x = Dropout(dropout_rate)(x)
  return x


def fusion_block(tensors,name,type='add'):
    if(type=='add'):
        return Add(name='add_'+name)(tensors)

    if (type == 'max'):
        return Maximum(name='max_'+name)(tensors)


## Mish Activation Function
def mish(x):
    return tf.keras.layers.Lambda(lambda x: x*tf.math.tanh(tf.math.log(1+tf.exp(x))))(x)

def activation(x,ind,t='-'):
    if (t == 'l'):
        return LeakyReLU(name='leakyrelu_'+str(ind),alpha=0.2)(x)
    if (t == 'm'):
        return mish(x)
    return x

def atrous_block(x_i,ind=0,nf=32,fs=3,strides=1,act='l',dropout_rate=None,weight_decay=5e-4,pool=0,FUS='max'):
    ki = initializers.he_normal()
    kr = regularizers.l2(weight_decay)
    x=[]
    d=[]
    ab=3
    
    redu_r = np.shape(x_i)[-1] // 2
    if(ind>0):
        x_i = Conv2D(redu_r, (1, 1), strides=(1, 1), kernel_initializer=ki, kernel_regularizer=kr, padding='same', use_bias=False,name='conv_2d_redu_' + str(ind))(x_i)
        x_i = BatchNormalization(name='atrous_redu_bn_' + str(ind))(x_i)
        x_i = activation(x_i, 'atrous_redu_act_'+str(ind), act)

    def mininet(x,dr,ind):
        m = DepthwiseConv2D(kernel_size=fs, kernel_initializer=ki, kernel_regularizer=kr, strides=strides, padding='same', use_bias=False,dilation_rate=dr + 1, name='atrous_depth_conv_' + str(ind) + '_' + str(dr))(x)
        m = BatchNormalization(name='atrous_bn_'+ str(ind) + '_' + str(dr))(m)
        m = activation(m, 'atrous_act'+str(ind)+'_'+str(dr), act)
        # if dropout_rate != None and dropout_rate != 0.:
        #     m = Dropout(dropout_rate)(m)
        return m

    for i in range(ab):
        x.append(mininet(x_i,i,ind))
        d.append(x[i].shape[1])

    mr=[x_i]

    for i in range(0,len(d)):
        if(d[0]==d[i]):
            mr.append(x[i])

    if(len(mr) > 1):
        f = fusion_block(mr, str(ind), FUS)
    else:
        f=x[0]

    b = Conv2D(nf, (1, 1), strides=(1, 1), kernel_initializer=ki, kernel_regularizer=kr, padding='same', use_bias=False,name='conv_2d_' + str(ind))(f)
    b = BatchNormalization(name='cnv_bn_'+ str(ind))(b)
    b = activation(b, 'ccnv_act'+str(ind), act)


    if dropout_rate != None and dropout_rate != 0.:
       b = Dropout(dropout_rate)(b)

    return b

def make_ACFF_model(H,W,C, fus, act):
    input_shape = [H, W, 3]
    inp = Input(shape=input_shape)
    
    x = inp

    wd = 5e-4
    x = Conv2D(32, (5,5), name= 'convI',strides=2,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = activation(x, 'ccnv_act0', act)
    x = atrous_block(x,ind=1,nf=64,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)
    x = MaxPooling2D()(x)
    x = atrous_block(x,ind=2,nf=96,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)
    x = MaxPooling2D()(x)
    x = atrous_block(x,ind=3,nf=128,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)
    x = MaxPooling2D()(x)
    x = atrous_block(x,ind=4,nf=128,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)
    x = atrous_block(x,ind=5,nf=128,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)
    x = atrous_block(x,ind=6,nf=256,fs=3,strides=1,FUS=fus,weight_decay=wd,act=act)

    x = separable_conv_block(x, C, kernel_size=1, stride=1, weight_decay=wd, act=act,dropout_rate=0.)

    x = GlobalAveragePooling2D()(x)
    cls = Activation('softmax', name='class_branch')(x)
    
    return inp, cls
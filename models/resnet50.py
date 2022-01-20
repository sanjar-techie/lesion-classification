from asyncio import AbstractEventLoop
from cbam import channel_attention
from cbam import spatial_attention
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, MaxPool2D, GlobalAvgPool2D, Dense, GlobalMaxPooling2D, Activation, multiply, Concatenate, Reshape, Lambda
from tensorflow.keras import Model
import tensorflow.keras.backend as K

# implement a function that attaches cbam block to a network
def attach_cbam(net):
  '''
  Input:
    net(tensor): tensor object which is an input into cbam

  Return:
    tensor after applying cbam
  '''
  x = channel_attention(net)
  cbam_block = spatial_attention(x)
  return cbam_block


# implement the conv2d, batchnormalizetion and relu block
def conv_bn_relu(x, filters, kernel_size, strides, padding=0):
  '''
  Input:
  x: input tensor 
    filters: the number of filters -> int
    kernel_size: the sixe of the kernel -> int
    strides: the strides -> int
  
  Return:
    outout tensor from applying conv, batchNorm and Relu
  '''

  x = Conv2D(filters=filters, 
             kernel_size=kernel_size, 
             strides=strides,
             padding='same')(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  return x

# build identity block
def identity_block(x, filters, cbam):
  '''
  Input:
    x: input tensor
    filters: number of filters: -> int
    cbam(boolean): attach cbma or not

  Return:
  output tensor from applying 1x1, 3x3, 1x1 conv_bn_relu block followed by batchnorm and input tensor
  '''
  input_tensor = x
  x = conv_bn_relu(x, filters=filters, kernel_size=1, strides=1)
  x = conv_bn_relu(x, filters=filters, kernel_size=3, strides=1)
  x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
  x = BatchNormalization()(x)

  x = Add()([x, input_tensor])
  x = ReLU()(x)
  
  # attach cbam block
  if cbam:
    x = attach_cbam(x)
  return x

  # bild the projection block
def projection_block(x, filters, strides, cbam): # strides are added because it varies in the downsampling
  '''
  Input:
    x: input tensor
    filters: number of filters -> int
    strides: the strides -> int 
    cbam(boolean): attach cbma or not


  Return:
    putput tensor after applying projection block's elements 
  '''
  # left stream
  input_tensor = x
  x = conv_bn_relu(x, filters=filters, kernel_size=1, strides=strides)
  x = conv_bn_relu(x, filters=filters, kernel_size=3, strides=1)
  x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
  x = BatchNormalization()(x)

  #right stream
  skip = Conv2D(filters=4*filters, kernel_size=1, strides=strides)(input_tensor)
  skip = BatchNormalization()(skip)

  x = Add()([x, skip])
  x = ReLU()(x)

  # attach cbam
  if cbam:
    x = attach_cbam(x)
  return x


def res_block(x, filters, reps, strides, cbam):  # reps paremeter is added since resnet is a projection block followed by a repetition of identity 
  '''
  Input:
    x: input tensor
    filters: the number of filters -> int
    reps: number of repetitions -> int
    strides: the strides -> int

  Return:
  outputs tensor after applying (reps-1) number of identity block
  '''
  x = projection_block(x, filters, strides, cbam=cbam)
  for _ in range(reps-1):
    x = identity_block(x, filters=filters, cbam=cbam)
  return x


# build the whole ResNet50 model
def resnet50(img_size, cbam=True):
  input = Input(shape=(img_size, img_size, 3))
  x = conv_bn_relu(input, filters=64, kernel_size=7, strides=2)
  x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

  x = res_block(x, filters=64, reps=3, strides=1, cbam=cbam)
  x = res_block(x, filters=128, reps=4, strides=2, cbam=cbam)  
  x = res_block(x, filters=256, reps=6, strides=2, cbam=cbam)  
  x = res_block(x, filters=512, reps=3, strides=2, cbam=cbam)  

  x = GlobalAvgPool2D()(x)
  output = Dense(units=7, activation='softmax')(x)
  model = Model(input, output)
  return model

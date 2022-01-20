
# import the neccessary packages
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, MaxPool2D, GlobalAvgPool2D, Dense, GlobalMaxPooling2D, Activation, multiply, Concatenate, Reshape, Lambda
from tensorflow.keras import Model
import tensorflow.keras.backend as K

# implement channel attention
def channel_attention(x, ratio=8):
  '''
  # Arguments
    x: input tensor of 4D (batch, h, w, c)
    ratio(int): the reduction ratio for the hidden layer
  # Returns 
    x(tensor): the output tensor after appying channel attention module
  '''
  # get the channel
  channel = x.get_shape()[-1] if K.image_data_format() == 'channels_last' else 1
  #avg & max pooling
  avg_pool = GlobalAvgPool2D()(x)
  avg_pool = Reshape((1,1,channel))(avg_pool)
  max_pool = GlobalMaxPooling2D()(x)
  max_pool = Reshape((1,1,channel))(max_pool)
  assert avg_pool.get_shape()[1:] == (1, 1, channel)
  assert max_pool.get_shape()[1:] == (1, 1, channel)
  # shared dense layers
  shared_layer1 = Dense(units=channel//ratio,
                    activation='relu',
                    kernel_initializer='he_normal',
                    use_bias=True,
                    bias_initializer='zeros')
  
  shared_layer2 = Dense(units=channel, 
                     kernel_initializer='he_normal',
                     use_bias='zeros')

  avg_dense1 = shared_layer1(avg_pool)
  assert avg_dense1.get_shape()[1:] == (1, 1, channel//ratio)
  avg_dense2 = shared_layer2(avg_dense1)
  assert avg_dense2.get_shape()[1:] == (1, 1, channel)

  max_dense1 = shared_layer1(max_pool)
  assert max_dense1.get_shape()[1:] == (1, 1, channel//ratio)
  max_dense2 = shared_layer2(max_dense1)
  assert max_dense2.get_shape()[1:] == (1, 1, channel)

  cbam_feature = Add()([avg_dense2, max_dense2])
  cbam_feature = Activation('sigmoid')(cbam_feature)


  return multiply([cbam_feature, x])

# Test 
input_shape = (4, 64, 64, 128)
test_tensor = tf.random.normal(input_shape)
channel = test_tensor.get_shape()[-1] if K.image_data_format() == 'channels_last' else 1
out = channel_attention(test_tensor)
print(out.get_shape())


# implement spacial attention
def spatial_attention(x):
  '''
  # Arguments
    x(tensor): Input tensor of 3D
  # Return
    Tensor after applying spacial attention
  '''
  channel = x.get_shape()[-1]
  avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
  assert avg_pool.get_shape()[-1] == 1
  max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
  assert max_pool.get_shape()[-1] == 1

  concat = Concatenate(axis=3)([avg_pool, max_pool])
  cbam_feature = Conv2D(filters = 1,
					kernel_size=7,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
  cbam_feature = Activation('sigmoid')(cbam_feature)
     
  return multiply([x, cbam_feature],)


# Test 
input_shape = (4, 64, 64, 128)
test_tensor = tf.random.normal(input_shape)
channel = test_tensor.get_shape()[-1] if K.image_data_format() == 'channels_last' else 1
out = spatial_attention(test_tensor)
print(out.get_shape())

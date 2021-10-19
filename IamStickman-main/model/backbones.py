from blocks import residual_block, inverted_residual_block, conv_block, up_sample

import tensorflow as tf
import sys


def Identity(x):
	print("IDENTITY")
	return x

def create_vgg(x):
	x = tf.keras.layers.Conv2D(16,
		kernel_size=7,
		strides=4,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(32,
		kernel_size=3,
		strides=2,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(32,
		kernel_size=3,
		strides=1,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	return x

def create_resnet8(x):
	"""
	This architecture contains : 
	Image input is 224x224*3
	input_shape = (len(images), 224, 224, 3)
	"""
  
	x = conv_block(x, num_filters = 64, kernel_size = 7, strides = 2, activation="relu")

	x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding = 'same')(x)

	x = residual_block(x, k=2, kernel_size=3, num_filters=64)
	x = tf.keras.layers.Activation('relu')(x)
	x = residual_block(x, k=2, kernel_size=3, num_filters=64)
	x = tf.keras.layers.Activation('relu')(x)
	x = residual_block(x, k=2, kernel_size=3, num_filters=128)
	x = tf.keras.layers.Activation('relu')(x)
	x = residual_block(x, k=2, kernel_size=3, num_filters=128)
	x = tf.keras.layers.Activation('relu')(x)

	x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=None , padding="same", data_format=None)(x)

	return x

possible_backbones = {
	'VGG':create_vgg, 
	'Identity':Identity,
	'Resnet8' : create_resnet8,
}

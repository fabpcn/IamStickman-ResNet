import tensorflow as tf

def create_scalar_heads(x, num_key_points, scale):
	y = tf.keras.layers.Flatten()(x)
	#y = tf.keras.layers.Dense(units = 1024, activation = 'relu')(y)
	#y = tf.keras.layers.Dense(units = 200, activation = 'relu')(y)
	#y = tf.keras.layers.Dense(units = 84, activation = 'softmax')(y)
	outputs = scale * tf.keras.layers.Dense(
		2*num_key_points,
		activation='sigmoid',
		use_bias=False,
		kernel_initializer='he_normal')(y)
	return outputs

possible_heads = {
	'scalar' : create_scalar_heads,
}
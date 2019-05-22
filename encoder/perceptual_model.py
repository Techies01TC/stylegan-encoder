import numpy as np
import tensorflow as tf
import tensorflow.keras.models
from keras.preprocessing import image
from functools import partial
import dnnlib.tflib as tflib


def load_images(images_list, img_size):
	loaded_images = list()
	for img_path in images_list:
		img = image.load_img(img_path, target_size=(img_size, img_size))
		img = np.expand_dims(img, 0)
		loaded_images.append(img)
	loaded_images = np.vstack(loaded_images)

	# Reimplement tflib.convert_images_from_uint8 in numpy (with bugfix):
	preprocessed_images = np.copy(loaded_images).astype(dtype=np.float32)
	preprocessed_images = np.transpose(preprocessed_images, axes=(0, 3, 1, 2))
	drange = [-1,1]
	preprocessed_images = (preprocessed_images - drange[0]) * ((drange[1] - drange[0]) / 255) + drange[0]
	# ("+ drange[0]" at the end is a fix of a bug in tflib.convert_images_from_uint8())

	# NHWC --> NCHW:
	# preprocessed_images = tflib.convert_images_from_uint8(loaded_images, nhwc_to_nchw=True)
	# preprocessed_images = np.transpose(np.copy(loaded_images), axes=(0, 3, 1, 2))
	return loaded_images, preprocessed_images


def create_stub(name, batch_size):
	return tf.constant(0, dtype='float32', shape=(batch_size, 0))


class PerceptualDiscriminatorModel:
	def __init__(self, img_size, batch_size=1, sess=None):
		self.sess = tf.get_default_session() if sess is None else sess
		self.img_size = img_size
		self.batch_size = batch_size

		self.perceptual_model = None
		self.ref_img_features = None
		self.features_weight = None
		self.loss = None

	def build_perceptual_model(self, discriminator_network, generator_output_tensor, generated_image_tensor,
			vars_to_optimize,
			initial_learning_rate=0.05, learning_rate_decay_steps=175, learning_rate_decay_rate=0.5):

		def generated_image_tensor_fn(name):
			return generator_output_tensor

		discriminator_network.run(
				np.zeros((self.batch_size, 3, 1024, 1024)), None,
				minibatch_size=self.batch_size,
				custom_inputs=[generated_image_tensor_fn,
							  partial(create_stub, batch_size=self.batch_size)],
				structure='fixed')

		self.graph = tf.get_default_graph()
		
		# Learning rate
		global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
		incremented_global_step = tf.assign_add(global_step, 1)
		self._reset_global_step = tf.assign(global_step, 0)
		self.learning_rate = tf.train.exponential_decay(initial_learning_rate, incremented_global_step,
				learning_rate_decay_steps, learning_rate_decay_rate, staircase=True)
		self.sess.run([self._reset_global_step])

		self.discriminator_input = self.graph.get_tensor_by_name("D/_Run/D/images_in:0") # (?, 3, 1024, 1024)

		# Pull out a tensor from the discriminator net at each level of detail, including the raw image in:
		tensor_name_list = [
			"D/_Run/D/images_in:0",								# (1, 3, 1024, 1024)
			"D/_Run/D/1024x1024/Conv0/LeakyReLU/IdentityN:0",	# (1, 16, 1024, 1024)
			"D/_Run/D/512x512/Conv0/LeakyReLU/IdentityN:0",		# (1, 32, 512, 512)
			"D/_Run/D/256x256/Conv0/LeakyReLU/IdentityN:0",		# (1, 64, 256, 256)
			"D/_Run/D/128x128/Conv0/LeakyReLU/IdentityN:0",		# (1, 128, 128, 128)
			"D/_Run/D/64x64/Conv0/LeakyReLU/IdentityN:0",		# (1, 256, 64, 64)
			"D/_Run/D/32x32/Conv0/LeakyReLU/IdentityN:0",		# (1, 512, 32, 32)
			"D/_Run/D/16x16/Conv0/LeakyReLU/IdentityN:0",		# (1, 512, 16, 16)
			"D/_Run/D/8x8/Conv0/LeakyReLU/IdentityN:0",			# (1, 512, 8, 8)
			"D/_Run/D/4x4/Conv/LeakyReLU/IdentityN:0",			# (1, 512, 4, 4)
			"D/_Run/D/4x4/Dense0/LeakyReLU/IdentityN:0",		# (1, 512)
		]

		# Just mash them all together, unweighted, into one n-dimensional vector.
		# (I spent hours picking and choosing combinations of layers, doing weighted averages to
		# accommodate different layers being different sizes, even sinusoidally shuffling the weights
		# during training while lerping toward even weighting... none of it worked as well as this.)
		tensors = [tf.reshape(self.graph.get_tensor_by_name(t), shape=[-1]) for t in tensor_name_list]
		self.discriminator_output = tf.concat(tensors, axis=0)

		# Image
		generated_image = tf.image.resize_images(
				generated_image_tensor, (self.img_size, self.img_size), method=1)
		self.reference_image = tf.get_variable('ref_img', shape=generated_image.shape,
												dtype='float32', initializer=tf.initializers.zeros())
		self._assign_reference_image_ph = tf.placeholder(tf.float32, name="assign_ref_img_ph")
		self._assign_reference_image = tf.assign(self.reference_image, self._assign_reference_image_ph)

		# Perceptual image features
		generated_img_features = self.discriminator_output
		self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
												dtype='float32', initializer=tf.initializers.zeros())
		self._assign_reference_img_feat_ph = tf.placeholder(tf.float32, name="assign_ref_img_feat_ph")
		self._assign_reference_img_feat = tf.assign(self.ref_img_features, self._assign_reference_img_feat_ph)

		# Feature weights
		self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
											   dtype='float32', initializer=tf.initializers.zeros())

		self._assign_features_weight_ph = tf.placeholder(tf.float32, name="assign_features_weight_ph")
		self._assign_features_weight = tf.assign(self.features_weight, self._assign_features_weight_ph)
		self.sess.run([self.features_weight.initializer])

		# Loss
		self.loss = tf.losses.mean_squared_error(self.features_weight * self.ref_img_features,
												 self.features_weight * generated_img_features)

		# Also report L2 loss even though we don't optimize based on this op
		# (though we do include the raw image tensor in the tensors that we pulled from the discriminator).
		# Pixel values are [0, 255] but feature values are ~[-1, 1], so divide by 128^2:
		self.l2_loss = tf.losses.mean_squared_error(self.reference_image, generated_image) / 128**2

		# Optimizer
		vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.train_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
		self._init_optimizer_vars = tf.variables_initializer(optimizer.variables())

	def set_reference_images(self, images_list):
		assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
		loaded_image, preprocessed_images = load_images(images_list, self.img_size)
		image_features = self.sess.run([self.discriminator_output],
				{self.discriminator_input:preprocessed_images})
		image_features = image_features[0]

		# in case if number of images less than actual batch size
		# can be optimized further
		weight_mask = np.ones(self.features_weight.shape)
		if len(images_list) != self.batch_size:
			raise NotImplementedError("We don't support image lists not divisible by batch size.")
			features_space = list(self.features_weight.shape[1:])
			existing_features_shape = [len(images_list)] + features_space
			empty_features_shape = [self.batch_size - len(images_list)] + features_space

			existing_examples = np.ones(shape=existing_features_shape)
			empty_examples = np.zeros(shape=empty_features_shape)
			weight_mask = np.vstack([existing_examples, empty_examples])

			image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

		self.sess.run([self._assign_features_weight], {self._assign_features_weight_ph: weight_mask})
		self.sess.run([self._assign_reference_img_feat], {self._assign_reference_img_feat_ph: image_features})
		self.sess.run([self._assign_reference_image], {self._assign_reference_image_ph: loaded_image})

	def optimize(self, iterations):
		self.sess.run([self._init_optimizer_vars, self._reset_global_step])
		fetch_ops = [self.train_op, self.loss, self.l2_loss, self.learning_rate]
		for _ in range(iterations):
			_, loss, l2_loss, lr = self.sess.run(fetch_ops)
			yield {"loss":loss, "l2_loss":l2_loss, "lr": lr}


import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial


def create_stub(name, batch_size):
	return tf.constant(0, dtype='float32', shape=(batch_size, 0))


def create_variable_for_generator(name, batch_size, tiled_dlatent):
	if tiled_dlatent:
		low_dim_dlatent = tf.get_variable('learnable_dlatents',
			shape=(batch_size, 512),
			dtype='float32',
			initializer=tf.initializers.random_normal())
		return tf.tile(tf.expand_dims(low_dim_dlatent, axis=1), [1, 18, 1])
	else:
		return tf.get_variable('learnable_dlatents',
			shape=(batch_size, 18, 512),
			dtype='float32',
			initializer=tf.initializers.random_normal())


class Generator:
	def __init__(self, model, batch_size, tiled_dlatent, randomize_noise):
		self.batch_size = batch_size
		self.tiled_dlatent=tiled_dlatent

		if tiled_dlatent:
			self.initial_dlatents = np.zeros((self.batch_size, 512))
			model.components.synthesis.run(np.zeros((self.batch_size, 18, 512)),
				randomize_noise=randomize_noise, minibatch_size=self.batch_size,
				custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, tiled_dlatent=True),
												partial(create_stub, batch_size=batch_size)],
				structure='fixed')
		else:
			self.initial_dlatents = np.zeros((self.batch_size, 18, 512))
			model.components.synthesis.run(self.initial_dlatents,
				randomize_noise=randomize_noise, minibatch_size=self.batch_size,
				custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, tiled_dlatent=False),
												partial(create_stub, batch_size=batch_size)],
				structure='fixed')

		self.sess = tf.get_default_session()
		self.graph = tf.get_default_graph()

		self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
		self._assign_dlatent_ph = tf.placeholder(tf.float32, name="assign_dlatent_ph")
		self._assign_dlantent = tf.assign(self.dlatent_variable, self._assign_dlatent_ph)
		self.set_dlatents(self.initial_dlatents)

		try:
			self.generator_output = self.graph.get_tensor_by_name('G_synthesis_1/_Run/concat:0')
		except KeyError:
			# If we loaded only Gs and didn't load G or D, then scope "G_synthesis_1" won't exist in the graph.
			self.generator_output = self.graph.get_tensor_by_name('G_synthesis/_Run/concat:0')
		self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
		self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

		# Implement stochastic clipping similar to what is described in https://arxiv.org/abs/1702.04782
		# (Slightly different in that the latent space is normal gaussian here and was uniform in [-1, 1] in that paper,
		# so we clip any vector components outside of [-2, 2]. It seems fine, but I haven't done an ablation check.)
		clipping_mask = tf.math.logical_or(self.dlatent_variable > 2.0, self.dlatent_variable < -2.0)
		clipped_values = tf.where(clipping_mask, tf.random_normal(shape=self.dlatent_variable.shape), self.dlatent_variable)
		self.stochastic_clip_op = tf.assign(self.dlatent_variable, clipped_values)

	def reset_dlatents(self):
		self.set_dlatents(self.initial_dlatents)

	def set_dlatents(self, dlatents):
		if self.tiled_dlatent:
				assert (dlatents.shape == (self.batch_size, 512))
		else:
				assert (dlatents.shape == (self.batch_size, 18, 512))
		self.sess.run([self._assign_dlantent], {self._assign_dlatent_ph: dlatents})

	def stochastic_clip_dlatents(self):
		self.sess.run(self.stochastic_clip_op)

	def get_dlatents(self):
		return self.sess.run(self.dlatent_variable)

	def generate_images(self, dlatents=None):
		if dlatents is not None:
				self.set_dlatents(dlatents)
		return self.sess.run(self.generated_image_uint8)

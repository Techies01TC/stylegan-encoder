import os
import subprocess
import glob
import PIL
import PIL.ImageDraw
import PIL.ImageFont
import pickle
import numpy as np

import dnnlib.tflib as tflib

def image_generator_fn(latents, Gs, max_batch_size, image_size=None):
	fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
	# Generate a batch of images.
	for i in range(0, len(latents), max_batch_size):
		images = Gs.run(np.array(latents[i:min(len(latents), i + max_batch_size)]),
				None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
		for idx in range(images.shape[0]):
			if image_size is None:
				yield PIL.Image.fromarray(images[idx], 'RGB')
			else:
				yield PIL.Image.fromarray(images[idx], 'RGB').resize(image_size, resample=PIL.Image.LANCZOS)

def dlatents_image_generator_fn(dlatents, Gs, max_batch_size=25, image_size=None):
	synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
	if np.shape(dlatents)[1:] == (512,):
		# Broadcast dlatents from [?, 512] to [?, 18, 512].
		broadcast_shape = Gs.components.synthesis.input_shape
		broadcast_shape[0] = len(dlatents)
		dlatents = np.broadcast_to(np.expand_dims(dlatents, axis=1), broadcast_shape)
	elif np.shape(dlatents)[1:] != (18, 512):
		raise ValueError("dlatents shape was {} but should be [?, 512] or [?, 18, 512]".format(np.shape(dlatents)))
	# Generate a batch of images.
	for i in range(0, len(dlatents), max_batch_size):
		dlatents_batch = dlatents[i:min(len(dlatents), i + max_batch_size)]
		images = Gs.components.synthesis.run(dlatents_batch, randomize_noise=False, **synthesis_kwargs)
		for idx in range(images.shape[0]):
			if image_size is None:
				yield PIL.Image.fromarray(images[idx], 'RGB')
			else:
				yield PIL.Image.fromarray(images[idx], 'RGB').resize(image_size, resample=PIL.Image.LANCZOS)

def save_video(image_generator, vid_name, total_frames=None):
	# See https://stackoverflow.com/questions/43650860/pipe-pil-images-to-ffmpeg-stdin-python
	path = '{}.mp4'.format(vid_name)
	if os.path.exists(path):
		os.remove(path)
	cmd_out = ['ffmpeg',
			'-f', 'image2pipe',
			'-r', '30',  # FPS 
			'-i', '-',  # Indicated input comes from pipe
			'-c:v', 'libx264',
			'-profile:v', 'high',
			'-crf', '20',
			'-pix_fmt', 'yuv420p',
			path]
	# Remove the `stderr=subprocess.DEVNULL` param to debug:
	pipe = subprocess.Popen(cmd_out, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	for i, image in enumerate(image_generator):
		image.save(pipe.stdin, 'BMP')
		if total_frames is None:
			print("\rSaved frame {:,}".format(i+1), end="")
		else:
			print("\rSaved frame {:,}/{:,} ({}%)".format(i+1, total_frames, int(100 * (i + 1) / total_frames)), end="")
	pipe.stdin.close()
	pipe.wait()
	# Make sure all went well
	if pipe.returncode != 0:
		raise subprocess.CalledProcessError(pipe.returncode, cmd_out)
	else:
		print("\nSaved {}.".format(path))

def save_images_to_grid(out_path, image_generator, image_count, grid_columns, image_size, with_numbers=True):
	grid_rows = int((image_count + grid_columns - 1) / grid_columns)
	canvas = PIL.Image.new('RGB', (image_size[0] * grid_columns, image_size[1] * grid_rows), 'white')
	if with_numbers:
		font = PIL.ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 15)
	draw_canvas = PIL.ImageDraw.Draw(canvas)
	for i, image in enumerate(image_generator):
		canvas_loc = ((i % grid_columns) * image_size[0], int(i / grid_columns) * image_size[1])
		canvas.paste(image.resize(image_size, resample=PIL.Image.LANCZOS), canvas_loc)
		label_loc = (canvas_loc[0] + 10, canvas_loc[1] + image_size[1] - 25)
		if with_numbers:
			draw_canvas.text(label_loc, str(i), font=font, fill=(255, 255, 255))
		print("\rSaved image {}/{}".format(i+1, image_count), end="")
	canvas.save(out_path)

def image_grid_generator_fn(image_generator_list, grid_columns, image_size):
	grid_rows = int((len(image_generator_list) + grid_columns - 1) / grid_columns)
	for images in zip(*image_generator_list):
		canvas = PIL.Image.new('RGB', (image_size[0] * grid_columns, image_size[1] * grid_rows), 'white')
		draw_canvas = PIL.ImageDraw.Draw(canvas)
		for i, image in enumerate(images):
			canvas_loc = ((i % grid_columns) * image_size[0], int(i / grid_columns) * image_size[1])
			canvas.paste(image.resize(image_size, resample=PIL.Image.LANCZOS), canvas_loc)
		del draw_canvas
		yield canvas

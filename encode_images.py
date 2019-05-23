import os
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
import bw_utils
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualDiscriminatorModel

def split_to_batches(l, n):
	for i in range(0, len(l), n):
		yield l[i:i + n]

def main():
	src_dir = os.path.join("output", "aligned_images")
	generated_images_dir = os.path.join("output", "generated_images")
	generated_videos_dir = os.path.join("output", "generated_videos")
	dlatent_dir = os.path.join("output", "latent_representations")

	# for now it's unclear if larger batch leads to better performance/quality
	# Also, I may have broken >1 batch sizes, but happily they didn't seem to provide meaningful time savings anyway.
	batch_size = 1

	# Perceptual model params
	image_size = 1024
	iterations = 600
	# Generator params
	randomize_noise = False

	ref_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
	ref_images = list(sorted(filter(os.path.isfile, ref_images)))

	if len(ref_images) == 0:
		raise Exception('%s is empty' % src_dir)

	os.makedirs(generated_images_dir, exist_ok=True)
	os.makedirs(dlatent_dir, exist_ok=True)

	# Initialize generator and perceptual model
	tflib.init_tf()
	# I saved the FFHQ network as a pickle file to my hard drive to avoid relying on the Nvidia Google Drive share.
	local_network_path = "karras2019stylegan-ffhq-1024x1024.pkl"
	if os.path.exists(local_network_path):
		with open(local_network_path, "rb") as f:
			generator_network, discriminator_network, Gs_network = pickle.load(f)
	else:
		URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
		with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
			generator_network, discriminator_network, Gs_network = pickle.load(f)

	# Set tiled_dlatent=False if you want to generate an 18x512 dlatent like in Puzer's original repo.
	# Set tiled_dlatent=True if you want to generate a 1x512 dlatent (subsequently tiled back to 18x512)
	# like the mapping network outputs.
	generator = Generator(Gs_network, batch_size, randomize_noise=randomize_noise, tiled_dlatent=True)
	perceptual_model = PerceptualDiscriminatorModel(image_size, batch_size=batch_size)
	perceptual_model.build_perceptual_model(discriminator_network,
			generator.generator_output, generator.generated_image, generator.dlatent_variable)

	# Optimize (only) dlatents by minimizing perceptual loss
	# between reference and generated images in feature space
	images = []
	video_frames = 100 # Set to >0 to save a video of the training, or to 0 to disable.
	if video_frames > 0:
		steps_per_frame = iterations / video_frames
		steps_until_frame = 0
	for images_batch in split_to_batches(ref_images, batch_size):
		names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
		perceptual_model.set_reference_images(images_batch)
		op = perceptual_model.optimize(iterations=iterations)
		pbar = tqdm(op, leave=False, total=iterations)
		best_loss = None
		best_dlatent = None
		dlatent_frames = []
		for loss_dict in pbar:
			pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v)
					for k, v in loss_dict.items()]))
			if best_loss is None or loss_dict["loss"] < best_loss:
				best_loss = loss_dict["loss"]
				best_dlatent = generator.get_dlatents()
			if video_frames > 0:
				# If we're recording a video, consider taking a dlatent snapshot for later assembly.
				if steps_until_frame <= 0:
					dlatent_frames.append(generator.get_dlatents()[0])
					steps_until_frame += steps_per_frame
				steps_until_frame -= 1.
			generator.stochastic_clip_dlatents()
		print(" ".join(names), " Loss {:.4f}".format(best_loss))

		# Generate images from found dlatents and save them.
		generated_images = generator.generate_images(dlatents=best_dlatent)
		generated_dlatents = generator.get_dlatents()
		for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
			img = PIL.Image.fromarray(img_array, 'RGB')
			images.append(PIL.Image.open(os.path.join(src_dir, "{}.png".format(img_name))))
			images.append(img)
			img.save(os.path.join(generated_images_dir, '{}.png'.format(img_name)), 'PNG')
			np.save(os.path.join(dlatent_dir, '{}.npy'.format(img_name)), dlatent)
		generator.reset_dlatents()
		bw_utils.save_images_to_grid(os.path.join(generated_images_dir, "grid.png"),
				images, len(images), 2, (1024, 1024), with_numbers=False)

		# Save video of training
		if video_frames > 0:
			os.makedirs(generated_videos_dir, exist_ok=True)
			video_name = os.path.join(generated_videos_dir, " ".join(names))
			# np.save(video_name + ".npy", np.array(dlatent_frames))
			# print("Saved dlatent video frames as {}.".format(video_name + ".npy"))
			image_generator = bw_utils.dlatents_image_generator_fn(dlatent_frames, Gs_network)
			bw_utils.save_video(image_generator, video_name)

if __name__ == "__main__":
	main()

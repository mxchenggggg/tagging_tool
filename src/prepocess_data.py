import cv2
import numpy as np 
import os, glob, pickle
from scipy.misc import imresize, imread

def read_img(image_file,tag=1):
	return cv2.imread(image_file, tag)

def fill_empty(image):
	musk_0 = (image<10)
	musk_1 = (image>=10)
	median = np.median(np.extract(musk_1, image))
	image += musk_0*np.array([median], dtype=np.uint8)
	return image

def main():
	image_folder = '/media/ac12/Data/tagging_tool/images'
	labels_folder = '/media/ac12/Data/tagging_tool/labels'
	tags_folder = '/media/ac12/Data/tagging_tool/tag'

	cracked_folder = '/media/ac12/Data/tagging_tool/cracked_relabeled'
	not_cracked_folder = '/media/ac12/Data/tagging_tool/not_cracked_relabeled'

	shadow_folder = '/media/ac12/Data/tagging_tool/shadow_relabeled'
	not_shadow_folder = '/media/ac12/Data/tagging_tool/not_shadow_relabeled'

	# images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
	# labels = sorted(glob.glob(os.path.join(labels_folder, '*color.png')))
	# tags_files = sorted(glob.glob(os.path.join(tags_folder, '*.txt')))
	# tags_dic = {}
	# for f in tags_files:
	# 	with open(f) as tags:
	# 		line = tags.readline()
	# 		while True:
	# 			line = tags.readline()
	# 			if not line:
	# 				break
	# 			tags_dic[line.split()[0]] = int(line.split()[16])
	
	# for i in range(len(images)):
	# 	img = read_img(images[i]).astype(np.uint8)
	# 	label = read_img(labels[i], 0)
	# 	label_musk = (label == 70).astype(np.uint8)
	# 	label_musk = np.repeat(label_musk.reshape((label_musk.shape[0], label_musk.shape[1], -1)), 3, axis=2)
	# 	img_musk = label_musk * img
	# 	if tags_dic[os.path.basename(images[i])] == 1:
	# 		name = os.path.join(shadow_folder ,str(i).zfill(4)+'.jpg')
	# 	else:
	# 		name = os.path.join(not_shadow_folder ,str(i).zfill(4)+'.jpg')
	# 	cv2.imwrite(name, img_musk)

	#### cracked ####
	# resized_folder = '/media/ac12/Data/tagging_tool/resized_new'

	# cra_imgs = sorted(glob.glob(os.path.join(cracked_folder, '*.jpg')))
	# not_cra_imgs = sorted(glob.glob(os.path.join(not_cracked_folder, '*.jpg')))
	# cracked = set()
	# print(len(cra_imgs), len(not_cra_imgs))

	# for c in cra_imgs:
	# 	basename = os.path.basename(c)
	# 	cracked.add(basename)
	# 	resized_name = os.path.join(resized_folder, basename)
	# 	img = fill_empty(imread(c, mode='L')[540:, :])
	# 	resized = imresize(img, (180, 640), interp='bicubic')
	# 	cv2.imwrite(resized_name, resized)
	# pickle.dump(cracked, open(os.path.join(resized_folder, 'cracked.p'), 'wb'))
	
	# for nc in not_cra_imgs[len(not_cra_imgs)-len(cra_imgs):]:
	# 	basename = os.path.basename(nc)
	# 	resized_name = os.path.join(resized_folder, basename)
	# 	img = fill_empty(imread(nc, mode='L')[540:, :])
	# 	resized = imresize(img, (180, 640), interp='bicubic')
	# 	cv2.imwrite(resized_name, resized)

	#### shadow ####
	resized_shadow_folder = '/media/ac12/Data/tagging_tool/resized_shadow'

	sha_imgs = sorted(glob.glob(os.path.join(shadow_folder, '*.jpg')))
	not_sha_imgs = sorted(glob.glob(os.path.join(not_shadow_folder, '*.jpg')))
	shadow = set()
	print(len(sha_imgs), len(not_sha_imgs))

	for s in sha_imgs:
		basename = os.path.basename(s)
		shadow.add(basename)
		resized_name = os.path.join(resized_shadow_folder, basename)
		img = fill_empty(imread(s, mode='L')[540:, :])
		resized = imresize(img, (180, 640), interp='bicubic')
		cv2.imwrite(resized_name, resized)
	pickle.dump(shadow, open(os.path.join(resized_shadow_folder, 'shadow.p'), 'wb'))

	for ns in not_sha_imgs:
		basename = os.path.basename(ns)
		resized_name = os.path.join(resized_shadow_folder, basename)
		img = fill_empty(imread(ns, mode='L')[540:, :])
		resized = imresize(img, (180, 640), interp='bicubic')
		cv2.imwrite(resized_name, resized)

if __name__ == "__main__":
	main()
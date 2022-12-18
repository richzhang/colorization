
# # python video_colorization.py -i input.jpg -o output.jpg

# import argparse
# import matplotlib.pyplot as plt
# from colorizers import *

# parser = argparse.ArgumentParser()
# parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
# parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
# parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
# opt = parser.parse_args()

# colorizer_eccv16 = eccv16(pretrained=True).eval()
# colorizer_siggraph17 = siggraph17(pretrained=True).eval()
# if(opt.use_gpu):
# 	colorizer_eccv16.cuda()
# 	colorizer_siggraph17.cuda()

# img = load_img('vid/%s'%opt.img_path)
# (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
# if(opt.use_gpu):
# 	tens_l_rs = tens_l_rs.cuda()

# img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig), dim=1))

# out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
# plt.imsave('vid_out/output_eccv16.png', out_img_eccv16)

# # out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
# # plt.imsave('vid_out/output_siggraph17.png', out_img_siggraph17)

# ------------------------------

# # https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames/33399711#33399711

# import cv2
# vidcap = cv2.VideoCapture('vid/big_buck_bunny_720p_5mb.mp4')
# success,image = vidcap.read()
# count = 0

# while success:
#   cv2.imwrite("vid_out/frame%d.jpg" % count, image)     # save frame as JPEG file      
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1

# ---------------------------------

# https://www.geeksforgeeks.org/python-create-video-using-multiple-images-using-opencv/

# import os
# import cv2
# from PIL import Image

# print(os.getcwd())
# os.chdir(r"C:\Users\Vicky\Desktop\Repository\colorization\vid_out")
# path = r"C:\Users\Vicky\Desktop\Repository\colorization\vid_out"

# mean_height = 0
# mean_width = 0

# num_of_images = len(os.listdir('.'))
# print(num_of_images)

# for file in os.listdir('.'):
# 	im = Image.open(os.path.join(path, file))
# 	width, height = im.size
# 	mean_width += width
# 	mean_height += height

# mean_width = int(mean_width / num_of_images)
# mean_height = int(mean_height / num_of_images)

# for file in os.listdir('.'):
# 	if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
# 		im = Image.open(os.path.join(path, file))

# 		width, height = im.size
# 		imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
# 		imResize.save( file, 'JPEG', quality = 95)
# 		# print(im.filename.split('\\')[-1], " is resized")

# def generate_video():
# 	image_folder = r'C:\Users\Vicky\Desktop\Repository\colorization\vid_out'
# 	video_name = 'mygeneratedvideo.avi'
# 	os.chdir(r"C:\Users\Vicky\Desktop\Repository\colorization\vid_out")
	
# 	images = [img for img in os.listdir(image_folder)
# 			if img.endswith(".jpg") or
# 				img.endswith(".jpeg") or
# 				img.endswith("png")]
	
# 	frame = cv2.imread(os.path.join(image_folder, images[0]))
# 	height, width, layers = frame.shape
# 	video = cv2.VideoWriter(video_name, 0, 1, (width, height))

# 	for image in images:
# 		video.write(cv2.imread(os.path.join(image_folder, image)))
	
# 	cv2.destroyAllWindows()
# 	video.release()

# generate_video()

# ---------------------------------------

# https://stackoverflow.com/questions/63631973/how-can-i-use-python-to-speed-up-a-video-without-dropping-frames/63632689#63632689

from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

in_loc = 'vid_out/mygeneratedvideo.avi'
out_loc = 'vid/dummy_out.mp4'

# Import video clip
clip = VideoFileClip(in_loc)
print("fps: {}".format(clip.fps))

# Modify the FPS
clip = clip.set_fps(clip.fps * 3)

# Apply speed up
final = clip.fx(vfx.speedx, 25)
print("fps: {}".format(final.fps))

# Save video clip
final.write_videofile(out_loc)
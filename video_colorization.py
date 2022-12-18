
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


import cv2
vidcap = cv2.VideoCapture('vid/big_buck_bunny_720p_5mb.mp4')
success,image = vidcap.read()
count = 0

while success:
  cv2.imwrite("vid_out/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
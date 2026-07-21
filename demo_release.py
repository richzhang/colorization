
import argparse
import matplotlib.pyplot as plt

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
# TODO can I load pretrained weights for attention26 if I added a layer?
colorizer_attention26 = attention26(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_attention26.cuda()


# 2. Freeze early layers (adjust this list to match your definition of "early")
modules_to_freeze = [colorizer_attention26.model1, colorizer_attention26.model2, colorizer_attention26.model3, colorizer_attention26.model4]

for module in modules_to_freeze:
    for param in module.parameters():
        param.requires_grad = False
# BatchNorm stats: If you freeze model1–model4, their BN layers will still update during train(). 
# To fully freeze them, call module.eval() on those blocks during training, 
# or use torch.nn.SyncBatchNorm.convert_sync_batchnorm() with track_running_stats=False

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_attention26 = postprocess_tens(tens_l_orig, colorizer_attention26(tens_l_rs).cpu())

# plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
# plt.imsave('%s_attention26.png'%opt.save_prefix, out_img_attention26)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(out_img_attention26)
plt.title('Output (ATTENTION 26)')
plt.axis('off')
plt.show()

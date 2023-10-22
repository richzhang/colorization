from PIL import Image
import typing as T
import numpy as np
import torch
from skimage import color
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from colorizers.siggraph17 import siggraph17
from colorizers.eccv16 import eccv16
from colorizers.modified import modified_colorizer

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(img.resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:,:,0]
    img_ab_orig = img_lab_orig[:,:,1:3]

    img_l_rs = img_lab_rs[:,:,0]
    img_ab_rs = img_lab_rs[:,:,1:3]

    tens_orig_l = torch.Tensor(img_l_orig)[None,:,:]
    tens_orig_ab = torch.Tensor(img_ab_orig.transpose((2,0,1)))

    tens_rs_l = torch.Tensor(img_l_rs)[None,:,:]
    tens_rs_ab = torch.Tensor(img_ab_rs.transpose((2,0,1)))
    tens_rs_ab = torch.randint(0, 313, (256, 256))

    return tens_orig_l, tens_orig_ab, tens_rs_l, tens_rs_ab

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return preprocess_img(img, HW=(256,256))

    def __len__(self):
        return 64

def mock_trainloader(batch_size: int = 8, num_workers: int = 0):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    trainset = MyDataset(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader


def build_optimizer(type: str, args: T.Dict[str, T.Any]) -> torch.optim.Optimizer:
    """
    Build an optimizer from a string and a set of arguments.
    """
    return getattr(torch.optim, type)(**args)

def build_criterion(type: str, args: T.Dict[str, T.Any]) -> torch.nn.Module:
    """
    Build a criterion from a string and a set of arguments.
    """
    return getattr(torch.nn, type)(**args)


def train(net, optimizer, trainloader, device, criterion, n_epochs):
    net.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        desc = 'Epoch %d/%d' % (epoch + 1, n_epochs)
        bar_fmt = '{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]'
        with tqdm(desc=desc, total=len(trainloader), leave=True, miniters=1, unit='ex',
                unit_scale=True, bar_format=bar_fmt, position=0) as progbar:
            for i, data in enumerate(trainloader):
                _, _, rs_l, rs_ab = data
                batch_size = rs_l.shape[0]
                inputs = rs_l.to(device)
                labels = rs_ab.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()/batch_size
                progbar.set_postfix({'loss': '%.3g' % (running_loss / (i+1))})
                progbar.update(1)

    print('\nFinished Training')

#TODO(Sebastian) implement this. Requires determining the Q=313 buckets of ab 
# colors from training data (in-gamut colors).
def mock_color_label_to_ab(label: int):
    ab = torch.rand((2)) * 200 - 100
    return ab

#TODO(Sebastian) Got lazy here. This is not the correct way to index tensors.
# will do this once above is done.
def convet_output_to_rgb(out_ab_dist, orig_l):
    out_ab_choice = out_ab_dist.argmax(dim=1)
    B, H, W = out_ab_choice.shape
    out_ab = torch.zeros((B, 2, H, W))
    for b in range(B):
        for h in range(H):
            for w in range(W):
                out_ab[b, :, h, w] = mock_color_label_to_ab(out_ab_choice[b, h, w].item())
    return postprocess_tens(orig_l, out_ab)

def eval_model(net, dataloader, device, criterion):
    net.eval()
    with torch.no_grad():
        running_loss = 0.0
        for data in dataloader:
            _, _, rs_l, rs_ab = data
            batch_size = rs_l.shape[0]
            inputs = rs_l.to(device)
            labels = rs_ab.to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()/batch_size
    return running_loss/len(dataloader)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#NOTE(Sebastian) This training pipeline only works for the modified colorizer.
# MSE loss can be used for the other two models but the output needs to be
# unnormalized first. Currently, neither pretrained model retunrs an ouput 
# distribution over the ab color space.

trainloader = mock_trainloader()
net = modified_colorizer()
net.to(device)
optimizer = build_optimizer('Adam', {'params': net.parameters(), 'lr': 0.001})
criterion = build_criterion('CrossEntropyLoss', {})
n_epochs = 1
train(net, optimizer, trainloader, device, criterion, n_epochs)
eval_model(net, trainloader, device, criterion)

# batch = next(iter(trainloader))
# orig_l, orig_ab, rs_l, rs_ab = batch
# out_ab_dist = net(rs_l)
# out_img = convet_output_to_rgb(out_ab_dist, orig_l)
# import matplotlib.pyplot as plt
# plt.imshow(out_img)
# plt.show()



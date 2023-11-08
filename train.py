from PIL import Image
from pathlib import Path
import typing as T
import numpy as np
import torch
from skimage import color
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorizers.modified import modified_colorizer
from torch.utils.data import DataLoader, Dataset

def resize_img(img: Image, HW: T.Tuple[int, int] = (256,256), resample: int = 3) -> np.ndarray:
    """
    Resize an image to a given size.
    """
    return np.asarray(img.resize((HW[1],HW[0]), resample=resample))

def preprocess_img(
        img_rgb_orig: Image, HW: T.Tuple[int, int] = (256,256), resample: int = 3
    ) -> T.Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Preprocess an image for training.
    """
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

def postprocess_tens(tens_orig_l: Tensor, out_ab: Tensor, mode='bilinear') -> np.ndarray:
    """
    Postprocess a tensor for visualization.
    """
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode)
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

class MyDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, index: int) -> T.Tuple[Tensor, Tensor, Tensor, Tensor]:
        img, _ = self.dataset[index]
        return preprocess_img(img, HW=(256,256))

    def __len__(self):
        return len(self.dataset)

def mock_trainloader(batch_size: int = 8, num_workers: int = 0) -> DataLoader:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    trainset = MyDataset(trainset)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader


def build_optimizer(type: str, args: T.Dict[str, T.Any]) -> Optimizer:
    """
    Build an optimizer from a string and a set of arguments.
    """
    return getattr(torch.optim, type)(**args)

def build_criterion(type: str, args: T.Dict[str, T.Any]) -> nn.Module:
    """
    Build a criterion from a string and a set of arguments.
    """
    return getattr(torch.nn, type)(**args)

class TrainingLogger:
    def __init__(self) -> None:
        self.train_loss = []
        self.eval_loss = []

    def log_train_loss(self, iteration: int, loss: float) -> None:
        """
        Log the training loss at a given iteration.
        """
        self.train_loss.append((iteration, loss))

    def log_eval_loss(self, iteration: int, loss: float) -> None:
        """
        Log the evaluation loss at a given iteration.
        """
        self.eval_loss.append((iteration, loss))

    def save_plot(self, output_dir: Path, model_name: str) -> None:
        """
        Save a plot of the training and evaluation loss.
        """
        plt.plot(*zip(*self.train_loss), label='train', color='blue')
        plt.plot(*zip(*self.eval_loss), label='eval', color='orange')
        plt.legend()
        plt.savefig(output_dir / f'{model_name}_loss.png')

def eval(
    net: nn.Module, dataloader: DataLoader, device: torch.device, criterion: nn.Module
) -> float:
    """
    Evaluate a model on a dataset using a given criterion. Return the average loss.
    """
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


def train(
    net: nn.Module, optimizer: Optimizer, trainloader: DataLoader, testloader: DataLoader, 
    device: torch.device, criterion: nn.Module, n_epochs: int, logger: TrainingLogger = None
) -> T.Tuple[float, T.Dict[str, T.Any]]:
    """
    Train a model on a dataset using a given criterion and optimizer. Return the best
    evaluation loss and the best model parameters during training.
    """
    best_eval_loss = np.inf
    best_model = None

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
                iter_loss = loss.item()/batch_size
                if logger is not None:
                    logger.log_train_loss(i+1, iter_loss)
                running_loss += iter_loss
                progbar.set_postfix({'loss': '%.3g' % (running_loss / (i+1))})
                progbar.update(1)
            
            eval_loss = eval(net, testloader, device, criterion)
            if logger is not None:
                logger.log_eval_loss(i+1, eval_loss)
            progbar.set_postfix({'loss': '%.3g' % (running_loss / (i+1)), 'eval_loss': '%.3g' % eval_loss})
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model = net.state_dict()
    print('\nFinished Training')
    return best_eval_loss, best_model

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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#NOTE(Sebastian) This training pipeline only works for the modified colorizer.
# MSE loss can be used for the other two models but the output needs to be
# unnormalized first. Currently, neither pretrained model retunrs an ouput 
# distribution over the ab color space.

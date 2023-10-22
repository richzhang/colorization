
import torch.nn as nn
from .layers import build_basic_block
from .base_color import *

class ModifiedColorizer(BaseColor):
    def __init__(self):
        super(ModifiedColorizer, self).__init__()

        self.model1 = build_basic_block(channels=[1, 64, 64], kernel_size=3, stride=[1, 2])
        self.model2 = build_basic_block(channels=[64, 128, 128], kernel_size=3, stride=[1, 2])
        self.model3 = build_basic_block(channels=[128, 256, 256, 256], kernel_size=3, stride=[1, 1, 2])
        self.model4 = build_basic_block(channels=[256, 512, 512, 512], kernel_size=3)
        self.model5 = build_basic_block(channels=[512, 512, 512, 512], kernel_size=3, dilation=2, padding=2)
        self.model6 = build_basic_block(channels=[512, 512, 512, 512], kernel_size=3, dilation=2, padding=2)
        self.model7 = build_basic_block(channels=[512, 512, 512, 512], kernel_size=3)
        self.model8 = build_basic_block(
              channels=[512, 256, 256, 256], kernel_size=[4, 3, 3], stride=[2, 1, 1], 
              norm_layer=False, conv_type=[nn.ConvTranspose2d, nn.Conv2d, nn.Conv2d]
        )
        self.model9 = build_basic_block(
              channels=[256, 256, 256, 256], kernel_size=[4, 3, 3], stride=[2, 1, 1], 
              norm_layer=False, conv_type=[nn.ConvTranspose2d, nn.Conv2d, nn.Conv2d]
        )
        self.model10 = build_basic_block(
              channels=[256, 128, 128, 128], kernel_size=[4, 3, 3], stride=[2, 1, 1], 
              norm_layer=False, conv_type=[nn.ConvTranspose2d, nn.Conv2d, nn.Conv2d]
        )
        self.model11 = build_basic_block(
              channels=[128, 64, 64, 64], kernel_size=[4, 3, 3], stride=[2, 1, 1], 
              norm_layer=False, conv_type=[nn.ConvTranspose2d, nn.Conv2d, nn.Conv2d]
        )
        self.model11.append(nn.Conv2d(64, 313, kernel_size=1, stride=1, padding=0, bias=True))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        conv9_3 = self.model9(conv8_3)
        conv10_3 = self.model10(conv9_3)
        conv11_3 = self.model11(conv10_3)
        out_reg = self.softmax(conv11_3)
        return out_reg

def modified_colorizer():
	model = ModifiedColorizer()
	return model

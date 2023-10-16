
import torch.nn as nn
from .layers import build_basic_block
from .base_color import *

class ECCVGenerator(BaseColor):
    def __init__(self):
        super(ECCVGenerator, self).__init__()

        self.model1 = build_basic_block(channels=[1, 64, 64], kernel_size=3, stride=[1, 2])
        self.model2 = build_basic_block(channels=[64, 128, 128], kernel_size=3, stride=[1, 2])
        self.model3 = build_basic_block(channels=[128, 256, 256, 256], kernel_size=3, stride=[1, 2, 2])
        self.model4 = build_basic_block(channels=[256, 512, 512, 512], kernel_size=3)
        self.model5 = build_basic_block(channels=[512, 512, 512, 512], kernel_size=3, dilation=2, padding=2)
        self.model6 = build_basic_block(channels=[512, 512, 512, 512], kernel_size=3, dilation=2, padding=2)
        self.model7 = build_basic_block(channels=[512, 512, 512, 512], kernel_size=3)
        self.model8 = build_basic_block(
              channels=[512, 256, 256, 256], kernel_size=[4, 3, 3], stride=[2, 1, 1], 
              norm_layer=False, conv_type=[nn.ConvTranspose2d, nn.Conv2d, nn.Conv2d]
        )
        self.model8.append(nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True))

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

def eccv16(pretrained=True):
	model = ECCVGenerator()
	if(pretrained):
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model

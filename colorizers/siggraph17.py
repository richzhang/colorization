import torch
import torch.nn as nn
from .layers import build_basic_block
from .base_color import *

# NOTE(Sebastian): Not a super neat way to update this model architecture 
# without messing up loading of pretrained weights. For hyperparemeter search
# it is still possible the build_layer function from layers.py by setting 
# configs. Same could be apply to this model for the sake of consistency, but 
# it's not super worth it.

class SIGGRAPHGenerator(BaseColor):
    def __init__(self, classes=529):
        super(SIGGRAPHGenerator, self).__init__()

        self.model1 = build_basic_block(channels=[4, 64, 64], kernel_size=3)
        # add a subsampling operation
        self.model2 = build_basic_block(channels=[64, 128, 128], kernel_size=3)
        # add a subsampling layer operation
        self.model3 = build_basic_block(channels=[128, 256, 256, 256], kernel_size=3)
        # add a subsampling layer operation
        self.model4 = build_basic_block(channels=[256, 512, 512, 512], kernel_size=3)
        self.model5 = build_basic_block(channels=[512, 512, 512, 512], kernel_size=3, dilation=2, padding=2)
        self.model6 = build_basic_block(channels=[512, 512, 512, 512], kernel_size=3, dilation=2, padding=2)
        self.model7 = build_basic_block(channels=[512, 512, 512, 512], kernel_size=3)
        self.model8up = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True))
        self.model3short8 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True))
        self.model8 = build_basic_block(channels=[256, 256, 256], kernel_size=3, init_relu=True)
        self.model9up = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True))
        self.model2short9 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True))
        # add the two feature maps above
        self.model9 = build_basic_block(channels=[128, 128], kernel_size=3, init_relu=True)
        self.model10up = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True))
        self.model1short10 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True))
        # add the two feature maps above
        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=.2),
        )
        # classification output
        self.model_class = nn.Sequential(nn.Conv2d(256, classes, kernel_size=1, stride=1, bias=True))
        # regression output
        self.model_out = nn.Sequential(nn.Conv2d(128, 2, kernel_size=1, stride=1, bias=True), nn.Tanh())
        self.upsample4 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'))
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

    def forward(self, input_A, input_B=None, mask_B=None):
        if(input_B is None):
            input_B = torch.cat((input_A*0, input_A*0), dim=1)
        if(mask_B is None):
            mask_B = input_A*0

        conv1_2 = self.model1(torch.cat((self.normalize_l(input_A),self.normalize_ab(input_B),mask_B),dim=1))
        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        return self.unnormalize_ab(out_reg)

def siggraph17(pretrained=True):
    model = SIGGRAPHGenerator()
    if(pretrained):
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth',map_location='cpu',check_hash=True))
    return model


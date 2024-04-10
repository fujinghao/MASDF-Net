from lib.backbone.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.ops.Criss_Cross_Attention import CrissCrossAttention as SIG
from mmcv.cnn import constant_init, kaiming_init


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3),padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel, n_class):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = BasicConv2d(3*channel, n_class, 1, padding=0)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(2, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x  # [N, C, H, W]
            input_x = input_x.view(batch, channel, height * width)  # [N, C, H*W]
            input_x = input_x.unsqueeze(1)  # [N, 1, C, H*W]
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            c = torch.cat([avg_out, max_out], dim=1)
            context_mask = self.conv_mask(c)  # [N, 1, H, W]
            cm = context_mask
            context_mask = context_mask.view(batch, 1, height * width)  # [N, 1, H*W]
            context_mask = self.softmax(context_mask)  # [N, 1, H*W]
            context_mask = context_mask.unsqueeze(-1)  # [N, 1, H*W, 1]
            context = torch.matmul(input_x, context_mask)  # [N, 1, C, H*W] * [N, 1, H*W, 1] = [N, 1, C, 1]
            context = context.view(batch, channel, 1, 1)  # [N, C, 1, 1]
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        y = out
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class Laplace(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(Laplace, self).__init__()
        kernel = [[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).expand(ou_ch, in_ch, 3, 3)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, tensor_image):
        x = torch.nn.functional.conv2d(tensor_image, self.weight, stride=1, padding=1)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride=1, padding=0, dilation=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class BasicConv2d(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size = 1, stride=1, padding =0,dilation = 1,  bn=True, relu=True, bias=True):
        super(BasicConv2d, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size = kernel_size, stride = stride, padding=padding, bias=bias, dilation=dilation)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * input


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Scale_Aware(nn.Module):
    def __init__(self, in_channels):
        super(Scale_Aware, self).__init__()

        # self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels)])
        self.conv1x1 = nn.ModuleList(
            [nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0),
             nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)])
        self.conv3x3_1 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1)])
        self.conv3x3_2 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1)])
        self.conv_last =  BasicConv2d(in_channels, in_channels, kernel_size=1, dilation=1, padding=0,relu=True, bn=True),

        self.relu = nn.ReLU()
    def forward(self, x_l, x_h):
        feat = torch.cat([x_l, x_h], dim=1)
        # feat=feat_cat.detach()
        feat = self.relu(self.conv1x1[0](feat))
        feat = self.relu(self.conv3x3_1[0](feat))
        att = self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)

        fusion_1_2 = att_1 * x_l + att_2 * x_h
        return fusion_1_2
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class MASDF_Net(nn.Module):
    def __init__(self, num_classes = 1,drop_rate=0.2):
        super(MASDF_Net, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r'D:\TransFuse-main\pretrained\pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.rfb4 = RFB_modified(512, 32)
        self.rfb3 = RFB_modified(320, 32)
        self.rfb2 = RFB_modified(128, 32)

        self.ParDec = aggregation(32, 64)

        self.gc_block = ContextBlock(512, 2)
        self.drop = nn.Dropout2d(drop_rate)
        self.crossatt1 = SIG(320)
        self.crossatt2 = SIG(128)
        self.crossatt3 = SIG(64)
        self.up1 = nn.ConvTranspose2d(512, 320, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(320, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.double_conv1 = DoubleConv(640, 320, kernel_size=1, padding=0)
        self.double_conv2 = DoubleConv(256, 128, kernel_size=1, padding=0)
        self.double_conv3 = DoubleConv(128, 64, kernel_size=1, padding=0)
        self.sw = Scale_Aware(64)
        self.final_1 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=3, dilation=1, padding=1,relu=True, bn=True),
            BasicConv2d(64, num_classes, kernel_size=3, dilation=1,padding=1,relu=False, bn=False)
            )

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0] # 1,64,56,56
        x2 = pvt[1] # 1,128,28,28
        x3 = pvt[2] # 1,320,14,14
        x4 = pvt[3] # 1,512,7,7

        x4 = self.gc_block(x4) # 1,512,7,7
        c1 = self.up1(x4) # 1,320,14,14
        x3 = self.crossatt1(c1, x3)  # 1,320,14,14
        x3 = self.crossatt1(x3, x3)  # 1,320,14,14
        merge3 = torch.cat([c1, x3], dim=1) #1,640,14,14
        y3 = self.double_conv1(merge3)  # 1,320,14,14
        y3 = self.drop(y3)
        c2 = self.up2(y3) # 1,128,28,28
        x2 = self.crossatt2(c2, x2)  # 1, 128,28,28
        x2 = self.crossatt2(x2, x2)  # 1, 128,28,28
        merge2 = torch.cat([c2, x2], dim=1) # 1,256,28,28
        y2 = self.double_conv2(merge2)  # 1, 128,28,28
        y2 = self.drop(y2)
        c3 = self.up3(y2) # 1,64,56,56
        x1 = self.crossatt3(c3, x1) # 1,64,56,56
        x1 = self.crossatt3(x1, x1)# 1,64,56,56
        merge1 = torch.cat([c3, x1], dim=1) # 1,128,56,56
        y1 = self.double_conv3(merge1)  # 1,64,56,56
        y1 = self.drop(y1)
        x4 = self.rfb4(x4)
        y3 = self.rfb3(y3)
        y2 = self.rfb2(y2)
        dsv = self.ParDec(x4, y3, y2)

        dsv = F.interpolate(dsv, scale_factor=2, mode='bilinear')
        y = self.sw(dsv, y1)
        out1 = F.interpolate(self.final_1(y), scale_factor=4, mode='bilinear')
        return out1

if __name__ == '__main__':
    model = MASDF_Net().cuda()
    from thop import profile
    import torch

    input = torch.randn(1, 3, 224, 224).to('cuda')
    macs, params = profile(model, inputs=(input,))
    print('macs:', macs / 1000000000)
    print('params:', params / 1000000)

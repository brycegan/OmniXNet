import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpDS(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels // 2, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x):
        x = self.up(x)
        out = self.conv(x)

        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class CAFusion(nn.Module):
    def __init__(self, in_channels=[192, 384, 768, 1536], reduction=16):
        super(CAFusion, self).__init__()
        self.num_levels = len(in_channels)
        self.in_channels = in_channels

        self.ca_modules = nn.ModuleList([ChannelAttention(in_channels[i]*2, reduction) 
                                         for i in range(self.num_levels)])
        self.conv1x1 = nn.ModuleList([nn.Sequential(
                            nn.Conv2d(in_channels[i]*2, in_channels[i]*2, kernel_size=1, bias=False),
                            nn.BatchNorm2d(in_channels[i]*2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels[i]*2, in_channels[i], kernel_size=1, bias=False),
                            nn.BatchNorm2d(in_channels[i]),
                            nn.ReLU(inplace=True))
                            for i in range(self.num_levels)])


    def forward(self, feats_cnn, feats_trans):

        fused_feats = []

        for i in range(self.num_levels):
            x = torch.cat([feats_cnn[i], feats_trans[i]], dim=1)  # [B, 2C, H, W]

            x = self.ca_modules[i](x)

            x = self.conv1x1[i](x)

            fused_feats.append(x)

        return fused_feats

class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=192):
        super(FPN_fuse, self).__init__()
        # assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

        self.up1 = UpDS(fpn_out, fpn_out // 2, True)
        self.up2 = UpDS(fpn_out // 2, fpn_out // 4, True)

    def forward(self, features):

        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        x = self.conv_fusion(torch.cat((P), dim=1))
        x = self.up1(x)
        x = self.up2(x)

        return x

class OSFF(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256, drop_path=0.1):
        super(OSFF, self).__init__()

        self.multi_feat_fuse = CAFusion(in_channels=feature_channels)

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, fpn_out, kernel_size=1) for ch in feature_channels
        ])
        
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1) for _ in feature_channels
        ])

        self.restore_convs = nn.ModuleList([
            nn.Conv2d(fpn_out, ch, kernel_size=3, padding=1) for ch in feature_channels
        ])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, cnn_features, trans_features):

        fused_feats = []

        features = self.multi_feat_fuse(cnn_features, trans_features)
        laterals = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]

        for i in range(len(laterals)-1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='bilinear', align_corners=False)
            laterals[i-1] = laterals[i-1] + up
        
        outs = [smooth(l) for l, smooth in zip(laterals, self.smooth_convs)]

        outs = [restore(out) for out, restore in zip(outs, self.restore_convs)]

        for origin_feat, fpn_feat in zip(features, outs):
            fused = origin_feat + fpn_feat
            fused_feats.append(fused)

        return fused_feats




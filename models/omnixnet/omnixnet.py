import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from omnixnet.backbone.local_branch import ConvNeXt
from omnixnet.backbone.global_branch import SwinTransformerV2
from omnixnet.head import OmnixnetHead
from omnixnet.modules import BCM, LFE, GFE
from omnixnet.layers import OSFF

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.local_branch = ConvNeXt(in_chans=args.in_chans, dims=args.feature_channels, depths=args.local_branch_depths)
        self.global_branch = SwinTransformerV2(img_size=args.img_size, in_chans=args.in_chans, 
                                              num_classes=args.num_classes, patch_size=4, 
                                              window_size=4,embed_dim=args.feature_channels[0], depths=args.global_branch_depths)
        self.decoder_head = OmnixnetHead(num_classes=self.num_classes, in_channels=args.feature_channels)

        # local stage
        self.conv_downsample_layers = self.local_branch.downsample_layers
        self.conv_stages = self.local_branch.stages
        self.conv_norm_layers = self.local_branch.norm_layers
        
        # global stage
        self.swin_patch_embed = self.global_branch.patch_embed
        self.swin_pos_drop = self.global_branch.pos_drop
        self.swin_stages = self.global_branch.layers
        
        # LFE
        self.local_enhancer = nn.ModuleList([
            BCM(dim, freqfusion_type=LFE) for dim in args.feature_channels
        ])
        
        # GFE
        self.global_enhancer = nn.ModuleList([
            BCM(dim, freqfusion_type=GFE, levels=1) 
            for dim in args.feature_channels
        ])

        self.osff = OSFF(args.feature_channels, args.feature_channels[0])
    
    def forward(self, x):
        # Stage 0
        x_cnn = x
        x_swin= self.swin_pos_drop(self.swin_patch_embed(x))

        local_multi_features = []
        global_multi_features = []
        multi_features = []

        for i in range(4):
            # local stage
            x_cnn_feat = self.conv_downsample_layers[i](x_cnn)
            x_cnn_feat = self.conv_stages[i](x_cnn_feat)
            x_cnn_feat = self.conv_norm_layers[i](x_cnn_feat)

            # global Stage
            for blk in self.swin_stages[i].blocks:
                x_swin = blk(x_swin)
            x_swin_h, x_swin_w = int(np.sqrt(x_swin.shape[1])), int(np.sqrt(x_swin.shape[1]))
            x_swin_feat = rearrange(x_swin, 'b (h w) c -> b c h w', h=x_swin_h, w=x_swin_w)


            # Feature Fusion
            fused_trans_feat = self.local_enhancer[i](x_swin_feat, x_cnn_feat)
            fused_trans_feat = x_swin_feat + fused_trans_feat

            fused_cnn_feat = self.global_enhancer[i](x_cnn_feat, x_swin_feat)
            fused_cnn_feat = x_cnn_feat + fused_cnn_feat

            local_multi_features.append(fused_cnn_feat)
            global_multi_features.append(fused_trans_feat)

            # down
            fused_trans_feat = rearrange(fused_trans_feat, 'b c h w -> b (h w) c')
            if self.swin_stages[i].downsample is not None:
                fused_trans_feat, _ = self.swin_stages[i].downsample(fused_trans_feat)
            
            x_cnn = fused_cnn_feat
            x_swin = fused_trans_feat

        # osff fusion
        multi_features = self.osff(local_multi_features, global_multi_features)

        # head
        output = self.decoder_head(multi_features)

        return output
    

    

from functools import partial

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from inspect import isfunction
from einops import rearrange, repeat
from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import DropPath

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


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
            nn.GroupNorm(12, mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.GroupNorm(12, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        
        # print(x.shape)
        if len(x.shape) == 4:
            x_h, x_w = x.shape[2:]
            x = rearrange(x, 'b c h w -> b (h w) c')
        elif len(x.shape) == 3:
            x_h, x_w = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))


        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        if len(context.shape) == 4:
            context = rearrange(context, 'b c h w -> b (h w) c')
        
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # print(out.shape)

        out = self.to_out(out)
        # print(out.shape)
        out = rearrange(out, 'b (h w) n -> b n h w', h=x_h, w=x_w)
        
        return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dim = dim
        self.softmax = nn.Softmax(dim=-1)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _forward(self, q, kv):
        k,v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax(attn)
        out = (attn @ v)
        return out

    def forward(self, query, feat):
        feat = rearrange(feat, 'b (h w) c -> b c h w', w=int(math.sqrt(feat.shape[1])), h=int(math.sqrt(feat.shape[1])))
        query = rearrange(query, 'b (h w) c -> b c h w', w=int(math.sqrt(query.shape[1])), h=int(math.sqrt(query.shape[1])))
        self.h, self.w = feat.shape[2:]

        q = self.q_dwconv(self.q(query))
        kv = self.kv_dwconv(self.kv(feat))
        out = self._forward(q, kv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=kv.shape[-2], w=kv.shape[-1])
        out = self.project_out(out)
        return out
    

class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(FFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.hid_fea = hidden_features
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = rearrange(x, 'b (h w) c -> b c h w', w=int(math.sqrt(x.shape[1])), h=int(math.sqrt(x.shape[1])))
        self.h, self.w = x.shape[2:]
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    

class LFE(nn.Module):
    def __init__(self, in_features, levels=3):
        super().__init__()
        
        self.levels = levels

        self.norm1 = nn.LayerNorm(in_features)
        self.attn1 = Attention(in_features)
        self.norm2 = nn.LayerNorm(in_features)
        self.ffn1 = FFN(in_features)
        self.dim = in_features

        self.norm3 = nn.LayerNorm(in_features * 3)
        self.attn2 = Attention(in_features * 3)
        self.norm4 = nn.LayerNorm(in_features * 3)
        self.ffn2 = FFN(in_features * 3)


    def forward(self, query, fuse_feat):
        q_L, q_H = query
        q_H = rearrange(q_H[0], 'b c n h w -> b (c n) h w')
        fuse_L, fuse_H = fuse_feat
        fuse_H = rearrange(fuse_H[0], 'b c n h w -> b (c n) h w')


        assert len(q_H) == len(fuse_H)

        # process lowfreq
        q = rearrange(q_L, 'b c h w -> b (h w) c')
        f = rearrange(fuse_L, 'b c h w -> b (h w) c')

        q_L = q_L + self.attn1(self.norm1(q), f)
        x = rearrange(q_L, 'b c h w -> b (h w) c')
        q_L = q_L + self.ffn1(self.norm2(x))


        # process highfreq
        q = rearrange(q_H, 'b c h w -> b (h w) c')
        f = rearrange(fuse_H, 'b c h w -> b (h w) c')

        q_H = q_H + self.attn2(self.norm3(q), f)
        x = rearrange(q_H, 'b c h w -> b (h w) c')
        q_H = q_H + self.ffn2(self.norm4(x))

        q_H = rearrange(q_H, 'b (c n) h w -> b c n h w', n=3)

        return q_L, (q_H,)


class GFE(nn.Module):
    def __init__(self, in_features, out_features=None, levels=3, drop_path=0.1, ratio=0.25):
        super().__init__()

        in_features = in_features * levels
        hidden_features = int(in_features * ratio)
        out_features = hidden_features
        
        self.levels = levels

        self.q_reduce_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.f_reduce_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0, bias=False)

        self.l_conv = nn.Sequential(
            nn.Conv2d(in_features // levels, in_features // levels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, in_features // levels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // levels, in_features // levels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, in_features // levels),
            nn.ReLU(inplace=True),
        )

        self.q_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features, out_features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, out_features),
            nn.ReLU(inplace=True),
        )

        self.f_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features, out_features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, out_features),
            nn.ReLU(inplace=True),
        )

        self.conv_final_h = nn.Sequential(nn.Conv2d(2*hidden_features, in_features, 1))

        self.gamma = nn.Parameter(torch.ones(in_features // levels, 1, 1), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, query, fuse_feat):
        q_L, q_H = query
        fuse_L, fuse_H = fuse_feat

        assert len(q_H) == len(fuse_H)
        levels = len(q_H)

        # process lowfreq
        l_fused = self.l_conv(fuse_L)
        q_L = q_L + self.drop_path(self.gamma * l_fused)

        # process highfreq
        for i in range(levels):
            q_H[i] = rearrange(q_H[i], 'b c n h w -> b (c n) h w')
            fuse_H[i] = rearrange(fuse_H[i], 'b c n h w -> b (c n) h w')

            q_reduced = self.q_reduce_conv(q_H[i])
            fuse_reduced = self.f_reduce_conv(fuse_H[i])

            fused = torch.cat([self.q_conv(q_reduced), self.f_conv(fuse_reduced)], dim=1)
            fused = self.conv_final_h(fused)

            q_H[i] = q_H[i] + self.drop_path(fused)
            q_H[i] = rearrange(q_H[i], 'b (c n) h w -> b c n h w', n=levels)

        return q_L, q_H
    

class BCM(nn.Module):
    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., drop_path=0.1, levels=3, freqfusion_type=GFE):
        super().__init__()

        assert freqfusion_type in [GFE, LFE]

        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)

        self.attn = CrossAttention(dim, dim)

        self.freqfusion = freqfusion_type(dim, levels=levels)

        self.dwt = DWTForward(J=levels, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma = nn.Parameter(init_values * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, query, feat):
        assert query.shape == feat.shape

        q_freq = self.dwt(query)
        
        query_ = rearrange(query, 'b c h w -> b (h w) c')
        feat_ = rearrange(feat, 'b c h w -> b (h w) c')

        att_feat = self.attn(self.query_norm(query_), self.feat_norm(feat_))
        f_freq = self.dwt(att_feat)

        fused_L, fused_H = self.freqfusion(q_freq, f_freq)

        q_recon = self.idwt((fused_L, fused_H))

        q_recon = q_recon + self.drop_path(self.gamma * att_feat)

        return q_recon
    


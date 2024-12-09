import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model

import math


# 输入x的维度为 [B, N, C]
# 输出x的维度为 [B, N, C]
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        # 3×3的深度可分离卷积

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        # x：[B,C,H,W]
        x = self.dwconv(x)
        # 经过卷积核大小为3×3的深度可分离卷积
        x = x.flatten(2).transpose(1, 2)
        # x：[B, N, C]

        return x

# 输入x的维度为 [B, N, C]
# 输出x的维度为 [B, N, C]
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 输入x的维度为 [B, N, C]
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        # kv_bias=False, qk_scale=None
        # attn_drop=0., proj_drop=0.

        # num_heads=[1, 2, 4, 8]
        # sr_ratios=[8, 4, 2, 1]
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 每个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        # sr_ratios=[8, 4, 2, 1]
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            # sr_ratios = 8
            # self.sr = nn.Conv2d(dim, dim, kernel_size = 8, stride = 8)

            # sr_ratios = 4
            # self.sr = nn.Conv2d(dim, dim, kernel_size = 4, stride = 4)

            # sr_ratios = 2
            # self.sr = nn.Conv2d(dim, dim, kernel_size = 2, stride = 2)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

# 输入x的维度为 [B, N, C]
    def forward(self, x, H, W):
        # num_heads=[1, 2, 4, 8] 注意力头数量
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # self.q(x).reshape(B, N, self.num_heads, C // self.num_heads) ----->[B, N, head, C // head]
        # q----> [B, head, N, C // head]

        if self.sr_ratio > 1:
            # 如果sr_ratio为2、4、8。
            # 其实就是对x下采样2倍、4倍、8倍。
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # 将x的维度变为[B, C, H, W]
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # [B, C, H, W]--->[B, C, H/2, W/2]--->[B, C, HW // 4]--->[B, HW // 4, C]
            # x_：sr_ratio为2 ---> [B, HW // 4, C]
            # x_：sr_ratio为4 ---> [B, HW // 16, C]
            # x_：sr_ratio为8 ---> [B, HW // 64, C]

            x_ = self.norm(x_)
            # 经过层归一化
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # self.kv(x_) --->[B, HW // 4, 2C]
            # reshape --->[B, HW // 4, 2, head, C // head]
            # permute --->[2, B, head, HW // 4, C // head]

            # sr_ratio为2 ---> kv：[2, B, head, HW // 4, C // head]
            # sr_ratio为4 ---> kv：[2, B, head, HW // 16, C // head]
            # sr_ratio为8 ---> kv：[2, B, head, HW // 64, C // head]
        else:
            # sr_ratio为 1 的情况
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # self.kv(x_) --->[B, N, 2C]
            # reshape --->[B, N, 2, head, C // head]
            # permute --->[2, B, head, N, C // head]

            # sr_ratio为1---> kv：[2, B, head, N, C // head]
        k, v = kv[0], kv[1]
        # sr_ratio为 1 ：
        # k --->[B, head, N, C // head]
        # v --->[B, head, N, C // head]
        # sr_ratio为 2 ：
        # k --->[B, head, HW // 4, C // head]
        # v --->[B, head, HW // 4, C // head]
        # sr_ratio为 4 ：
        # k --->[B, head, HW // 16, C // head]
        # v --->[B, head, HW // 16, C // head]
        # sr_ratio为 8 ：
        # k --->[B, head, HW // 64, C // head]
        # v --->[B, head, HW // 64, C // head]

        # q----> [B, head, N, C // head]
        attn_ = (q @ k.transpose(-2, -1))
        # sr_ratio为 1 时：
        # q----> [B, head, N, C // head]
        # k, v --->[B, head, N, C // head]
        # attn_ --->[B, head, N, N]

        # sr_ratio为 2 时：
        # q----> [B, head, N, C // head]
        # k, v --->[B, head, HW // 4, C // head]
        # attn_ --->[B, head, N, HW // 4]

        # sr_ratio为 4 时：
        # attn_ --->[B, head, N, HW // 16]

        # sr_ratio为 8 时：
        # attn_ --->[B, head, N, HW // 64]
        attn = attn_ * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # (attn @ v) --->[B, head, N, C // head]
        # transpose --->[B, N, head, C // head]
        # reshape --->[B, N, C]
        # 对sr_ratio为 1、 2、 4、 8 时, x的维度均为 [B, N, C]。
        x = self.proj(x)
        x = self.proj_drop(x)

        attn_copy = attn_.clone().reshape(B, self.num_heads, H, W, attn.shape[-1],)
        # sr_ratio为1时，attn_ --->[B, head, N, N]
        # sr_ratio为2时，attn_ --->[B, head, N, HW // 4]
        # sr_ratio为4时，attn_ --->[B, head, N, HW // 16]
        # sr_ratio为8时，attn_ --->[B, head, N, HW // 64]
        # attn_.clone()创建了attn_的一个副本。
        # reshape ---> [B, head, H, W, N]或者[B, head, H, W, HW // 4]或者[B, head, H, W, HW // 16]或者[B, head, H, W, HW // 64]

        if self.sr_ratio > 1:
            attn_copy = F.avg_pool3d(attn_copy, kernel_size=(self.sr_ratio, self.sr_ratio, 1),
                                     stride=(self.sr_ratio, self.sr_ratio, 1))
            # 3d平均池化操作。
            # 如果sr_ratio 为2，经过3d平均池化维度变为[B, head, H // 2, W // 2, HW // 4]
            # 如果sr_ratio 为4，经过3d平均池化维度变为[B, head, H // 4, W // 4, HW // 16]
            # 如果sr_ratio 为8，经过3d平均池化维度变为[B, head, H // 8, W // 8, HW // 64]
        attn_copy = attn_copy.reshape(-1, self.num_heads, attn.shape[-1], attn.shape[-1])
        # sr_ratio为2时，attn_copy的维度为：[B, head, HW // 4, HW // 4]
        # sr_ratio为4时，attn_copy的维度为：[B, head, HW // 16, HW // 16]
        # sr_ratio为8时，attn_copy的维度为：[B, head, HW // 64, HW // 64]

        return x, attn_copy
        # x的维度为[B, N, C]

        # 当sr_ratio为1时，attn_copy的维度为：[B, head, N, N]
        # 当sr_ratio为2时，attn_copy的维度为：[B, head, HW // 4, HW // 4]
        # 当sr_ratio为4时，attn_copy的维度为：[B, head, HW // 16, HW // 16]
        # 当sr_ratio为8时，attn_copy的维度为：[B, head, HW // 64, HW // 64]

# 输入x的维度为[B, N, C]
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., sr_ratio=1,
                # Attention 中的注意力机制
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # dim = embed_dims = [64, 128, 256, 512]
        # num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1]
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            # Attention (self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 实例化DropPath()层
        # 指定的丢弃路径的概率drop_path
        # DropPath()层概率性地丢弃网络中的路径
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Mlp的 hidden_dim： dim ----> 4 * dim
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # 输入x的维度为[B, N, C]
        _x, _attn = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(_x)
        # 对_x进行丢弃操作。
        # 以一定的概率将输入的 _x 中的部分元素设置为 0,
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # 输出x的维度为[B, N, C]
        return x, _attn

# 输入[B, C, H, W]
# 输出[B, N, C]
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        # embed_dims=[64, 128, 256, 512]

        # self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
        #                                       embed_dim=embed_dims[0])
        # self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
        #                                       embed_dim=embed_dims[1])
        # self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
        #                                       embed_dim=embed_dims[2])
        # self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
        #                                       embed_dim=embed_dims[3])
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # to_2tuple函数用于确保img_size、patch_size是包含两个元素的元组。

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        # self.H, self.W = 224 // 7, 224 // 7
        # self.H, self.W = 32, 32
        self.num_patches = self.H * self.W
        # self.num_patches = 32 * 32 = 1024
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        # self.proj = nn.Conv2d(     3,     768,     kernel_size=7,          stride=4,     padding=(3, 3))

        # patch_embed1  img_size = img_size = 224
        # self.proj = nn.Conv2d(     3,     64,      kernel_size=7,          stride=4,     padding=(3, 3))

        # patch_embed2  img_size=img_size // 4 = 56
        # self.proj = nn.Conv2d(     64,    128,     kernel_size=3,          stride=2,     padding=(1, 1))

        # patch_embed3  img_size=img_size // 8 = 28
        # self.proj = nn.Conv2d(     128,   256,     kernel_size=3,          stride=2,     padding=(1, 1))

        # patch_embed4  img_size=img_size // 16 = 14
        # self.proj = nn.Conv2d(     256,   512,     kernel_size=3,          stride=2,     padding=(1, 1))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        # patch_embed1：[B, 64, 56, 56]
        # patch_embed2：[B, 128, 28, 28]
        # patch_embed3：[B, 256, 14, 14]
        # patch_embed4：[B, 512, 7, 7]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # patch_embed1：[B, 3136, 64]
        # patch_embed2：[B, 784, 128]
        # patch_embed3：[B, 196, 256]
        # patch_embed4：[B, 49, 512]
        x = self.norm(x)
        return x, H, W


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],

                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,  # Attention注意力机制中的参数
                 # drop_rate=0. 和 attn_drop_rate=0. 应该都是0  也就是注意力机制中的

                 drop_path_rate=0.,

                 norm_layer=nn.LayerNorm,

                 depths=[3, 4, 6, 3]):
        # @register_model
        # class pvt_v2_b2(PyramidVisionTransformerImpr):
        #     def __init__(self, **kwargs):
        #         super(pvt_v2_b2, self).__init__(
        #             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        #             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        #             drop_rate=0.0, drop_path_rate=0.1)

        # drop_path_rate=0.1
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        # img_size = 224   embed_dims[0] = 64
        # [B, 3, 224, 224] ------> [B, 64, 56, 56]
        # patch_embed1：[B, 3136, 64]

        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        # img_size // 4 = 56   embed_dims[0] = 64  embed_dims[1] = 128
        # [B, 64, 56, 56] ------> [B, 128, 28, 28]
        # patch_embed2：[B, 784, 128]

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        # img_size // 8 = 28   embed_dims[1] = 128  embed_dims[2] = 256
        # [B, 128, 28, 28] ------> [B, 256, 14, 14]
        # patch_embed3：[B, 196, 256]

        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        # img_size // 16 = 14  embed_dims[2] = 256  embed_dims[3] = 512
        # [B, 256, 14, 14] ------> [B, 512, 7, 7]
        # patch_embed4：[B, 49, 512]

        # transformer encoder
        # depths = [3, 4, 6, 3] sum(depths) = 16
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # torch.linspace(0, drop_path_rate, sum(depths)) 区间0 - drop_path_rate 等分成 sum(depths)个间隔
        # 如果 drop_path_rate 为 0.5，而 depths 列表为 [3, 4, 6, 3]，sum(depths) 就是 16（3 + 4 + 6 + 3）。在从 0 到 0.5 之间生成 16 个等间隔的数值。
        # 创建了一个列表dpr。 其中包含了从0 - drop_path_rate（包括 drop_path_rate）之间均匀间隔的数值，并且这个区间被分成了 sum(depths) 个部分。
        # 如果drop_path_rate为0，那么dpr列表中的所有元素都将是0.

        # drop_path_rate = 0.1
        # dpr列表包含16个元素，这些元素从0开始，均匀分布在 0 - 0.1之间。
        # [0.0, 0.006666666828095913, 0.013333333656191826, 0.019999999552965164, 0.02666666731238365,
        # 0.03333333507180214, 0.03999999910593033, 0.046666666865348816, 0.0533333346247673,
        # 0.06000000238418579, 0.06666667014360428, 0.07333333790302277,
        # 0.07999999821186066, 0.08666667342185974, 0.09333333373069763, 0.10000000149011612]
        cur = 0
        # depths=[3, 4, 6, 3]

        # Blcok() 输入维度为[B, N, C]
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], sr_ratio=sr_ratios[0],
                   # embed_dims=[64, 128, 256, 512] num_heads=[1, 2, 4, 8] sr_ratios=[8, 4, 2, 1]
                   # embed_dims[0] = 64
                   # num_heads[0] = 1
                   # mlp_ratios[0] = 4
                   # sr_ratios[0] = 8
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            # 注意力机制中的参数
            drop_path=dpr[cur + i], norm_layer=norm_layer)
            # depths[0] = 3
            # i为 0、1、2
            for i in range(depths[0])])
        # 第一个块 block1 会执行depths[0]次。

        self.norm1 = norm_layer(embed_dims[0])
        # 层归一化

        cur += depths[0]
        # cur = 3

        # Blcok() 输入维度为[B, N, C]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], sr_ratio=sr_ratios[1],
            # embed_dims=[64, 128, 256, 512] num_heads=[1, 2, 4, 8] sr_ratios=[8, 4, 2, 1]
            # embed_dims[1] = 128
            # num_heads[1] = 2
            # mlp_ratios[1] = 4
            # sr_ratios[1] = 4
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            # 注意力机制中的参数
            drop_path=dpr[cur + i], norm_layer=norm_layer)
            # depths[1] = 4
            # i 为 0、1、2、3
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        # depths=[3, 4, 6, 3]
        cur += depths[1]
        # cur = 7
        # Blcok() 输入维度为[B, N, C]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], sr_ratio=sr_ratios[2],
            # embed_dims=[64, 128, 256, 512] num_heads=[1, 2, 4, 8] sr_ratios=[8, 4, 2, 1]
            # embed_dims[2] = 256
            # num_heads[2] = 4
            # sr_ratios[2] = 2
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            # 注意力机制中的参数
            drop_path=dpr[cur + i], norm_layer=norm_layer)
            # depths[2] = 6
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        # depths=[3, 4, 6, 3]
        cur += depths[2]
        # cur = 7+6 = 13
        # Blcok() 输入维度为[B, N, C]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], sr_ratio=sr_ratios[3],
            # embed_dims=[64, 128, 256, 512] num_heads=[1, 2, 4, 8] sr_ratios=[8, 4, 2, 1]
            # embed_dims[3] = 512
            # num_heads[3] = 8
            # mlp_ratios[3] = 4
            # sr_ratios[3] = 1
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            # 注意力机制中的参数
            drop_path=dpr[cur + i], norm_layer=norm_layer)
            # depths[3] = 3
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = 1
            #load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        # drop_path_rate=0.1
        # depths=[3, 4, 6, 3]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # 生成一个从 0 - drop_path_rate的数列。一个等差数列。
        # 数列的长度是 sum(self.depths)
        cur = 0
        # depths=[3, 4, 6, 3]
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def _get_pos_embed(self, pos_embed, patch_embed, H, W):
    #     if H * W == self.patch_embed1.num_patches:
    #         return pos_embed
    #     else:
    #         return F.interpolate(
    #             pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
    #             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        attns = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        # [B, 3, 224, 224] ------> [B, 64, 56, 56]
        # x [B, 3136, 64]
        # H W 为 56.
        # 经过 self.patch_embed1(x)  x为[B, N, C]的形式。 N = H*W = 56×56 = 3136。 C = 64
        for i, blk in enumerate(self.block1):
            #         self.block1 = nn.ModuleList([Block(
            #             dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], sr_ratio=sr_ratios[0],
            #          embed_dims=[64, 128, 256, 512] num_heads=[1, 2, 4, 8] sr_ratios=[8, 4, 2, 1]
            # self.block1是一个 nn.ModuleList，包含了多个 Block 块的堆叠。
            # self.block1是一个堆叠的模块列表，这个列表中的每个元素都是一个 Block 模块。
            # blk 是 self.block1 中的一个 Block 模块。
            # 通过此循环，可依次访问并操作 self.block1 中的每个 Block 模块。
            x, attn = blk(x, H, W)
            # 将输入 x 与两个额外的参数H和W传递给block
            # 输入 x 经过Attention操作和MLP操作。
            # x 的维度保持不变。
            # sr_ratios为8、num_head为1。------>attn的维度[B, head, HW // 64, HW // 64]
            attns.append(attn)
        x = self.norm1(x)
        # x经过层归一化。
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 将x转换为[b, c, h, w]的形状。 [B, 64, 56, 56]
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        # [B, 64, 56, 56] ----self.patch_embed2(x)---->[B, 128, 28, 28]
        # x [B, 784, 128]
        # H = 28、W = 28
        # 输出的 x 的维度是[b, n, c]的形式。 H = 28、W = 28.
        for i, blk in enumerate(self.block2):
            # self.block2 = nn.ModuleList([Block(
            # dim=embed_dims[1] = 128, num_heads=num_heads[1] = 2, mlp_ratio=mlp_ratios[1], sr_ratio=sr_ratios[1] = 4)
            #        embed_dims=[64, 128, 256, 512] num_heads=[1, 2, 4, 8] sr_ratios=[8, 4, 2, 1]
            x, attn = blk(x, H, W)
            # 将输入 x 与两个额外的参数 H 和 W 传递给block。
            # 将 x 经过Attention操作和MLP操作。
            # num_heads = 2、sr_ratio = 4。------>attn的维度[B, head, HW // 16, HW // 16]
            # x 的维度保持不变。
            attns.append(attn)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 将x有[b, n, c]的形状转换为[b, c, h, w]的形状。
        # x [B, 128, 28, 28]
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        # [B, 128, 28, 28]----self.patch_embed3(x)--->[B, 256, 14, 14]
        # x [B, 196, 256]
        # H = 14、W = 14
        # 将x由[b, c, h, w]的形状转换为[b, n, c]的形状。
        for i, blk in enumerate(self.block3):
            # self.block3 = nn.ModuleList([Block(
            # dim=embed_dims[2] = 256, num_heads=num_heads[2] = 4, mlp_ratio=mlp_ratios[2], sr_ratio=sr_ratios[2] = 2,
            # embed_dims=[64, 128, 256, 512] num_heads=[1, 2, 4, 8] sr_ratios=[8, 4, 2, 1]
            x, attn = blk(x, H, W)
            # 将输入 x 与两个额外的参数 H 和 W 传递给block。
            # 将 x 经过Attention操作和MLP操作。
            # num_heads = 2、sr_ratio = 2。------>attn的维度[B, head, HW // 4, HW // 4]
            attns.append(attn)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 将x由[b, n, c]的形状转换为[b, c, h, w]的形状。
        # x [B, 256, 14, 14]
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        # [B, 256, 14, 14]----self.patch_embed4(x)---> [B, 512, 7, 7]
        # x [B, 49, 512]
        # H = 7、W = 7
        # 将x由[b, c, h, w]的形状转换为[b, n, c]的形状。
        for i, blk in enumerate(self.block4):
            # self.block4 = nn.ModuleList([Block(
            #      dim=embed_dims[3] = 512, num_heads=num_heads[3] = 8, mlp_ratio=mlp_ratios[3], sr_ratio=sr_ratios[3] = 1,
            # embed_dims=[64, 128, 256, 512] num_heads=[1, 2, 4, 8] sr_ratios=[8, 4, 2, 1]
            x, attn = blk(x, H, W)
            # 将输入 x 与两个额外的参数 H 和 W 传递给block。
            # 将 x 经过Attention操作和MLP操作。
            # num_heads = 8、sr_ratio = 1。------>attn的维度[B, head, N, N]
            attns.append(attn)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 将 x 由[b, n, c]的形状转换为[b, c , h, w]的形状。
        # x [B, 512, 7, 7]
        outs.append(x)

        return outs, attns

        # return x.mean(dim=1)

    def forward(self, x):
        x, attns = self.forward_features(x)
        # x = self.head(x)

        return x, attns

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

# 定义了一系列名为 pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5 的模型类.
@register_model
class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



@register_model
class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

@register_model
class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

@register_model
class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

@register_model
class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@register_model
class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


if __name__ =='__main__':
    model = pvt_v2_b2().cuda()
    model.eval()
    inputs = torch.randn(2, 3, 352, 352).cuda()
    output = model(inputs)
    print(model)


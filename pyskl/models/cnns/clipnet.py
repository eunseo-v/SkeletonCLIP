# YuanLin added
import warnings
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from mmcv.cnn import ConvModule, build_activation_layer, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint, load_state_dict
from mmcv.utils import _BatchNorm
from torch.nn.modules.utils import _ntuple, _triple

from ...utils import cache_checkpoint, get_root_logger
from ..builder import BACKBONES, build_backbone

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

@BACKBONES.register_module()
class BkCLIP(nn.Module):
    """ skeleton CLIP backbone
    
    Args: 
        video_encoder (dict): 调用现有的ResNet3dSlowOnly
    """ 

    def __init__(
        self, 
        video_encoder,
        text_encoder
    ):
        super().__init__()
        self.video_encoder = build_backbone(video_encoder)
        self.text_encoder = build_backbone(text_encoder)

    def init_weights(self):
        self.video_encoder.init_weights()
        self.text_encoder.init_weights()

    def forward(self, imgs, texts):
        # imgs [N, C, T, H, W]
        # texts [N, n_ctx]
        img_features = self.video_encoder(imgs) # [N, C', T', H', W']
        text_features = self.text_encoder(texts) # [N, C']
        return img_features, text_features


@BACKBONES.register_module()
class TextCLIP(nn.Module):
    """ text CLIP part of skeleton CLIP backbone
        在init中完成text_prompt工作
    """
    def __init__(
        self,
        context_length: int,    # 77
        vocab_size: int,        # 49408
        transformer_width: int, # 512
        transformer_heads: int, # 8
        transformer_layers: int,# 12
        embed_dim: int,          # 也设成512，与posec3d的输出维度对应
        frozen_stages = -1,      # >=0就参数全部冻结
        pretrained = None     # 是否有预训练模型
    ):
        super().__init__()
        self.context_length = context_length
        
        # text encoder
        self.transformer = Transformer(
            width = transformer_width,
            layers = transformer_layers,
            heads = transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(
            vocab_size, transformer_width
        )
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim)
        )

        # 这一部分移到clip_head中
        # 0.07是初始化可学习的temperature parameter
        # self.logit_scale = nn.Parameter(
        #     torch.ones([]) * np.log(1/ 0.07)
        # )
        self.frozen_stages = frozen_stages
        self.pretrained = pretrained

        # 这一部分移到了recognizerclip中，这样的话textclip的输入已经是tokenize后的文本信息
        # self.classes 存储每一种text_prompt形式的tokenize编号 [(num_text_aug* num_classes), n_ctx]
        # self.num_text_aug 有多少种text_prompt形式
        # self.text_dict 与self.classes类似
        # classes[num_classes*i+j] = text_dict[i][j] i = 0,...,num_text_aug-1, j=0,...,num_classes-1
        # self.classes, self.num_text_aug, self.text_dict = text_prompt(self.class_list)
        
    def init_weights(self, pretrained = None):
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will override the original 'pretrained' if set.
                The arg is added to be compatible with mmdet. Default: None.
        """
        # 先按照正常的初始化策略
        self._initialize_parameters()
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load TextCLIP model from: {self.pretrained}')
            ckpt = cache_checkpoint(self.pretrained)
            # 只加载已有的参数权重
            state_dict = _load_checkpoint(ckpt).state_dict()
            # 将这些不是可学习的参数剔除
            for key in ["input_resolution", "context_length", "vocab_size"]:
                if key in state_dict:
                    del state_dict[key]
            load_state_dict(self, state_dict, strict=False, logger = logger)

    def _initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    
    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        'self.frozen_stages'."""
        if self.frozen_stages >=0:
            self.transformer.eval()
            for param in self.transformer.parameters(): # len=144
                param.requires_grad = False
            for param in self.token_embedding.parameters(): # len=1
                param.requires_grad = False
            for param in self.ln_final.parameters():    # len=2
                param.requires_grad = False
            self.positional_embedding.requires_grad = False
            # self.logit_scale.requires_grad = False            
    
    @property
    def dtype(self):
        return self.token_embedding.weight.dtype
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask    
        
    def forward(self, text):
        # 输入text [N, n_ctx = 77]
        x = self.token_embedding(text).type(self.dtype) # [N n_ctx=77 C=512]
        x = x + self.positional_embedding.type(self.dtype) # [N n_ctx=77 C=512]
        x = x.permute(1, 0, 2) # [L, N, C]
        x = self.transformer(x) # [L, N, C]
        x = x.permute(1, 0, 2) # [N, L, C]
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection # [B, C=512]
        return x

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()


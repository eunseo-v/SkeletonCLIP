from mmcv.cnn import normal_init
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from ..builder import HEADS, build_loss
from .base import BaseHead

@HEADS.register_module()
class CLIPHead(nn.Module, metaclass = ABCMeta):
    def __init__(
        self,   
        in_channels,        # encoder后的特征通道数
        img_loss_cls = dict(type='KLLoss'),
        text_loss_cls = dict(type = 'KLLoss'),
        dropout = 0.5,
        logit_scale_pretrained = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.imgs_loss_cls = build_loss(img_loss_cls)
        self.text_loss_cls = build_loss(text_loss_cls)
        self.dropout_ratio = dropout
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.logit_scale_pretrained = logit_scale_pretrained
        # 0.07是初始化可学习的temperature parameter
        if logit_scale_pretrained is False:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1/ 0.07)
            )
        else:
            self.logit_scale = nn.Parameter(
                torch.tensor(4.6052, dtype=torch.float32)
            )

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # 0.07是初始化可学习的temperature parameter
        if self.logit_scale_pretrained is False:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1/ 0.07)
            )
        else:
            self.logit_scale = nn.Parameter(
                torch.tensor(4.6052, dtype=torch.float32)
            )

    def forward(self, imgs, texts):
        """Defines the computation performed at every call.
        """
        pool = nn.AdaptiveAvgPool3d(1)
        if isinstance(imgs, tuple) or isinstance(imgs, list):
            imgs = torch.cat(imgs, dim=1)
        imgs = pool(imgs)
        imgs = imgs.view(imgs.shape[:2]) # [N, C]
        # normalized features L2正则化，各通道的平方和为1
        imgs = imgs/imgs.norm(dim = -1, keepdim=True)
        texts = texts/texts.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * imgs @ texts.t()    # [每一个图片N, 每一个文本N]
        logits_per_text = logit_scale * texts @ imgs.t()     # [每一个文本N, 每一个图片N]
        return logits_per_image, logits_per_text

    def gen_label(self, labels):
        # 若为正样本对，取值为1，否则取值为0
        num = len(labels)
        gt = np.zeros(shape=(num,num))
        for i, label in enumerate(labels):
            for k in range(num):
                if labels[k] == label:
                    gt[i,k] = 1
        return gt

    def top_k_accuracy(self, simi, gt, topk = (1,)):
        """ 计算topk_1或者topk_5准确率
            计算topk_1或者topk_5其实有多个正样本的位置，
            正样本点中任意一个得分最高也算这个样本分类成功
            因为具体的类别信息在矩阵相乘的过程中被忽略了
            思路就是similarity从高到低排序，
            topk1的indice索引到ground_truth矩阵中，如果值为1的话，就认为topk1分类成功
            topk5的indice索引到ground_truth矩阵中，有任意一个值为1，就认为topk5分类成功
        """
        res = []
        for k in topk:
            num = 0
            max_k_preds = np.argsort(simi, axis=1)[:, -k:][:, ::-1]
            if k == 1:
                for i, idx in enumerate(max_k_preds):
                    if gt[i][idx] == 1:
                        num +=1
                top1_acc_score = num/gt.shape[0]
                res.append(top1_acc_score)
            if k == 5:
                for i, idxs in enumerate(max_k_preds):
                    if sum(gt[i][idxs]) >=1:
                        num+=1
                top5_acc_score = num/gt.shape[0]
                res.append(top5_acc_score)
        return res

    def loss(self, logits_per_image, logits_per_text, labels):
        # logits_per_image  [每一个图片N, 每一个文本N]
        # logits_per_text   [每一个文本N, 每一个图片N]
        # labels            [N]
        # 需要完成计算这一个batch的top1_acc, top5_acc以及loss
        losses = dict()
        ground_truth = torch.tensor(
            self.gen_label(labels), 
            dtype = logits_per_image.dtype,
            device=logits_per_image.device
        )   # 为1的点表示该图像文本对为正样本，否则为负样本
        # 计算topk_1或者topk_5其实有多个正样本的位置，
        # 正样本点中任意一个得分最高也算这个样本分类成功
        # 因为具体的类别信息在矩阵相乘的过程中被忽略了
        # 思路就是similarity从高到低排序，
        # topk1的indice索引到ground_truth矩阵中，如果值为1的话，就认为topk1分类成功
        # topk5的indice索引到ground_truth矩阵中，有任意一个值为1，就认为topk5分类成功
        similarity = (100* logits_per_image @ logits_per_text.t()).softmax(dim=-1) # [N, N]
        top_k_acc = self.top_k_accuracy(
            simi = similarity.detach().cpu().numpy(),
            gt = ground_truth.detach().cpu().numpy(),
            topk=(1, 5)
        )
        losses['top1_acc'] = torch.tensor(
            top_k_acc[0], device = logits_per_image.device
        )
        losses['top5_acc'] = torch.tensor(
            top_k_acc[1], device = logits_per_image.device
        )

        loss_imgs = self.imgs_loss_cls(logits_per_image, ground_truth)
        loss_texts = self.text_loss_cls(logits_per_text, ground_truth)
        total_loss = (loss_imgs + loss_texts)/2
        losses['loss_cls'] = total_loss
        return losses
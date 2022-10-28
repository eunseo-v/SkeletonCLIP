# YuanLin added
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer
import pandas as pd
import numpy as np
from pyskl.utils.text_prompt import text_prompt

@RECOGNIZERS.register_module()
class RecognizerCLIP(BaseRecognizer):
    def __init__(self, 
        backbone, cls_head, class_list, 
        train_cfg = dict(), test_cfg = dict()
    ):
        super().__init__(backbone=backbone, cls_head=cls_head,train_cfg=train_cfg, test_cfg=test_cfg)
        self.class_list = class_list
        # self.classes 存储每一种text_prompt形式的tokenize编号 [(num_text_aug* num_classes), n_ctx]
        # self.num_text_aug 有多少种text_prompt形式
        # self.text_dict 与self.classes类似
        # classes[num_classes*i+j] = text_dict[i][j] i = 0,...,num_text_aug-1, j=0,...,num_classes-1
        class_all = pd.read_csv(self.class_list)
        self.classes, self.num_text_aug, self.text_dict = text_prompt(class_all.values.tolist())

    def extract_feat(self, imgs, texts):
        return self.backbone(imgs, texts)

    def forward_train(self, imgs, label, **kwargs):
        # imgs [N, 1, C, T, H, W]
        # label [N, 1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        label = label.squeeze()
        text_id = np.random.randint(
            self.num_text_aug, size = len(label)
        )   # batch_size长，取值为0,...,self.num_text_aug-1
        # text_id 选择哪一种aug模式, label是对应的类别
        texts = torch.stack([self.text_dict[j][i,:] for i,j in zip(label, text_id)]).to(imgs.device) # [N, n_ctx]
        losses = dict()
        # imgs [B, C, T, H, W] texts [B, C]
        imgs, texts = self.extract_feat(imgs, texts)
        # self.cls_head 完成计算logits_per_image, logits_per_text
        logits_per_image, logits_per_text = self.cls_head(imgs, texts)
        # self.cls_head.loss()需要完成计算这一个batch的top1_acc, top5_acc以及loss
        gt_label = label.squeeze()
        loss_cls = self.cls_head.loss(logits_per_image, logits_per_text, gt_label)
        losses.update(loss_cls)
        return losses

    def forward_test(self, imgs, **kwargs):
        # imgs [N, num_segs, C, T, H, W]
        batch_size = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) # [(N num_segs), C, T, H, W]
        # 选择将self.classes作为texts进行文本特征提取
        texts = self.classes.to(imgs.device) # [(num_text_aug num_classes), n_ctx]
        # self.extract_feat输出 
        #   imgs [(N num_segs), C', T', H', W']
        #   texts[(num_text_aug num_classes), C']
        imgs, texts = self.extract_feat(imgs, texts)
        # self.cls_head 完成计算logits_per_image. logits_per_text忽略
        # logits_per_image [(N num_segs), (num_text_aug num_classes)]
        logits_per_image, _ = self.cls_head(imgs, texts)
        # logits_per_image [(N num_segs), num_text_aug, num_classes]
        logits_per_image = logits_per_image.reshape(
            logits_per_image.shape[0], self.num_text_aug, -1
        ).softmax(dim = -1)
        # logits_per_image [(N num_segs), num_classes]
        logits_per_image = logits_per_image.mean(dim=1, keepdim=False)
        # logits_per_image [N, num_segs, num_classes]
        logits_per_image = logits_per_image.reshape(
            batch_size, num_segs, logits_per_image.shape[-1]
        )
        # logits_per_image [N, num_classes]
        logits_per_image = logits_per_image.mean(dim=1)
        return logits_per_image.detach().cpu().numpy()
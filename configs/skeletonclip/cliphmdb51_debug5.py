model = dict(
    type='RecognizerCLIP',
    backbone = dict(
        type='BkCLIP',
        video_encoder = dict(
            type='ResNet3dSlowOnly',
            in_channels=17, # 输入数据的通道数
            base_channels=32,   # 输入网络的特征通道数
            num_stages=3,   # 有多少个stage (论文中表格的一行，一系列resblock作为一个stage)
            out_indices=(2, ),  # 将哪些res_layer的输出作为前向输出，层数从0开始计数
            stage_blocks=(3, 4, 6), # 每一个stage/res_layer包含的resblock数 (论文中的一个方括号)
            conv1_stride=(1, 1),    # 应用于self.conv1的stride
            pool1_stride=(1, 1),    # 应用于self.maxpool的stride
            inflate=(0, 1, 1),  # 每一个res_layer是否有inflate，就是卷积核为(3,1,1)
            spatial_strides=(2, 2, 2),  # 每一个res_layer对应的spatial_stride
            temporal_strides=(1, 1, 2),  # 每一个res_layer对应的temporal_stride
            # frozen_stages=3,  只要frozen_stages≥0， self.conv1就被冻结, 值是多少，就冻结多少个res_layer层  
        ),      
        text_encoder = dict(
            type='TextCLIP',
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
            embed_dim=512,
            # frozen_stages = -1, # >=0就冻结除了text_projection的所有参数
            pretrained='/home/yl/ActionCLIP/.cache/clip/ViT-B-16.pt'
        ),
    ),
    cls_head = dict(
        type = 'CLIPHead',
        in_channels= 512,
        img_loss_cls = dict(type='KLLoss'),
        text_loss_cls = dict(type='KLLoss'),
        dropout = 0.0,
        logit_scale_pretrained=True # 是否加载预训练的temperature parameter
    ),
    class_list = '/home/yl/pyskl/lists/hmdb51_labels.csv',
)
dataset_type = 'PoseDataset'
ann_file = '/home/yl/public_datasets/heatmap/hmdb51_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(48, 48), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train1', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='test1', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='test1', pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='AdamW', lr=0.1, weight_decay=0.0001,
    paramwise_cfg = dict(
        custom_keys = {
            'backbone.text_encoder': dict(lr_mult=0.01, decay_mult=0.9),
            'cls_head': dict(lr_mult=0.1, decay_mult=0.9)
        }
    )
)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy = 'CosineAnnealing',
    by_epoch = False,
    min_lr = 0,
    warmup = 'linear',
    warmup_iters = 2,
    warmup_ratio = 0.1,
    warmup_by_epoch = True
)
total_epochs = 20
checkpoint_config = dict(interval=20)    # interval表示每多少个epoch，save checkpoints
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/actionclip/mydebug5'
# load_from = 'https://download.openmmlab.com/mmaction/skeleton/posec3d/k400_posec3d-041f49c6.pth'  # noqa: E501
find_unused_parameters = False
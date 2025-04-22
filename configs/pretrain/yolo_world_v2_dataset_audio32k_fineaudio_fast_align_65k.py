_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# hyper-parameters
num_classes = 1203
num_training_classes = 80
max_epochs = 40  # Maximum training epochs
close_mosaic_epochs = 20
save_epoch_intervals = 1
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 1e-3
weight_decay = 0.025
train_batch_size_per_gpu = 16
val_batch_size_per_gpu   = 1
# load_from = 'work_dirs/yolo_world_v2_dataset_audio/pretrained.pth'
# load_from = 'pretrained/audio_ckp.pth'
load_from = 'pretrained/audio_ckp_align_mix.pth'

# img_scale = (1280, 1280)
img_scale = _base_.img_scale
model_fp = '/home/yangwenhao/project/SpeechCLIP/exp/mix/no_image_text_relate2_aug/epoch=11-step=46979-val_recall_mean_10=85.7600.ckpt'

# model settings
model = dict(
    type='AudioYOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalAudioYOLOBackbone',
        image_model={{_base_.model.backbone}},
        audio_model=model_fp,
        audio_blank='demo/audio_blank.npy',
        frozen_stages=4,
        freeze_audio=1,
        with_audio_model=True),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              # freeze_all=True,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    # freeze_all=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'audio',
                    'flip_direction', 'texts'))
]
# text_transform = [
#     *_base_.test_pipeline[:-1],
#     dict(type='LoadText'),
#     dict(type='mmdet.PackDetInputs',
#          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'audio',
#                     'flip_direction', 'texts'))
# ]


audio_transform = [
    dict(type='LoadAudio',
        max_duration=2,),
]

train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=img_scale,
         # center_ratio_range=(0.25, 0.75),
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *audio_transform,
    *text_transform,
]


train_pipeline_stage2 = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *audio_transform,
    *text_transform
]

flickr_train_dataset = dict(
    type='YOLOv5MixedGroundingAudioDataset',
    data_root='data/flickr_audio_tmp/',
    ann_file='annotations/final_flickr_separateGT_train_32k_scores_valid.json',
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline)

# coco_train_dataset = dict(
#     type='YOLOv5MixedGroundingAudioDataset',
#     data_root='data/coco2014/',
#     # ann_file='annotations/final_flickr_separateGT_train_8k2.json',
#     ann_file='annotations/final_mixed_train_8k_norm_scores.json',
#     # ann_file='annotations/final_flickr_separateGT_train_32k3.json',
#     data_prefix=dict(img='images/'),
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=train_pipeline)

coco_train_dataset = dict(
    type='YOLOv5MixedGroundingAudioDataset',
    data_root='data/coco2014_caps/',
    # ann_file='annotations/final_flickr_separateGT_train_8k2.json',
    ann_file='annotations/final_mixed_train_65k_caps.json',
    # ann_file='annotations/final_flickr_separateGT_train_32k3.json',
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline)


gqa_train_dataset = dict(
    type='YOLOv5MixedGroundingAudioDataset',
    # data_root='data/GQA/',
    data_root='data/GQA_caps/',
    # ann_file='annotations/final_mixed_train_8k.json',
    ann_file='annotations/final_mixed_train_65k_caps.json',
    # ann_file='annotations/final_flickr_separateGT_train_1.json',
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(_delete_=True,
                                     type='ConcatDataset',
                                     datasets=[
                                         flickr_train_dataset,
                                         coco_train_dataset,
                                         gqa_train_dataset
                                     ],
                                     ignore_keys=['classes', 'palette']))
test_audio_transform = [
    dict(type='LoadAudio',
        # audio_path='data/audios/lvis_v1_class_audio2.json'),
        audio_path='data/audios/coco_class_audio.json',),
    # dict(type='LoadRandomAudio',
    #      # audio_path='data/audios/lvis_v1_class_audio2.json',
    #      audio_path='data/audios/coco_class_audio.json',
    #      num_neg_samples=(num_classes, num_classes),
    #      max_num_samples=num_training_classes,
    #      padding_to_max=True,),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    *test_audio_transform,
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'audio', 'audio_length',
                    'scale_factor', 'pad_param', 'texts'))
]

# coco_val_dataset = dict(
#     _delete_=True,
#     type='MultiModalDataset',
#     dataset=dict(type='YOLOv5LVISV1Dataset',
#                  data_root='data/coco/',
#                  test_mode=True,
#                  ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
#                  data_prefix=dict(img=''),
#                  batch_shapes_cfg=None),
#     class_text_path='data/texts/lvis_v1_class_texts.json',
#     pipeline=test_pipeline)

# flickr_val_dataset = dict(
#     _delete_=True,
#     type='MultiModalDataset',
#     dataset=dict(type='YOLOv5MixedGroundingAudioDataset',
#                  data_root='data/flickr_audio_tmp/',
#                  test_mode=True,
#                  ann_file='annotations/final_flickr_separateGT_val.json',
#                  data_prefix=dict(img='images/'),
#                  batch_shapes_cfg=None),
#     # class_text_path='data/texts/lvis_v1_class_texts.json',
#     pipeline=test_pipeline)

# flickr_val_dataset = dict(
#     type='YOLOv5MixedGroundingAudioDataset',
#     data_root='data/flickr_audio_tmp/',
#     ann_file='annotations/final_flickr_separateGT_val.json',
#     data_prefix=dict(img='images/'),
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=test_pipeline)

# val_dataloader = dict(dataset=coco_val_dataset,
#                      batch_size=val_batch_size_per_gpu,)#,collate_fn=dict(type='predict_collate')) #collate_fn=dict(type='yolow_collate'),
# collate_fn=dict(type='predict_collate'),
# val_dataloader = dict(dataset=flickr_val_dataset) #collate_fn=dict(type='yolow_collate'),

# test_dataloader = val_dataloader

# val_evaluator = dict(type='mmdet.LVISMetric',
#                      ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
#                      metric='bbox')
# val_evaluator = dict(type='mmdet.LVISMetric',
#                      ann_file='data/flickr_audio_tmp/annotations/final_flickr_separateGT_val.json',
#                      metric='bbox')

# test_pipeline = [
#     *_base_.test_pipeline[:-1],
#     dict(type='LoadText'),
#     *test_audio_transform,
#     dict(type='mmdet.PackDetInputs',
#          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                     'audio',
#                     'scale_factor', 'pad_param', 'texts'))
# ]

# coco_val_dataset = dict(
#     _delete_=True,
#     type='MultiModalDataset',
#     dataset=dict(type='YOLOv5LVISV1Dataset',
#                  data_root='data/coco/',
#                  test_mode=True,
#                  ann_file='lvis/lvis_v1_val_sub512.json',
#                  data_prefix=dict(img=''),
#                  batch_shapes_cfg=None),
#     class_text_path='data/texts/lvis_v1_class_texts.json',
#     pipeline=test_pipeline)

coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='data/coco2014/',
                 test_mode=True,
                 ann_file='annotations/instances_val2014_neg_2k.json',
                 data_prefix=dict(img='val2014'),
                 batch_shapes_cfg=None),
    class_text_path='data/texts/coco_class_texts.json',
    pipeline=test_pipeline)

# coco_val_dataset = dict(
#     _delete_=True,
#     type='MultiModalDataset',
#     dataset=dict(type='YOLOv5LVISV1Dataset',
#                  data_root='data/coco/',
#                  test_mode=True,
#                  ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
#                  data_prefix=dict(img=''),
#                  batch_shapes_cfg=None),
#     class_text_path='data/texts/lvis_v1_class_texts.json',
#     pipeline=test_pipeline)


val_dataloader = dict(dataset=coco_val_dataset,
                     batch_size=val_batch_size_per_gpu,)
test_dataloader = val_dataloader

# val_evaluator = dict(type='mmdet.LVISMetric',
#                      ann_file='data/coco/lvis/lvis_v1_val_sub512.json',
#                      metric='bbox')

val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='data/coco2014/annotations/instances_val2014_neg_2k.json',
                     metric='bbox')

test_evaluator = val_evaluator

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater'))
custom_hooks = [
    # dict(type='EMAHook',
    #      ema_type='ExpMomentumEMA',
    #      momentum=0.0001,
    #      update_buffers=True,
    #      strict_load=False,
    #      priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=1,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])

optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(bias_decay_mult=0.0,
                        norm_decay_mult=0.0,
                        custom_keys={
                            # 'backbone':
                            # dict(lr_mult=0.0),
                            'backbone.image_model':
                            dict(lr_mult=0.0),
                            'backbone.audio_model':
                            dict(lr_mult=0.01, weight_decay=1e-5),
                            'bbox_head':
                            dict(lr_mult=0.1),
                            'neck':
                            dict(lr_mult=0.1),
                            'logit_scale':
                            dict(weight_decay=0.0)
                        }),
    accumulative_counts=2,
    constructor='YOLOWv5OptimizerConstructor')

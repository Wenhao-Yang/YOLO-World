# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
import pickle
import os
import time

@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks,
                                     batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              txt_masks,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        text_output = self.backbone.forward_text(texts)
        if isinstance(text_output, tuple):
            self.text_feats = text_output[0]
        else:
            self.text_feats = text_output

        # self.text_feats, _ = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, (txt_feats,
                        txt_masks) = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats, txt_masks
@MODELS.register_module()
class AudioYOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 frozen_list = [],
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.frozen_list = frozen_list
        
        super().__init__(*args, **kwargs)
        
        self.frozen()
        
    def frozen(self):
        model_list = set(['backbone.audio_model',
                      'backbone.image_model',
                      'neck', 'head'])
        
        for i in self.frozen_list:
            if i in model_list:
                m = self
                for s in i.split('.'):
                    m = getattr(m, s)

                # m = getattr(self, i)
                for param in m.parameters():
                    param.requires_grad = False

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, audio_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        # print(audio_feats.shape, [i.shape for i in img_feats])
        losses = self.bbox_head.loss(img_feats, audio_feats, batch_data_samples)
        for k in losses:
            if torch.isnan(losses[k]):
                pickle_name = 'tmp/input.nan.{}.pickle'.format(time.strftime('%Y%m%d_%H%M%S',time.localtime()))
                # if not os.path.exists('input.nan.pickle'):
                with open(pickle_name, 'wb') as f:
                    pickle.dump({'batch_inputs': batch_inputs,
                                'batch_data_samples': batch_data_samples,
                                'img_feats': img_feats,
                                'audio_feats': audio_feats}, f)
                break
        
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        # print(batch_inputs)
#         with open('tmp.batch_data_samples.pickle', 'wb') as f:
#             pickle.dump(batch_data_samples[0], f)
#         print(type(batch_data_samples[0].audio))
        img_feats, audio_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = audio_feats[0].shape[0]
        # print(audio_feats.shape, [i.shape for i in img_feats])
        results_list = self.bbox_head.predict(img_feats,
                                              audio_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        
        return batch_data_samples

    def reparameterize(self, audio: List[List[Tensor]]) -> None:
        # encode text embeddings into the detector
        self.audio = audio
        self.audio_feats = self.backbone.forward_text(audio)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, audio_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, audio_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        audio_feats = None
        if batch_data_samples is None:
            audio = self.audio
            audio_feats = self.audio_feats
        elif isinstance(batch_data_samples,
                        dict) and 'audio' in batch_data_samples:
            audio = batch_data_samples['audio']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'audio'):
            audio = [data_sample.audio for data_sample in batch_data_samples]
        elif hasattr(self, 'audio_feats'):
            audio = self.audio
            audio_feats = self.audio_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
            
        if isinstance(audio[0], torch.Tensor): 
            # if padding into tensor [batch, classes, audio_dim] 
            #         ==> audio list [batch*classes, audio_dim] 

            if 'audio_length' in batch_data_samples[0]:
                audios = []
                for au,le in zip(audio, batch_data_samples):
                    for a, l in zip(au,le.audio_length):
                        audios.append(a[:l])
                
                audio = audios 
              
        if audio_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, audio_feats = self.backbone(batch_inputs, audio)
        
        
        if len(audio_feats.shape) == 2:
            
            audio_feats = audio_feats.reshape(len(img_feats[0]), -1, audio_feats.shape[-1])
        
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, audio_feats)
            else:
                img_feats = self.neck(img_feats)
                
        return img_feats, audio_feats


@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


@MODELS.register_module()
class AudioSimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, audio_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, audio_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, audio_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  audio_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, audio_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, audio_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            audio_feats = self.embeddings[None]
            if self.adapter is not None:
                audio_feats = self.adapter(audio_feats) + audio_feats
                audio_feats = nn.functional.normalize(audio_feats, dim=-1, p=2)
            audio_feats = audio_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            audio_feats = None
            
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, audio_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, audio_feats
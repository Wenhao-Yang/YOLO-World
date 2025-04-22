# Copyright (c) Tencent Inc. All rights reserved.
import itertools
from typing import List, Sequence, Tuple
import torch
import os
import numpy as np
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType
from transformers import (AutoTokenizer, AutoModel, CLIPTextConfig)
from transformers import CLIPTextModelWithProjection as CLIPTP

#  Speech encoder mdoels
import avssl.model
import avssl
import math
import pickle
# from yolo_world.models.base import OrderedNamespace
# from .speech import FairseqSpeechEncoder_Hubert

@MODELS.register_module()
class HuggingVisionBackbone(BaseModule):
    def __init__(self,
                 model_name: str,
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 norm_eval: bool = True,
                 frozen_modules: Sequence[str] = (),
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.norm_eval = norm_eval
        self.frozen_modules = frozen_modules
        self.model = AutoModel.from_pretrained(model_name)

        self._freeze_modules()

    def forward(self, image: Tensor) -> Tuple[Tensor]:
        encoded_dict = self.image_model(pixel_values=image,
                                        output_hidden_states=True)
        hidden_states = encoded_dict.hidden_states
        img_feats = encoded_dict.get('reshaped_hidden_states', hidden_states)
        img_feats = [img_feats[i] for i in self.image_out_indices]
        return tuple(img_feats)

    def _freeze_modules(self):
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class HuggingCLIPLanguageBackbone(BaseModule):
    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 add_mask: bool = False,
                 training_use_cache: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.add_mask = add_mask
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name,
                                                     attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()

    def forward_tokenizer(self, texts):
        if not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors='pt', padding=True)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        if self.add_mask:
            text_mask = torch.tensor([x != self.pad_value for x in text],
                                     requires_grad=False).to(self.model.device)
        text = self.tokenizer(text=text, return_tensors='pt', padding=True)
        text = text.to(device=self.model.device)

        if len(self.frozen_modules) > 0:
            with torch.no_grad():
                txt_outputs = self.model(**text)
                txt_feats = txt_outputs.text_embeds
        else:
            txt_outputs = self.model(**text)
            txt_feats = txt_outputs.text_embeds

        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0],
                                      txt_feats.shape[-1])
        if self.add_mask:
            text_mask = text_mask.reshape(-1, num_per_batch[0]).to(txt_feats)
        else:
            text_mask = None
        return txt_feats, text_mask
    
    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()

@MODELS.register_module()
class HuggingSpeechCLIPLanguageBackbone(BaseModule):

    def __init__(self,
                 model_name: str,
                #  config: OrderedNamespace,
                 frozen_modules: Sequence[str] = (),
                #  dropout: float = 0.0,
                 training_use_cache: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # clip_config = CLIPTextConfig.from_pretrained(model_name,
        #                                              attention_dropout=dropout)
        # self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        # self.model = FairseqSpeechEncoder_Hubert(**config.audio_encoder)
        self.model = avssl.model.KWClip_GeneralTransformer.load_from_checkpoint(model_name)

        self._freeze_modules()
        

    def forward(self, wav_data: List[List[Tensor]]) -> Tensor:
        audio_batches = []
        num_wavs = []
        for t in audio_batch:
            num_wavs.append(len(t))
            audio_batches.extend(t)
        audio_batches = [t.squeeze() for t in audio_batches] 
        
        output = self.model.encode_speech(wav=audio_batches)
        
        return output

    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()


@MODELS.register_module()
class PseudoLanguageBackbone(BaseModule):
    """Pseudo Language Backbone
    Args:
        text_embed_path (str): path to the text embedding file
    """
    def __init__(self,
                 text_embed_path: str = "",
                 test_embed_path: str = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        # {text:embed}
        self.text_embed = torch.load(text_embed_path, map_location='cpu')
        if test_embed_path is None:
            self.test_embed = self.text_embed
        else:
            self.test_embed = torch.load(test_embed_path)
        self.register_buffer("buff", torch.zeros([
            1,
        ]))

    def forward_cache(self, text: List[List[str]]) -> Tensor:
        if not hasattr(self, "cache"):
            self.cache = self.forward_text(text)
        return self.cache

    def forward(self, text: List[List[str]]) -> Tensor:
        if self.training:
            return self.forward_text(text)
        else:
            return self.forward_cache(text)

    def forward_text(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        if self.training:
            text_embed_dict = self.text_embed
        else:
            text_embed_dict = self.test_embed
        text_embeds = torch.stack(
            [text_embed_dict[x.split("/")[0]] for x in text])
        # requires no grad and force to float
        text_embeds = text_embeds.to(
            self.buff.device).requires_grad_(False).float()
        text_embeds = text_embeds.reshape(-1, num_per_batch[0],
                                          text_embeds.shape[-1])
        return text_embeds


@MODELS.register_module()
class MultiModalYOLOBackbone(BaseModule):
    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        self.with_text_model = with_text_model
        self.image_model = MODELS.build(image_model)
        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()

    def forward(self, image: Tensor,
                text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.image_model(image)
        if text is not None and self.with_text_model:
            txt_feats = self.text_model(text)
            return img_feats, txt_feats
        else:
            return img_feats, None

    def forward_text(self, text: List[List[str]]) -> Tensor:
        assert self.with_text_model, "forward_text() requires a text model"
        txt_feats = self.text_model(text)
        return txt_feats

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)


### audio Multi Modal
@MODELS.register_module()
class MultiModalAudioYOLOBackbone(BaseModule):

    def __init__(self,
                 image_model: ConfigType,
                 audio_model: str = '',
                 audio_blank: str = '',
                 frozen_stages: int = -1,
                 freeze_audio: int = 2,
                 with_audio_model: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        self.with_audio_model = with_audio_model
        self.image_model = MODELS.build(image_model)
        if self.with_audio_model:
            # self.audio_model = MODELS.build(audio_model)
            self.audio_model = avssl.model.KWClip_GeneralTransformer.load_from_checkpoint(audio_model)
            self.audio_model.clip = None
            self.audio_model.criterion = None
            self.audio_model.audio_augment = None
            if freeze_audio:
                self.audio_model.freeze(freeze_audio)
                
            if os.path.exists(audio_blank):
                self.audio_blank_embeddings = torch.nn.Parameter(
                        torch.from_numpy(np.load(audio_blank)).float())
                self.audio_blank_embeddings.requires_grad = False
        else:
            self.audio_model = None
            
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()

    def forward(self, image: Tensor,
                audio: List[List[Tensor]]) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.image_model(image)
        
        if self.with_audio_model or audio is None:
            audio_feats = self.forward_audio(audio)
            
            return img_feats, audio_feats
        else:
            return img_feats, None

    def forward_audio(self, audio: List[List[Tensor]]) -> Tensor:

        if audio is None:
            return None
        
        assert self.with_audio_model, "forward_audio() requires a audio model"
        # print(audio)
        
        if isinstance(audio[0], list) and self.training:
            audios, num_audio = [], []
            for t in audio:
                num_audio.append(len(t))
                audios.extend(t)
            
            # negative zeros
            audios.append(torch.zeros(max(num_audio)))

            audio_feats = self.audio_model.encode_speech_alone(wav=audios)
            audio_feats = audio_feats['parallel_audio_feat']

            s = 0
            total = []
            for t in num_audio:
                # total.append(audio_feats[s:s+t])
                # tt = torch.cat([audio_feats[s:s+t], self.audio_blank_embeddings.unsqueeze(0).repeat(80-t, 1)])
                tt = torch.cat([audio_feats[s:s+t], audio_feats[-1].unsqueeze(0).repeat(80-t, 1)])
                total.append(tt)
                # total.append(self.audio_blank_embeddings.unsqueeze(0).repeat(80-t, 1))
                s += t
            audio_feats = torch.stack(total, dim=0)
        
        elif isinstance(audio[0], list) and len(audio[0]) == len(audio[1]) and len(audio[0]) == len(audio[2]):
            
#             wavs = audio[0]
#             audio_feats = []
#             for i in range(int(math.ceil(len(wavs)/64))):
#                 audio_feat = self.audio_model.encode_speech_alone(wav=wavs[i*64:(i*64+64)])
#                 audio_feats.append(audio_feat['parallel_audio_feat'])
#             audio_feats = torch.cat(audio_feats, dim=0)
            # audio_feats = self.audio_model.encode_speech_alone(wav=audios[0])
            # audio_feats = audio_feats['parallel_audio_feat']  
        
            audio_feats = []
            for wavs in audio:
                audio_feat = self.audio_model.encode_speech_alone(wav=wavs)
                audio_feats.append(audio_feat['parallel_audio_feat'])
                
            audio_feats = torch.stack(audio_feats, dim=0)
            
        elif isinstance(audio[0], torch.Tensor):
            # print('audio_clip: ', audio[0].shape)
            if len(audio[0].shape) == 2:
                wavs = []
                for i in audio:
                    wavs.extend([a for a in i])
                    
            elif len(audio[0].shape) == 1:
                wavs = [i for i in audio]
            
            # print('audio_clip wav : ', len(wavs))
            # if not os.path.exists('tmp/class_coco_clip.pickle'):
            #     with open('tmp/class_coco_clip.pickle', 'wb') as f:
            #         pickle.dump(wavs, f)
                # np.save('tmp/class_coco_clip.npy', audio_feats.cpu().squeeze().numpy())
            
            audio_feats = []
            for i in range(int(math.ceil(len(wavs)/64))):
                audio_feat = self.audio_model.encode_speech_alone(wav=wavs[i*64:(i*64+64)])
                audio_feats.append(audio_feat['parallel_audio_feat'])
                
            audio_feats = torch.cat(audio_feats, dim=0)
            # print('audio_clip audio_feats: ', audio_feats.shape)
        
        audio_feats = audio_feats / audio_feats.norm(p=2, dim=-1, keepdim=True)
        
        # if not os.path.exists('tmp/class_coco_clip.npy'):
        #     np.save('tmp/class_coco_clip.npy', audio_feats.cpu().squeeze().numpy())
            
        # print(audio_feats.shape)
        return audio_feats

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)

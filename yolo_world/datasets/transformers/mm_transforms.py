# Copyright (c) Tencent Inc. All rights reserved.
import json
import random
from typing import Tuple

import numpy as np
from mmyolo.registry import TRANSFORMS
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence

@TRANSFORMS.register_module()
class RandomLoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '') -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value  = padding_value
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        num_classes = len(class_texts)
        if 'gt_labels' in results:
            gt_label_tag = 'gt_labels'
        elif 'gt_bboxes_labels' in results:
            gt_label_tag = 'gt_bboxes_labels'
        else:
            raise ValueError('No valid labels found in results.')
        positive_labels = set(results[gt_label_tag])

        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))

        num_neg_samples = min(
            min(num_classes, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels:
                candidate_neg_labels.append(idx)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['gt_bboxes']), dtype=bool)
        for idx, label in enumerate(results[gt_label_tag]):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results[gt_label_tag][idx] = label2ids[label]
        results['gt_bboxes'] = results['gt_bboxes'][gt_valid_mask]
        results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]

        if 'gt_ignore_flags' in results:
            results['gt_ignore_flags'] = results['gt_ignore_flags'][gt_valid_mask]

        if 'instances' in results:
            retaged_instances = []
            for idx, inst in enumerate(results['instances']):
                label = inst['bbox_label']
                if label in label2ids:
                    inst['bbox_label'] = label2ids[label]
                    retaged_instances.append(inst)
            results['instances'] = retaged_instances

        texts = []
        for label in sampled_labels:
            cls_caps = class_texts[label]
            assert len(cls_caps) > 0
            cap_id = random.randrange(len(cls_caps))
            sel_cls_cap = self.prompt_format.format(cls_caps[cap_id])
            texts.append(sel_cls_cap)

        if 'audio' in results:
            audios = []
            for label in sampled_labels:
                cls_caps = results['audio'][label]
                audios.append(cls_caps)
                
            results['audio'] = audios

        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        results['texts'] = texts

        return results


@TRANSFORMS.register_module()
class LoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 multi_prompt_flag: str = '/') -> None:
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        texts = []
        for idx, cls_caps in enumerate(class_texts):
            assert len(cls_caps) > 0
            sel_cls_cap = cls_caps[0]
            sel_cls_cap = self.prompt_format.format(sel_cls_cap)
            texts.append(sel_cls_cap)

        results['texts'] = texts

        return results
    

@TRANSFORMS.register_module()
class LoadAudio:

    def __init__(self,
                 audio_path: str = None,
                 prompt_format: str = '{}',
                 max_duration: int = 4,
                 multi_prompt_flag: str = '/') -> None:
        
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        self.max_duration = max_duration
        
        if audio_path is not None:
            with open(audio_path, 'r') as f:
                self.class_audios = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'audio' in results or hasattr(self, 'class_audios'), (
            'No audio found in results. ', results.keys())
        
        class_audios = results.get(
            'audio',
            getattr(self, 'class_audios', None))

        audios = []
        sr = 16000
        
        for idx, audio_caps in enumerate(class_audios):
            this_wav = []
            assert len(audio_caps) > 0
            # print(audio_caps)
            # if isinstance(audio_caps[0], list):
            #     cap_id = random.randrange(len(audio_caps))
            #     audio_f, s, e = audio_caps[cap_id]
            # else:
            
            if isinstance(audio_caps[0], list):
                for cls_caps in audio_caps:
                    # print()
                    if isinstance(cls_caps[0], list):
                        cap_id = random.randrange(len(cls_caps))

                        audio_f, s, e = cls_caps[cap_id]
                    else:
                        audio_f, s, e = cls_caps

                    # audio_f, s, e = ac
                    wav, sr = torchaudio.load(audio_f, frame_offset=int(s*sr), num_frames=int(e*sr))
                    this_wav.append(wav[0])
            else:
                audio_f, s, e = audio_caps
                # audio_f, s, e = ac
                wav, sr = torchaudio.load(audio_f, frame_offset=int(s*sr), num_frames=int(e*sr))
                this_wav.append(wav[0])
                
            wav = torch.cat(this_wav)
            if self.max_duration*sr < len(wav):
                s = np.random.randint(0, len(wav)-int(self.max_duration*sr))
                e = s + int(self.max_duration*sr)
                wav = wav[s:e]
                
            if len(wav) == 0:
                wav = torch.zeros(16000)
                
            # if len(wav[0]) > 0:
            audios.append(wav)
            
        ### padding to 10 % min
        audio_length = torch.LongTensor([len(i) for i in audios])
        min_length = int(np.percentile(audio_length, 10))
        
        for i, a in enumerate(audios):
            if audio_length[i] < min_length:
                audios[i] = torch.nn.functional.pad(a, (0, min_length-len(a)), "constant", 0)
        
        if hasattr(self, 'class_audios'):
            # audios = torch.stack(audios)
            # audio_length = torch.LongTensor([len(i) for i in audios])
            audios = pad_sequence(audios, batch_first=True)
        
        results['audio_length'] = audio_length
        results['audio'] = audios
        
        # print(results.keys())

        return results
    

@TRANSFORMS.register_module()
class LoadRandomAudio:

    def __init__(self,
                 audio_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = True,
                 multi_prompt_flag: str = '/') -> None:
        
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max  = padding_to_max
        
        if audio_path is not None:
            with open(audio_path, 'r') as f:
                self.class_audios = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'audio' in results or hasattr(self, 'class_audios'), (
            'No audio found in results. ', results.keys())
        
        class_audios = results.get(
            'audio',
            getattr(self, 'class_audios', None))

        sr = 16000
        num_classes = len(class_audios)
        if 'gt_labels' in results:
            gt_label_tag = 'gt_labels'
        elif 'gt_bboxes_labels' in results:
            gt_label_tag = 'gt_bboxes_labels'
        else:
            raise ValueError('No valid labels found in results.')
            
        positive_labels = set(results[gt_label_tag])

        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))

        num_neg_samples = min(
            min(num_classes, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        
        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels:
                candidate_neg_labels.append(idx)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['gt_bboxes']), dtype=bool)
        for idx, label in enumerate(results[gt_label_tag]):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results[gt_label_tag][idx] = label2ids[label]
                
        results['gt_bboxes'] = results['gt_bboxes'][gt_valid_mask]
        results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]

        if 'instances' in results:
            retaged_instances = []
            for idx, inst in enumerate(results['instances']):
                label = inst['bbox_label']
                if label in label2ids:
                    inst['bbox_label'] = label2ids[label]
                    retaged_instances.append(inst)
            results['instances'] = retaged_instances
        
        # audio_length
        audios = []
        for label in sampled_labels:
            cls_caps = class_audios[label]
            assert len(cls_caps) > 0
            if isinstance(cls_caps[0], list):
                cap_id = random.randrange(len(cls_caps))

                audio_f, s, e = cls_caps[cap_id]
            else:
                audio_f, s, e = cls_caps
                
            wav, sr = torchaudio.load(audio_f, frame_offset=int(s*sr), num_frames=int(e*sr))
            audios.append(wav[0])
            
        if self.padding_to_max:
            # if hasattr(self, 'class_audios'):
            audio_length = torch.LongTensor([len(i) for i in audios])
            audios = pad_sequence(audios, batch_first=True)
            results['audio_length'] = audio_length

        results['audio'] = audios

        return results


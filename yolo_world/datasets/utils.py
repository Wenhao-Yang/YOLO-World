# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmengine.dataset import COLLATE_FUNCTIONS
from torch.nn.utils.rnn import pad_sequence

@COLLATE_FUNCTIONS.register_module()
def yolow_collate(data_batch: Sequence,
                  use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs = []
    batch_bboxes_labels = []
    batch_masks = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']
        batch_imgs.append(inputs)

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks.to(
                dtype=torch.bool, device=gt_bboxes.device)
            batch_masks.append(masks)
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)

    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0)
        }
    }
    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        collated_results['inputs'] = torch.stack(batch_imgs, 0)

    if hasattr(data_batch[0]['data_samples'], 'texts'):
        batch_texts = [meta['data_samples'].texts for meta in data_batch]
        collated_results['data_samples']['texts'] = batch_texts

    if hasattr(data_batch[0]['data_samples'], 'audio'):
        batch_audio = [meta['data_samples'].audio for meta in data_batch]
        collated_results['data_samples']['audio'] = batch_audio

    if hasattr(data_batch[0]['data_samples'], 'is_detection'):
        # detection flag
        batch_detection = [meta['data_samples'].is_detection
                           for meta in data_batch]
        collated_results['data_samples']['is_detection'] = torch.tensor(
            batch_detection)

    return collated_results


@COLLATE_FUNCTIONS.register_module()
def predict_collate(data_batch):
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    # print(data_batch[0])
    if hasattr(data_batch[0]['data_samples'], 'audio'):
        for meta in data_batch:
            audio = meta['data_samples'].audio
            audio_length = torch.LongTensor([len(i) for i in audio])
            # meta['data_samples'].audio = pad_sequence(audio, batch_first=True)
            meta['data_samples'].set_field(pad_sequence(audio, batch_first=True), 'audio', field_type='metainfo')
            meta['data_samples'].audio_length = audio_length
    
    # print(data_batch)
    # print(data_batch[0])
    return data_batch


@COLLATE_FUNCTIONS.register_module()
def mm_default_collate(data_batch: Sequence):
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each data_itement in ``data_batch``.

    The default behavior of dataloader is to merge a list of samples to form
    a mini-batch of Tensor(s). However, in MMEngine, ``pseudo_collate``
    will not stack tensors to batch tensors, and convert int, float, ndarray to
    tensors.

    This code is referenced from:
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.

    Args:
        data_batch (Sequence): Batch of data from dataloader.

    Returns:
        Any: Transversed Data in the same format as the data_itement of
        ``data_batch``.
    """  # noqa: E501
    data_item = data_batch[0]
    data_item_type = type(data_item)
    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named tuple
        return data_item_type(*(pseudo_collate(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [pseudo_collate(samples)
                    for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type(
                    [pseudo_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [pseudo_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({
            key: pseudo_collate([d[key] for d in data_batch])
            for key in data_item
        })
    else:
        return data_batch
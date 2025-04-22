import logging
import types

from fairseq.models.hubert.hubert import HubertConfig, HubertModel
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder

logger = logging.getLogger(__name__)

from typing import Dict, List, Optional, Tuple, Union

import fairseq
import numpy as np
import torch
import torch.nn.functional as F
# from fairseq.utils import index_put
from s3prl import hub
from s3prl.utility.download import _urls_to_filepaths
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.file_utils import copy_func

# from ..data import random_crop_max_length
# from ..util import freeze_model, init_weights
# from .weighted_sum import WeightedSumLayer

FEAT_SELECT_IDX_WEIGHTED_SUM_MODE = "weighted_sum"

def random_crop_max_length(
    audio: torch.Tensor, max_len: int, orig_len: int = 1000000000
) -> torch.Tensor:
    """Randomly crop an audio feature sequence into max_len.

    Args:
        audio (torch.Tensor): Audio features (T, *)
        max_len (int): Maximum length.
        orig_len (int, optional): Original length of audio. Defaults to 1000000000.

    Returns:
        torch.Tensor: Cropped audio features.
    """
    audio_len = min(len(audio), orig_len)
    if audio_len <= max_len or max_len < 0:
        return audio[:audio_len]

    offset = np.random.randint(audio_len - max_len)
    return audio[offset : offset + max_len]


def freeze_model(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False

def init_weights(m: nn.Module) -> None:
    """Initialize module's weights

    Args:
        m (nn.Module): Module.
    """
    if isinstance(m, nn.Module) and hasattr(m, "reset_parameters"):
        m.reset_parameters()



class WeightedSumLayer(nn.Module):
    def __init__(self, n_weights: int, normalize_features: bool = False):
        """Weighted sum layer with learnable weights.

        Args:
            n_weights (int): Number of weights, i.e., number of hidden representations.
        """

        super().__init__()

        self.n_weights = n_weights
        self.weights = nn.Parameter(torch.zeros((n_weights,), dtype=torch.float))
        self.normalize_features = normalize_features
        if self.normalize_features:
            logger.info("Normalize feature before weighted sum")

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Weighted sum a list of tensors.

        Args:
            x (List[torch.Tensor]): Representations to be weighted summed.

        Returns:
            torch.Tensor: Weighted summed representations.
        """

        assert len(x) == self.n_weights, len(x)

        weights = torch.softmax(self.weights, dim=0)
        weights = weights.view(-1, *([1] * x[0].dim()))
        x = torch.stack(x, dim=0)
        if self.normalize_features:
            x = F.layer_norm(x, (x.shape[-1],))
        x = (weights * x).sum(0)

        return x


def customFunc_hubert_forward(
    self,
    source: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    mask: bool = True,
    output_layer: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """output layer is 1-based"""
    features = self.forward_features(source)

    features = features.transpose(1, 2)
    features = self.layer_norm(features)
    unmasked_features = features.clone()

    if padding_mask is not None:
        padding_mask = self.forward_padding_mask(features, padding_mask)

    if self.post_extract_proj is not None:
        features = self.post_extract_proj(features)

    features = self.dropout_input(features)
    unmasked_features = self.dropout_features(unmasked_features)

    if mask:
        x, mask_indices = self.apply_mask(features, padding_mask, None)
    else:
        x = features
        mask_indices = None

    # feature: (B, T, D), float
    # target: (B, T), long
    # x: (B, T, D), float
    # padding_mask: (B, T), bool
    # mask_indices: (B, T), bool
    x, layer_results = self.encoder(
        x,
        padding_mask=padding_mask,
        layer=None if output_layer is None else output_layer - 1,
    )

    return {"x": x, "layer_results": layer_results}


class FairseqSpeechEncoder_Hubert(nn.Module):
    """FairseqSpeechEncoder_Hubert

    For Extracting HuBERT hiddenstates
    HuBERT load from fairseq

    """

    MODEL2URL = {
        "hubert": "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
        "hubert_base": "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
        "hubert_large_ll60k": "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt",
    }

    MODEL_DOWNSAMPLE_RATE = {
        "hubert": 320,
        "hubert_base": 320,
        "hubert_large_ll60k": 320,
    }

    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        trainable: bool = False,
        device: str = "cpu",
        feat_select_idx: Union[str, list] = "all",
        layer_drop: Union[str, float] = 0.0,
        max_audio_len: int = -1,
        reinit_layers: List[int] = [],
        unfreeze_layers: List[int] = [],
        normalize_hiddenstates: bool = False,
        normalize_type: str = "s3prl",
        **kwargs,
    ):
        """Speech Encoder with S3PRL (v0.3.1)
        Args:
            name (str): Name of speech encoder.
            pretrained (bool, optional): Init with pretrained model. Defaults to False.
            trainable (bool, optional): Whether to update the model while training. Defaults to False.
            device (str, optional): Device. Defaults to "cpu".
            feat_select_idx (Union[str, list], optional): Feature selection indices. Defaults to "all".
            layerdrop (Union[str, float], optional): Layer drop rate. Defaults to 0.0.
        """
        super().__init__()

        assert name in self.MODEL2URL, "Model name({}) should be in {}".format(
            name, self.MODEL2URL.keys()
        )
        self.name = name
        self.pretrained = pretrained
        self.trainable = trainable
        # self.device = device
        self.feat_select_idx = feat_select_idx
        self.max_audio_len = max_audio_len
        self.reinit_layers = reinit_layers
        self.unfreeze_layers = unfreeze_layers
        self.normalize_hiddenstates = normalize_hiddenstates
        assert normalize_type in ["s3prl", "method1", "method2"], normalize_type
        if self.normalize_hiddenstates:
            logger.info("Normalize hidden states ({})".format(normalize_type))
        self.normalize_type = normalize_type

        ckpt = _urls_to_filepaths(self.MODEL2URL[self.name], refresh=False)

        # modify Hubert Functions for extracting hidden states
        self.apply_customHubertForward()

        model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.encoder = model[0]
        self.encoder_task = task
        logger.info(f"Normalize waveform = ({self.encoder_task.cfg.normalize:})")

        if hasattr(self.encoder, "get_downsample_rates"):
            self.downsample_rate = self.encoder.get_downsample_rates("hidden_states")
        else:
            self.downsample_rate = self.MODEL_DOWNSAMPLE_RATE[self.name]

        if not pretrained:
            self.encoder.apply(init_weights)

        if not trainable:
            freeze_model(self.encoder)
            self.encoder.eval()

        if self.name.startswith("hubert"):
            if (
                isinstance(layer_drop, float)
                and layer_drop >= 0.0
                and layer_drop <= 1.0
            ):
                self.encoder.encoder.layerdrop = layer_drop
            elif layer_drop == "original":
                pass
            else:
                raise ValueError(f"layer_drop = {layer_drop} is not supported.")

            assert not (len(reinit_layers) > 0 and len(unfreeze_layers) > 0)
            if len(reinit_layers) > 0:
                logger.warning(f"Reinitializing encoder layers {reinit_layers}")
                assert self.trainable
                for i, layer in enumerate(self.encoder.encoder.layers):
                    if i in reinit_layers:
                        layer.apply(init_weights)
                    else:
                        freeze_model(layer)

                freeze_model(self.encoder.encoder.pos_conv)
                freeze_model(self.encoder.layer_norm)
                freeze_model(self.encoder.feature_extractor)
                freeze_model(self.encoder.post_extract_proj)
                self.encoder.feature_grad_mult = 0

            if len(unfreeze_layers) > 0:
                logger.warning(f"Freezing encoder layers excluding {unfreeze_layers}")
                assert self.trainable
                for i, layer in enumerate(self.encoder.encoder.layers):
                    if i in unfreeze_layers:
                        pass
                        # layer.apply(init_weights)
                    else:
                        freeze_model(layer)

                freeze_model(self.encoder.encoder.pos_conv)
                freeze_model(self.encoder.layer_norm)
                freeze_model(self.encoder.feature_extractor)
                freeze_model(self.encoder.post_extract_proj)
                self.encoder.feature_grad_mult = 0

        self.out_dim = 0
        with torch.no_grad():
            wav = [torch.randn(16000, dtype=torch.float, device="cpu")]
            padded_wav, wav_padding_mask = self.preprocess_input(wavs=wav)
            output = self.encoder.customHubertForward(
                source=padded_wav,
                padding_mask=wav_padding_mask,
                mask=None,
            )
            # feat = self.encoder(wav)
            # self.out_dim = feat["last_hidden_state"].shape[2]
            self.upstream_model_hiddenstates_len = len(output["layer_results"])
            self.out_dim = output["x"].shape[2]

        logger.info(
            f"Loaded s3prl speech encoder ({name}): out_dim = {self.out_dim} layer_drop = {self.encoder.encoder.layerdrop}"
        )

        if self.feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            logger.info(
                f"Using weighted sum for all hiddenstates({self.upstream_model_hiddenstates_len})"
            )
            assert self.upstream_model_hiddenstates_len > 0

            self.weightedsum_layer = WeightedSumLayer(
                n_weights=self.upstream_model_hiddenstates_len,
                normalize_features=self.normalize_hiddenstates
                and self.normalize_type == "s3prl",
            )

    def trainable_params(self) -> list:
        if self.trainable and len(self.reinit_layers) == 0:
            return list(self.parameters())
        if self.trainable and len(self.reinit_layers) > 0:
            params = []
            for i in self.reinit_layers:
                params += list(self.encoder.encoder.layers[i].parameters())
            if not self.encoder.encoder.layer_norm_first:
                params += list(self.encoder.encoder.layer_norm.parameters())
            return params
        else:
            if self.feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
                logger.info("Adding weightedsum params")
                params = list(self.weightedsum_layer.parameters())
                return params
            else:
                return []

    def apply_customHubertForward(self):
        # add method
        # self.encoder.encoder.extract_features = copy_func(custom_FairseqTransformerEncoder_extract_features)
        TransformerEncoder.extract_features = copy_func(
            custom_FairseqTransformerEncoder_extract_features
        )
        # add method
        # self.encoder.customHubertForward = copy_func(customFunc_hubert_forward)
        HubertModel.customHubertForward = copy_func(customFunc_hubert_forward)

    def preprocess_input(self, wavs):
        if self.encoder_task.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        return padded_wav, wav_padding_mask

    def forward(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
        feat_select_idx: Union[str, list] = None,
        return_hidden_states: bool = False,
    ) -> Tuple[Union[torch.Tensor, list], torch.Tensor]:
        """Forward function for S3PRL speech encoder
        Args:
            wav (Union[torch.Tensor, list]): List of waveforms. (L, )
            wav_len (Union[torch.Tensor, list]): List of waveforms' lengths. Defaults to [].
            feat_select_idx (Union[str, list], optional): Feature selection indices. Defaults to None.
        Raises:
            KeyError: feat_select_idx is not "all", "hidden_states",
                      "last_hidden_state", or list.
        Returns:
            Tuple[Union[torch.Tensor, list], torch.Tensor]: Hidden features and their lengths.
        """

        if isinstance(wav, torch.Tensor):
            if wav.dim() == 2:
                if len(wav_len) > 0:
                    wav = [wav[b, : wav_len[b]] for b in range(len(wav))]
                else:
                    wav = [wav[b] for b in range(len(wav))]
            elif wav.dim() == 1:
                wav = [wav]

        if self.training:
            wav = [
                random_crop_max_length(wav[b], self.max_audio_len, len(wav[b]))
                for b in range(len(wav))
            ]

        padded_wav, wav_padding_mask = self.preprocess_input(wavs=wav)

        if self.trainable:
            # feat = self.encoder(wav)
            features = self.encoder.customHubertForward(
                padded_wav,
                padding_mask=wav_padding_mask,
                mask=None,
            )
        else:
            with torch.no_grad():
                # feat = self.encoder(wav)
                features = self.encoder.customHubertForward(
                    padded_wav,
                    padding_mask=wav_padding_mask,
                    mask=None,
                )

        if self.normalize_hiddenstates:
            # pass
            if self.normalize_type.startswith("method"):
                for i in range(len(features["layer_results"])):
                    if self.normalize_type == "method1":
                        # method1
                        features["layer_results"][i] = features["layer_results"][i] / (
                            torch.norm(
                                features["layer_results"][i], dim=-1, keepdim=True
                            )
                            + 1e-8
                        )
                    elif self.normalize_type == "method2":
                        # method2
                        features["layer_results"][i] = features["layer_results"][
                            i
                        ] / torch.mean(
                            torch.norm(features["layer_results"][i], dim=-1), dim=-1
                        ).view(
                            -1, 1, 1
                        )
                        # s3prl
                        # stacked_feature = F.layer_norm(stacked_feature, (stacked_feature.shape[-1],))

        feat = {
            "last_hidden_state": features["layer_results"][-1],
            "hidden_states": tuple(features["layer_results"]),
        }

        # if len(wav_len) == 0:
        wav_len = [len(w) for w in wav]

        feat_len = (
            torch.LongTensor([round(l / self.downsample_rate) for l in wav_len])
            .type_as(feat["last_hidden_state"])
            .long()
        )
        # .to(feat["last_hidden_state"].device)

        feat_len = torch.clamp_max(feat_len, feat["last_hidden_state"].shape[1])

        if feat_select_idx is None:
            feat_select_idx = self.feat_select_idx

        return_list = []
        if feat_select_idx == "all":
            return_list.extend([feat, feat_len])
        elif feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            return_list.extend(
                [self.weightedsum_layer(feat["hidden_states"]), feat_len]
            )
        elif isinstance(feat_select_idx, list):
            feat = [feat["hidden_states"][i] for i in feat_select_idx]
            return_list.extend([feat, feat_len])
        elif feat_select_idx in feat:
            return_list.extend([feat[feat_select_idx], feat_len])
        else:
            raise KeyError(feat_select_idx)

        if return_hidden_states:
            return_list.append(feat["hidden_states"])

        return tuple(return_list)

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from textlesslib.textless.data.hubert_feature_reader import HubertFeatureReader
from textlesslib.textless.data.kmeans_quantizer import KMeansQuantizer
from textlesslib.textless.checkpoint_manager import CHECKPOINT_MANAGER

# Values are expected to be (DenseModelClass, DenseModelBasename, DenseModelLayer, kwargs)
DENSE_MODELS = {
    "hubert-base-ls960": (HubertFeatureReader, "hubert-base-ls960", 6),
    "hubert-base-ls960-layer-9": (HubertFeatureReader, "hubert-base-ls960", 9),
    "mhubert-base-vp_mls_cv_8lang": (HubertFeatureReader, "mhubert-base-vp_mls_cv_8lang", 12),
    "mhubert-base-25hz": (HubertFeatureReader, "mhubert-base-25hz", 11, {'feat_hop_size': 640})
}

QUANTIZER_MODELS = {
    "kmeans": KMeansQuantizer,
    "kmeans-expresso": KMeansQuantizer,
}


# TODO: add kwargs everywhere
def dispatch_dense_model(name: str, **kwargs):
    model_class, model_basename, model_layer, *model_kwargs = DENSE_MODELS[name]
    # Add other model arguments to kwargs (i.e. {'feat_hop_size': 640})
    for mkwargs in model_kwargs:
        for k, v in mkwargs.items():
            # don't overwrite keys in kwargs
            if k not in kwargs:
                kwargs[k] = v
    checkpoint_path = CHECKPOINT_MANAGER.get_by_name(model_basename)
    return model_class(checkpoint_path, layer=model_layer, **kwargs)


def dispatch_quantizer(dense_model_name: str, quantizer_name: str, vocab_size: int):
    quantizer_checkpoint_name = f"{dense_model_name}-{quantizer_name}-{vocab_size}"
    checkpoint_path = CHECKPOINT_MANAGER.get_by_name(quantizer_checkpoint_name)
    quantizer = QUANTIZER_MODELS[quantizer_name](checkpoint_path)
    return quantizer

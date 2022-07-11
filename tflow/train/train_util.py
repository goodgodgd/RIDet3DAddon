import numpy as np
import config as cfg
import tensorflow as tf


def do_nothing(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


mode_decor = None
if cfg.Train.MODE in ["graph", "distribute"]:
    mode_decor = tf.function
else:
    mode_decor = do_nothing


def gt_feat_rename(features):
    new_feat = {"inst": {}, "feat2d": [], "feat3d": []}
    for key, val in features.items():
        if "feat_lane" in key:
            new_feat["feat_lane"] = val
        elif "feat2d" in key:
            new_feat["feat2d"].extend(val)
        elif "feat3d" in key:
            new_feat["feat3d"].extend(val)
        elif "image" in key:
            new_feat[key] = val
        elif "depth" in key:
            new_feat[key] = val
        else:
            new_feat["inst"][key] = val
    return new_feat


def create_batch_featmap(features_feat, featmap, key):
    if f"feat{key}" not in features_feat.keys():
        features_feat[f"feat{key}"] = []
    for scale, value in enumerate(featmap):
        value = value[np.newaxis, ...]
        if len(features_feat[f"feat{key}"]) < len(cfg.ModelOutput.FEATURE_SCALES):
            features_feat[f"feat{key}"].append(value)
        else:
            features_feat[f"feat{key}"][scale] = np.concatenate([features_feat[f"feat{key}"][scale], value], axis=0)
    return features_feat[f"feat{key}"]


def load_weights(model, ckpt_file):
    model.load_weights(ckpt_file)
    return model

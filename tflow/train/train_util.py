import numpy as np
import tensorflow as tf

import RIDet3DAddon.config as cfg3d
from train.tflow.train_util import do_nothing


mode_decor = None
if cfg3d.Train.MODE in ["graph", "distribute"]:
    mode_decor = tf.function
else:
    mode_decor = do_nothing


def gt_feat_rename(features):
    new_feat = {(f"inst{key[-2:]}" if "bboxes" in key else key): list() for key in features.keys()}
    for key, val in features.items():
        if isinstance(features[key], list):
            pass
        else:
            new_feat[key] = features[key]


        if "feat_lane" in key:
            new_feat[key] = val
        elif "feat2d" in key:
            new_feat[key].extend(val)
        elif "feat3d" in key:
            new_feat[key].extend(val)
        elif "inst2d" in key:
            new_feat["inst2d"].extend(val)
        elif "inst2d" in key:
            new_feat["inst3d"].extend(val)
        elif "image" in key:
            new_feat[key] = val
        elif "depth" in key:
            new_feat[key] = val
        elif "intrinsic" in key:
            new_feat[key] = val
    return new_feat


def create_batch_featmap(features_feat, featmap, key):
    if f"feat{key}" not in features_feat.keys():
        features_feat[f"feat{key}"] = []
    for scale, value in enumerate(featmap):
        value = value[np.newaxis, ...]
        if len(features_feat[f"feat{key}"]) < len(cfg3d.ModelOutput.FEATURE_SCALES):
            features_feat[f"feat{key}"].append(value)
        else:
            features_feat[f"feat{key}"][scale] = np.concatenate([features_feat[f"feat{key}"][scale], value], axis=0)
    return features_feat[f"feat{key}"]


def box_preprocess(grtr, pred, scale):
    h = pred["hwl"][scale][..., :1]
    wl = pred["hwl"][scale][..., 1:3]
    M = tf.cast(tf.cos(2*(pred["theta"][scale] - grtr["theta"][scale])) > 0, dtype=tf.float32)
    lw = tf.concat([pred["hwl"][scale][..., 2:3], pred["hwl"][scale][..., 1:2]], axis=-1)
    wl = M * wl + (1 - M) * lw
    pred["hwl"][scale] = tf.concat([h, wl], axis=-1)
    return pred

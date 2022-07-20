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
        if "feat_lane" in key:
            new_feat[key] = val
        elif "feat2d" in key:
            new_feat[key].extend(val)
        elif "feat3d" in key:
            new_feat[key].extend(val)
        elif "bboxes2d" in key:
            new_feat["inst2d"].extend(val)
        elif "bboxes3d" in key:
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

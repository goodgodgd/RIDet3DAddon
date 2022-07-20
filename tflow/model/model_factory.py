import tensorflow as tf
import numpy as np
from tensorflow import keras

import model.tflow.backbone as back
import model.tflow.head as head2d
import RIDet3DAddon.tflow.model.head_3d as head3d
import model.tflow.neck as neck
import RIDet3DAddon.tflow.model.decoder_2d as decoder2d
import RIDet3DAddon.tflow.model.decoder_3d as decoder3d
import utils.tflow.util_function as uf
import config as cfg
import RIDet3DAddon.config as cfg3d
import model.tflow.model_util as mu


class ModelFactory:
    def __init__(self, batch_size, input_shape, anchors_per_scale,
                 backbone_name=cfg3d.Architecture.BACKBONE,
                 neck_name=cfg3d.Architecture.NECK,
                 head_name=cfg3d.Architecture.HEAD,
                 backbone_conv_args=cfg3d.Architecture.BACKBONE_CONV_ARGS,
                 neck_conv_args=cfg3d.Architecture.NECK_CONV_ARGS,
                 head_conv_args=cfg3d.Architecture.HEAD_CONV_ARGS,
                 num_anchors_per_scale=cfg3d.ModelOutput.NUM_ANCHORS_PER_SCALE,
                 pred_composition=cfg3d.ModelOutput.PRED_HEAD_COMPOSITION,
                 pred3d_composition=cfg3d.ModelOutput.PRED_3D_HEAD_COMPOSITION,
                 out_channels=cfg3d.ModelOutput.NUM_MAIN_CHANNELS,
                 training=True
                 ):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors_per_scale = anchors_per_scale
        self.backbone_name = backbone_name
        self.neck_name = neck_name
        self.head_name = head_name
        self.backbone_conv_args = backbone_conv_args
        self.neck_conv_args = neck_conv_args
        self.head_conv_args = head_conv_args
        self.num_anchors_per_scale = num_anchors_per_scale
        self.pred_composition = pred_composition
        self.pred3d_composition = pred3d_composition
        self.out_channels = out_channels
        self.training = training
        mu.CustomConv2D.CALL_COUNT = -1
        print(f"[ModelFactory] batch size={batch_size}, input shape={input_shape}")
        print(f"[ModelFactory] backbone={self.backbone_name}, NECK={self.neck_name} ,HEAD={self.head_name}")

    def get_model(self):
        backbone_model = back.backbone_factory(self.backbone_name, self.backbone_conv_args, self.training)
        neck_model = neck.neck_factory(self.neck_name, self.neck_conv_args, self.training,
                                       self.num_anchors_per_scale, self.out_channels, None, None)
        head_model = head2d.head_factory(self.head_name, self.head_conv_args, self.training, self.num_anchors_per_scale,
                                         self.pred_composition, None, None)
        head3d_model = head3d.head_factory(self.head_name, self.head_conv_args, self.training,
                                           self.num_anchors_per_scale, self.pred3d_composition)

        input_tensor1 = tf.keras.layers.Input(shape=self.input_shape, batch_size=self.batch_size)
        input_tensor2 = tf.keras.layers.Input(shape=(3, 4), batch_size=self.batch_size)
        input_tensor = {"image": input_tensor1, "intrinsic": input_tensor2}

        backbone_features = backbone_model(input_tensor["image"])
        neck_features = neck_model(backbone_features)

        head_features = head_model(neck_features)

        output_features = {"inst": {}, "feat2d": [], "feat3d": [], "raw_feat": []}
        decode_2d_features = decoder2d.FeatureDecoder(self.anchors_per_scale)
        decode_3d_features = decoder3d.FeatureDecoder(self.anchors_per_scale)
        feature_scales = [scale for scale in head_features.keys() if "feature" in scale]
        for i, scale in enumerate(feature_scales):
            output_features["feat2d"].append(decode_2d_features(head_features[scale], i))
            if cfg3d.ModelOutput.FEAT_RAW:
                output_features["raw_feat"].append(head_features[scale])
                # output_features["backraw"].append(backbone_features[scale])
        head3d_features = head3d_model(neck_features, output_features["feat2d"])
        for i, scale in enumerate(feature_scales):
            output_features["feat3d"].append(decode_3d_features(head3d_features[scale], i, input_tensor["intrinsic"],
                                                                output_features["feat2d"][i][..., :4]))
        yolo_model = tf.keras.Model(inputs=input_tensor, outputs=output_features, name="yolo_model")
        return yolo_model


# ==================================================


def test_model_factory():
    print("===== start test_model_factory")
    anchors = [np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               ]
    batch_size = 1
    imshape = (512, 1280, 3)
    model = ModelFactory(batch_size, imshape, anchors).get_model()
    input_tensor = tf.zeros((batch_size, 512, 1280, 3))
    # print(model.summary())
    keras.utils.plot_model(model, to_file='Efficient_test.png', show_shapes=True)
    output = model(input_tensor)
    print("print output key and tensor shape")
    uf.print_structure("output", output)

    print("!!! test_model_factory passed !!!")


if __name__ == "__main__":
    uf.set_gpu_configs()
    test_model_factory()

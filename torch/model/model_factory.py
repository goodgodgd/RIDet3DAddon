import numpy as np
import torch

import model.framework.backbone as back
import model.framework.neck as neck
import RIDet3DAddon.torch.model.head as head2d
import RIDet3DAddon.torch.model.architecture as arch
import RIDet3DAddon.torch.model.head_3d as head3d
import RIDet3DAddon.torch.model.decoder_2d as decoder2d
import RIDet3DAddon.torch.model.decoder_3d as decoder3d
import utils.framework.util_function as uf
import config as cfg
import RIDet3DAddon.torch.config as cfg3d
import model.framework.model_util as mu


class ModelFactory:
    def __init__(self, batch_size, input_shape, anchors_per_scale,
                 architecture=cfg3d.Architecture.ARCHITECTURE,
                 backbone_name=cfg.Architecture.BACKBONE,
                 neck_name=cfg.Architecture.NECK,
                 head_name=cfg.Architecture.HEAD,
                 backbone_conv_args=cfg.Architecture.BACKBONE_CONV_ARGS,
                 neck_conv_args=cfg.Architecture.NECK_CONV_ARGS,
                 head_conv_args=cfg.Architecture.HEAD_CONV_ARGS,
                 num_anchors_per_scale=cfg.ModelOutput.NUM_ANCHORS_PER_SCALE,
                 pred2d_composition=cfg3d.ModelOutput.PRED_HEAD_COMPOSITION,
                 pred3d_composition=cfg3d.ModelOutput.PRED_3D_HEAD_COMPOSITION,
                 out_channels=cfg.ModelOutput.NUM_MAIN_CHANNELS,
                 training=True
                 ):

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors_per_scale = anchors_per_scale
        self.architecture = architecture
        self.backbone_name = backbone_name
        self.neck_name = neck_name
        self.head_name = head_name
        self.backbone_conv_args = backbone_conv_args
        self.neck_conv_args = neck_conv_args
        self.head_conv_args = head_conv_args
        self.num_anchors_per_scale = num_anchors_per_scale
        self.pred2d_composition = pred2d_composition
        self.pred3d_composition = pred3d_composition
        self.out_channels = out_channels
        self.training = training
        mu.CustomConv2D.CALL_COUNT = -1
        print(f"[ModelFactory] batch size={batch_size}, input shape={input_shape}")
        print(f"[ModelFactory] backbone={self.backbone_name}, NECK={self.neck_name} ,HEAD={self.head_name}")

    def get_model(self):
        backbone_model = back.backbone_factory(self.backbone_name, self.backbone_conv_args, self.training)
        neck_model = neck.neck_factory(self.neck_name, self.neck_conv_args, backbone_model.out_channels)
        head_model = head2d.head_factory(self.head_name, self.head_conv_args, neck_model.out_channels,
                                         self.num_anchors_per_scale, self.pred2d_composition)
        head3d_model = head3d.head_factory(self.head_name, self.head_conv_args, head_model.out_channels,
                                           self.num_anchors_per_scale, self.pred3d_composition)

        decode_2d = decoder2d.FeatureDecoder(self.anchors_per_scale)
        decode_3d = decoder3d.FeatureDecoder3D(self.anchors_per_scale)
        Architecture = arch.select_model(self.architecture)

        model = Architecture(backbone_model, neck_model, head_model, decode_2d, head3d_model, decode_3d)

        return model


# ==================================================


def test_model_factory():
    anchors = [np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               ]
    batch_size = 1
    imshape = (512, 1280, 3)
    input_test = torch.tensor(np.zeros((1, 3, 512, 1280)), device="cuda",dtype=torch.float32, requires_grad=True)
    model = ModelFactory(batch_size, imshape, anchors).get_model()
    output = model(input_test)
    uf.print_structure("output", output)
    # make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(
    #     "rnn_torchviz", format="png")



if __name__ == "__main__":
    uf.set_gpu_configs()
    test_model_factory()

import numpy as np


class LossComb:
    STANDARD = {"iou": ([1., 1., 1.], "IouLoss", True), "object_2d": ([1., 1., 1.], "BoxObjectnessLoss", 1, 1),
                "category_2d": ([1., 1., 1.], "MajorCategoryLoss"), "category_3d": ([1., 1., 1.], "MajorCategoryLoss"),
                "object_3d": ([1., 1., 1.], "Objectness3DLoss"), "theta": ([1., 1., 1.], "ThetaLoss", 1.1, 1.5)}


class Anchor:
    """
    anchor order MUST be compatible with Config.ModelOutput.FEATURE_ORDER
    in the current setting, the smallest anchor comes first
    """
    COCO_YOLOv3 = np.array([[13, 10], [30, 16], [23, 33],
                            [61, 30], [45, 62], [119, 59],
                            [90, 116], [198, 156], [326, 373]], dtype=np.float32)

    KITTI_YOLOv4 = np.array([[42, 51], [121, 52], [79, 52],
                             [51, 323], [251, 112], [166, 231],
                             [85, 692], [92, 1079], [282, 396]], dtype=np.float32)

    KITTI_RESOLUTION = (416, 416)


class TrainingPlan:
    KITTI_SIMPLE = [
        ("kitti", 10, 0.0001, LossComb.STANDARD, True),
        ("kitti", 50, 0.00001, LossComb.STANDARD, True)
    ]


class TfrParams:
    MIN_PIX = {'train': {"Pedestrian": 0, "Car": 0, "Cyclist": 0,
                         },
               'val': {"Pedestrian": 0, "Car": 0, "Cyclist": 0,
                       }
               }

    CATEGORY_NAMES = {"category": ["Pedestrian", "Car", "Cyclist"],
                      "dont": ["DontCare"]
                      }


class TrainParams:
    @classmethod
    def get_pred_composition(cls, iou_aware, categorized=False):
        cls_composition = {"category": len(TfrParams.CATEGORY_NAMES["category"])}
        reg_composition = {"yxhw": 4, "object": 1}
        if iou_aware:
            reg_composition["ioup"] = 1
        composition = {"reg": reg_composition, "cls": cls_composition}

        if categorized:
            out_composition = {name: sum(list(subdic.values())) for name, subdic in composition.items()}
        else:
            out_composition = dict()
            for names, subdic in composition.items():
                out_composition.update(subdic)

        return out_composition

    @classmethod
    def get_3d_pred_composition(cls, iou_aware, categorized=False):
        cls_composition = {"category": len(TfrParams.CATEGORY_NAMES["category"])}
        reg_composition = {"yxhwl": 5, "z": 1, "theta": 1, "object": 1}
        if iou_aware:
            reg_composition["ioup"] = 1
        composition = {"reg": reg_composition, "cls": cls_composition}

        if categorized:
            out_composition = {name: sum(list(subdic.values())) for name, subdic in composition.items()}
        else:
            out_composition = dict()
            for names, subdic in composition.items():
                out_composition.update(subdic)

        return out_composition


assert list(TfrParams.MIN_PIX["train"].keys()) == TfrParams.CATEGORY_NAMES["category"]
assert list(TfrParams.MIN_PIX["val"].keys()) == TfrParams.CATEGORY_NAMES["category"]

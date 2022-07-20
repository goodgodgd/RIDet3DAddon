class LossComb:
    STANDARD = {"bboxes_2d": {"iou": ([1., 1., 1.], "IouLoss", True),
                              "object_2d": ([1., 1., 1.], "BoxObjectnessLoss", 1, 1),
                              "category_2d": ([1., 1., 1.], "MajorCategoryLoss")},
                "bboxes_3d": {"category_3d": ([1., 1., 1.], "MajorCategoryLoss"),
                              "box_3d": ([1., 1., 1.], "Box3DLoss"), "theta": ([1., 1., 1.], "ThetaLoss", 1.1, 1.5)}}


class TrainingPlan:
    KITTI_SIMPLE = [
        ("kitti", 10, 0.001, LossComb.STANDARD, True),
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

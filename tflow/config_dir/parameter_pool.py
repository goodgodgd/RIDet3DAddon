class LossComb:
    STANDARD = {"box_2d": ([1., 1., 1.], "CiouLoss"),
                "object": ([1., 1., 1.], "BoxObjectnessLoss", 2, 3),
                # "neg_object": ([1., 1., 1.], "NegBoxObjectnessLoss", 1),
                "category_2d": ([1., 1., 1.], "MajorCategoryLoss", "2d"),
                # "category_3d": ([1., 1., 1.], "MajorCategoryLoss", "3d"),
                "yx": ([1., 1., 1.], "YXLoss"),
                "hwl": ([1., 1., 1.], "HWLLoss"),
                "depth": ([1., 1., 1.], "DepthLoss"),
                "theta": ([1., 1., 1.], "ThetaLoss", 1.1, 1.5)}
    # STANDARD = {"box_2d": ([1., 1., 1.], "IouL1SmoothLoss", True),
    #             "object": ([1., 1., 1.], "BoxObjectnessLoss", 1, 3),
    #             "category_2d": ([1., 1., 1.], "MajorCategoryLoss", "2d")}
    STANDARD_SCALE = {"box_2d": ([3., 3., 3.], "CiouLoss"),
                "object": ([2., 2., 2.], "BoxObjectnessLoss", 1, 3),
                # "neg_object": ([1., 1., 2.], "NegBoxObjectnessLoss", 1),
                "category_2d": ([5., 3., 1.], "MajorCategoryLoss", "2d"),
                # "category_3d": ([1., 1., 1.], "MajorCategoryLoss", "3d"),
                "yx": ([5., 5., 5.], "YXLoss"),
                "hwl": ([10., 10., 10.], "HWLLoss"),
                "depth": ([10., 10., 10.], "DepthLoss"),
                "theta": ([10., 10., 10.], "ThetaLoss", 1.1, 1.5)}


class TrainingPlan:
    KITTI_SIMPLE = [
        # ("kitti", 1, 0.00001, LossComb.STANDARD, True),
        ("kitti", 20, 0.001, LossComb.STANDARD, True),
        # ("kitti", 25, 0.001, LossComb.STANDARD_SCALE, True),
        ("kitti", 20, 0.0001, LossComb.STANDARD, True),
        # ("kitti", 50, 0.0001, LossComb.STANDARD_SCALE, True),
        ("kitti", 50, 0.00001, LossComb.STANDARD_SCALE, True),
    ]


class TfrParams:
    MIN_PIX = {'train': {"Bgd": 0, "Pedestrian": 0, "Car": 0, "Cyclist": 0},
               'val': {"Bgd": 0, "Pedestrian": 0, "Car": 0, "Cyclist": 0
                       }
               }

    CATEGORY_NAMES = {"category": ["Bgd", "Pedestrian", "Car", "Cyclist"],
                      "dont": ["DontCare"]
                      }


class TrainParams:
    @classmethod
    def get_pred_composition(cls, iou_aware, categorized=False):
        cls_composition = {"category": len(TfrParams.CATEGORY_NAMES["category"])}
        reg_composition = {"yxhw": 4, "z": 1, "object": 1}
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
        # reg_composition = {"yxz": 3, "hwl": 3, "theta": 1}
        reg_composition = {"yx": 2, "hwl": 3, "theta": 1}
        if iou_aware:
            reg_composition["ioup"] = 1
        composition = {"reg": reg_composition}

        if categorized:
            out_composition = {name: sum(list(subdic.values())) for name, subdic in composition.items()}
        else:
            out_composition = dict()
            for names, subdic in composition.items():
                out_composition.update(subdic)

        return out_composition


assert list(TfrParams.MIN_PIX["train"].keys()) == TfrParams.CATEGORY_NAMES["category"]
assert list(TfrParams.MIN_PIX["val"].keys()) == TfrParams.CATEGORY_NAMES["category"]

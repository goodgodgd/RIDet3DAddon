class LossComb:
    STANDARD = {"box_2d": ([1., 1., 1.], "CiouLoss"),
                "object": ([4., 2., 1.], "BoxObjectnessLoss", 1, 5),
                # "neg_object": ([1., 1., 1.], "NegBoxObjectnessLoss", 1),
                "category": ([5., 5., 5.], "MajorCategoryLoss"),
                # "category_3d": ([1., 1., 1.], "MajorCategoryLoss", "3d"),
                "yx": ([10., 10., 10.], "YXLoss"),
                "hwl": ([100., 100., 100.], "HWLLoss"),
                "depth": ([1., 1., 1.], "DepthLoss"),
                "theta": ([10., 10., 10.], "ThetaLoss", 1.1, 1.5),
                "occluded": ([5., 5., 5.], "OccludedLoss")}
    # STANDARD = {"box_2d": ([1., 1., 1.], "IouL1SmoothLoss", True),
    #             "object": ([1., 1., 1.], "BoxObjectnessLoss", 1, 3),
    #             "category_2d": ([1., 1., 1.], "MajorCategoryLoss", "2d")}
    STANDARD_SCALE = {"box_2d": ([3., 3., 3.], "CiouLoss"),
                      "object": ([4., 2., 1.], "BoxObjectnessLoss", 2, 10),
                      # "neg_object": ([1., 1., 2.], "NegBoxObjectnessLoss", 1),
                      "category": ([5., 5., 5.], "MajorCategoryLoss"),
                      # "category_3d": ([1., 1., 1.], "MajorCategoryLoss", "3d"),
                      "yx": ([100., 100., 100.], "YXLoss"),
                      "hwl": ([100., 100., 100.], "HWLLoss"),
                      "depth": ([10., 10., 10.], "DepthLoss"),
                      "theta": ([10., 10., 10.], "ThetaLoss", 1.1, 1.5),
                      "occluded": ([5., 5., 5.], "OccludedLoss")}


class TrainingPlan:
    KITTI_SIMPLE = [
        # ("kitti", 10, 0.004, LossComb.STANDARD, True),
        # ("kitti", 10, 0.0025, LossComb.STANDARD, True),
        # ("kitti", 6, 0.002, LossComb.STANDARD, True),
        # ("kitti", 10, 0.0015, LossComb.STANDARD, True),

        # ("kitti", 10, 0.001, LossComb.STANDARD, True),
        # # ("kitti", 20, 0.0005, LossComb.STANDARD, True),
        # ("kitti", 10, 0.0001, LossComb.STANDARD, True),
        # ("kitti", 10, 0.0001, LossComb.STANDARD_SCALE, True),
        # ("kitti", 10, 0.00005, LossComb.STANDARD_SCALE, True),
        # # ("kitti", 20, 0.00005, LossComb.STANDARD_SCALE, True),
        # ("kitti", 10, 0.00001, LossComb.STANDARD_SCALE, True),

        ("kitti", 10, 0.0008, LossComb.STANDARD, True),
        ("kitti", 10, 0.0006, LossComb.STANDARD, True),
        ("kitti", 10, 0.0004, LossComb.STANDARD, True),
        ("kitti", 10, 0.0002, LossComb.STANDARD, True),

        ("kitti", 10, 0.0002, LossComb.STANDARD_SCALE, True),
        ("kitti", 10, 0.00008, LossComb.STANDARD_SCALE, True),
        ("kitti", 10, 0.00006, LossComb.STANDARD_SCALE, True),
        ("kitti", 10, 0.00004, LossComb.STANDARD_SCALE, True),
        ("kitti", 10, 0.00002, LossComb.STANDARD_SCALE, True),

        ("kitti", 10, 0.000008, LossComb.STANDARD_SCALE, True),
        ("kitti", 10, 0.000006, LossComb.STANDARD_SCALE, True),
        ("kitti", 10, 0.000004, LossComb.STANDARD_SCALE, True),
        ("kitti", 10, 0.000002, LossComb.STANDARD_SCALE, True),
        ("kitti", 10, 0.000001, LossComb.STANDARD_SCALE, True),
    ]


class TfrParams:
    MIN_PIX = {'train': {"Bgd": 0, "Pedestrian": 0, "Car": 0, "Cyclist": 0},
               'val': {"Bgd": 0, "Pedestrian": 0, "Car": 0, "Cyclist": 0
                       }
               }

    CATEGORY_NAMES = {"category": ["Bgd", "Pedestrian", "Car", "Cyclist"],
                      "dont": ["DontCare"]
                      }
    # MIN_PIX = {'train': {"Bgd": 0, "Car": 0},
    #            'val': {"Bgd": 0, "Car": 0
    #                    }
    #            }
    #
    # CATEGORY_NAMES = {"category": ["Bgd",  "Car"],
    #                   "dont": ["DontCare"]
    #                   }


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
        cls_composition = {"occluded": 3}
        # reg_composition = {"yxz": 3, "hwl": 3, "theta": 1}
        reg_composition = {"yx": 2, "hwl": 3, "theta": 1}
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

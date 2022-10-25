import numpy as np


class LossComb:
    STANDARD = {"ciou": ([1., 1., 1.], "CiouLoss"), "object": ([1., 1., 1.], "BoxObjectnessLoss", 1, 1),
                "category": ([1., 1., 1.], "MajorCategoryLoss")}

    BEV = {"box_2d": ([1., 1., 1.], "L1smooth"),
           "object": ([.1, .1, .1], "BoxObjectnessLoss", 1, 1),
           "category_2d": ([.1, .1, .1], "MajorCategoryLoss", "2d"),
           "category_3d": ([.1, .1, .1], "MajorCategoryLoss", "3d"),
           "box_3d": ([1., 1., 1.], "Box3DLoss"),
           "theta": ([2., 2., 2.], "ThetaLoss", 1.1, 1.5)
           }
    # BEV = {
    #        "object": ([1., 1., 1.], "BoxObjectnessLoss", 1, 1),
    #
    #      }
    # STANDARD = {"iou": ([1., 1., 1.], "IoUL1smooth"),
    #             "object_2d": ([1., 1., 1.], "BoxObjectnessLoss", 1, 1),
    #             "category_2d": ([1., 1., 1.], "MajorCategoryLoss", "2d"),
    #             "category_3d": ([1., 1., 1.], "MajorCategoryLoss", "3d"),
    #             "box_3d": ([1., 1., 1.], "Box3DLoss"),
    #             "theta": ([1., 1., 1.], "ThetaLoss", 1.1, 1.5)
    #             }


class KittiBEVParam:
    class Deg0:
        CELL_SIZE = 1 / 16
        TILT_ANGLE = 0
        INPUT_RESOLUTION = [768, 768]
        POINT_XY_RANGE = [0, INPUT_RESOLUTION[0] * CELL_SIZE / np.cos(TILT_ANGLE),
                          -INPUT_RESOLUTION[1] * CELL_SIZE / 2, INPUT_RESOLUTION[1] * CELL_SIZE / 2]
        # POINT_XY_RANGE = [0, 48, -24, 24]
        # INPUT_RESOLUTION = [int((POINT_XY_RANGE[1] - POINT_XY_RANGE[0]) * np.cos(TILT_ANGLE) / CELL_SIZE),
        #                     int((POINT_XY_RANGE[3] - POINT_XY_RANGE[2]) / CELL_SIZE),
        #                     ]

    class Deg30:
        CELL_SIZE = 1 / 16
        TILT_ANGLE = np.pi / 6
        INPUT_RESOLUTION = [640, 768]
        POINT_XY_RANGE = [0, INPUT_RESOLUTION[0] * CELL_SIZE / np.cos(TILT_ANGLE),
                          -INPUT_RESOLUTION[1] * CELL_SIZE / 2, INPUT_RESOLUTION[1] * CELL_SIZE / 2]
        # POINT_XY_RANGE = [0, 48.5, -24, 24]
        # INPUT_RESOLUTION = [int((POINT_XY_RANGE[1] - POINT_XY_RANGE[0]) * np.cos(TILT_ANGLE) / CELL_SIZE),
        #                     int((POINT_XY_RANGE[3] - POINT_XY_RANGE[2]) / CELL_SIZE),
        #                     ]

    class Deg45:
        CELL_SIZE = 1 / 16
        TILT_ANGLE = np.pi / 4
        INPUT_RESOLUTION = [512, 768]
        POINT_XY_RANGE = [0, INPUT_RESOLUTION[0] * CELL_SIZE / np.cos(TILT_ANGLE),
                          -INPUT_RESOLUTION[1] * CELL_SIZE / 2, INPUT_RESOLUTION[1] * CELL_SIZE / 2]
        # POINT_XY_RANGE = [0, 48.1, -24, 24]
        # INPUT_RESOLUTION = [int((POINT_XY_RANGE[1] - POINT_XY_RANGE[0]) * np.cos(TILT_ANGLE) / CELL_SIZE),
        #                     int((POINT_XY_RANGE[3] - POINT_XY_RANGE[2]) / CELL_SIZE),
        #                     ]

    class Deg60:
        CELL_SIZE = 1 / 16
        TILT_ANGLE = np.pi / 3
        INPUT_RESOLUTION = [384, 768]
        POINT_XY_RANGE = [0, INPUT_RESOLUTION[0] * CELL_SIZE / np.cos(TILT_ANGLE),
                          -(INPUT_RESOLUTION[1] * CELL_SIZE) / 2, (INPUT_RESOLUTION[1] * CELL_SIZE) / 2]
        # POINT_XY_RANGE = [0, 64, -32, 32]
        # INPUT_RESOLUTION = [int((POINT_XY_RANGE[1] - POINT_XY_RANGE[0]) * np.cos(TILT_ANGLE) / CELL_SIZE),
        #                     int((POINT_XY_RANGE[3] - POINT_XY_RANGE[2]) / CELL_SIZE),
        #                     ]

    Deg = Deg60


class TrainingPlan:
    KITTIBEV_SIMPLE = [
        # ("kittibev", 1, 0.0000001, LossComb.BEV, True),
        ("kittibev", 1, 1e-06, LossComb.BEV, True),
        ("kittibev", 50, 1e-04, LossComb.BEV, True),
        ("kittibev", 50, 1e-05, LossComb.BEV, True),
        ("kittibev", 50, 1e-06, LossComb.BEV, True),
        ("kittibev", 50, 1e-07, LossComb.BEV, True)
    ]


class TfrParams:
    MIN_PIX = {'train': {"Bgd": 0, "Person": 0, "Car": 0, "Bicycle": 0,
                         },
               'val': {"Bgd": 0, "Person": 0, "Car": 0, "Bicycle": 0,
                         }
               }

    CATEGORY_NAMES = {"category": ["Bgd", "Person", "Car", "Bicycle", ],
                      "dont": ["Don't Care"],
                      }


class TrainParams:
    @classmethod
    def get_pred_composition(cls, categorized=False):
        cls_composition = {"category": len(TfrParams.CATEGORY_NAMES["category"])}
        reg_composition = {"yxhw": 4, "object": 1}
        composition = {"reg": reg_composition, "cls": cls_composition}

        if categorized:
            out_composition = {name: sum(list(subdic.values())) for name, subdic in composition.items()}
        else:
            out_composition = dict()
            for names, subdic in composition.items():
                out_composition.update(subdic)

        return out_composition

    @classmethod
    def get_3d_pred_composition(cls, categorized=False):
        cls_composition = {"category": len(TfrParams.CATEGORY_NAMES["category"])}
        reg_composition = {"xyz": 3, "lwh": 3, "theta": 1}
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

import os
import numpy as np
import os.path as op

import RIDet3DAddon.config as cfg3d
import utils.tflow.util_function as uf


class SavePred:
    def __init__(self, result_path):
        self.result_path = result_path
        if not op.isdir(result_path):
            os.makedirs(result_path, exist_ok=True)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg3d.Dataloader.CATEGORY_NAMES["category"])}

    def __call__(self, step, grtr, pred):
        batch, _, __ = pred["inst2d"]["yxhw"].shape
        for i in range(batch):
            filename = op.join(self.result_path, f"{step * batch + (i):06d}.txt")
            im_shape = grtr["image_shape"][i]
            bbox_2d = self.extract_valid_data(pred["inst2d"], i, "object")
            bbox_3d = self.extract_valid_data(pred["inst3d"], i, "category")
            tlbr_bboxes = uf.convert_box_format_yxhw_to_tlbr(bbox_2d["yxhw"])
            file = open(os.path.join(filename), 'w')
            text_to_write = ''
            for n in range(bbox_2d["yxhw"].shape[0]):
                tlbr = tlbr_bboxes[n]
                ctgr = bbox_2d["category"][n].astype(int)
                yx = bbox_3d["yx"][n]
                z = bbox_2d["z"][n]
                hwl = bbox_3d["hwl"][n]
                ry = bbox_3d["theta"][n]
                score = bbox_2d["score"][n]
                alpha = np.arctan2(z, yx[-1])
                ctgr = self.categories[ctgr[0]]
                text_to_write += (f'{ctgr} {-1} {-1} {alpha[0]:.6f} '
                                  f'{tlbr[1] * im_shape[1]:.6f} {tlbr[0] * im_shape[0]:.6f} {tlbr[3] * im_shape[1]:.6f} '
                                  f'{tlbr[2] * im_shape[0]:.6f} {hwl[0]:.6f} {hwl[1]:.6f} {hwl[2]:.6f} {yx[1]:.6f} '
                                  f'{yx[0] + (hwl[0]/2):.6f} {z[0]:.6f} {ry[0]:.6f} {score[0]} \n')
            file.write(text_to_write)
            file.close()

    def extract_valid_data(self, inst_data, i, mask_key):
        """
        remove zero padding from bboxes
        """
        valid_data = {}
        valid_mask = (inst_data[mask_key][i] > 0).flatten()
        for key, data in inst_data.items():
            valid_data[key] = data[i][valid_mask]
        return valid_data




import os
import numpy as np
import os.path as op

import RIDet3DAddon.torch.config as cfg3d
import utils.framework.util_function as uf


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
            im_shape = grtr["image"][i].shape
            bbox_2d = self.extract_valid_data(pred["inst2d"], i, "object")
            bbox_3d = self.extract_valid_data(pred["inst3d"], i, "category")
            tlbr_bboxes = uf.convert_box_format_yxhw_to_tlbr(bbox_2d["yxhw"])
            file = open(os.path.join(filename), 'w')
            text_to_write = ''
            for n in range(bbox_2d["yxhw"].shape[0]):
                tlbr = tlbr_bboxes[n]
                yxhw = bbox_2d["yxhw"][n]
                ctgr = bbox_2d["category"][n].astype(int)
                lxyz = bbox_3d["lxyz"][n]
                cxyz = bbox_3d["cxyz"][n]

                hwl = bbox_3d["hwl"][n]
                ry = bbox_3d["theta"][n]
                score = bbox_2d["score"][n]
                alpha = np.arctan2(cxyz[-1:], cxyz[:1])
                # yxz = [0, 0, 0]
                # hwl = [0, 0, 0]
                # ry = [0]
                # score = [0]
                # alpha = 0
                ctgr = self.categories[ctgr[0]]
                # TODO score를 기준으로 짤라서 저장, 데이터 50m 제한두기
                # if score >= 0.24:
                text_to_write += (
                        '{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                        + '{:.6f} {:.6f} '
                        + '{:.6f} {:.6f}\n').format(ctgr, alpha[0], yxhw[0] * im_shape[0],
                                                    yxhw[1] * im_shape[1],
                                                    yxhw[2] * im_shape[0],
                                                    yxhw[3] * im_shape[1],
                                                    lxyz[0], lxyz[1], lxyz[2],
                                                    hwl[0], hwl[1], hwl[2],
                                                    cxyz[0], cxyz[1], cxyz[2], ry[0], score[0])
            file.write(text_to_write)
            file.close()

        self.save_data(pred)

    def save_data(self, pred):
        pass

    def extract_valid_data(self, inst_data, i, mask_key):
        """
        remove zero padding from bboxes
        """
        valid_data = {}
        valid_mask = (inst_data[mask_key][i] > 0).flatten()
        for key, data in inst_data.items():
            valid_data[key] = data[i][valid_mask]
        return valid_data

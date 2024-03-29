import os
import os.path as op
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from matplotlib import pyplot as plt

import settings
import RIDet3DAddon.config as cfg3d
import RIDet3DAddon.tflow.config_dir.util_config as uc3d
from dataloader.tflow.dataset_reader import DatasetReader
from RIDet3DAddon.tflow.model.model_factory import ModelFactory
import RIDet3DAddon.tflow.utils.util_function as uf3d
import utils.tflow.util_function as uf
import train.framework.train_util as tu
import RIDet3DAddon.tflow.model.nms as nms
from RIDet3DAddon.tflow.log.metric import count_true_positives
from RIDet3DAddon.tflow.train.feature_generator import FeatureMapDistributer


class EvaluateNmsParams:
    """
    evaluate performance for each param combination
    -> total_eval_result.csv
    """

    def __init__(self):
        self.dataset_name = cfg3d.Datasets.TARGET_DATASET
        self.ckpt_path = op.join(cfg3d.Paths.CHECK_POINT, cfg3d.Train.CKPT_NAME)
        self.feat_scales = cfg3d.ModelOutput.FEATURE_SCALES
        self.nms = nms.NonMaximumSuppression()
        self.num_ctgr = len(cfg3d.Dataloader.CATEGORY_NAMES["category"])

    def create_eval_file(self):
        dataset, steps, model, anchors_per_scale, feature_creator = self.load_dataset_model(self.dataset_name, self.ckpt_path)
        perf_data = self.collect_recall_precision(dataset, steps, model, self.num_ctgr, anchors_per_scale, feature_creator)
        self.save_results(perf_data, self.ckpt_path)

    def load_dataset_model(self, dataset_name, ckpt_path):
        batch_size, train_mode, anchors = cfg3d.Train.BATCH_SIZE, cfg3d.Train.MODE, cfg3d.AnchorGeneration.ANCHORS
        tfrd_path = cfg3d.Paths.DATAPATH
        dataset, steps, imshape, anchors_per_scale \
            = self.get_dataset(tfrd_path, dataset_name, False, batch_size, "val", anchors)
        model = ModelFactory(batch_size, imshape, anchors_per_scale).get_model()
        model = self.try_load_weights(ckpt_path, model)
        feature_creator = FeatureMapDistributer(cfg3d.FeatureDistribPolicy.POLICY_NAME, imshape, anchors_per_scale)
        return dataset, steps, model, anchors_per_scale, feature_creator

    def get_dataset(self, tfrd_path, dataset_name, shuffle, batch_size, split, anchors):
        tfrpath = op.join(tfrd_path, f"{dataset_name}_{split}")
        reader = DatasetReader(tfrpath, shuffle, batch_size, 1)
        dataset = reader.get_dataset()
        frames = reader.get_total_frames()
        tfr_cfg = reader.get_dataset_config()
        image_shape = tfr_cfg["image"]["shape"]
        if "depth" in tfr_cfg:
            image_shape[-1] += tfr_cfg["depth"]["shape"][-1]
        # anchor sizes per scale in pixel
        anchors_per_scale = np.array([anchor / np.array([image_shape[:2]]) for anchor in anchors], dtype=np.float32)
        print(f"[get_dataset] dataset={dataset_name}, image shape={image_shape}, "
              f"frames={frames},\n\tanchors={anchors_per_scale}")
        return dataset, frames // batch_size, image_shape, anchors_per_scale

    def try_load_weights(self, ckpt_path, model, weights_suffix='latest'):
        ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
        if op.isfile(ckpt_file):
            print(f"===== Load weights from checkpoint: {ckpt_file}")
            model.load_weights(ckpt_file)
        else:
            print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
        return model

    def collect_recall_precision(self, dataset, steps, model, num_ctgr, anchors_per_scale, feature_creator):
        results = {"max_box": [], "iou_thresh": [], "score_thresh": []}
        for max_box in cfg3d.NmsOptim.MAX_OUT_CANDIDATES:
            for iou_thresh in cfg3d.NmsOptim.IOU_CANDIDATES:
                for score_thresh in cfg3d.NmsOptim.SCORE_CANDIDATES[::-1]:
                    results["max_box"].append(max_box)
                    results["iou_thresh"].append(iou_thresh)
                    results["score_thresh"].append(score_thresh)

        results = {key: np.array(val) for key, val in results.items()}
        num_params = results["max_box"].shape[0]
        accum_keys = ["trpo", "grtr", "pred"]
        init_data = {key: np.zeros((num_params, num_ctgr), dtype=np.float32) for key in accum_keys}
        results.update(init_data)
        for step, grtr in enumerate(dataset):
            grtr["image"] = np.concatenate([grtr["image"], grtr["depth"]], axis=-1)
            # grtr = tu.gt_feat_rename(grtr)
            # for i in range(grtr["image"].shape[0]):
            #     feat_sizes = [np.array(grtr["image"][i].shape[:2]) // scale for scale in self.feat_scales]
            #     featmap = feature_creator.create(grtr["inst"]["bboxes"][i].numpy(), feat_sizes)
            #     grtr["feat"] = tu.create_batch_featmap(grtr, featmap)
            start = timer()
            # grtr_slices = uf3d.merge_and_slice_features(grtr, True)
            # for key, feature_slices in grtr_slices.items():
            #     grtr_slices[key] = uf.convert_tensor_to_numpy(feature_slices)
            pred = model((grtr["image"], grtr["intrinsic"]))
            # pred_slices = uf3d.merge_and_slice_features(pred, False)

            for i in range(num_params):
                max_box = np.ones((num_ctgr,), dtype=np.float32) * results["max_box"][i]
                iou_thresh = np.ones((num_ctgr,), dtype=np.float32) * results["iou_thresh"][i]
                score_thresh = np.ones((num_ctgr,), dtype=np.float32) * results["score_thresh"][i]
                pred_bboxes, _ = self.nms(pred, max_box, iou_thresh, score_thresh)
                pred_bboxes = uf3d.slice_feature(pred_bboxes, uc3d.get_bbox_composition(False))
                grtr_bboxes = uf3d.slice_feature(grtr["inst2d"], uc3d.get_bbox_composition(True))
                grtr_bboxes = uf.convert_tensor_to_numpy(grtr_bboxes)
                pred_bboxes = self.convert_tensor_to_numpy(pred_bboxes)
                count_per_class = count_true_positives(grtr_bboxes, pred_bboxes,
                                                       num_ctgr, iou_thresh=cfg3d.Validation.MAP_TP_IOU_THRESH,
                                                       per_class=True)

                for key in accum_keys:
                    results[key][i] += count_per_class[key]

            uf.print_progress(f"=== step: {step}/{steps}, took {timer() - start:1.2f}s")
            # if step > 1:
            #     break

        results["recall"] = np.divide(results["trpo"], results["grtr"], out=np.zeros_like(results["trpo"]),
                                      where=(results["grtr"] != 0))
        results["precision"] = np.divide(results["trpo"], results["pred"], out=np.zeros_like(results["trpo"]),
                                         where=(results["pred"] != 0))
        results["min_perf"] = np.minimum(results["recall"], results["precision"])
        results["avg_perf"] = (results["recall"] + results["precision"]) / 2.

        for key, val in results.items():
            print(f"results: {key}\n{val[:10]}")
        return results

    def convert_tensor_to_numpy(self, feature):
        numpy_feature = dict()
        for key, value in feature.items():
            if isinstance(value, dict):
                sub_feature = dict()
                for sub_key, sub_value in value.items():
                    sub_feature[sub_key] = sub_value.numpy()
                numpy_feature[key] = sub_feature
            else:
                numpy_feature[key] = value.numpy()
        return numpy_feature

    def save_results(self, perf_data, ckpt_path):
        param_path = op.join(ckpt_path, "nms_param")
        os.makedirs(param_path, exist_ok=True)
        intkeys = ["trpo", "grtr", "pred"]

        specific_summary = self.specific_data(perf_data, self.num_ctgr)
        specific_summary.to_csv(op.join(param_path, "specific_summary.csv"), index=False, float_format="%1.4f")

    def specific_data(self, data, num_ctgr):
        specific_summary = dict()
        class_order_data = {"class": np.tile(np.arange(0, num_ctgr, 1), data["score_thresh"].shape[0])}
        thresh_dict, re_pr_dict = self.change_data_form(data, num_ctgr)
        for update_data in [thresh_dict, class_order_data, re_pr_dict]:
            specific_summary.update(update_data)
        columns = specific_summary.keys()
        specific_values = np.stack(specific_summary.values(), axis=0).transpose()
        specific_summary = pd.DataFrame(specific_values, columns=columns)
        return specific_summary

    def change_data_form(self, data, num_categories):
        dim_one_data = dict()
        dim_two_data = dict()
        need_key = ["max_box", "iou_thresh", "score_thresh", "recall", "precision", "average_prec", "mean_ap",
                    "min_perf", "trpo", "grtr", "pred"]
        for key, data in data.items():
            if data.ndim == 1:
                if key in need_key:
                    dim_one_data[key] = np.repeat(data, num_categories, axis=0)
            elif data.ndim == 2:
                if key in need_key:
                    dim_two_data[key] = data.flatten()
        return dim_one_data, dim_two_data


class FindBestParamByAP:
    """
    find the best param combination by AP
    -> optim_result.csv : best param per class
        pr_curve_{class}.png
    """

    def __init__(self):
        self.file_dir = op.join(cfg3d.Paths.CHECK_POINT, cfg3d.Train.CKPT_NAME)
        self.filename = self.find_nms_path(self.file_dir)
        self.num_ctgr = len(cfg3d.Dataloader.CATEGORY_NAMES["category"])

    def find_nms_path(self, ckpt_path):
        param_dir = ckpt_path + "/nms_param"
        ckpt_dir = os.listdir(ckpt_path)
        if "nms_param" in ckpt_dir:
            return param_dir + "/specific_summary.csv"
        else:
            assert 0, f"not exist {param_dir}"

    def create_all_param_ap(self):
        data = pd.read_csv(self.filename)
        best_params = self.param_summarize(data, self.num_ctgr)
        ap_all_data, mean_ap_all_data, wt_ap_all_data = self.compute_ap_all_class(data, self.num_ctgr)
        total_params = {"best_params": best_params, "ap_data": ap_all_data, "mean_ap_all_data": mean_ap_all_data,
                        "wt_ap_all_data": wt_ap_all_data}
        self.save_results(total_params)

    def param_summarize(self, data, num_ctgr):
        params = pd.DataFrame()
        for n_class in range(num_ctgr):
            class_data = data.loc[data["class"] == n_class]
            max_index = np.argmax(class_data["min_perf"])
            select_param = class_data.iloc[max_index]
            params = params.append(select_param, ignore_index=True)
        del params["min_perf"]
        return params

    def compute_ap_all_class(self, data, num_ctgr):
        max_out_vals = data["max_box"].unique()
        iou_vals = data["iou_thresh"].unique()
        ap_outputs = []
        for max_out in max_out_vals:
            for iou in iou_vals:
                for ctgr in range(num_ctgr):
                    ap_out = {"max_box": max_out, "iou_thresh": iou, "class": ctgr}
                    ap_out["ap"], ap_out["grtr"] = self.compute_ap(data, max_out, iou, ctgr)
                    ap_outputs.append(ap_out)
        ap_outputs = pd.DataFrame(ap_outputs)

        mean_ap_outputs = []
        wt_ap_outputs = []
        for max_out in max_out_vals:
            for iou in iou_vals:
                mean_ap_out = {"max_box": max_out, "iou_thresh": iou}
                wt_ap_out = {"max_box": max_out, "iou_thresh": iou}
                mean_ap_out["mean_ap"], wt_ap_out["wt_ap"] = self.compute_mean_ap(ap_outputs, max_out, iou)
                mean_ap_outputs.append(mean_ap_out)
                wt_ap_outputs.append(wt_ap_out)
        mean_ap_outputs = pd.DataFrame(mean_ap_outputs)
        wt_ap_outputs = pd.DataFrame(wt_ap_outputs)
        return ap_outputs, mean_ap_outputs, wt_ap_outputs

    def compute_ap(self, data, max_out, iou, ctgr):
        mask = (data['iou_thresh'] == iou) & (data['max_box'] == max_out) & (data['class'] == ctgr)
        data = data.loc[mask, :]
        data = data.reset_index()
        apdata = data.loc[:, ["recall", "precision", "grtr"]]
        apdata = apdata.sort_values(by="recall")
        apdata = apdata.reset_index()
        length = apdata.shape[0]
        max_pre = apdata["precision"].copy()
        for score in range(length - 2, -1, -1):
            max_pre[score] = np.maximum(max_pre[score], max_pre[score + 1])
        ap = 0
        recall = apdata["recall"]
        precision = max_pre
        for i in range(apdata.shape[0] - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]
        return ap, apdata["grtr"].sum()

    def compute_mean_ap(self, data, max_out, iou):
        mask = (data['iou_thresh'] == iou) & (data['max_box'] == max_out)
        data = data.loc[mask, :]
        data = data.reset_index()
        grtr_num = data["grtr"]
        ap = data["ap"]
        wt_ap = np.sum(ap * grtr_num) / np.sum(grtr_num)
        mean_ap = np.mean(ap, axis=0)
        return mean_ap, wt_ap

    def save_results(self, total_params):
        for key, data in total_params.items():
            data.to_csv(op.join(self.file_dir, f"nms_param/{key}.csv"), index=False, float_format="%1.4f")

    def draw_main(self, max_box=None, iou_thresh=None):
        data = pd.read_csv(self.filename)
        nms_param_dir = self.file_dir + "/nms_param"
        if iou_thresh is None and max_box is None:
            iou_thresh = data["iou_thresh"].unique()
            max_box = data["max_box"].unique()
        elif max_box is None:
            max_box = data["max_box"].unique()
        elif iou_thresh is None:
            iou_thresh = data["iou_thresh"].unique()
        ap_outputs = []
        for max_out in max_box:
            for iou in iou_thresh:
                for ctgr in range(self.num_ctgr):
                    ap_out = {"max_box": max_out, "iou_thresh": iou, "class": ctgr}
                    ap_out["ap"], ap_out["grtr"] = self.compute_ap(data, max_out, iou, ctgr)
                    self.draw_select_ap_curve(data, max_out, iou, ctgr)
                    ap_outputs.append(ap_out)
        ap_outputs = pd.DataFrame(ap_outputs)

        mean_ap_outputs = []
        wt_ap_outputs = []
        for max_out in max_box:
            for iou in iou_thresh:
                mean_ap_out = {"max_box": max_out, "iou_thresh": iou}
                wt_ap_out = {"max_box": max_out, "iou_thresh": iou}
                mean_ap_out["mean_ap"], wt_ap_out["wt_ap"] = self.compute_mean_ap(ap_outputs, max_out, iou)
                mean_ap_outputs.append(mean_ap_out)
                wt_ap_outputs.append(wt_ap_out)
        mean_ap_outputs = pd.DataFrame(mean_ap_outputs)
        wt_ap_outputs = pd.DataFrame(wt_ap_outputs)

        ap_outputs.to_csv(op.join(nms_param_dir, f"select_ap.csv"), index=False, float_format="%1.4f")
        mean_ap_outputs.to_csv(op.join(nms_param_dir, f"select_mean_ap.csv"), index=False, float_format="%1.4f")
        wt_ap_outputs.to_csv(op.join(nms_param_dir, f"select_wt_ap.csv"), index=False, float_format="%1.4f")

    def draw_select_ap_curve(self, data, select_max_out, select_iou, category):
        mask = (data['iou_thresh'] == select_iou) & (data['max_box'] == select_max_out) & (data['class'] == category)
        data = data.loc[mask, :]
        data = data.reset_index()
        data = data.loc[:, ["recall", "precision"]]
        data = data.sort_values(by="recall")
        data = data.reset_index()
        length = data.shape[0]
        max_pre = data["precision"].copy()
        for score in range(length - 2, -1, -1):
            max_pre[score] = np.maximum(max_pre[score], max_pre[score + 1])
        data["max_pre"] = max_pre
        name_suff = f"{select_max_out}_{select_iou}"
        self.draw_ap_curve(data["recall"], data["precision"], data["max_pre"], category, name_suff)

    def draw_ap_curve(self, recall, precision, max_pre, ctgr, name_suff):
        image_dir = self.file_dir + "/nms_param/curve_image"
        if not op.isdir(image_dir):
            os.makedirs(image_dir, exist_ok=True)
        plt.subplot(121)
        plt.plot(recall, precision, 'r-o')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")

        plt.subplot(122)
        plt.plot(recall, max_pre, 'b-o')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("AP")

        plt.savefig(op.join(image_dir, f"ap_fig_{name_suff}_{ctgr}.png"))
        plt.subplot(121)
        plt.cla()
        plt.subplot(122)
        plt.cla()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    uf.set_gpu_configs()
    eval_param = EvaluateNmsParams()
    eval_param.create_eval_file()
    # TODO ap check
    param_optimizer = FindBestParamByAP()
    param_optimizer.create_all_param_ap()
    # param_optimizer.draw_main([11, 12, 13], [0.3, 0.4])
    print("end optimizer")


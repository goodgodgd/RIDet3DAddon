import os
import os.path as op
import sys
import importlib
import zipfile
import shutil
import numpy as np


class CodeSnapshot:
    def __init__(self, ckpt_path, start_epoch):
        self.ckpt_path = ckpt_path
        self.start_epoch = start_epoch

    def __call__(self):
        if self.start_epoch == 0:
            self.snapshot_project()
            self.copy_config()

        self.compare_config()

    def snapshot_project(self):
        if not op.isdir(self.ckpt_path):
            os.makedirs(self.ckpt_path, exist_ok=True)
        work_dir = os.getcwd()
        if self.start_epoch == 0:
            py_zips = zipfile.ZipFile(op.join(self.ckpt_path, "py_files.zip"), "w")
            os.chdir(work_dir)
            for (path, dir, files) in os.walk(work_dir):
                for file in files:
                    if file.endswith(".py"):
                        py_zips.write(op.join(op.relpath(path, work_dir), file))
            py_zips.close()
            os.chdir(work_dir)

    def copy_config(self):
        # work_dir = os.getcwd()
        work_dir = os.getcwd().replace("/train", "").replace("/tflow", "")
        cfg_copy_file = op.join(self.ckpt_path, f"config_ep{self.start_epoch:02d}.py")
        cfg_orig_file = work_dir + "/config.py"
        shutil.copy(cfg_orig_file, cfg_copy_file)

    def compare_config(self):
        if self.ckpt_path not in sys.path:
            sys.path.append(self.ckpt_path)
        cur_config = self.load_config("config")
        latest_config = [files for files in os.listdir(self.ckpt_path) if files.endswith(".py")]
        latest_config.sort()
        old_config = self.load_config(f"{latest_config[-1][:-3]}")
        self.compare_config_impl(cur_config, old_config)

    def load_config(self, modname):
        config = importlib.import_module(modname)
        conf_classes = []
        for item in dir(config):
            item = eval("config." + item)
            if "'type'" in f"{type(item)}":
                conf_classes.append(item)
        return conf_classes

    def compare_config_impl(self, cur_config, old_config):
        for cur_cfg_class, old_cfg_class in zip(cur_config, old_config):
            try:
                self.compare_class(eval("cur_cfg_class"), eval("old_cfg_class"))
            except AttributeError as ae:
                print(ae)

    def compare_class(self, cur_class, old_class):
        for item in dir(cur_class):
            if item.startswith("__"):
                continue
            cur_item = eval("cur_class." + item)
            old_item = eval("old_class." + item)
            if "'type'" in f"{type(cur_item)}":
                self.compare_class(cur_item, old_item)
            elif isinstance(cur_item, dict):
                if cur_item.keys() != old_item.keys():
                    self.copy_config()
                    break
                for (cur_key, cur_val), (old_key, old_val) in zip(cur_item.items(), old_item.items()):
                    if cur_key != old_key:
                        self.copy_config()
                        break
                    if "'type'" not in f"{type(cur_val)}":
                        if cur_val != old_val:
                            self.copy_config()
                            break
            elif type(cur_item) == np.ndarray:
                if not np.array_equal(cur_item, old_item):
                    self.copy_config()
                    break
            else:
                if cur_item != old_item:
                    self.copy_config()
                    break


def test_code_snapshot():
    CodeSnapshot("/home/eagle/mun_workspace/ckpt/tttest", 30)()


if __name__ == "__main__":
    test_code_snapshot()




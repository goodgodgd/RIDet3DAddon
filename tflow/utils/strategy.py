import tensorflow as tf
import config as cfg


class DistributionStrategy:
    strategy = None

    @classmethod
    def get_strategy(cls):
        if (cls.strategy is None) and (cfg.Train.MODE == "distribute"):
            cls.strategy = tf.distribute.MirroredStrategy()
            # cls.strategy = tf.distribute.MirroredStrategy(["/gpu:0", "/gpu:1"])

            cfg.Train.GLOBAL_BATCH = cls.strategy.num_replicas_in_sync * cfg.Train.DATA_BATCH_SIZE
            print(f"[DistributionStrategy] number of devices: {cls.strategy.num_replicas_in_sync}")
            print(f"[DistributionStrategy] global batch size: {cfg.Train.GLOBAL_BATCH}")
        return cls.strategy


class StrategyScope:
    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        if cfg.Train.MODE == "distribute":
            print("[StrategyScope]", self.func.__name__)
            strategy = DistributionStrategy.get_strategy()
            with strategy.scope():
                return self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)


class StrategyDataset:
    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        assert args[4] in ["train", "val"]
        if (cfg.Train.MODE == "distribute") and (args[4] == "train"):
            print("[StrategyDataset]", self.func.__name__, *args)
            strategy = DistributionStrategy.get_strategy()
            dataset, steps, image_shape, anchors_per_scale = self.func(*args, **kwargs)
            dist_dataset = strategy.experimental_distribute_dataset(dataset)
            return dist_dataset, steps, image_shape, anchors_per_scale
        else:
            return self.func(*args, **kwargs)


import yaml


class Config:
    def __init__(self, cfg_path: str):
        with open(cfg_path) as f:
            self.cfg_dict = yaml.load(f, yaml.SafeLoader)

    def validate(self, fields):
        for field in fields:
            if field not in self.cfg_dict:
                raise ValueError(f"Missing field: {field}")

    def __getattr__(self, item):
        return self.cfg_dict[item]


class TrainConfig(Config):
    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)
        fields = [
            "data_dir", "out_dir", "unique_name", "batch_size", "num_workers",
            "model_name", "device", "epochs", "scheduler", "optimizer", "lr", "weight_decay"
        ]
        self.validate(fields)


class CompareConfig(Config):
    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)
        fields = [
            "data_dir", "model_name", "device", "weights_path_base", "weights_path_proposed",
            "num_workers", "confidence_threshold", "out_dir", "grid"
        ]
        self.validate(fields)


class EvaluateConfig(Config):
    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)
        fields = [
            "data_dir", "model_name", "device", "weights_path", "num_workers",
            "batch_size", "dataframe_path"
        ]
        self.validate(fields)

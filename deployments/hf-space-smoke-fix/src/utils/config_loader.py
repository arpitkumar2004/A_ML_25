import yaml

class ConfigLoader:
    @staticmethod
    def load(config_path: str) -> dict:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

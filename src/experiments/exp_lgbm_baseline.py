# quick experiment entrypoint
import yaml
from ..pipelines.train_pipeline import run_train_pipeline

def run():
    cfg = yaml.safe_load(open("configs/model/lgbm.yaml"))
    run_train_pipeline(cfg)

if __name__ == "__main__":
    run()
    
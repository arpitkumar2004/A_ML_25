#!/usr/bin/env python3
import argparse
import yaml
from src.pipelines.train_pipeline import run_train_pipeline
from src.pipelines.inference_pipeline import run_inference_pipeline

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'infer'])
    parser.add_argument('--config', '-c', default='configs/model/lgbm.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode == 'train':
        run_train_pipeline(cfg)
    else:
        run_inference_pipeline(cfg)

if __name__ == "__main__":
    main()

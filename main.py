"""Entry point for CLI: python main.py train --config configs/model/lgbm.yaml"""
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','inference'], help='mode')
    parser.add_argument('--config', type=str, help='config path')
    args = parser.parse_args()
    print('Mode:', args.mode, 'Config:', args.config)

if __name__ == '__main__':
    main()


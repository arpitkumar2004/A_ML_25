from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
root_dir_str = str(ROOT_DIR)

if root_dir_str not in sys.path:
    sys.path.insert(0, root_dir_str)
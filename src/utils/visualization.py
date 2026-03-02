# src/utils/visualization.py
from typing import Sequence, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_bar(values: Sequence[float], labels: Optional[Sequence[str]] = None, title: str = "", rotate_xticks: bool = False, save_path: Optional[str] = None):
    plt.figure(figsize=(8, 4))
    indices = np.arange(len(values))
    plt.bar(indices, values)
    if labels is not None:
        plt.xticks(indices, labels, rotation=45 if rotate_xticks else 0, ha='right')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_scatter(x, y, xlabel='x', ylabel='y', title=''):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=6, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


# for optimization of hyperparameters using Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from ..training.metrics import smape
from ..models.lgbm_model import LGBMModel
from ..models.catboost_model import CatBoostModel
from ..utils.seed_everything import seed_everything
from ..models.base_model import cross_validate_model
import warnings
warnings.filterwarnings("ignore")
import time
import os
import joblib
from ..utils.logging_utils import get_logger
logger = get_logger("hyperparam_opt")

def objective(trial, X, y, cfg):
    seed_everything(cfg['training']['seed'])
    model = LGBMModel(trial, cfg)
    return cross_validate_model(model, X, y, cfg)

def objective_catboost(trial, X, y, cfg):
    seed_everything(cfg['training']['seed'])
    model = CatBoostModel(trial, cfg)
    return cross_validate_model(model, X, y, cfg)

def optimize_hyperparameters(X, y, cfg, n_trials=50, direction='minimize', model_type='lgbm'):
    sampler = TPESampler(seed=cfg['training']['seed'])
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    if model_type == 'lgbm':
        func = lambda trial: objective(trial, X, y, cfg)
    elif model_type == 'catboost':
        func = lambda trial: objective_catboost(trial, X, y, cfg)
    else:
        raise ValueError("Unsupported model type for hyperparameter optimization.")
    study.optimize(func, n_trials=n_trials, timeout=None)
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.params}")
    # save study
    os.makedirs("experiments/hyperparam_studies", exist_ok=True)
    joblib.dump(study, f"experiments/hyperparam_studies/{model_type}_study.pkl")
    return study.best_trial.params
def load_study(filepath):
    return joblib.load(filepath)
def get_best_hyperparameters(study):
    return study.best_trial.params
def get_study_summary(study):
    return {
        'n_trials': len(study.trials),
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params
    }
    
def plot_optimization_history(study):
    from optuna.visualization import plot_optimization_history
    fig = plot_optimization_history(study)
    fig.show()
def plot_param_importances(study):
    from optuna.visualization import plot_param_importances
    fig = plot_param_importances(study)
    fig.show()
    
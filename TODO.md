# TODO: Fix Code Issues and Improve ML Pipeline for Price Prediction

## Overview
Fix critical errors, refactor redundant code, enhance data parsing, integrate embeddings, ensure model compatibility, and build a robust end-to-end pipeline for multimodal price prediction (text + images).

## Steps

### 1. Fix Critical Errors and Typos
- [ ] Rename `src/features/image_embedings.py` to `src/features/image_embeddings.py` (fix typo).
- [ ] Correct imports in `main.py`: Remove duplicates, fix paths (e.g., `from src.data.dataset_loader import load_train_df`).
- [ ] Fix `src/training/hyperparam_opt.py`: Update `objective` and `objective_catboost` to pass correct params to models.
- [ ] Fix `src/models/bert_regressor.py`: Correct `cross_validate_model` to pass texts directly to `fit`.
- [ ] Fix `src/models/fusion_nn.py`: Update `evaluate_model` call in `cross_validate_model`.
- [ ] Add missing imports (e.g., `smape` in `src/training/evaluator.py`).
- [ ] Remove example code from `src/data/parse_features.py`.
- [ ] Clean `src/models/__init__.py`: Remove duplicate code, only import classes.

### 2. Refactor Redundant Code
- [ ] Merge `run_*_pipeline` and `run_*_pipeline_full` in pipeline files into single functions with optional params.
- [ ] Extract common logic (data loading, feature building) into utility functions in `src/utils/`.
- [ ] Move duplicate functions (`cross_validate_model`, `evaluate_model`, `get_feature_importance`) to `src/models/base_model.py` or a shared module.
- [ ] Remove unused code (e.g., empty `src/utils/visualization.py`, placeholders in `src/data/augmentations.py`).

### 3. Enhance Data Parsing
- [ ] Improve `src/data/parse_features.py`: Refine regex and logic for extracting item_name, bullet_points, product_description, value, unit from catalog_content. Add error handling for malformed data.
- [ ] Integrate parsing into pipelines: Ensure `add_parsed_features` is called consistently.

### 4. Build Feature Engineering
- [ ] Update `src/features/build_features.py`: Integrate text embeddings (TF-IDF + BERT fallback from `text_embeddings.py`) and image embeddings (CLIP from `image_embeddings.py`).
- [ ] Add `load_tfidf` function to `src/features/text_embeddings.py` for inference.
- [ ] Ensure embeddings are combined properly (e.g., hstack for sparse matrices).

### 5. Model Integration and Fixes
- [ ] Ensure all models (LGBM, XGBoost, CatBoost, BERT, Fusion NN) handle embeddings correctly.
- [ ] Fix cross-validation in models to use proper seeds and metrics.
- [ ] Update hyperparameter optimization to work with all models.

### 6. Pipeline Construction
- [ ] Refactor `main.py`: Use pipelines as entry points, remove duplicate logic.
- [ ] Create end-to-end flow: Load -> Parse -> Feature Engineering -> Train (CV) -> Ensemble -> Save Models.
- [ ] Add inference pipeline: Load data -> Parse -> Features -> Load Models -> Predict -> Ensemble -> Output.
- [ ] Add config validation and error handling throughout.

### 7. Add Utilities and Best Practices
- [ ] Implement proper logging (use `src/utils/logging_utils.py` consistently).
- [ ] Add type hints and docstrings to key functions.
- [ ] Use `pathlib` for paths.
- [ ] Cache large models (e.g., BERT, CLIP) to avoid reloading.

### 8. Testing and Validation
- [ ] Add unit tests for parsing, embeddings, and basic model training.
- [ ] Test end-to-end pipeline on sample data.
- [ ] Validate outputs: Check embedding shapes, prediction accuracy, ensemble results.

### 9. Config and Documentation Updates
- [ ] Update YAML configs for consistency (e.g., add missing keys like `inference_path`).
- [ ] Update README.md with pipeline usage instructions.

## Followup
- Run pipeline on full dataset.
- Optimize for performance (e.g., batch processing for embeddings).
- Add monitoring/logging for production readiness.

# ---------------------------
# Flatten embedding columns
# ---------------------------
def flatten_embeddings(df, feat_cols):
    print("Flattening embedding columns...")
    def flatten_row(row):
        flattened = []
        for col in feat_cols:
            val = row[col]
            if isinstance(val, (list, np.ndarray)):
                flattened.extend(val)
            else:
                raise ValueError(f"Column {col} contains non-array values")
        return flattened
    X = df.apply(flatten_row, axis=1).to_list()
    X = np.array(X)
    print(f"Flattened features shape: {X.shape}")
    return X

# ---------------------------
# Train multiple models
# ---------------------------
def train_multiple_models(df, feat_cols, target_col, save_models=True):
    print("Starting model training pipeline...")

    X = flatten_embeddings(df, feat_cols)
    y = df[target_col].astype(float).values
    print(f"Target variable shape: {y.shape}")

    # Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Define models
    models = {
        # "Ridge": Ridge(alpha=1.0),
        "MLPRegressor": MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500, random_state=42),
        # "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42),
        # "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=-1, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining model: {name} ...")
        model.fit(X_train, y_train)
        print(f"{name} training completed.")

        y_pred = model.predict(X_test)
        score = smape(y_test, y_pred)
        print(f"{name} SMAPE on test set: {score:.4f}%")

        if save_models:
            model_file = f"{name}_price_model.pkl"
            joblib.dump(model, model_file)
            print(f"{name} model saved as {model_file}")

        results[name] = score

    print("All models trained and evaluated.")
    return results

# ---------------------------
# Example usage in Colab
# ---------------------------
df = merged_df
feat_cols = ['image_embedding', 'measure_embedding', 'item_embedding_vector']
target_col = 'price'

print("Starting model training pipeline...")

X = flatten_embeddings(df, feat_cols)
y = df[target_col].astype(float).values
print(f"Target variable shape: {y.shape}")

# Train-test split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Train all models
results = train_multiple_models(merged_df, feat_cols, target_col, save_models=True)

# Print SMAPE results
print("\nSMAPE results for all models:")
for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}%")

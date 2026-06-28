from __future__ import annotations

import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from scipy.stats import entropy, iqr, kurtosis, skew, variation
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text


PROJECT_ROOT = Path("/Users/samehissa/Downloads/XAI-Med-Images-Stat-Visual-Rules-main 2")
DATA_ROOT = Path("/Users/samehissa/Downloads/COVID-CT-Dataset-master")
RUN_ROOT = PROJECT_ROOT / "covid_ct_run" / "exact_sequence"
OUTPUT_DIR = RUN_ROOT / "outputs"
NOTEBOOK_DIR = RUN_ROOT / "notebooks"
MODEL_DIR = OUTPUT_DIR / "models"
FEATURE_DIR = OUTPUT_DIR / "features"
RULE_DIR = OUTPUT_DIR / "rules"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
AUGMENTED_DIR = OUTPUT_DIR / "augmented_train_images"
PLOT_DIR = OUTPUT_DIR / "plots"
MPLCONFIG_DIR = OUTPUT_DIR / "_mplconfig"

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["NonCOVID", "COVID"]
CLASS_TO_LABEL = {"NonCOVID": 0, "COVID": 1}
LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}


def setup_environment() -> None:
    for directory in [
        OUTPUT_DIR,
        NOTEBOOK_DIR,
        MODEL_DIR,
        FEATURE_DIR,
        RULE_DIR,
        HEATMAP_DIR,
        AUGMENTED_DIR,
        PLOT_DIR,
        MPLCONFIG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
    os.environ["KERAS_HOME"] = str(Path.home() / ".keras")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    random.seed(SEED)
    np.random.seed(SEED)


def read_split_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return [line.strip() for line in handle if line.strip()]


def build_official_split_table() -> pd.DataFrame:
    split_specs = [
        ("train", "COVID", DATA_ROOT / "Data-split/COVID/trainCT_COVID.txt", DATA_ROOT / "Images-processed/CT_COVID"),
        ("val", "COVID", DATA_ROOT / "Data-split/COVID/valCT_COVID.txt", DATA_ROOT / "Images-processed/CT_COVID"),
        ("test", "COVID", DATA_ROOT / "Data-split/COVID/testCT_COVID.txt", DATA_ROOT / "Images-processed/CT_COVID"),
        ("train", "NonCOVID", DATA_ROOT / "Data-split/NonCOVID/trainCT_NonCOVID.txt", DATA_ROOT / "Images-processed/CT_NonCOVID"),
        ("val", "NonCOVID", DATA_ROOT / "Data-split/NonCOVID/valCT_NonCOVID.txt", DATA_ROOT / "Images-processed/CT_NonCOVID"),
        ("test", "NonCOVID", DATA_ROOT / "Data-split/NonCOVID/testCT_NonCOVID.txt", DATA_ROOT / "Images-processed/CT_NonCOVID"),
    ]
    rows = []
    missing = []
    for split, class_name, split_file, image_dir in split_specs:
        for filename in read_split_file(split_file):
            image_path = image_dir / filename
            if not image_path.exists():
                missing.append(str(image_path))
                continue
            rows.append(
                {
                    "filepath": str(image_path),
                    "filename": filename,
                    "split": split,
                    "class_name": class_name,
                    "label": CLASS_TO_LABEL[class_name],
                    "source": "original",
                }
            )
    if missing:
        raise FileNotFoundError(f"{len(missing)} split files are missing. First: {missing[0]}")
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "covid_ct_official_splits.csv", index=False)
    return df


def load_official_split_table() -> pd.DataFrame:
    path = OUTPUT_DIR / "covid_ct_official_splits.csv"
    if path.exists():
        return pd.read_csv(path)
    return build_official_split_table()


def augment_image(image: Image.Image, rng: random.Random, index: int) -> Image.Image:
    image = image.convert("RGB")
    if rng.random() < 0.55:
        image = ImageOps.mirror(image)
    angle = rng.uniform(-7.0, 7.0)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    scale = rng.uniform(0.94, 1.06)
    width, height = image.size
    scaled = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
    if scaled.size[0] >= width and scaled.size[1] >= height:
        left = (scaled.size[0] - width) // 2
        top = (scaled.size[1] - height) // 2
        image = scaled.crop((left, top, left + width, top + height))
    else:
        canvas = Image.new("RGB", (width, height), 0)
        canvas.paste(scaled, ((width - scaled.size[0]) // 2, (height - scaled.size[1]) // 2))
        image = canvas
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.15))
    return image


def create_augmented_manifest() -> pd.DataFrame:
    setup_environment()
    splits = load_official_split_table()
    train = splits[splits["split"] == "train"].copy()
    val_test = splits[splits["split"].isin(["val", "test"])].copy()
    max_train_count = int(train.groupby("class_name").size().max())
    target_per_class = max_train_count * 2
    rng = random.Random(SEED)
    manifest_rows = []

    for _, row in train.iterrows():
        manifest_rows.append(row.to_dict())

    for class_name, class_df in train.groupby("class_name"):
        needed = target_per_class - len(class_df)
        class_out = AUGMENTED_DIR / class_name
        class_out.mkdir(parents=True, exist_ok=True)
        class_records = class_df.to_dict("records")
        for idx in range(needed):
            source = class_records[idx % len(class_records)]
            src_path = Path(source["filepath"])
            out_name = f"{src_path.stem}_aug_{idx:04d}.png"
            out_path = class_out / out_name
            if not out_path.exists():
                augmented = augment_image(Image.open(src_path), rng, idx)
                augmented.save(out_path)
            manifest_rows.append(
                {
                    "filepath": str(out_path),
                    "filename": out_name,
                    "split": "train",
                    "class_name": class_name,
                    "label": CLASS_TO_LABEL[class_name],
                    "source": "augmented",
                    "source_filepath": str(src_path),
                }
            )

    train_augmented = pd.DataFrame(manifest_rows)
    combined = pd.concat([train_augmented, val_test], ignore_index=True, sort=False)
    train_augmented.to_csv(OUTPUT_DIR / "01_augmented_train_manifest.csv", index=False)
    combined.to_csv(OUTPUT_DIR / "01_augmented_splits.csv", index=False)
    return combined


def load_augmented_splits() -> pd.DataFrame:
    path = OUTPUT_DIR / "01_augmented_splits.csv"
    if path.exists():
        return pd.read_csv(path)
    return create_augmented_manifest()


def load_image_batch(paths: list[str]) -> np.ndarray:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize(IMG_SIZE)
        images.append(np.asarray(image, dtype="float32"))
    return np.stack(images, axis=0)


def import_tensorflow():
    import tensorflow as tf

    tf.random.set_seed(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    return tf


def build_base_feature_model():
    setup_environment()
    tf = import_tensorflow()
    weights_path = Path.home() / ".keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Cached MobileNetV2 weights not found: {weights_path}")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None,
        pooling="avg",
        name="mobilenetv2_backbone",
    )
    base_model.load_weights(weights_path)
    base_model.trainable = False
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="ct_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    outputs = base_model(x, training=False)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_frozen_feature_extractor")


def extract_base_features(split_df: pd.DataFrame, output_path: Path) -> dict[str, np.ndarray]:
    feature_model = build_base_feature_model()
    arrays: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        df = split_df[split_df["split"] == split].reset_index(drop=True)
        features = []
        labels = []
        for start in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[start : start + BATCH_SIZE]
            images = load_image_batch(batch_df["filepath"].tolist())
            batch_features = feature_model.predict(images, verbose=0)
            features.append(batch_features)
            labels.append(batch_df["label"].values.astype("int32"))
        arrays[f"X_{split}"] = np.vstack(features).astype("float32")
        arrays[f"y_{split}"] = np.concatenate(labels).astype("int32")
        print(split, arrays[f"X_{split}"].shape, arrays[f"y_{split}"].shape)
    np.savez_compressed(output_path, **arrays)
    return arrays


def build_head_model(params: dict):
    tf = import_tensorflow()
    layers = tf.keras.layers
    dense_units = int(params.get("dense_units", 256))
    dropout = float(params.get("dropout", 0.5))
    learning_rate = float(params.get("learning_rate", 3e-4))
    inputs = tf.keras.Input(shape=(1280,), name="mobilenetv2_feature_vector")
    x = layers.Dense(dense_units, activation="relu", name="xai_dense_1")(inputs)
    x = layers.BatchNormalization(name="xai_bn_1")(x)
    x = layers.Dropout(dropout, seed=SEED, name="xai_dropout_1")(x)
    features = layers.Dense(128, activation="relu", name="xai_feature_layer")(x)
    x = layers.BatchNormalization(name="xai_bn_2")(features)
    x = layers.Dropout(dropout, seed=SEED + 1, name="xai_dropout_2")(x)
    outputs = layers.Dense(2, activation="softmax", name="classification")(x)
    model = tf.keras.Model(inputs, outputs, name="covid_ct_head")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def class_weight_from_labels(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(labels.astype("int32"), minlength=2)
    total = counts.sum()
    return {idx: float(total / (2 * max(counts[idx], 1))) for idx in range(2)}


def train_head_model(
    arrays: dict[str, np.ndarray],
    params: dict,
    epochs: int = 25,
    patience: int = 5,
    verbose: int = 0,
):
    tf = import_tensorflow()
    model = build_head_model(params)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
    ]
    history = model.fit(
        arrays["X_train"],
        arrays["y_train"],
        validation_data=(arrays["X_val"], arrays["y_val"]),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight_from_labels(arrays["y_train"]),
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history


def evaluate_probabilities(name: str, y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = np.argmax(probs, axis=1)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs[:, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }
    return metrics


def run_dl_grid_search() -> tuple[pd.DataFrame, dict]:
    split_df = load_augmented_splits()
    feature_path = FEATURE_DIR / "02_base_features_augmented.npz"
    if feature_path.exists():
        arrays = dict(np.load(feature_path))
    else:
        arrays = extract_base_features(split_df, feature_path)

    param_grid = [
        {"learning_rate": lr, "dense_units": units, "dropout": dropout}
        for lr in [1e-3, 3e-4]
        for units in [128, 256]
        for dropout in [0.3, 0.5]
    ]
    rows = []
    best_score = -math.inf
    best_params = None
    for params in param_grid:
        started = time.time()
        model, history = train_head_model(arrays, params, epochs=20, patience=4, verbose=0)
        val_probs = model.predict(arrays["X_val"], verbose=0)
        val_metrics = evaluate_probabilities("candidate", arrays["y_val"], val_probs)
        row = {
            **params,
            "epochs_run": len(history.history.get("loss", [])),
            "seconds": time.time() - started,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        rows.append(row)
        print(row)
        if row["val_balanced_accuracy"] > best_score:
            best_score = row["val_balanced_accuracy"]
            best_params = params

    results = pd.DataFrame(rows).sort_values("val_balanced_accuracy", ascending=False)
    best_params = dict(best_params or results.iloc[0][["learning_rate", "dense_units", "dropout"]])
    results.to_csv(OUTPUT_DIR / "02_dl_grid_search_results.csv", index=False)
    with (OUTPUT_DIR / "02_best_dl_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(best_params, handle, indent=2)
    return results, best_params


def build_full_model_from_head(head_model, params: dict):
    tf = import_tensorflow()
    weights_path = Path.home() / ".keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None,
        pooling="avg",
        name="mobilenetv2_backbone",
    )
    base_model.load_weights(weights_path)
    base_model.trainable = False

    dense_units = int(params.get("dense_units", 256))
    dropout = float(params.get("dropout", 0.5))
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="ct_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dense(dense_units, activation="relu", name="xai_dense_1")(x)
    x = layers.BatchNormalization(name="xai_bn_1")(x)
    x = layers.Dropout(dropout, seed=SEED, name="xai_dropout_1")(x)
    features = layers.Dense(128, activation="relu", name="xai_feature_layer")(x)
    x = layers.BatchNormalization(name="xai_bn_2")(features)
    x = layers.Dropout(dropout, seed=SEED + 1, name="xai_dropout_2")(x)
    outputs = layers.Dense(2, activation="softmax", name="classification")(x)
    full_model = tf.keras.Model(inputs, outputs, name="custom_mobilenetv2_covid_ct")
    for layer_name in ["xai_dense_1", "xai_bn_1", "xai_feature_layer", "xai_bn_2", "classification"]:
        full_model.get_layer(layer_name).set_weights(head_model.get_layer(layer_name).get_weights())
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(float(params.get("learning_rate", 3e-4))),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return full_model


def train_final_custom_mobilenet() -> dict:
    arrays = dict(np.load(FEATURE_DIR / "02_base_features_augmented.npz"))
    with (OUTPUT_DIR / "02_best_dl_hyperparameters.json").open("r", encoding="utf-8") as handle:
        best_params = json.load(handle)
    head_model, history = train_head_model(arrays, best_params, epochs=35, patience=6, verbose=2)
    probs = head_model.predict(arrays["X_test"], verbose=0)
    metrics = evaluate_probabilities("custom_mobilenetv2_complete", arrays["y_test"], probs)
    metrics["best_hyperparameters"] = best_params
    metrics["epochs_run"] = len(history.history.get("loss", []))

    head_model.save(MODEL_DIR / "03_custom_mobilenetv2_head.keras")
    full_model = build_full_model_from_head(head_model, best_params)
    full_model.save(MODEL_DIR / "03_custom_mobilenetv2_complete_covid_ct.keras")

    pd.DataFrame(history.history).to_csv(OUTPUT_DIR / "03_custom_mobilenetv2_training_history.csv", index=False)
    with (OUTPUT_DIR / "03_custom_mobilenetv2_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def extract_xai_features_from_full_model(split_df: pd.DataFrame, model_path: Path, output_path: Path) -> dict[str, np.ndarray]:
    tf = import_tensorflow()
    model = tf.keras.models.load_model(model_path)
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("xai_feature_layer").output)
    arrays = {}
    for split in ["train", "val", "test"]:
        df = split_df[split_df["split"] == split].reset_index(drop=True)
        features = []
        labels = []
        for start in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[start : start + BATCH_SIZE]
            images = load_image_batch(batch_df["filepath"].tolist())
            features.append(feature_model.predict(images, verbose=0))
            labels.append(batch_df["label"].values.astype("int32"))
        arrays[f"X_{split}"] = np.vstack(features).astype("float32")
        arrays[f"y_{split}"] = np.concatenate(labels).astype("int32")
        print(split, arrays[f"X_{split}"].shape)
    np.savez_compressed(output_path, **arrays)
    return arrays


def signal_to_noise(values: np.ndarray) -> float:
    return float(np.mean(values) / (np.std(values) + 1e-6))


def safe_autocorr(values: np.ndarray) -> float:
    if len(values) < 2 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return 0.0
    return float(np.nan_to_num(np.corrcoef(values[:-1], values[1:])[0, 1], nan=0.0, posinf=0.0, neginf=0.0))


def calculate_statistical_features(features: np.ndarray) -> pd.DataFrame:
    rows = []
    for vector in features:
        f = vector.astype("float64")
        positive = np.abs(f) + 1e-6
        probability = positive / positive.sum()
        row = {
            "mean": np.mean(f),
            "std_dev": np.std(f),
            "variance": np.var(f),
            "median": np.median(f),
            "range": np.ptp(f),
            "skewness": skew(f),
            "kurtosis": kurtosis(f),
            "entropy": entropy(positive),
            "energy": np.sum(f**2),
            "contrast": np.var(f),
            "mean_abs_dev": np.mean(np.abs(f - np.mean(f))),
            "min_value": np.min(f),
            "max_value": np.max(f),
            "iqr": iqr(f),
            "percentile_25": np.percentile(f, 25),
            "percentile_50": np.percentile(f, 50),
            "percentile_75": np.percentile(f, 75),
            "signal_to_noise": signal_to_noise(f),
            "coef_of_var": variation(f),
            "autocorrelation": safe_autocorr(f),
            "shannon_entropy": -np.sum(probability * np.log2(probability + 1e-12)),
            "root_mean_square": np.sqrt(np.mean(f**2)),
            "harmonic_mean": len(positive) / np.sum(1.0 / positive),
            "geometric_mean": np.exp(np.mean(np.log(positive))),
            "std_error_mean": np.std(f) / np.sqrt(len(f)),
            "median_abs_dev": np.median(np.abs(f - np.median(f))),
        }
        rows.append({key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key, value in row.items()})
    return pd.DataFrame(rows)


def run_statistical_features_zfmis() -> dict:
    split_df = load_augmented_splits()
    xai_path = FEATURE_DIR / "04_xai_deep_features_custom_mobilenetv2.npz"
    if xai_path.exists():
        arrays = dict(np.load(xai_path))
    else:
        arrays = extract_xai_features_from_full_model(
            split_df,
            MODEL_DIR / "03_custom_mobilenetv2_complete_covid_ct.keras",
            xai_path,
        )

    feature_sets = {}
    stats_frames = {}
    for split in ["train", "val", "test"]:
        stats = calculate_statistical_features(arrays[f"X_{split}"])
        stats["label"] = arrays[f"y_{split}"]
        stats_frames[split] = stats
        stats.to_csv(FEATURE_DIR / f"04_{split}_statistical_features.csv", index=False)

    feature_columns = [col for col in stats_frames["train"].columns if col != "label"]
    zero_fraction = (stats_frames["train"][feature_columns] == 0).mean().sort_values(ascending=False)
    zero_fraction.to_csv(FEATURE_DIR / "04_statistical_feature_zero_fraction.csv", header=["zero_fraction"])
    filtered_columns = zero_fraction[zero_fraction <= 0.50].index.tolist() or feature_columns
    feature_sets["all"] = feature_columns
    feature_sets["filtered"] = filtered_columns

    X_train = stats_frames["train"][filtered_columns]
    y_train = stats_frames["train"]["label"].values
    for k in [18, 15, 12, 9, 6, 3]:
        actual_k = min(k, len(filtered_columns))
        selector = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=SEED), k=actual_k)
        selector.fit(X_train, y_train)
        selected = list(np.array(filtered_columns)[selector.get_support()])
        feature_sets[f"selected_{k}"] = selected
        out_dir = FEATURE_DIR / f"04_selected_{k}_features"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, stats in stats_frames.items():
            selected_df = stats[selected + ["label"]]
            selected_df.to_csv(out_dir / f"{split}_selected_{k}_features.csv", index=False)

    with (FEATURE_DIR / "04_zfmis_feature_sets.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_sets, handle, indent=2)
    return feature_sets


def load_stat_frames(prefix: str = "04") -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_csv(FEATURE_DIR / f"{prefix}_{split}_statistical_features.csv")
        for split in ["train", "val", "test"]
    }


def evaluate_tree_grid(feature_set_names: list[str] | None = None) -> pd.DataFrame:
    frames = load_stat_frames()
    with (FEATURE_DIR / "04_zfmis_feature_sets.json").open("r", encoding="utf-8") as handle:
        feature_sets = json.load(handle)
    if feature_set_names is None:
        feature_set_names = ["all", "filtered", "selected_18", "selected_15", "selected_12", "selected_9", "selected_6", "selected_3"]
    rows = []
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 8, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
    }
    for name in feature_set_names:
        columns = feature_sets[name]
        X_train = frames["train"][columns].values
        y_train = frames["train"]["label"].values
        X_val = frames["val"][columns].values
        y_val = frames["val"]["label"].values
        X_test = frames["test"][columns].values
        y_test = frames["test"]["label"].values
        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=SEED),
            param_grid=param_grid,
            cv=5,
            scoring="balanced_accuracy",
            n_jobs=1,
        )
        grid.fit(X_train, y_train)
        clf = DecisionTreeClassifier(random_state=SEED, **grid.best_params_)
        clf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        metrics = evaluate_probabilities(
            f"decision_tree_{name}", y_test, np.column_stack([1 - probs, probs])
        )
        model_path = MODEL_DIR / f"05a_decision_tree_{name}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(clf, handle)
        rules_path = RULE_DIR / f"05a_decision_tree_{name}_rules.txt"
        rules_path.write_text(export_text(clf, feature_names=columns), encoding="utf-8")
        row = {
            "feature_set": name,
            "n_features": len(columns),
            "best_params": json.dumps(grid.best_params_),
            "cv_balanced_accuracy": float(grid.best_score_),
            "test_accuracy": metrics["accuracy"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_f1_weighted": metrics["f1_weighted"],
            "test_roc_auc": metrics["roc_auc"],
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            "model_path": str(model_path),
            "rules_path": str(rules_path),
        }
        rows.append(row)
        print(row)
    results = pd.DataFrame(rows).sort_values("test_balanced_accuracy", ascending=False)
    results.to_csv(OUTPUT_DIR / "05a_gridsearchfortree_results.csv", index=False)
    return results


def evaluate_rulefit_grid(feature_set_names: list[str] | None = None) -> pd.DataFrame:
    from rulefit import RuleFit

    frames = load_stat_frames()
    with (FEATURE_DIR / "04_zfmis_feature_sets.json").open("r", encoding="utf-8") as handle:
        feature_sets = json.load(handle)
    if feature_set_names is None:
        feature_set_names = ["selected_3"]
    param_grid = [
        {"tree_size": tree_size, "sample_fract": sample_fract, "max_rules": max_rules}
        for tree_size in [3, 4]
        for sample_fract in [0.8]
        for max_rules in [50, 100]
    ]
    rows = []
    for name in feature_set_names:
        columns = feature_sets[name]
        X_train = frames["train"][columns].values
        y_train = frames["train"]["label"].values
        X_val = frames["val"][columns].values
        y_val = frames["val"]["label"].values
        X_test = frames["test"][columns].values
        y_test = frames["test"]["label"].values
        best = None
        for params in param_grid:
            started = time.time()
            rf = RuleFit(rfmode="classify", model_type="r", random_state=SEED, **params)
            rf.fit(X_train, y_train, feature_names=columns)
            val_pred = rf.predict(X_val).astype("int32")
            val_score = balanced_accuracy_score(y_val, val_pred)
            candidate = {**params, "val_balanced_accuracy": float(val_score), "seconds": time.time() - started}
            if best is None or candidate["val_balanced_accuracy"] > best["val_balanced_accuracy"]:
                best = candidate
        final_rf = RuleFit(
            rfmode="classify",
            model_type="r",
            tree_size=int(best["tree_size"]),
            sample_fract=float(best["sample_fract"]),
            max_rules=int(best["max_rules"]),
            random_state=SEED,
        )
        final_rf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]), feature_names=columns)
        test_pred = final_rf.predict(X_test).astype("int32")
        rules = final_rf.get_rules()
        rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
        rules_path = RULE_DIR / f"05b_rulefit_{name}_rules.csv"
        rules.to_csv(rules_path, index=False)
        row = {
            "feature_set": name,
            "n_features": len(columns),
            "best_params": json.dumps({k: best[k] for k in ["tree_size", "sample_fract", "max_rules"]}),
            "val_balanced_accuracy": best["val_balanced_accuracy"],
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
            "test_f1_weighted": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
            "confusion_matrix": json.dumps(confusion_matrix(y_test, test_pred).tolist()),
            "n_nonzero_rules": int(len(rules)),
            "rules_path": str(rules_path),
        }
        rows.append(row)
        print(row)
    results = pd.DataFrame(rows).sort_values("test_balanced_accuracy", ascending=False)
    results.to_csv(OUTPUT_DIR / "05b_rulefitgridsearchcode_results.csv", index=False)
    return results


def extract_final_rules() -> dict:
    tree_results = pd.read_csv(OUTPUT_DIR / "05a_gridsearchfortree_results.csv")
    rulefit_results = pd.read_csv(OUTPUT_DIR / "05b_rulefitgridsearchcode_results.csv")
    best_tree = tree_results.sort_values("test_balanced_accuracy", ascending=False).iloc[0].to_dict()
    best_rulefit = rulefit_results.sort_values("test_balanced_accuracy", ascending=False).iloc[0].to_dict()
    tree_rules = Path(best_tree["rules_path"]).read_text(encoding="utf-8")
    tree_out = RULE_DIR / "06_best_decision_tree_rules.txt"
    tree_out.write_text(tree_rules, encoding="utf-8")
    rulefit_rules = pd.read_csv(best_rulefit["rules_path"])
    rulefit_out = RULE_DIR / "06_best_rulefit_rules.csv"
    rulefit_rules.to_csv(rulefit_out, index=False)
    summary = {
        "best_decision_tree": best_tree,
        "best_rulefit": best_rulefit,
        "decision_tree_rules": str(tree_out),
        "rulefit_rules": str(rulefit_out),
    }
    with (OUTPUT_DIR / "06_rule_extraction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    return (values - values.min()) / (values.max() - values.min() + 1e-8)


def make_heatmap_overlay(raw_rgb: np.ndarray, heatmap: np.ndarray) -> Image.Image:
    heatmap_u8 = (normalize_map(heatmap) * 255).astype("uint8")
    heatmap_resized = cv2.resize(heatmap_u8, IMG_SIZE)
    color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(raw_rgb.astype("uint8"), 0.58, color, 0.42, 0)
    return Image.fromarray(blended)


def run_sfmov_heatmaps(samples_per_class: int = 4) -> list[str]:
    tf = import_tensorflow()
    model = tf.keras.models.load_model(MODEL_DIR / "03_custom_mobilenetv2_complete_covid_ct.keras")
    backbone = model.get_layer("mobilenetv2_backbone")
    conv_model = tf.keras.Model(inputs=backbone.input, outputs=backbone.get_layer("Conv_1_bn").output)
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("xai_feature_layer").output)
    splits = load_official_split_table()
    test_df = splits[splits["split"] == "test"].reset_index(drop=True)
    sample_df = test_df.groupby("class_name", group_keys=False).sample(n=samples_per_class, random_state=SEED)
    saved = []
    for _, row in sample_df.iterrows():
        raw = np.asarray(Image.open(row["filepath"]).convert("RGB").resize(IMG_SIZE), dtype="float32")
        batch = np.expand_dims(raw, axis=0)
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(batch.copy())
        conv = conv_model.predict(preprocessed, verbose=0)[0]
        dense = feature_model.predict(batch, verbose=0)[0]
        mean_map = np.mean(conv, axis=-1)
        skewness_map = np.zeros(conv.shape[:2], dtype="float64")
        entropy_map = np.zeros(conv.shape[:2], dtype="float64")
        for channel in range(conv.shape[-1]):
            fmap = conv[:, :, channel]
            skewness_map += float(np.nan_to_num(skew(fmap.ravel()), nan=0.0)) * fmap
            entropy_map += float(entropy(np.abs(fmap.ravel()) + 1e-6)) * fmap
        dense_mean = float(np.mean(dense))
        dense_skewness = abs(float(np.nan_to_num(skew(dense), nan=0.0)))
        dense_entropy = float(entropy(np.abs(dense) + 1e-6))
        combined = normalize_map(dense_mean * mean_map + dense_skewness * skewness_map + dense_entropy * entropy_map)
        maps = {
            "mean": mean_map,
            "skewness": skewness_map,
            "entropy": entropy_map,
            "combined": combined,
        }
        safe_stem = Path(row["filename"]).stem.replace(" ", "_").replace("/", "_")
        for name, heatmap in maps.items():
            out_path = HEATMAP_DIR / f"{row['class_name']}_{safe_stem}_{name}.png"
            make_heatmap_overlay(raw, heatmap).save(out_path)
            saved.append(str(out_path))
    with (OUTPUT_DIR / "07_sfmov_heatmap_files.json").open("w", encoding="utf-8") as handle:
        json.dump(saved, handle, indent=2)
    return saved


def final_sequence_summary() -> pd.DataFrame:
    rows = []
    custom_metrics = json.load(open(OUTPUT_DIR / "03_custom_mobilenetv2_test_metrics.json", encoding="utf-8"))
    rows.append(
        {
            "stage": "custom_mobilenetv2_complete",
            "accuracy": custom_metrics["accuracy"],
            "balanced_accuracy": custom_metrics["balanced_accuracy"],
            "f1_weighted": custom_metrics["f1_weighted"],
            "roc_auc": custom_metrics["roc_auc"],
        }
    )
    if (OUTPUT_DIR / "05a_gridsearchfortree_results.csv").exists():
        tree = pd.read_csv(OUTPUT_DIR / "05a_gridsearchfortree_results.csv").iloc[0]
        rows.append(
            {
                "stage": f"best_decision_tree_{tree['feature_set']}",
                "accuracy": tree["test_accuracy"],
                "balanced_accuracy": tree["test_balanced_accuracy"],
                "f1_weighted": tree["test_f1_weighted"],
                "roc_auc": tree["test_roc_auc"],
            }
        )
    if (OUTPUT_DIR / "05b_rulefitgridsearchcode_results.csv").exists():
        rulefit = pd.read_csv(OUTPUT_DIR / "05b_rulefitgridsearchcode_results.csv").iloc[0]
        rows.append(
            {
                "stage": f"best_rulefit_{rulefit['feature_set']}",
                "accuracy": rulefit["test_accuracy"],
                "balanced_accuracy": rulefit["test_balanced_accuracy"],
                "f1_weighted": rulefit["test_f1_weighted"],
                "roc_auc": None,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "final_exact_sequence_metrics_summary.csv", index=False)
    return summary

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from .common import set_deterministic_seed, write_json


def concept_score_names(concept_columns: list[str]) -> list[str]:
    return [
        column.removeprefix("concept_").removesuffix("_label")
        for column in concept_columns
    ]


def med_feature_names(concept_columns: list[str]) -> list[str]:
    return [f"concept_score_{name}" for name in concept_score_names(concept_columns)] + [
        "med_task_prob_covid",
        "med_neural_prob_covid",
    ]


class Trainer(Protocol):
    def fit(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame | None,
        concept_columns: list[str],
        output_dir: Path,
        model_id: str,
        fixed_epochs: int | None = None,
    ) -> dict[str, Any]: ...

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame: ...


class ConstantAwareLogistic:
    def __init__(self, seed: int):
        self.seed = seed
        self.constant: float | None = None
        self.model: LogisticRegression | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ConstantAwareLogistic":
        unique = np.unique(y)
        if len(unique) == 1:
            self.constant = float(unique[0])
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=self.seed, class_weight="balanced")
            self.model.fit(x, y)
        return self

    def predict_probability(self, x: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(x), self.constant, dtype=float)
        assert self.model is not None
        classes = list(self.model.classes_)
        if 1 not in classes:
            return np.zeros(len(x), dtype=float)
        return self.model.predict_proba(x)[:, classes.index(1)]


def _smoke_image_features(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for path in frame["image_path"]:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L").resize((16, 16)), dtype=float) / 255.0
        histogram, _ = np.histogram(array, bins=8, range=(0.0, 1.0), density=True)
        rows.append(
            np.concatenate(
                [
                    [array.mean(), array.std(), np.quantile(array, 0.25), np.median(array), np.quantile(array, 0.75)],
                    histogram,
                    array.mean(axis=0)[::4],
                    array.mean(axis=1)[::4],
                ]
            )
        )
    return np.asarray(rows, dtype=float)


class SmokeMedTrainer:
    """Fast workflow verifier. Its outputs are not scientific Med-MICN results."""

    def __init__(self, config: dict, seed: int):
        self.config = config
        self.seed = seed
        self.concept_models: list[ConstantAwareLogistic] = []
        self.task_model = ConstantAwareLogistic(seed)
        self.neural_model = ConstantAwareLogistic(seed + 100)
        self.concept_columns: list[str] = []
        self.model_id = ""

    def fit(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame | None,
        concept_columns: list[str],
        output_dir: Path,
        model_id: str,
        fixed_epochs: int | None = None,
    ) -> dict[str, Any]:
        set_deterministic_seed(self.seed)
        output_dir.mkdir(parents=True, exist_ok=False)
        self.concept_columns = list(concept_columns)
        self.model_id = model_id
        x_train = _smoke_image_features(train_frame)
        y_train = train_frame["true_label"].to_numpy(dtype=int)
        self.concept_models = []
        for index, column in enumerate(concept_columns):
            model = ConstantAwareLogistic(self.seed + index).fit(x_train, train_frame[column].to_numpy(dtype=int))
            self.concept_models.append(model)
        concept_train = np.column_stack([model.predict_probability(x_train) for model in self.concept_models])
        self.task_model.fit(x_train, y_train)
        self.neural_model.fit(concept_train, y_train)
        checkpoint = output_dir / "smoke_model.joblib"
        joblib.dump(self, checkpoint)
        provenance = train_frame[["image_id", "image_path", "original_split", "group_id"]].copy()
        provenance["role"] = "fit"
        if validation_frame is not None and len(validation_frame):
            validation = validation_frame[["image_id", "image_path", "original_split", "group_id"]].copy()
            validation["role"] = "inner_validation"
            provenance = pd.concat([provenance, validation], ignore_index=True)
        provenance.to_csv(output_dir / "training_provenance.csv", index=False)
        metadata = {
            "backend": "smoke",
            "scientific_result": False,
            "model_id": model_id,
            "checkpoint": str(checkpoint),
            "training_rows": int(len(train_frame)),
            "validation_rows": int(0 if validation_frame is None else len(validation_frame)),
            "feature_order": med_feature_names(concept_columns),
            "fixed_epochs": fixed_epochs,
            "config": self.config,
        }
        write_json(output_dir / "checkpoint_metadata.json", metadata)
        return metadata

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = _smoke_image_features(frame)
        concept_probabilities = np.column_stack(
            [model.predict_probability(x) for model in self.concept_models]
        )
        task_probability = self.task_model.predict_probability(x)
        neural_probability = self.neural_model.predict_probability(concept_probabilities)
        result = pd.DataFrame(
            concept_probabilities,
            columns=[f"concept_score_{name}" for name in concept_score_names(self.concept_columns)],
        )
        result["med_task_prob_covid"] = task_probability
        result["med_neural_prob_covid"] = neural_probability
        return result


class TorchMedMICNTrainer:
    """Real Med-MICN trainer, imported lazily so audits work without PyTorch."""

    def __init__(self, config: dict, seed: int, med_source_root: Path):
        self.config = copy.deepcopy(config)
        self.seed = seed
        self.med_source_root = med_source_root.resolve()
        self.model = None
        self.device = None
        self.concept_columns: list[str] = []
        self.model_id = ""

    @staticmethod
    def dependency_status() -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, "available"

    def _imports(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms

        med_root = str(self.med_source_root)
        if med_root not in sys.path:
            sys.path.insert(0, med_root)
        import torch_explain as te
        from models import Neural_Concat_Model
        from torch_explain.nn.concepts import ConceptReasoningLayer

        return torch, nn, functional, DataLoader, Dataset, models, transforms, te, Neural_Concat_Model, ConceptReasoningLayer

    def _device(self, torch):
        requested = str(self.config.get("device", "auto"))
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _transforms(self, transforms):
        preprocessing = self.config["preprocessing"]
        normalization = preprocessing["normalization"]
        train_steps = [transforms.Resize(int(preprocessing["resize_shorter_side"]))]
        augmentation = self.config.get("training_augmentation", {})
        if augmentation.get("enabled", False):
            train_steps.append(
                transforms.RandomAffine(
                    degrees=float(augmentation.get("rotation_degrees", 0.0)),
                    translate=tuple(augmentation.get("translate_fraction", [0.0, 0.0])),
                    scale=tuple(augmentation.get("scale_range", [1.0, 1.0])),
                    fill=0,
                )
            )
        train_steps.extend(
            [
                transforms.CenterCrop(int(preprocessing["crop_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalization["mean"], std=normalization["std"]),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(int(preprocessing["resize_shorter_side"])),
                transforms.CenterCrop(int(preprocessing["crop_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalization["mean"], std=normalization["std"]),
            ]
        )
        return transforms.Compose(train_steps), eval_transform

    def _build_model(self, models, nn, te, Neural_Concat_Model, ConceptReasoningLayer, n_concepts: int):
        backbone_name = str(self.config["model"]["backbone"]).lower()
        pretrained = bool(self.config["model"].get("pretrained", True))
        if backbone_name in {"rn50", "resnet50"}:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            concept_hidden, fused_hidden = 10, 1000
        elif backbone_name in {"densenet", "densenet169"}:
            weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
            backbone = models.densenet169(weights=weights)
            concept_hidden, fused_hidden = 128, 64
        elif backbone_name in {"vgg", "vgg16"}:
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            backbone = models.vgg16(weights=weights)
            concept_hidden, fused_hidden = 10, 1000
        else:
            raise ValueError(f"unsupported backbone: {backbone_name}")
        embedding_size = int(self.config["model"].get("embedding_size", 8))
        concept_encoder = nn.Sequential(
            nn.Linear(1000, concept_hidden),
            nn.LeakyReLU(),
            te.nn.ConceptEmbedding(concept_hidden, n_concepts, embedding_size),
        )
        neural = ConceptReasoningLayer(embedding_size, 2)
        task = nn.Sequential(
            nn.Linear(n_concepts * embedding_size + 1000, fused_hidden),
            nn.LeakyReLU(),
            nn.Dropout(float(self.config["model"].get("dropout", 0.2))),
            nn.Linear(fused_hidden, 2),
        )
        concept_task = nn.Linear(n_concepts * embedding_size, 2)
        return Neural_Concat_Model(backbone, concept_encoder, neural, task, concept_task)

    def _set_backbone_mode(self, model, mode: str) -> None:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "final_block":
            candidates = []
            for name in ("layer4", "fc", "classifier"):
                module = getattr(model.backbone, name, None)
                if module is not None:
                    candidates.append(module)
            features = getattr(model.backbone, "features", None)
            if features is not None:
                children = list(features.children())
                if children:
                    candidates.append(children[-1])
            if not candidates:
                raise ValueError("could not identify the final backbone block")
            for module in candidates:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        elif mode not in {"frozen", "full"}:
            raise ValueError(f"unsupported fine_tune mode: {mode}")

    def _parameter_counts(self, model) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        backbone_trainable = sum(
            parameter.numel() for parameter in model.backbone.parameters() if parameter.requires_grad
        )
        return {
            "total": int(total),
            "trainable": int(trainable),
            "frozen": int(total - trainable),
            "backbone_trainable": int(backbone_trainable),
            "head_trainable": int(trainable - backbone_trainable),
        }

    def _optimizer(self, torch, model):
        head_parameters = []
        backbone_parameters = []
        backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            (backbone_parameters if id(parameter) in backbone_ids else head_parameters).append(parameter)
        groups = []
        if backbone_parameters:
            groups.append({"params": backbone_parameters, "lr": float(self.config["optimization"]["backbone_learning_rate"])})
        if head_parameters:
            groups.append({"params": head_parameters, "lr": float(self.config["optimization"]["head_learning_rate"])})
        return torch.optim.AdamW(groups, weight_decay=float(self.config["optimization"].get("weight_decay", 0.0)))

    def fit(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame | None,
        concept_columns: list[str],
        output_dir: Path,
        model_id: str,
        fixed_epochs: int | None = None,
    ) -> dict[str, Any]:
        available, reason = self.dependency_status()
        if not available:
            raise RuntimeError(f"real Med-MICN backend unavailable: {reason}")
        (
            torch,
            nn,
            functional,
            DataLoader,
            Dataset,
            models,
            transforms,
            te,
            Neural_Concat_Model,
            ConceptReasoningLayer,
        ) = self._imports()
        set_deterministic_seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        output_dir.mkdir(parents=True, exist_ok=False)
        self.concept_columns = list(concept_columns)
        self.model_id = model_id
        self.device = self._device(torch)
        train_transform, eval_transform = self._transforms(transforms)

        class FrameDataset(Dataset):
            def __init__(self, frame: pd.DataFrame, transform):
                self.frame = frame.reset_index(drop=True)
                self.transform = transform

            def __len__(self):
                return len(self.frame)

            def __getitem__(self, index):
                row = self.frame.iloc[index]
                with Image.open(row["image_path"]) as image:
                    tensor = self.transform(image.convert("RGB"))
                return tensor, int(row["true_label"]), row[concept_columns].to_numpy(dtype=np.float32)

        batch_size = int(self.config["optimization"].get("batch_size", 16))
        workers = int(self.config["optimization"].get("num_workers", 2))
        if workers < 0:
            raise ValueError("optimization.num_workers must be non-negative")
        loader_options = {"num_workers": workers, "pin_memory": self.device.type == "cuda"}
        train_loader = DataLoader(
            FrameDataset(train_frame, train_transform),
            batch_size=batch_size,
            shuffle=True,
            **loader_options,
        )
        validation_loader = None
        if validation_frame is not None and len(validation_frame):
            validation_loader = DataLoader(
                FrameDataset(validation_frame, eval_transform),
                batch_size=batch_size,
                shuffle=False,
                **loader_options,
            )

        model = self._build_model(models, nn, te, Neural_Concat_Model, ConceptReasoningLayer, len(concept_columns)).to(self.device)
        phases = self.config["fine_tuning"].get("phases")
        if fixed_epochs is not None:
            phases = [{"mode": self.config["fine_tuning"].get("final_mode", "frozen"), "epochs": fixed_epochs}]
        elif not phases:
            phases = [{"mode": self.config["fine_tuning"]["mode"], "epochs": int(self.config["optimization"]["max_epochs"])}]
        task_loss_fn = nn.CrossEntropyLoss()
        concept_loss_fn = nn.BCELoss()
        neural_loss_fn = nn.BCELoss()
        loss_weights = self.config["loss_weights"]
        history = []
        best_state = None
        best_score = float("-inf")
        best_epoch = 0
        global_epoch = 0
        patience = int(self.config["optimization"].get("early_stopping_patience", 5))
        stale = 0

        def evaluate(loader):
            model.eval()
            labels_all, task_all, task_probs_all, neural_all, losses = [], [], [], [], []
            with torch.no_grad():
                for images, labels, concepts in loader:
                    images, labels, concepts = images.to(self.device), labels.to(self.device), concepts.to(self.device)
                    _, task_logits, neural_probs, _, concept_probs, _ = model(images)
                    task_loss = task_loss_fn(task_logits, labels)
                    concept_loss = concept_loss_fn(concept_probs, concepts)
                    neural_target = functional.one_hot(labels, num_classes=2).float()
                    neural_loss = neural_loss_fn(neural_probs, neural_target)
                    loss = task_loss + float(loss_weights["concept"]) * concept_loss + float(loss_weights["neural"]) * neural_loss
                    probabilities = torch.softmax(task_logits, dim=1)
                    labels_all.extend(labels.cpu().numpy())
                    task_all.extend(probabilities.argmax(1).cpu().numpy())
                    task_probs_all.extend(probabilities[:, 1].cpu().numpy())
                    neural_all.extend(neural_probs.argmax(1).cpu().numpy())
                    losses.append(float(loss.cpu()))
            return {
                "loss": float(np.mean(losses)),
                "balanced_accuracy": float(balanced_accuracy_score(labels_all, task_all)),
                "task_f1_macro": float(f1_score(labels_all, task_all, average="macro", zero_division=0)),
                "neural_balanced_accuracy": float(balanced_accuracy_score(labels_all, neural_all)),
                "task_log_loss": float(log_loss(labels_all, np.clip(task_probs_all, 1e-7, 1 - 1e-7), labels=[0, 1])),
            }

        for phase_index, phase in enumerate(phases):
            phase_mode = str(phase["mode"])
            self._set_backbone_mode(model, phase_mode)
            optimizer = self._optimizer(torch, model)
            stale = 0
            for _ in range(int(phase["epochs"])):
                global_epoch += 1
                model.train()
                if phase_mode in {"frozen", "final_block"}:
                    model.backbone.eval()
                    if phase_mode == "final_block":
                        for name in ("layer4", "fc", "classifier"):
                            module = getattr(model.backbone, name, None)
                            if module is not None:
                                module.train()
                train_losses = []
                for images, labels, concepts in train_loader:
                    images, labels, concepts = images.to(self.device), labels.to(self.device), concepts.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    _, task_logits, neural_probs, _, concept_probs, _ = model(images)
                    task_loss = task_loss_fn(task_logits, labels)
                    concept_loss = concept_loss_fn(concept_probs, concepts)
                    neural_target = functional.one_hot(labels, num_classes=2).float()
                    neural_loss = neural_loss_fn(neural_probs, neural_target)
                    loss = task_loss + float(loss_weights["concept"]) * concept_loss + float(loss_weights["neural"]) * neural_loss
                    loss.backward()
                    optimizer.step()
                    train_losses.append(float(loss.detach().cpu()))
                metrics = evaluate(validation_loader) if validation_loader is not None else {"loss": float(np.mean(train_losses)), "balanced_accuracy": float("nan"), "task_f1_macro": float("nan"), "neural_balanced_accuracy": float("nan"), "task_log_loss": float("nan")}
                history.append({"epoch": global_epoch, "phase": phase_index, "mode": phase["mode"], "train_loss": float(np.mean(train_losses)), **{f"val_{key}": value for key, value in metrics.items()}})
                score = metrics["balanced_accuracy"] if validation_loader is not None else -metrics["loss"]
                if score > best_score:
                    best_score, best_epoch, stale = score, global_epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                if validation_loader is not None and stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        counts = self._parameter_counts(model)
        checkpoint_path = output_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": self.config,
                "concept_cols": concept_columns,
                "feature_order": med_feature_names(concept_columns),
                "model_id": model_id,
                "best_epoch": best_epoch,
                "parameter_counts": counts,
            },
            checkpoint_path,
        )
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        provenance = train_frame[["image_id", "image_path", "original_split", "group_id"]].copy()
        provenance["role"] = "fit"
        if validation_frame is not None:
            validation = validation_frame[["image_id", "image_path", "original_split", "group_id"]].copy()
            validation["role"] = "inner_validation"
            provenance = pd.concat([provenance, validation], ignore_index=True)
        provenance.to_csv(output_dir / "training_provenance.csv", index=False)
        metadata = {
            "backend": "med_micn",
            "scientific_result": True,
            "model_id": model_id,
            "checkpoint": str(checkpoint_path),
            "training_rows": int(len(train_frame)),
            "validation_rows": int(0 if validation_frame is None else len(validation_frame)),
            "feature_order": med_feature_names(concept_columns),
            "parameter_counts": counts,
            "best_epoch": best_epoch,
            "device": str(self.device),
            "config": self.config,
        }
        write_json(output_dir / "checkpoint_metadata.json", metadata)
        return metadata

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("trainer has not been fitted")
        torch, _, _, DataLoader, Dataset, _, transforms, _, _, _ = self._imports()
        _, eval_transform = self._transforms(transforms)
        concept_columns = self.concept_columns
        device = self.device

        class PredictionDataset(Dataset):
            def __init__(self, rows):
                self.rows = rows.reset_index(drop=True)

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                with Image.open(self.rows.iloc[index]["image_path"]) as image:
                    return eval_transform(image.convert("RGB"))

        loader = DataLoader(PredictionDataset(frame), batch_size=int(self.config["optimization"].get("batch_size", 16)), shuffle=False, num_workers=0)
        concepts_all, task_all, neural_all = [], [], []
        self.model.eval()
        with torch.no_grad():
            for images in loader:
                _, task_logits, neural_probs, _, concept_probs, _ = self.model(images.to(device))
                concepts_all.append(concept_probs.cpu().numpy())
                task_all.append(torch.softmax(task_logits, dim=1)[:, 1].cpu().numpy())
                neural_all.append(neural_probs[:, 1].cpu().numpy())
        result = pd.DataFrame(np.vstack(concepts_all), columns=[f"concept_score_{name}" for name in concept_score_names(concept_columns)])
        result["med_task_prob_covid"] = np.concatenate(task_all)
        result["med_neural_prob_covid"] = np.concatenate(neural_all)
        return result


def trainer_for_backend(backend: str, config: dict, seed: int, med_source_root: Path) -> Trainer:
    if backend == "smoke":
        return SmokeMedTrainer(config, seed)
    if backend == "med_micn":
        return TorchMedMICNTrainer(config, seed, med_source_root)
    raise ValueError(f"unknown backend: {backend}")


def configuration_id(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]

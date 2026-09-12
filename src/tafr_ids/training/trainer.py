"""Deterministic continual trainer for Naive, Uniform Replay, and TAFR variants."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tafr_ids.data.inspection import sha256
from tafr_ids.data.loader import CLASSES, CLASS_TO_INDEX
from tafr_ids.evaluation.continual import build_matrices, continual_summary
from tafr_ids.evaluation.metrics import classification_metrics
from tafr_ids.evaluation.reporting import write_json_atomic
from tafr_ids.models.mlp import TabularMLP
from tafr_ids.training.checkpoint import atomic_save, load_checkpoint
from tafr_ids.training.quota import allocate_quotas
from tafr_ids.training.replay_buffer import ReplayBuffer, selected_ids_fingerprint
from tafr_ids.training.reproducibility import capture_rng_state, data_loader_generator, restore_rng_state
from tafr_ids.training.signals import forgetting_scores, normalize_signal, priorities_for_method, rarity_scores
from tafr_ids.training.strategies.naive import NaiveStrategy
from tafr_ids.training.strategies.tafr import TAFRStrategy
from tafr_ids.training.strategies.uniform import UniformReplayStrategy

EXPERIENCES = ("E1", "E2", "E3", "E4")
METHODS = ("naive", "uniform", "tafr_f", "tafr_fu", "tafr")


@dataclass(frozen=True)
class PreparedData:
    arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]
    metadata: dict[str, Any]
    fingerprint: str
    input_dim: int


def canonical_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def validate_prepared_data(prepared_dir: Path, config: dict, repository: Path) -> PreparedData:
    """Validate the complete training-only artifact boundary before loading arrays."""
    directory = prepared_dir.resolve()
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Prepared metadata.json is missing; run preparation explicitly")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared_files = metadata.get("files", {})
    expected_names = {
        f"{experience}_{partition}_{suffix}.npy"
        for experience in EXPERIENCES
        for partition in ("development", "validation")
        for suffix in ("X", "y", "indices")
    } | {"assignments.csv", "preprocessor.joblib", "preprocessor_fit_indices.npy"}
    if set(declared_files) != expected_names:
        raise ValueError("Prepared artifact manifest has missing, extra, or logical-test files")
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_names != expected_names | {"metadata.json"}:
        raise ValueError("Prepared directory has unmanifested, missing, or logical-test files")
    for name, expected_hash in declared_files.items():
        if sha256(directory / name) != expected_hash:
            raise ValueError(f"Prepared artifact fingerprint mismatch: {name}")

    split_path = (repository / config["data"]["split_manifest"]).resolve()
    experience_path = (repository / config["data"]["experience_config"]).resolve()
    if sha256(split_path) != metadata.get("split_manifest_sha256"):
        raise ValueError("Dataset split-manifest fingerprint does not match prepared data")
    if sha256(experience_path) != metadata.get("experience_config_sha256"):
        raise ValueError("Experience protocol fingerprint does not match prepared data")
    split_config = _load_toml(split_path)
    experience_config = _load_toml(experience_path)
    protocol = metadata.get("protocol", {})
    if split_config["training"]["sha256"] != protocol.get("logical_training_sha256"):
        raise ValueError("Logical-training dataset fingerprint does not match preparation protocol")
    if experience_config.get("split_assignment_fingerprint") != declared_files.get("assignments.csv"):
        raise ValueError("Assignment fingerprint does not match locked experience protocol")
    if experience_config.get("global_class_mapping") != dict(CLASS_TO_INDEX):
        raise ValueError("Tracked global class mapping differs from package mapping")
    if protocol.get("global_class_mapping") != dict(CLASS_TO_INDEX):
        raise ValueError("Preparation global class mapping differs from locked mapping")
    if metadata.get("logical_test") != "Integrity verification only; no targets profiled or arrays prepared":
        raise ValueError("Prepared metadata does not attest the logical-test boundary")

    arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    all_indices: list[np.ndarray] = []
    dimensions: set[int] = set()
    for experience in EXPERIENCES:
        arrays[experience] = {}
        for partition in ("development", "validation"):
            prefix = f"{experience}_{partition}"
            features = np.load(directory / f"{prefix}_X.npy", allow_pickle=False)
            labels = np.load(directory / f"{prefix}_y.npy", allow_pickle=False)
            indices = np.load(directory / f"{prefix}_indices.npy", allow_pickle=False)
            if features.dtype != np.float32 or features.ndim != 2 or not np.isfinite(features).all():
                raise ValueError(f"{prefix} features must be a finite float32 matrix")
            if labels.dtype != np.int64 or labels.ndim != 1:
                raise ValueError(f"{prefix} labels must be an int64 vector")
            if indices.dtype.kind not in "iu" or indices.ndim != 1:
                raise ValueError(f"{prefix} identifiers must be an integer vector")
            if len(features) != len(labels) or len(labels) != len(indices):
                raise ValueError(f"{prefix} arrays have inconsistent row counts")
            if labels.size == 0 or labels.min() < 0 or labels.max() >= len(CLASSES):
                raise ValueError(f"{prefix} has invalid global class IDs")
            if len(np.unique(indices)) != len(indices):
                raise ValueError(f"{prefix} has duplicate sample identifiers")
            summary = metadata.get("subsets", {}).get(prefix)
            if summary != {"rows": len(features), "features": features.shape[1]}:
                raise ValueError(f"{prefix} shape differs from preparation manifest")
            dimensions.add(features.shape[1])
            all_indices.append(indices)
            arrays[experience][partition] = (features, labels, indices.astype(np.int64, copy=False))
    joined = np.concatenate(all_indices)
    if len(np.unique(joined)) != len(joined):
        raise ValueError("Experience sample identifiers are not mutually disjoint")
    if len(dimensions) != 1:
        raise ValueError("Prepared feature dimension changes across experiences")
    input_dim = dimensions.pop()
    if input_dim != config["data"]["expected_input_dim"] or input_dim != len(metadata.get("feature_names", [])):
        raise ValueError("Prepared input dimension does not match locked configuration")
    return PreparedData(arrays, metadata, sha256(metadata_path), input_dim)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def environment_record(repository: Path, device: torch.device) -> dict:
    def git(*arguments: str) -> str:
        return subprocess.run(("git", *arguments), cwd=repository, check=True, capture_output=True, text=True).stdout.strip()

    cpu = platform.processor() or None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_state": "dirty" if git("status", "--porcelain") else "clean",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "cpu": cpu or platform.machine() or None,
        "cpu_count": os.cpu_count(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }


def _loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, generator=None) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)), batch_size=batch_size,
        shuffle=shuffle, num_workers=0, generator=generator,
    )


@torch.inference_mode()
def evaluate(model: nn.Module, datasets: list[tuple[np.ndarray, np.ndarray]], device: torch.device, batch_size: int) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    loss_sum = 0.0
    total = 0
    for features, labels in datasets:
        for batch_features, batch_labels in _loader(features, labels, batch_size, False):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            logits = model(batch_features)
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite logits")
            loss_sum += float(criterion(logits, batch_labels).item())
            total += len(batch_labels)
            targets.append(batch_labels.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return classification_metrics(np.concatenate(targets), np.concatenate(predictions), loss_sum / total, CLASSES)


@torch.inference_mode()
def _memory_recall(model: nn.Module, memory: ReplayBuffer, device: torch.device, batch_size: int) -> dict[int, float]:
    if not len(memory.sample_ids):
        return {}
    model.eval()
    predictions = []
    for features, _ in _loader(memory.features, memory.labels, batch_size, False):
        logits = model(features.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits in replay recall")
        predictions.append(logits.argmax(1).cpu().numpy())
    predicted = np.concatenate(predictions)
    return {
        int(class_id): float(np.mean(predicted[memory.labels == class_id] == class_id))
        for class_id in np.unique(memory.labels)
    }


@torch.inference_mode()
def _candidate_uncertainty(
    model: nn.Module, candidates: ReplayBuffer, device: torch.device, batch_size: int
) -> dict[int, float]:
    model.eval()
    uncertainty = []
    for features, _ in _loader(candidates.features, candidates.labels, batch_size, False):
        logits = model(features.to(device))
        probabilities = torch.softmax(logits, dim=1)
        values = 1.0 - probabilities.max(dim=1).values
        if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all() or not torch.isfinite(values).all():
            raise ValueError("Non-finite uncertainty calculation")
        uncertainty.append(values.cpu().numpy())
    values = np.concatenate(uncertainty)
    return {
        int(class_id): float(values[candidates.labels == class_id].mean())
        for class_id in np.unique(candidates.labels)
    }


def _state(
    *, model, optimizer, completed_experience, current_experience, completed_epoch, config,
    config_fingerprint, prepared_fingerprint, history, runtime, buffer, cumulative_counts,
    best_recall, discarded_ids, counted_through, replay_sampler_state,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "completed_experience": completed_experience,
        "current_experience": current_experience,
        "completed_epoch": completed_epoch,
        "rng_states": capture_rng_state(),
        "config": config,
        "config_fingerprint": config_fingerprint,
        "prepared_data_fingerprint": prepared_fingerprint,
        "metric_history": history,
        "runtime": runtime,
        "replay_buffer": buffer.checkpoint_state(),
        "cumulative_class_counts": cumulative_counts,
        "best_replay_recall": best_recall,
        "discarded_historical_ids": np.asarray(sorted(discarded_ids), dtype=np.int64),
        "counted_through_experience": counted_through,
        "replay_sampler_state": replay_sampler_state,
    }


def _update_memory(
    *, method: str, model: nn.Module, previous: ReplayBuffer, current: tuple[np.ndarray, np.ndarray, np.ndarray],
    experience_number: int, cumulative_counts: dict[int, int], best_recall: dict[int, float],
    discarded_ids: set[int], capacity: int, seed: int, version: int, device: torch.device, batch_size: int,
) -> tuple[ReplayBuffer, dict, dict[int, float], set[int]]:
    features, labels, sample_ids = current
    candidates = previous.candidates(features, labels, sample_ids, experience_number)
    if discarded_ids.intersection(map(int, candidates.sample_ids)):
        raise ValueError("A permanently discarded historical sample reappeared")
    seen_classes = sorted(map(int, np.unique(candidates.labels)))
    current_recall = _memory_recall(model, previous, device, batch_size)
    raw_f = forgetting_scores(seen_classes, best_recall, current_recall)
    raw_u = _candidate_uncertainty(model, candidates, device, batch_size)
    raw_r = rarity_scores(seen_classes, cumulative_counts)
    norm_f, norm_u, norm_r = normalize_signal(raw_f), normalize_signal(raw_u), normalize_signal(raw_r)
    priorities, fallback = priorities_for_method(method, norm_f, norm_u, norm_r)
    candidate_counts = {class_id: int(np.sum(candidates.labels == class_id)) for class_id in seen_classes}
    quotas, allocation = allocate_quotas(candidate_counts, priorities, capacity)
    selected = candidates.select(quotas, seed, f"E{experience_number}", version)
    selected.validate(capacity)
    target = min(capacity, len(candidates.sample_ids))
    if len(selected.sample_ids) != target or set(map(int, selected.sample_ids)) - set(map(int, candidates.sample_ids)):
        raise ValueError("Replay selection violates target or candidate eligibility")
    newly_discarded = set(map(int, candidates.sample_ids)) - set(map(int, selected.sample_ids))
    discarded_ids = discarded_ids | newly_discarded
    selected_recall = _memory_recall(model, selected, device, batch_size)
    updated_best = dict(best_recall)
    for class_id, recall in selected_recall.items():
        updated_best[class_id] = max(float(updated_best.get(class_id, recall)), recall)
    allocation.update({
        "experience": f"E{experience_number}",
        "buffer_update_version": version,
        "raw_signals": {"forgetting": raw_f, "uncertainty": raw_u, "rarity": raw_r},
        "normalized_signals": {"forgetting": norm_f, "uncertainty": norm_u, "rarity": norm_r},
        "uniform_fallback": fallback,
        "previous_buffer_recall": current_recall,
        "best_replay_recall_after_update": updated_best,
        "selected_buffer_recall": selected_recall,
        "selected_ids_fingerprint": selected_ids_fingerprint(selected.sample_ids),
        "buffer_size": len(selected.sample_ids),
        "buffer_capacity": capacity,
        "unique_ids": len(np.unique(selected.sample_ids)) == len(selected.sample_ids),
        "eligible_only": True,
        "source_experience_counts": {
            f"E{source}": int(np.sum(selected.source_experiences == source))
            for source in sorted(np.unique(selected.source_experiences))
        },
        "discarded_historical_count": len(discarded_ids),
        "forgetting_definition": "Fixed-memory replay-proxy recall; discarded samples are never recovered.",
    })
    return selected, allocation, updated_best, discarded_ids


def run_experiment(
    *, config: dict, prepared: PreparedData, output_dir: Path, repository: Path, resume: bool = False
) -> dict:
    method = config["method"]
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    training = config["training"]
    replay = config.get("replay")
    if method == "naive" and replay is not None:
        raise ValueError("Naive must not define replay")
    if method != "naive" and replay is None:
        raise ValueError("Replay method is missing its locked replay configuration")
    device = resolve_device(config["device"])
    model = TabularMLP(prepared.input_dim, len(CLASSES), config["model"]["hidden_dims"], config["model"]["dropout"]).to(device)
    config_fingerprint = canonical_fingerprint(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir, latest = output_dir / "checkpoints", output_dir / "checkpoints" / "latest.pt"
    history: dict[str, Any] = {"evaluations": {}, "seen_unions": {}, "epochs": [], "replay_updates": {}}
    runtime: dict[str, Any] = {"prior_elapsed_seconds": 0.0, "experiences": {}}
    capacity = int(replay["capacity"]) if replay else 0
    buffer = ReplayBuffer.empty(prepared.input_dim)
    cumulative_counts: dict[int, int] = {}
    best_recall: dict[int, float] = {}
    discarded_ids: set[int] = set()
    counted_through = 0
    replay_sampler_state = None
    completed_experience = resume_experience = resume_epoch = 0
    resume_optimizer_state = None
    if resume:
        if not latest.is_file():
            raise ValueError("--resume requested but checkpoints/latest.pt does not exist")
        state = load_checkpoint(
            latest, config_fingerprint=config_fingerprint, prepared_data_fingerprint=prepared.fingerprint,
            map_location=device,
        )
        if state.get("config") != config:
            raise ValueError("Checkpoint complete configuration does not match")
        model.load_state_dict(state["model"])
        completed_experience = int(state["completed_experience"])
        resume_experience = int(state["current_experience"])
        resume_epoch = int(state["completed_epoch"])
        resume_optimizer_state = state["optimizer"]
        history, runtime = state["metric_history"], state["runtime"]
        buffer = ReplayBuffer.from_checkpoint(state["replay_buffer"], capacity or 0) if method != "naive" else ReplayBuffer.empty(prepared.input_dim)
        cumulative_counts = {int(k): int(v) for k, v in state["cumulative_class_counts"].items()}
        best_recall = {int(k): float(v) for k, v in state["best_replay_recall"].items()}
        discarded_ids = set(map(int, state["discarded_historical_ids"]))
        counted_through = int(state["counted_through_experience"])
        replay_sampler_state = state["replay_sampler_state"]
        restore_rng_state(state["rng_states"])
        if completed_experience == len(EXPERIENCES) and (output_dir / "results.json").is_file():
            existing = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            if existing.get("status") != "completed" or existing.get("config_fingerprint") != config_fingerprint or existing.get("prepared_data_fingerprint") != prepared.fingerprint:
                raise ValueError("Completed results do not match the validated checkpoint")
            return existing
    elif any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory or use --resume")

    batch_size = training["batch_size"]
    run_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if "random_init" not in history["evaluations"]:
        history["evaluations"]["random_init"] = {
            experience: evaluate(model, [prepared.arrays[experience]["validation"][:2]], device, batch_size)
            for experience in EXPERIENCES
        }

    optimizer = None
    checkpoint_size = latest.stat().st_size if latest.is_file() else 0
    for number, experience in enumerate(EXPERIENCES, start=1):
        if number <= completed_experience:
            continue
        features, labels, ids = prepared.arrays[experience]["development"]
        if counted_through < number:
            for class_id, count in zip(*np.unique(labels, return_counts=True), strict=True):
                cumulative_counts[int(class_id)] = cumulative_counts.get(int(class_id), 0) + int(count)
            counted_through = number
        optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
        start_epoch = 1
        if resume_experience == number and resume_epoch > 0:
            if resume_optimizer_state is None:
                raise ValueError("Mid-experience checkpoint is missing optimizer state")
            optimizer.load_state_dict(resume_optimizer_state)
            start_epoch = resume_epoch + 1
        strategy: NaiveStrategy
        if method == "uniform":
            strategy = UniformReplayStrategy(optimizer, training["gradient_clip_norm"])
        elif method.startswith("tafr"):
            strategy = TAFRStrategy(optimizer, training["gradient_clip_norm"])
        else:
            strategy = NaiveStrategy(optimizer, training["gradient_clip_norm"])
        experience_started = time.perf_counter()
        previous_exp_runtime = float(runtime["experiences"].get(experience, {}).get("seconds", 0.0))
        for epoch in range(start_epoch, training["epochs_per_experience"] + 1):
            epoch_started = time.perf_counter()
            if number == 1 or method == "naive":
                batches = strategy.training_batches(
                    _loader(features, labels, batch_size, True, data_loader_generator(config["seed"], number, epoch))
                )
                exposure = {"current": len(labels), "replay": 0, "stored_buffer": len(buffer.sample_ids)}
            else:
                mix_seed = config["seed"] + number * 100_000 + epoch
                batches = strategy.mixed_batches(features, labels, buffer, seed=mix_seed)
                expected_replay = sum(round(len(labels[start : start + 192]) * 0.25 / 0.75) for start in range(0, len(labels), 192))
                exposure = {"current": len(labels), "replay": expected_replay, "stored_buffer": len(buffer.sample_ids)}
                replay_sampler_state = {"algorithm": "NumPy PCG64", "deterministic_seed": mix_seed, "epoch_completed": epoch}
            model.train()
            loss_sum = 0.0
            sample_count = 0
            try:
                for batch_features, batch_labels in batches:
                    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                    loss = nn.functional.cross_entropy(model(batch_features), batch_labels)
                    strategy.observe(loss, model)
                    loss_sum += float(loss.item()) * len(batch_labels)
                    sample_count += len(batch_labels)
            except RuntimeError as error:
                if "determin" in str(error).lower():
                    raise RuntimeError(f"Deterministic training operation failed: {error}") from error
                raise
            epoch_seconds = time.perf_counter() - epoch_started
            history["epochs"].append({
                "experience": experience, "epoch": epoch, "training_loss": loss_sum / sample_count,
                "runtime_seconds": epoch_seconds, "exposures": exposure,
            })
            runtime["experiences"][experience] = {
                "seconds": previous_exp_runtime + time.perf_counter() - experience_started,
                "epoch_seconds": [item["runtime_seconds"] for item in history["epochs"] if item["experience"] == experience],
            }
            runtime["prior_elapsed_seconds"] += time.perf_counter() - run_started
            run_started = time.perf_counter()
            checkpoint = _state(
                model=model, optimizer=optimizer, completed_experience=completed_experience,
                current_experience=number, completed_epoch=epoch, config=config,
                config_fingerprint=config_fingerprint, prepared_fingerprint=prepared.fingerprint,
                history=history, runtime=runtime, buffer=buffer, cumulative_counts=cumulative_counts,
                best_recall=best_recall, discarded_ids=discarded_ids, counted_through=counted_through,
                replay_sampler_state=replay_sampler_state,
            )
            atomic_save(checkpoint, latest)
            atomic_save(checkpoint, checkpoint_dir / f"{experience}_epoch_{epoch:02d}.pt")

        row_name = f"after_{experience}"
        history["evaluations"][row_name] = {
            validation_experience: evaluate(model, [prepared.arrays[validation_experience]["validation"][:2]], device, batch_size)
            for validation_experience in EXPERIENCES
        }
        history["seen_unions"][row_name] = evaluate(
            model, [prepared.arrays[item]["validation"][:2] for item in EXPERIENCES[:number]], device, batch_size,
        )
        if method != "naive":
            buffer, audit, best_recall, discarded_ids = _update_memory(
                method=method, model=model, previous=buffer, current=(features, labels, ids),
                experience_number=number, cumulative_counts=cumulative_counts, best_recall=best_recall,
                discarded_ids=discarded_ids, capacity=capacity, seed=config["seed"],
                version=int(replay["buffer_update_version"]), device=device, batch_size=batch_size,
            )
            history["replay_updates"][experience] = audit
        runtime["experiences"][experience]["training_seconds"] = runtime["experiences"][experience]["seconds"]
        runtime["experiences"][experience]["seconds"] = previous_exp_runtime + time.perf_counter() - experience_started
        completed_experience = number
        runtime["prior_elapsed_seconds"] += time.perf_counter() - run_started
        run_started = time.perf_counter()
        checkpoint = _state(
            model=model, optimizer=optimizer, completed_experience=completed_experience,
            current_experience=number, completed_epoch=training["epochs_per_experience"], config=config,
            config_fingerprint=config_fingerprint, prepared_fingerprint=prepared.fingerprint,
            history=history, runtime=runtime, buffer=buffer, cumulative_counts=cumulative_counts,
            best_recall=best_recall, discarded_ids=discarded_ids, counted_through=counted_through,
            replay_sampler_state=replay_sampler_state,
        )
        checkpoint_size = atomic_save(checkpoint, latest)
        atomic_save(checkpoint, checkpoint_dir / f"after_{experience}.pt")
        resume_experience = resume_epoch = 0
        resume_optimizer_state = None

    matrices = build_matrices(history["evaluations"], EXPERIENCES)
    summary = continual_summary(matrices)
    summary["final_seen_class_macro_f1"] = history["seen_unions"]["after_E4"]["macro_f1"]
    summary["per_task_forgetting_by_experience"] = dict(zip(EXPERIENCES, summary["per_task_forgetting"], strict=True))
    summary["accuracy_per_task_forgetting_by_experience"] = dict(zip(EXPERIENCES, summary["accuracy_per_task_forgetting"], strict=True))
    cpu_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resource_report = {
        "total_runtime_seconds": runtime["prior_elapsed_seconds"],
        "runtime_per_experience": runtime["experiences"],
        "peak_cpu_resident_memory_bytes": int(cpu_peak * 1024) if sys.platform.startswith("linux") else None,
        "peak_cpu_measurement_note": "ru_maxrss process-lifetime peak" if sys.platform.startswith("linux") else "unsupported platform",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "cuda_measurement_note": None if device.type == "cuda" else "CUDA not used",
        "parameter_count": model.parameter_count,
        "checkpoint_size_bytes": checkpoint_size,
    }
    report = {
        "status": "completed", "method": method, "seed": config["seed"], "config": config,
        "config_fingerprint": config_fingerprint, "prepared_data_fingerprint": prepared.fingerprint,
        "prepared_dataset_fingerprint": prepared.metadata["protocol"]["logical_training_sha256"],
        "prepared_protocol_fingerprints": {
            "split_manifest_sha256": prepared.metadata["split_manifest_sha256"],
            "experience_config_sha256": prepared.metadata["experience_config_sha256"],
            "assignment_sha256": prepared.metadata["files"]["assignments.csv"],
        },
        "environment": environment_record(repository, device), "history": history, "matrices": matrices,
        "matrix_axes": {"rows": ["random_init", "after_E1", "after_E2", "after_E3", "after_E4"], "columns": list(EXPERIENCES)},
        "continual_metrics": summary, "resources": resource_report,
        "protocol_notes": {
            "logical_test_evaluated": False,
            "model_selection": "Final epoch; validation never changes training.",
            "normal_samples": "Task validation subsets contain disjoint Normal samples and different introduced attack classes.",
            "replay_proxy": "Forgetting allocation uses only retained training-memory recall, never validation or test data." if method != "naive" else None,
        },
    }
    write_json_atomic(output_dir / "results.json", report)
    return report


def run_naive(*, config: dict, prepared: PreparedData, output_dir: Path, repository: Path, resume: bool = False) -> dict:
    if config.get("method") != "naive":
        raise ValueError("run_naive requires the naive method")
    return run_experiment(config=config, prepared=prepared, output_dir=output_dir, repository=repository, resume=resume)

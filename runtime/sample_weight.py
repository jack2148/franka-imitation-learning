"""Direct operator labels, independent of BIW stage expansion and loss weights.

Owned exclusively by the collection worker's control loop. HTTP threads enqueue
commands; neither HTTP nor camera processes read or mutate this scalar.
"""
from __future__ import annotations

import csv
from pathlib import Path
import uuid

import numpy as np


def check_weight(value):
    if type(value) is not int or value not in (1, 2, 3):
        raise ValueError("weight must be the integer 1, 2 or 3")
    return value


class SampleWeight:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_weight = 1
        self.recording_id = uuid.uuid4().hex
        self.request_id = None
        self.values = []

    def set(self, value, recording_id, request_id):
        check_weight(value)
        # A delayed request from a previous episode must never label a new one.
        if recording_id != self.recording_id:
            return False
        self.current_weight = value
        self.request_id = request_id
        return True

    def append(self, frame_index, *, value=None):
        if frame_index != len(self.values):
            raise ValueError("weight must be captured once per contiguous recorder sample")
        self.values.append(self.current_weight if value is None else check_weight(value))

    def snapshot(self):
        return dict(current_weight=self.current_weight, recording_id=self.recording_id,
                    request_id=self.request_id, frames=len(self.values))

    def metadata(self):
        return dict(schema_version="operator_weight_v1", values=[1, 2, 3], default=1,
                    reset="every_recording_start", canonical="samples_20hz.npz:weight",
                    csv="samples_20hz.csv", timing="observation_t_before_step -> action_t",
                    policy_input=False, applied_to_loss=False)

    def arrays(self, count):
        if count != len(self.values):
            raise ValueError("weight/sample count mismatch")
        return {"weight": np.asarray(self.values, dtype=np.int64)}


def validate_weight(samples, count, metadata=None):
    """Missing labels stay missing for legacy data; never synthesize defaults."""
    if "weight" not in samples:
        if (metadata or {}).get("sample_weight"):
            raise ValueError("declared sample_weight is missing from canonical samples")
        return None
    value = np.asarray(samples["weight"])
    if value.shape != (count,) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("weight must be an integer array with shape (N,)")
    if not np.isin(value, (1, 2, 3)).all():
        raise ValueError("weight contains values outside {1,2,3}")
    return value


def csv_columns(samples):
    """Lossless numeric NPZ projection in existing field order, weight last."""
    columns = []
    for key in samples:
        if key == "weight":
            continue
        value = np.asarray(samples[key])
        if value.ndim == 1:
            columns.append((key, key, ()))
        else:
            columns.extend((key + "_" + "_".join(map(str, index)), key, index)
                           for index in np.ndindex(value.shape[1:]))
    return columns + [("weight", "weight", ())]


def save_weighted_samples(directory, samples, annotation):
    count = len(samples["action"])
    samples.update(annotation.arrays(count))
    validate_weight(samples, count)
    np.savez(Path(directory) / "samples_20hz.npz", **samples)
    columns = csv_columns(samples)
    with (Path(directory) / "samples_20hz.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([name for name, _, _ in columns])
        writer.writerows([samples[key][(i,) + index].item() for _, key, index in columns]
                         for i in range(count))


def validate_weight_csv(directory, samples):
    """Strict index/timestamp/label alignment, with no timestamp interpolation."""
    weights = validate_weight(samples, len(samples["action"]))
    expected = [name for name, _, _ in csv_columns(samples)]
    with (Path(directory) / "samples_20hz.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected or expected.count("weight") != 1:
            raise ValueError("weight CSV columns mismatch; weight must be rightmost")
        count = 0
        for index, row in enumerate(reader):
            if (weights is None or index >= len(weights) or None in row
                    or row["weight"] != str(int(weights[index]))
                    or int(row["dataset_frame_index"]) != index
                    or float(row["sim_time"]) != float(samples["sim_time"][index])):
                raise ValueError("weight CSV is not index-aligned with canonical samples")
            count += 1
        if count != len(weights):
            raise ValueError("weight CSV sample count mismatch")

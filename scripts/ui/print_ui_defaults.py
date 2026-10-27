#!/usr/bin/env python3
"""Print the launcher-selected task profile without starting any process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from runtime.collection_workspace import research_root
collection_root = research_root()
collection_suffix = "_test" if os.environ.get("MIMIC_UI_GEOMAGIC_SOURCE") == "synthetic_udp" else ""
experiment = os.environ["MIMIC_UI_EXPERIMENT"]
print(
    json.dumps(
        {
            
            "collection_workspace": str(collection_root) if collection_root else None,
            "task": os.environ["MIMIC_UI_TASK"],
            "leader": os.environ.get("MIMIC_UI_LEADER", "omy"),
            "input_backend": os.environ.get("VITE_MIMIC_INPUT_BACKEND", "zmq"),
            "haptics_enabled": (os.environ.get("MIMIC_UI_LEADER") == "geomagic"
                                and os.environ.get("MIMIC_UI_GEOMAGIC_SOURCE") == "physical_hdapi"
                                and os.environ.get("MIMIC_GEOMAGIC_HAPTICS", "1") == "1"),
            "geomagic_source": os.environ.get("MIMIC_UI_GEOMAGIC_SOURCE") or None,
            "collection_plan_id": None if collection_root else (os.environ.get('MIMIC_UI_COLLECTION_PLAN') or 'square_restricted30_v1')
                if os.environ.get('MIMIC_UI_LEADER') == 'geomagic'
                and os.environ.get('MIMIC_UI_GEOMAGIC_SOURCE') == 'physical_hdapi'
                and os.environ['MIMIC_UI_TASK'] == 'mimicgen_square' else
                'square_restricted30_v1' if os.environ['MIMIC_UI_TASK'] == 'mimicgen_square' else None,
            "sync_capture_root": os.environ.get("GEOMAGIC_SYNC_CAPTURE") or None,
            "sync_capture_seconds": os.environ.get("GEOMAGIC_SYNC_SECONDS") or None,
            "sync_capture_records": os.environ.get("GEOMAGIC_SYNC_RECORDS") or None,
            "experiment": experiment,
            "dataset_version": os.environ["MIMIC_UI_DATASET_VERSION"],
            "raw_root": str(collection_root / "raw" / collection_suffix) if collection_root else str(root / "experiments" / os.environ["VITE_MIMIC_DATASET_OUTPUT_ROOT"])
            if os.environ.get("VITE_MIMIC_DATASET_OUTPUT_ROOT") else
            str(root / "experiments" / experiment / "datasets" / "raw"),
            "processed_root": str(collection_root / "processed" / collection_suffix) if collection_root else str(
                root / "experiments" / experiment / "datasets" / "processed"
            ),
            "training_root": str(collection_root / "training" / collection_suffix) if collection_root else str(root / "experiments" / experiment / "training"),
        },
        sort_keys=True,
    )
)

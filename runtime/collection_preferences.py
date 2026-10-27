"""Persist an operator's folder choice independently of the UI process."""
import json
import os
from pathlib import Path
import tempfile

PREFERENCES = Path(__file__).resolve().parents[1]/"outputs/ui_preferences/touch_square.json"


def load_output(path=PREFERENCES):
    try:
        value=json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    if value.get("schema")!="touch_collection_preferences_v1" or not isinstance(value.get("output_folder"),str):
        raise ValueError("Invalid Touch collection preferences")
    # Keep the selection even if a drive is temporarily unavailable. Validation
    # at recording start blocks writes; never silently switch output folders.
    selected=Path(value["output_folder"])
    if not selected.is_absolute():
        raise ValueError("Saved collection folder must be absolute")
    return selected


def save_output(folder,path=PREFERENCES):
    from .collection_counts import validate_output
    selected=validate_output(folder)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,prefix=".touch_preferences_",delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(dict(schema="touch_collection_preferences_v1",output_folder=str(selected)),stream,indent=2)
            stream.write("\n");stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    return selected

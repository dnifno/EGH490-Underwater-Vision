"""
Configuration and class-label helpers for the Coralscapes evaluation notebooks.

- load_config: read configs/config.yaml.
- resolve_id2label: get the model's class-id -> class-name table.
- pick_mapping: build the lookup table that translates model class ids into the project class scheme, matching by class NAME (never by id).
"""

import numpy as np
import yaml

V2_DATASET_ID = "josauder/coralscapesV2"
UNMAPPED = 255  # lookup-table value for a model class that has no project class


def load_config(path):
    """Read the YAML config. Colours are returned as an (n_classes, 3) uint8 array."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    names = cfg["classes"]["names"]
    colours = np.asarray(cfg["classes"]["colours"], dtype=np.uint8)
    if colours.shape != (len(names), 3):
        raise ValueError(f"classes.colours must hold one RGB triple per class name ({len(names)} expected), got shape {colours.shape}")
    cfg["classes"]["colours"] = colours
    cfg["classes"]["ignore_index"] = int(cfg["classes"]["ignore_index"])
    return cfg


def fetch_v2_id2label(repo_id=V2_DATASET_ID):
    """Download the CoralscapesV2 class table from the dataset repo. Returns {} if unavailable."""
    import json
    from huggingface_hub import hf_hub_download
    try:
        with open(hf_hub_download(repo_id, "id2label.json", repo_type="dataset")) as f:
            return {int(k): str(v) for k, v in json.load(f).items()}
    except Exception as exc:  # offline, renamed repo, etc.
        print(f"  (could not load id2label.json from {repo_id}: {type(exc).__name__})")
        return {}


def _normalise_id2label(table, n_labels):
    """Return {int: str} with exactly n_labels entries, or None if the table is unusable."""
    if not table:
        return None
    table = {int(k): str(v) for k, v in table.items()}
    if all(v.lower().startswith("label_") for v in table.values()):  # generic placeholder names
        return None
    if 0 not in table and len(table) == n_labels - 1:
        table[0] = "unlabelled"
    return dict(sorted(table.items())) if len(table) == n_labels else None


def resolve_id2label(model, cfg):
    """
    Return (id2label, source). The model config is preferred; the dataset's
    id2label.json is the fallback. Raises if neither gives a usable table.
    """
    n_labels = model.config.num_labels
    table = _normalise_id2label(model.config.id2label, n_labels)
    if table is not None:
        return table, "model config"
    table = _normalise_id2label(fetch_v2_id2label(), n_labels)
    if table is not None:
        return table, "dataset id2label.json"
    raise ValueError(
        f"No usable class names for a {n_labels}-output model. Provide model.id2label_file in the config."
    )


def pick_mapping(id2label, cfg, names):
    """
    Build the model-id -> project-class lookup table.

    The mapping section is chosen by cfg['model']['mapping_key']; when that is
    absent, the first mapping whose class names all exist in the model is used.
    Returns (mapping_key, lut) where lut is a uint8 array indexed by model class
    id: the project class index, or 255 for a model class that is not mapped.
    A class name in the mapping that the model does not have raises an error
    listing every class the model does have.
    """
    name_to_id = {n.lower(): i for i, n in id2label.items()}

    def build(key):
        mapping = cfg[key]
        lut = np.full(max(id2label) + 1, UNMAPPED, dtype=np.uint8)
        missing, clashes = [], []
        for project_class, model_names in mapping.items():
            if project_class not in names:
                raise ValueError(f"Mapping '{key}' refers to '{project_class}', which is not in classes.names {names}")
            for model_name in model_names or []:
                idx = name_to_id.get(model_name.lower())
                if idx is None:
                    missing.append(model_name)
                    continue
                if lut[idx] != UNMAPPED and lut[idx] != names.index(project_class):
                    clashes.append(model_name)
                lut[idx] = names.index(project_class)
        return lut, missing, clashes

    key = cfg["model"].get("mapping_key")
    if key is None:
        for candidate in ("coralscapes_mapping", "v1_coralscapes_mapping"):
            if candidate in cfg and not build(candidate)[1]:
                key = candidate
                break
        if key is None:
            raise ValueError("No mapping in the config matches this model's class names.")

    lut, missing, clashes = build(key)
    if missing:
        listing = ", ".join(f"{i}: {n}" for i, n in sorted(id2label.items()))
        raise ValueError(f"Mapping '{key}' names classes the model does not have: {missing}\nModel classes: {listing}")
    if clashes:
        raise ValueError(f"Mapping '{key}' assigns these model classes to more than one project class: {clashes}")
    return key, lut

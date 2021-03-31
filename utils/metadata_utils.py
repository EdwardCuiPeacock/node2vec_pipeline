"""Pipeline Running Utilities. DO NOT MOVE!"""

import os
import pathlib
from typing import Dict
import yaml

# Get project directory
PROJECT_DIR = str(pathlib.Path(__file__).parent.parent)
DEFAULT_METADATA = os.path.join(PROJECT_DIR, "metadata.yaml")


def get_metadata(metadata_file: str = DEFAULT_METADATA) -> Dict:
    """Return the metadata dictionary."""
    with open(metadata_file, "r") as fid:
        metadata = yaml.safe_load(fid)

    return metadata


def get_config(metadata: Dict, field: str = "system_configurations") -> Dict:
    """Return the pipeline config dictionary."""
    config = {k: v_dict["value"] for k, v_dict in metadata[field].items()}
    return config
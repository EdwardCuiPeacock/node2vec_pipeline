"""Pipeline Running Utilities. DO NOT MOVE!"""

import os
import pathlib
from typing import Dict
import yaml

# Get project directory
PROJECT_DIR = str(pathlib.Path(__file__).parent.parent)
DEFAULT_METADATA = os.path.join(PROJECT_DIR, "metadata.yaml")


def get_metadata(metadata_file: str = DEFAULT_METADATA, additional_configs=["tfx"]) -> Dict:
    """Return the metadata dictionary."""
    metadata_file = DEFAULT_METADATA if metadata_file is None else metadata_file
    with open(metadata_file, "r") as fid:
        metadata = yaml.safe_load(fid)
    
    # Additional configs
    for con in additional_configs:
        if con == "tfx":
            metadata = make_tfx_configs(metadata)
        else:
            raise(NotImplementedError(f"Unrecognized config type: {con}"))

    return metadata


def get_config(metadata: Dict, field: str = "system_configurations") -> Dict:
    """Return the pipeline config dictionary."""
    config = {k: v_dict["value"] for k, v_dict in metadata[field].items()}
    return config

def parse_templated_fields(metadata):
    """Parse any string or array(string) fields that are templated."""
    parse_dict = {}
    for field in metadata:
        if "configurations" not in field:
            parse_dict.update({field: metadata[field]})
        else:
            parse_dict.update(get_config(metadata, field))

    # looping over config sections:
    for config_sec, configs in metadata.items():
        # looping over each field in the current config section
        for cur_key, cur_val in configs.items():
            if cur_val["type"] not in ["string", "array"]:
                continue # not string fields, template does not support
            
            if cur_val["type"] == "string" and "{" in cur_val and "}" in cur_val["value"]:
                cur_val["value"] = cur_val["value"].format(**parse_dict)
            else: # array
                for index, s in enumerate(cur_val["value"]):
                    cur_val["value"][index] = s.format(**parse_dict)
        
            metadata[config_sec][cur_key]["value"] = cur_val["value"]
            
    return metadata


def make_tfx_configs(metadata: Dict) -> Dict:
    """Add additional metadata fields for TFX processing."""
    system_config = get_config(metadata, "system_configurations")
   

    # %% pipeline_root
    # TFX produces two types of outputs, files and metadata.
    # - Files will be created under "pipeline_root" directory.
    pipeline_root = {
        "description": """TFX produces two types of outputs, files and metadata.
    Files will be created under 'pipeline_root' directory.""",
        "type": "string",
        "value": os.path.join(
            system_config["gcs_bucket_name"],
            "tfx_pipeline_output",
            metadata["pipeline_name"] + "_" + metadata["pipeline_version"],
        ),
    }
    metadata["system_configurations"]["pipeline_root"] = pipeline_root

    # %% model_serve_dir
    # The last component of the pipeline, "Pusher" will produce serving model under
    # model_serve_dir.
    model_serve_dir = {
        "description": "",
        "type": "string",
        "value": os.path.join(pipeline_root["value"], "serving_model"),
    }
    metadata["system_configurations"]["model_serve_dir"] = model_serve_dir

    return metadata
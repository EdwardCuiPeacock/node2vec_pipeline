"""Pipeline Running Utilities. DO NOT MOVE!"""

import os
import pathlib
from typing import Dict, List, Text, Optional
import yaml
import json
import jinja2

# Get project directory
PROJECT_DIR = str(pathlib.Path(__file__).parent.parent)
DEFAULT_METADATA = os.path.join(PROJECT_DIR, "metadata.yaml")


def get_metadata(metadata_file: str = DEFAULT_METADATA) -> Dict:
    """Return the metadata dictionary."""
    metadata_file = DEFAULT_METADATA if metadata_file is None else metadata_file
    with open(metadata_file, "r") as fid:
        metadata = yaml.safe_load(fid)
    
    # Parse any templated fields
    metadata = parse_templated_fields(metadata)

    return metadata


def get_config(metadata: Dict, field: Optional[Text] = "system_configurations", 
               filter_type: Optional[List[Text]] = None) -> Dict:
    """Return the pipeline config dictionary."""
    if filter_type is not None:
        config = {k: v_dict["value"] for k, v_dict in metadata[field].items() if v_dict["type"] in filter_type}
    else:
        config = {k: v_dict["value"] for k, v_dict in metadata[field].items()}
    return config




def parse_templated_fields(metadata: Dict) -> Dict:
    """Parse any strings, array(string), or dict fields that are templated."""
    parse_dict = {}
    for field in metadata:
        if "configurations" not in field:
            parse_dict.update({field: metadata[field]})
        else:
            parse_dict.update(get_config(metadata, field))
            
    
    def _recursive_render(s, cur_key):
        counter = 0
        while "{{ " in s and " }}" in s:
            s = jinja2.Template(s).render(**parse_dict)
            counter += 1
            if counter > 100:
                raise(ValueError(f"Cannot parse templated field {cur_key}"))
        return s

    # looping over config sections:
    for config_sec, configs in metadata.items():
        if"configurations" not in config_sec:
            continue
        # looping over each field in the current config section
        for cur_key, cur_val in configs.items():
            if cur_val["type"] in ["string", "str"]:
                cur_val["value"] = _recursive_render(cur_val["value"], cur_key)
            elif cur_val["type"] == "array":
                for index, s in enumerate(cur_val["value"]):                    
                    cur_val["value"][index] = _recursive_render(s, cur_key)
            elif cur_val["type"] == "object": # a dict
                # convert to json string
                object_str = json.dumps(cur_val["value"]).replace("\\", "")
                # parse it like a string
                object_str = _recursive_render(object_str, cur_key)
                # convert it back to dict
                cur_val["value"] = json.loads(object_str)
                
            metadata[config_sec][cur_key]["value"] = cur_val["value"]
            
    return metadata


if __name__ == '__main__':
    metadata = get_metadata()
    yaml.dump(metadata, open("test.yaml", "w"))
"""Utilities to interact with query strings / files."""

import os
import pathlib
from typing import Text, Dict, Optional


# Get project directory
PROJECT_DIR = str(pathlib.Path(__file__).parent.parent)

def load_query_string(query_file_path:Text, field_dict:Optional[Dict] = None) -> Text:
    """Load a query string from a .sql file and parse the fields."""
    # Open the file and read all its content
    with open(os.path.join(PROJECT_DIR, query_file_path), "r") as fid:
        query_str = fid.read() # read everything
    
    # Apply / parse any templated fields
    if field_dict is not None:
        query_str = query_str.format(**field_dict)
    
    return query_str
        
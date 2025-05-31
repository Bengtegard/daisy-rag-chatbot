"""
Utility functions for the Data Science RAG system.
"""

import os
import hashlib
import time
import yaml
from typing import List, Dict, Any
from pathlib import Path

def get_document_hash(content: str) -> str:
    """Generate a unique hash for document content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Parsed config as a nested dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config

def summarize_materials(data_dir: str) -> Dict[str, Any]:
    """Generate a summary of the materials in the data directory."""
    summary = {
        "total_files": 0,
        "file_types": {},
        "categories": {},
        "total_size_mb": 0
    }
    
    data_path = Path(data_dir)
    if not data_path.exists():
        return summary
    
    # Process all files recursively
    for file_path in data_path.glob('**/*'):
        if file_path.is_file():
            # Count files
            summary["total_files"] += 1
            
            # Count file types
            ext = file_path.suffix.lower()
            if ext in summary["file_types"]:
                summary["file_types"][ext] += 1
            else:
                summary["file_types"][ext] = 1
            
            # Count categories (using parent folder name)
            category = file_path.parent.name
            if category in summary["categories"]:
                summary["categories"][category] += 1
            else:
                summary["categories"][category] = 1
            
            # Calculate size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            summary["total_size_mb"] += size_mb
    
    # Round the total size
    summary["total_size_mb"] = round(summary["total_size_mb"], 2)
    
    return summary
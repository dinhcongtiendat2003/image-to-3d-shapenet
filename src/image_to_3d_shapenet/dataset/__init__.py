"""ShapeNetCore download and preparation API."""

from .archive import UnsafeArchiveError, safe_extract_zip
from .auth import ACCESS_INSTRUCTIONS, MissingTokenError, get_hf_token
from .download import DatasetAccessError, DatasetResourceError, download_category
from .meshes import find_normalized_meshes, normalize_mesh, validate_mesh
from .pipeline import prepare_dataset
from .splits import create_splits
from .base_mesh_cache import generate_base_mesh_cache
from .training_dataset import RenderedRefinementDataset, single_item_collate

__all__ = [
    "ACCESS_INSTRUCTIONS",
    "DatasetAccessError",
    "DatasetResourceError",
    "MissingTokenError",
    "UnsafeArchiveError",
    "create_splits",
    "download_category",
    "find_normalized_meshes",
    "get_hf_token",
    "normalize_mesh",
    "prepare_dataset",
    "safe_extract_zip",
    "validate_mesh",
    "generate_base_mesh_cache",
    "RenderedRefinementDataset",
    "single_item_collate",
]



# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from pathlib import Path
from typing import Optional

import vllm_logits.envs as envs
from vllm_logits.connections import global_http_connection

VLLM_LOGITS_S3_BUCKET_URL = "https://vllm_logits-public-assets.s3.us-west-2.amazonaws.com"


def get_cache_dir() -> Path:
    """Get the path to the cache for storing downloaded assets."""
    path = Path(envs.VLLM_LOGITS_ASSETS_CACHE)
    path.mkdir(parents=True, exist_ok=True)

    return path


@lru_cache
def get_vllm_logits_public_assets(filename: str,
                           s3_prefix: Optional[str] = None) -> Path:
    """
    Download an asset file from ``s3://vllm_logits-public-assets``
    and return the path to the downloaded file.
    """
    asset_directory = get_cache_dir() / "vllm_logits_public_assets"
    asset_directory.mkdir(parents=True, exist_ok=True)

    asset_path = asset_directory / filename
    if not asset_path.exists():
        if s3_prefix is not None:
            filename = s3_prefix + "/" + filename
        global_http_connection.download_file(
            f"{VLLM_LOGITS_S3_BUCKET_URL}/{filename}",
            asset_path,
            timeout=envs.VLLM_LOGITS_IMAGE_FETCH_TIMEOUT)

    return asset_path

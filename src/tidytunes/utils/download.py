from pathlib import Path

import requests
from platformdirs import user_cache_dir
from tqdm import tqdm


def download_github(tag: str, filename: str) -> Path:
    """
    Downloads an artifact from a GitHub release and caches it locally.

    Args:
        tag (str): GitHub release tag.
        filename (str): Name of the file in the release.

    Returns:
        Path: Path to the downloaded or cached file.
    """
    cache_dir = Path(user_cache_dir("tidytunes", version=tag))
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = (
        f"https://github.com/meaningTeam/tidy-tunes/releases/download/{tag}/{filename}"
    )
    cached_file = cache_dir / filename

    if cached_file.exists():
        return cached_file

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(cached_file, "wb") as file, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {url} to {cached_file}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress.update(len(chunk))

    return cached_file

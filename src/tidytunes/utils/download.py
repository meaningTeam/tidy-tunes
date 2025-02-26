from pathlib import Path

import requests
from platformdirs import user_cache_dir
from tqdm import tqdm
from importlib.metadata import version


def download_github(filename: str, tag: str = None) -> Path:
    """
    Downloads an artifact from a GitHub release and caches it locally.

    Args:
        filename (str): Name of the file in the release.
        tag (str): GitHub release tag.

    Returns:
        Path: Path to the downloaded or cached file.
    """

    if tag is None:
        tag = "v" + version("tidytunes")

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

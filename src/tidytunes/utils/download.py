from importlib.metadata import version
from pathlib import Path

import requests
from platformdirs import user_cache_dir
from tqdm import tqdm


def download_github(filename: str | list[str], tag: str = None) -> Path:
    """
    Downloads an artifact from a GitHub release and caches it locally.
    If a list of filenames is provided, downloads all parts and reassembles
    them into a single file.

    Args:
        filename (str | list[str]): Name of the file in the release, or a list
            of filenames representing parts to be concatenated.
        tag (str): GitHub release tag.

    Returns:
        Path: Path to the downloaded or cached file.
    """

    if isinstance(filename, list):
        return _download_and_reassemble(filename, tag)

    return _download_single(filename, tag)


def _download_single(filename: str, tag: str = None) -> Path:
    """
    Downloads a single file from a GitHub release and caches it locally.

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


def _download_and_reassemble(filenames: list[str], tag: str = None) -> Path:
    """
    Downloads multiple file parts from a GitHub release and reassembles them
    into a single file.

    Args:
        filenames (list[str]): List of filenames representing parts to be concatenated.
            The output filename is derived from the first part by removing common
            split suffixes (e.g., ".part-aa", ".part-01", ".part1").
        tag (str): GitHub release tag.

    Returns:
        Path: Path to the reassembled file.
    """
    import re

    if not filenames:
        raise ValueError("filenames list cannot be empty")

    if tag is None:
        tag = "v" + version("tidytunes")

    cache_dir = Path(user_cache_dir("tidytunes", version=tag))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filename by removing split suffix from first part
    first_part = filenames[0]
    output_filename = re.sub(
        r"\.part[-_]?[a-z0-9]+$", "", first_part, flags=re.IGNORECASE
    )
    if output_filename == first_part:
        # No suffix matched, just use the first filename without extension + combined
        output_filename = (
            first_part.rsplit(".", 1)[0] if "." in first_part else first_part
        )

    output_file = cache_dir / output_filename

    if output_file.exists():
        return output_file

    # Download all parts
    part_paths = [_download_single(f, tag) for f in filenames]

    # Reassemble into single file
    with open(output_file, "wb") as out:
        for part in part_paths:
            with open(part, "rb") as f:
                out.write(f.read())

    return output_file

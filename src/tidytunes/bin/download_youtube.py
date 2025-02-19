import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click

from tidytunes.bin import cli
from tidytunes.utils import setup_logger


def download_video(
    video_id: str,
    download_dir: Path,
    proxy_api_key: str,
    proxy_endpoint: str,
    subtitle_language: str,
    logger: logging.Logger,
):
    """
    Download a YouTube video using yt-dlp with a proxy.

    Args:
        video_id (str): The ID of the YouTube video.
        download_dir (Path): The directory to save downloaded files.
        proxy_api_key (str): API key for the proxy server.
        proxy_endpoint (str): Proxy endpoint address.
        subtitle_language (str): Language of subtitles.
    """

    folder = Path(download_dir) / video_id[:2]
    root = folder / video_id
    root.mkdir(exist_ok=True, parents=True)
    audio_path = root / f"{video_id}.flac"
    metadata_path = root / f"{video_id}.info.json"

    if audio_path.exists() and metadata_path.exists():
        logger.info(f"Video {video_id} is already processed. Skipping.")
        return

    proxy_url = f"http://{proxy_api_key}:@{proxy_endpoint}"
    command = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "flac",
        "-o",
        f"{video_id}.%(ext)s",
        "-P",
        f"{root}",
        "--write-info-json",
        "--write-auto-subs",
        "--sub-lang",
        subtitle_language,
        "--sub-format",
        "srt",
        "--proxy",
        proxy_url,
        "--no-check-certificate",
        f"https://www.youtube.com/watch?v={video_id}",
    ]

    try:
        subprocess.run(command, check=True)
        logger.info(f"Successfully processed video {video_id}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading video {video_id}: {e}")


@cli.command()
@click.argument("video-ids", nargs=-1, type=str)
@click.option("--download_dir", "-d", type=click.Path(), help="Ouput directory")
@click.option("--proxy-api-key", "-k", type=str, help="API key for the proxy server.")
@click.option(
    "--proxy-endpoint",
    "-e",
    default="api.zyte.com:8011",
    help="Proxy endpoint address.",
)
@click.option(
    "--max-workers",
    "-w",
    default=4,
    type=int,
    help="Number of parallel download threads.",
)
@click.option(
    "--subtitle-language",
    "-l",
    default="en",
    type=str,
    help="Language code for substitle language.",
)
def download_youtube(
    video_ids,
    download_dir,
    proxy_api_key,
    proxy_endpoint,
    max_workers,
    subtitle_language,
):

    logger = setup_logger("tidytunes")

    proxy_api_key = proxy_api_key or os.getenv("API_KEY")

    if not proxy_api_key:
        raise ValueError("API key for your proxy endpoint is missing!")

    click.secho(
        f"Going to download {len(video_ids)} to {download_dir}", fg="green", bold=True
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(
            lambda vid: download_video(
                vid,
                Path(download_dir),
                proxy_api_key,
                proxy_endpoint,
                subtitle_language,
                logger,
            ),
            video_ids,
        )

    click.secho(f"Downloading finished!", fg="green", bold=True)

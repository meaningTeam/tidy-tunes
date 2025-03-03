import json
from pathlib import Path

import click
import hdbscan
from tqdm import tqdm

from tidytunes.bin.cli import cli
from tidytunes.pipeline_components.speaker_segmentation import compute_mean_embeddings
from tidytunes.utils import Audio, chunk_list, setup_logger


@cli.command()
@click.argument("audio-paths", nargs=-1, type=str)
@click.option(
    "--out",
    "-o",
    required=True,
    type=click.Path(),
    default="speaker_clusters.json",
    help="Path to the speaker map json file.",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    help="Device on which to run the pipeline.",
)
@click.option(
    "--min-cluster-size",
    "-c",
    type=int,
    default=5,
    help="Minimum cluster size for HDBSCAN.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=100,
    help="Batch size for processing audio files.",
)
def cluster_speakers(audio_paths, out, device, min_cluster_size, batch_size):

    out_path = Path(out)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    click.secho(f"Processing {len(audio_paths)} audio files")

    speaker_embeddings = []
    full_paths = []
    for chunk in tqdm(
        chunk_list(audio_paths, batch_size), desc="Computing speaker embeddings"
    ):
        audio = [Audio.from_file(pth) for pth in chunk]
        embeddings = compute_mean_embeddings(audio, device)
        speaker_embeddings.extend(embeddings)
        full_paths.extend([a.origin.path for a in audio])

    click.secho(f"Clustering speakers with HDBSCAN ({min_cluster_size=})")
    speaker_labels = (
        hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        .fit(speaker_embeddings)
        .labels_
    )
    speaker_map = {
        str(pth): int(label) for pth, label in zip(full_paths, speaker_labels)
    }

    with open(out_path, "w") as f:
        json.dump(speaker_map, f, indent=4)

from datetime import timedelta
from pathlib import Path

import click
import torch
import yaml

from tidytunes.bin.cli import cli
from tidytunes.pipeline_components import (
    find_segments_with_single_speaker,
    find_segments_with_speech,
    find_segments_without_music,
    get_denoised_pesq,
    get_dnsmos,
    get_language_probabilities,
    get_rolloff_frequency,
)
from tidytunes.pipeline_components.dnsmos import load_dnsmos_model
from tidytunes.utils import (
    Audio,
    configure_logger,
    get_logger,
    partition,
    trim_audios,
)
from tidytunes.utils.memory import garbage_collection_cuda, is_oom_error

PIPELINE_FUNCTIONS = {
    "voice_separation": find_segments_without_music,
    "speech_segmentation": find_segments_with_speech,
    "speaker_segmentation": find_segments_with_single_speaker,
    "rolloff_filtering": get_rolloff_frequency,
    "denoising": get_denoised_pesq,
    "mos_filtering": get_dnsmos,
    "language_filtering": get_language_probabilities,
}


def process_audio(audios, device, pipeline_components):
    assert len(audios) == 1
    logger = get_logger()
    throughput_stats = {}
    audio_segments = audios

    logger.info(
        f"Starting audio processing pipeline with {len(pipeline_components)} components"
    )
    logger.info(f"Initial audio duration: {audios[0].duration:.2f} seconds")

    for name, func, kwargs, filter_fn in pipeline_components:
        logger.info(f"Starting pipeline step: {name}")
        logger.debug(
            f"Step {name} - Input segments: {len(audio_segments)}, Total duration: {sum(a.duration for a in audio_segments):.2f}s"
        )

        # to improve batching efficiency
        audio_segments = sorted(audio_segments, key=lambda x: x.duration)

        try:
            values = func(audio_segments, device=device, **kwargs)
            if filter_fn:
                audio_segments, _ = partition(
                    audio_segments, by=[filter_fn(v) for v in values]
                )
            else:
                audio_segments = trim_audios(audio_segments, values)

            throughput_stats[name] = sum(a.duration for a in audio_segments)
            logger.info(
                f"Completed pipeline step: {name} - Remaining segments: {len(audio_segments)}, Duration: {throughput_stats[name]:.2f}s"
            )

        except RuntimeError as e:
            if not is_oom_error(e):
                raise
            garbage_collection_cuda()
            audio_segments = []
            logger.error(
                f"Pipeline step {name} failed with OOM error, skipping remaining processing"
            )
            click.secho(
                "Failed to process a possibly too large audio file! Skipping ...",
                fg="red",
                bold=True,
            )

        if not audio_segments:
            logger.warning(f"No audio segments remaining after step: {name}")
            return audio_segments, throughput_stats

    logger.info(
        f"Audio processing pipeline completed. Final segments: {len(audio_segments)}"
    )
    return audio_segments, throughput_stats


@cli.command()
@click.argument("audio-paths", nargs=-1, type=str)
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, readable=True, path_type=Path),
    help="Path to the YAML config file defining pipeline components and parameters.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=click.Path(),
    help="Destination directory to save processed files.",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    help="Device on which to run the pipeline.",
)
@click.option("--overwrite", "-w", is_flag=True, help="Overwrite processed files.")
def process_audios(audio_paths, config, out, device, overwrite):
    # Configure the shared logger
    configure_logger(log_file="pipeline.log.txt")
    logger = get_logger()

    logger.info("Starting audio processing batch")
    logger.info(f"Config file: {config}")
    logger.info(f"Output directory: {out}")
    logger.info(f"Device: {device}")
    logger.info(f"Overwrite: {overwrite}")

    with open(config, "r") as f:
        config_data = yaml.safe_load(f)

    pipeline_components = []
    for component in config_data.get("pipeline", []):
        name = component["name"]
        if name not in PIPELINE_FUNCTIONS:
            raise ValueError(f"Unknown pipeline component: {name}")

        func = PIPELINE_FUNCTIONS[name]
        params = component.get("params", {})
        condition = eval(component["condition"]) if "condition" in component else None

        pipeline_components.append((name, func, params, condition))
        logger.info(f"Configured pipeline component: {name} with params: {params}")

    logger.info(
        f"Pipeline configured with {len(pipeline_components)} components: {[c[0] for c in pipeline_components]}"
    )

    out_path = Path(out)
    out_path.mkdir(exist_ok=True, parents=True)
    out_processed = out_path / "pipeline.processed.json"

    torch.backends.cudnn.benchmark = False
    load_dnsmos_model(torch.device(device), True, 8)

    processed_paths = set()
    if not overwrite and out_processed.exists():
        with open(out_processed, "r") as f:
            processed_paths = {l.strip() for l in f}

    paths = set(audio_paths) - processed_paths
    click.secho(
        f"Going to process {len(paths)} recordings ({len(audio_paths)} found, {len(audio_paths) - len(paths)} already processed)",
        fg="green",
        bold=True,
    )

    with open(out_processed, "a+") as f:
        for i, pth in enumerate(paths, 1):
            logger.info(f"Processing file {i}/{len(paths)}: {pth}")
            try:
                audio = Audio.from_file(pth)
                logger.debug(
                    f"Loaded audio file - Duration: {audio.duration:.2f}s, Sample rate: {audio.sampling_rate}"
                )
            except Exception as e:
                logger.error(f"Failed to load audio {pth}: {e}")
                click.secho(f"Failed to load audio {pth}: {e}", fg="red", bold=True)
                continue

            audio_segments, throughput_stats = process_audio(
                [audio], device, pipeline_components
            )

            preserved = sum(a.duration for a in audio_segments)
            throughput_str = ", ".join(
                f"{k}: {(throughput_stats[k] / audio.duration * 100):.1f}%"
                for k in throughput_stats
            )
            logger.info(
                f"Audio {audio.origin.id}, preserved {timedelta(seconds=int(preserved))} ({throughput_str})"
            )
            print(pth, file=f, flush=True)

            for a in audio_segments:
                a.to_file(root=out_path / a.origin.id)

            logger.info(
                f"Saved {len(audio_segments)} audio segments for {audio.origin.id}"
            )

    logger.info("Processing finished!")
    click.secho(f"Processing finished!", fg="green", bold=True)

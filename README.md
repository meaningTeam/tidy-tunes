# Tidy Tunes

Tidy Tunes is an easy-to-use pipeline for mining high-quality audio data for speech generation models. To do so, it chains multiple open source models while minimizing dependencies.

Specifically, the pipeline includes (see also https://www.arxiv.org/pdf/2409.03283):
- voice source segmentation
- segmentation based on voice activity detection
- speaker segmentation 
- rolloff frequency filtering
- filtering based on PESQ of denoised audio 
- DNSMOS filtering
- spoken language identification filtering

It provides two commands, one for downloading audios from sources like YouTube, and one for processing the downloaded audio efficiently.

## Installation

You can install Tidy Tunes directly from the GitHub repository using pip:

```sh
pip install git+https://github.com/meaningTeam/tidy-tunes.git
```
If you wish, clone the repository or install additional dependencies:
```sh
git clone git@github.com:meaningTeam/tidy-tunes.git && cd tidy-tunes
pip install -e .[dev]
```

## Usage

The CLI offers two commands, one for downloading audio from videos from the internet (YouTube), the other one for processing audio files with a stack of data-cleaning models.

To download the audios, use:
```
tidytunes download-youtube [OPTIONS] [VIDEO_IDS]...

Options:
  -d, --download_dir PATH       Ouput directory
  -k, --proxy-api-key TEXT      API key for the proxy server.  [required]
  -e, --proxy-endpoint TEXT     Proxy endpoint address.
  -w, --max-workers INTEGER     Number of parallel download threads.
  -l, --subtitle-language TEXT  Language code for substitle language.
```
You can place your secret proxy API_KEY into `~/.tidytunes` and it will be auto-discovered by the script, for example:
```
API_KEY=Wk1yM8sY9M4S7rT3g5Xq2L6bV0NfJHp
```

To process the downloaded audios, use:
```
tidytunes process-audios [OPTIONS] [AUDIO_PATHS]...

Options:
  -c, --config PATH               Path to the YAML config file defining pipeline components and parameters.  [required]
  -o, --out PATH                  Destination directory to save processed
                                  files.  [required]
  -d, --device [cpu|cuda]         Device on which to run the pipeline.
  -w, --overwrite                 Overwrite processed files.
```
You can find the default config at `configs/full_en.yaml`. If needed, modify or disable the pipeline components and their parameters.

## Using in Python

Tidy Tunes can also be used directly in Python. Example usage:

```python
from tidytunes.utils import Audio, trim_audios, partition
from tidytunes.pipeline_components import find_segments_with_single_speaker, get_dnsmos

device = "cuda"
path = "path/to/my_audio.flac"
output_dir = "processed/"

audio_segments = [Audio.from_file(path)]

timestamps = find_segments_with_single_speaker(audio_segments, min_duration=3.2, device=device)
audio_segments = trim_audios(audio_segments, timestamps)

mos = get_dnsmos(audio_segments, device=device)
audio_segments, poor_quality_audio_segments = partition(audio_segments, by=mos > 3.3)

for segment in zipaudio_segments:
    segment.play()
    segment.to_file(output_dir)
``````


## Citation

If used, please cite it as follows:

```
@misc{tidytunes,
  author = {Tomas Nekvinda and Jan Vainer and Vojtech Srdecny},
  title = {Tidy Tunes: An Easy-to-Use Pipeline for Mining High-Quality Audio Data},
  year = {2025},
  howpublished = {\url{https://github.com/meaningTeam/tidy-tunes}},
  note = {Version 1.0.0}
}
```

## License

Tidy Tunes is released under the MIT License. See [LICENSE](LICENSE) for details.

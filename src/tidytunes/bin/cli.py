import click


@click.group()
def cli(invoke_without_command=True):
    """🧼🎶 TidyTunes - download and clean audio files 🧼🎶"""
    import os

    from dotenv import load_dotenv

    load_dotenv(os.path.expanduser("~/.tidytunes"))

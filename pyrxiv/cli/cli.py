import click
from pathlib import Path
import time


@click.group(help="Entry point to run `bam_masterdata` CLI commands.")
def cli():
    pass


@cli.command(
    name="search_and_download",
    help="Searchs papers in arXiv for a specified category and downloads them in a specified path.",
)
@click.option(
    "--category",
    "-c",
    type=str,
    default="cond-mat.str-el",
    required=False,
    help="""
    (Optional) The arXiv category on which the papers will be searched. Defaults to "cond-mat.str-el".
    """,
)
@click.option(
    "--download-path",
    "-path"
    type=str,
    default="./data/",
    required=False,
    help="""
    (Optional) The path for downloading the arXiv PDFs. Defaults to "./data/".
    """,
)
# add more options: regex pattern, n papers, etc
def fill_masterdata(category, download_path):
    start_time = time.time()

    # check if `download_path` exists, and if not, create it
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)

    elapsed_time = time.time() - start_time
    click.echo(f"Downloaded arXiv papers in {elapsed_time:.2f} seconds\n\n")
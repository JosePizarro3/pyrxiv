import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrxiv.datamodel import ArxivPaper

import click

from pyrxiv.download import ArxivDownloader
from pyrxiv.extract import TextExtractor
from pyrxiv.fetch import ArxivFetcher
from pyrxiv.logger import logger


def run_search_and_download(
    download_path: Path = Path("data"),
    category: str = "cond-mat.str-el",
    n_papers: int = 5,
    regex_pattern: str = "",
    start_id: str | None = None,
    start_from_filepath: bool = False,
    loader: str = "pdfminer",
) -> tuple[list[Path], list["ArxivPaper"]]:
    """
    Searches for a specific number of papers `n_papers` in arXiv for a specified `category` and downloads
    them in a `download_path`.

    If `regex_pattern` is specified, only the papers that contain the pattern will be downloaded.
    If `start_id` is specified, the search will start from that ID.
    If `start_from_filepath` is True, the search will start from the last downloaded paper's ID.
    If `loader` is specified, the text will be extracted using the corresponding loader.


    Args:
        download_path (Path, optional): The path for downloading the arXiv PDFs. Defaults to Path("data").
        category (str, optional): The arXiv category on which the papers will be searched. Defaults to "cond-mat.str-el".
        n_papers (int, optional): The number of arXiv papers to be fetched and downloaded.
            If `regex_pattern` is not specified, this would correspond to the n_papers starting from the newest in the `category`. Defaults to 5.
        regex_pattern (str, optional): If specified, this regex pattern is searched in the arXiv papers so only the ones with
            the corresponding match will be downloaded. Defaults to "".
        start_id (str | None, optional): If specified, the search will start from this arXiv ID. Defaults to None.
        start_from_filepath (bool, optional): If True, the search will start from the last downloaded arXiv ID. Otherwise, it will start from the
            newest papers in the `category`. Defaults to False.
        loader (str, optional): PDF loader to use for extracting text from the downloaded PDFs.
            Defaults to "pdfminer". Available loaders: "pdfminer", "pypdf".

    Returns:
        tuple[list[Path], list[ArxivPaper]]: A tuple containing a list of Paths to the downloaded PDFs and a list of ArxivPaper objects
            with the extracted text.
    """
    if loader not in ["pdfminer", "pypdf"]:
        raise ValueError(
            f"Invalid loader: {loader}. Available loaders: 'pdfminer', 'pypdf'."
        )

    # check if `download_path` exists, and if not, create it
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)

    # Initializing classes
    fetcher = ArxivFetcher(
        download_path=download_path,
        category=category,
        start_id=start_id,
        start_from_filepath=start_from_filepath,
        logger=logger,
    )
    downloader = ArxivDownloader(download_path=download_path, logger=logger)
    extractor = TextExtractor(logger=logger)

    pattern_files: list[Path] = []
    pattern_papers: list[ArxivPaper] = []
    with click.progressbar(
        length=n_papers, label="Downloading and processing papers"
    ) as bar:
        while len(pattern_papers) < n_papers:
            papers = fetcher.fetch()
            for paper in papers:
                pdf_path = downloader.download_pdf(arxiv_paper=paper)
                text = extractor.get_text(pdf_path=pdf_path, loader=loader)
                if not text:
                    logger.info("No text extracted from the PDF.")
                    continue

                # Deleting downloaded PDFS that do not match the regex pattern
                regex = re.compile(regex_pattern) if regex_pattern else None
                if regex and not regex.search(text):
                    pdf_path.unlink()
                    continue

                # If the paper matches the regex_pattern, store text in the corresponding ArxivPaper object
                paper.text = text

                pattern_files.append(pdf_path)
                pattern_papers.append(paper)
                bar.update(1)
    return pattern_files, pattern_papers


@click.group(help="Entry point to run `pyrxiv` CLI commands.")
def cli():
    pass


@cli.command(
    name="search_and_download",
    help="Searchs papers in arXiv for a specified category and downloads them in a specified path.",
)
@click.option(
    "--download-path",
    "-path",
    type=str,
    default="data",
    required=False,
    help="""
    (Optional) The path for downloading the arXiv PDFs. Defaults to "data".
    """,
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
    "--n-papers",
    "-n",
    type=int,
    default=5,
    required=False,
    help="""
    (Optional) The number of arXiv papers to be fetched and downloaded. If `regex-pattern` is not specified, this
    would correspond to the n_papers starting from the newest in the `category`. Defaults to 5.
    """,
)
@click.option(
    "--regex-pattern",
    "-regex",
    type=str,
    required=False,
    help="""
    (Optional) If specified, this regex pattern is searched in the arXiv papers so only the ones with
    the corresponding match will be downloaded.
    """,
)
@click.option(
    "--start-id",
    "-s",
    type=str,
    required=False,
    help="""
    (Optional) If specified, the search will start from this arXiv ID. This is useful for resuming the search
    from a specific point. If not specified, the search will start from the newest papers in the `category`.
    """,
)
@click.option(
    "--start-from-filepath",
    "-sff",
    type=bool,
    default=False,
    required=False,
    help="""
    (Optional) If specified, the search will start from the last downloaded arXiv ID. This is useful for resuming
    the search from a specific point. If not specified, the search will start from the newest papers in the `category`.
    """,
)
@click.option(
    "--loader",
    "-l",
    type=click.Choice(["pdfminer", "pypdf"], case_sensitive=False),
    default="pdfminer",
    required=False,
    help="""
    (Optional) PDF loader to use for extracting text from the downloaded PDFs. Defaults to "pdfminer".
    Available loaders: "pdfminer", "pypdf".
    """,
)
def search_and_download(
    download_path,
    category,
    n_papers,
    regex_pattern,
    start_id,
    start_from_filepath,
    loader,
):
    start_time = time.time()

    run_search_and_download(
        download_path=Path(download_path),
        category=category,
        n_papers=n_papers,
        regex_pattern=regex_pattern,
        start_id=start_id,
        start_from_filepath=start_from_filepath,
        loader=loader,
    )

    elapsed_time = time.time() - start_time
    click.echo(f"Downloaded arXiv papers in {elapsed_time:.2f} seconds\n\n")

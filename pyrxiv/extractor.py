import re
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structlog._config import BoundLoggerLazyProxy

import requests
import xmltodict
from langchain_community.document_loaders import PDFMinerLoader, PyPDFLoader

from pyrxiv.datamodel import ArxivPaper, Author
from pyrxiv.logger import logger

DMFT_PATTERN = re.compile(r"\bDMFT\b|\bDynamical Mean[- ]Field Theory\b", re.IGNORECASE)


class TextExtractor:
    """
    Extract text from the PDF file using LangChain implementation of PDF loaders. This class also
    implements the text cleaning methods.
    """

    def __init__(self, **kwargs):
        self.logger = kwargs.get("logger", logger)

        # Implemented loaders from LangChain
        self.available_loaders = {
            "pypdf": PyPDFLoader,
            "pdfminer": PDFMinerLoader,
        }

    def _check_pdf_path(self, pdf_path: str | None = ".") -> bool:
        """
        Check if the PDF path is valid.

        Args:
            pdf_path (str | None): The path to the PDF file. If None, it will return False.

        Returns:
            bool: True if the PDF path is valid, False otherwise.
        """
        pdf_path = str(pdf_path)  # to avoid potential problems when being a Path object
        if not pdf_path:
            self.logger.error(
                "No PDF path provided. Returning an empty string for the text."
            )
            return False
        return Path(pdf_path).exists() and pdf_path.endswith(".pdf")

    def get_text(self, pdf_path: str | None = ".", loader: str = "pdfminer") -> str:
        """
        Extract text from the PDF file using LangChain implementation of PDF loaders.

        Read more: https://python.langchain.com/docs/how_to/document_loader_pdf/

        Args:
            pdf_path (str | None, optional): The path to the PDF file. Defaults to ".", the root project directory.
            loader (str, optional): The loader to use for extracting text from the PDF file. Defaults to "pdfminer".

        Returns:
            str: The extracted text from the PDF file.
        """
        # Check if the PDF path is valid
        if not self._check_pdf_path(pdf_path=pdf_path):
            return []
        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path)
        filepath = pdf_path

        # Check if the loader is available
        if loader not in self.available_loaders.keys():
            self.logger.error(
                f"Loader {loader} not available. Available loaders: {self.available_loaders.keys()}"
            )
            return []
        loader_cls = self.available_loaders[loader](filepath)

        # Extract text
        text = ""
        for page in loader_cls.lazy_load():
            text += page.page_content
        return text

    def delete_references(self, text: str = "") -> str:
        """
        Delete the references section from the text by detecting where its section might be.

        Args:
            text (str): The text to delete the references section from.

        Returns:
            str: The text without the references section if a match is found.
        """
        pattern_start = "(?:\nReferences\n|\nBibliography\n|\n\[1\] *[A-Z])"
        pattern_end = "(?:\nSupplemental Material[\:\n]*|\nSupplemental Information[\:\n]*|\nAppendices[\:\n]*)"

        match_start = re.search(pattern_start, text, flags=re.IGNORECASE)
        match_end = re.search(pattern_end, text, flags=re.IGNORECASE)
        if match_start:
            start = match_start.start()
            if match_end:
                end = match_end.start()
                return text[:start] + text[end:]
            return text[:start]
        return text

    def clean_text(self, text: str = "") -> str:
        """
        Clean and normalize extracted PDF text.

        - Remove hyphenation across line breaks.
        - Normalize excessive line breaks and spacing.
        - Remove arXiv identifiers and footnotes.
        - Strip surrounding whitespace.

        Args:
            text (str): Raw text extracted from a PDF.

        Returns:
            str: Cleaned text.
        """
        if not text:
            self.logger.warning("No text provided for cleaning.")
            return ""

        # Fix hyphenated line breaks: e.g., "super-\nconductivity" → "superconductivity"
        text = re.sub(r"-\s*\n\s*", "", text)

        # Replace multiple newlines with a single newline
        text = re.sub(r"\n{2,}", "\n\n", text)

        # Remove arXiv identifiers like 'arXiv:2301.12345'
        text = re.sub(r"arXiv:\d{4}\.\d{4,5}(v\d+)?", "", text)

        # Normalize spacing
        text = re.sub(r"[ \t]+", " ", text)  # collapse multiple spaces/tabs
        text = re.sub(r"\n[ \t]+", "\n", text)  # remove indentations

        # Replace newline characters with spaces
        text = re.sub(r"\n+", " ", text)

        return text.strip()


def arxiv_fetch_and_extract(
    category: str = "cond-mat.str-el",
    max_results: int = 100,
    data_folder: str = "data",
    fetched_arxiv_ids_file: str = "fetched_arxiv_ids.txt",
    batch_size: int = 10,
    loader: str = "pdfminer",
    logger: "BoundLoggerLazyProxy" = logger,
) -> list[ArxivPaper]:
    """
    Fetch papers from arXiv and extract text from the queried PDFs.

    This function initializes the `ArxivFetecher` and `TextExtractor` classes, following the workflow:

        -> Fetches the papers from arXiv and stores them in a list of `ArxivPaper` pydantic models.
        -> For each paper, it downloads the PDF storing it in `data_folder` and extracts the text from it.
        -> For each paper, it deletes the references section and cleans the extracted text.
        -> The text is stored in the `text` attribute of each `ArxivPaper` object.

    Args:
        category (str, optional): The arXiv category. Defaults to "cond-mat.str-el".
        max_results (int, optional): The maximum number of results to fetch from arXiv. A typical value when
            running the code would be 1000 (see https://info.arxiv.org/help/api/user-manual.html#3112-start-and-max_results-paging). Default is 5.
        data_folder (str, optional): The folder where to store the PDFs. Defaults to "data".
        fetched_arxiv_ids_file (str, optional): The file where to store the fetched arXiv IDs. Defaults to "fetched_arxiv_ids.txt".
        batch_size (int, optional): The number of papers to fetch in each request. Default is 10.
        loader (str, optional): The loader to use for extracting text from the PDF file. Defaults to "pdfminer".
        logger (BoundLoggerLazyProxy, optional): The logger to log messages. Defaults to logger.

    Returns:
        list[ArxivPaper]: A list of ArxivPaper objects with the text extracted from the PDFs.
    """
    # Initializes the `fetcher` (in this case from arXiv) and the `text_extractor` classes
    fetcher = ArxivFetcher(
        logger=logger,
        category=category,
        max_results=max_results,
        data_folder=data_folder,
        fetched_arxiv_ids_file=fetched_arxiv_ids_file,
    )
    text_extractor = TextExtractor(logger=logger)

    # Fetch the papers from arXiv and stores them in a list of `ArxivPaper` pydantic models
    papers = fetcher.fetch(batch_size=batch_size)
    logger.info(f"{max_results} papers fetched from arXiv, {category}.")

    # For each paper, it downloads the PDF storing it in `data_folder` and extracts the text from it
    # The text is stored in the `text` attribute of each `ArxivPaper` object
    for paper in papers:
        # ! note it is more efficient to download the PDF and extract the text because long papers cannot extract
        # ! the text on the fly and the connection times out
        # Download the PDF to `data_folder`
        pdf_path = fetcher.download_pdf(arxiv_paper=paper)

        # Extract text from the PDF
        text = text_extractor.get_text(pdf_path=pdf_path, loader=loader)
        if not text:
            logger.info("No text extracted from the PDF.")
            continue

        # Delete references and clean text
        # ! note that deleting the references must be done first
        text = text_extractor.delete_references(text=text)
        text = text_extractor.clean_text(text=text)

        # Store the text in the `text` attribute of the `ArxivPaper` object
        paper.text = text
        logger.info(f"Text extracted from {paper.id} and stored in model.")
    return papers


def download_pattern_papers(
    category: str = "cond-mat.str-el",
    max_results: int = 100,
    data_folder: str = "data",
    fetched_arxiv_ids_file: str = "fetched_arxiv_ids.txt",
    batch_size: int = 10,
    n_fetch_batches: int = 1,
    loader: str = "pdfminer",
    pattern: str = DMFT_PATTERN,
    logger: "BoundLoggerLazyProxy" = logger,
) -> tuple[list[Path], list[ArxivPaper]]:
    """
    Downloads locally all papers from arXiv that match a given pattern in their text.

    This function initializes the `ArxivFetecher` and `TextExtractor` classes, following the workflow:

        -> Fetches the papers from arXiv and stores them in a list of `ArxivPaper` pydantic models.
        -> For each paper, it downloads the PDF storing it in `data_folder` and extracts the text from it.
        -> For each paper, it deletes the references section and cleans the extracted text.
        -> The text is stored in the `text` attribute of each `ArxivPaper` object.

    Args:
        category (str, optional): The arXiv category. Defaults to "cond-mat.str-el".
        max_results (int, optional): The maximum number of results to fetch from arXiv. A typical value when
            running the code would be 1000 (see https://info.arxiv.org/help/api/user-manual.html#3112-start-and-max_results-paging). Default is 5.
        data_folder (str, optional): The folder where to store the PDFs. Defaults to "data".
        fetched_arxiv_ids_file (str, optional): The file where to store the fetched arXiv IDs. Defaults to "fetched_arxiv_ids.txt".
        batch_size (int, optional): The number of papers to fetch in each request. Default is 10.
        loader (str, optional): The loader to use for extracting text from the PDF file. Defaults to "pdfminer"
        n_fetch_batches (int, optional): The number of batches to fetch from arXiv. Defaults to 1.
        pattern (str, optional): The pattern to search for in the text. Defaults to DMFT_PATTERN.
        logger (BoundLoggerLazyProxy, optional): The logger to log messages. Defaults to logger.

    Returns:
        list[Path]: A list of paths to the downloaded PDFs that contain the pattern in their text.
    """
    # Initializes the `fetcher` (in this case from arXiv) and the `text_extractor` classes
    fetcher = ArxivFetcher(
        logger=logger,
        category=category,
        max_results=max_results,
        data_folder=data_folder,
        fetched_arxiv_ids_file=fetched_arxiv_ids_file,
    )
    text_extractor = TextExtractor(logger=logger)

    # Fetch the papers in a for loop, to avoid fetching too many papers at once
    files = []
    all_papers = []
    for _ in range(n_fetch_batches):
        papers = fetcher.fetch(batch_size=batch_size)
        for paper in papers:
            pdf_path = fetcher.download_pdf(arxiv_paper=paper)

            # Extract text from the PDF
            text = text_extractor.get_text(pdf_path=pdf_path, loader=loader)
            if not text:
                logger.info("No text extracted from the PDF.")
                continue

            # Deleting downloaded PDFs that do not match the pattern
            if not pattern.search(text):
                pdf_path.unlink()
                continue

            # Proceed with text cleaning and assigning to the `ArxivPaper` object
            text = text_extractor.delete_references(text=text)
            text = text_extractor.clean_text(text=text)
            paper.text = text

            # Store the local file paths and `ArxivPaper` objects in lists
            files.append(pdf_path)
            all_papers.append(paper)
    return files, all_papers
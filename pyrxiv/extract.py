import re
from pathlib import Path

from langchain_community.document_loaders import PDFMinerLoader, PyPDFLoader

from pyrxiv.logger import logger


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

    def _check_pdf_path(self, pdf_path: str | None | Path = ".") -> bool:
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

    def get_text(
        self, pdf_path: str | None | Path = ".", loader: str = "pdfminer"
    ) -> str:
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
            return ""
        if pdf_path is not None and isinstance(pdf_path, str):
            pdf_path = Path(pdf_path)
        filepath = pdf_path

        # Check if the loader is available
        if loader not in self.available_loaders.keys():
            self.logger.error(
                f"Loader {loader} not available. Available loaders: {self.available_loaders.keys()}"
            )
            return ""
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

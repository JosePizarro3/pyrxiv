from pathlib import Path

import requests

from pyrxiv.datamodel import ArxivPaper
from pyrxiv.logger import logger


class ArxivDownloader:
    def __init__(self, download_path: Path = Path("data"), **kwargs):
        """
        Initializes the ArxivDownloader with a specified download path and logger.

        Args:
            download_path (str): The path where the downloaded PDFs will be stored. Defaults to "data".
            logger: A logger instance for logging messages. If None, a default logger will be used.
        """
        download_path.mkdir(parents=True, exist_ok=True)
        self.download_path = download_path

        self.logger = kwargs.get("logger", logger)

        # ! an initial short paper is used to warm up the `requests` session connection
        # ! otherwise, long papers get stuck on `requests.get()` due to connection timeouts
        self.session = requests.Session()  # Reuse TCP connection
        self.session.head("http://arxiv.org/pdf/2502.10309v1", timeout=30)

    def download_pdf(self, arxiv_paper: ArxivPaper, write: bool = True) -> Path:
        """
        Download the PDF of the arXiv paper and stores it in the `download_path` folder using the `arxiv_paper.id` to name the PDF file.

        Args:
            arxiv_paper (ArxivPaper): The arXiv paper object to be queried and stored.
            write (bool): If True, the PDF will be written to the `data/` folder. Defaults to True.

        Returns:
            Path: The path to the downloaded PDF file.
        """

        pdf_path = Path("")
        try:
            response = self.session.get(arxiv_paper.pdf_url, stream=True, timeout=60)
            response.raise_for_status()

            pdf_path = self.download_path / f"{arxiv_paper.id}.pdf"
            if write:
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            self.logger.info(f"PDF downloaded: {pdf_path}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download PDF: {e}")
            pdf_path = None
        return pdf_path

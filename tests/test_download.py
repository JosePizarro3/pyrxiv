from pathlib import Path

import pytest

from pyrxiv.download import ArxivDownloader
from tests.conftest import generate_arxiv_paper


class TestArxivDownloader:
    @pytest.mark.parametrize(
        "id, pdf_path_result, log_msg",
        [
            (
                "1234.5678v1",
                None,
                {
                    "level": "error",
                    "event": "Failed to download PDF: 404 Client Error: Not Found for url: http://arxiv.org/pdf/1234.5678v1",
                },
            ),
            (
                "2502.10309v1",
                Path("tests/data/2502.10309v1.pdf"),
                {
                    "level": "info",
                    "event": "PDF downloaded: tests/data/2502.10309v1.pdf",
                },
            ),
        ],
    )
    def test_download_pdf(
        self, cleared_log_storage: list, id: str, pdf_path_result: str, log_msg: dict
    ):
        """Tests the `download_pdf` method of the `ArxivFetcher` class."""
        arxiv_paper = generate_arxiv_paper(id=id)
        arxiv_downloader = ArxivDownloader(download_path=Path("tests/data"))
        # no writing to file
        pdf_path = arxiv_downloader.download_pdf(arxiv_paper, write=False)
        assert pdf_path == pdf_path_result
        assert len(cleared_log_storage) == 1
        assert cleared_log_storage[0]["level"] == log_msg["level"]
        assert cleared_log_storage[0]["event"] == log_msg["event"]

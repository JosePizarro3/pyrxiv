import datetime
import os
from pathlib import Path

import pytest

from pyrxiv.datamodel import ArxivPaper
from pyrxiv.logger import log_storage

if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@pytest.fixture(autouse=True)
def cleared_log_storage():
    """Fixture to clear the log storage before each test."""
    log_storage.clear()
    yield log_storage


def generate_arxiv_paper(id: str = "1234.5678v1"):
    return ArxivPaper(
        id=id,
        url=f"http://arxiv.org/abs/{id}",
        pdf_url=f"http://arxiv.org/pdf/{id}",
        title="Test Title",
        summary="A summary or abstract.",
        authors=[],
        comment="",
        categories=[],
        updated=datetime.datetime(2024, 4, 25, 0, 0, tzinfo=datetime.timezone.utc),
        published=datetime.datetime(2024, 4, 25, 0, 0, tzinfo=datetime.timezone.utc),
    )

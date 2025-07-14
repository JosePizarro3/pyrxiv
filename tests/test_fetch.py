import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyrxiv.datamodel import ArxivPaper
from pyrxiv.fetch import ArxivFetcher


class TestArxivFetcher:
    """Tests for the `ArxivFetcher` class."""

    @pytest.mark.parametrize(
        "start_id, start_from_filepath, fetched_ids_file, result",
        [
            (None, False, "", [None, False, "tests/data/fetched_arxiv_ids.txt"]),
            (
                "1234.5678",
                False,
                "",
                ["1234.5678", True, "tests/data/fetched_arxiv_ids.txt"],
            ),
            (
                None,
                True,
                "fetched_arxiv_ids.txt",
                ["2507.02753v1", True, "tests/data/fetched_arxiv_ids.txt"],
            ),
        ],
    )
    def test_init(
        self,
        start_id: str,
        start_from_filepath: bool,
        fetched_ids_file: str,
        result: list,
    ):
        """Tests the initialization of the `ArxivFetcher` class for when the fetching starts from a specific ID."""
        arxiv_fetcher = ArxivFetcher(
            download_path=Path("tests/data"),
            start_id=start_id,
            start_from_filepath=start_from_filepath,
            fetched_ids_file=fetched_ids_file,
        )
        assert arxiv_fetcher.start_id == result[0]
        assert arxiv_fetcher.skip_newer_ids == result[1]
        assert str(arxiv_fetcher.fetched_ids_file) == result[2]

    @pytest.mark.parametrize(
        "fetched_ids_file, result",
        [
            # no file
            ("", ""),
            # correct
            ("tests/data/fetched_arxiv_ids.txt", "2507.02753v1"),
        ],
    )
    def test_last_fetched_id(self, fetched_ids_file: str, result: str):
        """Tests the `last_fetched_id` property of the `ArxivFetcher` class."""
        arxiv_fetcher = ArxivFetcher()
        assert (
            arxiv_fetcher._last_fetched_id(fetched_ids_file=fetched_ids_file) == result
        )

    @pytest.mark.parametrize(
        "comment, result",
        [
            # no comment
            ("", (None, None)),
            # no figures or pages
            ("5 pages", (None, None)),
            # no alphanumeric figures
            ("5 pages, figures", (None, None)),
            # correct
            ("5 pages, 3 figures", (5, 3)),
            # correct with different formats
            ("  12 pages ,  7 figures ", (12, 7)),
            ("This paper has 6 pages, 4 figures and 2 tables", (6, 4)),
            ("See also: 3 pages, 1 figures", (3, 1)),
            ("Includes 10pages,2figures", (10, 2)),
            # incorrect formats
            ("pages: 10, figures: 2", (None, None)),
        ],
    )
    def test_get_pages_and_figures(self, comment, result):
        """Tests the `_get_pages_and_figures` method of the `ArxivFetcher` class."""
        arxiv_fetcher = ArxivFetcher()
        assert arxiv_fetcher._get_pages_and_figures(comment) == result

    @pytest.mark.parametrize(
        "paper_id, reference_id, result",
        [
            # no ids
            ("", "", False),
            # one empty id
            ("2201.00001", "", False),
            # one invalid id
            ("2201.00001", "aaa", False),
            # no newer id
            ("2201.00001", "2201.00002", False),
            # same year and month, different version
            ("2201.00002", "2201.00001", True),
            # different year
            ("2201.00001", "1909.00002", True),
            # including version
            ("2201.00002v1", "1909.00002", True),
        ],
    )
    def test_is_newer_than(self, paper_id: str, reference_id: str, result: bool):
        """Tests the `is_newer_than` method of the `ArxivFetcher` class."""
        arxiv_fetcher = ArxivFetcher()
        assert arxiv_fetcher.is_newer_than(paper_id, reference_id) == result

    @pytest.mark.parametrize(
        "arxiv_response, log_msg, result",
        [
            # Empty response
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom"></feed>
                """,
                {"level": "info", "event": "No papers found in the response"},
                {},
            ),
            # Error in title when fetching
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                    <entry>
                        <title>Error when fetching the paper</title>
                    </entry>
                </feed>
                """,
                {"level": "error", "event": "Error fetching the paper"},
                {},
            ),
            # Id not in the correct format
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                    <entry>
                        <title>Test Paper Title</title>
                        <id>not a proper arxiv id</id>
                    </entry>
                </feed>
                """,
                {
                    "level": "error",
                    "event": "Paper without a valid URL id: not a proper arxiv id",
                },
                {},
            ),
            # Missing summary
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                    <entry>
                        <title>Test Paper Title</title>
                        <id>http://arxiv.org/abs/1234.5678v1</id>
                    </entry>
                </feed>
                """,
                {
                    "level": "error",
                    "event": "Paper http://arxiv.org/abs/1234.5678v1 without summary/abstract",
                },
                {},
            ),
            # Missing authors
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                    <entry>
                        <title>Test Paper Title</title>
                        <id>http://arxiv.org/abs/1234.5678v1</id>
                        <summary>This is a test abstract.</summary>
                    </entry>
                </feed>
                """,
                {},
                {
                    "id": "1234.5678v1",
                    "url": "http://arxiv.org/abs/1234.5678v1",
                    "pdf_url": "http://arxiv.org/pdf/1234.5678v1",
                    "updated": None,
                    "published": None,
                    "title": "Test Paper Title",
                    "summary": "This is a test abstract.",
                    "authors": [],
                    "comment": "",
                    "n_pages": None,
                    "n_figures": None,
                    "categories": [],
                    "text": "",
                },
            ),
            # Successful response
            (
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                    <entry>
                        <id>http://arxiv.org/abs/1234.5678v1</id>
                        <updated>2024-04-25T00:00:00Z</updated>
                        <published>2024-04-24T00:00:00Z</published>
                        <title>Test Paper Title</title>
                        <summary>This is a test abstract.</summary>
                        <author>
                            <name>John Doe</name>
                            <affiliation>University of Test</affiliation>
                        </author>
                        <category term="cond-mat.str-el"/>
                        <arxiv:comment xmlns:arxiv="http://arxiv.org/schemas/atom">10 pages, 2 figures</arxiv:comment>
                    </entry>
                </feed>
                """,
                {},
                {
                    "id": "1234.5678v1",
                    "url": "http://arxiv.org/abs/1234.5678v1",
                    "pdf_url": "http://arxiv.org/pdf/1234.5678v1",
                    "updated": datetime.datetime(
                        2024, 4, 25, 0, 0, tzinfo=datetime.timezone.utc
                    ),
                    "published": datetime.datetime(
                        2024, 4, 24, 0, 0, tzinfo=datetime.timezone.utc
                    ),
                    "title": "Test Paper Title",
                    "summary": "This is a test abstract.",
                    "authors": [
                        {
                            "name": "John Doe",
                            "affiliation": "University of Test",
                            "email": None,
                        }
                    ],
                    "comment": "10 pages, 2 figures",
                    "n_pages": 10,
                    "n_figures": 2,
                    "categories": ["cond-mat.str-el"],
                    "text": "",
                },
            ),
        ],
    )
    @patch("urllib.request.urlopen")
    def test_fetch(
        self,
        mock_urlopen: MagicMock,
        cleared_log_storage: list,
        arxiv_response: str,
        log_msg: dict,
        result: dict,
    ):
        """Tests the `fetch` method of the `ArxivFetcher` class."""
        mock_response = MagicMock()
        mock_response.read.return_value = arxiv_response.encode("utf-8")
        mock_urlopen.return_value = mock_response

        fetcher = ArxivFetcher(max_results=1, download_path=Path("tests/data"))
        papers = fetcher.fetch(write=False)
        if log_msg:
            assert len(cleared_log_storage) in [1, 2]
            assert cleared_log_storage[0]["level"] == log_msg["level"]
            assert cleared_log_storage[0]["event"] == log_msg["event"]
        if papers and all(isinstance(p, ArxivPaper) for p in papers):
            assert papers[0].model_dump() == result

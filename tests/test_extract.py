import pytest

from pyrxiv.extract import TextExtractor


class TestTextExtractor:
    @pytest.mark.parametrize(
        "pdf_path, result",
        [
            # no path
            ("", False),
            # no valid path
            ("sample", False),
            # no pdf file
            ("tests/data/no_pdf_paper.txt", False),
            # successful
            ("tests/data/sample.pdf", True),
        ],
    )
    def test__check_pdf_path(self, pdf_path: str, result: bool):
        """Tests the `_check_pdf_path` method of the `TextExtractor` class."""
        check = TextExtractor()._check_pdf_path(pdf_path=pdf_path)
        assert check == result

    @pytest.mark.parametrize(
        "pdf_path, loader, length_text",
        [
            # no path
            ("", "pypdf", 0),
            # no valid path
            ("sample", "pypdf", 0),
            # no pdf file
            ("tests/data/no_pdf_paper.txt", "pypdf", 0),
            # no pdf loader implemented
            ("tests/data/sample.pdf", "no-pdf-loader-implemented", 0),
            # successful with pypdf
            ("tests/data/sample.pdf", "pypdf", 2876),
            # successful with pdfminer
            ("tests/data/sample.pdf", "pdfminer", 3118),
        ],
    )
    def test_get_text(self, pdf_path: str, loader: str, length_text: int):
        """Tests the `with_pdfminer` method of the `TextExtractor` class."""
        text = TextExtractor().get_text(pdf_path=pdf_path, loader=loader)
        assert len(text) == length_text

    @pytest.mark.parametrize(
        "input_text, expected_output",
        [
            # No references section
            (
                "This is some main body text.\nIntroduction\nMethods\nResults\nConclusion\n",
                "This is some main body text.\nIntroduction\nMethods\nResults\nConclusion\n",
            ),
            # References found, no Supplemental Material
            (
                "Main text content.\nReferences\n[1] A. Author, Title, 2024.\n[2] B. Author, Title, 2023.\n",
                "Main text content.",
            ),
            # Bibliography found
            ("Main body.\nBibliography\n[1] C. Author, 2022.\n", "Main body."),
            # References and Supplemental Material delimiters found
            (
                "Body text.\nReferences\n[1] Author 1\n[2] Author 2\nSupplemental Material:\nAdditional stuff.\n",
                "Body text.\nSupplemental Material:\nAdditional stuff.\n",
            ),
            # No references section but has [1] directly
            (
                "Introduction\n[1] Z. Researcher\n[2] X. Scientist\nConclusion\n",
                "Introduction",
            ),
        ],
    )
    def test_delete_references(self, input_text: str, expected_output: str):
        """Tests the `delete_references` method of the `TextExtractor` class."""
        output = TextExtractor().delete_references(text=input_text)
        assert output == expected_output

    def test_clean_text(self):
        """Tests the `clean_text` method of the `TextExtractor` class."""
        extractor = TextExtractor()
        old_text = extractor.get_text(pdf_path="tests/data/sample.pdf")
        text = extractor.clean_text(text=old_text)
        assert (
            old_text[:100]
            == "Sample PDF\nThis is a simple PDF ﬁle. Fun fun fun.\n\nLorem ipsum dolor  sit amet,  consectetuer  adipi"
        )
        assert (
            text[:100]
            == "Sample PDF This is a simple PDF ﬁle. Fun fun fun. Lorem ipsum dolor sit amet, consectetuer adipiscin"
        )

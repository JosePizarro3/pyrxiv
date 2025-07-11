class ArxivDownloader:
    def download_pdf(self, arxiv_paper: ArxivPaper, write: bool = True) -> Path:
        """
        Download the PDF of the arXiv paper and stores it in the `data` folder using the `arxiv_paper.id` to name the PDF file.

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

            pdf_path = self.data_folder / f"{arxiv_paper.id}.pdf"

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

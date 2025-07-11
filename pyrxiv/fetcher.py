class ArxivFetcher:
    """
    Fetch papers from arXiv and extract text from the queried PDFs.
    """

    def __init__(
        self,
        category: str = "cond-mat.str-el",
        max_results: int = 100,
        data_folder: str = "data",
        fetched_arxiv_ids_file: str = "fetched_arxiv_ids.txt",
        **kwargs,
    ):
        """
        Initialize the ArxivFetcher class.
        This class fetches papers from arXiv and extracts text from the queried PDFs.
        It uses the `requests` library to fetch the papers and the `xmltodict` library to parse the XML response.
        It also uses the `PyPDFLoader` and `PDFMinerLoader` from LangChain to extract text from the PDFs.

        Args:
            category (str, optional): The arXiv category to fetch papers from. Default is "cond-mat.str-el".
            max_results (int, optional): The maximum number of results to fetch from arXiv. A typical value when
                running the code would be 1000 (see https://info.arxiv.org/help/api/user-manual.html#3112-start-and-max_results-paging). Default is 5.
            data_folder (str, optional): The folder where to store the PDFs and other data. Default is "data".
            fetched_arxiv_ids_file (str, optional): The file where to store the fetched arXiv IDs. Default is "fetched_arxiv_ids.txt".
        """
        self.category = category
        self.max_results = max_results

        # check if `data_folder` exists, and if not, create it
        Path(data_folder).mkdir(parents=True, exist_ok=True)
        self.data_folder = Path(data_folder)
        # file to store fetched arXiv IDs
        self.fetched_ids_file = self.data_folder / fetched_arxiv_ids_file

        self.logger = kwargs.get("logger", logger)
        self.session = requests.Session()  # Reuse TCP connection
        # ! an initial short paper is used to warm up the `requests` session connection
        # ! otherwise, long papers get stuck on `requests.get()` due to connection timeouts
        self.session.head("http://arxiv.org/pdf/2502.10309v1", timeout=30)

    @property
    def fetched_ids(self) -> set[str]:
        """
        Get the set of fetched arXiv IDs from the `fetched_arxiv_ids.txt` file.

        Returns:
            set[str]: A set of fetched arXiv IDs.
        """
        if not self.fetched_ids_file.exists():
            return set()
        with open(self.fetched_ids_file) as f:
            return set(line.strip() for line in f if line.strip())

    def fetch(self, batch_size: int = 100) -> list:
        """
        Fetch new papers from arXiv, skipping already fetched ones, and stores their metadata in an `ArxivPaper`
        pydantic models. New fetched arXiv IDs will be appended to `data/fetched_arxiv_ids.txt`.

        Args:
            batch_size (int, optional): The number of papers to fetch in each request. Default is 100.

        Returns:
            list: A list of `ArxivPaper` objects with the metadata of the papers fetched from arXiv.
        """

        def _get_pages_and_figures(comment: str) -> tuple[int | None, int | None]:
            """
            Gets the number of pages and figures from the comment of the arXiv paper.

            Args:
                comment (str): A string containing the comment of the arXiv paper.

            Returns:
                tuple[int | None, int | None]: A tuple containing the number of pages and figures.
                    If not found, returns (None, None).
            """
            pattern = r"(\d+) *pages*, *(\d+) *figures*"
            match = re.search(pattern, comment)
            if match:
                n_pages, n_figures = match.groups()
                return int(n_pages), int(n_figures)
            return None, None

        # Load already fetched IDs into a set
        fetched_ids = self.fetched_ids

        new_papers = []
        start_index = 0
        while len(new_papers) < self.max_results:
            remaining = self.max_results - len(new_papers)  # remaining papers to fetch
            current_batch_size = min(batch_size, remaining)  # current batch to fetch

            # Fetch request from arXiv API and parsing the XML response
            url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=cat:{self.category}&start={start_index}&max_results={current_batch_size}&"
                f"sortBy=submittedDate&sortOrder=descending"
            )

            request = urllib.request.urlopen(url)
            data = request.read().decode("utf-8")
            data_dict = xmltodict.parse(data)

            # Extracting papers from the XML response
            papers = data_dict.get("feed", {}).get("entry", [])
            if not papers:
                self.logger.info("No papers found in the response")
                return []
            # In case `max_results` is 1, the response is not a list
            if not isinstance(papers, list):
                papers = [papers]

            # Store papers object ArxivPaper in a list
            for paper in papers:
                # If there is an error in the fetching, skip the paper
                if "Error" in paper.get("title", ""):
                    self.logger.error("Error fetching the paper")
                    new_papers = papers
                    continue

                # If there is no `id`, skip the paper
                url_id = paper.get("id")
                if not url_id or "arxiv.org" not in url_id:
                    self.logger.error(f"Paper without a valid URL id: {url_id}")
                    new_papers = papers
                    continue

                # If there is no `summary`, skip the paper
                summary = paper.get("summary")
                if not summary:
                    self.logger.error(f"Paper {url_id} without summary/abstract")
                    new_papers = papers
                    continue

                # Getting arXiv `id`, and skipping if already fetched
                arxiv_id = url_id.split("/")[-1].replace(".pdf", "")
                if arxiv_id in fetched_ids:
                    continue

                # Extracting `authors` from the XML response
                paper_authors = paper.get("author", [])
                if not isinstance(paper_authors, list):
                    paper_authors = [paper_authors]
                authors = [
                    Author(
                        name=author.get("name"), affiliation=author.get("affiliation")
                    )
                    for author in paper_authors
                ]
                if not authors:
                    self.logger.info("\tPaper without authors.")

                # Extracting `categories` from the XML response
                arxiv_categories = paper.get("category", [])
                if not isinstance(arxiv_categories, list):
                    categories = [arxiv_categories.get("@term")]
                else:
                    categories = [
                        category.get("@term") for category in arxiv_categories
                    ]

                # Extracting pages and figures from the comment
                comment = paper.get("arxiv:comment", {}).get("#text", "")
                n_pages, n_figures = _get_pages_and_figures(comment)

                # Storing the ArxivPaper object in the list
                new_papers.append(
                    ArxivPaper(
                        id=arxiv_id,
                        url=url_id,
                        pdf_url=url_id.replace("abs", "pdf"),
                        updated=paper.get("updated"),
                        published=paper.get("published"),
                        title=paper.get("title"),
                        summary=summary,
                        authors=authors,
                        comment=comment,
                        n_pages=n_pages,
                        n_figures=n_figures,
                        categories=categories,
                    )
                )

                fetched_ids.add(arxiv_id)
                self.logger.info(f"Paper {arxiv_id} fetched from arXiv.")

                if len(new_papers) >= self.max_results:
                    break

            start_index += batch_size

        # Save newly fetched IDs if these are all `ArxivPaper` objects and contain the `id` attribute
        if all(isinstance(p, ArxivPaper) and p.id is not None for p in new_papers):
            with open(self.fetched_ids_file, "a") as f:
                for paper in new_papers:
                    f.write(f"{paper.id}\n")

        # If not all papers are `ArxivPaper` objects, return an empty list
        if not all(isinstance(p, ArxivPaper) for p in new_papers):
            return []

        return new_papers

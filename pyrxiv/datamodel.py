import datetime

from pydantic import BaseModel, Field


class Author(BaseModel):
    name: str = Field(..., description="The name of the author.")
    affiliation: str | None = Field(None, description="The affiliation of the author.")
    email: str | None = Field(None, description="The email of the author.")


class ArxivPaper(BaseModel):
    """The data model for the arXiv paper metadata."""

    id: str = Field(
        ...,
        description="The arXiv ID of the paper. Example: 2502.1234v1.",
    )

    url: str = Field(
        ...,
        description="The URL of the arXiv paper. Example: http://arxiv.org/abs/2502.1234v1",
    )

    pdf_url: str = Field(
        ...,
        description="The URL of the PDF of the arXiv paper. Example: http://arxiv.org/pdf/2502.1234v1",
    )

    updated: datetime.datetime | None = Field(
        None, description="The date when the paper was updated."
    )

    published: datetime.datetime | None = Field(
        None, description="The date when the paper was published."
    )

    title: str = Field(..., description="The title of the arXiv paper.")

    summary: str = Field(..., description="The summary of the arXiv paper.")

    authors: list[Author]

    comment: str | None = Field(None, description="The comment of the arXiv paper.")

    n_pages: int | None = Field(
        None, description="The number of pages of the arXiv paper."
    )

    n_figures: int | None = Field(
        None, description="The number of figures of the arXiv paper."
    )

    categories: list[str] = Field(
        ...,
        description="The categories of the arXiv paper. Example: ['cond-mat.str-el', 'cond-mat.mtrl-sci'].",
    )

    text: str = Field(
        default="",
        description="The text of the arXiv paper. It is the text of the paper after cleaning and deleting references.",
    )

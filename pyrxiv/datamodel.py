import datetime

import h5py
import numpy as np
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

    pdf_loader: str | None = Field(
        default=None,
        description="The name of the PDF loader used to extract the text from the PDF.",
    )

    text: str = Field(
        default="",
        description="The text of the arXiv paper. It is the text of the paper after cleaning and deleting references.",
    )

    def to_hdf5(self, hdf_file: h5py.File) -> h5py.Group:
        """
        Stores the ArxivPaper metadata and text dataset in an HDF5 file.

        Args:
            hdf_file (h5py.File): The HDF5 file to store the metadata and text.

        Returns:
            h5py.Group: The group in the HDF5 file where the metadata and text are stored.
        """
        group = hdf_file.require_group(self.id)
        sub_group = group.require_group("arxiv_paper")
        for key in self.model_fields:
            if key == "id":
                continue
            value = getattr(self, key)
            if key in ["updated", "published"]:
                value = getattr(self, key).isoformat() if getattr(self, key) else None
            if key == "authors":
                value = [author.name for author in self.authors]

            # Skip none values
            if value is None:
                continue

            # overwrite existing dataset for `text`
            if key == "text":
                if key in sub_group:
                    del sub_group[key]
                sub_group.create_dataset(key, data=value.encode("utf-8"))
                continue
            # handle lists of strings
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                if key in sub_group:
                    del sub_group[key]
                sub_group.create_dataset(key, data=value)
                continue
            # all other attributes
            sub_group.attrs[key] = value
        return group

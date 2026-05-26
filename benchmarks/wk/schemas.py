from typing import Optional
from pydantic import BaseModel


class Term(BaseModel):
    term: str
    pref_label: str
    alt_labels: list[str]
    definition: str
    rank: int
    lang: str
    categories: Optional[list[str]] = None


class Section(BaseModel):
    section_label: str
    section_name: str
    seq_idx: int
    section_content: Optional[str] = None
    paragraphs: Optional[list[str]] = None
    num_paragraphs: Optional[int] = None
    num_chars: Optional[int] = None
    children: Optional[list["Section"]] = None


class ParseResult(BaseModel):
    root: list[Section]

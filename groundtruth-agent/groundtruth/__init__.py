"""groundtruth: an agent that answers questions from your PDFs, with citations.

Read in this order: config -> parse -> chunk -> index -> tool -> agent -> ingest.
"""

from .agent import Answer, Citation, ask
from .index import BM25Index
from .ingest import ingest_pdf
from .parse import make_claude_describer

__all__ = ["Answer", "Citation", "ask", "BM25Index", "ingest_pdf", "make_claude_describer"]

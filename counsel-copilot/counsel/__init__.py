"""counsel-copilot: a paralegal-style crew over your legal documents.

Reading order: config → ingest → search → llm → clause_finder → researcher
→ drafter → crew → ../app.py
"""

from .config import NOT_LEGAL_ADVICE, SETTINGS
from .crew import CrewResult, run_crew
from .ingest import Chunk, chunk_document
from .llm import make_client
from .search import ClauseIndex

__all__ = ["NOT_LEGAL_ADVICE", "SETTINGS", "Chunk", "ClauseIndex", "CrewResult",
           "chunk_document", "make_client", "run_crew"]

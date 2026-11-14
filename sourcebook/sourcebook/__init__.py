"""Sourcebook: a from-scratch, NotebookLM-style research notebook.

Read the modules in pipeline order: sources -> chunk -> retrieve -> memory ->
answer -> notebook -> podcast -> tts. `app.py` is the Streamlit UI on top.
"""

from .answer import Answer, Citation, ask
from .notebook import Notebook, list_notebooks
from .podcast import Script, generate_script
from .sources import Source, load_pdf, load_text, load_url

__all__ = [
    "Answer", "Citation", "ask", "Notebook", "list_notebooks", "Script",
    "generate_script", "Source", "load_pdf", "load_text", "load_url",
]

"""Storage and retrieval: the three memories in SQLite + sqlite-vec."""

from pathlib import Path

import pytest

from vault import HashEmbedder, Vault
from vault.ingest import chunk_file, chunk_text
from vault.sqlite_store import SqliteVecStore

DOCS = Path(__file__).parent.parent / "sample_docs"


def _ingest_samples(vault):
    for path in sorted(DOCS.glob("*.md")):
        vault.ingest(path.name, path.read_bytes())


# --- ingest ------------------------------------------------------------------------

def test_chunks_overlap_and_cover_the_text():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), words=180, overlap=40)
    assert [len(c.text.split()) for c in chunks] == [180, 180, 180, 80]
    # Consecutive chunks share 40 words, and the last word is included.
    assert chunks[0].text.split()[-40:] == chunks[1].text.split()[:40]
    assert chunks[-1].text.split()[-1] == "w499"
    assert [c.location for c in chunks] == ["§1", "§2", "§3", "§4"]


def _tiny_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one line of text per page, built by hand."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objs)} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def test_pdf_pages_become_chunk_locations():
    chunks = chunk_file("report.pdf", _tiny_pdf(["Alpha page text", "Beta page text"]))
    assert [(c.location, c.text) for c in chunks] == [("p. 1", "Alpha page text"), ("p. 2", "Beta page text")]


def test_unsupported_and_empty_files_are_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        chunk_file("image.png", b"\x89PNG")
    with pytest.raises(ValueError, match="empty"):
        chunk_file("blank.txt", b"   \n ")


# --- memory 1: knowledge base ---------------------------------------------------------

def test_search_finds_the_right_passage(vault):
    _ingest_samples(vault)
    hit = vault.search("turbidity UV filter cartridge tanks")[0]
    assert hit.doc_name == "kestrel-station-handbook.md"
    assert "UV filter cartridge" in hit.text
    hit = vault.search("hand pollination tomatoes toothbrush")[0]
    assert hit.doc_name == "greenhouse-notes.md"
    scores = [h.score for h in vault.search("tomatoes pollination")]
    assert scores == sorted(scores, reverse=True)


def test_same_bytes_are_stored_once(vault):
    data = (DOCS / "greenhouse-notes.md").read_bytes()
    doc, created = vault.ingest("greenhouse-notes.md", data)
    again, created_again = vault.ingest("renamed.md", data)
    assert created and not created_again and again.id == doc.id
    assert vault.store.stats()["documents"] == 1


def test_deleting_a_document_removes_its_vectors(vault):
    _ingest_samples(vault)
    handbook = next(d for d in vault.store.list_documents() if d.name.startswith("kestrel"))
    vault.store.delete_document(handbook.id)
    assert {h.doc_name for h in vault.search("battery solar water visitors", k=20)} == {"greenhouse-notes.md"}
    n_vectors = vault.store.db.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
    assert n_vectors == vault.store.stats()["chunks"]


def test_empty_knowledge_base_returns_nothing(vault):
    assert vault.search("anything") == []


# --- memory 2: conversation history -----------------------------------------------------

def test_threads_keep_ordered_messages_and_limit_returns_the_newest(vault):
    s = vault.store
    a, b = s.create_thread("A"), s.create_thread("B")
    for i in range(5):
        s.add_message(a.id, "user" if i % 2 == 0 else "assistant", f"a{i}")
    s.add_message(b.id, "user", "b0", trace=[{"tool": "recall", "input": {"query": "x"}, "notes": []}])
    assert [m.content for m in s.get_messages(a.id)] == ["a0", "a1", "a2", "a3", "a4"]
    assert [m.content for m in s.get_messages(a.id, limit=2)] == ["a3", "a4"]
    assert s.get_messages(b.id)[0].trace[0]["tool"] == "recall"
    # Most recently active first.
    assert [t.id for t in s.list_threads()][0] == b.id
    s.delete_thread(a.id)
    assert s.get_messages(a.id) == [] and [t.title for t in s.list_threads()] == ["B"]


# --- memory 3: notes -------------------------------------------------------------------

def test_notes_dedupe_search_and_delete(vault):
    t = vault.store.create_thread("t")
    n1, c1 = vault.remember("User prefers answers in metric units", "preference", t.id)
    n2, c2 = vault.remember("User prefers answers in metric units", "preference", t.id)
    n3, c3 = vault.remember("User's project deadline is March 3", "task", None)
    assert c1 and not c2 and n2.id == n1.id and c3
    top = vault.recall("which units does the user prefer")[0]
    assert top.id == n1.id and top.score is not None
    vault.store.delete_note(n1.id)
    assert [n.id for n in vault.store.list_notes()] == [n3.id]
    assert all(n.id != n1.id for n in vault.recall("metric units"))


def test_notes_survive_their_thread(vault):
    t = vault.store.create_thread("t")
    vault.remember("User lives in Lagos", "fact", t.id)
    vault.store.delete_thread(t.id)
    assert vault.store.list_notes()[0].content == "User lives in Lagos"


# --- the database file itself ----------------------------------------------------------

def test_everything_persists_across_reopen(tmp_path):
    emb = HashEmbedder()
    v = Vault(SqliteVecStore(tmp_path / "v.db", emb.name, emb.dim), emb)
    _ingest_samples(v)
    t = v.store.create_thread("kept")
    v.store.add_message(t.id, "user", "hello")
    v.remember("User is a hydrologist", "fact", t.id)
    v.store.close()

    v2 = Vault(SqliteVecStore(tmp_path / "v.db", emb.name, emb.dim), emb)
    assert v2.store.stats() == {"documents": 2, "chunks": v2.store.stats()["chunks"],
                                "threads": 1, "messages": 1, "notes": 1}
    assert v2.search("satellite phone Friday")[0].doc_name == "kestrel-station-handbook.md"
    assert v2.recall("what is the user's job")[0].content == "User is a hydrologist"


def test_a_database_refuses_a_different_embedder(tmp_path):
    SqliteVecStore(tmp_path / "v.db", "hash:256", 256).close()
    with pytest.raises(RuntimeError, match="built with embedder"):
        SqliteVecStore(tmp_path / "v.db", "fastembed:bge", 384)

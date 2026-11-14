import pytest

from vault import HashEmbedder, Vault
from vault.sqlite_store import SqliteVecStore


@pytest.fixture
def vault(tmp_path):
    """A fresh vault on a temp SQLite file, with the no-download hash embedder."""
    emb = HashEmbedder()
    v = Vault(SqliteVecStore(tmp_path / "vault.db", emb.name, emb.dim), emb)
    yield v
    v.store.close()

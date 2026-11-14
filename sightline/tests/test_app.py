"""The Streamlit app itself, run headless (Streamlit AppTest) in demo mode."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from sight import config

APP = Path(__file__).resolve().parent.parent / "app.py"


def start(monkeypatch, tmp_path) -> AppTest:
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)       # no key -> demo mode on
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # ignore a real .env
    monkeypatch.setattr(config, "LOG_PATH", tmp_path / "decisions.jsonl")
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_sample_image_flow_queue_and_history(monkeypatch, tmp_path):
    at = start(monkeypatch, tmp_path)
    assert at.toggle[0].value is True                            # demo mode by default
    at.sidebar.radio[0].set_value("Sample image").run()
    at.button[0].click().run()                                   # "Analyse the sample desk"
    assert not at.exception

    assert at.metric[0].value == "laptop"                        # classified
    queue_tab = at.tabs[1]
    assert queue_tab.label.startswith("🟠 Review queue (")
    n_queued = int(queue_tab.label.split("(")[1].rstrip(")"))
    assert n_queued == 1                                         # the book box (confidence 0.48)

    # Resolve one queued item: it leaves the queue and lands in the log.
    save = next(b for b in at.button if b.key and b.key.startswith("save_"))
    save.click().run()
    assert not at.exception
    assert at.tabs[1].label == f"🟠 Review queue ({n_queued - 1})"
    rows = [l for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert any('"type": "review"' in r for r in rows)

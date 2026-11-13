"""Render the Streamlit app headless with a finished (fake-Claude) analysis.

AppTest can't drive the file uploader, so we run the pipeline with the fake
client and hand the result to the app through session_state, the same place
the "Analyse paper" button stores it.
"""

from streamlit.testing.v1 import AppTest

from paperlab.pipeline import analyse_pdf


def test_app_renders_report(sample_pdf, fake_client):
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.session_state["client"] = fake_client
    at.session_state["result"] = analyse_pdf(sample_pdf, fake_client)
    at.run()

    assert not at.exception
    assert at.title[0].value == "Soil Enzymes Under Drought"
    assert [m.value for m in at.metric] == ["1", "1", "1"]   # verified / wrong page / not found
    labels = [t.label for t in at.tabs]
    assert labels == ["📋 Report", "🖼️ Figures", "📝 Analyst drafts", "🔎 Trace"]
    captions = " ".join(c.value for c in at.caption)
    assert "actually on p. 2" in captions and "not found in the PDF" in captions


def test_app_idle_without_result():
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.session_state["client"] = object()   # never called
    at.run()
    assert not at.exception and not at.title   # waits for an upload

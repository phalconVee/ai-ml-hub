"""Smoke-test the Streamlit page headlessly (streamlit.testing AppTest).

We put a FakeClient into session_state before the first run, so app.py uses it
instead of building a real client. Then: load the sample lease, ask a question,
and check the notice, the cited answer and the source list are on the page.
"""

from streamlit.testing.v1 import AppTest

from tests.test_crew import LATE, FakeClient, cite, msg, report, text


def test_ask_about_sample_lease():
    fake = FakeClient([
        msg("tool_use", report([LATE])),
        msg("end_turn", text("The late fee is 10%.", [cite(LATE, "ten percent (10%)",
                                                           "sample_lease.pdf, p.1, 3. Late Payment")])),
    ])
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.session_state["client"] = fake
    at.run()
    assert "Not legal advice" in at.warning[0].value

    at.button[0].click().run()                      # "Load the sample lease"
    assert "sample_lease.pdf" in " ".join(m.value for m in at.markdown)

    at.chat_input[0].set_value("What is the late fee?").run()
    assert not at.exception and not at.error
    page = " ".join(m.value for m in at.markdown)
    assert "The late fee is 10%. [1]" in page
    assert "3. Late Payment" in page                # the source list
    assert len(fake.requests) == 2                   # finder + drafter, no web

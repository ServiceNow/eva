from eva.user_simulator.cascade.stt import TranscriptBuffer


def test_partial_updates_replace_the_in_flight_text():
    buffer = TranscriptBuffer()

    buffer.apply_partial("Let me check")
    buffer.apply_partial("Let me check that for")

    assert buffer.in_flight == "Let me check that for"
    assert buffer.committed == ""


def test_commit_appends_and_clears_the_partial():
    buffer = TranscriptBuffer()
    buffer.apply_partial("Let me check that")

    buffer.commit("Let me check that for you.")

    assert buffer.committed == "Let me check that for you."
    assert buffer.in_flight == ""


def test_successive_commits_accumulate_with_spaces():
    buffer = TranscriptBuffer()

    buffer.commit("First sentence.")
    buffer.commit("Second sentence.")

    assert buffer.committed == "First sentence. Second sentence."


def test_take_committed_drains_the_buffer():
    buffer = TranscriptBuffer()
    buffer.commit("All done.")

    assert buffer.take_committed() == "All done."
    assert buffer.committed == ""

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


def test_current_text_marks_the_in_flight_partial_as_incomplete():
    buffer = TranscriptBuffer()
    buffer.commit("I found your order.")
    buffer.apply_partial("It includes a keyboa")

    assert buffer.current_text() == "I found your order. It includes a keyboa [CURRENTLY SPEAKING, INCOMPLETE]"


def test_current_text_omits_the_marker_when_nothing_is_in_flight():
    buffer = TranscriptBuffer()
    buffer.commit("I found your order.")

    assert buffer.current_text() == "I found your order."


def test_current_text_does_not_consume_the_committed_text():
    buffer = TranscriptBuffer()
    buffer.commit("hello")

    buffer.current_text()

    assert buffer.take_committed() == "hello"


def test_heard_text_never_carries_the_prompt_marker_into_the_transcript():
    buffer = TranscriptBuffer()
    buffer.commit("I found your order.")
    buffer.apply_partial("It includes a keyboa")

    assert buffer.heard_text() == "I found your order. It includes a keyboa"

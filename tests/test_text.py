from silentdict.text import accept_rewrite, align, corpus_wer, normalize, sentence_case, wer


def test_normalize_matches_lrs3_style():
    assert normalize("Don’t stop-believing, OK?") == "DON'T STOP BELIEVING OK"


def test_wer_counts_word_edits():
    assert wer("see you at noon", "see you at moon") == (1, 4)
    assert wer("see you", "") == (2, 2)
    assert corpus_wer([("a b c d", "a b c d"), ("a b", "a")]) == 1 / 6


def test_align_marks_each_word():
    ops = align("in any case we can", "any case we can do")
    assert [o["op"] for o in ops] == ["del", "ok", "ok", "ok", "ok", "ins"]
    ops = align("send the slides", "send the sides")
    assert ops[-1] == {"op": "sub", "ref": "slides", "hyp": "sides"}


def test_sentence_case():
    assert sentence_case("i'll be there in five minutes") == "I'll be there in five minutes."
    assert sentence_case("") == ""
    assert sentence_case("is it ready?") == "Is it ready?"


def test_llm_rewrite_guard():
    offered = ["i'll be there in fine minutes", "i'll be there in five minutes"]
    assert accept_rewrite("I'll be there in five minutes.", offered)       # an offered candidate
    assert accept_rewrite("I'll be there in nine minutes.", offered)       # one word changed
    assert not accept_rewrite("I'll be there soon, see you.", offered)     # new content
    assert not accept_rewrite("I will be there in nine minutes.", offered)  # two words changed

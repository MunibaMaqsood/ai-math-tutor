from src.study_tools import MCQ, clean_mcqs, strip_dollars


def test_strip_dollars():
    assert strip_dollars("$ax^2+bx+c=0$") == "ax^2+bx+c=0"
    assert strip_dollars("  a+b ") == "a+b"


def test_clean_mcqs_drops_bad_items():
    good = MCQ(question="q", options=["a", "b", "c", "d"], answer_index=2, explanation="e")
    bad_count = MCQ(question="q", options=["a", "b"], answer_index=0, explanation="e")
    bad_index = MCQ(question="q", options=["a", "b", "c", "d"], answer_index=7, explanation="e")
    assert clean_mcqs([good, bad_count, bad_index]) == [good]

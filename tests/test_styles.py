import pytest

from docparse.styles import transfer_aligned_styles
from docparse.textlayer import Line, Span


def _line(*spans):
    return Line(list(spans), [0, 0, 300, 20])


def _span(text, **styles):
    return Span(text, [0, 0, 100, 20], 10, **styles)


def test_styles_correct_repeated_occurrence():
    line = _line(_span("total "), _span("total", bold=True))
    assert transfer_aligned_styles("total total", [line]) == "total **total**"


def test_abstains_when_repeated_occurrence_is_missing():
    line = _line(_span("total "), _span("total", bold=True))
    assert transfer_aligned_styles("total", [line]) == "total"


def test_single_character_superscript_stays_inside_word():
    line = _line(_span("x"), _span("2", sup=True), _span(" increases"))
    assert transfer_aligned_styles("x2 increases", [line]) == "x<sup>2</sup> increases"


def test_keeps_target_characters_when_native_text_differs():
    line = _line(_span("The "), _span("increase", bold=True), _span(" was large."))
    assert transfer_aligned_styles("The increases were large.", [line]) == "The **increases** were large."


def test_preserves_existing_bold_without_duplicate_markers():
    line = _line(_span("A "), _span("bold word", bold=True), _span(" remains."))
    assert transfer_aligned_styles("A **bold word** remains.", [line]) == "A **bold word** remains."


def test_protects_inline_math_and_code():
    line = _line(_span("x2", bold=True, sup=True), _span(" and "), _span("sample", bold=True, sup=True))
    target = "$x2$ and `sample`"
    assert transfer_aligned_styles(target, [line]) == target


def test_preserves_markup_while_adding_subword_superscript():
    line = _line(_span("x", bold=True), _span("2", bold=True, sup=True))
    assert transfer_aligned_styles("**x2**", [line]) == "**x<sup>2</sup>**"


def test_ligature_alignment_preserves_target_spelling():
    line = _line(_span("The "), _span("in\ufb02ation", bold=True), _span(" rate."))
    assert transfer_aligned_styles("The inflation rate.", [line]) == "The **inflation** rate."


def test_abstains_on_unrelated_text():
    line = _line(_span("Annual revenue", bold=True))
    assert transfer_aligned_styles("Unrelated sentence", [line]) == "Unrelated sentence"


def test_style_intersections_are_properly_nested():
    line = _line(_span("first ", bold=True), _span("middle", bold=True, underline=True), _span(" last", underline=True))
    assert transfer_aligned_styles("first middle last", [line]) == "**first <u>middle</u>**<u> last</u>"


def test_extends_partial_existing_bold_without_duplicate_markers():
    line = _line(_span("one two", bold=True))
    assert transfer_aligned_styles("**one** two", [line]) == "**one** **two**"


def test_crossing_existing_style_is_split_into_valid_containers():
    line = _line(_span("one two", italic=True))
    assert transfer_aligned_styles("**one** two", [line]) == "***one*** *two*"


def test_styles_prose_on_both_sides_of_protected_code():
    line = _line(_span("before code after", bold=True))
    assert transfer_aligned_styles("before `code` after", [line]) == "**before** `code` **after**"


@pytest.mark.parametrize("target", ["***one***", "___one___"])
def test_preserves_combined_bold_and_italic(target):
    line = _line(_span("one", bold=True, italic=True))
    assert transfer_aligned_styles(target, [line]) == target


@pytest.mark.parametrize("target,style", [("__one__", "bold"), ("_one_", "italic")])
def test_preserves_underscore_emphasis(target, style):
    line = _line(_span("one", **{style: True}))
    assert transfer_aligned_styles(target, [line]) == target


def test_does_not_expand_native_partial_word_bolding():
    line = _line(_span("AB", bold=True), _span("CD"))
    assert transfer_aligned_styles("ABCD", [line]) == "**AB**CD"


@pytest.mark.parametrize("target", ["before ``code`` after", "before <code>code</code> after", r"before \(code\) after"])
def test_adds_styles_around_other_protected_syntax(target):
    line = _line(_span("before code after", bold=True))
    expected = target.replace("before", "**before**").replace("after", "**after**")
    assert transfer_aligned_styles(target, [line]) == expected


@pytest.mark.parametrize("target", ["**AB**CD", "AB**CD**", "**A**B**C**"])
def test_joins_adjacent_bold_at_both_existing_boundaries(monkeypatch, target):
    monkeypatch.setenv("DOCPARSE_STYLE_JOIN_ADJACENT", "1")
    text = target.replace("**", "")
    line = _line(_span(text, bold=True))
    assert transfer_aligned_styles(target, [line]) == f"**{text}**"


def test_joins_question_label_to_native_bold_sentence(monkeypatch):
    monkeypatch.setenv("DOCPARSE_STYLE_JOIN_ADJACENT", "1")
    line = _line(_span("Ques: An example question.", bold=True))
    assert transfer_aligned_styles("**Ques**: An example question.", [line]) == "**Ques: An example question.**"


def test_join_does_not_change_existing_combined_emphasis(monkeypatch):
    monkeypatch.setenv("DOCPARSE_STYLE_JOIN_ADJACENT", "1")
    line = _line(_span("one", bold=True, italic=True), _span(" two", bold=True))
    assert transfer_aligned_styles("***one*** two", [line]) == "***one*** **two**"


def test_join_keeps_superscript_nested_inside_bold(monkeypatch):
    monkeypatch.setenv("DOCPARSE_STYLE_JOIN_ADJACENT", "1")
    line = _line(_span("x", bold=True), _span("2", bold=True, sup=True))
    assert transfer_aligned_styles("**x**2", [line]) == "**x<sup>2</sup>**"


def test_default_joins_adjacent_bold(monkeypatch):
    monkeypatch.delenv("DOCPARSE_STYLE_JOIN_ADJACENT", raising=False)
    line = _line(_span("ABCD", bold=True))
    assert transfer_aligned_styles("**AB**CD", [line]) == "**ABCD**"

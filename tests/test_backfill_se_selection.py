import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import backfill_se_selection as selection  # noqa: E402


def test_replace_selection_section_is_idempotent():
    section = "\n".join([selection.MARKER_START, "## Example", selection.MARKER_END, ""])
    first = selection.replace_selection_section("# Digest\n", section)
    second = selection.replace_selection_section(first, section)
    assert second == first


def test_selection_markdown_keeps_only_relevant_se_papers():
    items = [
        {"title": "Keep", "abs": "https://arxiv.org/abs/1", "categories": ["cs.SE"], "se_selection": {"relevant": True, "score": 0.9, "reason_zh": "direct"}},
        {"title": "Drop", "abs": "https://arxiv.org/abs/2", "categories": ["cs.SE"], "se_selection": {"relevant": False, "score": 0.2, "reason_zh": "not related"}},
        {"title": "AI", "abs": "https://arxiv.org/abs/3", "categories": ["cs.AI"], "se_selection": {"relevant": True, "score": 1.0, "reason_zh": "wrong category"}},
    ]
    section = selection.selection_markdown(items)
    assert "Keep" in section
    assert "Drop" not in section
    assert "wrong category" not in section


def test_classify_unique_allows_an_empty_se_batch():
    decisions, usage = selection.classify_unique([], workers=1, batch_size=20)
    assert decisions == {}
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

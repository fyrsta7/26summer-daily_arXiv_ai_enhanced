import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prune_se_delivery_papers as delivery  # noqa: E402


def test_keep_for_delivery_drops_only_unselected_se():
    assert delivery.keep_for_delivery({"categories": ["cs.AI"]})
    assert delivery.keep_for_delivery({"categories": ["cs.SE"], "se_selection": {"relevant": True}})
    assert not delivery.keep_for_delivery({"categories": ["cs.SE"], "se_selection": {"relevant": False}})


def test_keep_for_delivery_requires_a_se_decision():
    with pytest.raises(ValueError):
        delivery.keep_for_delivery({"id": "x", "categories": ["cs.SE"]})


def test_sync_enhanced_copies_final_decision_and_preserves_summary():
    decision = {"relevant": True, "reason_zh": "程序性能优化"}
    source = [{"id": "keep", "se_selection": decision}]
    enhanced = [
        {"id": "drop", "AI": {"tldr": "排除"}},
        {"id": "keep", "AI": {"tldr": "保留原来的摘要"}},
    ]
    actual = delivery.sync_enhanced_records(enhanced, source)
    assert actual == [{"id": "keep", "AI": {"tldr": "保留原来的摘要"}, "se_selection": decision}]
    assert "se_selection" not in enhanced[1]

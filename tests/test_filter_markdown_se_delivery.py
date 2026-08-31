import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import filter_markdown_se_delivery as markdown_filter  # noqa: E402


def test_filter_markdown_removes_only_unselected_se_blocks_and_addendum():
    markdown = """# Table of Contents

- [cs.SE](#cs.se) [Total: 2]
- [cs.AI](#cs.ai) [Total: 1]

# cs.SE [[Back]](#toc)

### [1] [Drop](https://arxiv.org/abs/drop)

Main category: cs.SE

Text to drop.

### [2] [Keep](https://arxiv.org/abs/keep)

Main category: cs.SE

Text to keep.

# cs.AI [[Back]](#toc)

### [3] [AI](https://arxiv.org/abs/ai)

Main category: cs.AI

<!-- semopt-se-selection:start -->
## SE 精选
<!-- semopt-se-selection:end -->
"""
    actual, removed = markdown_filter.filter_markdown(markdown, {"keep"})
    assert removed == 1
    assert "Drop" not in actual
    assert "Text to drop" not in actual
    assert "Keep" in actual and "AI" in actual
    assert "SE 精选" not in actual
    assert "[Total: 1]" in actual
    assert "### [1] [Keep]" in actual
    assert "### [2] [AI]" in actual

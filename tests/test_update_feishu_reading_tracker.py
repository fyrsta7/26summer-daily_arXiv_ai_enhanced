from scripts.update_feishu_reading_tracker import append_content


def test_appending_non_month_start_adds_one_checkbox():
    assert append_content("2026-08-16") == '<checkbox done="false">2026-08-16</checkbox>'


def test_appending_month_start_adds_a_month_heading():
    assert append_content("2026-09-01") == (
        '<h1>2026-09</h1><checkbox done="false">2026-09-01</checkbox>'
    )

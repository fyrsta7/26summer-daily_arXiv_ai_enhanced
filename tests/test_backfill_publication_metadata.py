import unittest

from scripts.backfill_publication_metadata import (
    parse_arxiv_feed,
    patch_markdown,
    plausible_official_affiliation,
)


class BackfillPublicationMetadataTests(unittest.TestCase):
    def test_parse_arxiv_publication_metadata_and_affiliation(self):
        xml = b'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
        <entry><id>https://arxiv.org/abs/2608.11513v1</id>
        <author><name>A. Author</name><arxiv:affiliation>Example University</arxiv:affiliation></author>
        <arxiv:comment>Accepted for publication</arxiv:comment>
        <arxiv:journal_ref>Example Journal 1 (2026)</arxiv:journal_ref>
        <arxiv:doi>10.1000/example</arxiv:doi></entry></feed>'''
        item = parse_arxiv_feed(xml)["2608.11513"]
        self.assertEqual(item["journal_ref"], "Example Journal 1 (2026)")
        self.assertEqual(item["author_affiliations"][0]["affiliation"], "Example University")

    def test_patch_markdown_is_idempotent(self):
        original = """# cs.SE\n\n### [1] [Paper](https://arxiv.org/abs/2608.11513)\n*Author*\n\nMain category: cs.SE\n\nTL;DR: text\n"""
        metadata = {
            "2608.11513": {
                "comment": "Accepted",
                "journal_ref": "Journal",
                "doi": "10.1000/example",
                "author_affiliations": [],
            }
        }
        once = patch_markdown(original, metadata)
        twice = patch_markdown(once, metadata)
        self.assertEqual(once, twice)
        self.assertEqual(once.count("Comments: Accepted"), 1)
        self.assertIn("Journal reference: Journal", once)

    def test_obviously_corrupt_single_name_affiliation_is_rejected(self):
        self.assertFalse(plausible_official_affiliation("Peter"))
        self.assertTrue(plausible_official_affiliation("NMSU"))
        self.assertTrue(plausible_official_affiliation("University of Example"))


if __name__ == "__main__":
    unittest.main()

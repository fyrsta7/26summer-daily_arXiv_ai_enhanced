import unittest

from to_md.convert import build_publication_metadata


class ConvertMetadataTests(unittest.TestCase):
    def test_publication_status_metadata_is_rendered(self):
        metadata = build_publication_metadata(
            {
                "comment": "Accepted for publication. 37 pages.",
                "journal_ref": "Empirical Software Engineering 32 (2026) 13",
                "doi": "10.1007/example",
                "author_affiliations": [
                    {"author": "A. Author", "affiliation": "Example University"}
                ],
            }
        )
        self.assertIn("Comments: Accepted for publication", metadata)
        self.assertIn("Journal reference: Empirical Software Engineering", metadata)
        self.assertIn("DOI: [10.1007/example](https://doi.org/10.1007/example)", metadata)
        self.assertIn("A. Author: Example University", metadata)

    def test_missing_optional_metadata_is_omitted(self):
        self.assertEqual(build_publication_metadata({"comment": None}), "")


if __name__ == "__main__":
    unittest.main()

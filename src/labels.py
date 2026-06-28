"""Label definitions for the four NER heads.

O is always index 0 so that special tokens / padding safely default to O.

CoNLL-2003 coverage:
  identity → PER, ORG, MISC
  location → LOC
  temporal → (none; add OntoNotes data for DATE/TIME/DURATION/SET)
  domain   → (none; supply a domain-specific annotated corpus)

ADDR (street-level address) is in the location head alongside LOC / GPE / FAC.
Example: "10 Downing Street", "1600 Pennsylvania Avenue", "5th Avenue"
"""

IDENTITY_LABELS = ["O", "B-PER",  "I-PER",  "B-ORG",   "I-ORG",   "B-MISC",  "I-MISC"]
LOCATION_LABELS = ["O", "B-LOC",  "I-LOC",  "B-GPE",   "I-GPE",   "B-FAC",   "I-FAC",  "B-ADDR", "I-ADDR"]
TEMPORAL_LABELS = ["O", "B-DATE", "I-DATE", "B-TIME",  "I-TIME",  "B-DUR",   "I-DUR",  "B-SET", "I-SET"]
DOMAIN_LABELS   = ["O", "B-PROD", "I-PROD", "B-EVENT", "I-EVENT", "B-LAW",   "I-LAW"]

IDENTITY_TO_ID = {l: i for i, l in enumerate(IDENTITY_LABELS)}
LOCATION_TO_ID = {l: i for i, l in enumerate(LOCATION_LABELS)}
TEMPORAL_TO_ID = {l: i for i, l in enumerate(TEMPORAL_LABELS)}
DOMAIN_TO_ID   = {l: i for i, l in enumerate(DOMAIN_LABELS)}

_CONLL_IDENTITY = {
    "B-PER": "B-PER", "I-PER": "I-PER",
    "B-ORG": "B-ORG", "I-ORG": "I-ORG",
    "B-MISC": "B-MISC", "I-MISC": "I-MISC",
}
_CONLL_LOCATION = {
    "B-LOC": "B-LOC", "I-LOC": "I-LOC",
}


def conll_tag_to_ids(tag: str):
    """Return (identity_id, location_id, temporal_id, domain_id) for a CoNLL-2003 BIO tag."""
    return (
        IDENTITY_TO_ID[_CONLL_IDENTITY.get(tag, "O")],
        LOCATION_TO_ID[_CONLL_LOCATION.get(tag, "O")],
        TEMPORAL_TO_ID["O"],
        DOMAIN_TO_ID["O"],
    )

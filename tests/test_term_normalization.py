from kg_gen.ontology.term_normalization import normalize_ontology_term
from kg_gen.utils.term_normalization import normalized_label_key


def test_lexical_normalization_unifies_common_label_styles():
    assert normalized_label_key("PlantPart") == "plant part"
    assert normalized_label_key("plant_part") == "plant part"
    assert normalized_label_key("plant-part") == "plant part"
    assert normalized_label_key("XMLParser") == "xml parser"
    assert normalized_label_key("  Has  Ability! ") == "has ability"


def test_class_morphology_is_conservative_and_idempotent():
    cases = {
        "PlantParts": "plant part",
        "herbivorous animals": "herbivore",
        "OmnivorousAnimal": "omnivore",
        "software": "software",
        "data": "data",
        "process": "process",
        "class": "class",
    }
    for source, expected in cases.items():
        assert normalize_ontology_term(source, "class") == expected
        assert normalize_ontology_term(expected, "class") == expected


def test_property_morphology_preserves_auxiliaries_and_direction():
    cases = {
        "purchasesGameOffering": "purchase game offering",
        "eats": "eat",
        "uses platform": "use platform",
        "has_parts": "has parts",
        "is_eaten_by": "is eaten by",
    }
    for source, expected in cases.items():
        assert normalize_ontology_term(source, "property") == expected
        assert normalize_ontology_term(expected, "property") == expected

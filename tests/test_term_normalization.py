from kg_gen.ontology.term_normalization import (
    normalize_ontology_term,
    normalize_term_assignments,
)
from kg_gen.utils.term_normalization import normalized_label_key


def test_lexical_normalization_unifies_common_label_styles():
    assert normalized_label_key("PlantPart") == "plant part"
    assert normalized_label_key("AssetCollection") == "asset collection"
    assert normalized_label_key("WhiteBordeaux") == "white bordeaux"
    assert normalized_label_key("YearValue") == "year value"
    assert normalized_label_key("plant_part") == "plant part"
    assert normalized_label_key("plant-part") == "plant part"
    assert normalized_label_key("XMLParser") == "xml parser"
    assert normalized_label_key("  Has  Ability! ") == "has ability"


def test_normalization_preserves_every_cq_assignment():
    assert normalize_term_assignments(
        {
            "CQ1": {"PlantParts"},
            "CQ2": {"plant_parts"},
            "CQ3": {"plant-parts"},
        },
        "class",
    ) == {
        "CQ1": {"plant part"},
        "CQ2": {"plant part"},
        "CQ3": {"plant part"},
    }


def test_class_morphology_is_conservative_and_idempotent():
    cases = {
        "PlantParts": "plant part",
        "AssetCollection": "asset collection",
        "WhiteBordeaux": "white bordeaux",
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
        "YearValue": "year value",
        "hasWineDescriptor": "has wine descriptor",
        "eats": "eat",
        "uses platform": "use platform",
        "has_parts": "has parts",
        "is_eaten_by": "is eaten by",
    }
    for source, expected in cases.items():
        assert normalize_ontology_term(source, "property") == expected
        assert normalize_ontology_term(expected, "property") == expected

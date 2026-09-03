"""Deduplication must not turn class assertions into ordinary relations.

`Graph.relations` is a *view*: it appends one `is a` triple per typed entity on
top of `relations_wo_class_assertions`, and `output_class_assertions = False`
suppresses them. Deduplication used to read that view and write the result back
into the stored field, which baked the assertions in permanently -- after that
no consumer could switch them off, and anything scoring the graph against a
gold standard of real relations counted every `is a` as a false positive.
"""

from kg_gen.models import Entity, EntityType, Graph, Relation, TypedEntity
from kg_gen.utils.deduplicate import run_semhash_deduplication


def _graph() -> Graph:
    college = TypedEntity(
        surface_form="AWH Engineering College",
        type=EntityType(label="University", uri="http://x#University"),
    )
    city = TypedEntity(
        surface_form="Kuttikkattoor",
        type=EntityType(label="City", uri="http://x#City"),
    )
    return Graph(
        typed_entities={college, city},
        relations_wo_class_assertions=[
            Relation(
                subject=Entity(surface_form="AWH Engineering College"),
                predicate=Entity(surface_form="city"),
                object=Entity(surface_form="Kuttikkattoor"),
            )
        ],
    )


def _dedup(graph: Graph) -> Graph:
    return run_semhash_deduplication(
        graph,
        model=None,
        entity_similarity_threshold=0.97,
        edge_similarity_threshold=0.9,
        deduplicate_edges=True,
        use_embeddings=False,
    )


def test_class_assertions_stay_suppressible_after_dedup():
    out = _dedup(_graph())
    out.output_class_assertions = False
    predicates = [r.predicate.surface_form for r in out.relations]
    assert predicates == ["city"], (
        "class assertions were materialised into relations_wo_class_assertions "
        f"and can no longer be switched off; got {predicates}"
    )


def test_class_assertions_are_still_available_after_dedup():
    """The view must still produce them when asked -- they are regenerated from
    the deduplicated typed entities, not stored."""
    out = _dedup(_graph())
    assert out.output_class_assertions is True
    assertions = [
        (r.subject.surface_form, r.object.surface_form)
        for r in out.relations
        if r.predicate.surface_form == "is a"
    ]
    assert sorted(assertions) == [
        ("AWH Engineering College", "University"),
        ("Kuttikkattoor", "City"),
    ]


def test_the_real_relation_survives_dedup():
    out = _dedup(_graph())
    assert len(out.relations_wo_class_assertions) == 1
    assert out.relations_wo_class_assertions[0].predicate.surface_form == "city"

import logging

from kg_gen.kg_gen import KGGen
from kg_gen.models import Graph, Entity, Relation, TypedEntity


def test_empty_graph_clustering(kg: KGGen):
    # Test with empty graph
    empty_graph = Graph(typed_entities=set(), relations_wo_class_assertions=[])
    clustered, stats = kg.deduplicate(empty_graph)

    assert len(clustered.entities) == 0
    assert len(clustered.edges) == 0
    assert len(clustered.relations) == 0
    assert clustered.entity_clusters is None
    assert clustered.edge_clusters is None


def test_semhash_deduplication(kg: KGGen):
    """
    Test SEMHASH deduplication method.
    SEMHASH should catch:
    - Plurals vs singulars (cat/cats, dog/dogs)
    - Case variations (Person/person/PERSON)
    - Very similar strings through normalization

    SEMHASH should NOT catch:
    - True synonyms (CEO/Chief Executive Officer)
    - Semantic equivalents (joyful/happy)
    """
    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="cat"),
            TypedEntity(surface_form="cats"),
            TypedEntity(surface_form="kitten"),  # Plurals - should be caught
            TypedEntity(surface_form="dog"),
            TypedEntity(surface_form="dogs"),  # Plurals - should be caught
            TypedEntity(surface_form="Person"),
            TypedEntity(surface_form="person"),
            TypedEntity(surface_form="PERSON"),  # Case variations - should be caught
            TypedEntity(surface_form="CEO"),
            TypedEntity(
                surface_form="Chief Executive Officer"
            ),  # Synonyms - should NOT be caught by semhash
            TypedEntity(surface_form="happy"),
            TypedEntity(
                surface_form="joyful"
            ),  # Synonyms - should NOT be caught by semhash
        },
        relations_wo_class_assertions=[
            Relation(
                subject=TypedEntity(surface_form="cat"),
                predicate=Entity(surface_form="likes"),
                object=TypedEntity(surface_form="dog"),
            ),
            Relation(
                subject=TypedEntity(surface_form="cats"),
                predicate=Entity(surface_form="like"),
                object=TypedEntity(surface_form="dogs"),
            ),
            Relation(
                subject=TypedEntity(surface_form="Person"),
                predicate=Entity(surface_form="manages"),
                object=TypedEntity(surface_form="CEO"),
            ),
            Relation(
                subject=TypedEntity(surface_form="person"),
                predicate=Entity(surface_form="Manages"),
                object=TypedEntity(surface_form="Chief Executive Officer"),
            ),
            Relation(
                subject=TypedEntity(surface_form="CEO"),
                predicate=Entity(surface_form="supervises"),
                object=TypedEntity(surface_form="happy"),
            ),
            Relation(
                subject=TypedEntity(surface_form="Chief Executive Officer"),
                predicate=Entity(surface_form="oversees"),
                object=TypedEntity(surface_form="joyful"),
            ),
        ],
    )

    deduplicated, stats = kg.deduplicate(
        graph=graph,
        entity_similarity_threshold=0.95,
    )

    # SEMHASH should merge plurals
    assert (
        TypedEntity(surface_form="cat") in deduplicated.typed_entities
        or TypedEntity(surface_form="cats") in deduplicated.typed_entities
    )
    assert not (
        TypedEntity(surface_form="cat") in deduplicated.typed_entities
        and TypedEntity(surface_form="cats") in deduplicated.typed_entities
    )

    # SEMHASH should merge case variations
    deduped_person = [
        e for e in deduplicated.entities if e.surface_form.lower() == "person"
    ]
    assert len(deduped_person) == 1, "Case variations should be merged to one"

    # SEMHASH should NOT merge true synonyms (they're different words)
    # Both CEO and Chief Executive Officer should still exist
    ceo_variants = [
        e
        for e in deduplicated.entities
        if "ceo" in e.surface_form.lower()
        or "chief executive" in e.surface_form.lower()
    ]
    assert len(ceo_variants) == 2, (
        "SEMHASH should not merge CEO and Chief Executive Officer"
    )

    # SEMHASH should NOT merge happy and joyful (different words)
    emotion_variants = [
        e
        for e in deduplicated.entities
        if e.surface_form.lower() in ["happy", "joyful"]
    ]
    assert len(emotion_variants) == 2, "SEMHASH should not merge happy and joyful"

    # Check edges
    assert Entity(surface_form="supervises") in deduplicated.edges
    assert Entity(surface_form="oversees") in deduplicated.edges
    assert Entity(surface_form="supervises") != Entity(surface_form="oversees"), (
        "SEMHASH should not merge synonym edges"
    )

    print(
        f"Original entities: {len(graph.entities)}, Deduplicated: {len(deduplicated.entities)}"
    )
    print(
        f"Original edges: {len(graph.edges)}, Deduplicated: {len(deduplicated.edges)}"
    )


def test_semhash_deduplication_lowthreshold_mpnet(kg: KGGen):
    """
    Test SEMHASH deduplication method.
    SEMHASH should catch:
    - Plurals vs singulars (cat/cats, dog/dogs)
    - Case variations (Person/person/PERSON)
    - Very similar strings through normalization
    - True synonyms (CEO/Chief Executive Officer)
    - Semantic equivalents (joyful/happy)
    """
    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="Person"),
            TypedEntity(surface_form="person"),
            TypedEntity(surface_form="CEO"),
            TypedEntity(surface_form="Chief Executive Officer"),
            TypedEntity(surface_form="happy"),
            TypedEntity(surface_form="joyful"),
        },
        relations_wo_class_assertions=[
            Relation(
                subject=TypedEntity(surface_form="Person"),
                predicate=Entity(surface_form="manages"),
                object=TypedEntity(surface_form="CEO"),
            ),
            Relation(
                subject=TypedEntity(surface_form="person"),
                predicate=Entity(surface_form="Manages"),
                object=TypedEntity(surface_form="Chief Executive Officer"),
            ),
            Relation(
                subject=TypedEntity(surface_form="CEO"),
                predicate=Entity(surface_form="supervises"),
                object=TypedEntity(surface_form="happy"),
            ),
            Relation(
                subject=TypedEntity(surface_form="Chief Executive Officer"),
                predicate=Entity(surface_form="oversees"),
                object=TypedEntity(surface_form="joyful"),
            ),
        ],
    )

    deduplicated, stats = kg.deduplicate(
        graph=graph,
        entity_similarity_threshold=0.5,
        edge_similarity_threshold=0.25,
        use_embeddings=True,
    )

    # SEMHASH should merge case variations
    deduped_person = [
        e for e in deduplicated.entities if e.surface_form.lower() == "person"
    ]
    assert len(deduped_person) == 1, "Case variations should be merged to one"

    # SEMHASH should NOT merge true synonyms (they're different words)
    # Both CEO and Chief Executive Officer should still exist
    ceo_variants = [
        e
        for e in deduplicated.entities
        if "ceo" in e.surface_form.lower()
        or "chief executive" in e.surface_form.lower()
    ]
    assert len(ceo_variants) == 1, (
        "SEMHASH should merge CEO and Chief Executive Officer"
    )

    # SEMHASH should NOT merge happy and joyful (different words)
    emotion_variants = [
        e
        for e in deduplicated.entities
        if e.surface_form.lower() in ["happy", "joyful"]
    ]
    assert len(emotion_variants) == 1, "SEMHASH should merge happy and joyful"

    # Check edges
    assert len(deduplicated.edges) == 1, deduplicated.edges
    assert len(deduplicated.edge_clusters.popitem()[1]) == 4

    logging.info(
        f"Original entities: {len(graph.entities)}, Deduplicated: {len(deduplicated.entities)}"
    )
    logging.info(
        f"Original edges: {len(graph.edges)}, Deduplicated: {len(deduplicated.edges)}"
    )


def test_deduplicate_edges_false_leaves_predicates_untouched(kg: KGGen):
    """
    With deduplicate_edges=False, predicates must not be clustered even with a
    permissive threshold that would otherwise merge them (as it does in
    test_semhash_deduplication_lowthreshold_mpnet above) — this is the mode
    used for ontology-guided extraction, where predicates are already
    canonical and should be left alone.
    """
    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="Person"),
            TypedEntity(surface_form="person"),
            TypedEntity(surface_form="CEO"),
            TypedEntity(surface_form="Chief Executive Officer"),
        },
        relations_wo_class_assertions=[
            Relation(
                subject=TypedEntity(surface_form="Person"),
                predicate=Entity(surface_form="manages"),
                object=TypedEntity(surface_form="CEO"),
            ),
            Relation(
                subject=TypedEntity(surface_form="person"),
                predicate=Entity(surface_form="Manages"),
                object=TypedEntity(surface_form="Chief Executive Officer"),
            ),
        ],
    )

    deduplicated, stats = kg.deduplicate(
        graph=graph,
        entity_similarity_threshold=0.5,
        edge_similarity_threshold=0.25,
        deduplicate_edges=False,
    )

    # Entities still get clustered.
    deduped_person = [
        e for e in deduplicated.entities if e.surface_form.lower() == "person"
    ]
    assert len(deduped_person) == 1, "Case variations should be merged to one"

    # Edges are left exactly as they were, despite the permissive threshold.
    assert len(deduplicated.edges) == 2
    assert not deduplicated.edge_clusters


def test_deduplication_preserves_provenance(kg: KGGen):
    """
    Test that deduplication preserves and merges provenance_ids of entities and relations.
    """
    ceo_1 = TypedEntity(surface_form="CEO", provenance_ids=["p1"])
    ceo_2 = TypedEntity(surface_form="Chief Executive Officer", provenance_ids=["p2"])
    employee = TypedEntity(surface_form="employee", provenance_ids=["p3"])

    graph = Graph(
        typed_entities={ceo_1, ceo_2, employee},
        relations_wo_class_assertions=[
            Relation(
                subject=ceo_1,
                predicate=Entity(surface_form="manages"),
                object=employee,
                provenance_ids=["r1"],
            ),
            Relation(
                subject=ceo_2,
                predicate=Entity(surface_form="Manager of"),
                object=employee,
                provenance_ids=["r2"],
            ),
        ],
    )

    deduplicated, _ = kg.deduplicate(
        graph=graph,
        entity_similarity_threshold=0.6,
        edge_similarity_threshold=0.6,
        use_embeddings=True,
    )

    # There should be 2 typed entities after merge: one for CEO/Chief Exec, one for employee
    assert len(deduplicated.typed_entities) == 2

    # Find the merged CEO entity
    ceo_entities = [
        e
        for e in deduplicated.typed_entities
        if "ceo" in e.surface_form.lower()
        or "chief executive" in e.surface_form.lower()
    ]
    assert len(ceo_entities) == 1
    ceo_entity = ceo_entities[0]

    # The provenance should be the union of the originals
    assert set(ceo_entity.provenance_ids) == {"p1", "p2"}

    # Find the employee entity
    employee_entity = next(
        e for e in deduplicated.typed_entities if e.surface_form == "employee"
    )
    assert set(employee_entity.provenance_ids) == {"p3"}

    assert len(deduplicated.edges) == 1

    # Now check relations. The two relations should be merged into one.
    assert len(deduplicated.relations_wo_class_assertions) == 1

    merged_relation = deduplicated.relations_wo_class_assertions[0]
    assert set(merged_relation.provenance_ids) == {"r1", "r2"}


def test_deduplication_with_descriptions(kg: KGGen):
    """
    Test that deduplication preserves and merges provenance_ids of entities and relations.
    """
    apple_1 = TypedEntity(
        surface_form="Apple Macintosh", description="A fruit", provenance_ids=["p1"]
    )
    apple_2 = TypedEntity(
        surface_form="Mac apple", description="A fruit", provenance_ids=["p2"]
    )
    apple_3 = TypedEntity(
        surface_form="Macintosh Apple",
        description="A technology company",
        provenance_ids=["p3"],
    )
    graph = Graph(
        typed_entities={apple_1, apple_2, apple_3},
        relations_wo_class_assertions=[],
    )
    deduplicated, _ = kg.deduplicate(
        graph=graph,
        entity_similarity_threshold=0.85,
        edge_similarity_threshold=0.75,
        use_embeddings=True,
    )
    # There should be 2 typed entities after merge: fruit and company
    assert len(deduplicated.typed_entities) == 2

    apple_1 = TypedEntity(surface_form="Apple Macintosh", provenance_ids=["p1"])
    apple_2 = TypedEntity(surface_form="Mac apple", provenance_ids=["p2"])
    apple_3 = TypedEntity(surface_form="Macintosh Apple", provenance_ids=["p3"])
    graph = Graph(
        typed_entities={apple_1, apple_2, apple_3},
        relations_wo_class_assertions=[],
    )
    deduplicated, _ = kg.deduplicate(
        graph=graph,
        entity_similarity_threshold=0.75,
        edge_similarity_threshold=0.75,
        use_embeddings=True,
    )
    # There should be 2 typed entities after merge: apple
    assert len(deduplicated.typed_entities) == 1


def test_merge_provenance_unions_without_duplicates():
    """Unit-test the provenance merge on its own: union, order-stable, no repeats."""
    from kg_gen.utils.deduplicate import _merge_provenance

    assert _merge_provenance([["a", "b"], ["b", "c"], []]) == ["a", "b", "c"]
    assert _merge_provenance([]) == []
    # Duplicates within a single list are collapsed too.
    assert _merge_provenance([["a", "a"], ["a"]]) == ["a"]


def test_deduplication_does_not_duplicate_provenance(kg: KGGen):
    """A merged entity's provenance must contain each id exactly once.

    Regression test. ``run_semhash_deduplication`` used to assign cluster
    provenance while iterating a list that holds each canonical entity once per
    cluster member; because the canonical is itself in its own cluster, every
    repeat folded its own already-merged list back in. A two-member cluster came
    out with three ids.

    Note this asserts on the *list*, not on ``set(...)``. The existing
    provenance test compares sets, which is why the inflation went unnoticed --
    the ids were all correct, just repeated.
    """
    ceo_1 = TypedEntity(surface_form="CEO", provenance_ids=["p1"])
    ceo_2 = TypedEntity(surface_form="Chief Executive Officer", provenance_ids=["p2"])
    employee = TypedEntity(surface_form="employee", provenance_ids=["p3"])

    graph = Graph(
        typed_entities={ceo_1, ceo_2, employee},
        relations_wo_class_assertions=[
            Relation(
                subject=ceo_1,
                predicate=Entity(surface_form="manages"),
                object=employee,
                provenance_ids=["r1"],
            ),
            Relation(
                subject=ceo_2,
                predicate=Entity(surface_form="Manager of"),
                object=employee,
                provenance_ids=["r2"],
            ),
        ],
    )

    deduplicated, _ = kg.deduplicate(
        graph=graph,
        entity_similarity_threshold=0.6,
        edge_similarity_threshold=0.6,
        use_embeddings=True,
    )

    for entity in deduplicated.typed_entities:
        assert len(entity.provenance_ids) == len(set(entity.provenance_ids)), (
            f"{entity.surface_form!r} has repeated provenance ids: "
            f"{entity.provenance_ids}"
        )
    merged_ceo = next(
        e for e in deduplicated.typed_entities if "ceo" in e.surface_form.lower()
    )
    assert sorted(merged_ceo.provenance_ids) == ["p1", "p2"]

    for relation in deduplicated.relations_wo_class_assertions:
        assert len(relation.provenance_ids) == len(set(relation.provenance_ids)), (
            f"relation {relation} has repeated provenance ids: "
            f"{relation.provenance_ids}"
        )


def test_string_only_deduplication_merges_without_embeddings(kg: KGGen):
    """With use_embeddings off, plural/singular and case variants still merge.

    The string grouping is not a no-op: surface forms are keyed by their
    normalized, singularized form before any embedding comparison, which is where
    the large majority of real duplicates are caught.
    """
    cats = TypedEntity(surface_form="Cats", provenance_ids=["p1"])
    cat = TypedEntity(surface_form="Cat", provenance_ids=["p2"])
    dog = TypedEntity(surface_form="Dog", provenance_ids=["p3"])
    graph = Graph(
        typed_entities={cats, cat, dog},
        relations_wo_class_assertions=[
            Relation(
                subject=cats,
                predicate=Entity(surface_form="chases"),
                object=dog,
                provenance_ids=["r1"],
            )
        ],
    )

    deduplicated, _ = kg.deduplicate(graph, use_embeddings=False)

    surface_forms = {e.surface_form for e in deduplicated.typed_entities}
    assert len(deduplicated.typed_entities) == 2, surface_forms
    merged = next(e for e in deduplicated.typed_entities if e.surface_form != "Dog")
    assert sorted(merged.provenance_ids) == ["p1", "p2"]


def test_string_only_deduplication_keeps_semantic_pairs_apart(kg: KGGen):
    """Without embeddings, an alias pair that needs semantics stays separate.

    This is the documented cost of the default: "North Yemen" and "Yemen Arab
    Republic" are the same country, and only the embedding pass can tell.
    """
    a = TypedEntity(surface_form="North Yemen")
    b = TypedEntity(surface_form="Yemen Arab Republic")
    graph = Graph(typed_entities={a, b}, relations_wo_class_assertions=[])

    deduplicated, _ = kg.deduplicate(graph, use_embeddings=False)
    assert len(deduplicated.typed_entities) == 2


def test_singularization_leaves_numerals_acronyms_and_compounds_alone():
    """Over-eager singularization merges unconditionally, at any threshold.

    Each of these used to collapse two distinct entities into one: decades into
    years, an acronym into a shorter one, and -- via inflect stringifying its own
    False for hyphenated compounds -- every hyphenated title sharing a suffix.
    """
    from kg_gen.utils.deduplicate import DeduplicateList

    dedup = DeduplicateList()

    def singular(text: str) -> str:
        return dedup.singularize(dedup.normalize(text))

    # A decade is not the plural of a year.
    assert singular("1970s") == "1970s"
    assert singular("1970s") != singular("1970")
    assert singular("CBS") != singular("CB")
    assert singular("Governor-General") != singular("Secretary-General")
    # Genuine plurals must still be singularized.
    assert singular("Cats") == singular("Cat")
    assert singular("Comanches") == singular("Comanche")
    assert singular("PGA Championships") == singular("PGA Championship")

from kg_gen.models import Graph, KGGenStats, Relation, TypedEntity, Entity


def test_knowledge_graph_clusters():
    # Setup a graph with clusters
    graph = Graph(
        typed_entities={
            TypedEntity(surface_form="Person 1", uri="http://example.org/p1"),
            TypedEntity(surface_form="Company A", uri="http://example.org/c1"),
        },
        relations_wo_class_assertions=[
            Relation(
                subject=TypedEntity(
                    surface_form="Person 1", uri="http://example.org/p1"
                ),
                predicate=Entity(
                    surface_form="works at", uri="http://example.org/worksAt"
                ),
                object=TypedEntity(
                    surface_form="Company A", uri="http://example.org/c1"
                ),
            )
        ],
        entity_clusters={
            "Person 1": [
                TypedEntity(surface_form="P1", description="Short name"),
                TypedEntity(surface_form="Person One", description="Full name"),
            ]
        },
        edge_clusters={
            "works at": [
                Entity(surface_form="employed by"),
                Entity(surface_form="works for"),
            ]
        },
    )

    stats = KGGenStats()
    kg_output = graph.to_knowledge_graph(stats)

    # Verify entity clusters
    assert kg_output.clusters is not None
    assert kg_output.clusters.entities is not None
    # Find the ID for "Person 1"
    person_id = next(
        id for id, e in kg_output.entities.items() if e.surface_form == "Person 1"
    )
    assert person_id in kg_output.clusters.entities
    assert len(kg_output.clusters.entities[person_id]) == 2
    assert any(m.surface_form == "P1" for m in kg_output.clusters.entities[person_id])
    assert any(
        m.surface_form == "Person One" for m in kg_output.clusters.entities[person_id]
    )

    # Verify edge clusters
    assert kg_output.clusters.edges is not None
    assert "works at" in kg_output.clusters.edges
    assert len(kg_output.clusters.edges["works at"]) == 2
    assert any(
        m.surface_form == "employed by" for m in kg_output.clusters.edges["works at"]
    )

    # Verify predicate label
    rel = kg_output.relations[0]
    assert rel.predicate.label == "works at"
    assert rel.predicate.uri == "http://example.org/worksAt"


def test_optional_ontology_extensions():
    graph = Graph(typed_entities=set(), relations_wo_class_assertions=[])
    stats = KGGenStats()

    # No extensions
    kg_output = graph.to_knowledge_graph(stats)
    assert kg_output.ontology_extensions is None

    # With extensions
    graph.typed_entities.add(
        TypedEntity(
            surface_form="e",
            type={"label": "NewClass", "uri": "http://ext.org/NewClass"},
        )
    )
    kg_output = graph.to_knowledge_graph(
        stats, ontology=None
    )  # ontology=None means all are extensions if it were used, but wait

    # Actually extensions are calculated only if ontology is provided
    from kg_gen.models import Ontology

    ontology = Ontology(classes=[], predicates=[])
    kg_output = graph.to_knowledge_graph(stats, ontology=ontology)
    assert kg_output.ontology_extensions is not None
    assert len(kg_output.ontology_extensions.classes) == 1
    assert kg_output.ontology_extensions.classes[0].label == "NewClass"

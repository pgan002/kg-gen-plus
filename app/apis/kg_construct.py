from __future__ import annotations

from rdflib.exceptions import ParserError
from typing import Optional

from fastapi import UploadFile, File, Query, HTTPException, APIRouter, Depends, Response
from pydantic import ValidationError
from rdflib import Graph as RDFGraph, RDFS, OWL, XSD, URIRef, RDF, Literal

from app.apis.deps import get_kg_gen
from app.kggen_logger import kggen_logger
from app.schemas import OntologyConversionInput, GenerateSingleInput
from app.utils import parse_ontology, GenerationMetadata
from kg_gen.kg_gen import KGGen
from kg_gen.models import (
    EntityType,
    Ontology,
    Graph,
    KGGenStats,
    InputData,
)


kgc_router = APIRouter(prefix="/api")


@kgc_router.post("/convert_ontology", tags=["KG Construction"])
async def convert_ontology(input_data: OntologyConversionInput) -> Response:
    """
    Takes a list of entity types (EntityType) and a list of OntologyPredicate
    and outputs a .ttl file structured for /generate.
    """
    g = RDFGraph()
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)

    for entity_type in input_data.classes:
        if entity_type.uri:
            class_uri = URIRef(entity_type.uri)
            g.add((class_uri, RDF.type, OWL.Class))
            if entity_type.label:
                g.add((class_uri, RDFS.label, Literal(entity_type.label)))
            if entity_type.description:
                g.add((class_uri, RDFS.comment, Literal(entity_type.description)))

    def is_datatype(entity_type: EntityType) -> bool:
        if entity_type.uri:
            return entity_type.uri.startswith(str(XSD))
        return False

    for predicate in input_data.predicates:
        if predicate.uri:
            prop_uri = URIRef(predicate.uri)

            is_data_prop = False
            if predicate.range:
                if all(is_datatype(r) for r in predicate.range):
                    is_data_prop = True

            prop_type = OWL.DatatypeProperty if is_data_prop else OWL.ObjectProperty
            g.add((prop_uri, RDF.type, prop_type))

            if predicate.label:
                g.add((prop_uri, RDFS.label, Literal(predicate.label)))
            if predicate.description:
                g.add((prop_uri, RDFS.comment, Literal(predicate.description)))

            for domain in predicate.domain:
                if domain.uri:
                    g.add((prop_uri, RDFS.domain, URIRef(domain.uri)))
            for range_ in predicate.range:
                if range_.uri:
                    g.add((prop_uri, RDFS.range, URIRef(range_.uri)))

    ttl_content = g.serialize(format="turtle")

    return Response(content=ttl_content, media_type="application/x-turtle")


@kgc_router.post("/parse_ontology", tags=["KG Construction"])
async def parse_ontology_api(
    ontology_file: UploadFile = File(
        ...,
        description=parse_ontology.__doc__,
    ),
) -> Ontology:
    """
    Endpoint to parse an ontology file and return the parsed ontology as an object.

    Parses the provided ontology file using the `parse_ontology` function and
    returns the resulting `Ontology` object.

    Args:
        ontology_file: The ontology file to be parsed. Must be uploaded as an
            `UploadFile`. The description of this parameter is derived from
            the documentation of the `parse_ontology` function.

    Returns:
        Ontology: The parsed ontology object.
    """
    onto: Ontology = parse_ontology(ontology_file.file)
    return onto


@kgc_router.post("/generate", tags=["KG Construction"])
async def generate_graph(
    response: Response,
    input_data: GenerateSingleInput,
    meta: GenerationMetadata = Query(...),
    ontology_file: Optional[UploadFile] = File(
        None,
        description=parse_ontology.__doc__,
    ),
    kg_gen: KGGen = Depends(get_kg_gen),
) -> Graph:
    """
    Generates a knowledge graph from a single text document.
    """
    onto = None
    if ontology_file:
        try:
            onto: Ontology = parse_ontology(ontology_file.file)
            types = list(onto.classes)
            predicates = list(onto.predicates)
        except (ParserError, SyntaxError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not parse the provided ontology: {exc}"
            )
        finally:
            ontology_file.file.close()
    else:
        types = None
        predicates = None

    kggen_logger.info(
        f"Generating graph for a single document via KGGen: {meta.model = }, {meta.retrieval_model = }, doc_id={input_data.id}"
    )
    if onto:
        kggen_logger.info(
            f"With ontology: {len(onto.classes) = }, {len(onto.predicates) = }"
        )
    else:
        kggen_logger.info("No ontology provided")
    kggen_logger.info(f"{meta.entity_context = }\n{meta.relation_context = }")

    total_gen_stats = KGGenStats()
    try:
        doc = InputData(id=input_data.id, text=input_data.text, terms=input_data.terms)
        try:
            graph, gen_stats = await kg_gen.generate(
                input_data=doc,
                temperature=meta.temperature,
                types=types,
                terms=doc.terms,
                predicate_domain_range=predicates,
                entity_context=meta.entity_context or "",
                relation_context=meta.relation_context or "",
            )
            total_gen_stats += gen_stats
        except ValidationError as exc:
            kggen_logger.exception("KGGen returned validation error")
            raise HTTPException(status_code=400, detail=f"Invalid graph result: {exc}")
        except Exception as exc:
            kggen_logger.exception("KGGen generation failed")
            raise HTTPException(status_code=500, detail=f"KGGen failed: {exc}")
    except ValidationError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid document format: {e.errors()}"
        )

    kggen_logger.info(
        "Graph generation complete: entities=%s relations=%s",
        len(graph.entities),
        len(graph.relations),
    )

    response.headers["X-KG-Gen-Stats"] = total_gen_stats.model_dump_json()

    gen_overall_stats = total_gen_stats.overall_usage
    response.headers["X-KG-Gen-Time"] = str(gen_overall_stats.execution_time)
    response.headers["X-KG-Gen-Input-Tokens"] = str(
        gen_overall_stats.lm_usage.prompt_tokens
    )
    response.headers["X-KG-Gen-Output-Tokens"] = str(
        gen_overall_stats.lm_usage.completion_tokens
    )

    return graph


@kgc_router.post("/aggregate_and_deduplicate", tags=["KG Construction"])
async def aggregate_and_deduplicate_graphs(
    response: Response,
    graphs: list[Graph],
    kg_gen: KGGen = Depends(get_kg_gen),
) -> Graph:
    """
    Aggregates a list of graphs and performs deduplication.
    """
    if not graphs:
        raise HTTPException(
            status_code=400, detail="No graphs provided for aggregation."
        )

    kggen_logger.info(f"Aggregating {len(graphs)} graphs.")
    aggregated_graph = kg_gen.aggregate(graphs)
    kggen_logger.info(
        "Aggregation complete: entities=%s relations=%s",
        len(aggregated_graph.entities),
        len(aggregated_graph.relations),
    )

    kggen_logger.info("Performing deduplication.")
    deduplicated_graph, dedup_stats = kg_gen.deduplicate(aggregated_graph)
    kggen_logger.info(
        "Deduplication complete: entities=%s relations=%s",
        len(deduplicated_graph.entities),
        len(deduplicated_graph.relations),
    )

    response.headers["X-KG-Gen-Dedup-Stats"] = dedup_stats.model_dump_json()

    return deduplicated_graph

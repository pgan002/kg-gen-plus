from __future__ import annotations

from rdflib.exceptions import ParserError
from typing import Optional, Annotated

from fastapi import UploadFile, File, Query, HTTPException, APIRouter, Response, Header
from pydantic import ValidationError
from rdflib import Graph as RDFGraph, RDFS, OWL, XSD, URIRef, RDF, Literal

from app.apis.deps import get_kg_gen
from app.kggen_logger import kggen_logger
from app.schemas import (
    OntologyConversionInput,
    GenerationMetadata,
    DeduplicationMetadata,
)
from app.utils import parse_ontology
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
    meta: Annotated[GenerationMetadata, Query()],
    x_api_key: Optional[str] = Header(
        None, description="API key for the LLM provider."
    ),
    ontology_file: Optional[UploadFile] = File(
        None,
        description=parse_ontology.__doc__,
    ),
    corpus_file: UploadFile = File(
        ...,
        description="A JSONL file where each line is a JSON object representing a document, with `id`, `text`, and `terms` fields. "
        "The terms are instances of #/components/schemas/TypedEntity.",
    ),
) -> list[Graph]:
    """
    Generates a knowledge graph from a single text document.
    """
    if corpus_file.filename is not None and not corpus_file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Corpus must be a .jsonl file")

    onto = None
    rdflib_onto = None
    if ontology_file:
        try:
            onto, rdflib_onto = parse_ontology(ontology_file.file)
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

    if onto:
        kggen_logger.info(
            f"With ontology: {len(onto.classes) = }, {len(onto.predicates) = }"
        )
    else:
        kggen_logger.info("No ontology provided")
    kggen_logger.info(f"{meta.entity_context = }\n{meta.relation_context = }")

    total_gen_stats = KGGenStats()
    kg_gen = get_kg_gen(
        api_key=x_api_key,
        api_base=meta.api_base,
        model=meta.model,
        enforce_type_conformance=meta.enforce_type_conformance,
        enforce_domain_conformance=meta.enforce_domain_conformance,
        enforce_range_conformance=meta.enforce_range_conformance,
        enforce_predicate_conformance=meta.enforce_predicate_conformance,
    )
    graphs = []
    kggen_logger.info(f"Generating graph via KGGen: {meta.model = }")
    try:
        for line in corpus_file.file:
            try:
                doc = InputData.model_validate_json(line)
                kggen_logger.info(f"Generating graph for doc_id={doc.id}")
            except ValidationError as e:
                raise HTTPException(
                    status_code=422, detail=f"Invalid document format: {e.errors()}"
                )
            try:
                graph, gen_stats = await kg_gen.generate(
                    input_data=doc,
                    temperature=meta.temperature,
                    types=types,
                    ontology=rdflib_onto,
                    predicate_domain_range=predicates,
                    entity_context=meta.entity_context or "",
                    relation_context=meta.relation_context or "",
                    enforce_type_conformance=meta.enforce_type_conformance,
                    enforce_domain_conformance=meta.enforce_domain_conformance,
                    enforce_range_conformance=meta.enforce_range_conformance,
                    enforce_predicate_conformance=meta.enforce_predicate_conformance,
                )
                kggen_logger.info(
                    f"Graph generation complete: {len(graph.entities) = }, {len(graph.relations) = }"
                )
                graphs.append(graph)
                total_gen_stats += gen_stats
            except ValidationError as exc:
                kggen_logger.exception("KGGen returned validation error")
                raise HTTPException(
                    status_code=400, detail=f"Invalid graph result: {exc}"
                )
            except Exception as exc:
                kggen_logger.exception("KGGen generation failed")
                raise HTTPException(status_code=500, detail=f"KGGen failed: {exc}")
    except ValidationError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid document format: {e.errors()}"
        )

    kggen_logger.info(
        f"Graphs generation complete: {len(graphs) = }, "
        f"total entities: {sum(len(g.entities) for g in graphs)}, "
        f"total relations: {sum(len(g.relations) for g in graphs)}."
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

    return graphs


@kgc_router.post("/aggregate_and_deduplicate", tags=["KG Construction"])
async def aggregate_and_deduplicate_graphs(
    response: Response,
    graphs: list[Graph],
    meta: Annotated[DeduplicationMetadata, Query()],
) -> Graph:
    """
    Aggregates a list of graphs and performs deduplication.
    """
    if not graphs:
        raise HTTPException(
            status_code=400, detail="No graphs provided for aggregation."
        )

    kg_gen = get_kg_gen(retrieval_model=meta.retrieval_model)
    kggen_logger.info(f"Aggregating {len(graphs)} graphs.")
    aggregated_graph = kg_gen.aggregate(graphs)
    kggen_logger.info(
        "Aggregation complete: entities=%s relations=%s",
        len(aggregated_graph.entities),
        len(aggregated_graph.relations),
    )

    if meta.deduplicate:
        kggen_logger.info("Performing deduplication.")
        deduplicated_graph, dedup_stats = kg_gen.deduplicate(aggregated_graph)
        kggen_logger.info(
            "Deduplication complete: entities=%s relations=%s",
            len(deduplicated_graph.entities),
            len(deduplicated_graph.relations),
        )
        response.headers["X-KG-Gen-Dedup-Stats"] = dedup_stats.model_dump_json()
        return deduplicated_graph
    else:
        return aggregated_graph

from __future__ import annotations

from rdflib.exceptions import ParserError
from typing import Optional

from fastapi import UploadFile, File, Query, HTTPException, APIRouter, Depends
from pydantic import BaseModel, ValidationError
from rdflib import Graph as RDFGraph, RDFS, OWL, XSD, URIRef, RDF, Literal
from starlette.responses import Response

from app.apis.deps import get_kg_gen
from app.kggen_logger import kggen_logger
from app.utils import parse_ontology, GenerationMetadata
from kg_gen.kg_gen import KGGen
from kg_gen.models import (
    EntityType,
    OntologyPredicate,
    Ontology,
    Graph,
    KGGenStats,
    InputData,
    StepStats,
)


kgc_router = APIRouter(prefix="/api")


class OntologyConversionInput(BaseModel):
    classes: list[EntityType]
    predicates: list[OntologyPredicate]


@kgc_router.post("/api/convert_ontology", tags=["KG Construction"])
async def convert_ontology(input_data: OntologyConversionInput) -> Response:
    """
    Takes a list of entity types (EntityType) and a list of OntologyPredicate
    and outputs a .ttl file structured for api/generate.
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


@kgc_router.post("/api/parse_ontology", tags=["KG Construction"])
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


@kgc_router.post("/api/generate", tags=["KG Construction"])
async def generate_graph(
    response: Response,
    meta: GenerationMetadata = Query(...),
    corpus_file: UploadFile = File(
        ...,
        description="A JSONL file where each line is a JSON object representing a document, with `id`, `text`, and `terms` fields. "
        "The terms are instances of #/components/schemas/TypedEntity.",
    ),
    ontology_file: Optional[UploadFile] = File(
        None,
        description=parse_ontology.__doc__,
    ),
    kg_gen: KGGen = Depends(get_kg_gen),
) -> Graph:
    if corpus_file.filename is not None and not corpus_file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Corpus must be a .jsonl file")

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
        f"Generating graph via KGGen: {meta.model = }, {meta.deduplicate = }, {meta.retrieval_model = }, {corpus_file.filename = }"
    )
    if onto:
        kggen_logger.info(
            f"With ontology: {len(onto.classes) = }, {len(onto.predicates) = }"
        )
    else:
        kggen_logger.info("No ontology provided")
    kggen_logger.info(f"{meta.entity_context = }\n{meta.relation_context = }")

    graphs = []
    total_gen_stats = KGGenStats()
    try:
        for line in corpus_file.file:
            try:
                doc = InputData.model_validate_json(line)
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
    finally:
        corpus_file.file.close()

    graph = kg_gen.aggregate(graphs)
    dedup_stats = StepStats()
    if meta.deduplicate:
        graph, dedup_stats = kg_gen.deduplicate(graph)

    kggen_logger.info(
        "Graph generation complete: entities=%s relations=%s",
        len(graph.entities),
        len(graph.relations),
    )

    total_gen_stats.deduplicate = dedup_stats
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

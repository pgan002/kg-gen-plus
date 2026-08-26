from typing import Union, Optional, Any, Callable
import asyncio
import time
from rdflib import Graph as RDFGraph

from app.kggen_logger import kggen_logger
from kg_gen.steps._1_get_entities import type_terms, extract_entities
from kg_gen.steps._2_get_relations import get_relations_typed
from kg_gen.utils.deduplicate import run_semhash_deduplication
from kg_gen.utils.visualize_kg import visualize as visualize_kg
from kg_gen.models import (
    Graph,
    Relation,
    KGGenStats,
    StepStats,
    LMUsage,
    Entity,
    EntityType,
    TypedEntity,
    OntologyPredicate,
    InputData,
)
import dspy
import os
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import logging

logger = logging.getLogger(__name__)

# SentenceTransformer models are large (hundreds of MB) and stateless for
# inference, so a single instance per model name is shared across all KGGen
# instances instead of loading a fresh copy for each one.
_SENTENCE_TRANSFORMER_CACHE: dict[str, SentenceTransformer] = {}


def _get_shared_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Return a shared SentenceTransformer, loading it once per model name."""
    if model_name not in _SENTENCE_TRANSFORMER_CACHE:
        _SENTENCE_TRANSFORMER_CACHE[model_name] = SentenceTransformer(model_name)
    return _SENTENCE_TRANSFORMER_CACHE[model_name]


def _log_generation_progress(done: int, total: int, start_time: float) -> None:
    """Emit a tqdm-like progress line for parallel document generation.

    Logs on the first and last document and roughly every 5% in between, so the
    number of log lines stays bounded (~20) regardless of corpus size.
    """
    step = max(1, total // 20)
    if not (done == 1 or done == total or done % step == 0):
        return

    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")
    pct = 100.0 * done / total if total else 100.0
    eta_str = f"{eta:.1f}s" if eta != float("inf") else "?"
    kggen_logger.info(
        f"[generate] {done}/{total} docs ({pct:.1f}%) "
        f"elapsed={elapsed:.1f}s rate={rate:.2f} doc/s ETA={eta_str}"
    )


# Monkey-patch dspy.predict.refine.OfferFeedback to avoid type mismatch warnings.
# Refine.forward stringifies these fields, so we update the signature to expect strings.
try:
    from dspy.predict.refine import OfferFeedback

    for field_name in ["target_threshold", "reward_value", "module_names"]:
        if field_name in OfferFeedback.model_fields:
            OfferFeedback.model_fields[field_name].annotation = str
    OfferFeedback.model_rebuild(force=True)
except (ImportError, AttributeError):
    pass


class KGGen:
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_tokens: int = 16000,  # minimum for gpt-5 family models
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        retrieval_model: Optional[str] = "sentence-transformers/all-mpnet-base-v2",
        disable_cache: bool = False,
        enable_thinking: Optional[bool] = False,
        enforce_type_conformance: bool = False,
        enforce_domain_conformance: bool = True,
        enforce_range_conformance: bool = True,
        enforce_predicate_conformance: bool = False,
    ):
        """Initialize KGGen with optional model configuration

        Args:
            model: Name of model to use (e.g. 'gpt-4')
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: Specify the base URL endpoint for making API calls to a language model service
            enforce_type_conformance: If True, predicted types should necessarily come from the list of types, if provided.
            enforce_domain_conformance: If True, predicted relations should necessarily come from the list of types, if provided.
            enforce_range_conformance: If True, predicted relations should necessarily come from the list of types, if provided.
            enforce_predicate_conformance: If True, predicted predicates should necessarily come from the list of predicates, if provided.
        """
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.retrieval_model: Optional[SentenceTransformer] = None
        self.retrieval_model_name: Optional[str] = retrieval_model
        self._lm = None
        self.disable_cache = disable_cache
        self.enable_thinking = enable_thinking
        self.enforce_type_conformance = enforce_type_conformance
        self.enforce_domain_conformance = enforce_domain_conformance
        self.enforce_range_conformance = enforce_range_conformance
        self.enforce_predicate_conformance = enforce_predicate_conformance

        self.validate_temperature(self.temperature)
        self.validate_max_tokens(self.max_tokens)

    @property
    def lm(self):
        if self._lm is None:
            self.init_model()
        return self._lm

    def validate_temperature(self, temperature: float):
        if "gpt-5" in self.model and temperature < 1.0:
            kggen_logger.warning(
                f"Temperature must be 1.0 for gpt-5 family models, {temperature = }."
            )
            self.temperature = 1.0

    def validate_max_tokens(self, max_tokens: int):
        if "gpt-5" in self.model and max_tokens < 16000:
            kggen_logger.warning("Max tokens must be 16000 for gpt-5 family models")
            self.max_tokens = 16000

    def init_model(
        self,
    ):
        """Initialize or reinitialize the model with new parameters

        Args:
            model: Name of model to use (e.g. 'gpt-4')
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: API base for model access
            retrieval_model: Name of retrieval model to use
            reasoning_effort: Reasoning effort for model
            max_tokens: Maximum tokens for model
            temperature: Temperature for model sampling
        """
        if self.retrieval_model_name is not None:
            self.retrieval_model = _get_shared_sentence_transformer(
                self.retrieval_model_name
            )

        # Initialize dspy LM with current settings
        settings_dict: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "reasoning": {"effort": self.reasoning_effort}
            if self.reasoning_effort
            else None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base,
            "cache": not self.disable_cache,
            "model_type": "chat",
        }
        if (
            self.enable_thinking is not None
            and "gpt" not in self.model
            and "o1" not in self.model
        ):
            settings_dict["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        if self.api_key:
            settings_dict["api_key"] = self.api_key

        self._lm = dspy.LM(**settings_dict)

    @staticmethod
    def from_dict(graph_dict: dict) -> Graph:
        return Graph(**graph_dict)

    async def generate(
        self,
        input_data: Union[InputData, list[InputData]],
        ontology: Optional[RDFGraph] = None,
        entity_context: str = "",
        types: Optional[list[EntityType]] = None,
        relation_context: str = "",
        predicate_domain_range: Optional[list[OntologyPredicate]] = None,
        deduplicate: bool = True,
        temperature: float | int | None = None,
        enforce_type_conformance: Optional[bool] = None,
        enforce_domain_conformance: Optional[bool] = None,
        enforce_range_conformance: Optional[bool] = None,
        enforce_predicate_conformance: Optional[bool] = None,
        n_parallel: int = 10,
        entity_similarity_threshold: float = 0.8,
        edge_similarity_threshold: float = 0.9,
        deduplicate_with_embeddings: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[Graph, KGGenStats]:
        # Normalize parameters
        enforce_type_conformance = (
            enforce_type_conformance
            if enforce_type_conformance is not None
            else self.enforce_type_conformance
        )
        enforce_domain_conformance = (
            enforce_domain_conformance
            if enforce_domain_conformance is not None
            else self.enforce_domain_conformance
        )
        enforce_range_conformance = (
            enforce_range_conformance
            if enforce_range_conformance is not None
            else self.enforce_range_conformance
        )
        enforce_predicate_conformance = (
            enforce_predicate_conformance
            if enforce_predicate_conformance is not None
            else self.enforce_predicate_conformance
        )
        # Note: use `is not None` rather than `or` so an explicit temperature=0.0
        # (greedy decoding) is respected instead of falling back to the default.
        temperature = temperature if temperature is not None else self.temperature

        kggen_logger.info(
            "Generating Knowledge Graph"
            f"{enforce_domain_conformance = }, {enforce_range_conformance = }, {enforce_predicate_conformance = }, {enforce_type_conformance = }, "
            f"{deduplicate = }, {temperature = }"
        )

        async def _process_single(item: InputData) -> tuple[Graph, KGGenStats]:
            content = item.text
            item_terms = item.terms
            lm = self.lm.copy()

            # Monkey-patch copy to preserve history reference
            # This is needed because dspy.Refine and other modules might copy the LM,
            # which isolates token usage stats.
            original_copy = lm.copy

            def shared_history_copy(**kwargs):
                new_instance = original_copy(**kwargs)
                new_instance.history = lm.history
                return new_instance

            lm.copy = shared_history_copy

            with dspy.context(lm=lm):
                step_stats = {}
                self.reset_token_usage(lm)
                start_time = time.time()
                if not item_terms:
                    typed_entities = await extract_entities(
                        content,
                        context=entity_context,
                        types=types,
                        temperature=temperature,
                        enforce_type_conformance=enforce_type_conformance,
                        provenance_ids=[item.id],
                    )
                    step_stats["extract_entities"] = StepStats(
                        lm_usage=LMUsage(**self.extract_token_usage_from_history(lm)),
                        execution_time=time.time() - start_time,
                    )
                else:
                    entities = []
                    for t in item_terms:
                        if isinstance(t, str):
                            entities.append(Entity(surface_form=t))
                        elif isinstance(t, TypedEntity):
                            entities.append(t)
                        elif isinstance(t, dict):
                            entities.append(TypedEntity(**t))
                        else:
                            # Fallback if it's already an Entity but not TypedEntity
                            entities.append(
                                TypedEntity(surface_form=t.surface_form, uri=t.uri)
                            )
                    step_stats["get_entities"] = StepStats(
                        lm_usage=LMUsage(), execution_time=0.0
                    )

                    self.reset_token_usage(lm)
                    start_time = time.time()

                    already_typed = []
                    to_type = []
                    for e in entities:
                        if isinstance(e, TypedEntity) and e.type is not None:
                            already_typed.append(e)
                        else:
                            to_type.append(e)

                    if not to_type:
                        typed_entities = entities
                    else:
                        newly_typed = await type_terms(
                            input_data=content,
                            terms=[e.surface_form for e in to_type],
                            types=types,
                            temperature=temperature,
                            provenance_ids=[item.id],
                            context=entity_context,
                            enforce_type_conformance=enforce_type_conformance,
                        )
                        typed_entities = already_typed + newly_typed

                    step_stats["type_terms"] = StepStats(
                        lm_usage=LMUsage(**self.extract_token_usage_from_history(lm)),
                        execution_time=time.time() - start_time,
                    )

                self.reset_token_usage(lm)
                start_time = time.time()
                relations = await get_relations_typed(
                    content,
                    typed_entities=typed_entities,
                    ontology=ontology,
                    predicate_domain_range=predicate_domain_range,
                    context=relation_context,
                    temperature=temperature,
                    provenance_ids=[item.id],
                    enforce_domain_conformance=enforce_domain_conformance or False,
                    enforce_range_conformance=enforce_range_conformance or False,
                    enforce_predicate_conformance=enforce_predicate_conformance
                    or False,
                    enforce_type_conformance=enforce_type_conformance,
                    allowed_types=types,
                )
                step_stats["get_relations_typed"] = StepStats(
                    lm_usage=LMUsage(**self.extract_token_usage_from_history(lm)),
                    execution_time=time.time() - start_time,
                )

                graph = Graph(
                    typed_entities=set(typed_entities),
                    relations_wo_class_assertions=relations,
                )
                return graph, KGGenStats(**step_stats)

        if isinstance(input_data, list):
            """Generates a Knowledge Graph from multiple input documents in parallel."""
            semaphore = asyncio.Semaphore(n_parallel)
            total = len(input_data)
            # asyncio runs these coroutines on a single thread, so the plain
            # counter increment below needs no lock (there is no await between
            # the read and the write).
            progress = {"done": 0}
            progress_start = time.time()

            async def process_with_semaphore(item: InputData):
                # Progress must advance (and be reported) even when this
                # document raises -- otherwise one failing document leaves
                # the job's progress/percent_complete permanently short of
                # 100%, since gather() below tolerates individual failures
                # and keeps going instead of aborting the whole batch.
                try:
                    async with semaphore:
                        result = await _process_single(item)
                finally:
                    progress["done"] += 1
                    done = progress["done"]
                    _log_generation_progress(done, total, progress_start)
                    if progress_callback is not None:
                        progress_callback(done, total)
                return result

            tasks = [process_with_semaphore(i) for i in input_data]
            if not tasks:
                return Graph(
                    typed_entities=set(), relations_wo_class_assertions=[]
                ), KGGenStats()

            kggen_logger.info(f"[generate] starting on {total} docs, {n_parallel = }")
            # return_exceptions=True: one document producing a malformed/
            # truncated LM response (e.g. a JSONAdapter parse failure) must not
            # discard every other document's already-successful results. Each
            # failure is recorded in failed_documents instead; the request
            # only fails outright if *every* document failed.
            results = await asyncio.gather(*tasks, return_exceptions=True)

            graphs = []
            total_gen_stats = KGGenStats()
            failed_documents: list[dict[str, str]] = []
            for item, result in zip(input_data, results):
                if isinstance(result, BaseException):
                    kggen_logger.error(
                        f"[generate] document {item.id!r} failed and was skipped: {result}"
                    )
                    failed_documents.append({"id": item.id, "error": str(result)})
                    continue
                graph, gen_stats = result
                graphs.append(graph)
                total_gen_stats += gen_stats

            if not graphs:
                failure_summary = "; ".join(
                    f"{f['id']}: {f['error']}" for f in failed_documents
                )
                raise RuntimeError(
                    f"KGGen failed on every document ({len(failed_documents)}/{total}): "
                    f"{failure_summary}"
                )

            if failed_documents:
                kggen_logger.warning(
                    f"[generate] {len(failed_documents)}/{total} document(s) failed "
                    f"and were skipped: {[f['id'] for f in failed_documents]}"
                )
            total_gen_stats.failed_documents = failed_documents

            kggen_logger.info(
                f"Graphs generation complete: {len(graphs) = }, "
                f"total entities: {sum(len(g.entities) for g in graphs)}, "
                f"total relations: {sum(len(g.relations) for g in graphs)}."
            )

            kggen_logger.info(f"Aggregating {len(graphs)} graphs.")
            final_graph = self.aggregate(graphs)
            final_stats = total_gen_stats
        else:
            final_graph, final_stats = await _process_single(input_data)

        if deduplicate:
            # Predicates are drawn from the ontology's controlled vocabulary, so
            # they are already canonical; only entity surface forms need
            # semantic clustering when an ontology guides extraction.
            deduplicate_edges = ontology is None
            kggen_logger.info(f"Performing deduplication ({deduplicate_edges = }).")
            final_graph, dedup_stats = self.deduplicate(
                final_graph,
                entity_similarity_threshold=entity_similarity_threshold,
                edge_similarity_threshold=edge_similarity_threshold,
                deduplicate_edges=deduplicate_edges,
                use_embeddings=deduplicate_with_embeddings,
            )
            final_stats.deduplicate = dedup_stats

        # Calculate class and predicate usage
        class_usage = {}
        for te in final_graph.typed_entities:
            if te.type:
                key = te.type.uri or te.type.label
                class_usage[key] = class_usage.get(key, 0) + 1

        predicate_usage = {}
        for rel in final_graph.relations_wo_class_assertions:
            key = rel.predicate.uri or rel.predicate.surface_form
            predicate_usage[key] = predicate_usage.get(key, 0) + 1

        final_stats.class_usage = class_usage
        final_stats.predicate_usage = predicate_usage

        return final_graph, final_stats

    def deduplicate(
        self,
        graph: Graph,
        entity_similarity_threshold: float = 0.8,
        edge_similarity_threshold: float = 0.9,
        deduplicate_edges: bool = True,
        use_embeddings: bool = False,
    ) -> tuple[Graph, StepStats]:
        """Merge duplicate entities (and optionally predicates) in ``graph``.

        ``use_embeddings`` selects the semantic pass on top of the string
        grouping. It is off by default because it dominates the runtime while
        contributing a small minority of the merges -- see
        ``run_semhash_deduplication`` for the measurements.
        """
        start_time = time.time()
        if not graph.entities and not graph.edges:
            return graph, StepStats(execution_time=0.0)
        deduplicated_graph = run_semhash_deduplication(
            graph,
            model=self.retrieval_model,
            entity_similarity_threshold=entity_similarity_threshold,
            edge_similarity_threshold=edge_similarity_threshold,
            deduplicate_edges=deduplicate_edges,
            use_embeddings=use_embeddings,
        )
        stats = StepStats(
            execution_time=time.time() - start_time,
        )
        return deduplicated_graph, stats

    def aggregate(self, graphs: list[Graph]) -> Graph:
        # Key entities/relations on (surface form + type), excluding URIs, since
        # LLM-generated URIs are unreliable and would otherwise over-count
        # otherwise-identical entities. Type is kept so homonyms with distinct
        # types (e.g. "Mercury" the planet vs. the element) stay separate.
        all_typed_entities: dict[tuple, TypedEntity] = {}
        all_relations: dict[tuple, Relation] = {}
        all_entity_metadata: dict[TypedEntity, set[str]] = {}

        def _prefer_non_null_uri(target: Entity, source: Entity) -> None:
            """Fill target.uri from source.uri when the survivor lacks one."""
            if target.uri is None and source.uri is not None:
                target.uri = source.uri

        for graph in graphs:
            for entity in graph.typed_entities:
                entity_key = (entity.surface_form, entity.type)
                if entity_key in all_typed_entities:
                    existing_entity = all_typed_entities[entity_key]
                    existing_entity.provenance_ids = list(
                        set(existing_entity.provenance_ids + entity.provenance_ids)
                    )
                    # Preserve a non-null URI on the entity and its type.
                    _prefer_non_null_uri(existing_entity, entity)
                    if (
                        existing_entity.type is not None
                        and existing_entity.type.uri is None
                        and entity.type is not None
                        and entity.type.uri is not None
                    ):
                        existing_entity.type = existing_entity.type.model_copy(
                            update={"uri": entity.type.uri}
                        )
                else:
                    all_typed_entities[entity_key] = entity

            for relation in graph.relations_wo_class_assertions:
                relation_key = (
                    relation.subject.surface_form,
                    relation.predicate.surface_form,
                    relation.object.surface_form,
                )
                if relation_key in all_relations:
                    existing_relation = all_relations[relation_key]
                    existing_relation.provenance_ids = list(
                        set(existing_relation.provenance_ids + relation.provenance_ids)
                    )
                    # Preserve non-null URIs on the merged relation's components.
                    _prefer_non_null_uri(existing_relation.subject, relation.subject)
                    _prefer_non_null_uri(
                        existing_relation.predicate, relation.predicate
                    )
                    _prefer_non_null_uri(existing_relation.object, relation.object)
                else:
                    all_relations[relation_key] = relation

            if graph.entity_metadata:
                for entity, metadata_set in graph.entity_metadata.items():
                    if entity in all_entity_metadata:
                        all_entity_metadata[entity].update(metadata_set)
                    else:
                        all_entity_metadata[entity] = metadata_set.copy()

        return Graph(
            typed_entities=set(all_typed_entities.values()),
            relations_wo_class_assertions=list(all_relations.values()),
            entity_metadata=all_entity_metadata if all_entity_metadata else None,
        )

    @staticmethod
    def visualize(graph: Graph, output_path: str, open_in_browser: bool = False):
        visualize_kg(graph, output_path, open_in_browser=open_in_browser)

    def _parse_embedding_model(
        self, model: Optional[SentenceTransformer] = None
    ) -> Optional[SentenceTransformer]:
        if model is None:
            model = self.retrieval_model
        if model is None:
            raise ValueError("No retrieval model provided")
        return model

    @staticmethod
    def to_nx(graph: Graph) -> nx.DiGraph:
        G = nx.DiGraph()
        for entity in graph.entities:
            G.add_node(entity)

        for relation in graph.relations:
            G.add_edge(relation.subject, relation.object, relation=relation.predicate)
        return G

    def generate_embeddings(
        self,
        graph: Union[Graph, nx.DiGraph],
        model: Optional[SentenceTransformer] = None,
    ) -> tuple[dict[TypedEntity, np.ndarray], dict[Entity, np.ndarray]]:
        model = self._parse_embedding_model(model)
        if isinstance(graph, Graph):
            graph = self.to_nx(graph)

        node_embeddings = {
            node: model.encode(str(node)).tolist() for node in graph.nodes
        }
        relation_embeddings = {
            rel: model.encode(str(rel)).tolist()
            for rel in set(edge[2]["relation"] for edge in graph.edges(data=True))
        }
        return node_embeddings, relation_embeddings

    def retrieve(
        self,
        query: str,
        node_embeddings: dict[TypedEntity, np.ndarray],
        graph: nx.DiGraph,
        model: Optional[SentenceTransformer] = None,
        k: int = 8,
        verbose: bool = False,
    ) -> tuple[list[tuple[TypedEntity, float]], set[str], str]:
        model = self._parse_embedding_model(model)
        top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, model, k)
        context = set()
        for node, _ in top_nodes:
            node_context = self.retrieve_context(node, graph)
            if verbose:
                print(f"Context for node {node}: {node_context}")
            context.update(node_context)
        context_text = " ".join(context)
        if verbose:
            print(f"Combined context: '{context_text}'\n---")
        return top_nodes, context, context_text

    @staticmethod
    def retrieve_relevant_nodes(
        query: str,
        node_embeddings: dict[TypedEntity, np.ndarray],
        model: SentenceTransformer,
        k: int = 8,
    ) -> list[tuple[TypedEntity, float]]:
        query_embedding = model.encode(query).reshape(1, -1)
        similarities = []
        for node, embed in node_embeddings.items():
            target_embedding = np.array(embed).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, target_embedding)[0][0]
            similarities.append((node, similarity))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:k]

    @staticmethod
    def retrieve_context(
        node: TypedEntity, graph: nx.DiGraph, depth: int = 2
    ) -> list[str]:
        context = set()

        def explore_neighbors(current_node, current_depth):
            if current_depth > depth:
                return
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]["relation"]
                context.add(f"{current_node} {rel} {neighbor}.")
                explore_neighbors(neighbor, current_depth + 1)
            for neighbor in graph.predecessors(current_node):
                rel = graph[neighbor][current_node]["relation"]
                context.add(f"{neighbor} {rel} {current_node}.")
                explore_neighbors(neighbor, current_depth + 1)

        explore_neighbors(node, 1)
        return list(context)

    @staticmethod
    def export_graph(graph: Graph, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        graph.to_file(output_path)

    def reset_token_usage(self, lm: Optional[dspy.LM] = None):
        target_lm = lm if lm is not None else self.lm
        target_lm.history = []

    def extract_token_usage_from_history(
        self, lm: Optional[dspy.LM] = None
    ) -> dict[str, int]:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        target_lm = lm if lm is not None else self.lm
        for entry in target_lm.history:
            usage = None
            if isinstance(entry, dict):
                # Try multiple possible locations for usage
                usage = (
                    entry.get("usage")
                    or entry.get("response", {}).get("usage")
                    or entry.get("model_info", {}).get("usage")
                )

                # If still not found, check if it's a cached entry which might have different structure
                if not usage and "cache_hit" in entry:
                    # Often cached entries have 0 usage anyway, but just in case
                    pass

            if usage:
                # Map various token field names
                p_tokens = 0
                c_tokens = 0
                t_tokens = 0

                if isinstance(usage, dict):
                    p_tokens = (
                        usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                    )
                    c_tokens = (
                        usage.get("completion_tokens")
                        or usage.get("output_tokens")
                        or 0
                    )
                    t_tokens = usage.get("total_tokens") or (p_tokens + c_tokens)
                else:
                    # Handle cases where usage might be an object
                    p_tokens = (
                        getattr(usage, "prompt_tokens", None)
                        or getattr(usage, "input_tokens", 0)
                        or 0
                    )
                    c_tokens = (
                        getattr(usage, "completion_tokens", None)
                        or getattr(usage, "output_tokens", 0)
                        or 0
                    )
                    t_tokens = getattr(usage, "total_tokens", None) or (
                        p_tokens + c_tokens
                    )

                total_prompt_tokens += p_tokens
                total_completion_tokens += c_tokens
                total_tokens += t_tokens

        return {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        }

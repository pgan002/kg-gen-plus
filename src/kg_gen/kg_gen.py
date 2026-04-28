from typing import Union, Dict, Optional
from typing_extensions import deprecated
import time  # Import time module

from kg_gen.steps._1_get_entities import get_entities, type_terms
from kg_gen.steps._2_get_relations import get_relations_typed
from kg_gen.steps._3_deduplicate import run_deduplication, DeduplicateMethod
from kg_gen.utils.chunk_text import chunk_text
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure dspy logging to only show errors
import logging

logger = logging.getLogger(__name__)


class KGGen:
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_tokens: int = 16000,  # minimum for gpt-5 family models
        temperature: float = 0.0,
        reasoning_effort: str = None,
        api_key: str = None,
        api_base: str = None,
        retrieval_model: Optional[str] = None,
        disable_cache: bool = False,
    ):
        """Initialize KGGen with optional model configuration

        Args:
            model: Name of model to use (e.g. 'gpt-4')
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: Specify the base URL endpoint for making API calls to a language model service
        """
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.retrieval_model: Optional[SentenceTransformer] = None
        self.lm = None
        self.disable_cache = disable_cache

        self.init_model(
            model=model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            retrieval_model=retrieval_model,
        )
        dspy.configure(track_usage=True)

    def validate_temperature(self, temperature: float):
        if "gpt-5" in self.model and temperature < 1.0:
            raise ValueError(
                f"Temperature must be 1.0 for gpt-5 family models, {temperature = }."
            )

    def validate_max_tokens(self, max_tokens: int):
        if "gpt-5" in self.model and max_tokens < 16000:
            raise ValueError("Max tokens must be 16000 for gpt-5 family models")

    def init_model(
        self,
        model: str = None,
        reasoning_effort: str = None,
        max_tokens: int = None,
        temperature: float = None,
        retrieval_model: str = None,
        api_key: str = None,
        api_base: str = None,
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

        # Update instance variables if new values provided
        if model is not None:
            self.model = model
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if api_key is not None:
            self.api_key = api_key
        if api_base is not None:
            self.api_base = api_base
        if temperature is not None:
            self.temperature = temperature
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
        if retrieval_model is not None:
            self.retrieval_model = SentenceTransformer(retrieval_model)

        self.validate_temperature(self.temperature)
        self.validate_max_tokens(self.max_tokens)

        # Initialize dspy LM with current settings
        if self.api_key:
            self.lm = dspy.LM(
                model=self.model,
                api_key=self.api_key,
                reasoning={"effort": self.reasoning_effort}
                if self.reasoning_effort
                else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                cache=not self.disable_cache,
                model_type="chat",
            )
        else:
            self.lm = dspy.LM(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                reasoning={"effort": self.reasoning_effort}
                if self.reasoning_effort
                else None,
                cache=not self.disable_cache,
                model_type="chat",
            )

    @staticmethod
    def from_file(file_path: str) -> Graph:
        return Graph.from_file(file_path)

    @staticmethod
    def from_dict(graph_dict: dict) -> Graph:
        return Graph(**graph_dict)

    def generate(
        self,
        input_data: InputData,
        entity_context: str = "",
        terms: Optional[list[str]] = None,
        types: Optional[list[EntityType]] = None,
        relation_context: str = "",
        predicate_domain_range: Optional[list[OntologyPredicate]] = None,
        dedup_context: str = "",
        chunk_size: Optional[int] = None,
        deduplication_method: DeduplicateMethod | None = DeduplicateMethod.SEMHASH,
        temperature: float = None,
        output_folder: Optional[str] = None,
    ) -> tuple[Graph, KGGenStats]:
        processed_input: str = input_data.text
        all_chunk_stats = []

        def _process(content, lm):
            with dspy.context(lm=lm):
                step_stats = {}
                self.reset_token_usage()
                start_time = time.time()
                if not terms:
                    entities = get_entities(
                        content,
                        context=entity_context,
                        types=types,
                        temperature=temperature or self.temperature,
                    )
                    step_stats["get_entities"] = StepStats(
                        lm_usage=LMUsage(**self.extract_token_usage_from_history()),
                        execution_time=time.time() - start_time,
                    )
                else:
                    entities = [Entity(surface_form=t) for t in terms]
                    step_stats["get_entities"] = StepStats(
                        lm_usage=LMUsage(), execution_time=0.0
                    )

                self.reset_token_usage()
                start_time = time.time()
                typed_entities = type_terms(
                    input_data=content,
                    terms=[e.surface_form for e in entities],
                    types=types,
                    temperature=temperature or self.temperature,
                    provenance_ids=[input_data.id],
                )
                step_stats["type_terms"] = StepStats(
                    lm_usage=LMUsage(**self.extract_token_usage_from_history()),
                    execution_time=time.time() - start_time,
                )

                self.reset_token_usage()
                start_time = time.time()
                relations = get_relations_typed(
                    content,
                    typed_entities=typed_entities,
                    predicate_domain_range=predicate_domain_range,
                    context=relation_context,
                    temperature=temperature or self.temperature,
                    provenance_ids=[input_data.id],
                )
                step_stats["get_relations_typed"] = StepStats(
                    lm_usage=LMUsage(**self.extract_token_usage_from_history()),
                    execution_time=time.time() - start_time,
                )

                return typed_entities, relations, step_stats

        if not chunk_size:
            try:
                typed_entities, relations, chunk_stats = _process(
                    processed_input, self.lm
                )
                all_chunk_stats.append(chunk_stats)
            except Exception as e:
                if "context length" in str(e).lower():
                    logger.warning(
                        f"Context length error: {e}. Chunking text with chunk size 16384."
                    )
                    chunk_size = 16384
                else:
                    raise e

        if chunk_size:
            chunks = chunk_text(processed_input, chunk_size)
            typed_entities = set()
            relations: list[Relation] = []

            with ThreadPoolExecutor() as executor:
                future_to_chunk = {
                    executor.submit(_process, chunk, self.lm): chunk for chunk in chunks
                }

                for i, future in enumerate(as_completed(future_to_chunk)):
                    chunk_typed_entities, chunk_relations, chunk_stats = future.result()
                    typed_entities.update(chunk_typed_entities)
                    relations.extend(chunk_relations)
                    all_chunk_stats.append(chunk_stats)

        aggregated_stats = {
            "get_entities": StepStats(lm_usage=LMUsage(), execution_time=0.0),
            "type_terms": StepStats(lm_usage=LMUsage(), execution_time=0.0),
            "get_relations_typed": StepStats(lm_usage=LMUsage(), execution_time=0.0),
        }

        for chunk_stats in all_chunk_stats:
            for step_name, step_stat in chunk_stats.items():
                aggregated_stats[step_name] += step_stat

        kg_gen_stats = KGGenStats(**aggregated_stats)

        graph = Graph(
            typed_entities=set(typed_entities),
            relations_wo_class_assertions=relations,
        )

        if deduplication_method:
            graph, dedup_stats = self.deduplicate(
                graph, method=deduplication_method, context=dedup_context
            )
            kg_gen_stats.deduplicate = dedup_stats

        if output_folder:
            self.export_graph(graph, os.path.join(output_folder, "graph.json"))

        return graph, kg_gen_stats

    @deprecated("Use KGGen.deduplicate() method instead")
    def cluster(
        self,
        graph: Graph,
        **kwargs,
    ) -> tuple[Graph, StepStats]:
        return self.deduplicate(graph, **kwargs)

    def deduplicate(
        self,
        graph: Graph,
        method: DeduplicateMethod = DeduplicateMethod.FULL,
        semhash_similarity_threshold: float = 0.95,
        model: str = None,
        temperature: float = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",
    ) -> tuple[Graph, StepStats]:
        if any([model, temperature, api_key, api_base]):
            self.init_model(
                model=model or self.model,
                temperature=temperature or self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
            )

        self.reset_token_usage()
        start_time = time.time()
        graph = run_deduplication(
            lm=self.lm,
            graph=graph,
            method=method,
            retrieval_model=self.retrieval_model,
            semhash_similarity_threshold=semhash_similarity_threshold,
        )
        stats = StepStats(
            lm_usage=LMUsage(**self.extract_token_usage_from_history()),
            execution_time=time.time() - start_time,
        )
        return graph, stats

    def aggregate(self, graphs: list[Graph]) -> Graph:
        all_typed_entities: dict[str, TypedEntity] = {}
        all_relations: dict[str, Relation] = {}
        all_entity_metadata: dict[TypedEntity, set[str]] = {}

        for graph in graphs:
            for entity in graph.typed_entities:
                if entity.surface_form in all_typed_entities:
                    existing_entity = all_typed_entities[entity.surface_form]
                    existing_entity.provenance_ids.extend(entity.provenance_ids)
                else:
                    all_typed_entities[entity.surface_form] = entity

            for relation in graph.relations_wo_class_assertions:
                relation_key = (
                    f"{relation.subject.surface_form}-"
                    f"{relation.predicate.surface_form}-"
                    f"{relation.object.surface_form}"
                )
                if relation_key in all_relations:
                    existing_relation = all_relations[relation_key]
                    existing_relation.provenance_ids.extend(relation.provenance_ids)
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

    def reset_token_usage(self):
        self.lm.history = []

    def extract_token_usage_from_history(self) -> Dict[str, int]:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for entry in self.lm.history:
            if isinstance(entry, dict):
                usage = entry.get("usage") or entry.get("response", {}).get("usage")

                if usage:
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)

        return {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        }

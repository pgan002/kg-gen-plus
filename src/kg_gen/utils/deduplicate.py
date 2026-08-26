import re
import unicodedata
from collections import defaultdict
from typing import Iterable

from semhash.utils import Encoder

from kg_gen.models import Graph, Relation, Entity, TypedEntity, EntityOrSubclass
from semhash import SemHash
import inflect


# Upper-case runs, optionally with dots/ampersands/hyphens: "CBS", "U.S.", "AT&T".
_ACRONYM_RE = re.compile(r"^[A-Z0-9.&-]+$")


class DeduplicateList:
    inflect_engine: inflect.engine
    original_map: dict[str, str]
    items_map: dict[str, str]
    surface_form2entity_map: dict[str, TypedEntity]
    duplicates: dict[str, str]
    deduplicated: list[str]

    # Stats values
    total_items: int
    deduplicated_items: int
    duplicate_items: int
    reduction: float

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.inflect_engine = inflect.engine()
        self.original_map = {}
        self.items_map = {}
        self.surface_form2entity_map = {}
        self.duplicates = {}
        self.deduplicated = []

    def normalize(self, text: str) -> str:
        """
        Normalize a text.
        """
        return unicodedata.normalize("NFKC", text)

    def singularize(self, text: str) -> str:
        """
        Singularize a text.
        """
        # singularize each token when it looks like a plural noun
        return " ".join(self._singularize_token(t) for t in text.split()).strip()

    def _singularize_token(self, token: str) -> str:
        """Singularize one token, leaving alone the things that are not plurals.

        Surface forms are grouped by their singular form *before* any embedding
        comparison, so anything this collapses is merged unconditionally, at any
        threshold. That makes over-eager singularization worse than missing a
        plural: it silently fuses two distinct entities.
        """
        # Numerals are not nouns. Singularizing them turns the decade "1970s"
        # into the year "1970" -- a distinction temporal questions turn on.
        if any(character.isdigit() for character in token):
            return token
        # Acronyms are not plurals: inflect renders "CBS" as "CB".
        if len(token) > 1 and _ACRONYM_RE.match(token):
            return token

        singular = self.inflect_engine.singular_noun(token)
        if not isinstance(singular, str) or not singular:
            return token
        # inflect returns its own False *stringified* for a hyphenated compound
        # whose leading part it cannot singularize, so "Governor-General" and
        # "Secretary-General" both come back as "False-General" and collapse
        # into a single entity. Any hyphenated title sharing a suffix does.
        if "False" in singular and "False" not in token:
            return token
        return singular

    def deduplicate(
        self,
        items: list[TypedEntity] | list[Entity],
        model: Encoder = None,
        use_embeddings: bool = True,
    ):
        """
        Deduplicate a list of items using semantic hashing.
        Before deduplication, items are normalized and singularized.

        Args:
            items: List of items to deduplicate
            model: Encoder for the embedding pass; unused when ``use_embeddings``
                is False.
            use_embeddings: Run the semantic pass on top of the string grouping.
                Off is markedly cheaper -- embedding every surface form is the
                dominant cost of deduplication, and on a 200-document MuSiQue run
                it was 86% of the job's wall clock while accounting for under 2%
                of the merges. See ``run_semhash_deduplication``.
        """
        self.total_items = len(items)

        if not items:
            return

        # 1. Normalize, singularize and case-fold each item's surface form. The
        # result is the grouping key: everything sharing one is merged before any
        # embedding comparison. Case-folding belongs here rather than being left
        # to semantic similarity -- "Person"/"person" is a string fact, and
        # spending an embedding on it is both slower and less reliable.
        singular_to_record = {}
        record_to_singular = {}
        key_to_forms: dict[str, set[str]] = defaultdict(set)

        for item in items:
            normalized = self.normalize(item.surface_form)
            singular = self.singularize(normalized)
            key = singular.casefold()
            self.original_map[item.surface_form] = key
            self.surface_form2entity_map[item.surface_form] = item
            key_to_forms[key].add(item.surface_form)

            if key not in singular_to_record:
                # Include the description in the semantic representation if it
                # exists. The record keeps the original casing: it is what gets
                # embedded, and case carries signal for the encoder.
                record = singular
                description = item.description
                if description:
                    record = f"{singular} - {description}"

                singular_to_record[key] = record
                record_to_singular[record] = key

        # Elect a stable representative per key: the lexicographically smallest
        # surface form, matching how a cluster representative is chosen below.
        # Items arrive from a set, whose iteration order varies with the hash
        # seed, so "last one wins" would make the output differ between runs.
        for key, forms in key_to_forms.items():
            self.items_map[key] = min(forms)

        if not use_embeddings:
            # The grouping above is already a merge: items_map holds one surface
            # form per singular form, so every item sharing a singular form
            # resolves to the same canonical entity. That accounts for the large
            # majority of real-world duplicates ("Cats"/"Cat", the same name
            # typed two ways) at no embedding cost.
            self.deduplicated = list(singular_to_record)
            self.deduplicated_items = len(self.deduplicated)
            self.duplicate_items = self.total_items - self.deduplicated_items
            self.reduction = (
                (self.duplicate_items / self.total_items) * 100
                if self.total_items > 0
                else 0
            )
            return

        # 2. Run semantic hashing
        records_list = list(singular_to_record.values())
        semhash = SemHash.from_records(records=records_list, model=model)
        deduplication_result = semhash.self_deduplicate(threshold=self.threshold)

        self.deduplicated_items = len(deduplication_result.selected)
        self.duplicate_items = len(deduplication_result.filtered)
        self.reduction = (
            (self.duplicate_items / self.total_items) * 100
            if self.total_items > 0
            else 0
        )

        # 3. Build a graph of duplicate relationships
        adj = defaultdict(set)
        for duplicate in deduplication_result.filtered:
            record_str = duplicate.record
            record_singular = record_to_singular[record_str]
            for dup_item_str, _ in duplicate.duplicates:
                dup_item_singular = record_to_singular[dup_item_str]
                adj[record_singular].add(dup_item_singular)
                adj[dup_item_singular].add(record_singular)

        # 4. Find connected components (clusters of duplicates)
        clusters = []
        visited = set()
        for node in adj:
            if node not in visited:
                component = []
                q = [node]
                visited.add(node)
                while q:
                    curr = q.pop(0)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                clusters.append(component)

        # 5. Elect a representative for each cluster and update the mapping
        for cluster in clusters:
            # Deterministically elect a representative (e.g., the first one alphabetically)
            cluster.sort()
            representative_singular = cluster[0]
            canonical_surface_form = self.items_map[representative_singular]

            # Map all members of the cluster to the canonical surface form
            for member_singular in cluster:
                self.items_map[member_singular] = canonical_surface_form

        self.deduplicated = [
            record_to_singular[r] for r in deduplication_result.selected
        ]

    def stats(self) -> str:
        return f"Total items: {self.total_items}; Deduplicated items: {self.deduplicated_items}; Duplicate items: {self.duplicate_items}; Reduction: {self.reduction:.1f}"


def _merge_provenance(id_lists: "Iterable[list[str]]") -> list[str]:
    """Union of provenance id lists, first-seen order, without duplicates.

    Matches what ``KGGen.aggregate`` already does when it merges entities across
    documents, so the two paths agree on what an entity's provenance means. Order
    is preserved to keep output stable between runs.
    """
    merged: dict[str, None] = {}
    for ids in id_lists:
        for provenance_id in ids:
            merged[provenance_id] = None
    return list(merged)


def run_semhash_deduplication(
    graph: Graph,
    model: Encoder = None,
    entity_similarity_threshold: float = 0.9,
    edge_similarity_threshold: float = 0.75,
    deduplicate_edges: bool = True,
    use_embeddings: bool = False,
) -> Graph:
    """
    Deduplicate the graph.

    Args:
        deduplicate_edges: Whether to also semantically cluster predicates. Skip
            this when predicates already come from a fixed ontology vocabulary,
            since they are canonical by construction and clustering them by
            embedding similarity risks merging distinct ontology predicates.
        use_embeddings: Whether to run the semantic pass on top of the string
            grouping. Defaults to off, because it is expensive and buys little.
            Measured on a 200-document MuSiQue run at ``entity_threshold=0.97``:
            embedding every surface form took 56 minutes, 86% of the job's wall
            clock, and produced 38 clusters -- of which 32 merged identical
            surface forms and 3 merged a plural with its singular, both of which
            the string grouping does for free. The 3 that genuinely needed
            embeddings were 2 correct merges and one wrong one ("John Green"
            with his own novel "Looking for Alaska"). Lowering the threshold does
            not help: precision of the additional merges falls to 30-40%.

            Turn it on for a small corpus, or to find alias pairs that string
            matching cannot ("North Yemen" / "Yemen Arab Republic"). Because it
            needs no LLM, it is also cheap to run offline afterwards on a saved
            graph rather than inline.
    """
    # Deduplicate each graph components
    entities_dedup = DeduplicateList(entity_similarity_threshold)
    entities_dedup.deduplicate(
        list(graph.typed_entities), model=model, use_embeddings=use_embeddings
    )
    if deduplicate_edges:
        edges_dedup = DeduplicateList(edge_similarity_threshold)
        edges_dedup.deduplicate(
            list(graph.edges), model=model, use_embeddings=use_embeddings
        )

    def get_canonical_entity(
        entity: EntityOrSubclass, dedup_list: DeduplicateList
    ) -> EntityOrSubclass:
        surface_form = entity.surface_form
        if surface_form not in dedup_list.original_map:
            return entity

        singular_sf = dedup_list.original_map[surface_form]
        canonical_original_sf = dedup_list.items_map[singular_sf]

        return dedup_list.surface_form2entity_map[canonical_original_sf]

    def _get_relation(relation: Relation) -> Relation:
        """
        Get the transformed relation.
        """
        new_subject = get_canonical_entity(relation.subject, entities_dedup)
        new_object = get_canonical_entity(relation.object, entities_dedup)
        if deduplicate_edges:
            new_predicate = get_canonical_entity(relation.predicate, edges_dedup)
            return Relation(
                subject=new_subject, predicate=new_predicate, object=new_object
            )
        else:
            return Relation(
                subject=new_subject, predicate=relation.predicate, object=new_object
            )

    # Deduplicate the graph
    entity2canonical: dict[TypedEntity, TypedEntity] = {
        entity: get_canonical_entity(entity, entities_dedup)
        for entity in graph.typed_entities
    }
    canonical2cluster = defaultdict(list)
    new_entities: list[TypedEntity] = []
    for entity, canonical in entity2canonical.items():
        canonical2cluster[canonical.surface_form].append(entity)
        new_entities.append(canonical)
    relation2canonical: dict[Relation, Relation] = {
        relation: _get_relation(relation) for relation in graph.relations
    }
    canonical_relation2cluster = defaultdict(list)
    for relation, canonical in relation2canonical.items():
        canonical_relation2cluster[canonical].append(relation)
    #
    if deduplicate_edges:
        edge2canonical: dict[Entity, Entity] = {
            edge: get_canonical_entity(edge, edges_dedup) for edge in graph.edges
        }
        canonical_edge2cluster = defaultdict(list)
        for edge, canonical in edge2canonical.items():
            canonical_edge2cluster[canonical.surface_form].append(edge)
    else:
        canonical_edge2cluster = dict()

    # Assign each surviving entity the union of its cluster's provenance.
    #
    # Iterate the canonical objects *once*. ``new_entities`` holds one entry per
    # pre-deduplication entity, so a canonical appears once per cluster member --
    # and since the canonical is itself in its own cluster, re-assigning while
    # walking that list folded its own already-merged list back in on every
    # repeat, inflating provenance with duplicates (a 2-member cluster came out
    # with 3 ids). Dropping the repeats also removes work quadratic in cluster
    # size.
    seen_canonicals: set[int] = set()
    for canonical_entity in new_entities:
        if id(canonical_entity) in seen_canonicals:
            continue
        seen_canonicals.add(id(canonical_entity))
        cluster = canonical2cluster[canonical_entity.surface_form]
        canonical_entity.provenance_ids = _merge_provenance(
            ent.provenance_ids for ent in cluster if isinstance(ent, TypedEntity)
        )
    new_relations = list(set(relation2canonical.values()))
    for canonical_relation in new_relations:
        cluster = canonical_relation2cluster[canonical_relation]
        canonical_relation.provenance_ids = _merge_provenance(
            rel.provenance_ids for rel in cluster
        )

    # Update entity_metadata keys to match deduplicated entity names
    new_entity_metadata: dict[TypedEntity, set[str]] | None = None
    if graph.entity_metadata:
        new_entity_metadata = {}
        for original_entity, metadata_set in graph.entity_metadata.items():
            deduped_entity = get_canonical_entity(original_entity, entities_dedup)
            # Merge metadata sets when entities are deduplicated together
            if deduped_entity in new_entity_metadata:
                new_entity_metadata[deduped_entity].update(metadata_set)
            else:
                new_entity_metadata[deduped_entity] = metadata_set.copy()

    return Graph(
        typed_entities=set(new_entities),
        relations_wo_class_assertions=new_relations,
        entity_clusters={k: vs for k, vs in canonical2cluster.items() if len(vs) > 1},
        edge_clusters={
            k: vs for k, vs in canonical_edge2cluster.items() if len(vs) > 1
        },
        entity_metadata=new_entity_metadata,
    )

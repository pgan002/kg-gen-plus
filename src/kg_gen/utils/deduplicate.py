import unicodedata
from collections import defaultdict
from kg_gen.models import Graph, Relation, Entity, TypedEntity, EntityOrSubclass
from semhash import SemHash
import inflect


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
        tokens = []
        for tok in text.split():
            sing = self.inflect_engine.singular_noun(tok)
            tokens.append(sing if isinstance(sing, str) and sing else tok)
        return " ".join(tokens).strip()

    def deduplicate(self, items: list[TypedEntity] | list[Entity]):
        """
        Deduplicate a list of items using semantic hashing.
        Before deduplication, items are normalized and singularized.

        Args:
            items: List of items to deduplicate
        """
        self.total_items = len(items)

        if not items:
            return

        # 1. Normalize and singularize each item's surface form
        normalized_items = set()
        for item in items:
            normalized = self.normalize(item.surface_form)
            singular = self.singularize(normalized)
            self.original_map[item.surface_form] = singular
            self.items_map[singular] = item.surface_form
            normalized_items.add(singular)
            self.surface_form2entity_map[item.surface_form] = item

        # 2. Run semantic hashing
        semhash = SemHash.from_records(records=list(normalized_items))
        deduplication_result = semhash.self_deduplicate(threshold=self.threshold)

        self.deduplicated_items = len(deduplication_result.selected)
        self.duplicate_items = len(deduplication_result.duplicates)
        self.reduction = (
            (self.duplicate_items / self.total_items) * 100
            if self.total_items > 0
            else 0
        )

        # 3. Build a graph of duplicate relationships
        adj = defaultdict(set)
        for duplicate in deduplication_result.duplicates:
            record = duplicate.record
            for dup_item, _ in duplicate.duplicates:
                adj[record].add(dup_item)
                adj[dup_item].add(record)

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

        self.deduplicated = deduplication_result.selected

    def stats(self) -> str:
        return f"Total items: {self.total_items}; Deduplicated items: {self.deduplicated_items}; Duplicate items: {self.duplicate_items}; Reduction: {self.reduction:.1f}"


def run_semhash_deduplication(
    graph: Graph,
    similarity_threshold: float = 0.95,
) -> Graph:
    """
    Deduplicate the graph.
    """
    # Deduplicate each graph components
    entities_dedup = DeduplicateList(similarity_threshold)
    entities_dedup.deduplicate(list(graph.typed_entities))
    edges_dedup = DeduplicateList(similarity_threshold)
    edges_dedup.deduplicate(list(graph.edges))

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
        new_predicate = get_canonical_entity(relation.predicate, edges_dedup)
        return Relation(subject=new_subject, predicate=new_predicate, object=new_object)

    # Deduplicate the graph
    new_entities = {
        get_canonical_entity(entity, entities_dedup) for entity in graph.entities
    }
    new_relations = [_get_relation(relation) for relation in graph.relations]

    # Remove duplicate relations
    new_relations = list(set(new_relations))

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
        typed_entities=new_entities,
        relations_wo_class_assertions=new_relations,
        entity_metadata=new_entity_metadata,
    )

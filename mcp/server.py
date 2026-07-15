"""kg-gen MCP server.

Exposes the deterministic scaffolding for agent-driven knowledge-graph
extraction — ontology processing, target-type derivation, predicate suggestion,
conformance/schema validation, graph serialization, and cluster-based
deduplication (embeddings propose candidate clusters, the agent reviews them,
a separate tool applies the merge). The cognitive work (NER, entity typing,
relation extraction, cluster review) is performed by the agent, guided by the
``kg-extract`` skill.

Run standalone with::

    fastmcp run mcp/server.py

It is also mounted in-process by the FastAPI app at ``/mcp`` (see app/server.py).
"""

from __future__ import annotations

import os
import sys

# The tool bodies import ``app`` and ``kg_gen``; make sure the repo root (the
# parent of this file's directory) is importable when this module is loaded by
# ``fastmcp run`` from an arbitrary working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Allow ``import tools`` regardless of the launching working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastmcp import FastMCP  # noqa: E402

import tools  # noqa: E402

mcp = FastMCP(
    name="kg-gen",
    instructions=(
        "Deterministic tools for ontology-guided knowledge-graph extraction. "
        "The agent performs NER, entity typing, and relation extraction; these "
        "tools parse the ontology, suggest compatible predicates, validate "
        "conformance, and serialize the final graph. The server is stateless: "
        "pass the ontology Turtle to each tool that needs it."
    ),
)

# Register the deterministic tool bodies. Each is a thin wrapper (no LLM).
mcp.tool(tools.parse_ontology)
mcp.tool(tools.list_target_types)
mcp.tool(tools.suggest_predicates)
mcp.tool(tools.validate_conformance)
mcp.tool(tools.validate_graph_schema)
mcp.tool(tools.serialize_graph)
mcp.tool(tools.convert_ontology)
mcp.tool(tools.suggest_clusters)
mcp.tool(tools.apply_clusters)


if __name__ == "__main__":
    mcp.run()

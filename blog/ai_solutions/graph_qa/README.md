# Graph QA Demo (Streamlit + NetworkX)

This directory contains a minimal working knowledge graph query demo using Streamlit with NetworkX for graph traversal and pattern matching over a toy dataset of people, organizations, and technologies.

## Quick start

* **Python 3.9+**
* **Install deps (CPU-friendly):**

```bash
pip install -U streamlit networkx matplotlib pandas
streamlit run app.py
```

## Architecture

```
+----------------------+     +---------------------------+
| Streamlit Frontend   | <-> | Graph Query Controller    |
| (Query interface)    |     | (Session state)           |
+-----------+----------+     +-------------+-------------+
            |                               |
            v                               v
  +-----------------+           +------------------------+
  | Query Types:    |           | NetworkX DiGraph       |
  | - Find Neighbors|           | (Knowledge base)       |
  | - Find Path     |  -------> | - Nodes (entities)     |
  | - Pattern Match |           | - Edges (relations)    |
  | - Full Graph    |           | - Attributes           |
  +-----------------+           +------------------------+
            |                               |
            v                               v
  Results table + Graph Visualization (Matplotlib)
```

* **In-memory graph**: NetworkX directed graph with labeled edges
* **Query patterns**: Neighbor traversal, shortest path, subgraph pattern matching
* **Visualization**: Spring layout with color-coded nodes by entity type
* **Extensibility**: Can be replaced with Neo4j, Amazon Neptune, or other graph databases

## Technical Summary

### Core Components

1. **Knowledge Graph Structure**
   - Directed graph with typed relations (works_at, uses, knows, expert_in)
   - Sample dataset covering people, companies, and technologies
   - Edge attributes store relationship metadata

2. **Query Operations**
   - **Neighbor queries**: Find incoming/outgoing connections
   - **Path finding**: Shortest path with relation labeling
   - **Pattern matching**: Complex subgraph patterns (e.g., "find people at companies using X")
   - **Full graph view**: Tabular display of all triples

3. **Visualization**
   - Force-directed layout for spatial organization
   - Color-coded nodes by entity type (people=blue, companies=green, tech=yellow)
   - Edge labels showing relationship types
   - Interactive matplotlib plots

## Files

* **app.py**: Streamlit app with graph creation, query interface, and visualization
* **README.md**: This file with setup, architecture, and references

## Demo Behavior

* **Sample graph** includes entities: Alice, Bob, Charlie (people), TechCorp, DataInc (companies), Python, Machine Learning (technologies)
* **Query types**:
  - Select a node to see its neighbors and connections
  - Find shortest path between any two entities
  - Run pattern queries to find complex relationships
  - View full knowledge graph as a table
* **Graph metrics** displayed in sidebar (node count, edge count)
* **Visual layout** updates automatically based on query context

## References

* **NetworkX**: Hagberg, Schult, Swart. "Exploring network structure, dynamics, and function using NetworkX". [https://networkx.org/](https://networkx.org/)
* **Graph Databases**: Robinson, Webber, Eifrem. "Graph Databases" (O'Reilly). [https://neo4j.com/graph-databases-book/](https://neo4j.com/graph-databases-book/)
* **Knowledge Graphs**: Hogan et al. "Knowledge Graphs" (ACM Computing Surveys, 2021). [https://arxiv.org/abs/2003.02320](https://arxiv.org/abs/2003.02320)
* **Streamlit docs**: [https://docs.streamlit.io/](https://docs.streamlit.io/)

## Notes for Hugging Face Space Integration

* **Add requirements.txt** with pinned versions:
  ```
  streamlit==1.28.0
  networkx==3.1
  matplotlib==3.7.2
  pandas==2.0.3
  ```
* **Lightweight deployment**: Pure Python implementation, no GPU needed
* **Multi-demo integration**: Expose as tab in unified Space launcher
* **Extensibility**: For production, consider:
  - Neo4j/Neptune for scalability
  - SPARQL/Cypher query languages
  - Graph embeddings (TransE, Node2Vec) for ML tasks
  - Real-time graph updates and streaming

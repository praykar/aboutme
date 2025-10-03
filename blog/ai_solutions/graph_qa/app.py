import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd

st.set_page_config(page_title="Graph QA Demo", layout="wide")

st.title("🔗 Graph QA Demo")
st.markdown(
    """A minimal knowledge graph query interface demonstrating simple graph traversal 
    and pattern matching over a toy dataset."""
)

# Sidebar
st.sidebar.header("Graph Settings")
query_type = st.sidebar.selectbox(
    "Query Type",
    ["Find Neighbors", "Find Path", "Subgraph Pattern", "Show Full Graph"]
)

# Initialize sample knowledge graph
@st.cache_resource
def create_sample_graph():
    G = nx.DiGraph()
    # Sample knowledge graph: People, Organizations, Technologies
    edges = [
        ("Alice", "works_at", "TechCorp"),
        ("Bob", "works_at", "TechCorp"),
        ("Alice", "knows", "Bob"),
        ("TechCorp", "uses", "Python"),
        ("TechCorp", "uses", "Machine Learning"),
        ("Bob", "expert_in", "Machine Learning"),
        ("Alice", "expert_in", "Python"),
        ("Charlie", "works_at", "DataInc"),
        ("DataInc", "uses", "Machine Learning"),
        ("Charlie", "knows", "Bob"),
    ]
    for src, rel, dst in edges:
        G.add_edge(src, dst, relation=rel)
    return G

G = create_sample_graph()

# Display graph statistics
st.sidebar.metric("Nodes", G.number_of_nodes())
st.sidebar.metric("Edges", G.number_of_edges())

# Query interface
st.subheader("Query Interface")

if query_type == "Find Neighbors":
    node = st.selectbox("Select Node", sorted(G.nodes()))
    if st.button("Find Neighbors"):
        neighbors = list(G.neighbors(node))
        predecessors = list(G.predecessors(node))
        
        st.write(f"**Outgoing connections from {node}:**")
        if neighbors:
            for n in neighbors:
                rel = G[node][n].get('relation', 'related_to')
                st.write(f"- {node} --[{rel}]--> {n}")
        else:
            st.write("No outgoing connections")
        
        st.write(f"**Incoming connections to {node}:**")
        if predecessors:
            for p in predecessors:
                rel = G[p][node].get('relation', 'related_to')
                st.write(f"- {p} --[{rel}]--> {node}")
        else:
            st.write("No incoming connections")

elif query_type == "Find Path":
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Source Node", sorted(G.nodes()))
    with col2:
        target = st.selectbox("Target Node", sorted(G.nodes()))
    
    if st.button("Find Path"):
        try:
            path = nx.shortest_path(G, source, target)
            st.success(f"Path found with {len(path)-1} hops")
            path_str = " → ".join(path)
            st.write(f"**Path:** {path_str}")
            
            # Show relations
            st.write("**Relations:**")
            for i in range(len(path)-1):
                rel = G[path[i]][path[i+1]].get('relation', 'related_to')
                st.write(f"{i+1}. {path[i]} --[{rel}]--> {path[i+1]}")
        except nx.NetworkXNoPath:
            st.error(f"No path found between {source} and {target}")

elif query_type == "Subgraph Pattern":
    st.write("**Pattern: Find all entities that 'work_at' companies that 'use' specific technology**")
    tech = st.selectbox("Select Technology", 
                        [n for n in G.nodes() if any(
                            G[src][n].get('relation') == 'uses' for src in G.predecessors(n)
                        )])
    
    if st.button("Find Pattern"):
        results = []
        for node in G.nodes():
            for neighbor in G.neighbors(node):
                if G[node][neighbor].get('relation') == 'works_at':
                    for tech_node in G.neighbors(neighbor):
                        if tech_node == tech and G[neighbor][tech_node].get('relation') == 'uses':
                            results.append((node, neighbor, tech))
        
        if results:
            st.success(f"Found {len(results)} matches")
            for person, company, technology in results:
                st.write(f"- {person} works at {company}, which uses {technology}")
        else:
            st.info("No matches found for this pattern")

else:  # Show Full Graph
    st.write("**Full Knowledge Graph:**")
    edges_data = []
    for src, dst, data in G.edges(data=True):
        edges_data.append({
            "Source": src,
            "Relation": data.get('relation', 'related_to'),
            "Target": dst
        })
    st.dataframe(pd.DataFrame(edges_data), use_container_width=True)

# Visualization
st.subheader("Graph Visualization")
fig, ax = plt.subplots(figsize=(12, 8))
try:
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes with different colors by type
    node_colors = []
    for node in G.nodes():
        if any(G[node][n].get('relation') == 'works_at' for n in G.neighbors(node) if G.has_edge(node, n)):
            node_colors.append('lightblue')  # People
        elif any(G[node][n].get('relation') == 'uses' for n in G.neighbors(node) if G.has_edge(node, n)):
            node_colors.append('lightgreen')  # Companies
        else:
            node_colors.append('lightyellow')  # Technologies/Skills
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, width=2, alpha=0.6, ax=ax)
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
    
    ax.set_title("Knowledge Graph Structure", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.error(f"Visualization error: {str(e)}")

# Footer
st.markdown("---")
st.caption(
    "This demo uses NetworkX for graph operations. For production systems, "
    "consider Neo4j, Amazon Neptune, or other graph databases with SPARQL/Cypher query languages."
)

import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from torch_geometric.data import Data
import torch
import plotly.express as px

# Lattice Configurations
LATTICE_CONFIGS = {
    "Simple Cubic": {
        "nodes": [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                 (0,0,1), (1,0,1), (0,1,1), (1,1,1)],
        "edges": [(0,1), (0,2), (0,4), (1,3), (1,5),
                 (2,3), (2,6), (3,7), (4,5), (4,6),
                 (5,7), (6,7)],
        "colors": ['#4299e1']*8
    },
    "BCC": {
        "nodes": [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                 (0,0,1), (1,0,1), (0,1,1), (1,1,1),
                 (0.5,0.5,0.5)],
        "edges": [(8,i) for i in range(8)] + [
                 (0,1), (0,2), (0,4), (1,3), (1,5),
                 (2,3), (2,6), (3,7), (4,5), (4,6),
                 (5,7), (6,7)],
        "colors": ['#4299e1']*8 + ['#48bb78']
    },
    "FCC": {
        "nodes": [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                 (0,0,1), (1,0,1), (0,1,1), (1,1,1),
                 (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
        "edges": [(0,8), (1,8), (2,8), (3,8),
                 (0,9), (1,9), (4,9), (5,9),
                 (0,10), (2,10), (4,10), (6,10),
                 (0,1), (0,2), (0,4), (1,3), (1,5),
                 (2,3), (2,6), (3,7), (4,5), (4,6),
                 (5,7), (6,7)],
        "colors": ['#4299e1']*8 + ['#48bb78']*3
    },
    "Octet": {
        "nodes": [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                 (0,0,1), (1,0,1), (0,1,1), (1,1,1),
                 (0.5,0.5,0.5)],
        "edges": [(0,8), (1,8), (2,8), (3,8), (4,8), (5,8), (6,8), (7,8),
                 (0,1), (0,2), (0,4), (1,3), (1,5),
                 (2,3), (2,6), (3,7), (4,5), (4,6),
                 (5,7), (6,7)],
        "colors": ['#4299e1']*8 + ['#48bb78']
    }
}

LATTICE_PROPERTIES = {
    "Simple Cubic": {
        "structural": "Regular cubic arrangement with uniform node distribution",
        "mechanical": "Uniform load distribution, predictable deformation",
        "applications": "Basic structural components, scaffolds, lightweight structures",
        "connectivity": "6 connections per internal node",
        "relative_density": "Low to medium (0.1-0.3)"
    },
    "BCC": {
        "structural": "Body-centered cubic with additional central node",
        "mechanical": "Enhanced load distribution and strength",
        "applications": "Load-bearing structures, energy absorption",
        "connectivity": "8 connections for center node, 4 for corner nodes",
        "relative_density": "Medium (0.2-0.4)"
    },
    "FCC": {
        "structural": "Face-centered cubic with nodes at face centers",
        "mechanical": "High strength-to-weight ratio, isotropic behavior",
        "applications": "High-performance structural applications",
        "connectivity": "12 connections per unit cell",
        "relative_density": "Medium to high (0.3-0.5)"
    },
    "Octet": {
        "structural": "Combination of octahedral and tetrahedral arrangements",
        "mechanical": "High stiffness-to-weight ratio",
        "applications": "Aerospace structures, high-performance materials",
        "connectivity": "12 connections per node",
        "relative_density": "Medium to high (0.3-0.5)"
    }
}

def create_lattice_viz(lattice_type, node_size=12, edge_width=3):
    """Create 3D visualization of lattice structure"""
    config = LATTICE_CONFIGS[lattice_type]
    nodes = config["nodes"]
    edges = config["edges"]
    node_colors = config["colors"]

    # Create edge traces
    edge_x, edge_y, edge_z = [], [], []
    for edge in edges:
        x0, y0, z0 = nodes[edge[0]]
        x1, y1, z1 = nodes[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=edge_width, color='#2d3748'),
        mode='lines',
        hoverinfo='none'
    )

    # Create node trace
    node_trace = go.Scatter3d(
        x=[n[0] for n in nodes],
        y=[n[1] for n in nodes],
        z=[n[2] for n in nodes],
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_colors,
            line=dict(width=1, color='#ffffff'),
            symbol='circle',
            opacity=0.9
        ),
        hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=True, zeroline=True,
                      gridcolor='#E2E8F0', zerolinecolor='#E2E8F0'),
            yaxis=dict(showbackground=False, showgrid=True, zeroline=True,
                      gridcolor='#E2E8F0', zerolinecolor='#E2E8F0'),
            zaxis=dict(showbackground=False, showgrid=True, zeroline=True,
                      gridcolor='#E2E8F0', zerolinecolor='#E2E8F0'),
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def create_graph_representation(lattice_type):
    """Create graph representation of lattice structure"""
    config = LATTICE_CONFIGS[lattice_type]
    nodes = config["nodes"]
    edges = config["edges"]
    
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(edges)
    
    return G, nodes, edges

def create_adjacency_matrix(G):
    """Create adjacency matrix visualization"""
    adj_matrix = nx.to_numpy_array(G)
    fig = go.Figure(data=go.Heatmap(
        z=adj_matrix,
        colorscale=[[0, '#1a365d'], [1, '#ffd700']],
        showscale=False
    ))
    
    fig.update_layout(
        xaxis=dict(title='Node Index', showgrid=False),
        yaxis=dict(title='Node Index', showgrid=False),
        height=300,
        margin=dict(l=40, r=40, t=20, b=40)
    )
    
    return fig

def display_metrics(G):
    """Display graph metrics"""
    col1, col2, col3 = st.columns(3)
    
    metrics = {
        "Average Degree": f"{sum(dict(G.degree()).values())/G.number_of_nodes():.2f}",
        "Graph Density": f"{nx.density(G):.3f}",
        "Connectivity": "Yes" if nx.is_connected(G) else "No"
    }
    
    for col, (metric, value) in zip([col1, col2, col3], metrics.items()):
        col.metric(label=metric, value=value)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Graph-based Lattice Explorer",
        page_icon="🔷",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #f8fafc;
        }
        .main > div {
            padding: 2rem;
        }
        .lattice-info {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Graph-based Lattice Explorer")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        lattice_type = st.selectbox(
            "Select Lattice Type",
            list(LATTICE_CONFIGS.keys())
        )
        
        st.subheader("Visualization Settings")
        node_size = st.slider("Node Size", 5, 20, 12)
        edge_width = st.slider("Edge Width", 1, 5, 3)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("3D Visualization")
        fig = create_lattice_viz(lattice_type, node_size, edge_width)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Graph Analysis")
        G, nodes, edges = create_graph_representation(lattice_type)
        display_metrics(G)
        
        st.subheader("Adjacency Pattern")
        adj_fig = create_adjacency_matrix(G)
        st.plotly_chart(adj_fig, use_container_width=True)
    
    # Properties section
    st.markdown("---")
    st.subheader("Lattice Properties")
    
    props = LATTICE_PROPERTIES[lattice_type]
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("##### Structural Characteristics")
        st.write(props["structural"])
        st.markdown("##### Mechanical Properties")
        st.write(props["mechanical"])
    
    with cols[1]:
        st.markdown("##### Applications")
        st.write(props["applications"])
        st.markdown("##### Connectivity")
        st.write(props["connectivity"])
        st.markdown("##### Relative Density Range")
        st.write(props["relative_density"])

if __name__ == "__main__":
    main()

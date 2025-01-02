import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from torch_geometric.data import Data
import torch
import plotly.express as px

def create_lattice_viz(lattice_type="Simple Cubic"):
    # Define lattice configurations
    lattice_configs = {
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
        }
    }
    
    config = lattice_configs[lattice_type]
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
        line=dict(width=3, color='#2d3748'),
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
            size=12,
            color=node_colors,
            line=dict(width=1, color='#ffffff'),
            symbol='circle',
            opacity=0.9
        ),
        hoverinfo='text'
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Update layout
    fig.update_layout(
        title="",
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
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_graph_representation(lattice_type):
    """Create graph representation for lattice analysis"""
    G = nx.Graph()
    
    # Define node positions and edges based on lattice type
    if lattice_type == "Simple Cubic":
        nodes = [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                (0,0,1), (1,0,1), (0,1,1), (1,1,1)]
        edges = [(0,1), (0,2), (0,4), (1,3), (1,5),
                (2,3), (2,6), (3,7), (4,5), (4,6),
                (5,7), (6,7)]
    else:  # BCC
        nodes = [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                (0,0,1), (1,0,1), (0,1,1), (1,1,1),
                (0.5,0.5,0.5)]
        edges = [(8,i) for i in range(8)] + [
                (0,1), (0,2), (0,4), (1,3), (1,5),
                (2,3), (2,6), (3,7), (4,5), (4,6),
                (5,7), (6,7)]
    
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
        title='Adjacency Matrix',
        xaxis=dict(title='Node Index', showgrid=False),
        yaxis=dict(title='Node Index', showgrid=False),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def display_metrics(G):
    """Display graph metrics in organized layout"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average Degree",
            value=f"{sum(dict(G.degree()).values())/G.number_of_nodes():.2f}"
        )
    
    with col2:
        st.metric(
            label="Graph Density",
            value=f"{nx.density(G):.3f}"
        )
    
    with col3:
        st.metric(
            label="Is Connected",
            value="Yes" if nx.is_connected(G) else "No"
        )

def main():
    # Page config
    st.set_page_config(
        page_title="LatticeViz Pro",
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
        .stMetric {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Graph based lattices")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        lattice_type = st.selectbox(
            "Select Lattice Type",
            ["Simple Cubic", "BCC"]
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("3D Visualization")
        fig = create_lattice_viz(lattice_type)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Structure Analysis")
        G, nodes, edges = create_graph_representation(lattice_type)
        display_metrics(G)
        
        st.subheader("Connectivity Pattern")
        adj_fig = create_adjacency_matrix(G)
        st.plotly_chart(adj_fig, use_container_width=True)
    
    # Properties section
    st.markdown("---")
    st.subheader("Lattice Properties")
    
    properties_col1, properties_col2 = st.columns(2)
    with properties_col1:
        st.markdown("**Structural Properties**")
        properties = {
            "Simple Cubic": "Regular cubic arrangement with uniform properties",
            "BCC": "Enhanced structure with central support node"
        }
        st.write(properties[lattice_type])
    
    with properties_col2:
        st.markdown("**Applications**")
        applications = {
            "Simple Cubic": "Basic structural components and scaffolds",
            "BCC": "Load-bearing applications with enhanced strength"
        }
        st.write(applications[lattice_type])

if __name__ == "__main__":
    main()

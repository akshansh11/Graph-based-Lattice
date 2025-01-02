import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from torch_geometric.data import Data
import torch
import pandas as pd

def create_3d_lattice_viz(lattice_type="Simple Cubic"):
    # Base coordinates for different lattice types
    lattice_configs = {
        "Simple Cubic": {
            "nodes": [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                     (0,0,1), (1,0,1), (0,1,1), (1,1,1)],
            "edges": [(0,1), (0,2), (1,3), (2,3), 
                     (4,5), (4,6), (5,7), (6,7),
                     (0,4), (1,5), (2,6), (3,7)]
        },
        "BCC": {
            "nodes": [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                     (0,0,1), (1,0,1), (0,1,1), (1,1,1),
                     (0.5,0.5,0.5)],  # Center node
            "edges": [(8,0), (8,1), (8,2), (8,3), 
                     (8,4), (8,5), (8,6), (8,7)]
        },
        "FCC": {
            "nodes": [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                     (0,0,1), (1,0,1), (0,1,1), (1,1,1),
                     (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
            "edges": [(0,8), (1,8), (2,8), (3,8),
                     (0,9), (1,9), (4,9), (5,9),
                     (0,10), (2,10), (4,10), (6,10)]
        }
    }
    
    nodes = lattice_configs[lattice_type]["nodes"]
    edges = lattice_configs[lattice_type]["edges"]
    
    # Create traces for visualization
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in edges:
        x0, y0, z0 = nodes[edge[0]]
        x1, y1, z1 = nodes[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # Create edge trace
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = [node[0] for node in nodes]
    node_y = [node[1] for node in nodes]
    node_z = [node[2] for node in nodes]
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=8,
            color=['#1f77b4' if i < 8 else '#d62728' for i in range(len(nodes))],
            symbol='circle',
            line=dict(width=1, color='#888')
        ))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Update layout
    fig.update_layout(
        title=f'{lattice_type} Lattice Structure',
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_graph_representation(lattice_type):
    """Create graph representation for different lattice types"""
    G = nx.Graph()
    
    if lattice_type == "Simple Cubic":
        nodes = [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                (0,0,1), (1,0,1), (0,1,1), (1,1,1)]
        edges = [(0,1), (0,2), (1,3), (2,3), 
                (4,5), (4,6), (5,7), (6,7),
                (0,4), (1,5), (2,6), (3,7)]
    elif lattice_type == "BCC":
        nodes = [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                (0,0,1), (1,0,1), (0,1,1), (1,1,1),
                (0.5,0.5,0.5)]
        edges = [(8,i) for i in range(8)]
    else:  # FCC
        nodes = [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                (0,0,1), (1,0,1), (0,1,1), (1,1,1),
                (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]
        edges = [(0,8), (1,8), (2,8), (3,8),
                (0,9), (1,9), (4,9), (5,9),
                (0,10), (2,10), (4,10), (6,10)]
    
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(edges)
    
    return G, nodes, edges

def main():
    st.set_page_config(layout="wide", page_title="Lattice Structure Explorer")
    
    st.title("Interactive Lattice Structure Explorer")
    
    # Sidebar
    st.sidebar.title("Controls")
    lattice_type = st.sidebar.selectbox(
        "Select Lattice Type",
        ["Simple Cubic", "BCC", "FCC"]
    )
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("3D Visualization")
        fig = create_3d_lattice_viz(lattice_type)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Graph Properties")
        G, nodes, edges = create_graph_representation(lattice_type)
        
        # Display basic graph properties
        st.write(f"Number of nodes: {G.number_of_nodes()}")
        st.write(f"Number of edges: {G.number_of_edges()}")
        st.write(f"Average degree: {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")
        
        # Display connectivity information
        st.write("Connectivity Analysis:")
        st.write(f"Is connected: {nx.is_connected(G)}")
        st.write(f"Graph density: {nx.density(G):.3f}")
        
        # Create a heatmap of the adjacency matrix
        adj_matrix = nx.to_numpy_array(G)
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=adj_matrix,
            colorscale='Viridis',
            showscale=True
        ))
        fig_heatmap.update_layout(
            title='Adjacency Matrix',
            xaxis_title='Node Index',
            yaxis_title='Node Index'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Additional Information Section
    st.markdown("---")
    st.subheader("Lattice Properties")
    
    # Create three columns for properties
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Structural Properties**")
        properties = {
            "Simple Cubic": "Simple cubic arrangement with 8 vertices",
            "BCC": "Body-centered cubic with additional center node",
            "FCC": "Face-centered cubic with nodes at face centers"
        }
        st.write(properties[lattice_type])
        
    with col2:
        st.markdown("**Applications**")
        applications = {
            "Simple Cubic": "Basic structural components, scaffolds",
            "BCC": "Enhanced mechanical properties, energy absorption",
            "FCC": "High strength-to-weight ratio applications"
        }
        st.write(applications[lattice_type])
        
    with col3:
        st.markdown("**Mechanical Behavior**")
        mechanics = {
            "Simple Cubic": "Regular deformation pattern, predictable behavior",
            "BCC": "Better load distribution, improved strength",
            "FCC": "High packing density, superior mechanical properties"
        }
        st.write(mechanics[lattice_type])

if __name__ == "__main__":
    main()

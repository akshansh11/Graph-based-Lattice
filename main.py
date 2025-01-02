import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from torch_geometric.data import Data
import torch
import plotly.express as px

def create_enhanced_lattice_viz(lattice_type="Simple Cubic"):
    # Enhanced lattice configurations with better connectivity
    lattice_configs = {
        "Simple Cubic": {
            "nodes": [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                     (0,0,1), (1,0,1), (0,1,1), (1,1,1)],
            "edges": [(0,1), (0,2), (0,4), (1,3), (1,5),
                     (2,3), (2,6), (3,7), (4,5), (4,6),
                     (5,7), (6,7)],
            "colors": ['#4299e1']*8  # Blue nodes
        },
        "BCC": {
            "nodes": [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                     (0,0,1), (1,0,1), (0,1,1), (1,1,1),
                     (0.5,0.5,0.5)],
            "edges": [(8,i) for i in range(8)] + [
                     (0,1), (0,2), (0,4), (1,3), (1,5),
                     (2,3), (2,6), (3,7), (4,5), (4,6),
                     (5,7), (6,7)],
            "colors": ['#4299e1']*8 + ['#48bb78']  # Blue + Green center
        },
        "Complex": {
            "nodes": [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                     (0,0,1), (1,0,1), (0,1,1), (1,1,1),
                     (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5),
                     (0.5,0.5,1), (0.5,1,0.5), (1,0.5,0.5)],
            "edges": [(i,j) for i in range(8) for j in range(8,14) 
                     if np.sum(np.abs(np.array(nodes[i]) - np.array(nodes[j]))) < 1.1],
            "colors": ['#4299e1']*8 + ['#48bb78']*6  # Blue corners + Green centers
        }
    }
    
    nodes = lattice_configs[lattice_type]["nodes"]
    edges = lattice_configs[lattice_type]["edges"]
    node_colors = lattice_configs[lattice_type]["colors"]
    
    # Create enhanced traces
    edge_trace = go.Scatter3d(
        x=[], y=[], z=[],
        line=dict(width=3, color='#2d3748'),  # Darker edges
        mode='lines',
        hoverinfo='none'
    )
    
    # Add edges
    for edge in edges:
        x0, y0, z0 = nodes[edge[0]]
        x1, y1, z1 = nodes[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
        edge_trace['z'] += (z0, z1, None)
    
    # Create enhanced node trace
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
    
    # Create figure with enhanced styling
    fig = go.Figure(data=[edge_trace, node_trace])
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
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def create_enhanced_adjacency(G):
    adj_matrix = nx.to_numpy_array(G)
    fig = go.Figure(data=go.Heatmap(
        z=adj_matrix,
        colorscale=[[0, '#1a365d'], [1, '#ffd700']],  # Dark blue to gold
        showscale=False
    ))
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        height=400
    )
    
    return fig

def display_metrics(G):
    cols = st.columns(3)
    metrics = {
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Avg Degree": f"{sum(dict(G.degree()).values())/G.number_of_nodes():.2f}",
        "Density": f"{nx.density(G):.3f}",
        "Connected": "Yes" if nx.is_connected(G) else "No",
        "Components": nx.number_connected_components(G)
    }
    
    for i, (metric, value) in enumerate(metrics.items()):
        with cols[i % 3]:
            st.metric(label=metric, value=value)

def main():
    st.set_page_config(layout="wide", page_title="LatticeViz Pro")
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #f8fafc;
        }
        .main > div {
            padding: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("LatticeViz Pro: Advanced Lattice Analysis")
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("Lattice Configuration")
        lattice_type = st.selectbox(
            "Select Structure Type",
            ["Simple Cubic", "BCC", "Complex"]
        )
        
        st.markdown("---")
        st.markdown("### Settings")
        show_metrics = st.checkbox("Show Metrics", value=True)
        show_adjacency = st.checkbox("Show Adjacency Matrix", value=True)
    
    # Main content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("3D Structure Visualization")
        fig = create_enhanced_lattice_viz(lattice_type)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if show_metrics:
            st.subheader("Structure Metrics")
            G, _, _ = create_graph_representation(lattice_type)
            display_metrics(G)
        
        if show_adjacency:
            st.subheader("Adjacency Pattern")
            adj_fig = create_enhanced_adjacency(G)
            st.plotly_chart(adj_fig, use_container_width=True)
    
    # Properties section
    st.markdown("---")
    with st.expander("Structure Properties", expanded=True):
        properties_col1, properties_col2 = st.columns(2)
        
        with properties_col1:
            st.markdown("### Mechanical Properties")
            properties = {
                "Simple Cubic": "Basic cubic structure with uniform properties",
                "BCC": "Enhanced load distribution with center support",
                "Complex": "Advanced topology with optimized strength distribution"
            }
            st.write(properties[lattice_type])
        
        with properties_col2:
            st.markdown("### Applications")
            applications = {
                "Simple Cubic": "Fundamental structural elements",
                "BCC": "Load-bearing applications",
                "Complex": "Optimized mechanical performance"
            }
            st.write(applications[lattice_type])

if __name__ == "__main__":
    main()

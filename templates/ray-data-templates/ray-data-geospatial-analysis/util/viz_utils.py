"""Visualization utilities for geospatial analysis templates."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_geospatial_heatmap(location_df):
    """Create interactive density heatmap of locations."""
    fig = px.density_mapbox(
        location_df, 
        lat='lat', 
        lon='lon',
        radius=10,
        title='Location Density Heatmap',
        mapbox_style='open-street-map',
        zoom=10,
        height=600
    )
    return fig


def create_trip_distance_analysis(location_df):
    """Create trip distance analysis visualizations."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Trip Distance Distribution', 'Distance by Area'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # Trip distance histogram
    trip_distances = location_df['trip_distance'].dropna()
    trip_distances = trip_distances[(trip_distances > 0) & (trip_distances < 50)]
    
    fig.add_trace(
        go.Histogram(x=trip_distances, nbinsx=30, marker_color='skyblue',
                    name="Trip Distance"),
        row=1, col=1
    )
    
    # Geographic scatter
    sample_data = location_df.sample(min(1000, len(location_df)))
    fig.add_trace(
        go.Scattermapbox(
            lat=sample_data['lat'],
            lon=sample_data['lon'],
            mode='markers',
            marker=dict(size=8, color=sample_data['trip_distance'],
                       colorscale='Viridis', showscale=True),
            name="Locations"
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=500, showlegend=False, mapbox_style='open-street-map')
    return fig


def create_spatial_clustering_viz(location_df):
    """Create spatial clustering visualization."""
    # Simple spatial clustering
    location_df = location_df.copy()
    location_df['lat_bucket'] = pd.cut(location_df['lat'], bins=20, labels=False)
    location_df['lon_bucket'] = pd.cut(location_df['lon'], bins=20, labels=False)
    location_df['spatial_cluster'] = location_df['lat_bucket'] * 20 + location_df['lon_bucket']
    
    cluster_sizes = location_df.groupby('spatial_cluster').size().reset_index()
    cluster_sizes.columns = ['cluster', 'count']
    cluster_sizes = cluster_sizes.nlargest(10, 'count')
    
    fig = px.bar(cluster_sizes, x='cluster', y='count',
                title='Top 10 Spatial Clusters by Activity',
                labels={'cluster': 'Cluster ID', 'count': 'Number of Points'})
    
    return fig


def create_3d_scatter(location_df):
    """Create 3D scatter plot for spatial analysis."""
    sample_data = location_df.sample(min(5000, len(location_df)))
    
    fig = go.Figure(data=[go.Scatter3d(
        x=sample_data['lon'],
        y=sample_data['lat'],
        z=sample_data['trip_distance'],
        mode='markers',
        marker=dict(
            size=3,
            color=sample_data['trip_distance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Trip Distance")
        )
    )])
    
    fig.update_layout(
        title='3D Spatial Analysis',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Trip Distance'
        ),
        height=700
    )
    
    return fig


def create_interactive_dashboard(dataset):
    """Create comprehensive interactive Plotly dashboard for geospatial analysis."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    poi_df = dataset.to_pandas()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Geographic Distribution', 'Category Analysis', 
                       'Rating Distribution', 'Metro Comparison'),
        specs=[[{"type": "scattermapbox"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "box"}]]
    )
    
    # 1. Geographic scatter map
    fig.add_trace(
        go.Scattermapbox(
            lat=poi_df['latitude'],
            lon=poi_df['longitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=poi_df['rating'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Rating", x=0.45)
            ),
            text=[f"Name: {name}<br>Category: {cat}<br>Rating: {rating:.1f}" 
                  for name, cat, rating in zip(poi_df['name'], poi_df['category'], poi_df['rating'])],
            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
            name="POIs"
        ),
        row=1, col=1
    )
    
    # 2. Category bar chart
    category_counts = poi_df['category'].value_counts()
    fig.add_trace(
        go.Bar(
            x=category_counts.index,
            y=category_counts.values,
            marker_color='lightblue',
            name="Categories"
        ),
        row=1, col=2
    )
    
    # 3. Rating histogram
    fig.add_trace(
        go.Histogram(
            x=poi_df['rating'],
            nbinsx=20,
            marker_color='lightgreen',
            name="Rating Distribution"
        ),
        row=2, col=1
    )
    
    # 4. Box plot by metro
    for metro in poi_df['metro_area'].unique():
        metro_data = poi_df[poi_df['metro_area'] == metro]
        fig.add_trace(
            go.Box(
                y=metro_data['rating'],
                name=metro,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Interactive Geospatial Analysis Dashboard",
        title_x=0.5,
        height=800,
        showlegend=False,
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=poi_df['latitude'].mean(), lon=poi_df['longitude'].mean()),
            zoom=8
        )
    )
    
    # Update axes titles
    fig.update_xaxes(title_text="Category", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Rating", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Rating", row=2, col=2)
    
    return fig


def create_interactive_heatmap(dataset):
    """Create interactive heatmap visualization."""
    import plotly.express as px
    import pandas as pd
    
    df = dataset.to_pandas()
    
    fig = px.density_mapbox(df, lat='lat', lon='lon', radius=15,
                           title='Interactive Location Heatmap',
                           mapbox_style='open-street-map',
                           zoom=11, height=600)
    
    return fig


def create_3d_density_plot(dataset):
    """Create 3D density visualization."""
    sample_df = dataset.to_pandas().sample(min(5000, len(dataset.to_pandas())))
    
    fig = go.Figure(data=[go.Scatter3d(
        x=sample_df['lon'],
        y=sample_df['lat'],
        z=sample_df.get('trip_distance', 0),
        mode='markers',
        marker=dict(size=3, color=sample_df.get('trip_distance', 0),
                   colorscale='Plasma', showscale=True)
    )])
    
    fig.update_layout(title='3D Spatial Density',
                     scene=dict(xaxis_title='Longitude',
                               yaxis_title='Latitude',
                               zaxis_title='Metric'),
                     height=700)
    
    return fig


def create_poi_category_chart(poi_df):
    """Create interactive POI category distribution chart."""
    category_counts = poi_df['category'].value_counts().head(10)
    
    fig = px.bar(
        x=category_counts.values,
        y=category_counts.index,
        orientation='h',
        title='Top 10 POI Categories',
        labels={'x': 'Number of Locations', 'y': 'Category'},
        color=category_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=500, showlegend=False)
    return fig


def create_metro_comparison(poi_df):
    """Create metro area comparison visualization."""
    metro_stats = poi_df.groupby('metro_area').agg({
        'rating': 'mean',
        'category': 'count'
    }).reset_index()
    metro_stats.columns = ['metro_area', 'avg_rating', 'poi_count']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='POI Count',
        x=metro_stats['metro_area'],
        y=metro_stats['poi_count'],
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='POI Distribution Across Metro Areas',
        xaxis_title='Metro Area',
        yaxis_title='Number of POIs',
        height=500
    )
    
    return fig


def create_rating_distribution(poi_df):
    """Create interactive rating distribution chart."""
    fig = px.histogram(
        poi_df,
        x='rating',
        nbins=20,
        title='POI Rating Distribution',
        labels={'rating': 'Rating', 'count': 'Frequency'},
        color_discrete_sequence=['skyblue']
    )
    
    fig.update_layout(height=400)
    return fig


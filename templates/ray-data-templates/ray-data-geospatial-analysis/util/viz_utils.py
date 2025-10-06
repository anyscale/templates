"""Visualization utilities for geospatial analysis templates."""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_geospatial_dashboard(dataset, sample_size=1000):
    """Generate a comprehensive geospatial data analysis dashboard."""
    # Sample data for analysis
    sample_data = dataset.take(sample_size)
    df = pd.DataFrame(sample_data)
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Geographic Distribution Map
    ax_map = fig.add_subplot(gs[0, :2])
    
    # Create scatter plot of locations
    scatter = ax_map.scatter(df['lon'], df['lat'], c=df['type'].astype('category').cat.codes, 
                           cmap='tab10', alpha=0.6, s=20)
    ax_map.set_title('Geographic Distribution of Locations', fontsize=14, fontweight='bold')
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    ax_map.grid(True, alpha=0.3)
    
    # Add city labels
    city_centers = df.groupby('city')[['lat', 'lon']].mean()
    for city, (lat, lon) in city_centers.iterrows():
        ax_map.annotate(city, (lon, lat), xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       fontsize=8)
    
    # 2. Location Type Distribution
    ax_types = fig.add_subplot(gs[0, 2:])
    type_counts = df['type'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
    
    wedges, texts, autotexts = ax_types.pie(type_counts.values, labels=type_counts.index, 
                                           autopct='%1.1f%%', colors=colors, startangle=90)
    ax_types.set_title('Location Type Distribution', fontsize=12, fontweight='bold')
    
    # 3. City Coverage Analysis
    ax_cities = fig.add_subplot(gs[1, :2])
    city_counts = df['city'].value_counts()
    bars = ax_cities.bar(range(len(city_counts)), city_counts.values,
                        color=plt.cm.viridis(np.linspace(0, 1, len(city_counts))))
    ax_cities.set_title('Location Count by City', fontsize=12, fontweight='bold')
    ax_cities.set_ylabel('Number of Locations')
    ax_cities.set_xticks(range(len(city_counts)))
    ax_cities.set_xticklabels(city_counts.index, rotation=45)
    
    # Add count labels
    for bar, count in zip(bars, city_counts.values):
        height = bar.get_height()
        ax_cities.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                      f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Latitude Distribution
    ax_lat = fig.add_subplot(gs[1, 2:])
    ax_lat.hist(df['lat'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax_lat.axvline(df['lat'].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {df["lat"].mean():.2f}')
    ax_lat.set_title('Latitude Distribution', fontsize=12, fontweight='bold')
    ax_lat.set_xlabel('Latitude (degrees)')
    ax_lat.set_ylabel('Frequency')
    ax_lat.legend()
    ax_lat.grid(True, alpha=0.3)
    
    # 5. Longitude Distribution
    ax_lon = fig.add_subplot(gs[2, :2])
    ax_lon.hist(df['lon'], bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
    ax_lon.axvline(df['lon'].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {df["lon"].mean():.2f}')
    ax_lon.set_title('Longitude Distribution', fontsize=12, fontweight='bold')
    ax_lon.set_xlabel('Longitude (degrees)')
    ax_lon.set_ylabel('Frequency')
    ax_lon.legend()
    ax_lon.grid(True, alpha=0.3)
    
    # 6. Type vs City Heatmap
    ax_heatmap = fig.add_subplot(gs[2, 2:])
    type_city_matrix = pd.crosstab(df['type'], df['city'])
    sns.heatmap(type_city_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax_heatmap)
    ax_heatmap.set_title('Location Types by City', fontsize=12, fontweight='bold')
    ax_heatmap.set_xlabel('City')
    ax_heatmap.set_ylabel('Location Type')
    
    # 7. Geographic Bounds Analysis
    ax_bounds = fig.add_subplot(gs[3, :2])
    ax_bounds.axis('off')
    
    # Calculate geographic bounds
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    
    bounds_text = "Geographic Analysis\n" + "="*40 + "\n"
    bounds_text += f"Total Locations: {len(df):,}\n"
    bounds_text += f"Latitude Range: {lat_min:.2f} to {lat_max:.2f}\n"
    bounds_text += f"Longitude Range: {lon_min:.2f} to {lon_max:.2f}\n"
    bounds_text += f"Latitude Span: {lat_range:.2f}\n"
    bounds_text += f"Longitude Span: {lon_range:.2f}\n"
    bounds_text += f"Cities Covered: {len(df['city'].unique())}\n"
    bounds_text += f"Location Types: {len(df['type'].unique())}\n"
    
    ax_bounds.text(0.05, 0.95, bounds_text, transform=ax_bounds.transAxes, 
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    # 8. Sample Data Table
    ax_table = fig.add_subplot(gs[3, 2:])
    ax_table.axis('off')
    
    # Create sample data table
    sample_df = df.head(8)[['location_id', 'city', 'type', 'lat', 'lon']].copy()
    
    table_text = "Sample Location Data\n" + "="*70 + "\n"
    table_text += f"{'ID':<12} {'City':<10} {'Type':<10} {'Lat':<8} {'Lon':<8}\n"
    table_text += "-"*70 + "\n"
    
    for _, row in sample_df.iterrows():
        table_text += f"{row['location_id']:<12} {row['city']:<10} {row['type']:<10} {row['lat']:<8.2f} {row['lon']:<8.2f}\n"
    
    ax_table.text(0.05, 0.95, table_text, transform=ax_table.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('Geospatial Data Analysis Dashboard', fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.close()
    
    # Print geospatial insights
    print(f" Geospatial Data Insights:")
    print(f"    Geographic coverage: {lat_range:.2f} lat × {lon_range:.2f} lon")
    print(f"    Location density: {len(df):,} points across {len(df['city'].unique())} cities")
    print(f"    Type diversity: {len(df['type'].unique())} different location types")
    print(f"    Coordinate range: ({lat_min:.2f}, {lon_min:.2f}) to ({lat_max:.2f}, {lon_max:.2f})")
    
    return df


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
            text=sample_data['trip_distance'],
            name="Trips"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Trip Distance Analysis',
        mapbox=dict(style='open-street-map', zoom=10),
        height=500
    )
    
    return fig


def create_cluster_visualization(location_df):
    """Create spatial cluster visualization."""
    fig = px.scatter_mapbox(
        location_df,
        lat='lat',
        lon='lon',
        color='cluster_id',
        size='cluster_size',
        hover_data=['location_id', 'city'],
        title='Spatial Clustering Results',
        mapbox_style='open-street-map',
        zoom=10,
        height=600
    )
    return fig


def create_poi_density_map(location_df):
    """Create point-of-interest density visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Location Density', 'Type Distribution'),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Density scatter
    fig.add_trace(
        go.Scatter(
            x=location_df['lon'],
            y=location_df['lat'],
            mode='markers',
            marker=dict(
                size=5,
                color=location_df['type'].astype('category').cat.codes,
                colorscale='Viridis',
                showscale=True
            ),
            text=location_df['type'],
            name="Locations"
        ),
        row=1, col=1
    )
    
    # Type distribution
    type_counts = location_df['type'].value_counts()
    fig.add_trace(
        go.Bar(
            x=type_counts.index,
            y=type_counts.values,
            marker_color='lightblue',
            name="Count"
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=500, showlegend=False,
                     title_text="Point of Interest Analysis")
    
    return fig


def create_route_analysis(route_df):
    """Create route analysis visualization."""
    fig = go.Figure()
    
    # Plot routes
    for route_id in route_df['route_id'].unique():
        route_data = route_df[route_df['route_id'] == route_id]
        
        fig.add_trace(go.Scattermapbox(
            lat=route_data['lat'],
            lon=route_data['lon'],
            mode='lines+markers',
            name=f'Route {route_id}',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Route Analysis',
        mapbox=dict(style='open-street-map', zoom=10),
        height=600,
        showlegend=True
    )
    
    return fig

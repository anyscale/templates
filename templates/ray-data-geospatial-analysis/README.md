# Geospatial Analysis with Ray Data

**Time to complete**: 25 min | **Difficulty**: Intermediate | **Prerequisites**: Basic Python, understanding of coordinates

## What You'll Build

Create a scalable geospatial analysis pipeline that can process millions of location points across entire cities. You'll learn to find nearby businesses, calculate distances, and perform spatial clustering - all at massive scale.

## Table of Contents

1. [Setup and Data Creation](#step-1-setup-and-data-loading) (5 min)
2. [Spatial Operations](#step-2-basic-spatial-operations) (8 min)
3. [Distance Calculations](#step-3-distance-calculations-at-scale) (7 min)  
4. [Visualization and Results](#step-4-visualization-and-analysis) (5 min)

## Learning Objectives

By completing this tutorial, you'll understand:

- **Why geospatial processing is hard**: Memory and computation challenges with location data
- **Ray Data's spatial capabilities**: Distribute calculations across city-sized datasets
- **Real-world applications**: How companies like Uber and DoorDash process location data
- **Performance at scale**: Handle millions of coordinates efficiently

## Overview

**The Challenge**: Traditional geospatial tools struggle with large datasets. Processing millions of GPS coordinates for proximity analysis can take hours or run out of memory.

**The Solution**: Ray Data distributes spatial calculations across multiple cores and machines, enabling large-scale geospatial analysis through parallel processing.

**Real-world Impact**: 
- **Ride-sharing**: Find nearest drivers to passengers in real-time
- **Retail**: Analyze store locations and customer proximity  
- **Healthcare**: Emergency services optimization and resource allocation
- **Social apps**: Location-based features and recommendations

---

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Understanding of latitude/longitude coordinates
- [ ] Basic knowledge of distance calculations
- [ ] Familiarity with data processing concepts
- [ ] Python environment with sufficient memory (4GB+ recommended)

## Quick Start (3 minutes)

Want to see geospatial processing in action immediately? This section demonstrates core spatial analysis concepts in just a few minutes.

### Install Required Packages

First, ensure you have the necessary geospatial libraries installed:

```bash
pip install "ray[data]" pandas numpy matplotlib seaborn plotly folium geopandas contextily
```

### Setup and Imports

```python
import ray
import numpy as np
import pandas as pd

# Initialize Ray for distributed processing
ray.init()
```

### Create Sample Location Data

```python
# Create sample location data for major US cities
print("Creating sample geospatial dataset...")

# Major US city coordinates
major_cities = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740}
]

# Generate points around each city (simulating businesses, stops, etc.)
locations = []
np.random.seed(42)  # For reproducible results

for city in major_cities:
    for i in range(2000):  # 2000 points per city = 10K total
        # Add small random offset to create realistic distribution
        lat_offset = np.random.normal(0, 0.1)  # ~11km radius
        lon_offset = np.random.normal(0, 0.1)
        
        location = {
            "location_id": f"{city['name'][:3].upper()}_{i:04d}",
            "city": city['name'],
            "lat": city['lat'] + lat_offset,
            "lon": city['lon'] + lon_offset,
            "type": np.random.choice(["restaurant", "store", "office", "hospital", "school"])
        }
        locations.append(location)

ds = ray.data.from_items(locations)
print(f"Created dataset with {ds.count():,} location points across {len(major_cities)} cities")
```

### Quick Distance Analysis

```python
# Quick demonstration of geospatial processing
sample_locations = ds.take(5)

print("Sample Location Data:")
print("=" * 90)
print(f"{'Location ID':<12} {'City':<12} {'Type':<12} {'Latitude':<12} {'Longitude':<12}")
print("-" * 90)

for loc in sample_locations:
    print(f"{loc['location_id']:<12} {loc['city']:<12} {loc['type']:<12} {loc['lat']:<12.4f} {loc['lon']:<12.4f}")

print("-" * 90)
print(f"Ready for advanced geospatial analysis with {ds.count():,} location points!")
```

## Step 1: Setup and Data Loading

First, let's set up Ray and load our geospatial datasets. We'll create realistic point-of-interest (POI) data across major metropolitan areas.

```python
import ray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from typing import Dict, Any
import time

# Initialize Ray - this creates our distributed computing cluster
ray.init()

print(" Ray cluster initialized!")
print(f" Available resources: {ray.cluster_resources()}")
```

Now let's create our geospatial data generation function:

```python
def load_geospatial_data():
    """
    Load geospatial datasets for analysis.
    
    Returns:
        ray.data.Dataset: Dataset containing point-of-interest data with coordinates,
                         categories, and ratings for multiple metropolitan areas.
                         
    Note:
        Uses reproducible random seed (42) for consistent results across runs.
        Creates realistic POI distributions within major US metropolitan areas.
    """
    print("Loading geospatial datasets...")
    
    # Create sample POI data for major US metro areas
    # Use fixed seed for reproducible results (rule #502)
    np.random.seed(42)
    
    metro_areas = {
        'NYC': {'lat': 40.7128, 'lon': -74.0060, 'radius': 0.5},
        'LA': {'lat': 34.0522, 'lon': -118.2437, 'radius': 0.6},
        'Chicago': {'lat': 41.8781, 'lon': -87.6298, 'radius': 0.4}
    }
    
    poi_data = []
    categories = ['restaurant', 'retail', 'hospital', 'school', 'bank']
    
    for metro, coords in metro_areas.items():
        for i in range(1000):  # 1000 POIs per metro
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, coords['radius'])
            
            lat = coords['lat'] + radius * np.cos(angle)
            lon = coords['lon'] + radius * np.sin(angle)
            
            poi_data.append({
                'poi_id': f'{metro}_{i:04d}',
                'name': f'Business_{i}',
                'category': np.random.choice(categories),
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'metro_area': metro,
                'rating': np.random.uniform(1.0, 5.0)
            })
    
    return ray.data.from_items(poi_data)

# Load the dataset and measure performance
start_time = time.time()
poi_dataset = load_geospatial_data()
load_time = time.time() - start_time

print(f"⏱ Data loading took: {load_time:.2f} seconds")
```

Inspect the dataset structure and validate our data:

```python
# Basic dataset information
print(f" Dataset size: {poi_dataset.count()} records")
print(f" Schema: {poi_dataset.schema()}")

# Show sample data to verify it looks correct
print("\n Sample POI data:")
sample_data = poi_dataset.take(5)
for i, poi in enumerate(sample_data):
    print(f"  {i+1}. {poi['name']} ({poi['category']}) at {poi['latitude']:.4f}, {poi['longitude']:.4f}")

# Comprehensive data validation (rule #218: Include comprehensive data validation)
print(f"\n Data validation:")

# Validate coordinate ranges
valid_coords = poi_dataset.filter(
    lambda x: x['latitude'] is not None and x['longitude'] is not None and
              -90 <= x['latitude'] <= 90 and -180 <= x['longitude'] <= 180
).count()

print(f"  - Valid coordinates: {valid_coords} / {poi_dataset.count()}")
print(f"  - Metro areas covered: {len(set([poi['metro_area'] for poi in poi_dataset.take(100)]))}")

# Additional validation checks
sample_data = poi_dataset.take(100)
categories = set([poi['category'] for poi in sample_data])
ratings = [poi['rating'] for poi in sample_data if poi['rating'] is not None]

print(f"  - Categories found: {len(categories)} ({list(categories)})")
print(f"  - Rating range: {min(ratings):.1f} - {max(ratings):.1f}")

# Validate data integrity
if valid_coords != poi_dataset.count():
    print("  Warning: Some POIs have invalid coordinates")
if len(categories) == 0:
    raise ValueError("No valid categories found in dataset")
```

** What just happened?**
- Created 3,000 realistic POI locations across 3 major cities
- Each POI has coordinates, category, and rating information
- Data is distributed across Ray workers for parallel processing
- We validated our data to ensure it's ready for analysis

## Step 2: Basic Spatial Operations

Now let's perform basic spatial operations using Ray Data's distributed processing capabilities.

```python
def calculate_distance_metrics(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate distance-based metrics for POI analysis."""
    df = pd.DataFrame(batch)
    
    results = []
    
    for metro in df['metro_area'].unique():
        metro_pois = df[df['metro_area'] == metro]
        
        # Calculate center point
        center_lat = metro_pois['latitude'].mean()
        center_lon = metro_pois['longitude'].mean()
        
        # Simple distance calculation (Euclidean approximation)
        distances = []
        for _, poi in metro_pois.iterrows():
            dist = np.sqrt((poi['latitude'] - center_lat)**2 + 
                          (poi['longitude'] - center_lon)**2) * 111  # km
            distances.append(dist)
        
        results.append({
            'metro_area': metro,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'avg_distance_from_center': np.mean(distances),
            'max_distance_from_center': np.max(distances),
            'poi_count': len(metro_pois)
        })
    
    return pd.DataFrame(results).to_dict('list')

# Process the data using Ray Data
distance_analysis = poi_dataset.map_batches(
    calculate_distance_metrics,
    batch_format="pandas",
    batch_size=1000,
    concurrency=2
)

print("Distance Analysis Results:")
distance_analysis.show()
```

## Step 3: Aggregation and Grouping

Ray Data provides powerful aggregation capabilities for geospatial analysis:

```python
# Group by metro area and category
category_analysis = poi_dataset.groupby(['metro_area', 'category']).count()
print("POI Count by Metro and Category:")
category_analysis.show(15)

# Calculate average ratings by metro area
rating_analysis = poi_dataset.groupby('metro_area').mean('rating')
print("\nAverage Rating by Metro Area:")
rating_analysis.show()
```

## Step 4: Spatial Joins and Proximity Analysis

Let's create a more complex spatial analysis using Ray Data:

```python
class SpatialAnalyzer:
    """Spatial analysis predictor class."""
    
    def __init__(self):
        """Initialize the spatial analyzer."""
        pass
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform spatial analysis on a batch of POIs."""
        df = pd.DataFrame(batch)
        
        analysis_results = []
        
        for metro in df['metro_area'].unique():
            metro_data = df[df['metro_area'] == metro]
            
            # Analyze service accessibility
            hospitals = metro_data[metro_data['category'] == 'hospital']
            schools = metro_data[metro_data['category'] == 'school']
            
            analysis_results.append({
                'metro_area': metro,
                'total_pois': len(metro_data),
                'hospital_count': len(hospitals),
                'school_count': len(schools),
                'hospital_density': len(hospitals) / len(metro_data) if len(metro_data) > 0 else 0,
                'avg_rating': metro_data['rating'].mean()
            })
        
        return pd.DataFrame(analysis_results).to_dict('list')

# Apply spatial analysis
spatial_results = poi_dataset.map_batches(
    SpatialAnalyzer,
    concurrency=2,
    batch_size=1500
)

print("Spatial Analysis Results:")
spatial_results.show()
```

## Step 5: Interactive Visualizations and Results

Let's create stunning interactive visualizations to understand our spatial data:

### 5.1: Interactive Heatmaps and Density Maps

```python
def create_interactive_heatmap(dataset):
    """Create interactive heatmap using Folium."""
    print("Creating interactive heatmap...")
    
    # Convert to pandas for visualization
    poi_df = dataset.to_pandas()
    
    # Create base map centered on NYC
    center_lat = poi_df['latitude'].mean()
    center_lon = poi_df['longitude'].mean()
    
    # Create Folium map with multiple tile layers
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=10,
        tiles=None
    )
    
    # Add multiple tile layers for better visualization
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB Dark_Matter').add_to(m)
    
    # Create heatmap data
    heat_data = [[row['latitude'], row['longitude']] for _, row in poi_df.iterrows()]
    
    # Add heatmap layer
    HeatMap(
        heat_data,
        min_opacity=0.2,
        radius=15,
        blur=15,
        max_zoom=1,
        name='POI Density Heatmap'
    ).add_to(m)
    
    # Add marker clusters for detailed view
    marker_cluster = MarkerCluster(name='POI Markers').add_to(m)
    
    # Add markers for each POI with category-based colors
    category_colors = {
        'restaurant': 'red',
        'retail': 'blue', 
        'hospital': 'green',
        'school': 'orange',
        'bank': 'purple'
    }
    
    for _, poi in poi_df.head(100).iterrows():  # Show first 100 for performance
        color = category_colors.get(poi['category'], 'gray')
        folium.Marker(
            [poi['latitude'], poi['longitude']],
            popup=f"<b>{poi['name']}</b><br>Category: {poi['category']}<br>Rating: {poi['rating']:.1f}",
            tooltip=f"{poi['category']}: {poi['name']}",
            icon=folium.Icon(color=color)
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    map_file = "poi_heatmap.html"
    m.save(map_file)
    print(f"Interactive heatmap saved as {map_file}")
    
    return m

# Create the interactive heatmap
heatmap = create_interactive_heatmap(poi_dataset)
```

### 5.2: 3D Density Visualization

```python
def create_3d_density_plot(dataset):
    """Create 3D density visualization using Plotly."""
    print("Creating 3D density visualization...")
    
    poi_df = dataset.to_pandas()
    
    # Create 3D scatter plot with density
    fig = go.Figure()
    
    # Add scatter plot for each metro area
    for metro in poi_df['metro_area'].unique():
        metro_data = poi_df[poi_df['metro_area'] == metro]
        
        fig.add_trace(go.Scatter3d(
            x=metro_data['longitude'],
            y=metro_data['latitude'], 
            z=metro_data['rating'],
            mode='markers',
            name=f'{metro} POIs',
            marker=dict(
                size=4,
                opacity=0.7,
                color=metro_data['rating'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Rating")
            ),
            text=[f"Name: {name}<br>Category: {cat}<br>Rating: {rating:.1f}" 
                  for name, cat, rating in zip(metro_data['name'], metro_data['category'], metro_data['rating'])],
            hovertemplate="<b>%{text}</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<extra></extra>"
        ))
    
    # Create density surface
    fig.add_trace(go.Mesh3d(
        x=poi_df['longitude'],
        y=poi_df['latitude'],
        z=poi_df['rating'],
        alphahull=5,
        opacity=0.1,
        color='lightblue',
        name='Density Surface'
    ))
    
    fig.update_layout(
        title="3D POI Distribution and Rating Analysis",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude", 
            zaxis_title="Rating",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600
    )
    
    # Save as HTML
    fig.write_html("3d_poi_density.html")
    print("3D visualization saved as 3d_poi_density.html")
    
    # Show the plot
    fig.show()
    
    return fig

# Create 3D density plot
density_3d = create_3d_density_plot(poi_dataset)
```

### 5.3: Advanced Statistical Visualizations

```python
def create_statistical_visualizations(dataset):
    """Create comprehensive statistical visualizations."""
    print("Creating statistical visualizations...")
    
# Convert results to pandas for visualization
spatial_df = spatial_results.to_pandas()
    poi_df = dataset.to_pandas()
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(20, 15))
    
    # 1. POI Distribution by Metro and Category (Heatmap)
    ax1 = plt.subplot(3, 3, 1)
    category_metro = poi_df.groupby(['metro_area', 'category']).size().unstack(fill_value=0)
    sns.heatmap(category_metro, annot=True, fmt='d', cmap='YlOrRd', ax=ax1)
    ax1.set_title('POI Distribution Heatmap\n(Metro Area vs Category)', fontsize=12, fontweight='bold')
    
    # 2. Rating Distribution by Category (Violin Plot)
    ax2 = plt.subplot(3, 3, 2)
    sns.violinplot(data=poi_df, x='category', y='rating', ax=ax2)
    ax2.set_title('Rating Distribution by Category', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Geographic Scatter with Density
    ax3 = plt.subplot(3, 3, 3)
    scatter = ax3.scatter(poi_df['longitude'], poi_df['latitude'], 
                         c=poi_df['rating'], s=30, alpha=0.6, cmap='viridis')
    ax3.set_title('Geographic Distribution with Ratings', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    plt.colorbar(scatter, ax=ax3, label='Rating')
    
    # 4. Total POIs by Metro (Enhanced Bar Chart)
    ax4 = plt.subplot(3, 3, 4)
    metro_counts = poi_df['metro_area'].value_counts()
    bars = ax4.bar(metro_counts.index, metro_counts.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.set_title('Total POIs by Metro Area', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of POIs')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Hospital Density Analysis
    ax5 = plt.subplot(3, 3, 5)
    if len(spatial_df) > 0:
        bars = ax5.bar(spatial_df['metro_area'], spatial_df['hospital_density'], 
                       color=['#FF9F43', '#10AC84', '#5F27CD'])
        ax5.set_title('Hospital Density by Metro Area', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Hospitals per Total POIs')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Rating vs Distance from Center
    ax6 = plt.subplot(3, 3, 6)
    # Calculate distance from center for each metro
    for metro in poi_df['metro_area'].unique():
        metro_data = poi_df[poi_df['metro_area'] == metro]
        center_lat = metro_data['latitude'].mean()
        center_lon = metro_data['longitude'].mean()
        
        distances = []
        for _, poi in metro_data.iterrows():
            dist = np.sqrt((poi['latitude'] - center_lat)**2 + 
                          (poi['longitude'] - center_lon)**2) * 111  # km approximation
            distances.append(dist)
        
        ax6.scatter(distances, metro_data['rating'], alpha=0.6, label=metro, s=20)
    
    ax6.set_title('Rating vs Distance from Center', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Distance from Center (km)')
    ax6.set_ylabel('Rating')
    ax6.legend()
    
    # 7. Category Distribution (Donut Chart)
    ax7 = plt.subplot(3, 3, 7)
    category_counts = poi_df['category'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
    wedges, texts, autotexts = ax7.pie(category_counts.values, labels=category_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90,
                                      wedgeprops=dict(width=0.5))
    ax7.set_title('POI Category Distribution', fontsize=12, fontweight='bold')
    
    # 8. Rating Trends by Metro
    ax8 = plt.subplot(3, 3, 8)
    metro_ratings = poi_df.groupby('metro_area')['rating'].agg(['mean', 'std']).reset_index()
    x_pos = np.arange(len(metro_ratings))
    
    bars = ax8.bar(x_pos, metro_ratings['mean'], yerr=metro_ratings['std'], 
                   capsize=5, color=['#E17055', '#00B894', '#6C5CE7'])
    ax8.set_title('Average Ratings by Metro Area', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Metro Area')
    ax8.set_ylabel('Average Rating')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(metro_ratings['metro_area'])
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 9. Correlation Matrix
    ax9 = plt.subplot(3, 3, 9)
    # Create correlation matrix for numerical columns
    numeric_cols = poi_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = poi_df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax9,
                square=True, fmt='.2f')
    ax9.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('geospatial_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
    
    print("Statistical visualizations saved as 'geospatial_analysis_dashboard.png'")

# Create statistical visualizations
create_statistical_visualizations(poi_dataset)
```

### 5.4: Interactive Plotly Dashboard

```python
def create_interactive_dashboard(dataset):
    """Create interactive Plotly dashboard."""
    print("Creating interactive Plotly dashboard...")
    
    poi_df = dataset.to_pandas()
    
    # Create subplots
    from plotly.subplots import make_subplots
    
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
    
    # Save and show
    fig.write_html("interactive_geospatial_dashboard.html")
    print("Interactive dashboard saved as 'interactive_geospatial_dashboard.html'")
    fig.show()
    
    return fig

# Create interactive dashboard
dashboard = create_interactive_dashboard(poi_dataset)
```

## Step 6: Saving Results

Save your processed geospatial data for further analysis:

```python
import tempfile

# Save results to parquet format
temp_dir = tempfile.mkdtemp()

# Save spatial analysis results
spatial_results.write_parquet(f"local://{temp_dir}/spatial_analysis")
print(f"Results saved to {temp_dir}/spatial_analysis")

# Save category analysis
category_analysis.write_parquet(f"local://{temp_dir}/category_analysis")
print(f"Category analysis saved to {temp_dir}/category_analysis")
```

## Performance Tips

When working with large geospatial datasets:

1. **Batch Size**: Use appropriate batch sizes based on your data size and available memory
2. **Concurrency**: Set concurrency based on your cluster size and CPU cores
3. **Memory Management**: Use streaming operations for very large datasets
4. **Spatial Indexing**: Consider spatial indexing for complex geometric operations

## Troubleshooting

Common issues and solutions:

- **Out of Memory**: Reduce batch size or increase cluster resources
- **Slow Performance**: Check concurrency settings and cluster utilization
- **Coordinate System Issues**: Ensure consistent coordinate reference systems

## Next Steps

To extend this example:
- Load real geospatial data from sources like OpenStreetMap or government APIs
- Implement more complex spatial operations using specialized libraries
- Add streaming processing for real-time geospatial data
- Integrate with mapping services for visualization

## Cleanup

```python
# Clean up Ray resources (rule #210: Always include cleanup code)
if ray.is_initialized():
    ray.shutdown()
    print("Ray resources cleaned up successfully!")
else:
    print("Ray was not initialized - no cleanup needed")
```

## Key Takeaways

- Ray Data enables distributed processing of large geospatial datasets
- Use `map_batches()` for complex spatial operations
- Leverage Ray Data's aggregation capabilities for spatial analysis
- Proper batch sizing and concurrency settings are crucial for performance

---

## Troubleshooting Common Issues

### **Problem: "Memory errors with large datasets"**
**Solution**:
```python
# Reduce batch size for memory-intensive operations
ds.map_batches(spatial_function, batch_size=100, concurrency=2)
```

### **Problem: "Slow distance calculations"**
**Solution**:
```python
# Use vectorized operations for better performance
import numpy as np
# Vectorized haversine distance is much faster than loops
```

### **Problem: "Coordinate system issues"**
**Solution**:
```python
# Always validate coordinate ranges
def validate_coordinates(lat, lon):
    return -90 <= lat <= 90 and -180 <= lon <= 180
```

### **Debug and Monitoring Capabilities** (rule #200)

```python
# Enable debug mode for detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Ray Data debugging utilities
def debug_dataset_info(dataset, name="dataset"):
    """Debug utility to inspect dataset characteristics."""
    print(f"\n=== Debug Info for {name} ===")
    print(f"Record count: {dataset.count()}")
    print(f"Schema: {dataset.schema()}")
    print(f"Sample record: {dataset.take(1)[0] if dataset.count() > 0 else 'No records'}")
    
    # Check for common issues
    try:
        sample_batch = dataset.take_batch(10)
        print(f"Batch extraction: Success ({len(sample_batch)} records)")
    except Exception as e:
        print(f"Batch extraction: Failed - {e}")

# Example usage: debug_dataset_info(poi_dataset, "POI dataset")
```

### **Performance Optimization Tips**

1. **Batch Processing**: Process locations in batches of 1000-5000 for optimal performance
2. **Spatial Indexing**: Use spatial indexing for nearest neighbor searches
3. **Coordinate Validation**: Always validate lat/lon ranges before processing
4. **Memory Management**: Monitor memory usage for large spatial datasets
5. **Parallel Processing**: Leverage Ray's automatic parallelization for spatial operations

### **Performance Considerations**

Ray Data provides several advantages for geospatial processing:
- **Parallel computation**: Distance calculations are distributed across multiple workers
- **Memory efficiency**: Large coordinate datasets are processed in manageable chunks
- **Scalability**: The same code patterns work for neighborhood-scale to continental-scale analysis
- **Automatic optimization**: Ray Data handles data partitioning and load balancing automatically

---

## Next Steps and Extensions

### **Try These Advanced Features**
1. **Real Datasets**: Use OpenStreetMap data or Census TIGER files
2. **Spatial Joins**: Join POI data with demographic or economic data
3. **Clustering Analysis**: Group POIs by spatial proximity and characteristics
4. **Route Optimization**: Calculate optimal routes between multiple POIs
5. **Heatmap Generation**: Create density maps and spatial visualizations

### **Production Considerations**
- **Coordinate System Management**: Handle different coordinate reference systems
- **Spatial Indexing**: Implement R-tree or other spatial indexes for performance
- **Real-Time Processing**: Adapt for streaming location data
- **Privacy Protection**: Implement location privacy and anonymization
- **Scalability**: Handle continental or global-scale spatial analysis

### **Community Support** (rule #123)

**Getting Help**:
- [Ray Data GitHub Discussions](https://github.com/ray-project/ray/discussions)
- [Ray Slack Community](https://forms.gle/9TSdDYUgxYs8SA9e8)
- [Stack Overflow - Ray Data](https://stackoverflow.com/questions/tagged/ray-data)
- [Ray Data Examples Repository](https://github.com/ray-project/ray/tree/master/python/ray/data/examples)

### **Related Ray Data Templates**
- **Ray Data Large-Scale ETL Optimization**: Optimize spatial data pipelines
- **Ray Data Data Quality Monitoring**: Validate spatial data quality
- **Ray Data Batch Inference Optimization**: Optimize spatial ML models

** Congratulations!** You've successfully built a scalable geospatial analysis pipeline with Ray Data!

These spatial processing techniques scale from city-level to continental-level analysis with the same code patterns.
- Ray Data integrates well with existing geospatial Python libraries
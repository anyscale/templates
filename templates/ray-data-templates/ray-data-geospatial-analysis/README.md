# Geospatial Data Analysis with Ray Data

**⏱️ Time to complete**: 25 min | **Difficulty**: Intermediate | **Prerequisites**: Basic Python, understanding of coordinates

## What You'll Build

Build a geospatial analysis pipeline that processes location points across cities. Learn to find nearby businesses, calculate distances, and perform spatial clustering using Ray Data's distributed processing capabilities.

## Table of Contents

1. [Setup and Data Creation](#step-1-setup-and-data-loading) (5 min)
2. [Spatial Operations](#step-2-basic-spatial-operations) (8 min)
3. [Distance Calculations](#step-3-distance-calculations-at-scale) (7 min)  
4. [Visualization and Results](#step-4-visualization-and-analysis) (5 min)

## Learning objectives

**Why geospatial analytics matters**: Location data supports ride-sharing, delivery, and real estate applications through spatial analysis. Understanding spatial patterns and relationships helps build location-based applications.

**Ray Data's spatial capabilities**: Distribute geographic calculations like spatial joins, clustering, and routing across distributed clusters. You'll learn how Ray Data handles spatial operations with large datasets.

**Real-world location applications**: Techniques used by transportation and delivery companies to process location events. These patterns apply across transportation, logistics, and location-based services.

**Spatial analysis techniques**: Learn geofencing, hot spot detection, route optimization, and location-based recommendations. These techniques support location intelligence applications.

**Deployment patterns**: Location processing, spatial indexing, and geographic data pipeline optimization strategies for geospatial applications.

## Overview

**Challenge**: Processing millions of GPS coordinates, calculating distances for billions of location pairs, and performing spatial joins across large datasets exceeds single-machine capacity. Traditional GIS tools struggle with:
- **Memory constraints**: Loading all coordinates into RAM
- **Sequential processing**: Calculating distances one-by-one
- **Complex spatial joins**: Matching locations within radius
- **Real-time requirements**: Sub-second proximity queries

**Solution**: Ray Data enables distributed geospatial processing that scales spatial calculations across clusters:

| Spatial Operation | Traditional Approach | Ray Data Approach | Scalability |
|-------------------|---------------------|-------------------|-------------|
| **Distance calculations** | Sequential loops | Parallel `map_batches()` | Linear scaling with nodes |
| **Proximity search** | Full dataset scan | Distributed filtering | Handles billions of points |
| **Spatial joins** | Memory-limited | Streaming joins | Unlimited dataset size |
| **Clustering** | Single-machine algorithms | Distributed `groupby()` | Scales to terabytes |

**Ray Data Benefits for Geospatial:**
- ✅ **Distributed haversine calculations**: Process millions of distance computations in parallel
- ✅ **Streaming spatial joins**: Match locations without loading full datasets into memory
- ✅ **Native aggregations**: Use `groupby()` for spatial clustering and zone analysis
- ✅ **Expression API**: Efficient filtering with `col()` and `lit()` for bounding box queries
- ✅ **Pipeline parallelism**: All spatial operations run concurrently for maximum throughput

**Applications**: Uber processes 10M+ trips daily using distributed spatial matching. DoorDash optimizes delivery zones across 10,000+ restaurants using spatial clustering. Zillow analyzes property locations and nearby amenities for 135M+ listings using geospatial joins.

```python
# Example: Real-time spatial matching like Uber/Lyftdef find_nearest_drivers(passenger_location, driver_locations, max_distance_km=5):
    """Find nearest available drivers using efficient spatial operations."""
    
    import math
    
    def calculate_distance(lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points."""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    # Find drivers within range
    nearby_drivers = []
    passenger_lat, passenger_lon = passenger_location
    
    for driver in driver_locations:
        distance = calculate_distance(
            passenger_lat, passenger_lon,
            driver['latitude'], driver['longitude']
        )
        
        if distance <= max_distance_km:
            nearby_drivers.append({
                'driver_id': driver['driver_id'],
                'distance_km': distance,
                'eta_minutes': distance * 2.5  # Estimated time
            })
    
    # Sort by distance
    nearby_drivers.sort(key=lambda x: x['distance_km'])
    
    return nearby_drivers[:3]  # Return top 3 nearest

print("Real-time spatial matching capabilities enabled")
```

This distributed approach enables real-world applications across industries. Ride-sharing platforms find nearest drivers to passengers in real-time using spatial indexing. Retail companies analyze store locations and customer proximity for optimal placement strategies. Healthcare systems optimize emergency services and resource allocation through geographic analysis, while social applications provide location-based features and recommendations using spatial intelligence.

---


### Approach comparison

| Traditional Approach | Ray Data Approach | Key Benefit |
|---------------------|-------------------|-------------|
| **Single-machine processing** | Distributed across cluster | Horizontal scalability |
| **Memory-limited** | Streaming execution | Handle large datasets |
| **Sequential operations** | Pipeline parallelism | Better resource utilization |
| **Manual optimization** | Automatic resource management | Simplified deployment |

## Prerequisites Checklist

Before starting this geospatial analysis template, ensure you have Python 3.8+ with basic geospatial processing experience and understanding of geographic coordinates. Knowledge of GIS concepts like projections and coordinate systems will help you understand the spatial transformations demonstrated.

**Required setup**:
- [ ] Python 3.8+ with geospatial processing experience
- [ ] Understanding of latitude/longitude coordinate systems
- [ ] Access to Ray cluster for distributed processing
- [ ] 8GB+ RAM for processing geographic datasets
- [ ] Optional: Experience with geospatial libraries (GeoPandas, Shapely)

## Quick start (3 minutes)

This section demonstrates core spatial analysis concepts using Ray Data in just a few minutes.

### Install Required Packages

ensure you have the necessary geospatial libraries installed:

```bash
pip install "ray[data]" pandas numpy matplotlib seaborn plotly folium geopandas contextily
```

### Setup and Imports

```python
import numpy as np
import pandas as pd
import ray

# Initialize Ray for distributed processingray.init()

# Configure Ray Data for optimal performance monitoringctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True
```

### Create Sample Location Data

```python
# Create sample location data for major US citiesprint("Creating sample geospatial dataset...")

# Major US city coordinates# Load NYC taxi trip data for geospatial analysistaxi_data = ray.data.read_parquet(
    "s3://ray-benchmark-data/nyc-taxi/yellow_tripdata_2023-01.parquet"
,
    num_cpus=0.025
).limit(50000)

# Extract location points from taxi data using Ray Data map_batchesdef extract_taxi_locations(batch):
    """Extract taxi pickup locations for geospatial analysis."""
    df = pd.DataFrame(batch)
    locations = []
    
    for _, row in df.iterrows():
        # Extract pickup locations with valid NYC coordinates
        if (row.get('pickup_latitude') is not None and 
            row.get('pickup_longitude') is not None and
            40.0 <= row['pickup_latitude'] <= 41.0 and  # Valid NYC latitude range
            -75.0 <= row['pickup_longitude'] <= -73.0):  # Valid NYC longitude range
            
            locations.append({
                'location_id': f"pickup_{row.name}",
                'lat': float(row['pickup_latitude']),
                'lon': float(row['pickup_longitude']),
                'type': 'taxi_pickup',
                'borough': 'NYC',
                'trip_distance': float(row.get('trip_distance', 0))
            })
    
    return locations

# Use Ray Data map_batches for efficient location extraction# Optimize batch size for memory efficiency with large datasetslocation_dataset = taxi_data.map_batches(
    extract_taxi_locations,
    batch_format="pandas",
    batch_size=500,  # Reduced batch size for memory efficiency
    concurrency=4    # Increase concurrency for better parallelization
).flatten()

print(f"Loaded NYC taxi location data: {location_dataset.count():,} location points")

### Nyc Geospatial Analysis Dashboard

Create engaging geospatial visualizations using utility functions:

```python
# Create interactive geospatial visualizations
from util.viz_utils import create_geospatial_heatmap, create_trip_distance_analysis, create_spatial_clustering_viz
import pandas as pd

# Convert to pandas for visualization
location_df = geospatial_dataset.to_pandas()

# 1. Interactive density heatmap
heatmap_fig = create_geospatial_heatmap(location_df)
heatmap_fig.show()

# 2. Trip distance analysis
distance_fig = create_trip_distance_analysis(location_df)
distance_fig.show()

# 3. Spatial clustering visualization
cluster_fig = create_spatial_clustering_viz(location_df)
cluster_fig.show()

# Print summary statistics
trip_distances = location_df['trip_distance'].dropna()
trip_distances = trip_distances[(trip_distances > 0) & (trip_distances < 50)]

print("\nNYC Geospatial Analysis Summary:")
print(f"  Total pickup locations: {len(location_df):,}")
print(f"  Average trip distance: {trip_distances.mean():.2f} miles")
print(f"  Median trip distance: {trip_distances.median():.2f} miles")
print(f"  Latitude range: {location_df['lat'].min():.4f} to {location_df['lat'].max():.4f}")
print(f"  Longitude range: {location_df['lon'].min():.4f} to {location_df['lon'].max():.4f}")
```

This comprehensive geospatial analysis reveals patterns crucial for optimizing ride-sharing operations, delivery routing, and urban planning decisions.
```

### Interactive Geospatial Visualization Dashboard

```python
# Create an engaging geospatial data visualization dashboarddef create_geospatial_dashboard(dataset, sample_size=1000):
    """Generate a comprehensive geospatial data analysis dashboard."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
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
    print(plt.limit(10).to_pandas())
    
    # Print geospatial insights
    print(f" Geospatial Data Insights:")
    print(f"    Geographic coverage: {lat_range:.2f} lat  {lon_range:.2f} lon")
    print(f"    Location density: {len(df):,} points across {len(df['city'].unique())} cities")
    print(f"    Type diversity: {len(df['type'].unique())} different location types")
    print(f"    Coordinate range: ({lat_min:.2f}, {lon_min:.2f}) to ({lat_max:.2f}, {lon_max:.2f})")
    
    return df

# Generate the geospatial dashboardgeospatial_df = create_geospatial_dashboard(geospatial_dataset)
```

**Why This Dashboard Matters:**
- **Geographic Understanding**: Visualize spatial distribution and coverage patterns
- **Data Quality**: Verify coordinate validity and geographic bounds
- **Pattern Recognition**: Identify clustering and distribution patterns across cities
- **Type Analysis**: Understand location type diversity and city coverage

## Step 1: Setup and Data Loading

you'll set up Ray and load our geospatial datasets. you'll create realistic point-of-interest (POI) data across major metropolitan areas.

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

# Initialize Ray - this creates our distributed computing clusterray.init()

# Configure Ray Data for optimal performance monitoringctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True

print(" Ray cluster initialized")
print(f" Available resources: {ray.cluster_resources()}")
```

Now you'll create our geospatial data generation function:

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

# Load the dataset and measure performancestart_time = time.time()
poi_dataset = load_geospatial_data()
load_time = time.time() - start_time

print(f"Data loading took: {load_time:.2f} seconds")
```

Inspect the dataset structure and validate our data:

```python
# Basic dataset informationprint(f" Dataset size: {poi_dataset.count()} records")
print(f" Schema: {poi_dataset.schema()}")

# Show sample data to verify it looks correctprint("\n Sample POI data:")
sample_data = poi_dataset.take(5)
for i, poi in enumerate(sample_data):
    print(f"  {i+1}. {poi['name']} ({poi['category']}) at {poi['latitude']:.4f}, {poi['longitude']:.4f}")

# Comprehensive data validation (rule #218: Include comprehensive data validation)print(f"\n Data validation:")

# Validate coordinate rangesvalid_coords = poi_dataset.filter(
    lambda x: x['latitude'] is not None and x['longitude'] is not None and
              -90 <= x['latitude'] <= 90 and -180 <= x['longitude'] <= 180
).count()

print(f"  - Valid coordinates: {valid_coords} / {poi_dataset.count()}")
print(f"  - Metro areas covered: {len(set([poi['metro_area'] for poi in poi_dataset.take(100)]))}")

# Additional validation checkssample_data = poi_dataset.take(100)
categories = set([poi['category'] for poi in sample_data])
ratings = [poi['rating'] for poi in sample_data if poi['rating'] is not None]

print(f"  - Categories found: {len(categories)} ({list(categories)})")
print(f"  - Rating range: {min(ratings):.1f} - {max(ratings):.1f}")

# Validate data integrityif valid_coords != poi_dataset.count():
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

Now you'll perform basic spatial operations using Ray Data's distributed processing capabilities.

```python
# Optimized: Reduce pandas usage and use native Ray Data operations where possibledef calculate_distance_metrics(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate distance-based metrics with optimized processing."""
    # Group records by metro area using native Python (avoid pandas groupby)
    metro_groups = {}
    for record in batch:
        metro = record.get('metro_area', 'unknown')
        if metro not in metro_groups:
            metro_groups[metro] = []
        metro_groups[metro].append(record)
    
    results = []
    
    for metro, pois in metro_groups.items():
        if not pois:
            continue
            
        # Calculate center point using native operations
        total_lat = sum(poi.get('latitude', 0) for poi in pois)
        total_lon = sum(poi.get('longitude', 0) for poi in pois)
        poi_count = len(pois)
        center_lat = total_lat / poi_count
        center_lon = total_lon / poi_count
        
        # Use vectorized haversine calculation for efficiency
        distances = []
        for poi in pois:
            lat1, lon1 = np.radians(center_lat), np.radians(center_lon)
            lat2, lon2 = np.radians(poi.get('latitude', 0)), np.radians(poi.get('longitude', 0))
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            dist = 6371 * c  # Earth's radius in kilometers
            distances.append(dist)
        
        results.append({
            'metro_area': metro,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'avg_distance_from_center': np.mean(distances),
            'max_distance_from_center': np.max(distances),
            'poi_count': poi_count
        })
    
    return results

# Process with optimized batch processingdistance_analysis = poi_dataset.map_batches(
    calculate_distance_metrics,
    batch_size=2000,    # Larger batch size for efficiency
    concurrency=4       # Increased concurrency
, batch_format="pandas")

print("Distance Analysis Results:")
print(distance_analysis.limit(10).to_pandas())
```

## Step 3: Advanced Spatial Operations with Ray Data

Ray Data provides capable native operations for geospatial analysis. This section demonstrates filtering, grouping, joining, and aggregating spatial data using Ray Data's distributed capabilities.

### Spatial Filtering and Selection

```python
# Best PRACTICE: Use Ray Data expressions API for optimized spatial queriesfrom ray.data.expressions import col, lit

# Find high-rated restaurants in NYC using expressions APIhigh_rated_restaurants = poi_dataset.filter(
    (col('category') == lit('restaurant')) & 
    (col('rating') > lit(4.0)) & 
    (col('metro_area') == lit('NYC'))
)

print(f"High-rated NYC restaurants: {high_rated_restaurants.count()} found")

# Filter POIs within specific geographic bounds using expressionsmanhattan_bounds = poi_dataset.filter(
    (col('latitude') >= lit(40.7000)) & 
    (col('latitude') <= lit(40.8000)) &
    (col('longitude') >= lit(-74.0200)) & 
    (col('longitude') <= lit(-73.9000)) &
    (col('metro_area') == lit('NYC'))
)

print(f"POIs in Manhattan bounds: {manhattan_bounds.count()} locations")
```

### Spatial Aggregations and Grouping

```python
# Use Ray Data's native groupby() for distributed spatial aggregationsprint("Performing distributed spatial aggregations...")

# Group by metro area and category using Ray Data native operationscategory_analysis = poi_dataset.groupby(['metro_area', 'category']).count()
print("POI Count by Metro and Category:")
print(category_analysis.limit(15).to_pandas())

# Calculate spatial statistics by metro areafrom ray.data.aggregate import Mean, Max, Min, Count
spatial_stats = poi_dataset.groupby('metro_area').aggregate(
    Count('poi_id'),
    Mean('rating'),
    Mean('latitude'),
    Mean('longitude')
)

print("\nSpatial Statistics by Metro Area:")
print(spatial_stats.limit(10).to_pandas())

# Advanced aggregation: Category distribution analysiscategory_distribution = poi_dataset.groupby('category').aggregate(
    Count('poi_id'),
    Mean('rating'),
    Max('rating'),
    Min('rating')
)

print("\nCategory Distribution Analysis:")
print(category_distribution.limit(10).to_pandas())
```

## Step 4: Advanced Spatial Joins and Analysis

Now you'll demonstrate Ray Data's capable join operations for complex spatial analysis. you'll create demographic data and join it with our POI data to understand location patterns.

### Creating Demographic Data for Spatial Joins

```python
# Create demographic data that we can join with POI datadef create_demographic_data():
    """Create realistic demographic data for spatial joins."""
    np.random.seed(42)  # Reproducible results
    
    demographics = []
    metro_areas = ['NYC', 'LA', 'Chicago']
    
    for metro in metro_areas:
        for zone_id in range(10):  # 10 zones per metro
            demographics.append({
                'metro_area': metro,
                'zone_id': f'{metro}_zone_{zone_id}',
                'population': np.random.randint(50000, 500000),
                'median_income': np.random.randint(35000, 150000),
                'age_median': np.random.uniform(25, 45),
                'density_per_km2': np.random.randint(1000, 15000)
            })
    
    return ray.data.from_items(demographics)

# Create demographic datasetdemographic_data = create_demographic_data()
print(f"Created demographic data: {demographic_data.count()} zones")
```

### Spatial Joins with Ray Data

```python
# Perform distributed spatial join using Ray Data's native join operationprint("Performing spatial join between POIs and demographics...")

# Join POI data with demographic data by metro areaspatial_join_result = poi_dataset.join(
    demographic_data,
    key='metro_area',  # Join on metro area
    join_type='inner'  # Inner join for complete matches
)

print(f"Spatial join completed: {spatial_join_result.count()} enriched records")

# Show sample of joined datajoined_sample = spatial_join_result.take(3)
for i, record in enumerate(joined_sample):
    print(f"  {i+1}. {record['name']} in {record['metro_area']} (Pop: {record['population']:,}, Income: ${record['median_income']:,})")
```

### Advanced Spatial Analytics with Ray Data

```python
# Use Ray Data's native sort() for geographic rankingprint("Ranking locations by spatial accessibility...")

# Sort POIs by rating within each metro areatop_rated_pois = spatial_join_result.sort(['metro_area', 'rating'], descending=[False, True])

print("Top-rated POIs by metro area:")
top_pois_sample = top_rated_pois.take(10)
for poi in top_pois_sample:
    print(f"  {poi['name']} ({poi['category']}) - Rating: {poi['rating']:.1f} in {poi['metro_area']}")

# Use Ray Data's union() operation to combine datasetsprint("\nCombining multiple geographic datasets...")

# Create additional POI data for demonstrationadditional_pois = ray.data.from_items([
    {'poi_id': 'new_001', 'name': 'Central Park', 'category': 'park', 
     'latitude': 40.7829, 'longitude': -73.9654, 'metro_area': 'NYC', 'rating': 4.8},
    {'poi_id': 'new_002', 'name': 'Golden Gate Bridge', 'category': 'landmark', 
     'latitude': 37.8199, 'longitude': -122.4783, 'metro_area': 'SF', 'rating': 4.9}
])

# Combine datasets using Ray Data's union operationcombined_pois = poi_dataset.union(additional_pois)
print(f"Combined POI dataset: {combined_pois.count()} total locations")

# Use Ray Data's limit() for efficient samplinggeographic_sample = combined_pois.limit(1000)
print(f"Geographic sample created: {geographic_sample.count()} locations")

# Use Ray Data's select() operation to focus on specific columnsspatial_coords = combined_pois.select_columns(['latitude', 'longitude', 'metro_area', 'category'])
print(f"Selected spatial coordinates: {spatial_coords.count()} records")

# Chain multiple Ray Data operations for complex spatial pipelinespatial_pipeline_result = (combined_pois
    .filter(lambda x: x['rating'] > 3.0,
    num_cpus=0.1
)  # Filter high-quality locations
    .groupby('category')                   # Group by POI category
    .count()                              # Count POIs per category
    .sort('count', descending=True)       # Sort by count
    .limit(10)                            # Take top 10 categories
)

print("\nTop POI categories by count:")
print(spatial_pipeline_result.limit(10).to_pandas())
```

### Distributed Spatial Clustering Analysis

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

# Apply spatial analysisspatial_results = poi_dataset.map_batches(
    SpatialAnalyzer,
    concurrency=2,
    batch_size=1500
, batch_format="pandas")

print("Spatial Analysis Results:")
print(spatial_results.limit(10).to_pandas())

# Save spatial analysis results using Ray Data's native write operationsprint("Saving geospatial analysis results...")

# Write enriched POI data to Parquet for efficient storagespatial_join_result.write_parquet("s3://your-bucket/geospatial-analysis/enriched-pois/",
    num_cpus=0.1
)

# Write category analysis results  category_distribution.write_parquet("s3://your-bucket/geospatial-analysis/category-stats/",
    num_cpus=0.1
)

# Write top locations for business intelligencetop_rated_pois.limit(100).write_json("s3://your-bucket/geospatial-analysis/top-locations.json",
    num_cpus=0.1
)

print("Geospatial analysis results saved using Ray Data native write operations")
```

## Step 5: Interactive Visualizations and Results

Create stunning interactive visualizations to understand our spatial data:

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

# Create the interactive heatmapheatmap = create_interactive_heatmap(poi_dataset)
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
    print(fig.limit(10).to_pandas())
    
    return fig

# Create 3D density plotdensity_3d = create_3d_density_plot(poi_dataset)
```

### 5.3: Advanced Statistical Visualizations

```python
def create_statistical_visualizations(dataset):
    """Create comprehensive statistical visualizations."""
    print("Creating statistical visualizations...")
    
# Convert results to pandas for visualizationspatial_df = spatial_results.to_pandas()
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
print(plt.limit(10).to_pandas())
    
    print("Statistical visualizations saved as 'geospatial_analysis_dashboard.png'")

# Create statistical visualizationscreate_statistical_visualizations(poi_dataset)
```

### 5.4: Interactive Plotly Dashboard

```python
# Create interactive dashboard using utility function
from util.viz_utils import create_interactive_dashboard

dashboard = create_interactive_dashboard(poi_dataset)
dashboard.write_html("interactive_geospatial_dashboard.html")
print("Interactive dashboard saved as 'interactive_geospatial_dashboard.html'")
```

## Step 6: Saving Results

Save your processed geospatial data for further analysis:

```python
import tempfile

# Save results to parquet formattemp_dir = tempfile.mkdtemp()

# Save spatial analysis resultsspatial_results.write_parquet(f"local://{temp_dir}/spatial_analysis",
    num_cpus=0.1
)
print(f"Results saved to {temp_dir}/spatial_analysis")

# Save category analysiscategory_analysis.write_parquet(f"local://{temp_dir}/category_analysis",
    num_cpus=0.1
)
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
# Clean up Ray resources when finished with geospatial analysisif ray.is_initialized():
    ray.shutdown()
    print("Ray cluster resources cleaned up successfully")
    print("Geospatial analysis pipeline completed")
```

## Key Takeaways

**Ray Data Native Operations Mastered**:
- `read_parquet(,
    num_cpus=0.025
)` for loading geospatial datasets efficiently
- `map_batches()` for distributed spatial calculations and transformations
- `filter()` for spatial queries and geographic bounds filtering
- `groupby()` and `aggregate()` for spatial statistics and analysis
- `join()` for combining POI data with demographic information
- `sort()` for geographic ranking and accessibility analysis
- `union()` for combining multiple geographic datasets
- `select_columns()` for focusing on spatial coordinates
- `limit()` and `take()` for efficient spatial sampling
- `write_parquet()` and `write_json()` for saving analysis results

**Geospatial Processing Excellence**: Ray Data's distributed architecture enables processing millions of location points with efficient spatial operations thwith large datasets horizontally across clusters while maintaining memory efficiency through streaming execution.

---

## Troubleshooting Common Issues

### Problem: "memory Errors with Large Datasets"
**Solution**:
```python
# Reduce batch size for memory-intensive operationsgeospatial_dataset.map_batches(spatial_function, batch_size=100, concurrency=2, batch_format="pandas")
```

### Problem: "slow Distance Calculations"
**Solution**:
```python
# Use vectorized operations for better performanceimport numpy as np
# Vectorized haversine distance is much faster than loops```

### Problem: "coordinate System Issues"
**Solution**:
```python
# Always validate coordinate rangesdef validate_coordinates(lat, lon):

    """Validate Coordinates."""
    return -90 <= lat <= 90 and -180 <= lon <= 180
```

### Debug and Monitoring Capabilities (rule #200)

```python
# Enable debug mode for detailed loggingimport logging
logging.basicConfig(level=logging.DEBUG)

# Ray Data debugging utilitiesdef debug_dataset_info(dataset, name="dataset"):
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

# Example usage: debug_dataset_info(poi_dataset, "POI dataset")```

## Troubleshooting Common Issues

### Problem: "invalid Coordinate Values"
**Solution**:
```python
# Validate coordinates before processingdef validate_coordinates(batch):

    """Validate Coordinates."""
    valid_records = []
    for record in batch:
        lat = record.get('latitude', 0)
        lon = record.get('longitude', 0)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            valid_records.append(record)
    return valid_records
```

### Problem: "memory Issues with Large Spatial Datasets"
**Solution**:
```python
# Use smaller batch sizes for memory-intensive spatial calculationsdataset.map_batches(spatial_function, batch_size=500, concurrency=2, batch_format="pandas")
```

### Problem: "slow Distance Calculations"
**Solution**:
```python
# Use vectorized operations for better performanceimport numpy as np

def fast_haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation."""
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))
```

### Performance Optimization Tips

1. **Batch Processing**: Process locations in batches of 1000-5000 for optimal performance
2. **Spatial Indexing**: Use spatial indexing for nearest neighbor searches
3. **Coordinate Validation**: Always validate lat/lon ranges before processing
4. **Memory Management**: Monitor memory usage for large spatial datasets
5. **Parallel Processing**: Leverage Ray's automatic parallelization for spatial operations

### Performance Considerations

Ray Data provides several advantages for geospatial processing:
- **Parallel computation**: Distance calculations are distributed across multiple workers
- **Memory efficiency**: Large coordinate datasets are processed in manageable chunks
- **Scalability**: The same code patterns work for neighborhood-scale to continental-scale analysis
- **Automatic optimization**: Ray Data handles data partitioning and load balancing automatically

---

## Next Steps and Extensions

### Try These Advanced Features
1. **Real Datasets**: Use OpenStreetMap data or Census TIGER files
2. **Spatial Joins**: Join POI data with demographic or economic data
3. **Clustering Analysis**: Group POIs by spatial proximity and characteristics
4. **Route Optimization**: Calculate optimal routes between multiple POIs
5. **Heatmap Generation**: Create density maps and spatial visualizations

### Production Considerations
- **Coordinate System Management**: Handle different coordinate reference systems
- **Spatial Indexing**: Implement R-tree or other spatial indexes for performance
- **Real-Time Processing**: Adapt for streaming location data
- **Privacy Protection**: Implement location privacy and anonymization
- **Scalability**: Handle continental or global-scale spatial analysis

### Community Support (rule #123)

**Getting Help**:
- [Ray Data GitHub Discussions](https://github.com/ray-project/ray/discussions)
- [Ray Slack Community](https://forms.gle/9TSdDYUgxYs8SA9e8)
- [Stack Overflow - Ray Data](https://stackoverflow.com/questions/tagged/ray-data)
- [Ray Data Examples Repository](https://github.com/ray-project/ray/tree/master/python/ray/data/examples)

### Related Ray Data Templates
- **Ray Data Large-Scale ETL Optimization**: Optimize spatial data pipelines
- **Ray Data Data Quality Monitoring**: Validate spatial data quality
- **Ray Data Batch Inference Optimization**: Optimize spatial ML models

## Cleanup and Resource Management

Always clean up Ray resources when done:

```python
# Clean up Ray resourcesray.shutdown()
print("Ray cluster shutdown complete")
```

** Congratulations!** You've successfully built a scalable geospatial analysis pipeline with Ray Data!

These spatial processing techniques scale from city-level to continental-level analysis with the same code patterns.
- Ray Data integrates well with existing geospatial Python libraries
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import warnings
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
import pandas as pd
from pygbif import occurrences
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Point, Polygon
from .references_data import REFERENCES
from shapely.geometry import box
from ipyleaflet import SplitMapControl, TileLayer, basemaps


def merge_touching_groups(gdf, buffer_distance=0):
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    gdf = gdf.copy()

    if gdf.crs.to_epsg() != 3395:
        gdf = gdf.to_crs(epsg=3395)

    # Apply small positive buffer if requested (only for matching)
    if buffer_distance > 0:
        gdf["geometry_buffered"] = gdf.geometry.buffer(buffer_distance)
    else:
        gdf["geometry_buffered"] = gdf.geometry

    # Build spatial index on buffered geometry
    sindex = gdf.sindex

    groups = []
    assigned = set()

    for idx, geom in gdf["geometry_buffered"].items():
        if idx in assigned:
            continue
        # Find all polygons that touch or intersect
        possible_matches_index = list(sindex.intersection(geom.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        touching = possible_matches[
            possible_matches["geometry_buffered"].touches(geom)
            | possible_matches["geometry_buffered"].intersects(geom)
        ]

        # Include self
        touching_idxs = set(touching.index.tolist())
        touching_idxs.add(idx)

        # Expand to fully connected group
        group = set()
        to_check = touching_idxs.copy()
        while to_check:
            checking_idx = to_check.pop()
            if checking_idx in group:
                continue
            group.add(checking_idx)
            checking_geom = gdf["geometry_buffered"].loc[checking_idx]
            new_matches_idx = list(sindex.intersection(checking_geom.bounds))
            new_matches = gdf.iloc[new_matches_idx]
            new_touching = new_matches[
                new_matches["geometry_buffered"].touches(checking_geom)
                | new_matches["geometry_buffered"].intersects(checking_geom)
            ]
            new_touching_idxs = set(new_touching.index.tolist())
            to_check.update(new_touching_idxs - group)

        assigned.update(group)
        groups.append(group)

    # Merge geometries and attributes
    merged_records = []
    for group in groups:
        group_gdf = gdf.loc[list(group)]

        # Merge original geometries (NOT buffered ones)
        merged_geom = unary_union(group_gdf.geometry)

        # Aggregate attributes
        record = {}
        for col in gdf.columns:
            if col in ["geometry", "geometry_buffered"]:
                record["geometry"] = merged_geom
            else:
                if np.issubdtype(group_gdf[col].dtype, np.number):
                    record[col] = group_gdf[
                        col
                    ].sum()  # Sum numeric fields like AREA, PERIMETER
                else:
                    record[col] = group_gdf[col].iloc[
                        0
                    ]  # Keep the first value for text/categorical columns

        merged_records.append(record)

    merged_gdf = gpd.GeoDataFrame(merged_records, crs=gdf.crs)

    # Reset warnings filter to default
    warnings.filterwarnings("default", category=RuntimeWarning)

    return merged_gdf


def classify_range_edges(gdf, largest_polygons):
    """
    Classifies polygons into leading (poleward), core, and trailing (equatorward)
    edges within each cluster based on distance from the centroid of the largest polygon within each cluster.
    Includes longitudinal relict detection.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame with 'geometry' and 'cluster' columns.

    Returns:
        GeoDataFrame: The original GeoDataFrame with a new 'category' column.
    """

    # Ensure CRS is in EPSG:3395 (meters)
    if gdf.crs is None or gdf.crs.to_epsg() != 3395:
        gdf = gdf.to_crs(epsg=3395)

    # Compute centroids and extract coordinates
    gdf["centroid"] = gdf.geometry.centroid
    gdf["latitude"] = gdf["centroid"].y
    gdf["longitude"] = gdf["centroid"].x
    gdf["area"] = gdf.geometry.area  # Compute area

    # Find the centroid of the largest polygon within each cluster
    def find_largest_polygon_centroid(sub_gdf):
        largest_polygon = sub_gdf.loc[sub_gdf["area"].idxmax()]
        return largest_polygon["centroid"]

    cluster_centroids = (
        gdf.groupby("cluster")
        .apply(find_largest_polygon_centroid)
        .reset_index(name="cluster_centroid")
    )

    gdf = gdf.merge(cluster_centroids, on="cluster", how="left")

    # Classify polygons within each cluster based on latitude and longitude distance
    def classify_within_cluster(sub_gdf):
        cluster_centroid = sub_gdf["cluster_centroid"].iloc[0]
        cluster_lat = cluster_centroid.y
        cluster_lon = cluster_centroid.x

        largest_polygon_area = largest_polygons[0]["AREA"]

        # Define long_value based on area size
        if largest_polygon_area > 100:
            long_value = 0.5  # for very large polygons, allow 10% longitude diff
        # elif largest_polygon_area > 200:
        # long_value = 1
        else:
            long_value = 0.05  # very small polygons, strict 1% longitude diff

        # Then calculate thresholds
        lat_threshold_01 = 0.1 * cluster_lat
        lat_threshold_05 = 0.05 * cluster_lat
        lat_threshold_02 = 0.02 * cluster_lat
        lon_threshold_01 = long_value * abs(cluster_lon)  # 5% of longitude

        def classify(row):
            lat_diff = row["latitude"] - cluster_lat
            lon_diff = row["longitude"] - cluster_lon

            # Relict by latitude
            if lat_diff <= -lat_threshold_01:
                return "relict (0.01 latitude)"
            # Relict by longitude
            if abs(lon_diff) >= lon_threshold_01:
                return "relict (longitude)"
            # Leading edge (poleward, high latitudes)
            if lat_diff >= lat_threshold_01:
                return "leading (0.99)"
            elif lat_diff >= lat_threshold_05:
                return "leading (0.95)"
            elif lat_diff >= lat_threshold_02:
                return "leading (0.9)"
            # Trailing edge (equatorward, low latitudes)
            elif lat_diff <= -lat_threshold_05:
                return "trailing (0.05)"
            elif lat_diff <= -lat_threshold_02:
                return "trailing (0.1)"
            else:
                return "core"

        sub_gdf["category"] = sub_gdf.apply(classify, axis=1)
        return sub_gdf

    gdf = gdf.groupby("cluster", group_keys=False).apply(classify_within_cluster)

    # Drop temporary columns
    gdf = gdf.drop(
        columns=["centroid", "latitude", "longitude", "area", "cluster_centroid"]
    )

    return gdf


import geopandas as gpd
import numpy as np
from scipy.spatial.distance import cdist


def update_polygon_categories(largest_polygons, classified_polygons):
    island_states_url = "https://raw.githubusercontent.com/anytko/biospat_large_files/main/island_states.geojson"

    # Load island states data
    island_states_gdf = gpd.read_file(island_states_url)
    island_states_gdf = island_states_gdf.to_crs("EPSG:3395")

    # Convert inputs to GeoDataFrames
    largest_polygons_gdf = gpd.GeoDataFrame(largest_polygons, crs="EPSG:3395")
    classified_polygons_gdf = gpd.GeoDataFrame(classified_polygons, crs="EPSG:3395")

    # Add category info to largest polygons
    largest_polygons_gdf = gpd.sjoin(
        largest_polygons_gdf,
        classified_polygons[["geometry", "category"]],
        how="left",
        predicate="intersects",
    )

    # Find polygons from classified set that overlap with island states
    overlapping_polygons = gpd.sjoin(
        classified_polygons_gdf, island_states_gdf, how="inner", predicate="intersects"
    )

    # Clean up overlapping polygons
    overlapping_polygons = overlapping_polygons.rename(
        columns={"index": "overlapping_index"}
    )
    overlapping_polygons_new = overlapping_polygons.drop_duplicates(subset="geometry")

    # Check for empty overlaps before proceeding
    if overlapping_polygons_new.empty:
        print("No overlapping polygons found — returning original classifications.")
        classified_polygons = classified_polygons.to_crs("EPSG:4236")
        return classified_polygons

    # Compute centroids for distance calculation
    overlapping_polygons_new["centroid"] = overlapping_polygons_new.geometry.centroid
    largest_polygons_gdf["centroid"] = largest_polygons_gdf.geometry.centroid

    # Extract coordinates of centroids
    overlapping_centroids = np.array(
        overlapping_polygons_new["centroid"].apply(lambda x: (x.x, x.y)).tolist()
    )
    largest_centroids = np.array(
        largest_polygons_gdf["centroid"].apply(lambda x: (x.x, x.y)).tolist()
    )

    # Compute distance matrix and find closest matches
    distances = cdist(overlapping_centroids, largest_centroids)
    closest_indices = distances.argmin(axis=1)

    # Assign categories from closest large polygons to overlapping polygons
    overlapping_polygons_new["category"] = largest_polygons_gdf.iloc[closest_indices][
        "category"
    ].values

    # Update the classified polygons with new categories
    updated_classified_polygons = classified_polygons_gdf.copy()
    updated_classified_polygons.loc[overlapping_polygons_new.index, "category"] = (
        overlapping_polygons_new["category"]
    )

    # Convert back to EPSG:4326 explicitly
    updated_classified_polygons = updated_classified_polygons.to_crs("EPSG:4326")

    # Ensure the CRS is explicitly set to 4326
    updated_classified_polygons.set_crs("EPSG:4326", allow_override=True, inplace=True)

    return updated_classified_polygons


def assign_polygon_clusters(polygon_gdf):

    island_states_url = "https://raw.githubusercontent.com/anytko/biospat_large_files/main/island_states.geojson"

    # Read the GeoJSON from the URL
    island_states_gdf = gpd.read_file(island_states_url)

    range_test = polygon_gdf.copy()

    # Step 1: Reproject if necessary
    if range_test.crs.is_geographic:
        range_test = range_test.to_crs(epsg=3395)

    range_test = range_test.sort_values(by="AREA", ascending=False)

    largest_polygons = []
    largest_centroids = []
    clusters = []

    # Add the first polygon as part of num_largest with cluster 0
    first_polygon = range_test.iloc[0]

    # Check if the first polygon intersects or touches any island-state polygons
    if (
        not island_states_gdf.intersects(first_polygon.geometry).any()
        and not island_states_gdf.touches(first_polygon.geometry).any()
    ):
        largest_polygons.append(first_polygon)
        largest_centroids.append(first_polygon.geometry.centroid)
        clusters.append(0)

    # Step 2: Loop through the remaining polygons and check area and proximity
    for i in range(1, len(range_test)):
        polygon = range_test.iloc[i]

        # Calculate the area difference between the largest polygon and the current polygon
        area_difference = abs(largest_polygons[0]["AREA"] - polygon["AREA"])

        # Set the polygon threshold dynamically based on the area difference
        if area_difference > 600:
            polygon_threshold = (
                0.2  # Use a smaller threshold (1% of the largest polygon's area)
            )
        elif area_difference > 200:
            polygon_threshold = 0.005
        else:
            polygon_threshold = (
                0.2  # Use a larger threshold (20% of the largest polygon's area)
            )

        # Check if the polygon's area is greater than or equal to the threshold
        if polygon["AREA"] >= polygon_threshold * largest_polygons[0]["AREA"]:

            # Check if the polygon intersects or touches any island-state polygons
            if (
                island_states_gdf.intersects(polygon.geometry).any()
                or island_states_gdf.touches(polygon.geometry).any()
            ):
                continue  # Skip the polygon if it intersects or touches an island-state polygon

            # Calculate the distance between the polygon's centroid and all existing centroids in largest_centroids
            distances = []
            for centroid in largest_centroids:
                lat_diff = abs(polygon.geometry.centroid.y - centroid.y)
                lon_diff = abs(polygon.geometry.centroid.x - centroid.x)

                # If both latitude and longitude difference is below the threshold, this polygon is close
                if lat_diff <= 5 and lon_diff <= 5:
                    distances.append((lat_diff, lon_diff))

            # Check if the polygon is not within proximity threshold
            if not distances:
                # Add to num_largest polygons if it's not within proximity and meets the area condition
                largest_polygons.append(polygon)
                largest_centroids.append(polygon.geometry.centroid)
                clusters.append(
                    len(largest_polygons) - 1
                )  # Assign a new cluster for the new largest polygon
        else:
            pass

    # Step 3: Assign clusters to the remaining polygons based on proximity to largest polygons
    for i in range(len(range_test)):
        polygon = range_test.iloc[i]

        # If the polygon is part of num_largest, it gets its own cluster (already assigned)
        if any(
            polygon.geometry.equals(largest_polygon.geometry)
            for largest_polygon in largest_polygons
        ):
            continue  # Skip, as the num_largest polygons already have their clusters

        # Find the closest centroid in largest_centroids
        closest_centroid_idx = None
        min_distance = float("inf")

        for j, centroid in enumerate(largest_centroids):
            lat_diff = abs(polygon.geometry.centroid.y - centroid.y)
            lon_diff = abs(polygon.geometry.centroid.x - centroid.x)

            distance = np.sqrt(lat_diff**2 + lon_diff**2)  # Euclidean distance
            if distance < min_distance:
                min_distance = distance
                closest_centroid_idx = j

        # Assign the closest cluster
        clusters.append(closest_centroid_idx)

    # Add the clusters as a new column to the GeoDataFrame
    range_test["cluster"] = clusters

    return range_test, largest_polygons


def process_gbif_csv(
    csv_path: str,
    columns_to_keep: list = [
        "species",
        "decimalLatitude",
        "decimalLongitude",
        "year",
        "basisOfRecord",
    ],
) -> dict:
    """
    Processes a GBIF download CSV, filters and cleans it, and returns a dictionary
    of species-specific GeoDataFrames (in memory only).

    Parameters:
    - csv_path (str): Path to the GBIF CSV download (tab-separated).
    - columns_to_keep (list): List of columns to retain from the CSV.

    Returns:
    - dict: Keys are species names (with underscores), values are GeoDataFrames.
    """

    # Load the CSV file
    df = pd.read_csv(csv_path, sep="\t")

    # Filter columns
    df_filtered = df[columns_to_keep]

    # Group by species
    species_grouped = df_filtered.groupby("species")

    # Prepare output dictionary
    species_gdfs = {}

    for species_name, group in species_grouped:
        species_key = species_name.replace(" ", "_")

        # Clean the data
        group_cleaned = group.dropna()
        group_cleaned = group_cleaned.drop_duplicates(
            subset=["decimalLatitude", "decimalLongitude", "year"]
        )

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            group_cleaned,
            geometry=gpd.points_from_xy(
                group_cleaned["decimalLongitude"], group_cleaned["decimalLatitude"]
            ),
            crs="EPSG:4326",
        )

        # Add to dictionary
        species_gdfs[species_key] = gdf

    return species_gdfs


# Generate a smaller gbif df - not recommended but an option


def fetch_gbif_data(species_name, limit=2000):
    """
    Fetches occurrence data from GBIF for a specified species, returning up to a specified limit.

    Parameters:
    - species_name (str): The scientific name of the species to query from GBIF.
    - limit (int, optional): The maximum number of occurrence records to retrieve.
            Defaults to 2000.

    Returns:
    - list[dict]: A list of occurrence records (as dictionaries) containing GBIF data.
    """
    all_data = []
    offset = 0  # Initialize the offset to 0
    page_limit = 300  # GBIF API maximum limit per request

    while len(all_data) < limit:
        # Fetch the data for the current page
        data = occurrences.search(
            scientificName=species_name,
            hasGeospatialIssue=False,
            limit=page_limit,  # Fetch up to 300 records per request
            offset=offset,  # Adjust offset for pagination
            hasCoordinate=True,  # Only include records with coordinates
        )

        # Add the fetched data to the list
        all_data.extend(data["results"])

        # If we have enough data, break out of the loop
        if len(all_data) >= limit:
            break

        # Otherwise, increment the offset for the next page of results
        offset += page_limit  # Increase by 300 each time since that's the max page size

    # Trim the list to exactly the new_limit size if needed
    all_data = all_data[:limit]

    # print(f"Fetched {len(all_data)} records (trimmed to requested limit)")
    return all_data


def convert_to_gdf(euc_data):
    """
    Converts raw GBIF occurrence data into a cleaned GeoDataFrame,
    including geometry, year, and basisOfRecord.

    Parameters:
    - euc_data (list): List of occurrence records (dicts) from GBIF.

    Returns:
    - gpd.GeoDataFrame: Cleaned GeoDataFrame with lat/lon as geometry.
    """
    records = []
    for record in euc_data:
        lat = record.get("decimalLatitude")
        lon = record.get("decimalLongitude")
        year = record.get("year")
        basis = record.get("basisOfRecord")
        scientific_name = record.get("scientificName", "")
        event_date = record.get("eventDate")
        species = " ".join(scientific_name.split()[:2]) if scientific_name else None
        if lat is not None and lon is not None:
            records.append(
                {
                    "species": species,
                    "decimalLatitude": lat,
                    "decimalLongitude": lon,
                    "year": year,
                    "eventDate": event_date,
                    "basisOfRecord": basis,
                    "geometry": Point(lon, lat),
                }
            )

    df = pd.DataFrame(records)

    df["eventDate"] = (
        df["eventDate"].astype(str).str.replace(r"[^0-9\-]", "", regex=True)
    )
    df["eventDate"] = df["eventDate"].str.extract(r"(\d{4}-\d{2}-\d{2})")

    df = df.drop_duplicates(subset=["decimalLatitude", "decimalLongitude", "year"])

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf


# I think go with stand alone functions for now (but keeping GBIF_map class as is for now)


def make_dbscan_polygons_with_points_from_gdf(
    gdf,
    eps=0.008,
    min_samples=3,
    lat_min=6.6,
    lat_max=83.3,
    lon_min=-178.2,
    lon_max=-49.0,
):
    """
    Performs DBSCAN clustering on a GeoDataFrame and returns a GeoDataFrame of
    polygons representing clusters with associated points and years.

    Parameters:
    - gdf (GeoDataFrame): Input GeoDataFrame with 'decimalLatitude', 'decimalLongitude', and 'year' columns.
    - eps (float): Maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    - lat_min, lat_max, lon_min, lon_max (float): Bounding box for filtering points. Default values are set to the extent of North America.

    Returns:
    - expanded_gdf (GeoDataFrame): GeoDataFrame of cluster polygons with retained point geometries and years.
    """

    if "decimalLatitude" not in gdf.columns or "decimalLongitude" not in gdf.columns:
        raise ValueError(
            "GeoDataFrame must contain 'decimalLatitude' and 'decimalLongitude' columns."
        )

    data = gdf.copy()

    # Clean and filter
    df = (
        data[["decimalLatitude", "decimalLongitude", "year", "eventDate"]]
        .drop_duplicates(subset=["decimalLatitude", "decimalLongitude"])
        .dropna(subset=["decimalLatitude", "decimalLongitude", "year"])
    )

    df = df[
        (df["decimalLatitude"] >= lat_min)
        & (df["decimalLatitude"] <= lat_max)
        & (df["decimalLongitude"] >= lon_min)
        & (df["decimalLongitude"] <= lon_max)
    ]

    coords = df[["decimalLatitude", "decimalLongitude"]].values
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="haversine").fit(
        np.radians(coords)
    )
    df["cluster"] = db.labels_

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]),
        crs="EPSG:4326",
    )

    cluster_polygons = {}
    for cluster_id in df["cluster"].unique():
        if cluster_id != -1:
            cluster_points = gdf_points[gdf_points["cluster"] == cluster_id].geometry
            if len(cluster_points) < 3:
                continue
            try:
                valid_points = [pt for pt in cluster_points if pt.is_valid]
                if len(valid_points) < 3:
                    continue
                hull = MultiPoint(valid_points).convex_hull
                if isinstance(hull, Polygon):
                    hull_coords = list(hull.exterior.coords)
                    corner_points = [Point(x, y) for x, y in hull_coords]
                    corner_points = [pt for pt in corner_points if pt in valid_points]
                    if len(corner_points) >= 3:
                        hull = MultiPoint(corner_points).convex_hull
                cluster_polygons[cluster_id] = hull
            except Exception as e:
                print(f"Error creating convex hull for cluster {cluster_id}: {e}")

    expanded_rows = []
    for cluster_id, cluster_polygon in cluster_polygons.items():
        cluster_points = gdf_points[gdf_points["cluster"] == cluster_id]
        for _, point in cluster_points.iterrows():
            if point.geometry.within(cluster_polygon) or point.geometry.touches(
                cluster_polygon
            ):
                expanded_rows.append(
                    {
                        "point_geometry": point["geometry"],
                        "polygon_geometry": cluster_polygon,
                        "year": point["year"],
                        "eventDate": point["eventDate"],
                    }
                )

    expanded_gdf = gpd.GeoDataFrame(
        expanded_rows,
        crs="EPSG:4326",
        geometry=[row["polygon_geometry"] for row in expanded_rows],
    )

    # Set 'geometry' column as active geometry column explicitly
    expanded_gdf.set_geometry("geometry", inplace=True)

    # Drop 'polygon_geometry' as it's no longer needed
    expanded_gdf = expanded_gdf.drop(columns=["polygon_geometry"])

    return expanded_gdf


def get_start_year_from_species(species_name):
    """
    Converts species name to 8-letter key and looks up the start year in REFERENCES.
    If the key is not found, returns 'NA'.
    """
    parts = species_name.strip().lower().split()
    if len(parts) >= 2:
        key = parts[0][:4] + parts[1][:4]
        return REFERENCES.get(key, "NA")
    return "NA"


def prune_by_year(df, start_year=1971, end_year=2025):
    """
    Prune a DataFrame to only include rows where 'year' is between start_year and end_year (inclusive).

    Parameters:
    - df: pandas.DataFrame or geopandas.GeoDataFrame with a 'year' column
    - start_year: int, start of the year range (default 1971)
    - end_year: int, end of the year range (default 2025)

    Returns:
    - pruned DataFrame
    """
    if "year" not in df.columns:
        raise ValueError("DataFrame must have a 'year' column.")

    pruned_df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    return pruned_df


def assign_polygon_clusters_gbif(polygon_gdf):

    island_states_url = "https://raw.githubusercontent.com/anytko/biospat_large_files/main/island_states.geojson"

    island_states_gdf = gpd.read_file(island_states_url)

    # Simplify geometries to avoid precision issues (optional)
    polygon_gdf["geometry"] = polygon_gdf.geometry.simplify(
        tolerance=0.001, preserve_topology=True
    )

    range_test = polygon_gdf.copy()

    # Transform to CRS for area calculation
    if range_test.crs.is_geographic:
        range_test = range_test.to_crs(epsg=3395)

    range_test["AREA"] = range_test.geometry.area / 1e6  # Calculate area first
    range_test = range_test.sort_values(by="AREA", ascending=False)

    largest_polygons = []
    largest_centroids = []
    clusters = []

    first_polygon = range_test.iloc[0]

    if (
        not island_states_gdf.intersects(first_polygon.geometry).any()
        and not island_states_gdf.touches(first_polygon.geometry).any()
    ):
        largest_polygons.append(first_polygon)
        largest_centroids.append(first_polygon.geometry.centroid)
        clusters.append(0)

    for i in range(1, len(range_test)):
        polygon = range_test.iloc[i]
        area_difference = abs(largest_polygons[0]["AREA"] - polygon["AREA"])

        polygon_threshold = 0.5  # Default threshold

        # if area_difference > 10000:
        # polygon_threshold = 0.2
        # else:
        # polygon_threshold = 0.5

        if polygon["AREA"] >= polygon_threshold * largest_polygons[0]["AREA"]:
            if (
                island_states_gdf.intersects(polygon.geometry).any()
                or island_states_gdf.touches(polygon.geometry).any()
            ):
                continue

            distances = []
            for centroid in largest_centroids:
                lat_diff = abs(polygon.geometry.centroid.y - centroid.y)
                lon_diff = abs(polygon.geometry.centroid.x - centroid.x)

                if lat_diff <= 5 and lon_diff <= 5:
                    distances.append((lat_diff, lon_diff))

            if not distances:
                largest_polygons.append(polygon)
                largest_centroids.append(polygon.geometry.centroid)
                clusters.append(len(largest_polygons) - 1)

    # Assign clusters to all polygons
    assigned_clusters = []
    for i in range(len(range_test)):
        polygon = range_test.iloc[i]

        # Use a tolerance when checking for geometry equality
        if any(
            polygon.geometry.equals_exact(lp.geometry, tolerance=0.00001)
            for lp in largest_polygons
        ):
            assigned_clusters.append(
                [
                    idx
                    for idx, lp in enumerate(largest_polygons)
                    if polygon.geometry.equals_exact(lp.geometry, tolerance=0.00001)
                ][0]
            )
            continue

        closest_centroid_idx = None
        min_distance = float("inf")

        for j, centroid in enumerate(largest_centroids):
            lat_diff = abs(polygon.geometry.centroid.y - centroid.y)
            lon_diff = abs(polygon.geometry.centroid.x - centroid.x)
            distance = np.sqrt(lat_diff**2 + lon_diff**2)

            if distance < min_distance:
                min_distance = distance
                closest_centroid_idx = j

        assigned_clusters.append(closest_centroid_idx)

    range_test["cluster"] = assigned_clusters

    # Return to the original CRS
    range_test = range_test.to_crs(epsg=4326)

    return range_test, largest_polygons


def classify_range_edges_gbif(df, largest_polygons):
    """
    Classifies polygons into leading (poleward), core, and trailing (equatorward)
    edges within each cluster based on distance from the centroid of the largest polygon within each cluster.
    Includes longitudinal relict detection.

    Parameters:
        df (GeoDataFrame): A GeoDataFrame with columns 'geometry' and 'cluster', and potentially repeated geometries.

    Returns:
        GeoDataFrame: The original GeoDataFrame with a new 'category' column merged in.
    """
    # Add unique ID for reliable merging
    df_original = df.copy().reset_index(drop=False).rename(columns={"index": "geom_id"})

    # Subset to unique geometry-cluster pairs with ID
    unique_geoms = (
        df_original[["geom_id", "geometry", "cluster"]].drop_duplicates().copy()
    )

    # Ensure proper CRS
    if unique_geoms.crs is None or unique_geoms.crs.to_epsg() != 3395:
        unique_geoms = unique_geoms.set_crs(df.crs).to_crs(epsg=3395)

    # Calculate centroids, lat/lon, area
    unique_geoms["centroid"] = unique_geoms.geometry.centroid
    unique_geoms["latitude"] = unique_geoms["centroid"].y
    unique_geoms["longitude"] = unique_geoms["centroid"].x
    unique_geoms["area"] = unique_geoms.geometry.area

    # Get centroid of largest polygon in each cluster
    def find_largest_polygon_centroid(sub_gdf):
        largest_polygon = sub_gdf.loc[sub_gdf["area"].idxmax()]
        return largest_polygon["centroid"]

    cluster_centroids = (
        unique_geoms.groupby("cluster")
        .apply(find_largest_polygon_centroid)
        .reset_index(name="cluster_centroid")
    )

    unique_geoms = unique_geoms.merge(cluster_centroids, on="cluster", how="left")

    # Classify within clusters
    def classify_within_cluster(sub_gdf):
        cluster_centroid = sub_gdf["cluster_centroid"].iloc[0]
        cluster_lat = cluster_centroid.y
        cluster_lon = cluster_centroid.x

        largest_polygon_area = largest_polygons[0]["AREA"]
        if largest_polygon_area > 150000:
            long_value = 0.2
        elif largest_polygon_area > 100000:
            long_value = 0.15
        else:
            long_value = 0.1
        # long_value = 0.15

        lat_threshold_01 = 0.1 * cluster_lat
        lat_threshold_05 = 0.05 * cluster_lat
        lat_threshold_02 = 0.02 * cluster_lat
        lon_threshold_01 = long_value * abs(cluster_lon)

        def classify(row):
            lat_diff = row["latitude"] - cluster_lat
            lon_diff = row["longitude"] - cluster_lon

            if lat_diff <= -lat_threshold_01:
                return "relict (0.01 latitude)"
            if abs(lon_diff) >= lon_threshold_01:
                return "relict (longitude)"
            if lat_diff >= lat_threshold_01:
                return "leading (0.99)"
            elif lat_diff >= lat_threshold_05:
                return "leading (0.95)"
            elif lat_diff >= lat_threshold_02:
                return "leading (0.9)"
            elif lat_diff <= -lat_threshold_05:
                return "trailing (0.05)"
            elif lat_diff <= -lat_threshold_02:
                return "trailing (0.1)"
            else:
                return "core"

        sub_gdf["category"] = sub_gdf.apply(classify, axis=1)
        return sub_gdf

    unique_geoms = unique_geoms.groupby("cluster", group_keys=False).apply(
        classify_within_cluster
    )

    # Prepare final mapping table and merge
    category_map = unique_geoms[["geom_id", "category"]]
    df_final = df_original.merge(category_map, on="geom_id", how="left").drop(
        columns="geom_id"
    )

    return df_final


def update_polygon_categories_gbif(largest_polygons_gdf, classified_polygons_gdf):
    """
    Updates polygon categories based on overlaps with island states and closest large polygon.

    Parameters:
        largest_polygons_gdf (GeoDataFrame): GeoDataFrame of largest polygons with 'geometry' and 'category'.
        classified_polygons_gdf (GeoDataFrame): Output from classify_range_edges_gbif with 'geom_id' and 'category'.
        island_states_gdf (GeoDataFrame): GeoDataFrame of island state geometries.

    Returns:
        GeoDataFrame: classified_polygons_gdf with updated 'category' values for overlapping polygons.
    """

    island_states_url = "https://raw.githubusercontent.com/anytko/biospat_large_files/main/island_states.geojson"

    island_states_gdf = gpd.read_file(island_states_url)

    # Ensure all CRS match
    crs = classified_polygons_gdf.crs or "EPSG:3395"
    island_states_gdf = island_states_gdf.to_crs(crs)

    if isinstance(largest_polygons_gdf, list):
        # Convert list of Series to DataFrame
        largest_polygons_gdf = pd.DataFrame(largest_polygons_gdf)
        largest_polygons_gdf = gpd.GeoDataFrame(
            largest_polygons_gdf,
            geometry="geometry",
            crs=crs,  # or whatever CRS you're using
        )

    largest_polygons_gdf = largest_polygons_gdf.to_crs(crs)
    classified_polygons_gdf = classified_polygons_gdf.to_crs(crs)

    unique_polygons = classified_polygons_gdf.drop_duplicates(
        subset="geometry"
    ).reset_index(drop=True)
    unique_polygons["geom_id"] = unique_polygons.index.astype(str)

    # Merge back geom_id to the full dataframe
    classified_polygons_gdf = classified_polygons_gdf.merge(
        unique_polygons[["geometry", "geom_id"]], on="geometry", how="left"
    )

    # Spatial join to find overlapping polygons with island states
    overlapping_polygons = gpd.sjoin(
        classified_polygons_gdf, island_states_gdf, how="inner", predicate="intersects"
    )
    overlapping_polygons = overlapping_polygons.drop_duplicates(subset="geom_id")

    # Compute centroids for distance matching
    overlapping_polygons["centroid"] = overlapping_polygons.geometry.centroid
    largest_polygons_gdf["centroid"] = largest_polygons_gdf.geometry.centroid

    # Extract coordinates
    overlapping_centroids = (
        overlapping_polygons["centroid"].apply(lambda x: (x.x, x.y)).tolist()
    )
    largest_centroids = (
        largest_polygons_gdf["centroid"].apply(lambda x: (x.x, x.y)).tolist()
    )

    # Compute distances and find nearest large polygon
    distances = cdist(overlapping_centroids, largest_centroids)
    closest_indices = distances.argmin(axis=1)

    # Assign nearest large polygon's category
    overlapping_polygons["category"] = largest_polygons_gdf.iloc[closest_indices][
        "category"
    ].values

    # Update classified polygons using 'geom_id'
    updated_classified_polygons = classified_polygons_gdf.copy()
    update_map = dict(
        zip(overlapping_polygons["geom_id"], overlapping_polygons["category"])
    )
    updated_classified_polygons["category"] = updated_classified_polygons.apply(
        lambda row: update_map.get(row["geom_id"], row["category"]), axis=1
    )

    return updated_classified_polygons


def merge_and_remap_polygons(gdf, buffer_distance=0):
    gdf = gdf.copy()

    # Ensure CRS is projected for buffering and spatial operations
    if gdf.crs.to_epsg() != 3395:
        gdf = gdf.to_crs(epsg=3395)

    # Step 1: Extract unique polygons
    unique_polys = gdf[["geometry"]].drop_duplicates().reset_index(drop=True)
    unique_polys = gpd.GeoDataFrame(unique_polys, geometry="geometry", crs=gdf.crs)

    # Apply buffering if necessary
    if buffer_distance > 0:
        unique_polys["geom_buffered"] = unique_polys["geometry"].buffer(buffer_distance)
    else:
        unique_polys["geom_buffered"] = unique_polys["geometry"]

    # Step 2: Merge only touching or intersecting polygons
    sindex = unique_polys.sindex
    assigned = set()
    groups = []

    for idx, geom in unique_polys["geom_buffered"].items():
        if idx in assigned:
            continue
        group = set([idx])
        queue = [idx]
        while queue:
            current = queue.pop()
            current_geom = unique_polys.loc[current, "geom_buffered"]
            matches = list(sindex.intersection(current_geom.bounds))
            for match in matches:
                if match not in group:
                    match_geom = unique_polys.loc[match, "geom_buffered"]
                    if current_geom.touches(match_geom) or current_geom.intersects(
                        match_geom
                    ):
                        group.add(match)
                        queue.append(match)
        assigned |= group
        groups.append(group)

    # Step 3: Build mapping from original polygon to merged geometry
    polygon_to_merged = {}
    merged_geoms = []

    for group in groups:
        group_polys = unique_polys.loc[list(group), "geometry"]
        merged = unary_union(group_polys.values)
        merged_geoms.append(merged)
        for poly in group_polys:
            polygon_to_merged[poly.wkt] = merged

    # Step 4: Map merged geometry back to each row in original gdf based on geometry
    gdf["merged_geometry"] = gdf["geometry"].apply(
        lambda poly: polygon_to_merged[poly.wkt]
    )

    # Step 5: Set the merged geometry as the active geometry column
    gdf["geometry"] = gdf["merged_geometry"]

    # Step 6: Remove temporary 'merged_geometry' column
    gdf = gdf.drop(columns=["merged_geometry"])

    # Step 7: Ensure that point geometries are correctly associated (keep them unchanged)
    gdf["point_geometry"] = gdf["point_geometry"]

    # Set the 'geometry' column explicitly as the active geometry column
    gdf.set_geometry("geometry", inplace=True)

    # Optional: reproject to WGS84 (EPSG:4326)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def update_polygon_categories_gbif_test(largest_polygons_gdf, classified_polygons_gdf):
    """
    Updates polygon categories based on overlaps with island states and nearest large polygon.

    Parameters:
        largest_polygons_gdf (GeoDataFrame): GeoDataFrame of largest polygons with 'geometry' and 'category'.
        classified_polygons_gdf (GeoDataFrame): GeoDataFrame of smaller polygons (one row per point) with potential duplicate geometries.

    Returns:
        GeoDataFrame: classified_polygons_gdf with updated 'category' values for overlapping polygons.
    """

    import geopandas as gpd
    import pandas as pd
    from scipy.spatial.distance import cdist

    # Load island states
    island_states_url = "https://raw.githubusercontent.com/anytko/biospat_large_files/main/island_states.geojson"
    island_states_gdf = gpd.read_file(island_states_url)

    # Ensure all CRS match
    crs = classified_polygons_gdf.crs or "EPSG:3395"
    island_states_gdf = island_states_gdf.to_crs(crs)

    if isinstance(largest_polygons_gdf, list):
        largest_polygons_gdf = pd.DataFrame(largest_polygons_gdf)
        largest_polygons_gdf = gpd.GeoDataFrame(
            largest_polygons_gdf, geometry="geometry", crs=crs
        )

    largest_polygons_gdf["category"] = "core"

    largest_polygons_gdf = largest_polygons_gdf.to_crs(crs)
    classified_polygons_gdf = classified_polygons_gdf.to_crs(crs)

    # Assign unique ID per unique geometry
    unique_polygons = classified_polygons_gdf.drop_duplicates(
        subset="geometry"
    ).reset_index(drop=True)
    unique_polygons["geom_id"] = unique_polygons.index.astype(str)

    # Merge geom_id back to full dataframe
    classified_polygons_gdf = classified_polygons_gdf.merge(
        unique_polygons[["geometry", "geom_id"]], on="geometry", how="left"
    )

    # Find overlaps with island states
    overlapping_polygons = gpd.sjoin(
        classified_polygons_gdf, island_states_gdf, how="inner", predicate="intersects"
    )
    overlapping_polygons = overlapping_polygons.drop_duplicates(subset="geom_id").copy()

    # Compute centroids
    overlapping_centroids = overlapping_polygons.geometry.centroid
    largest_centroids = largest_polygons_gdf.geometry.centroid

    # Compute distances between centroids
    distances = cdist(
        overlapping_centroids.apply(lambda x: (x.x, x.y)).tolist(),
        largest_centroids.apply(lambda x: (x.x, x.y)).tolist(),
    )
    closest_indices = distances.argmin(axis=1)

    # Assign categories from nearest large polygon
    overlapping_polygons["category"] = largest_polygons_gdf.iloc[closest_indices][
        "category"
    ].values

    # Update the categories in the original dataframe
    update_map = dict(
        zip(overlapping_polygons["geom_id"], overlapping_polygons["category"])
    )
    updated_classified_polygons = classified_polygons_gdf.copy()
    updated_classified_polygons["category"] = updated_classified_polygons.apply(
        lambda row: update_map.get(row["geom_id"], row["category"]), axis=1
    )

    return updated_classified_polygons


import geopandas as gpd


def remove_lakes_and_plot_gbif(polygons_gdf):
    """
    Removes lake polygons from range polygons and retains all rows in the original data,
    updating the geometry where lakes intersect with polygons.

    Parameters:
    - polygons_gdf: GeoDataFrame of range polygons.

    Returns:
    - Updated GeoDataFrame with lakes removed from intersecting polygons.
    """
    # Load lakes GeoDataFrame
    lakes_url = "https://raw.githubusercontent.com/anytko/biospat_large_files/main/lakes_na.geojson"
    lakes_gdf = gpd.read_file(lakes_url)

    # Ensure geometries are valid
    polygons_gdf = polygons_gdf[polygons_gdf.geometry.is_valid]
    lakes_gdf = lakes_gdf[lakes_gdf.geometry.is_valid]

    # Ensure CRS matches before performing spatial operations
    if polygons_gdf.crs != lakes_gdf.crs:
        print(f"CRS mismatch! Transforming {polygons_gdf.crs} -> {lakes_gdf.crs}")
        polygons_gdf = polygons_gdf.to_crs(lakes_gdf.crs)

    # Add an ID column to identify unique polygons (group points by shared polygons)
    polygons_gdf["unique_id"] = polygons_gdf.groupby("geometry").ngroup()

    # Deduplicate the range polygons by geometry and add ID to unique polygons
    unique_gdf = polygons_gdf.drop_duplicates(subset="geometry")
    unique_gdf["unique_id"] = unique_gdf.groupby(
        "geometry"
    ).ngroup()  # Assign shared unique IDs

    # Clip the unique polygons with the lake polygons (difference operation)
    polygons_no_lakes_gdf = gpd.overlay(unique_gdf, lakes_gdf, how="difference")

    # Merge the modified unique polygons back with the original GeoDataFrame using 'unique_id'
    merged_polygons = polygons_gdf.merge(
        polygons_no_lakes_gdf[["unique_id", "geometry"]], on="unique_id", how="left"
    )

    # Now update the geometry column with the new geometries from the modified polygons
    merged_polygons["geometry"] = merged_polygons["geometry_y"].fillna(
        merged_polygons["geometry_x"]
    )

    # Drop the temporary columns that were used for merging
    merged_polygons = merged_polygons.drop(
        columns=["geometry_y", "geometry_x", "unique_id"]
    )

    # Ensure the resulting DataFrame is still a GeoDataFrame
    merged_polygons = gpd.GeoDataFrame(merged_polygons, geometry="geometry")

    # Set CRS correctly
    merged_polygons.set_crs(polygons_gdf.crs, allow_override=True, inplace=True)

    # Return the updated GeoDataFrame
    return merged_polygons


def clip_polygons_to_continent_gbif(input_gdf):
    """
    Clips the polygon geometry associated with each point to the North American continent.
    Preserves one row per original point.

    Parameters:
    - input_gdf: GeoDataFrame with columns ['point_geometry', 'year', 'geometry'].

    Returns:
    - GeoDataFrame with same number of rows but clipped geometries.
    """
    from shapely.geometry import box

    # Load continent polygons (land areas)
    land_url = (
        "https://raw.githubusercontent.com/anytko/biospat_large_files/main/land.geojson"
    )
    continents_gdf = gpd.read_file(land_url)

    # Ensure valid geometries
    input_gdf = input_gdf[input_gdf["geometry"].is_valid]
    continents_gdf = continents_gdf[continents_gdf["geometry"].is_valid]

    # Reproject if needed
    if input_gdf.crs != continents_gdf.crs:
        input_gdf = input_gdf.to_crs(continents_gdf.crs)

    # Step 1: Assign unique polygon IDs for shared geometries
    input_gdf = input_gdf.copy()
    input_gdf["poly_id"] = input_gdf.groupby("geometry").ngroup()

    # Step 2: Clip only unique polygons
    unique_polygons = input_gdf.drop_duplicates(subset="geometry")[
        ["poly_id", "geometry"]
    ]
    clipped = gpd.overlay(unique_polygons, continents_gdf, how="intersection")

    # Step 3: Clip again to North America bounding box
    na_bbox = box(-178.2, 6.6, -49.0, 83.3)
    na_gdf = gpd.GeoDataFrame(geometry=[na_bbox], crs=input_gdf.crs)
    clipped = gpd.overlay(clipped, na_gdf, how="intersection")

    # Step 4: Collapse fragments back into one geometry per poly_id
    clipped = clipped.dissolve(by="poly_id")

    # Step 5: Merge clipped polygons back to original data
    result = input_gdf.merge(
        clipped[["geometry"]],
        left_on="poly_id",
        right_index=True,
        how="left",
        suffixes=("", "_clipped"),
    )

    # Use clipped geometry if available
    result["geometry"] = result["geometry_clipped"].fillna(result["geometry"])
    result = result.drop(columns=["geometry_clipped", "poly_id"])

    # Ensure it's still a GeoDataFrame with correct CRS
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=input_gdf.crs)
    result = result.to_crs(epsg=4326)

    return result


# This works the same as the assign_polygon_clusters_gbif function but subsets unique polygons first which may be quicker with more data


def assign_polygon_clusters_gbif_test(polygon_gdf):
    import hashlib

    island_states_url = "https://raw.githubusercontent.com/anytko/biospat_large_files/main/island_states.geojson"
    island_states_gdf = gpd.read_file(island_states_url)

    # Simplify to avoid precision issues (optional)
    polygon_gdf["geometry"] = polygon_gdf.geometry.simplify(
        tolerance=0.001, preserve_topology=True
    )

    # Create a unique ID for each geometry (by hashing WKT string)
    polygon_gdf = polygon_gdf.copy()
    polygon_gdf["geometry_id"] = polygon_gdf.geometry.apply(
        lambda g: hashlib.md5(g.wkb).hexdigest()
    )

    # Subset unique polygons
    unique_polys = polygon_gdf.drop_duplicates(subset="geometry_id").copy()

    # Calculate area (in meters^2)
    if unique_polys.crs.is_geographic:
        unique_polys = unique_polys.to_crs(epsg=3395)
    unique_polys["AREA"] = unique_polys.geometry.area / 1e6
    unique_polys = unique_polys.sort_values(by="AREA", ascending=False)

    # Start clustering
    largest_polygons = []
    largest_centroids = []
    cluster_ids = {}

    first_polygon = unique_polys.iloc[0]
    if (
        not island_states_gdf.intersects(first_polygon.geometry).any()
        and not island_states_gdf.touches(first_polygon.geometry).any()
    ):
        largest_polygons.append(first_polygon)
        largest_centroids.append(first_polygon.geometry.centroid)
        cluster_ids[first_polygon["geometry_id"]] = 0

    for i in range(1, len(unique_polys)):
        polygon = unique_polys.iloc[i]
        if polygon["geometry_id"] in cluster_ids:
            continue  # Already clustered

        # polygon_threshold = 0.3  # Default threshold

        # Dynamically set threshold based on size of largest polygon
        if largest_polygons[0]["AREA"] > 500000:
            polygon_threshold = 0.1
        elif largest_polygons[0]["AREA"] > 150000:
            polygon_threshold = 0.2
        else:
            polygon_threshold = 0.3

        if polygon["AREA"] >= polygon_threshold * largest_polygons[0]["AREA"]:
            if (
                island_states_gdf.intersects(polygon.geometry).any()
                or island_states_gdf.touches(polygon.geometry).any()
            ):
                continue

            centroid = polygon.geometry.centroid
            too_close = any(
                abs(centroid.x - c.x) <= 5 and abs(centroid.y - c.y) <= 5
                for c in largest_centroids
            )
            if not too_close:
                new_cluster = len(largest_polygons)
                largest_polygons.append(polygon)
                largest_centroids.append(centroid)
                cluster_ids[polygon["geometry_id"]] = new_cluster

    # Assign remaining polygons to nearest cluster
    for i, row in unique_polys.iterrows():
        geom_id = row["geometry_id"]
        if geom_id in cluster_ids:
            continue
        centroid = row.geometry.centroid
        distances = [
            np.sqrt((centroid.x - c.x) ** 2 + (centroid.y - c.y) ** 2)
            for c in largest_centroids
        ]
        cluster_ids[geom_id] = int(np.argmin(distances))

    # Map clusters back to full polygon_gdf
    polygon_gdf["cluster"] = polygon_gdf["geometry_id"].map(cluster_ids)

    polygon_gdf["AREA"] = polygon_gdf["geometry_id"].map(
        unique_polys.set_index("geometry_id")["AREA"]
    )

    # Return to original CRS
    polygon_gdf = polygon_gdf.to_crs(epsg=4326)

    return polygon_gdf, largest_polygons


from pygbif import occurrences
import time


def fetch_gbif_data_modern(species_name, limit=2000, end_year=2025, start_year=1971):
    """
    Fetches modern occurrence data from GBIF for a specified species between given years.
    Works backward from end_year to start_year until the limit is reached.
    """
    all_data = []
    page_limit = 300
    consecutive_empty_years = 0

    for year in range(end_year, start_year - 1, -1):
        offset = 0
        year_data = []

        while len(all_data) < limit:
            response = occurrences.search(
                scientificName=species_name,
                hasCoordinate=True,
                hasGeospatialIssue=False,
                year=year,
                limit=page_limit,
                offset=offset,
            )

            results = response.get("results", [])
            if not results:
                break

            year_data.extend(results)

            if len(results) < page_limit:
                break

            offset += page_limit

        if year_data:
            all_data.extend(year_data)
            consecutive_empty_years = 0
        else:
            consecutive_empty_years += 1

        if len(all_data) >= limit:
            all_data = all_data[:limit]
            break

        if consecutive_empty_years >= 5:
            print(
                f"No data found for 5 consecutive years before {year + 5}. Stopping early."
            )
            break

    return all_data


from pygbif import occurrences


def fetch_historic_records(species_name, limit=2000, year=1971):
    all_data = []
    year = year
    page_limit = 300
    consecutive_empty_years = 0  # stop if multiple years in a row return nothing

    while len(all_data) < limit and year >= 1960:
        offset = 0
        year_data = []
        while len(all_data) < limit:
            response = occurrences.search(
                scientificName=species_name,
                hasCoordinate=True,
                hasGeospatialIssue=False,
                year=year,
                limit=page_limit,
                offset=offset,
            )
            results = response.get("results", [])
            if not results:
                break
            year_data.extend(results)
            if len(results) < page_limit:
                break
            offset += page_limit

        if year_data:
            all_data.extend(year_data)
            consecutive_empty_years = 0  # reset
        else:
            consecutive_empty_years += 1

        if consecutive_empty_years >= 5:
            print(
                f"No data found for 5 consecutive years before {year + 5}. Stopping early."
            )
            break

        year -= 1

    return all_data[:limit]


def fetch_gbif_data_with_historic(
    species_name, limit=2000, start_year=1971, end_year=2025
):
    """
    Fetches both modern and historic occurrence data from GBIF for a specified species.

    Parameters:
        species_name (str): Scientific name of the species.
        limit (int): Max number of records to fetch for each (modern and historic).
        start_year (int): The earliest year for modern data and latest year for historic data.
        end_year (int): The most recent year to fetch from.

    Returns:
        dict: {
            'modern': [...],  # from start_year + 1 to end_year
            'historic': [...] # from start_year backwards to ~1960
        }
    """
    modern = fetch_gbif_data_modern(
        species_name=species_name,
        limit=limit,
        start_year=start_year + 1,
        end_year=end_year,
    )

    historic = fetch_historic_records(
        species_name=species_name,
        limit=limit,
        year=start_year,  # avoid overlap with modern
    )

    return {"modern": modern, "historic": historic}


def process_gbif_data_pipeline(
    gdf, species_name=None, is_modern=True, year_range=None, end_year=2025
):
    """
    Processes GBIF occurrence data through a series of spatial filtering and classification steps.

    Parameters:
        gdf (GeoDataFrame): Input GBIF occurrence data.
        species_name (str): Scientific name of the species. Required if year_range is not given.
        is_modern (bool): Whether the data is modern. If False, the pruning by year is skipped.
        year_range (tuple or None): Start and end years for pruning (only used for modern data).
        end_year (int): The end year for pruning modern data, default is 2025.

    Returns:
        GeoDataFrame: Classified polygons.
    """

    if is_modern and year_range is None:
        if species_name is None:
            raise ValueError("species_name must be provided if year_range is not.")

        # Get start year from species data if available, otherwise use a default
        start_year = get_start_year_from_species(species_name)
        if start_year == "NA":
            raise ValueError(f"Start year not found for species '{species_name}'.")
        start_year = int(start_year)

        # Use the provided end_year if available, otherwise default to 2025
        year_range = (start_year, end_year)

    # Step 1: Create DBSCAN polygons
    polys = make_dbscan_polygons_with_points_from_gdf(gdf)

    # Step 2: Optionally prune by year for modern data
    if is_modern:
        polys = prune_by_year(polys, *year_range)

    # Step 3: Merge and remap
    merged_polygons = merge_and_remap_polygons(polys, buffer_distance=100)

    # Step 4: Remove lakes
    unique_polys_no_lakes = remove_lakes_and_plot_gbif(merged_polygons)

    # Step 5: Clip to continents
    clipped_polys = clip_polygons_to_continent_gbif(unique_polys_no_lakes)

    # Step 6: Assign cluster ID and large polygon
    assigned_poly, large_poly = assign_polygon_clusters_gbif_test(clipped_polys)

    # Step 7: Classify edges
    classified_poly = classify_range_edges_gbif(assigned_poly, large_poly)

    return classified_poly


def analyze_species_distribution(species_name, record_limit=100, end_year=2025):
    """
    Fetches and processes both modern and historic GBIF data for a given species.

    Parameters:
        species_name (str): Scientific name of the species.
        record_limit (int): Max number of records to fetch from GBIF.
        end_year (int): The most recent year to fetch modern data for.

    Returns:
        Tuple: (classified_modern_polygons, classified_historic_polygons)
    """

    start_year = get_start_year_from_species(species_name)
    if start_year == "NA":
        raise ValueError(f"Start year not found for species '{species_name}'.")
    start_year = int(start_year)

    data = fetch_gbif_data_with_historic(
        species_name, limit=record_limit, start_year=start_year, end_year=end_year
    )

    print(f"Modern records (>= {start_year}):", len(data["modern"]))
    print(f"Historic records (< {start_year}):", len(data["historic"]))

    modern_data = data["modern"]  # This is a list of dictionaries
    historic_data = data["historic"]

    historic_gdf = convert_to_gdf(historic_data)
    modern_gdf = convert_to_gdf(modern_data)

    # Let the pipeline dynamically determine the year range
    classified_modern = process_gbif_data_pipeline(
        modern_gdf, species_name=species_name, is_modern=True, end_year=end_year
    )
    classified_historic = process_gbif_data_pipeline(
        historic_gdf, is_modern=False, end_year=end_year
    )

    classified_modern = calculate_density(classified_modern)
    classified_historic = calculate_density(classified_historic)

    return classified_modern, classified_historic


def collapse_and_calculate_centroids(gdf):
    """
    Collapses subgroups in the 'category' column into broader groups and calculates
    the centroid for each category.

    Parameters:
    - gdf: GeoDataFrame with a 'category' column and polygon geometries.

    Returns:
    - GeoDataFrame with one centroid per collapsed category.
    """

    # Step 1: Standardize 'category' names
    gdf["category"] = gdf["category"].str.strip().str.lower()

    # Step 2: Collapse specific subgroups
    category_mapping = {
        "leading (0.99)": "leading",
        "leading (0.95)": "leading",
        "leading (0.9)": "leading",
        "trailing (0.1)": "trailing",
        "trailing (0.05)": "trailing",
        "relict (0.01 latitude)": "relict",
        "relict (longitude)": "relict",
    }
    gdf["category"] = gdf["category"].replace(category_mapping)

    # Step 3: Calculate centroids per collapsed category
    centroids_data = []
    for category, group in gdf.groupby("category"):
        centroid = group.geometry.unary_union.centroid
        centroids_data.append({"category": category, "geometry": centroid})

    return gpd.GeoDataFrame(centroids_data, crs=gdf.crs)


def calculate_northward_change_rate(hist_gdf, new_gdf, species_name, end_year=2025):
    """
    Compare centroids within each group/category in two GeoDataFrames and calculate:
    - The northward change in kilometers
    - The rate of northward change in km per year

    Parameters:
    - hist_gdf: GeoDataFrame with historical centroids (1 centroid per category)
    - new_gdf: GeoDataFrame with new centroids (1 centroid per category)
    - species_name: Name of the species to determine start year
    - end_year: The final year of the new data (default 2025)

    Returns:
    - A DataFrame with category, northward change in km, and rate of northward change in km/year
    """

    # Dynamically get the starting year based on species
    start_year = int(get_start_year_from_species(species_name))

    # Calculate the time difference in years
    years_elapsed = end_year - start_year

    # Merge the two GeoDataFrames on the 'category' column
    merged_gdf = hist_gdf[["category", "geometry"]].merge(
        new_gdf[["category", "geometry"]], on="category", suffixes=("_hist", "_new")
    )

    # List to store the changes
    changes = []

    for _, row in merged_gdf.iterrows():
        category = row["category"]
        centroid_hist = row["geometry_hist"].centroid
        centroid_new = row["geometry_new"].centroid

        # Latitude difference
        northward_change_lat = centroid_new.y - centroid_hist.y
        northward_change_km = northward_change_lat * 111.32
        northward_rate_km_per_year = northward_change_km / years_elapsed

        changes.append(
            {
                "species": species_name,
                "category": category,
                "northward_change_km": northward_change_km,
                "northward_rate_km_per_year": northward_rate_km_per_year,
            }
        )

    return pd.DataFrame(changes)


def analyze_northward_shift(gdf_hist, gdf_new, species_name, end_year=2025):
    """
    Wrapper function that collapses categories and computes the rate of northward shift
    in km/year between historical and modern GeoDataFrames.

    Parameters:
    - gdf_hist: Historical GeoDataFrame with 'category' column and polygon geometries
    - gdf_new: Modern GeoDataFrame with 'category' column and polygon geometries
    - species_name: Name of the species to determine the starting year
    - end_year: The final year of modern data (default is 2025)

    Returns:
    - DataFrame with each category's northward change and rate of change
    """

    # Step 1: Collapse and calculate centroids
    hist_centroids = collapse_and_calculate_centroids(gdf_hist)
    new_centroids = collapse_and_calculate_centroids(gdf_new)

    # Step 2: Calculate northward movement
    result = calculate_northward_change_rate(
        hist_gdf=hist_centroids,
        new_gdf=new_centroids,
        species_name=species_name,
        end_year=end_year,
    )

    return result


def categorize_species(df):
    """
    Categorizes species into movement groups based on leading, core, and trailing rates.
    Handles both full (3-edge) and partial (2-edge) data cases.

    Parameters:
        df (pd.DataFrame): A DataFrame with columns ['species', 'category', 'northward_rate_km_per_year']

    Returns:
        pd.DataFrame: Categorized movement results with leading/core/trailing rates.
    """
    categories = []

    for species_name in df["species"].unique():
        species_data = df[df["species"] == species_name]

        # Extract available rates
        leading = species_data.loc[
            species_data["category"].str.contains("leading", case=False),
            "northward_rate_km_per_year",
        ].values
        core = species_data.loc[
            species_data["category"].str.contains("core", case=False),
            "northward_rate_km_per_year",
        ].values
        trailing = species_data.loc[
            species_data["category"].str.contains("trailing", case=False),
            "northward_rate_km_per_year",
        ].values

        leading = leading[0] if len(leading) > 0 else None
        core = core[0] if len(core) > 0 else None
        trailing = trailing[0] if len(trailing) > 0 else None

        # Count how many components are not None
        num_known = sum(x is not None for x in [leading, core, trailing])

        category = "uncategorized"  # default

        # ======= Full Data (3 values) =======
        if num_known == 3:
            if leading > 2 and core > 2 and trailing > 2:
                category = "positive moving together"
            elif leading < -2 and core < -2 and trailing < -2:
                category = "negative moving together"

            elif (leading > 2 and trailing < -2) or (trailing > 2 and leading < -2):
                category = "pull apart"
            elif (core > 2 and (leading > 2 or trailing < -2)) or (
                core < -2 and (leading < -2 or trailing > 2)
            ):
                category = "pull apart"

            elif (
                (leading < -2 and core >= -2 and trailing > 2)
                or (core > 2 and -2 <= leading <= 2 and trailing > 2)
                or (core < -2 and -2 <= trailing <= 2 and leading < -2)
                or (core > 2 and (leading <= 0))
                or (core < -2 and trailing >= 0)
            ):
                category = "reabsorption"

            elif -2 <= core <= 2 and (
                (-2 <= leading <= 2 and -2 <= trailing <= 2)
                or (-2 <= leading <= 2)
                or (-2 <= trailing <= 2)
            ):
                category = "stability"

            elif (
                (leading > 2 and core <= 2 and trailing < -2)
                or (leading > 2 and core > 2 and trailing < -2)
                or (leading > 2 and core < -2 and trailing < -2)
                or (-2 <= leading <= 2 and core < -2 and trailing < -2)
                or (leading > 2 and core > 2 and -2 <= trailing <= 2)
            ):
                category = "pulling apart"

            elif (
                (leading < -2 and core >= -2 and trailing > 2)
                or (leading <= 2 and core > 2)
                or (core < -2 and trailing <= 2)
                or (leading < -2 and core > 2 and trailing > 2)
                or (leading < -2 and core < -2 and trailing > 2)
            ):
                category = "reabsorption"

        # ======= Partial Data (2 values) =======
        elif num_known == 2:
            # Only leading and core
            if leading is not None and core is not None:
                if -2 <= leading <= 2 and -2 <= core <= 2:
                    category = "likely stable"
                elif leading > 2 and core > 2:
                    category = "likely positive moving together"
                elif leading < -2 and core < -2:
                    category = "likely negative moving together"
                elif leading > 2 and core < -2:
                    category = "likely pull apart"
                elif leading > 2 and -2 <= core <= 2:
                    category = "likely pull apart"
                elif leading < -2 and -2 <= core <= 2:
                    category = "likely reabsorption"
                elif leading < -2 and core > 2:
                    category = "likely reabsorption"

            # Only core and trailing
            elif core is not None and trailing is not None:
                if -2 <= core <= 2 and -2 <= trailing <= 2:
                    category = "likely stable"
                elif core > 2 and trailing > 2:
                    category = "likely moving together"
                elif core < -2 and trailing < -2:
                    category = "likely moving together"
                elif -2 <= core <= 2 and trailing < -2:
                    category = "likely pull apart"
                elif core > 2 and trailing < -2:
                    category = "likely pull apart"
                elif -2 <= core <= 2 and trailing > 2:
                    category = "likely reabsorption"
                elif core < -2 and trailing > 2:
                    category = "likely reabsorption"

        # ======= Final Append =======
        categories.append(
            {
                "species": species_name,
                "leading": leading,
                "core": core,
                "trailing": trailing,
                "category": category,
            }
        )

    return pd.DataFrame(categories)


def summarize_polygons_with_points(df):
    """
    Summarizes number of points per unique polygon (geometry_id), retaining one row per polygon.

    Parameters:
        df (pd.DataFrame): A DataFrame where each row represents a point with associated polygon metadata.

    Returns:
        gpd.GeoDataFrame: A summarized GeoDataFrame with one row per unique polygon and geometry set.
    """

    # Group by geometry_id and aggregate
    summary = (
        df.groupby("geometry_id")
        .agg(
            {
                "geometry": "first",  # keep one polygon geometry
                "category": "first",  # assume category is the same within a polygon
                "AREA": "first",  # optional: keep AREA of the polygon
                "cluster": "first",  # optional: keep cluster ID
                "point_geometry": "count",  # count how many points fall in this polygon
            }
        )
        .rename(columns={"point_geometry": "n_points"})
        .reset_index()
    )

    summary_gdf = gpd.GeoDataFrame(summary, geometry="geometry")

    return summary_gdf


def count_points_per_category(df):
    """
    Standardizes category labels and counts how many points fall into each simplified category.

    Parameters:
        df (pd.DataFrame): The original DataFrame with a 'category' column.

    Returns:
        pd.DataFrame: A DataFrame showing total points per simplified category.
    """
    category_mapping = {
        "leading (0.99)": "leading",
        "leading (0.95)": "leading",
        "leading (0.9)": "leading",
        "trailing (0.1)": "trailing",
        "trailing (0.05)": "trailing",
        "relict (0.01 latitude)": "relict",
        "relict (longitude)": "relict",
    }

    # Standardize the categories
    df["category"] = df["category"].replace(category_mapping)

    # Count the number of points per simplified category
    category_counts = df.groupby("category")["point_geometry"].count().reset_index()
    category_counts.columns = ["category", "n_points"]

    return category_counts


def prepare_data(df):
    # Group by polygon
    grouped = (
        df.groupby("geometry_id")
        .agg({"geometry": "first", "point_geometry": "count", "category": "first"})
        .rename(columns={"point_geometry": "point_count"})
        .reset_index()
    )
    gdf_polygons = gpd.GeoDataFrame(grouped, geometry="geometry")
    gdf_polygons = gdf_polygons.to_crs("EPSG:4326")  # MapLibre requires lat/lon
    return gdf_polygons


import pydeck as pdk
import tempfile
import webbrowser
import os
import json
import geopandas as gpd


def create_interactive_map(dataframe, if_save=False):
    # --- Split dataframe into polygons and points ---
    # Keep the polygon geometries
    polygon_gdf = dataframe.drop(
        columns=["point_geometry"]
    )  # Remove point geometry column from polygons
    polygon_gdf = gpd.GeoDataFrame(
        polygon_gdf, geometry="geometry"
    )  # Set 'geometry' as the geometry column

    # Create the point GeoDataFrame, setting 'point_geometry' as the geometry column
    point_gdf = dataframe.copy()
    point_gdf = point_gdf.drop(
        columns=["geometry"]
    )  # Remove the polygon geometry column from points
    point_gdf = gpd.GeoDataFrame(
        point_gdf, geometry="point_geometry"
    )  # Set 'point_geometry' as the geometry column

    # --- Convert to GeoJSON for the polygon layer ---
    polygon_json = json.loads(polygon_gdf.to_json())

    # Add columns for point locations (longitude, latitude)
    point_gdf["point_lon"] = point_gdf.geometry.x
    point_gdf["point_lat"] = point_gdf.geometry.y

    # Optional: Add elevation based on some attribute, e.g., year or cluster
    point_gdf["weight"] = 1  # Can also use year, cluster, etc.

    # --- Define the initial view state for the map ---
    view_state = pdk.ViewState(
        latitude=point_gdf["point_lat"].mean(),
        longitude=point_gdf["point_lon"].mean(),
        zoom=6,
        pitch=60,
    )

    # --- Polygon outline layer ---
    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        data=polygon_json,
        get_fill_color="[0, 0, 0, 0]",  # Transparent fill
        get_line_color=[120, 120, 120],
        line_width_min_pixels=1,
        pickable=True,
    )

    # --- Smooth elevation using HexagonLayer ---
    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=point_gdf,
        get_position=["point_lon", "point_lat"],
        radius=1500,  # Hexagon size in meters (adjust for smoothness)
        elevation_scale=100,  # Lower scale for smoother, less jagged effect
        get_elevation_weight="weight",  # Use 'weight' column for height (density)
        elevation_range=[0, 2000],  # Range for elevation (can adjust as needed)
        extruded=True,
        coverage=1,  # Coverage of hexagons, 1 = fully covered
        pickable=True,
    )

    # --- Create the pydeck map with the layers ---
    r = pdk.Deck(
        layers=[polygon_layer, hex_layer],
        initial_view_state=view_state,
        tooltip={"text": "Height (density): {elevationValue}"},
    )

    # --- Create and display the map in a temporary HTML file ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        # Get the temporary file path
        temp_file_path = tmp_file.name

        # Save the map to the temporary file
        r.to_html(temp_file_path)

        # Open the saved map in the default browser (automatically detects default browser)
        webbrowser.open(f"file://{temp_file_path}")

    if if_save:
        home_dir = os.path.expanduser("~")
        if os.name == "nt":  # Windows
            downloads_path = os.path.join(home_dir, "Downloads", "map.html")
        else:  # macOS or Linux
            downloads_path = os.path.join(home_dir, "Downloads", "map.html")

        try:
            # Save the map directly to the Downloads folder
            r.to_html(downloads_path)
            print(f"Map saved at {downloads_path}")
        except Exception as e:
            print(f"Error saving map to Downloads: {e}")


# REMEMBER that this is a proportional metric - meaning that decreases mean that category are holding proportionally less points across the range


def calculate_rate_of_change_first_last(
    historical_df, modern_df, species_name, custom_end_year=None
):
    from datetime import datetime
    import pandas as pd

    # Mapping of detailed categories to collapsed ones
    category_mapping = {
        "leading (0.99)": "leading",
        "leading (0.95)": "leading",
        "leading (0.9)": "leading",
        "trailing (0.1)": "trailing",
        "trailing (0.05)": "trailing",
        "relict (0.01 latitude)": "relict",
        "relict (longitude)": "relict",
    }

    # Apply mapping to both dataframes
    historical_df["collapsed_category"] = historical_df["category"].replace(
        category_mapping
    )
    modern_df["collapsed_category"] = modern_df["category"].replace(category_mapping)

    # Get species start year and define start time period
    start_year = int(get_start_year_from_species(species_name))
    first_period_start = (start_year // 10) * 10
    first_period_end = start_year
    adjusted_first_period = f"{first_period_start}-{first_period_end}"

    # Define end time period
    current_year = datetime.today().year
    modern_df["event_year"] = pd.to_datetime(
        modern_df["eventDate"], errors="coerce"
    ).dt.year
    last_event_year = modern_df["event_year"].dropna().max()

    if custom_end_year is not None:
        last_period_end = custom_end_year
        last_period_start = custom_end_year - 1
    else:
        last_period_start = min(last_event_year, current_year - 1)
        last_period_end = current_year

    adjusted_last_period = f"{last_period_start}-{last_period_end}"

    # Add time_period to each dataframe
    historical_df = historical_df.copy()
    historical_df["time_period"] = adjusted_first_period
    modern_df = modern_df.copy()
    modern_df["time_period"] = adjusted_last_period

    # Combine for grouped calculations
    combined_df = pd.concat([historical_df, modern_df], ignore_index=True)

    # Drop missing categories or time_periods
    combined_df = combined_df.dropna(subset=["collapsed_category", "time_period"])

    # Group and calculate percentages
    grouped = (
        combined_df.groupby(["collapsed_category", "time_period"])
        .size()
        .reset_index(name="count")
    )
    total_counts = grouped.groupby("collapsed_category")["count"].transform("sum")
    grouped["percentage"] = grouped["count"] / total_counts * 100

    # Calculate rate of change between the two periods
    rate_of_change_first_last = []
    for category in grouped["collapsed_category"].unique():
        cat_data = grouped[grouped["collapsed_category"] == category]
        periods = cat_data.set_index("time_period")
        if (
            adjusted_first_period in periods.index
            and adjusted_last_period in periods.index
        ):
            first = periods.loc[adjusted_first_period]
            last = periods.loc[adjusted_last_period]
            rate = (last["percentage"] - first["percentage"]) / (
                last_period_end - first_period_start
            )
            rate_of_change_first_last.append(
                {
                    "collapsed_category": category,
                    "start_time_period": adjusted_first_period,
                    "end_time_period": adjusted_last_period,
                    "rate_of_change_first_last": rate,
                }
            )

    return pd.DataFrame(rate_of_change_first_last)


from ipyleaflet import Map, TileLayer, GeoJSON
import ipywidgets as widgets


def recreate_layer(layer):
    """
    Safely recreate a common ipyleaflet layer (e.g., GeoJSON) from its core properties
    to avoid modifying the original object.
    """
    if isinstance(layer, GeoJSON):
        return GeoJSON(
            data=layer.data,
            style=layer.style or {},
            hover_style=layer.hover_style or {},
            name=layer.name or "",
        )
    elif isinstance(layer, TileLayer):
        return TileLayer(url=layer.url, name=layer.name or "")
    else:
        raise NotImplementedError(
            f"Layer type {type(layer)} not supported in recreate_layer."
        )


def create_opacity_slider_map(
    map1, map2, species_name, center=[40, -100], zoom=4, end_year=2025
):
    """
    Create a new map that overlays map2 on map1 with a year slider,
    fading opacity between the two. Original maps are unaffected.
    """
    # Initialize new map
    swipe_map = Map(center=center, zoom=zoom)

    # Re-add tile layers from both maps
    for layer in map1.layers + map2.layers:
        if isinstance(layer, TileLayer):
            swipe_map.add_layer(recreate_layer(layer))

    # Recreate and add overlay layers from both maps
    overlay_layers_1 = []
    overlay_layers_2 = []

    for layer in map1.layers:
        if not isinstance(layer, TileLayer):
            try:
                new_layer = recreate_layer(layer)
                overlay_layers_1.append(new_layer)
                swipe_map.add_layer(new_layer)
            except NotImplementedError:
                continue

    for layer in map2.layers:
        if not isinstance(layer, TileLayer):
            try:
                new_layer = recreate_layer(layer)
                overlay_layers_2.append(new_layer)
                swipe_map.add_layer(new_layer)
            except NotImplementedError:
                continue

    # Get year range
    start_year = int(get_start_year_from_species(species_name))
    end_year = end_year
    year_range = end_year - start_year

    # Create year slider with static labels
    slider = widgets.IntSlider(
        value=start_year,
        min=start_year,
        max=end_year,
        step=1,
        description="",
        layout=widgets.Layout(width="80%"),
        readout=False,
    )

    slider_box = widgets.HBox(
        [
            widgets.Label(str(start_year), layout=widgets.Layout(width="auto")),
            slider,
            widgets.Label(str(end_year), layout=widgets.Layout(width="auto")),
        ]
    )

    # Update opacity when slider changes
    def update_opacity(change):
        norm = (change["new"] - start_year) / year_range
        for layer in overlay_layers_1:
            if hasattr(layer, "style"):
                layer.style = {
                    **layer.style,
                    "opacity": 1 - norm,
                    "fillOpacity": 1 - norm,
                }
        for layer in overlay_layers_2:
            if hasattr(layer, "style"):
                layer.style = {**layer.style, "opacity": norm, "fillOpacity": norm}

    slider.observe(update_opacity, names="value")
    update_opacity({"new": start_year})  # Initialize

    return widgets.VBox([swipe_map, slider_box])


def get_species_code_if_exists(species_name):
    """
    Converts species name to 8-letter key and checks if it exists in REFERENCES.
    Returns the code if found, else returns False.
    """
    parts = species_name.strip().lower().split()
    if len(parts) >= 2:
        key = parts[0][:4] + parts[1][:4]
        return key if key in REFERENCES else False
    return False


def process_species_historical_range(new_map, species_name):
    """
    Wrapper function to process species range and classification using the HistoricalMap instance.
    Performs the following operations:
    1. Retrieves the species code using the species name.
    2. Loads the historic data for the species.
    3. Removes lakes from the species range.
    4. Merges touching polygons.
    5. Clusters and classifies the polygons.
    6. Updates the polygon categories.

    Args:
    - new_map (HistoricalMap): The map object that contains the species' historical data.
    - species_name (str): The name of the species to process.

    Returns:
    - updated_polygon: The updated polygon with classification and category information.
    """
    # Step 1: Get the species code
    code = get_species_code_if_exists(species_name)

    if not code:
        print(f"Species code not found for {species_name}.")
        return None

    # Step 2: Load historic data
    new_map.load_historic_data(species_name)

    # Step 3: Remove lakes from the species range
    range_no_lakes = new_map.remove_lakes(new_map.gdfs[code])

    # Step 4: Merge touching polygons
    merged_polygons = merge_touching_groups(range_no_lakes, buffer_distance=5000)

    # Step 5: Cluster and classify polygons
    clustered_polygons, largest_polygons = assign_polygon_clusters(merged_polygons)
    classified_polygons = classify_range_edges(clustered_polygons, largest_polygons)

    # Step 6: Update the polygon categories
    updated_polygon = update_polygon_categories(largest_polygons, classified_polygons)

    return updated_polygon


import os
import datetime


def save_results_as_csv(
    northward_rate_df,
    final_result,
    change,
    total_clim_result,
    category_clim_result,
    species_name,
):
    # Set up paths
    home_dir = os.path.expanduser("~")
    downloads_path = os.path.join(home_dir, "Downloads")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{species_name.replace(' ', '_')}_Results_{timestamp}"
    results_folder = os.path.join(downloads_path, folder_name)

    # Create results folder
    os.makedirs(results_folder, exist_ok=True)

    # Standardize the column name to 'category' and normalize categories to title case
    for df in [northward_rate_df, change, category_clim_result]:
        if "Category" in df.columns:
            df.rename(columns={"Category": "category"}, inplace=True)
        if "category" in df.columns:
            df["category"] = df["category"].str.title()

    # Merge the three DataFrames by category
    merged_df = northward_rate_df.merge(change, on="category", how="outer").merge(
        category_clim_result, on="category", how="outer"
    )

    # Drop duplicate species columns (if they exist)
    if "species_x" in merged_df.columns and "species_y" in merged_df.columns:
        merged_df.drop(columns=["species_x", "species_y"], inplace=True)

    merged_single = final_result.merge(total_clim_result, on="species", how="outer")

    # Save final_result as range_pattern.csv
    merged_single.to_csv(os.path.join(results_folder, "range_pattern.csv"), index=False)

    # Save the merged DataFrame (category_summary.csv)
    merged_df.to_csv(os.path.join(results_folder, "category_summary.csv"), index=False)

    # Optional: print file path
    # print(f"Results saved in folder: {results_folder}")


def save_modern_gbif_csv(classified_modern, species_name):
    # Set up paths
    home_dir = os.path.expanduser("~")
    downloads_path = os.path.join(home_dir, "Downloads")

    # Define the file name
    file_name = f"{species_name.replace(' ', '_')}_classified_modern.csv"

    # Save the DataFrame to CSV in the Downloads folder
    classified_modern.to_csv(os.path.join(downloads_path, file_name), index=False)


def save_historic_gbif_csv(classified_historic, species_name):
    # Set up paths
    home_dir = os.path.expanduser("~")
    downloads_path = os.path.join(home_dir, "Downloads")

    # Define the file name
    file_name = f"{species_name.replace(' ', '_')}_classified_historic.csv"

    # Save the DataFrame to CSV in the Downloads folder
    classified_historic.to_csv(os.path.join(downloads_path, file_name), index=False)


import requests
import geopandas as gpd
import pandas as pd
from rasterio import MemoryFile
from rasterstats import zonal_stats


def extract_raster_means_single_species(gdf, species_name):
    """
    gdf: GeoDataFrame with polygons (for a single species)
    species_name: string, the species name to assign to the output

    Returns:
    - total_df: DataFrame with species-wide averages
    - category_df: DataFrame with category-level averages
    """

    # Hardcoded GitHub raw URLs for rasters
    raster_urls = {
        "precipitation(mm)": "https://raw.githubusercontent.com/anytko/biospat_large_files/main/avg_precip.tif",
        "temperature(c)": "https://raw.githubusercontent.com/anytko/biospat_large_files/main/avg_temp.tif",
        "elevation(m)": "https://raw.githubusercontent.com/anytko/biospat_large_files/main/elev.tif",
    }

    # -------- Species-wide average --------
    row = {"species": species_name}

    for var_name, url in raster_urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            with MemoryFile(response.content) as memfile:
                with memfile.open() as src:
                    # Get zonal stats
                    stats = zonal_stats(
                        gdf.geometry,
                        src.read(1),
                        affine=src.transform,
                        nodata=src.nodata,
                        stats="mean",
                    )
                    values = [s["mean"] for s in stats if s["mean"] is not None]

                    # If zonal stats don't return valid values, use centroid fallback
                    if not values:
                        print(
                            f"No valid zonal stats for {var_name}, falling back to centroid method..."
                        )
                        values = []
                        for geom in gdf.geometry:
                            centroid = geom.centroid
                            row_idx, col_idx = src.index(centroid.x, centroid.y)
                            value = src.read(1)[row_idx, col_idx]
                            values.append(value)

                    # Ensure values are not empty before calculating the mean
                    if values:
                        row[var_name] = float(
                            sum(values) / len(values)
                        )  # Ensure the result is a float
                    else:
                        row[var_name] = None  # If no valid values, assign None
        except Exception as e:
            print(f"Error processing {var_name}: {e}")
            row[var_name] = None

    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds
    row["latitudinal_difference"] = maxy - miny
    row["longitudinal_difference"] = maxx - minx

    total_df = pd.DataFrame([row])

    # -------- Normalize and collapse category labels --------
    if "category" in gdf.columns:
        gdf["category"] = gdf["category"].str.strip().str.lower()

        category_mapping = {
            "leading (0.99)": "leading",
            "leading (0.95)": "leading",
            "leading (0.9)": "leading",
            "trailing (0.1)": "trailing",
            "trailing (0.05)": "trailing",
            "relict (0.01 latitude)": "relict",
            "relict (longitude)": "relict",
        }

        gdf["category"] = gdf["category"].replace(category_mapping)

    # -------- Category-level averages --------
    category_rows = []

    if "category" in gdf.columns:
        for category in gdf["category"].unique():
            subset = gdf[gdf["category"] == category]
            row = {
                "species": species_name,
                "category": category,
            }  # Reinitialize row here to avoid overwriting
            for var_name, url in raster_urls.items():
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with MemoryFile(response.content) as memfile:
                        with memfile.open() as src:
                            # Get zonal stats
                            stats = zonal_stats(
                                subset.geometry,
                                src.read(1),
                                affine=src.transform,
                                nodata=src.nodata,
                                stats="mean",
                            )
                            values = [s["mean"] for s in stats if s["mean"] is not None]

                            # If zonal stats don't return valid values, use centroid fallback
                            if not values:
                                # print(f"No valid zonal stats for category '{category}' and {var_name}, falling back to centroid method...")
                                values = []
                                for geom in subset.geometry:
                                    centroid = geom.centroid
                                    row_idx, col_idx = src.index(centroid.x, centroid.y)
                                    value = src.read(1)[row_idx, col_idx]
                                    values.append(value)

                            # Ensure values are not empty before calculating the mean
                            if values:
                                row[var_name] = float(
                                    sum(values) / len(values)
                                )  # Ensure the result is a float
                            else:
                                row[var_name] = None  # If no valid values, assign None
                except Exception as e:
                    print(f"Error processing {var_name} for category '{category}': {e}")
                    row[var_name] = None

            category_rows.append(row)

    category_df = pd.DataFrame(category_rows)

    return total_df, category_df


def calculate_density(df):
    # Count number of points per unique polygon (using geometry_id)
    point_counts = df.groupby("geometry_id").size().reset_index(name="point_count")

    # Merge point counts back into original dataframe
    df = df.merge(point_counts, on="geometry_id", how="left")

    # Calculate density: points per km²
    df["density"] = df["point_count"] / df["AREA"]
    df = df.drop(columns=["point_count"])

    return df


def merge_category_dataframes(northward_rate_df, change):
    """
    Merges three category-level dataframes on the 'category' column and returns the merged result.
    Standardizes 'category' casing to title case before merging.
    """
    import pandas as pd

    # Standardize 'category' column
    for df in [northward_rate_df, change]:
        if "Category" in df.columns:
            df.rename(columns={"Category": "category"}, inplace=True)
        if "category" in df.columns:
            df["category"] = df["category"].str.title()

    # Merge dataframes
    merged_df = northward_rate_df.merge(change, on="category", how="outer")

    # Drop duplicated species columns if they exist
    if "species_x" in merged_df.columns and "species_y" in merged_df.columns:
        merged_df.drop(columns=["species_x", "species_y"], inplace=True)

    cols_to_keep = [
        "species",
        "category",
        "northward_rate_km_per_year",
        "Rate of Change",
    ]
    merged_df = merged_df[[col for col in cols_to_keep if col in merged_df.columns]]

    return merged_df


import pandas as pd
import geopandas as gpd


def prepare_gdf_for_rasterization(gdf, df_values):
    """
    Merge polygon-level GeoDataFrame with range-level category values,
    and remove duplicate polygons.

    Parameters:
    - gdf: GeoDataFrame with polygons and category/density
    - df_values: DataFrame with category, northward_rate_km_per_year, Rate of Change

    Returns:
    - GeoDataFrame with merged attributes and unique geometries
    """

    # Standardize category column casing
    gdf["category"] = gdf["category"].str.title()
    df_values["category"] = df_values["category"].str.title()

    # Merge based on 'category'
    merged = gdf.merge(df_values, on="category", how="left")

    # Optional: handle missing Rate of Change or movement values
    merged.fillna({"Rate of Change": 0, "northward_rate_km_per_year": 0}, inplace=True)

    # Select relevant columns
    relevant_columns = [
        "geometry",
        "category",
        "density",
        "northward_rate_km_per_year",
        "Rate of Change",
    ]
    final_gdf = merged[relevant_columns]

    # Drop duplicate geometries
    final_gdf = final_gdf.drop_duplicates(subset="geometry")

    return final_gdf


def rasterize_multiband_gdf_match(
    gdf, value_columns, bounds=None, resolution=0.1666667
):
    """
    Rasterizes multiple value columns of a GeoDataFrame into a multiband raster with a specified resolution.

    Parameters:
    - gdf: GeoDataFrame with polygon geometries and numeric value_columns
    - value_columns: list of column names to rasterize into bands
    - bounds: bounding box (minx, miny, maxx, maxy). If None, computed from gdf.
    - resolution: The desired resolution of the raster in degrees (default is 10 minutes = 0.1666667 degrees).

    Returns:
    - 3D numpy array (bands, height, width)
    - affine transform
    - bounds used for rasterization
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    # Calculate bounds if not given
    if bounds is None:
        bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)

    minx, miny, maxx, maxy = bounds

    # Calculate the width and height of the raster
    width = int((maxx - minx) / resolution)  # number of cells in the x-direction
    height = int((maxy - miny) / resolution)  # number of cells in the y-direction

    # Create the transform based on bounds and resolution
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    bands = []

    for col in value_columns:
        shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[col])]
        raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=np.nan,
            dtype="float32",
        )
        bands.append(raster)

    stacked = np.stack(bands, axis=0)  # shape: (bands, height, width)
    return stacked, transform, (minx, miny, maxx, maxy)


def rasterize_multiband_gdf_world(gdf, value_columns, resolution=0.1666667):
    """
    Rasterizes multiple value columns of a GeoDataFrame into a multiband raster with a specified resolution
    covering the entire world.

    Parameters:
    - gdf: GeoDataFrame with polygon geometries and numeric value_columns
    - value_columns: list of column names to rasterize into bands
    - resolution: The desired resolution of the raster in degrees (default is 10 minutes = 0.1666667 degrees).

    Returns:
    - 3D numpy array (bands, height, width)
    - affine transform
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    # Define the bounds of the entire world
    minx, miny, maxx, maxy = -180, -90, 180, 90

    # Calculate the width and height of the raster based on the resolution
    width = int((maxx - minx) / resolution)  # number of cells in the x-direction
    height = int((maxy - miny) / resolution)  # number of cells in the y-direction

    # Create the transform based on the world bounds and new resolution
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    bands = []

    for col in value_columns:
        shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[col])]
        raster = rasterize(
            shapes,
            out_shape=(
                height,
                width,
            ),  # Ensure this matches the calculated height and width
            transform=transform,
            fill=np.nan,  # Fill areas outside the polygons with NaN
            dtype="float32",
        )
        bands.append(raster)

    stacked = np.stack(bands, axis=0)  # shape: (bands, height, width)
    return stacked, transform, (minx, miny, maxx, maxy)


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_propagule_pressure_range(stacked_raster, D=0.3, S=10.0, show_plot=True):
    # Extract input data
    density = stacked_raster[0]
    northward_rate = stacked_raster[1]  # in km/y
    category_raw = stacked_raster[3]

    # Replace NaNs with zeros
    density = np.nan_to_num(density, nan=0.0)
    northward_rate = np.nan_to_num(northward_rate, nan=0.0)
    category = np.nan_to_num(category_raw, nan=0).astype(int)

    # Identify occupied cells
    occupied_mask = density > 0

    # Compute distance and indices of nearest occupied cell
    distance, indices = distance_transform_edt(~occupied_mask, return_indices=True)

    # Gather source values
    nearest_y = indices[0]  # y-coordinate of nearest occupied cell
    current_y = np.indices(density.shape)[0]  # Current y-coordinates for each cell
    delta_y = (
        current_y - nearest_y
    )  # Distance from each cell to the nearest occupied cell

    # Debug: Check delta_y values for correct calculation
    # print(f"Delta Y (Calculated): {delta_y}")

    # Initialize direction modifier to 1
    direction_modifier = np.ones_like(northward_rate, dtype="float32")

    # Check northward rate for moving north or south and apply corresponding logic
    northward_mask = northward_rate > 0  # Mask for northward movement
    southward_mask = northward_rate < 0  # Mask for southward movement

    # Apply southward movement logic
    for y in range(density.shape[0]):
        for x in range(density.shape[1]):
            if occupied_mask[y, x]:
                rate = northward_rate[y, x]
                if rate != 0:
                    direction = 1 if rate < 0 else -1  # south = 1, north = -1
                    for dy in range(1, 4):  # How far outward to apply
                        ny = y + dy * direction
                        if 0 <= ny < density.shape[0]:
                            for dx in range(-dy, dy + 1):  # widen as you go further
                                nx = x + dx
                                if 0 <= nx < density.shape[1]:
                                    distance_factor = np.sqrt(dy**2 + dx**2)
                                    modifier = (abs(rate) * distance_factor) / S
                                    direction_modifier[ny, nx] += modifier

    # Clip to prevent out-of-bounds influence
    direction_modifier = np.clip(direction_modifier, 0.1, 2.0)

    # Apply northward movement logic
    # if np.any(northward_mask):
    # direction_modifier[northward_mask & (delta_y > 0)] = 1 - (np.abs(northward_rate[northward_mask & (delta_y > 0)]) * np.abs(delta_y[northward_mask & (delta_y > 0)])) / S
    # direction_modifier[northward_mask & (delta_y < 0)] = 1 + (np.abs(northward_rate[northward_mask & (delta_y < 0)]) * np.abs(delta_y[northward_mask & (delta_y < 0)])) / S
    # direction_modifier = np.clip(direction_modifier, 0.1, 2.0)

    for y in range(density.shape[0]):
        for x in range(density.shape[1]):
            if occupied_mask[y, x]:
                rate = northward_rate[y, x]
                if rate != 0:
                    direction = (
                        -1 if rate < 0 else 1
                    )  # north = -1, south = 1 (flipped direction)
                    for dy in range(1, 4):  # How far outward to apply
                        ny = y + dy * direction
                        if 0 <= ny < density.shape[0]:
                            for dx in range(-dy, dy + 1):  # widen as you go further
                                nx = x + dx
                                if 0 <= nx < density.shape[1]:
                                    distance_factor = np.sqrt(dy**2 + dx**2)
                                    modifier = (abs(rate) * distance_factor) / S
                                    direction_modifier[ny, nx] += modifier

    # Compute pressure from source density and distance

    pressure_nearest = density[nearest_y, indices[1]] * np.exp(-D * distance)

    D_self = density

    pressure = pressure_nearest + (D_self * np.exp(-D * 0))

    # pressure = density[nearest_y, indices[1]] * np.exp(-D * distance)
    # pressure = nearest_y * np.exp(-D * distance)

    # Apply directional influence (adjusting based on the direction_modifier)
    pressure_directional = pressure * direction_modifier

    # Apply category-based scaling
    scale_factors = {
        1: 1.5,  # Core
        2: 1.2,  # Leading
        3: 0.8,  # Trailing
        4: 1.0,  # Relict
    }
    scaling = np.ones_like(category, dtype="float32")
    for cat, factor in scale_factors.items():
        scaling[category == cat] = factor

    # Final pressure scaled
    pressure_scaled = pressure_directional * scaling

    edge_change_rate = np.nan_to_num(stacked_raster[2], nan=0.0)

    # Initialize modifier matrix (default = 1)
    edge_modifier = np.ones_like(edge_change_rate, dtype="float32")

    # Define which categories to include (Core=1, Leading=2, Trailing=3)
    target_categories = [1, 2, 3]

    for y in range(density.shape[0]):
        for x in range(density.shape[1]):
            if category[y, x] in target_categories:
                rate = edge_change_rate[y, x]
                if rate != 0:
                    # Spread influence outward from this cell
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            ny, nx = y + dy, x + dx
                            if (
                                0 <= ny < density.shape[0]
                                and 0 <= nx < density.shape[1]
                            ):
                                distance_factor = np.sqrt(dy**2 + dx**2)
                                if distance_factor == 0:
                                    distance_factor = 1  # to avoid division by zero
                                modifier = (rate * (1 / distance_factor)) / S
                                edge_modifier[ny, nx] += modifier

    # Clip to keep values within a stable range
    edge_modifier = np.clip(edge_modifier, 0.1, 2.0)

    # Apply additional edge-based pressure influence
    pressure_scaled *= edge_modifier

    return pressure_scaled


def cat_int_mapping(preped_gdf):
    """
    Copies the input GeoDataFrame, maps the 'category' column to integers,
    and transforms the CRS to EPSG:4326.

    Parameters:
        preped_gdf (GeoDataFrame): Input GeoDataFrame with a 'category' column.

    Returns:
        GeoDataFrame: Transformed GeoDataFrame with a new 'category_int' column and EPSG:4326 CRS.
    """
    category_map = {"Core": 1, "Leading": 2, "Trailing": 3, "Relict": 4}
    gdf = preped_gdf.copy()
    gdf["category_int"] = gdf["category"].map(category_map)
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


def full_propagule_pressure_pipeline(
    classified_modern, northward_rate_df, change, resolution=0.1666667
):
    """
    Full wrapper pipeline to compute propagule pressure from input data.

    Steps:
        1. Merge category dataframes.
        2. Prepare GeoDataFrame for rasterization.
        3. Map category strings to integers.
        4. Rasterize to show and save versions.
        5. Compute propagule pressure for both rasters.

    Parameters:
        classified_modern (GeoDataFrame): GeoDataFrame with spatial features and categories.
        northward_rate_df (DataFrame): Contains northward movement rate per point or cell.
        change (DataFrame): Contains rate of change per point or cell.

    Returns:
        tuple: (pressure_show, pressure_save), both as 2D numpy arrays
    """

    # Step 1: Merge data
    merged = merge_category_dataframes(northward_rate_df, change)

    # Step 2: Prepare for rasterization
    preped_gdf = prepare_gdf_for_rasterization(classified_modern, merged)

    # Step 3: Map category to integers
    preped_gdf_new = cat_int_mapping(
        preped_gdf
    )  # assumes this was renamed from cat_int_mapping

    # Step 4: Rasterize
    value_columns = [
        "density",
        "northward_rate_km_per_year",
        "Rate of Change",
        "category_int",
    ]
    raster_show, transform, show_bounds = rasterize_multiband_gdf_match(
        preped_gdf_new, value_columns, resolution=resolution
    )
    raster_save, transform, save_bounds = rasterize_multiband_gdf_world(
        preped_gdf_new, value_columns, resolution=resolution
    )

    # Step 5: Compute propagule pressure
    pressure_show = compute_propagule_pressure_range(raster_show)
    pressure_save = compute_propagule_pressure_range(raster_save)

    return pressure_show, pressure_save, show_bounds, save_bounds


import os
import rasterio
from rasterio.transform import from_bounds


def save_raster_to_downloads_range(array, bounds, species):
    """
    Saves a NumPy raster array as a GeoTIFF to the user's Downloads folder.

    Parameters:
        array (ndarray): The raster data to save.
        bounds (tuple): Bounding box in the format (minx, miny, maxx, maxy).
        species (str): The species name to use in the output filename.
    """
    try:
        # Clean filename
        clean_species = species.strip().replace(" ", "_")
        filename = f"{clean_species}_persistence_raster.tif"

        # Determine Downloads path
        home_dir = os.path.expanduser("~")
        downloads_path = os.path.join(home_dir, "Downloads", filename)

        # Generate raster transform
        transform = from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3], array.shape[1], array.shape[0]
        )

        # Write to GeoTIFF
        with rasterio.open(
            downloads_path,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(array, 1)

        # print(f"Raster successfully saved to: {downloads_path}")
        return downloads_path

    except Exception as e:
        print(f"Error saving raster: {e}")
        return None


def save_raster_to_downloads_global(array, bounds, species):
    """
    Saves a NumPy raster array as a GeoTIFF to the user's Downloads folder.

    Parameters:
        array (ndarray): The raster data to save.
        bounds (tuple): Bounding box in the format (minx, miny, maxx, maxy).
        species (str): The species name to use in the output filename.
    """
    try:
        # Clean filename
        clean_species = species.strip().replace(" ", "_")
        filename = f"{clean_species}_persistence_raster_global.tif"

        # Determine Downloads path
        home_dir = os.path.expanduser("~")
        downloads_path = os.path.join(home_dir, "Downloads", filename)

        # Generate raster transform
        transform = from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3], array.shape[1], array.shape[0]
        )

        # Write to GeoTIFF
        with rasterio.open(
            downloads_path,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(array, 1)

        # print(f"Raster successfully saved to: {downloads_path}")
        return downloads_path

    except Exception as e:
        print(f"Error saving raster: {e}")
        return None

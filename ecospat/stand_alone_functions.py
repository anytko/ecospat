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
        if largest_polygon_area > 1000:
            long_value = 0.5  # for very large polygons, allow 10% longitude diff
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
                return "relict (0.001 longitude)"
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


def update_polygon_categories(largest_polygons, classified_polygons):

    island_states_url = "https://raw.githubusercontent.com/anytko/biospat_large_files/main/island_states.geojson"

    island_states_gdf = gpd.read_file(island_states_url)

    largest_polygons_gdf = gpd.GeoDataFrame(largest_polygons)
    classified_polygons_gdf = gpd.GeoDataFrame(classified_polygons)
    island_states_gdf = island_states_gdf.to_crs("EPSG:3395")

    # Step 1: Set CRS for both GeoDataFrames (ensure consistency)
    largest_polygons_gdf.set_crs("EPSG:3395", inplace=True)
    classified_polygons_gdf.set_crs(
        "EPSG:3395", inplace=True
    )  # Ensure this matches the CRS of largest_polygons_gdf

    largest_polygons_gdf = gpd.sjoin(
        largest_polygons_gdf,
        classified_polygons[["geometry", "category"]],
        how="left",
        predicate="intersects",
    )

    # Step 3: Perform spatial join with classified_polygons_gdf and island_states_gdf (Assuming this is correct)
    overlapping_polygons = gpd.sjoin(
        classified_polygons_gdf, island_states_gdf, how="inner", predicate="intersects"
    )

    # Step 4: Clean up overlapping polygons
    overlapping_polygons = overlapping_polygons.rename(
        columns={"index": "overlapping_index"}
    )
    overlapping_polygons_new = overlapping_polygons.drop_duplicates(subset="geometry")

    # Step 5: Compute centroids for distance calculation
    overlapping_polygons_new["centroid"] = overlapping_polygons_new.geometry.centroid
    largest_polygons_gdf["centroid"] = largest_polygons_gdf.geometry.centroid

    # Step 6: Extract coordinates of centroids
    overlapping_centroids = (
        overlapping_polygons_new["centroid"].apply(lambda x: (x.x, x.y)).tolist()
    )
    largest_centroids = (
        largest_polygons_gdf["centroid"].apply(lambda x: (x.x, x.y)).tolist()
    )

    # Step 7: Calculate pairwise distance matrix
    distances = cdist(overlapping_centroids, largest_centroids)

    # Step 8: Find closest largest_polygon for each overlapping polygon
    closest_indices = distances.argmin(axis=1)

    # Step 9: Reassign 'category' from closest largest_polygons to overlapping_polygons
    overlapping_polygons_new["category"] = largest_polygons_gdf.iloc[closest_indices][
        "category"
    ].values

    # Step 10: Update categories in the original classified_polygons based on matching geometries
    # Here, we're only updating the category for polygons in the original gdf that overlap
    updated_classified_polygons = classified_polygons.copy()

    # Update only the overlapping polygons in the original GeoDataFrame
    updated_classified_polygons.loc[overlapping_polygons_new.index, "category"] = (
        overlapping_polygons_new["category"]
    )

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
        species = " ".join(scientific_name.split()[:2]) if scientific_name else None
        if lat is not None and lon is not None:
            records.append(
                {
                    "species": species,
                    "decimalLatitude": lat,
                    "decimalLongitude": lon,
                    "year": year,
                    "basisOfRecord": basis,
                    "geometry": Point(lon, lat),
                }
            )

    df = pd.DataFrame(records)
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
        data[["decimalLatitude", "decimalLongitude", "year"]]
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


def clip_polygons_to_continent(input_gdf):
    """
    Clips the input GeoDataFrame geometries to continent boundaries (e.g., North America).

    Parameters:
    - input_gdf: GeoDataFrame with columns ['point_geometry', 'year', 'geometry'].
    - continents_geojson: Path to the GeoJSON file with continent boundaries.

    Returns:
    - GeoDataFrame with same columns but clipped geometries.
    """
    # Load continents GeoJSON

    land_url = (
        "https://raw.githubusercontent.com/anytko/biospat_large_files/main/land.geojson"
    )

    continents_gdf = gpd.read_file(land_url)

    # Ensure valid geometries only
    input_gdf = input_gdf[input_gdf["geometry"].is_valid]
    continents_gdf = continents_gdf[continents_gdf["geometry"].is_valid]

    input_gdf.set_geometry(
        "geometry", inplace=True
    )  # or 'polygon_geometry' if that's the active column

    # Ensure CRS matches
    if input_gdf.crs != continents_gdf.crs:
        input_gdf = input_gdf.to_crs(continents_gdf.crs)

    # Clip input geometries to the continents
    clipped = gpd.overlay(
        input_gdf[["point_geometry", "year", "geometry"]],
        continents_gdf[["geometry"]],
        how="intersection",
    )

    # Define the bounding box for North America (Canada, US, Mexico)
    north_america_bbox = box(-178.2, 6.6, -49.0, 83.3)
    north_america_gdf = gpd.GeoDataFrame(
        geometry=[north_america_bbox], crs=input_gdf.crs
    )

    # Clip again to North America bounding box
    clipped = gpd.overlay(clipped, north_america_gdf[["geometry"]], how="intersection")

    clipped = clipped.to_crs(epsg=4326)

    # Check for empty geometries
    empty_geometries = clipped[clipped["geometry"].is_empty]
    if not empty_geometries.empty:
        print(
            f"Warning: Found {len(empty_geometries)} empty geometries after clipping."
        )

    return clipped


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

        if area_difference > 600:
            polygon_threshold = 0.2
        elif area_difference > 200:
            polygon_threshold = 0.005
        else:
            polygon_threshold = 0.2

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
        long_value = 1 if largest_polygon_area > 100000 else 0.5

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
                return "relict (0.001 longitude)"
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
    largest_polygons_gdf = largest_polygons_gdf.to_crs(crs)
    classified_polygons_gdf = classified_polygons_gdf.to_crs(crs)

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


def remove_lakes_and_plot_gbif(polygons_gdf):
    """
    Removes lake polygons from range polygons and plots them using Folium.

    Parameters:
    - polygons_gdf: GeoDataFrame of range polygons.

    Returns:
    - Updated GeoDataFrame with the lakes removed.
    """
    # Load lakes GeoDataFrame

    lakes_url = "https://raw.githubusercontent.com/anytko/biospat_large_files/main/lakes_na.geojson"

    lakes_gdf = gpd.read_file(lakes_url)

    # Ensure geometries are valid
    polygons_gdf = polygons_gdf[polygons_gdf.geometry.is_valid]
    lakes_gdf = lakes_gdf[lakes_gdf.geometry.is_valid]

    # Ensure CRS matches before performing spatial operations
    if polygons_gdf.crs != lakes_gdf.crs:
        polygons_gdf = polygons_gdf.to_crs(lakes_gdf.crs)

    # Deduplicate the range polygons by geometry
    unique_gdf = polygons_gdf.drop_duplicates(subset="geometry")

    # Clip the unique polygons with the lake polygons (difference operation)
    polygons_no_lakes_gdf = gpd.overlay(unique_gdf, lakes_gdf, how="difference")

    # Handle cases where no lake intersection was found
    polygons_no_lakes_gdf = polygons_no_lakes_gdf.copy()
    polygons_no_lakes_gdf["geometry"] = polygons_no_lakes_gdf["geometry"].fillna(
        unique_gdf["geometry"]
    )

    # Check for empty geometries in the resulting GeoDataFrame
    empty_geometries = polygons_no_lakes_gdf[polygons_no_lakes_gdf["geometry"].is_empty]

    if not empty_geometries.empty:
        print(
            f"Warning: Found {len(empty_geometries)} empty geometries after clipping."
        )

    # Return the updated GeoDataFrame
    return polygons_no_lakes_gdf

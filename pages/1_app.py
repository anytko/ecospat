import datetime
import solara
from solara import component, HTML, VBox, HBox
import ecospat


import datetime
from solara import component, reactive, HTML


# Reactive state
species_name = solara.reactive("")
gbif_limit = solara.reactive(2000)
end_date = solara.reactive(datetime.date.today().isoformat())
baseline_mortality = solara.reactive(0.1)
distance_decay = solara.reactive(0.3)
directional_modifier = solara.reactive(10)
resolution = solara.reactive(0.1666667)
user_start_year = solara.reactive(None)
lat_min = solara.reactive(6.6)
lat_max = solara.reactive(83.3)
lon_min = solara.reactive(-178.2)
lon_max = solara.reactive(-49.0)


basis_options = [
    "OBSERVATION",
    "OCCURRENCE",
    "FOSSIL_SPECIMEN",
    "HUMAN_OBSERVATION",
    "LIVING_SPECIMEN",
    "MACHINE_OBSERVATION",
    "MATERIAL_CITATION",
    "PRESERVED_SPECIMEN",
]

# reactive dict to hold which options are selected
basis_of_record = solara.reactive({key: False for key in basis_options})
selected_basis = solara.reactive([k for k, v in basis_of_record.value.items() if v])
search_clicked = solara.reactive(False)
filter_mode = solara.Reactive("Simple Filters")


@solara.component
def Sidebar():
    def on_search():
        # Trigger search
        search_clicked.set(not search_clicked.value)
        print("Search button clicked!")  # confirm click
        print(f"Species name: {species_name.value}")

    children = [
        solara.Markdown("## Filters"),
        solara.InputText(label="Species", value=species_name),
        solara.Button(
            "Search",
            on_click=on_search,
            style={"width": "100%", "marginTop": "10px", "marginBottom": "10px"},
        ),
        solara.ToggleButtonsSingle(
            value=filter_mode,
            values=["Simple Filters", "All Filters"],
            style={
                "display": "flex",
                "justifyContent": "center",
            },
        ),
    ]

    # GBIF Section

    def update_basis(val, option):
        basis_of_record.value[option] = val
        selected_basis.value = [k for k, v in basis_of_record.value.items() if v]

    gbif_children = []
    if filter_mode.value in ["Simple Filters", "All Filters"]:
        gbif_children.append(solara.InputInt("GBIF Limit", value=gbif_limit))
    if filter_mode.value == "All Filters":
        gbif_children.extend(
            [
                solara.InputText(
                    "Start Year (only required for global species)",
                    value=user_start_year,
                ),
                solara.InputText("End Date", value=end_date),
                solara.Markdown("### Bounding Box"),
                solara.InputText("Latitudinal Minimum", value=lat_min),
                solara.InputText("Latitudinal Maximum", value=lat_max),
                solara.InputText("Longitudinal Minimum", value=lon_min),
                solara.InputText("Longitudinal Maximum", value=lon_max),
                solara.Markdown("**Basis of Record**"),
                solara.Column(
                    [
                        solara.Checkbox(
                            label=option,
                            value=basis_of_record.value[option],
                            on_value=lambda val, opt=option: update_basis(val, opt),
                        )
                        for option in basis_options
                    ],
                    gap="1px",
                ),
            ]
        )
    children.append(solara.Card("GBIF Data", children=gbif_children))

    # Biology Section
    bio_children = []
    if filter_mode.value in ["Simple Filters", "All Filters"]:
        bio_children.append(
            solara.InputFloat("Baseline Mortality", value=baseline_mortality)
        )
    if filter_mode.value == "All Filters":
        bio_children.extend(
            [
                solara.InputFloat("Distance Decay", value=distance_decay),
                solara.InputInt("Directional Modifier", value=directional_modifier),
            ]
        )
    children.append(solara.Card("Biology", children=bio_children))

    # Output Section
    output_children = []
    if filter_mode.value in ["Simple Filters", "All Filters"]:
        output_children.append(solara.InputFloat("Raster Resolution", value=resolution))
    children.append(solara.Card("Output", children=output_children))

    return solara.Column(
        children,
        style={
            "width": "30vw",
            "minWidth": "10vw",
            "maxWidth": "25vw",
            "padding": "20px",
            "backgroundColor": "rgb(248, 249, 250)",
            "borderRight": "1px solid #ddd",
            "boxShadow": "3px 0 5px rgba(0,0,0,0.1)",
            "height": "100vh",
            "overflow": "auto",
        },
    )


import solara
import json
import ipyleaflet
import ecospat.ecospat as ecospat_full
from ecospat.stand_alone_functions import process_species_historical_range
from shapely.geometry import shape

# Define your master colors
master_category_colors = {
    "leading (0.99)": "#8d69b8",
    "leading (0.95)": "#519e3e",
    "leading (0.9)": "#ef8636",
    "core": "#3b75af",
    "trailing (0.1)": "#58bbcc",
    "trailing (0.05)": "#bcbd45",
    "relict (0.01 latitude)": "#84584e",
    "relict (longitude)": "#7f7f7f",
}

whole_colors = {
    "leading": "#519e3e",
    "core": "#3b75af",
    "trailing": "#bcbd45",
    "relict": "#84584e",
}


def style_callback(feature):
    category = feature["properties"].get("category", None)
    color = master_category_colors.get(category, "#000000")
    return {"color": color, "weight": 2, "fillColor": color, "fillOpacity": 0.5}


def style_callback_hist(feature):
    category = feature["properties"].get("category", None)
    color = whole_colors.get(category, "#000000")
    return {"color": color, "weight": 2, "fillColor": color, "fillOpacity": 0.5}


def style_callback_points(feature):
    edge_val = feature["properties"].get("edge_vals", "core")
    color = whole_colors.get(edge_val, "#3b75af")
    return {"color": color, "weight": 2, "fillColor": color, "fillOpacity": 0.5}


use_historic_map = solara.reactive(False)
hist_range_data = solara.reactive(None)

import ipywidgets as widgets


import json
import solara
import ipyleaflet
import ipywidgets as widgets
from shapely.geometry import Point, Polygon


@solara.component
def HistoricalRangeMap(species_name: str, hist_range):
    if not species_name:
        return solara.Text("Enter a species name and press search.")

    if hist_range is None:
        return solara.Text("No historic range available.")

    geojson_dict = json.loads(hist_range.to_json())

    # Info box
    original_message = "<b>Hover over a polygon</b>"
    info_box = widgets.HTML(value=original_message)

    # GeoJSON layer
    geojson_layer = ipyleaflet.GeoJSON(
        data=geojson_dict,
        style_callback=style_callback,
        hover_style={"fillOpacity": 0.7},
    )

    # Hover over polygon
    def on_hover(event, feature, **kwargs):
        if feature is not None:
            category = feature["properties"].get("category", "N/A")
            info_box.value = f"<b>Category:</b> {category}"

    geojson_layer.on_hover(on_hover)

    # Map widget
    map_widget = ipyleaflet.Map(
        center=(0, 0),
        zoom=2,
        scroll_wheel_zoom=True,
        layers=[
            ipyleaflet.TileLayer(
                url="https://tile.gbif.org/3857/omt/{z}/{x}/{y}@1x.png?style=gbif-classic",
                name="GBIF Classic",
                attribution="GBIF",
            ),
            geojson_layer,
        ],
    )

    # Prepare Shapely polygons for point-in-polygon check
    polygons = []
    for feature in geojson_dict["features"]:
        geom_type = feature["geometry"]["type"]
        coords = feature["geometry"]["coordinates"]

        if geom_type == "Polygon":
            poly_coords = [tuple(c) for c in coords[0]]
            polygons.append(
                (Polygon(poly_coords), feature["properties"].get("category", "N/A"))
            )

        elif geom_type == "MultiPolygon":
            for part in coords:
                poly_coords = [tuple(c) for c in part[0]]
                polygons.append(
                    (Polygon(poly_coords), feature["properties"].get("category", "N/A"))
                )

    # Reset info box when mouse leaves all polygons
    def on_mouse_move(**kwargs):
        if kwargs.get("type") != "mousemove":
            return  # Ignore non-mousemove events

        coords = kwargs.get("coordinates", None)
        if coords is None:
            return

        lat, lon = coords  # ipyleaflet returns [lat, lon]
        over_any = False
        for poly, category in polygons:
            if poly.covers(Point(lon, lat)):  # includes boundary points
                info_box.value = f"<b>Category:</b> {category}"
                over_any = True
                break

        if not over_any:
            info_box.value = original_message

    map_widget.on_interaction(on_mouse_move)

    csv_buffer = io.StringIO()
    hist_range.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    # Return layout with download button under map
    return solara.Column(
        [
            solara.Markdown(f"## {species_name} Historical Range Map"),
            info_box,
            map_widget,
            solara.FileDownload(
                data=csv_bytes,
                filename=f"{species_name}_historical_range.csv",
                label="Download Historical Range CSV",
            ),
        ]
    )


import solara
import pandas as pd
from ecospat.stand_alone_functions import analyze_species_distribution


@solara.component
def SpeciesSummary(
    northward_movement: pd.DataFrame,
    pop_change: pd.DataFrame,
    range_cat: pd.DataFrame,
):
    # Ensure category is a column (not index)
    if "category" not in northward_movement.columns:
        northward_movement = northward_movement.reset_index()
    if "category" not in pop_change.columns:
        pop_change = pop_change.reset_index()

    # Rename columns to avoid clashes
    northward_movement = northward_movement.rename(
        columns={
            "change_km": "northward_change_km",
            "rate_km_per_year": "northward_rate_km_per_year",
        }
    )
    pop_change = pop_change.rename(
        columns={
            "collapsed_category": "category",
            "rate_of_change_first_last": "rate_of_change_first_last",
            "start_time_period": "start_time_period",
            "end_time_period": "end_time_period",
        }
    )

    # Merge on category (outer to keep all)
    merged = pd.merge(
        northward_movement,
        pop_change[
            [
                "category",
                "start_time_period",
                "end_time_period",
                "rate_of_change_first_last",
            ]
        ],
        on="category",
        how="outer",
    )

    final_df = merged.rename(
        columns={
            "species": "Species",
            "category": "Category",
            "start_time_period": "Start Time Period",
            "end_time_period": "End Time Period",
            "northward_change_km": "Northward Change (km)",
            "northward_rate_km_per_year": "Northward Rate of Change (km/y)",
            "rate_of_change_first_last": "Population Rate of Change",
        }
    )

    cols = [
        "Species",
        "Category",
        "Start Time Period",
        "End Time Period",
        "Northward Change (km)",
        "Northward Rate of Change (km/y)",
        "Population Rate of Change",
    ]
    final_df = final_df[cols]

    display_df = final_df.copy()
    for col in display_df.select_dtypes(include="number").columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")

    range_cat_value = ""
    if "category" in range_cat.columns and not range_cat.empty:
        range_cat_value = str(range_cat["category"].iloc[0]).title()

    csv_buffer = io.StringIO()
    final_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    # Return layout
    with solara.Column(margin=4) as column_header_info:
        pass

    return solara.Column(
        margin=6,
        children=[
            solara.Markdown(f"## Species Summary for **{species_name.value}**"),
            solara.Markdown(f"###**Range Dynamic:** {range_cat_value}"),
            solara.Column(
                children=[
                    solara.DataFrame(
                        display_df,
                        column_header_info=column_header_info,
                        items_per_page=10,
                    ),
                    solara.FileDownload(
                        data=csv_bytes,
                        filename=f"{species_name.value}_summary.csv",
                        label="Download CSV",
                    ),
                ]
            ),
        ],
    )


import asyncio
import solara
import pandas as pd
from ecospat.stand_alone_functions import summarize_polygons_with_points


def HistoricFallbackMap(classified_historic: pd.DataFrame):
    """Render a fallback map using GeoJSON from the historic classified data."""
    if classified_historic.empty:
        return solara.Text("No historic data available.")

    polygons_gdf = summarize_polygons_with_points(classified_historic)

    if polygons_gdf.empty:
        return solara.Text("No historic data available.")

    geojson_dict = json.loads(polygons_gdf.to_json())

    # Info box
    original_message = "<b>Hover over a polygon</b>"
    info_box = widgets.HTML(value=original_message)

    # Create GeoJSON layer
    geojson_layer = ipyleaflet.GeoJSON(
        data=geojson_dict,
        style_callback=style_callback,
        hover_style={"fillOpacity": 0.7},
    )

    # Hover over polygon
    def on_hover(event, feature, **kwargs):
        if feature is not None:
            category = feature["properties"].get("category", "N/A")
            info_box.value = f"<b>Category:</b> {category}"

    geojson_layer.on_hover(on_hover)

    # Map widget
    map_widget = ipyleaflet.Map(
        center=(0, 0),
        zoom=2,
        scroll_wheel_zoom=True,
        layers=[
            ipyleaflet.TileLayer(
                url="https://tile.gbif.org/3857/omt/{z}/{x}/{y}@1x.png?style=gbif-classic",
                name="GBIF Classic",
                attribution="GBIF",
            ),
            geojson_layer,
        ],
    )

    # Prepare Shapely polygons for point-in-polygon check
    polygons = []
    for feature in geojson_dict["features"]:
        geom_type = feature["geometry"]["type"]
        coords = feature["geometry"]["coordinates"]

        if geom_type == "Polygon":
            poly_coords = [tuple(c) for c in coords[0]]
            polygons.append(
                (Polygon(poly_coords), feature["properties"].get("category", "N/A"))
            )

        elif geom_type == "MultiPolygon":
            for part in coords:
                poly_coords = [tuple(c) for c in part[0]]
                polygons.append(
                    (Polygon(poly_coords), feature["properties"].get("category", "N/A"))
                )

    # Reset info box when mouse leaves all polygons
    def on_mouse_move(**kwargs):
        if kwargs.get("type") != "mousemove":
            return  # Ignore non-mousemove events

        coords = kwargs.get("coordinates", None)
        if coords is None:
            return

        lat, lon = coords  # ipyleaflet returns [lat, lon]
        over_any = False
        for poly, category in polygons:
            if poly.covers(Point(lon, lat)):  # includes boundary points
                info_box.value = f"<b>Category:</b> {category}"
                over_any = True
                break

        if not over_any:
            info_box.value = original_message

    map_widget.on_interaction(on_mouse_move)

    csv_buffer = io.StringIO()
    polygons_gdf.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    # Return layout with download button under map
    return solara.Column(
        [
            solara.Markdown(f"## {species_name} Historical Range Map"),
            info_box,
            map_widget,
            solara.FileDownload(
                data=csv_bytes,
                filename=f"{species_name}_historical_range.csv",
                label="Download Historical Range CSV",
            ),
        ]
    )


def ModernMap(classified_modern: pd.DataFrame):
    """Render a fallback map using GeoJSON from the modern classified data."""
    if classified_modern.empty:
        return solara.Text("No modern data available.")

    polygons_gdf = summarize_polygons_with_points(classified_modern)

    if polygons_gdf.empty:
        return solara.Text("No modern data available.")

    geojson_dict = json.loads(polygons_gdf.to_json())

    # Info box
    original_message = "<b>Hover over a polygon</b>"
    info_box = widgets.HTML(value=original_message)

    # Create GeoJSON layer
    geojson_layer = ipyleaflet.GeoJSON(
        data=geojson_dict,
        style_callback=style_callback,
        hover_style={"fillOpacity": 0.7},
    )

    # Hover over polygon
    def on_hover(event, feature, **kwargs):
        if feature is not None:
            category = feature["properties"].get("category", "N/A")
            info_box.value = f"<b>Category:</b> {category}"

    geojson_layer.on_hover(on_hover)

    # Map widget
    map_widget = ipyleaflet.Map(
        center=(0, 0),
        zoom=2,
        scroll_wheel_zoom=True,
        layers=[
            ipyleaflet.TileLayer(
                url="https://tile.gbif.org/3857/omt/{z}/{x}/{y}@1x.png?style=gbif-classic",
                name="GBIF Classic",
                attribution="GBIF",
            ),
            geojson_layer,
        ],
    )

    # Prepare Shapely polygons for point-in-polygon check
    polygons = []
    for feature in geojson_dict["features"]:
        geom_type = feature["geometry"]["type"]
        coords = feature["geometry"]["coordinates"]

        if geom_type == "Polygon":
            poly_coords = [tuple(c) for c in coords[0]]
            polygons.append(
                (Polygon(poly_coords), feature["properties"].get("category", "N/A"))
            )

        elif geom_type == "MultiPolygon":
            for part in coords:
                poly_coords = [tuple(c) for c in part[0]]
                polygons.append(
                    (Polygon(poly_coords), feature["properties"].get("category", "N/A"))
                )

    # Reset info box when mouse leaves all polygons
    def on_mouse_move(**kwargs):
        if kwargs.get("type") != "mousemove":
            return  # Ignore non-mousemove events

        coords = kwargs.get("coordinates", None)
        if coords is None:
            return

        lat, lon = coords  # ipyleaflet returns [lat, lon]
        over_any = False
        for poly, category in polygons:
            if poly.covers(Point(lon, lat)):  # includes boundary points
                info_box.value = f"<b>Category:</b> {category}"
                over_any = True
                break

        if not over_any:
            info_box.value = original_message

    map_widget.on_interaction(on_mouse_move)

    csv_buffer = io.StringIO()
    polygons_gdf.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    # Return layout with download button under map
    return solara.Column(
        [
            solara.Markdown(f"## {species_name.value} Modern Range Map"),
            info_box,
            map_widget,
            solara.FileDownload(
                data=csv_bytes,
                filename=f"{species_name}_modern_range.csv",
                label="Download Modern Range CSV",
            ),
        ]
    )


import solara
import ipyleaflet
import numpy as np
import base64
import io
from PIL import Image


@solara.component
def RasterMap(
    raster_array: np.ndarray,
    transform,
    preped_new=None,
    value_columns=None,
    cmap="viridis",
    resolution=None,
    distance_decay=None,
    directional_modifier=None,
):
    """
    Display a raster array on an ipyleaflet map using Solara.
    Includes download buttons for the current raster and the world raster.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import ipyleaflet
    from PIL import Image
    import io, base64
    import solara
    import rasterio
    import numpy as np

    # --- Normalize and convert raster to RGBA ---
    arr = raster_array.astype(float)
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    normed = (arr - arr_min) / (arr_max - arr_min + 1e-9)
    colormap = cm.get_cmap(cmap)
    rgba_img = (colormap(normed) * 255).astype(np.uint8)

    # --- Convert to PNG in memory for map display ---
    image_pil = Image.fromarray(rgba_img, mode="RGBA")
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{img_b64}"

    # --- Compute bounds from transform ---
    height, width = raster_array.shape
    x_min, y_max = transform * (0, 0)
    x_max, y_min = transform * (width, height)
    bounds = ((y_min, x_min), (y_max, x_max))

    # --- Raster layer ---
    raster_layer = ipyleaflet.ImageOverlay.element(
        url=data_url,
        bounds=bounds,
        opacity=1,
    )

    # --- Helper to convert array to 8-bit RGB GeoTIFF bytes ---
    def raster_to_bytes_rgb(array, transform, cmap="viridis", crs="EPSG:4326"):
        arr_min, arr_max = np.nanmin(array), np.nanmax(array)
        normed = (array - arr_min) / (arr_max - arr_min + 1e-9)
        rgba_img = (cm.get_cmap(cmap)(normed)[:, :, :3] * 255).astype(
            np.uint8
        )  # RGB only
        bands = np.moveaxis(rgba_img, -1, 0)  # shape (3, height, width)

        buffer = io.BytesIO()
        with rasterio.open(
            buffer,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=3,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(bands)
        buffer.seek(0)
        return buffer.read()

    # --- Download buttons ---
    range_download = solara.FileDownload(
        data=raster_to_bytes_rgb(raster_array, transform, cmap=cmap),
        filename="range_raster.tif",
        label="Download Range Raster",
    )

    def compute_world_raster_bytes():
        # compute world raster dynamically
        raster_show, world_transform, show_bounds = rasterize_multiband_gdf_world(
            preped_new, value_columns, resolution
        )
        pressure_show = compute_propagule_pressure_range(
            raster_show, D=distance_decay, S=directional_modifier
        )
        return raster_to_bytes_rgb(pressure_show, world_transform, cmap=cmap)

    world_download = solara.FileDownload(
        data=compute_world_raster_bytes,
        filename="world_raster.tif",
        label="Download World Raster",
    )

    # --- Return map + download buttons ---
    map_widget = ipyleaflet.Map.element(
        center=((y_min + y_max) / 2, (x_min + x_max) / 2),
        zoom=3,
        scroll_wheel_zoom=True,
        layers=[
            ipyleaflet.TileLayer.element(
                url="https://tile.gbif.org/3857/omt/{z}/{x}/{y}@1x.png?style=gbif-classic",
                name="GBIF Classic",
                attribution="GBIF",
            ),
            raster_layer,
        ],
    )

    return solara.Column(
        [
            solara.Markdown(f"## {species_name.value} Propagule Pressure Rasters"),
            map_widget,
            solara.Row([range_download, world_download]),
        ]
    )


import json
import solara
import ipyleaflet

import solara
import ipyleaflet
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from ipyleaflet import Popup
import ipywidgets as widgets


@solara.component
def MapWithPolygonsAndPoints(df):
    import ipyleaflet
    import ipywidgets as widgets
    from shapely.geometry import Polygon, Point
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # --- Compute deviations ---
    df = df.copy()
    expected_1y = 1 - df["baseline_death"]
    df["deviation_1y"] = df["P_1y"] - expected_1y

    expected_5y = (1 - df["baseline_death"]) ** 5
    df["deviation_5y"] = df["P_5y"] - expected_5y

    # --- Prepare polygon GeoJSON ---
    polygons = df.copy()
    summary_polygons = summarize_polygons_for_point_plot(polygons)

    polygons_json = {"type": "FeatureCollection", "features": []}

    for _, row in summary_polygons.iterrows():
        geom = row["geometry"]

        # handle single Polygon or MultiPolygon
        polys_to_add = []
        if geom.geom_type == "Polygon":
            polys_to_add = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys_to_add = list(geom.geoms)  # iterate each polygon inside

        for poly in polys_to_add:
            # drop holes
            geom_no_holes = Polygon(poly.exterior)
            coords = list(geom_no_holes.exterior.coords)

            # ensure counterclockwise orientation
            if not Polygon(coords).exterior.is_ccw:
                coords = list(reversed(coords))

            polygons_json["features"].append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": row.drop("geometry").to_dict(),
                }
            )

    polygon_layer = ipyleaflet.GeoJSON(
        data=polygons_json,
        style_callback=style_callback_points,
        hover_style={"fillOpacity": 0.7},
    )

    # --- Polygon hover info box ---
    info_box = widgets.HTML(value="<b>Hover over a polygon</b>")

    # Precompute Shapely polygons for hover detection
    shapely_polys = []
    for feature in polygons_json["features"]:
        geom_type = feature["geometry"]["type"]
        coords = feature["geometry"]["coordinates"]
        if geom_type == "Polygon":
            shapely_polys.append((Polygon(coords[0]), feature["properties"]))
        elif geom_type == "MultiPolygon":
            for part in coords:
                shapely_polys.append((Polygon(part[0]), feature["properties"]))

    def on_mouse_move(**kwargs):
        if kwargs.get("type") != "mousemove":
            return
        coords = kwargs.get("coordinates")
        if coords is None:
            return
        lat, lon = coords
        over_any = False
        for poly, props in shapely_polys:
            if poly.covers(Point(lon, lat)):
                info_box.value = f"<b>Category:</b> {props.get('edge_vals', 'core')}"
                over_any = True
                break
        if not over_any:
            info_box.value = "<b>Hover over a polygon</b>"

    # --- Map colormap for points ---
    cmap = cm.get_cmap("PiYG")
    norm = mcolors.TwoSlopeNorm(
        vmin=df["deviation_5y"].min(), vcenter=0, vmax=df["deviation_5y"].max()
    )

    # --- Add points as CircleMarkers with click popups ---
    point_info_box = widgets.HTML(value="<b>Click a point</b>")

    point_layer = ipyleaflet.LayerGroup()
    for _, row in df.iterrows():
        x, y = row["point_geometry"].x, row["point_geometry"].y
        color = mcolors.to_hex(cmap(norm(row["deviation_5y"])))

        marker = ipyleaflet.CircleMarker(
            location=(y, x),
            radius=5,
            color=color,
            fill_color=color,
            fill_opacity=1,
            weight=1,
        )

        # still add to layer (hover will work if you later add)
        point_layer.add_layer(marker)

    # --- Map-level click handler to detect clicks near points ---
    def on_map_click(**kwargs):
        if kwargs.get("type") == "click":
            lat, lon = kwargs.get("coordinates")
            threshold = 0.05  # adjust depending on zoom/units (~5 km)
            for _, row in df.iterrows():
                px, py = row["point_geometry"].x, row["point_geometry"].y
                if abs(lat - py) < threshold and abs(lon - px) < threshold:
                    point_info_box.value = f"""
                    <b>Baseline Mortality:</b> {baseline_mortality.value:.2f}<br>
                    <b>Probability of Persistence at 1 year:</b> {row['P_1y']:.3f}<br>
                    <b>Probability of Persistence at 5 years:</b> {row['P_5y']:.3f}<br>
                    <b>Risk Decile:</b> {row['risk_decile']}
                    """
                    break
            else:
                point_info_box.value = "<b>Click a point</b>"

    center_lat = df["point_geometry"].apply(lambda p: p.y).mean()
    center_lon = df["point_geometry"].apply(lambda p: p.x).mean()

    map_widget = ipyleaflet.Map(
        center=(center_lat, center_lon),
        zoom=5,
        scroll_wheel_zoom=True,
        layers=[
            ipyleaflet.TileLayer(
                url="https://tile.gbif.org/3857/omt/{z}/{x}/{y}@1x.png?style=gbif-classic",
                name="GBIF Classic",
                attribution="GBIF",
            ),
            polygon_layer,
            point_layer,
        ],
    )

    map_widget.on_interaction(on_mouse_move)
    map_widget.on_interaction(on_map_click)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    # Return layout with download button under map
    return solara.Column(
        [
            solara.Markdown(f"## {species_name.value} Individual Persistence Map"),
            info_box,
            point_info_box,
            map_widget,
            solara.FileDownload(
                data=csv_bytes,
                filename=f"{species_name}_individual_persistence.csv",
                label="Download Individual Persistence CSV",
            ),
        ]
    )


from datetime import datetime
from ecospat.stand_alone_functions import (
    analyze_northward_shift,
    calculate_rate_of_change_first_last,
    merge_category_dataframes,
    prepare_gdf_for_rasterization,
    cat_int_mapping,
    categorize_species,
    rasterize_multiband_gdf_match,
    compute_propagule_pressure_range,
    compute_individual_persistence,
    rasterize_multiband_gdf_world,
    summarize_polygons_for_point_plot,
)


@solara.component
def MainContent():
    # State
    loading, set_loading = solara.use_state(False)
    results, set_results = solara.use_state(None)
    last_search, set_last_search = solara.use_state("")

    species = species_name.value.strip()

    # Async loader
    async def load_map_for_search(species_to_load):
        set_loading(True)
        try:
            hist_pipeline = ecospat_full.Map()
            hist_range = process_species_historical_range(
                new_map=hist_pipeline, species_name=species
            )

            # --- Heavy computation happens here ---
            classified_modern, classified_historic = analyze_species_distribution(
                species_name=species_to_load,
                record_limit=gbif_limit.value,
                end_year=datetime.fromisoformat(end_date.value).year,
                user_start_year=user_start_year.value,
                lat_min=float(lat_min.value),
                lat_max=float(lat_max.value),
                lon_min=float(lon_min.value),
                lon_max=float(lon_max.value),
                basisOfRecord=selected_basis.value if selected_basis.value else None,
            )

            if hist_range is not None:
                historic_component = HistoricalRangeMap(species_to_load, hist_range)
                historic_gdf = hist_range.copy()
            else:
                historic_component = HistoricFallbackMap(classified_historic)
                historic_gdf = classified_historic.copy()

            modern_component = ModernMap(classified_modern)

            northward_rate = analyze_northward_shift(
                gdf_hist=historic_gdf,
                gdf_new=classified_modern,
                species_name=species_to_load,
                end_year=datetime.fromisoformat(end_date.value).year,
                user_start_year=user_start_year.value,
            )

            print(northward_rate)

            north_copy = northward_rate.copy()
            north_copy = north_copy[
                north_copy["category"].isin(["leading", "core", "trailing"])
            ]
            north_copy["category"] = north_copy["category"].str.title()

            print(north_copy)

            range_cat = categorize_species(northward_rate)

            pop_change = calculate_rate_of_change_first_last(
                classified_historic,
                classified_modern,
                species_to_load,
                custom_end_year=datetime.fromisoformat(end_date.value).year,
                user_start_year=user_start_year.value,
            )

            pop_copy = pop_change.copy()
            pop_copy = pop_copy[
                pop_copy["collapsed_category"].isin(["leading", "core", "trailing"])
            ]
            pop_copy = pop_copy.rename(
                columns={
                    "collapsed_category": "Category",
                    "rate_of_change_first_last": "Rate of Change",
                    "start_time_period": "Start Years",
                    "end_time_period": "End Years",
                }
            )
            pop_copy["Category"] = pop_copy["Category"].str.title()

            merged = merge_category_dataframes(north_copy, pop_copy)
            preped = prepare_gdf_for_rasterization(classified_modern, merged)
            preped_new = cat_int_mapping(preped)

            value_columns = [
                "density",
                "northward_rate_km_per_year",
                "Rate of Change",
                "category_int",
            ]

            raster_show, gdf_transform, show_bounds = rasterize_multiband_gdf_match(
                preped_new, value_columns, resolution=resolution.value
            )

            pressure_show = compute_propagule_pressure_range(
                raster_show, D=distance_decay.value, S=directional_modifier.value
            )

            raster_component = RasterMap(
                raster_array=pressure_show,
                transform=gdf_transform,
                preped_new=preped_new,
                resolution=resolution.value,
                distance_decay=distance_decay.value,
                directional_modifier=directional_modifier.value,
                value_columns=value_columns,
            )

            points = compute_individual_persistence(
                points_gdf=classified_modern,
                raster_stack_arrays=raster_show,
                propagule_array=pressure_show,
                baseline_death=baseline_mortality.value,
                transform=gdf_transform,
            )

            individual_component = MapWithPolygonsAndPoints(points)

            # Store the fully built results in state
            set_results(
                solara.Column(
                    [
                        SpeciesSummary(
                            northward_movement=northward_rate,
                            pop_change=pop_change,
                            range_cat=range_cat,
                        ),
                        historic_component,
                        modern_component,
                        raster_component,
                        individual_component,
                    ],
                    style={"width": "70vw", "padding": "20px"},
                )
            )
        finally:
            set_loading(False)

    # Trigger search whenever button clicked (even for same species)
    if search_clicked.value and species:
        set_last_search(species)
        set_results(None)  # clear previous results for spinner
        search_clicked.value = False
        asyncio.create_task(load_map_for_search(species))

    # Render
    if loading:
        return solara.SpinnerSolara(size="100px")
    elif results:
        return results
    else:
        return solara.Markdown(
            "🔍 Enter a species name and click 'Search' to view the map."
        )


from pathlib import Path


@component
def Page():
    return VBox(
        [
            solara.Style(
                Path("/Users/alivianytko/Desktop/ecospat/solara/assets/custom.css")
            ),
            HTML(
                tag="div",
                style="""
        width: 100vw;
        max-width: 100vw;
        overflow-x: hidden;
        margin: 0;
        padding: 0;
    """,
                unsafe_innerHTML="",  # empty because we'll use children instead
            ),
            HBox([Sidebar(), MainContent()]),
        ]
    )


Page()

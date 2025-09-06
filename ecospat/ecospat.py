"""This module provides a custom Map class that extends ipyleaflet.Map to visualize range edge dynamics."""

import os
import ipyleaflet
import geopandas as gpd
import numpy as np
import json
import pandas as pd
from .references_data import REFERENCES
import requests
from io import BytesIO
from ipywidgets import widgets
from ipyleaflet import (
    GeoJSON,
    WidgetControl,
    CircleMarker,
    Popup,
    Polygon as LeafletPolygon,
)
from ipyleaflet import TileLayer
from .name_references import NAME_REFERENCES
from matplotlib import cm, colors
from matplotlib.colors import Normalize, to_hex
from matplotlib.colors import ListedColormap
from datetime import date
from IPython.display import display
from .stand_alone_functions import (
    get_species_code_if_exists,
    analyze_species_distribution,
    process_species_historical_range,
    summarize_polygons_with_points,
    create_opacity_slider_map,
    create_interactive_map,
    analyze_northward_shift,
    categorize_species,
    calculate_rate_of_change_first_last,
    save_results_as_csv,
    save_modern_gbif_csv,
    save_historic_gbif_csv,
    extract_raster_means_single_species,
    full_propagule_pressure_pipeline,
    save_raster_to_downloads_range,
    save_raster_to_downloads_global,
    summarize_polygons_for_point_plot,
    merge_category_dataframes,
    prepare_gdf_for_rasterization,
    cat_int_mapping,
    rasterize_multiband_gdf_match,
    compute_propagule_pressure_range,
    compute_individual_persistence,
    save_individual_persistence_csv,
    rasterize_multiband_gdf_world,
    analyze_species_distribution_south,
    categorize_species_south,
)


class Map(ipyleaflet.Map):
    def __init__(
        self,
        center=[42.94033923363183, -80.9033203125],
        zoom=4,
        height="600px",
        **kwargs,
    ):

        super().__init__(center=center, zoom=zoom, **kwargs)
        self.layout.height = height
        self.scroll_wheel_zoom = True
        self.github_historic_url = (
            "https://raw.githubusercontent.com/wpetry/USTreeAtlas/main/geojson"
        )
        self.github_state_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/10m_cultural"
        self.gdfs = {}
        self.references = REFERENCES
        self.master_category_colors = {
            "leading (0.99)": "#8d69b8",
            "leading (0.95)": "#519e3e",
            "leading (0.9)": "#ef8636",
            "core": "#3b75af",
            "trailing (0.1)": "#58bbcc",
            "trailing (0.05)": "#bcbd45",
            "relict (0.01 latitude)": "#84584e",
            "relict (longitude)": "#7f7f7f",
        }
        self.whole_colors = {
            "leading": "#519e3e",
            "core": "#3b75af",
            "trailing": "#bcbd45",
            "relict": "#84584e",
        }

    def show(self):
        display(self)

    def shorten_name(self, species_name):
        """
        Shorten a species name into an 8-character key.

        Takes the first four letters of the genus and first four letters of the species,
        converts to lowercase, and concatenates them. Used for indexing into REFERENCES.

        Parameters:
            species_name (str): Full scientific name of the species, e.g., 'Eucalyptus globulus'.

        Returns:
            str: 8-character lowercase key, e.g., 'eucaglob'.
        """
        return (species_name.split()[0][:4] + species_name.split()[1][:4]).lower()

    def load_historic_data(self, species_name, add_to_map=False):
        """
        Load historic range data for a species using Little maps from GitHub and add it as a layer to the map.

        Parameters:
            species_name (str): Full scientific name of the species (e.g., 'Eucalyptus globulus').
            add_to_map (bool, optional): If True, the historic range is added as a GeoJSON layer
                to the map. Defaults to False.
        """
        # Create the short name (first 4 letters of each word, lowercase)
        short_name = self.shorten_name(species_name)

        # Build the URL
        geojson_url = f"{self.github_historic_url}/{short_name}.geojson"

        try:
            # Download the GeoJSON file
            response = requests.get(geojson_url)
            response.raise_for_status()

            # Read it into a GeoDataFrame
            species_range = gpd.read_file(BytesIO(response.content))

            # Reproject to WGS84
            species_range = species_range.to_crs(epsg=4326)

            # Save it internally
            self.gdfs[short_name] = species_range

            geojson_dict = species_range.__geo_interface__

            # Only add to map if add_to_map is True
            if add_to_map:
                geojson_layer = GeoJSON(data=geojson_dict, name=species_name)
                self.add_layer(geojson_layer)

        except Exception as e:
            print(f"Error loading {geojson_url}: {e}")

    def remove_lakes(self, polygons_gdf):
        """
        Remove lakes from range polygons.

        Subtracts lake geometries from the input polygons to produce
        a cleaned GeoDataFrame representing land-only range areas. All spatial
        operations are performed in EPSG:3395 (Mercator) for consistency.

        Parameters:
            polygons_gdf (GeoDataFrame): A GeoDataFrame containing the species range
                polygons. CRS will be set to EPSG:4326 if not already defined.

        Returns:
            GeoDataFrame: A new GeoDataFrame with lake areas removed, in EPSG:3395 CRS.
                Empty geometries are removed.
        """

        lakes_url = "https://raw.githubusercontent.com/anytko/biospat_large_files/main/lakes_na.geojson"

        lakes_gdf = gpd.read_file(lakes_url)

        # Ensure valid geometries
        polygons_gdf = polygons_gdf[polygons_gdf.geometry.is_valid]
        lakes_gdf = lakes_gdf[lakes_gdf.geometry.is_valid]

        # Force both to have a CRS if missing
        if polygons_gdf.crs is None:
            polygons_gdf = polygons_gdf.set_crs("EPSG:4326")
        if lakes_gdf.crs is None:
            lakes_gdf = lakes_gdf.set_crs("EPSG:4326")

        # Reproject to EPSG:3395 for spatial ops
        polygons_proj = polygons_gdf.to_crs(epsg=3395)
        lakes_proj = lakes_gdf.to_crs(epsg=3395)

        # Perform spatial difference
        polygons_no_lakes_proj = gpd.overlay(
            polygons_proj, lakes_proj, how="difference"
        )

        # Remove empty geometries
        polygons_no_lakes_proj = polygons_no_lakes_proj[
            ~polygons_no_lakes_proj.geometry.is_empty
        ]

        # Stay in EPSG:3395 (no reprojecting back to 4326)
        return polygons_no_lakes_proj

    def load_states(self):
        """
        Load US states/provinces shapefile from GitHub and store as a GeoDataFrame.

        This method downloads the components of a shapefile (SHP, SHX, DBF) for
        administrative boundaries (states/provinces) from a GitHub repository,
        saves them temporarily, and reads them into a GeoDataFrame. The resulting
        GeoDataFrame is stored as the `states` attribute of the class.
        """
        # URLs for the shapefile components (shp, shx, dbf)
        shp_url = f"{self.github_state_url}/ne_10m_admin_1_states_provinces.shp"
        shx_url = f"{self.github_state_url}/ne_10m_admin_1_states_provinces.shx"
        dbf_url = f"{self.github_state_url}/ne_10m_admin_1_states_provinces.dbf"

        try:
            # Download all components of the shapefile
            shp_response = requests.get(shp_url)
            shx_response = requests.get(shx_url)
            dbf_response = requests.get(dbf_url)

            shp_response.raise_for_status()
            shx_response.raise_for_status()
            dbf_response.raise_for_status()

            # Create a temporary directory to store the shapefile components in memory
            with open("/tmp/ne_10m_admin_1_states_provinces.shp", "wb") as shp_file:
                shp_file.write(shp_response.content)
            with open("/tmp/ne_10m_admin_1_states_provinces.shx", "wb") as shx_file:
                shx_file.write(shx_response.content)
            with open("/tmp/ne_10m_admin_1_states_provinces.dbf", "wb") as dbf_file:
                dbf_file.write(dbf_response.content)

            # Now load the shapefile using geopandas
            state_gdf = gpd.read_file("/tmp/ne_10m_admin_1_states_provinces.shp")

            # Store it in the class as an attribute
            self.states = state_gdf

            print("Lakes data loaded successfully")

        except Exception as e:
            print(f"Error loading lakes shapefile: {e}")

    def get_historic_date(self, species_name):
        """
        Retrieve the historic reference date for a species.

        Generates an 8-letter key from the species name (first 4 letters of the
        genus and first 4 letters of the species), converts it to lowercase,
        and looks up the corresponding value in the `references` dictionary.

        Args:
            species_name (str): The full scientific name of the species.

        Returns:
            str: The historic reference date associated with the species key.
                Returns "Reference not found" if the key is not in `self.references`.
        """
        # Helper function to easily fetch the reference
        short_name = (species_name.split()[0][:4] + species_name.split()[1][:4]).lower()
        return self.references.get(short_name, "Reference not found")

    def add_basemap(self, basemap="OpenTopoMap"):
        """Add basemap to the map.

        Args:
            basemap (str, optional): Basemap name. Defaults to "OpenTopoMap".

        Available basemaps:
            - "OpenTopoMap": A topographic map.
            - "OpenStreetMap.Mapnik": A standard street map.
            - "Esri.WorldImagery": Satellite imagery.
            - "Esri.WorldTerrain": Terrain map from Esri.
            - "Esri.WorldStreetMap": Street map from Esri.
            - "CartoDB.Positron": A light, minimalist map style.
            - "CartoDB.DarkMatter": A dark-themed map style.
            - "GBIF.Classic": GBIF Classic tiles
        """

        if basemap == "GBIF.Classic":
            layer = TileLayer(
                url="https://tile.gbif.org/3857/omt/{z}/{x}/{y}@1x.png?style=gbif-classic",
                name="GBIF Classic",
                attribution="GBIF",
            )
        else:
            # fallback to ipyleaflet basemaps
            url = eval(f"basemaps.{basemap}").build_url()
            layer = TileLayer(url=url, name=basemap)

        self.add(layer)

    def add_basemap_gui(self, options=None, position="topleft"):
        """Adds a graphical user interface (GUI) for dynamically changing basemaps.

        Params:
            options (list, optional): A list of basemap options to display in the dropdown.
                Defaults to ["OpenStreetMap.Mapnik", "OpenTopoMap", "Esri.WorldImagery", "Esri.WorldTerrain", "Esri.WorldStreetMap", "CartoDB.DarkMatter", "CartoDB.Positron", "GBIF.Classic].
            position (str, optional): The position of the widget on the map. Defaults to "topright".

        Behavior:
            - A toggle button is used to show or hide the dropdown and close button.
            - The dropdown allows users to select a basemap from the provided options.
            - The close button removes the widget from the map.

        Event Handlers:
            - `on_toggle_change`: Toggles the visibility of the dropdown and close button.
            - `on_button_click`: Closes and removes the widget from the map.
            - `on_dropdown_change`: Updates the map's basemap when a new option is selected.

        Returns:
            None
        """
        if options is None:
            options = [
                "OpenStreetMap.Mapnik",
                "OpenTopoMap",
                "Esri.WorldImagery",
                "Esri.WorldTerrain",
                "Esri.WorldStreetMap",
                "CartoDB.DarkMatter",
                "CartoDB.Positron",
                "GBIF.Classic",
            ]

        toggle = widgets.ToggleButton(
            value=True,
            button_style="",
            tooltip="Click me",
            icon="map",
        )
        toggle.layout = widgets.Layout(width="38px", height="38px")

        dropdown = widgets.Dropdown(
            options=options,
            value=options[0],
            description="Basemap:",
            style={"description_width": "initial"},
        )
        dropdown.layout = widgets.Layout(width="250px", height="38px")

        button = widgets.Button(
            icon="times",
        )
        button.layout = widgets.Layout(width="38px", height="38px")

        hbox = widgets.HBox([toggle, dropdown, button])

        def on_toggle_change(change):
            if change["new"]:
                hbox.children = [toggle, dropdown, button]
            else:
                hbox.children = [toggle]

        toggle.observe(on_toggle_change, names="value")

        def on_button_click(b):
            hbox.close()
            toggle.close()
            dropdown.close()
            button.close()

        button.on_click(on_button_click)

        def on_dropdown_change(change):
            if change["new"]:
                # Remove all current basemap layers (TileLayer)
                tile_layers = [
                    layer
                    for layer in self.layers
                    if isinstance(layer, ipyleaflet.TileLayer)
                ]
                for tile_layer in tile_layers:
                    self.remove_layer(tile_layer)

                # Add new basemap
                if change["new"] == "GBIF.Classic":
                    new_tile_layer = ipyleaflet.TileLayer(
                        url="https://tile.gbif.org/3857/omt/{z}/{x}/{y}@1x.png?style=gbif-classic",
                        name="GBIF Classic",
                        attribution="GBIF",
                    )
                else:
                    url = eval(f"ipyleaflet.basemaps.{change['new']}").build_url()
                    new_tile_layer = ipyleaflet.TileLayer(url=url, name=change["new"])

                # Add as bottom layer
                self.layers = [new_tile_layer] + [
                    layer
                    for layer in self.layers
                    if not isinstance(layer, ipyleaflet.TileLayer)
                ]

        dropdown.observe(on_dropdown_change, names="value")

        control = ipyleaflet.WidgetControl(widget=hbox, position=position)
        self.add(control)

    def add_widget(self, widget, position="topright", **kwargs):
        """Add a widget to the map.

        Args:
            widget (ipywidgets.Widget): The widget to add.
            position (str, optional): Position of the widget. Defaults to "topright".
            **kwargs: Additional keyword arguments for the WidgetControl.
        """
        control = ipyleaflet.WidgetControl(widget=widget, position=position, **kwargs)
        self.add(control)

    def add_google_map(self, map_type="ROADMAP"):
        """Add Google Map to the map.

        Args:
            map_type (str, optional): Map type. Defaults to "ROADMAP".
        """
        map_types = {
            "ROADMAP": "m",
            "SATELLITE": "s",
            "HYBRID": "y",
            "TERRAIN": "p",
        }
        map_type = map_types[map_type.upper()]

        url = (
            f"https://mt1.google.com/vt/lyrs={map_type.lower()}&x={{x}}&y={{y}}&z={{z}}"
        )
        layer = ipyleaflet.TileLayer(url=url, name="Google Map")
        self.add(layer)

    def add_geojson(
        self,
        data,
        zoom_to_layer=True,
        hover_style=None,
        **kwargs,
    ):
        """Adds a GeoJSON layer to the map.

        Args:
            data (str or dict): The GeoJSON data. Can be a file path (str) or a dictionary.
            zoom_to_layer (bool, optional): Whether to zoom to the layer's bounds. Defaults to True.
            hover_style (dict, optional): Style to apply when hovering over features. Defaults to {"color": "yellow", "fillOpacity": 0.2}.
            **kwargs: Additional keyword arguments for the ipyleaflet.GeoJSON layer.

        Raises:
            ValueError: If the data type is invalid.
        """
        import geopandas as gpd

        if hover_style is None:
            hover_style = {"color": "yellow", "fillOpacity": 0.2}

        if isinstance(data, str):
            gdf = gpd.read_file(data)
            geojson = gdf.__geo_interface__
        elif isinstance(data, dict):
            geojson = data
        layer = ipyleaflet.GeoJSON(data=geojson, hover_style=hover_style, **kwargs)
        self.add_layer(layer)

        if zoom_to_layer:
            bounds = gdf.total_bounds
            self.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    def add_shp(self, data, **kwargs):
        """Adds a shapefile to the map.

        Args:
            data (str): The file path to the shapefile.
            **kwargs: Additional keyword arguments for the GeoJSON layer.
        """
        import geopandas as gpd

        gdf = gpd.read_file(data)
        gdf = gdf.to_crs(epsg=4326)
        geojson = gdf.__geo_interface__
        self.add_geojson(geojson, **kwargs)

    def add_shp_from_url(self, url, **kwargs):
        """Adds a shapefile from a URL to the map.
        Adds a shapefile from a URL to the map.

        This function downloads the shapefile components (.shp, .shx, .dbf) from the specified URL, stores them
        in a temporary directory, reads the shapefile using Geopandas, converts it to GeoJSON format, and
        then adds it to the map. If the shapefile's coordinate reference system (CRS) is not set, it assumes
        the CRS to be EPSG:4326 (WGS84).

        Args:
            url (str): The URL pointing to the shapefile's location. The URL should be a raw GitHub link to
                    the shapefile components (e.g., ".shp", ".shx", ".dbf").
            **kwargs: Additional keyword arguments to pass to the `add_geojson` method for styling and
                    configuring the GeoJSON layer on the map.
        """
        try:
            base_url = url.replace("github.com", "raw.githubusercontent.com").replace(
                "blob/", ""
            )
            shp_url = base_url + ".shp"
            shx_url = base_url + ".shx"
            dbf_url = base_url + ".dbf"

            temp_dir = tempfile.mkdtemp()

            shp_file = requests.get(shp_url).content
            shx_file = requests.get(shx_url).content
            dbf_file = requests.get(dbf_url).content

            with open(os.path.join(temp_dir, "data.shp"), "wb") as f:
                f.write(shp_file)
            with open(os.path.join(temp_dir, "data.shx"), "wb") as f:
                f.write(shx_file)
            with open(os.path.join(temp_dir, "data.dbf"), "wb") as f:
                f.write(dbf_file)

            gdf = gpd.read_file(os.path.join(temp_dir, "data.shp"))

            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)

            geojson = gdf.__geo_interface__

            self.add_geojson(geojson, **kwargs)

            shutil.rmtree(temp_dir)

        except Exception:
            pass

    def add_layer_control(self):
        """Adds a layer control widget to the map."""
        control = ipyleaflet.LayersControl(position="topright")
        self.add_control(control)

    def add_range_polygons(self, summarized_poly):
        """
        Add range polygons from a GeoDataFrame to an ipyleaflet map with interactive hover tooltips.

        This method converts a GeoDataFrame into a GeoJSON layer, applies custom styling,
        and attaches event handlers to display tooltips when polygons are hovered over.
        Tooltips are displayed in a widget positioned at the bottom-left of the map.

        Args:
            summarized_poly (geopandas.GeoDataFrame): A GeoDataFrame containing the polygons
                to be added to the map. Must have valid geometries.

        Returns:
            None
        """

        # Create the tooltip as an independent widget
        tooltip = widgets.HTML(value="")  # Start with an empty value
        tooltip.layout.margin = "10px"
        tooltip.layout.visibility = "hidden"
        tooltip.layout.width = "auto"
        tooltip.layout.height = "auto"

        tooltip.layout.display = "flex"  # Make it a flex container to enable alignment
        tooltip.layout.align_items = "center"  # Center vertically
        tooltip.layout.justify_content = "center"  # Center horizontally
        tooltip.style.text_align = "center"

        # Widget control for the tooltip, positioned at the bottom right of the map
        hover_control = WidgetControl(widget=tooltip, position="bottomleft")

        # Convert GeoDataFrame to GeoJSON format
        geojson_data = summarized_poly.to_json()

        # Load the GeoJSON string into a Python dictionary
        geojson_dict = json.loads(geojson_data)

        # Create GeoJSON layer for ipyleaflet
        geojson_layer = GeoJSON(
            data=geojson_dict,  # Pass the Python dictionary (not a string)
            style_callback=self.style_callback,
        )

        # Attach hover and mouseout event handlers
        geojson_layer.on_hover(self.handle_hover(tooltip, hover_control))
        geojson_layer.on_msg(self.handle_mouseout(tooltip, hover_control))

        # Add the GeoJSON layer to the map (now directly using self)
        self.add_layer(geojson_layer)

    def style_callback(self, feature):
        """
        Determine the visual style of GeoJSON range polygons based on their edge categories.

        This function is used as a callback for ipyleaflet GeoJSON layers to assign
        fill color, border color, line weight, and opacity according to the range edge
        'category' property of each polygon.

        Args:
            feature (dict): A GeoJSON feature dictionary. Should contain a
                'properties' key with a 'category' field.

        Returns:
            dict: A style dictionary with keys:
                - 'fillColor' (str): Fill color of the polygon.
                - 'color' (str): Border color of the polygon.
                - 'weight' (int): Border line width.
                - 'fillOpacity' (float): Opacity of the fill.
        """
        category = feature["properties"].get("category", "core")
        color = self.master_category_colors.get(category, "#3b75af")  # Fallback color
        return {"fillColor": color, "color": color, "weight": 2, "fillOpacity": 0.7}

    def handle_hover(self, tooltip, hover_control):
        """
        Create a hover event handler that displays a tooltip for a GeoJSON feature.

        This method returns a function that can be attached to a GeoJSON layer's
        `on_hover` event in ipyleaflet. When the mouse hovers over a feature, the
        tooltip is updated with the feature's category and made visible on the map.

        Args:
            tooltip (ipywidgets.HTML): The HTML widget used to display feature information.
            hover_control (ipyleaflet.WidgetControl): The map control containing the tooltip.

        Returns:
            function: An event handler function that takes a `feature` dictionary
                    and updates the tooltip when the mouse hovers over it.
        """

        def inner(feature, **kwargs):
            # Update the tooltip with feature info
            category_value = feature["properties"].get("category", "N/A").title()
            tooltip.value = f"<b>Category:</b> {category_value}"
            tooltip.layout.visibility = "visible"

            # Show the tooltip control
            self.add_control(hover_control)

        return inner

    def handle_hover_edge(self, tooltip, hover_control):
        """
        Create a hover event handler that displays a tooltip for a GeoJSON feature.

        This method returns a function that can be attached to a GeoJSON layer's
        `on_hover` event in ipyleaflet. When the mouse hovers over a feature, the
        tooltip is updated with the feature's category and made visible on the map.

        Args:
            tooltip (ipywidgets.HTML): The HTML widget used to display feature information.
            hover_control (ipyleaflet.WidgetControl): The map control containing the tooltip.

        Returns:
            function: An event handler function that takes a `feature` dictionary
                    and updates the tooltip when the mouse hovers over it.
        """

        def inner(feature, **kwargs):
            # Update the tooltip with feature info
            category_value = feature["properties"].get("edge_vals", "N/A").title()
            tooltip.value = f"<b>Category:</b> {category_value}"
            tooltip.layout.visibility = "visible"

            # Show the tooltip control
            self.add_control(hover_control)

        return inner

    def handle_mouseout(self, tooltip, hover_control):
        """
        Create a mouseout event handler that hides a tooltip for a GeoJSON feature.

        This method returns a function that can be attached to a GeoJSON layer's
        `on_msg` event in ipyleaflet. When the mouse moves out of a feature, the
        tooltip is cleared and hidden from the map.

        Args:
            tooltip (ipywidgets.HTML): The HTML widget used to display feature information.
            hover_control (ipyleaflet.WidgetControl): The map control containing the tooltip.

        Returns:
            function: An event handler function that takes event parameters (`_`, `content`, `buffers`)
                    and hides the tooltip when a "mouseout" event is detected.
        """

        def inner(_, content, buffers):
            event_type = content.get("type", "")
            if event_type == "mouseout":
                tooltip.value = ""
                tooltip.layout.visibility = "hidden"
                self.remove_control(hover_control)

        return inner

    def add_raster(self, filepath, **kwargs):
        """
        Add a raster file as a tile layer to the map using a local tile server.

        This method creates a `TileClient` for the given raster file and generates
        a Leaflet-compatible tile layer. The layer is added to the map, and the map
        view is updated to center on the raster with a default zoom.

        Args:
            filepath (str): Path to the raster file (e.g., GeoTIFF) to be added.
            **kwargs: Additional keyword arguments passed to `get_leaflet_tile_layer`
                    to control tile layer appearance and behavior (e.g., colormap, opacity).

        Returns:
            None
        """
        from localtileserver import TileClient, get_leaflet_tile_layer

        client = TileClient(filepath)
        tile_layer = get_leaflet_tile_layer(client, **kwargs)

        self.add(tile_layer)
        self.center = client.center()
        self.zoom = client.default_zoom

    def style_callback_point(self, feature):
        """
        Determine the visual style of GeoJSON range polygons based on their edge values.

        This function is used as a callback for ipyleaflet GeoJSON layers to assign
        fill color, border color, line weight, and opacity according to the range edge
        'edge_vals' property of each polygon.

        Args:
            feature (dict): A GeoJSON feature dictionary. Should contain a
                'properties' key with a 'edge_vals' field.

        Returns:
            dict: A style dictionary with keys:
                - 'fillColor' (str): Fill color of the polygon.
                - 'color' (str): Border color of the polygon.
                - 'weight' (int): Border line width.
                - 'fillOpacity' (float): Opacity of the fill.
        """
        edge_val = feature["properties"].get("edge_vals", "core")
        color = self.whole_colors.get(edge_val, "#3b75af")  # fallback
        return {
            "fillColor": color,  # polygon interior
            "color": color,  # polygon border
            "weight": 2,  # border width
            "opacity": 0.7,  # border transparency
            "fillOpacity": 0.4,  # interior transparency
        }

    def add_point_data(self, summarized_poly, use_gradient=False):
        """
        Add points and polygons from a GeoDataFrame to an ipyleaflet map, with optional gradient coloring.

        This method visualizes spatial data by adding both polygons and point markers to the map.
        Polygons are summarized and styled using a callback function. Points can optionally be
        colored based on deviation from an expected baseline (P_5y) using a pink-yellow-green gradient.

        Args:
            summarized_poly (GeoDataFrame): A GeoDataFrame containing polygon geometries and
                associated point data. Expected columns include:
                - 'point_geometry': shapely Point objects for each data point
                - 'P_1y' and 'P_5y': probabilities for 1-year and 5-year events
                - 'baseline_death': baseline probability used to calculate deviation
                - 'risk_decile': risk category for display in the popup
            use_gradient (bool, optional): If True, points are colored according to their deviation
                from expected baseline using a pink-yellow-green gradient. Defaults to False.

        Returns:
            None

        Notes:
            - Hovering over polygons displays a tooltip with category information.
            - Each point is displayed as a CircleMarker with a popup showing its P_1y, P_5y,
            and risk decile.
            - When `use_gradient=True`, deviations are normalized and clipped to the 5th-95th
            percentile range for better visual contrast.
        """
        tooltip = widgets.HTML(value="")
        tooltip.layout.margin = "10px"
        tooltip.layout.visibility = "hidden"
        tooltip.layout.width = "auto"
        tooltip.layout.height = "auto"
        tooltip.layout.display = "flex"
        tooltip.layout.align_items = "center"
        tooltip.layout.justify_content = "center"
        tooltip.style.text_align = "center"

        hover_control = WidgetControl(widget=tooltip, position="bottomleft")

        # --- Polygons ---
        polygon_copy = summarized_poly.copy()
        summary_polygons = summarize_polygons_for_point_plot(polygon_copy)
        # polygons_only = summarized_poly.drop(columns=['point_geometry'])
        geojson_data = summary_polygons.to_json()
        geojson_dict = json.loads(geojson_data)

        polygon_layer = GeoJSON(
            data=geojson_dict, style_callback=self.style_callback_point
        )
        polygon_layer.on_hover(self.handle_hover_edge(tooltip, hover_control))
        polygon_layer.on_msg(self.handle_mouseout(tooltip, hover_control))

        self.add_layer(polygon_layer)

        def lighten_cmap(cmap, factor=0.5):
            n = cmap.N
            colors_array = cmap(np.linspace(0, 1, n))
            white = np.ones_like(colors_array)
            new_colors = colors_array + (white - colors_array) * factor
            return ListedColormap(new_colors)

        # Create a lighter Spectral colormap
        # light_spectral = lighten_cmap(cm.PiYG, factor=0.3)

        # --- Points ---
        if use_gradient:

            expected_1y = 1 - summarized_poly["baseline_death"]
            summarized_poly["deviation_1y"] = summarized_poly["P_1y"] - expected_1y

            expected_5y = (1 - summarized_poly["baseline_death"]) ** 5
            summarized_poly["deviation_5y"] = summarized_poly["P_5y"] - expected_5y

            # Use 1y deviation for coloring (or max deviation if you prefer)
            # deviations = summarized_poly['deviation_1y'].values
            # max_abs_dev = np.max(np.abs(deviations))
            # norm = colors.Normalize(vmin=-max_abs_dev, vmax=max_abs_dev)

            # cmap = cm.get_cmap('PiYG')  # red = low, blue = high
            # cmap = light_spectral

            # Assign a color to each point
            # summarized_poly['color'] = [cmap(norm(x)) for x in deviations]

            deviations = summarized_poly[
                "deviation_5y"
            ].values  # or however you're storing them

            # Dynamic range but clipped to percentiles so you see stronger contrast
            vmin, vmax = np.percentile(deviations, [5, 95])  # clip extremes
            vcenter = 0

            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            cmap = cm.get_cmap("PiYG")

            summarized_poly["color"] = [
                colors.to_hex(cmap(norm(x))) for x in deviations
            ]

        for idx, row in summarized_poly.iterrows():
            if row["point_geometry"] is not None:
                coords = (row["point_geometry"].y, row["point_geometry"].x)  # lat, lon

                # Default marker properties
                color = "#000000"
                radius = 4

                if use_gradient:
                    # Use deviation_1y or deviation_5y (choose whichever you prefer)
                    dev = row["deviation_1y"]  # could also average 1y & 5y
                    color = to_hex(cmap(norm(dev)))

                # Popup HTML
                point_info = f"""
                <b>P_1y:</b> {row['P_1y']:.3f}<br>
                <b>P_5y:</b> {row['P_5y']:.3f}<br>
                <b>Risk Decile:</b> {row['risk_decile']}
                """

                marker = CircleMarker(
                    location=coords,
                    fill_color=color,
                    color=color,
                    radius=radius,
                    fill_opacity=1.0,
                )
                popup = Popup(
                    location=coords,
                    child=widgets.HTML(value=point_info),
                    close_button=True,
                    auto_close=False,
                    close_on_escape_key=False,
                )

                marker.popup = popup
                self.add_layer(marker)

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

    def add_control_panel(self):
        """
        Add an interactive control panel to the application interface.

        The panel allows users to display different range layers and data types,
        including historic range, modern range, propagule pressure raster, individual
        persistence, and summary statistics. It also supports saving options and triggers
        map rendering updates when toggles are changed.

        This function manages the layout, interactivity, and event handling for these
        controls, integrating them with the map visualization system.

        Args:
            None

        Returns:
            None
        """
        # Toggle button like in basemap GUI
        toggle = widgets.ToggleButton(
            value=True,
            button_style="",
            tooltip="Open/Close options panel",
            icon="gear",
            layout=widgets.Layout(
                width="38px",
                height="38px",
                display="flex",
                align_items="center",
                justify_content="center",
                padding="0px 0px 0px 0px",  # Top, Right, Bottom, Left — slight left shift
            ),
            style={"button_color": "white"},
        )

        output_toggle = widgets.ToggleButton(
            value=False,
            button_style="",
            tooltip="Show/Hide output panel",
            icon="eye",
            layout=widgets.Layout(
                width="38px",
                height="38px",
                display="flex",
                align_items="center",
                justify_content="center",
                padding="0px 0px 0px 0px",
            ),
            style={"button_color": "white"},
        )

        end_date_picker = widgets.DatePicker(
            description="End Date:", value=date.today()
        )

        gbif_limit_input = widgets.BoundedIntText(
            value=500,
            min=10,
            max=10000,
            step=10,
            description="GBIF Limit:",
            tooltip="Maximum number of GBIF records to use",
        )

        generate_3d_checkbox = widgets.Checkbox(
            value=False, description="Generate 3D population density map", indent=False
        )

        save_map_checkbox = widgets.Checkbox(
            value=False, description="Save 3D Population Density Map", indent=False
        )

        save_results_checkbox = widgets.Checkbox(
            value=False, description="Save Movement Results", indent=False
        )

        save_modern_label = widgets.Label("Save Selection:")

        save_modern_gbif_checkbox = widgets.Checkbox(
            value=False, description="Save Modern GBIF Data", indent=False
        )

        baseline_mortality = widgets.BoundedFloatText(
            value=0.1,
            min=0,
            max=1,
            step=0.1,
            description="Yearly Death:",
            tooltip="Baseline mortality of species",
        )

        # Stack them vertically
        save_modern_box = widgets.VBox([save_modern_label, save_modern_gbif_checkbox])

        save_historic_gbif_checkbox = widgets.Checkbox(
            value=False, description="Save Historic GBIF Data", indent=False
        )

        save_individual_persistence_checkbox = widgets.Checkbox(
            value=False, description="Save Individual Persistence Data", indent=False
        )

        save_raster_radio = widgets.RadioButtons(
            options=["Yes", "No"],
            description="Save Predicted Persistence Raster",
            value="No",
        )

        save_range_checkbox = widgets.Checkbox(
            description="Save to range extent", value=False
        )
        save_global_checkbox = widgets.Checkbox(
            description="Save to global extent", value=False
        )
        resolution_input = widgets.FloatText(value=0.1666667, description="Resolution")

        conditional_raster_box = widgets.VBox(
            children=[save_range_checkbox, save_global_checkbox, resolution_input]
        )
        conditional_raster_box.layout.display = "none"  # hidden initially

        toggle_buttons = widgets.HBox([toggle, output_toggle])

        # save_map_box = widgets.HBox([widgets.Label("    "), save_map_checkbox])

        process_button = widgets.Button(
            description="Run Analysis", button_style="success", icon="play"
        )

        # Radio buttons for map type (removed 'Split' option)
        map_type_radio = widgets.RadioButtons(
            options=["Modern", "Historic", "Individual Persistence"],
            description="Map Type:",
            disabled=False,
        )
        # Create output widget that will be shown in the bottom-right corner
        collapsible_output_area = widgets.Output()
        collapsible_output_area.layout.display = "none"  # Initially hidden
        collapsible_output_area.layout.padding = "0px 20px 0px 0px"

        species_input = widgets.Dropdown(
            options=sorted(NAME_REFERENCES.keys()),
            description="Species:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Add gbif_limit_input to controls_box
        controls_box = widgets.VBox(
            [
                species_input,
                gbif_limit_input,  # <- inserted here
                end_date_picker,
                baseline_mortality,
                map_type_radio,
                generate_3d_checkbox,
                save_modern_box,
                save_historic_gbif_checkbox,
                save_individual_persistence_checkbox,
                save_map_checkbox,
                save_results_checkbox,
                save_raster_radio,
                conditional_raster_box,
                process_button,
            ]
        )
        scrollable_controls = widgets.Box(
            [controls_box],
            layout=widgets.Layout(
                overflow_y="auto",  # vertical scroll
                height="600px",  # fix height to match map
                width="320px",  # fixed width
                border="1px solid lightgray",
                padding="5px",
                margin="0px",
            ),
        )

        # Put them side by side
        hbox = widgets.HBox([toggle, scrollable_controls])

        # Function to toggle the collapsible output widget
        def toggle_output_visibility(change):
            if change["new"]:
                collapsible_output_area.layout.display = "block"
            else:
                collapsible_output_area.layout.display = "none"

        def toggle_save_options(change):
            if change["new"] == "Yes":
                conditional_raster_box.layout.display = "flex"
            else:
                conditional_raster_box.layout.display = "none"

        # Function to toggle the control panel visibility
        def toggle_control_panel_visibility(change):
            if change["new"]:
                scrollable_controls.layout.display = "block"
            else:
                scrollable_controls.layout.display = "none"

        save_raster_radio.observe(toggle_save_options, names="value")

        toggle.observe(toggle_control_panel_visibility, names="value")
        output_toggle.observe(toggle_output_visibility, names="value")

        def on_run_button_clicked(b):
            species = species_input.value
            record_limit = gbif_limit_input.value
            end_date = end_date_picker.value
            end_year_int = end_date.year
            use_3d = generate_3d_checkbox.value
            save_map = save_map_checkbox.value
            save_results = save_results_checkbox.value
            resolution = resolution_input.value
            baseline_val = baseline_mortality.value

            collapsible_output_area.clear_output()

            collapsible_output_area.layout.display = "block"

            # Check if species exists in reference data
            species_code = get_species_code_if_exists(species)
            if species_code:
                print(f"Species {species} exists with code {species_code}")

                # Run the analysis
                classified_modern, classified_historic = analyze_species_distribution(
                    species, record_limit=record_limit, end_year=end_year_int
                )

                # We are going to use the existing map_widget for results display
                map_widget = self  # Assuming self is the existing map_widget

                # Process the historical range if species exists
                hist_range = process_species_historical_range(
                    new_map=map_widget, species_name=species
                )

                if hist_range is None or (
                    hasattr(hist_range, "empty") and hist_range.empty
                ):
                    hist_range = classified_historic

                northward_rate_df = analyze_northward_shift(
                    gdf_hist=hist_range,
                    gdf_new=classified_modern,
                    species_name=species,
                )
                northward_rate_df = northward_rate_df[
                    northward_rate_df["category"].isin(["leading", "core", "trailing"])
                ]
                northward_rate_df["category"] = northward_rate_df[
                    "category"
                ].str.title()

                print("Northward Rate of Change:")
                print(
                    northward_rate_df[["category", "northward_rate_km_per_year"]]
                    .rename(
                        columns={
                            "category": "Category",
                            "northward_rate_km_per_year": "Northward Movement (km/y)",
                        }
                    )
                    .to_string(index=False)
                )

                final_result = categorize_species(northward_rate_df)
                pattern_value = final_result["category"].iloc[0].title()
                print(f"Range movement pattern: {pattern_value}")

                change = calculate_rate_of_change_first_last(
                    classified_historic,
                    classified_modern,
                    species,
                    custom_end_year=end_year_int,
                )
                change = change[
                    change["collapsed_category"].isin(["leading", "core", "trailing"])
                ]
                change = change.rename(
                    columns={
                        "collapsed_category": "Category",
                        "rate_of_change_first_last": "Rate of Change",
                        "start_time_period": "Start Years",
                        "end_time_period": "End Years",
                    }
                )
                change["Category"] = change["Category"].str.title()

                merged = merge_category_dataframes(northward_rate_df, change)

                preped_gdf = prepare_gdf_for_rasterization(classified_modern, merged)
                preped_gdf_new = cat_int_mapping(preped_gdf)

                value_columns = [
                    "density",
                    "northward_rate_km_per_year",
                    "Rate of Change",
                    "category_int",
                ]
                raster_show, transform, show_bounds = rasterize_multiband_gdf_match(
                    preped_gdf_new, value_columns
                )
                pressure_show = compute_propagule_pressure_range(raster_show)

                points = compute_individual_persistence(
                    points_gdf=classified_modern,
                    raster_stack_arrays=raster_show,
                    propagule_array=pressure_show,
                    baseline_death=baseline_val,
                    transform=transform,
                )

                with collapsible_output_area:
                    # print(f"Running analysis for {species} until {end_date}")
                    # if use_3d:
                    # print("3D map will be generated.")
                    # if save_map:
                    # print("Map will be saved locally.")
                    # else:
                    # print("Standard map generation.")

                    # Map Type Based Actions (removed 'Split' case)
                    if map_type_radio.value == "Modern":
                        summarized_poly = summarize_polygons_with_points(
                            classified_modern
                        )
                        map_widget.add_range_polygons(summarized_poly)

                    elif map_type_radio.value == "Historic":
                        map_widget.add_range_polygons(hist_range)
                    elif map_type_radio.value == "Individual Persistence":
                        map_widget.add_point_data(points, use_gradient=True)

                    # Population map handling
                    if generate_3d_checkbox.value:
                        if map_type_radio.value == "Modern":
                            create_interactive_map(classified_modern, if_save=save_map)
                        elif map_type_radio.value == "Historic":
                            create_interactive_map(
                                classified_historic, if_save=save_map
                            )

                    # Display the results
                    print("Population Density:")
                    print(change.to_string(index=False))

                    mean_clim, clim_data = extract_raster_means_single_species(
                        classified_modern, species
                    )

                    if save_results_checkbox.value:
                        save_results_as_csv(
                            northward_rate_df,
                            final_result,
                            change,
                            mean_clim,
                            clim_data,
                            species,
                        )

                    if save_modern_gbif_checkbox.value:
                        save_modern_gbif_csv(classified_modern, species)

                    if save_historic_gbif_checkbox.value:
                        save_historic_gbif_csv(classified_historic, species)

                    if save_individual_persistence_checkbox.value:
                        save_individual_persistence_csv(points, species)

                    if save_raster_radio.value == "Yes":
                        if save_range_checkbox.value:
                            # Save the raster for the range extent
                            save_raster_to_downloads_range(
                                raster_show, show_bounds, species
                            )

                        # If you also need to save propagule pressure:
                        if save_global_checkbox.value:
                            global_show, global_transform, global_bounds = (
                                rasterize_multiband_gdf_world(
                                    preped_gdf_new, value_columns
                                )
                            )
                            global_show = compute_propagule_pressure_range(global_show)
                            save_raster_to_downloads_global(
                                global_show, global_bounds, species
                            )

                    if save_raster_radio.value == "Yes":
                        # Call the pipeline function once
                        full_show, full_save, show_bounds, save_bounds = (
                            full_propagule_pressure_pipeline(
                                classified_modern,
                                northward_rate_df,
                                change,
                                resolution=resolution,
                            )
                        )

                        if save_range_checkbox.value:
                            # Save the raster for the range extent
                            save_raster_to_downloads_range(
                                full_show, show_bounds, species
                            )

                        elif save_global_checkbox.value:
                            # Save the raster for the global extent
                            save_raster_to_downloads_global(
                                full_save, save_bounds, species
                            )

            else:
                collapsible_output_area.clear_output()
                collapsible_output_area.layout.display = "block"
                with collapsible_output_area:
                    print(
                        f"Species '{species}' not available in the reference data. Try another species."
                    )
            scrollable_controls.layout.display = "none"

        process_button.on_click(on_run_button_clicked)

        control = ipyleaflet.WidgetControl(widget=hbox, position="topright")
        self.add(control)

        # Add the collapsible output widget to the map in the bottom-left corner (updated position)
        output_box = widgets.HBox([output_toggle, collapsible_output_area])
        output_control = ipyleaflet.WidgetControl(
            widget=output_box, position="bottomright"
        )
        self.add(output_control)

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

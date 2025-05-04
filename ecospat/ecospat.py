"""This module provides a custom Map class that extends ipyleaflet.Map to visualize range edge dynamics."""

import os
import ipyleaflet
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry import mapping, box
from shapely.wkt import loads
from pygbif import species, occurrences
import json
import pandas as pd
from scipy.stats import linregress
from scipy.spatial.distance import pdist
from shapely.geometry import box
from .references_data import REFERENCES
import requests
from io import BytesIO
from scipy.spatial.distance import cdist
from ipywidgets import widgets
from ipyleaflet import Map, GeoJSON, WidgetControl


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
)


class HistoricalMap(ipyleaflet.Map):
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

    def show(self):
        display(self)

    def shorten_name(self, species_name):
        """Helper to shorten the species name."""
        return (species_name.split()[0][:4] + species_name.split()[1][:4]).lower()

    def load_historic_data(self, species_name):
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

            # No prints, no plots - clean loading!

        except Exception as e:
            print(f"Error loading {geojson_url}: {e}")

    def remove_lakes(self, polygons_gdf):
        """
        Removes lakes from range polygons and returns the resulting GeoDataFrame.
        All operations in EPSG:3395 for consistency.
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
        """

        url = eval(f"ipyleaflet.basemaps.{basemap}").build_url()
        layer = ipyleaflet.TileLayer(url=url, name=basemap)
        self.add(layer)

    def add_basemap_gui(self, options=None, position="topleft"):
        """Adds a graphical user interface (GUI) for dynamically changing basemaps.

        Params:
            options (list, optional): A list of basemap options to display in the dropdown.
                Defaults to ["OpenStreetMap.Mapnik", "OpenTopoMap", "Esri.WorldImagery", "Esri.WorldTerrain", "Esri.WorldStreetMap", "CartoDB.DarkMatter", "CartoDB.Positron"].
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
                url = eval(f"ipyleaflet.basemaps.{change['new']}").build_url()
                new_tile_layer = ipyleaflet.TileLayer(url=url, name=change["new"])

                # Add the new basemap as the bottom layer (first in the list)
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

        species_input = widgets.Text(
            description="Species:", placeholder="Enter species name"
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
            value=False,
            description="Generate 3D population density map",
            layout=widgets.Layout(
                margin="0px 0px 0px 0px", padding="0px 20px 0px 0px", width="auto"
            ),
        )

        save_map_checkbox = widgets.Checkbox(
            value=False,
            description="Save this map to your local device",
            layout=widgets.Layout(
                margin="0px 0px 0px 0px", padding="0px 20px 0px 0px", width="auto"
            ),
        )

        toggle_buttons = widgets.HBox([toggle, output_toggle])

        save_map_box = widgets.HBox([widgets.Label("    "), save_map_checkbox])

        process_button = widgets.Button(
            description="Run Analysis", button_style="success", icon="play"
        )

        # Radio buttons for map type (removed 'Split' option)
        map_type_radio = widgets.RadioButtons(
            options=["Modern", "Historic"], description="Map Type:", disabled=False
        )

        # Create output widget that will be shown in the bottom-right corner
        collapsible_output_area = widgets.Output()
        collapsible_output_area.layout.display = "none"  # Initially hidden
        collapsible_output_area.layout.padding = "0px 20px 0px 0px"

        # Add gbif_limit_input to controls_box
        controls_box = widgets.VBox(
            [
                species_input,
                gbif_limit_input,  # <- inserted here
                end_date_picker,
                map_type_radio,
                generate_3d_checkbox,
                save_map_box,
                process_button,
            ]
        )

        hbox = widgets.HBox([toggle, controls_box])

        # Function to toggle the collapsible output widget
        def toggle_output_visibility(change):
            if change["new"]:
                collapsible_output_area.layout.display = "block"
            else:
                collapsible_output_area.layout.display = "none"

        # Function to toggle the control panel visibility
        def toggle_control_panel_visibility(change):
            if change["new"]:
                controls_box.layout.display = "block"
            else:
                controls_box.layout.display = "none"

        toggle.observe(toggle_control_panel_visibility, names="value")
        output_toggle.observe(toggle_output_visibility, names="value")

        def on_run_button_clicked(b):
            species = species_input.value
            record_limit = gbif_limit_input.value
            end_date = end_date_picker.value
            end_year_int = end_date.year
            use_3d = generate_3d_checkbox.value
            save_map = save_map_checkbox.value

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
                        map_widget.add_gbif_polygons(summarized_poly)

                    elif map_type_radio.value == "Historic":
                        map_widget.add_gbif_polygons(hist_range)

                    # Population map handling
                    if generate_3d_checkbox.value:
                        if map_type_radio.value == "Modern":
                            create_interactive_map(classified_modern, if_save=save_map)
                        elif map_type_radio.value == "Historic":
                            create_interactive_map(
                                classified_historic, if_save=save_map
                            )

                    # Display the analysis results for northward change
                    northward_rate_df = analyze_northward_shift(
                        gdf_hist=hist_range,
                        gdf_new=classified_modern,
                        species_name=species,
                    )
                    northward_rate_df = northward_rate_df[
                        northward_rate_df["category"].isin(
                            ["leading", "core", "trailing"]
                        )
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

                    # Display the analysis results for range movement
                    final_result = categorize_species(northward_rate_df)
                    pattern_value = final_result["category"].iloc[0].title()
                    print(f"Range movement pattern: {pattern_value}")

                    # Display the analysis results for rate of change
                    change = calculate_rate_of_change_first_last(
                        classified_historic,
                        classified_modern,
                        species,
                        custom_end_year=end_year_int,
                    )
                    change = change[
                        change["collapsed_category"].isin(
                            ["leading", "core", "trailing"]
                        )
                    ]
                    change = change.rename(
                        columns={
                            "collapsed_category": "Category",
                            "rate_of_change_first_last": "Rate of Change",
                            "start_time_period": "Start Years",
                            "end_time_period": "End Years",
                        }
                    )

                    # Convert 'Category' column to title case
                    change["Category"] = change["Category"].str.title()

                    # Display the results
                    print("Population Density:")
                    print(change.to_string(index=False))

            else:
                collapsible_output_area.clear_output()
                collapsible_output_area.layout.display = "block"
                with collapsible_output_area:
                    print(
                        f"Species '{species}' not available in the reference data. Try another species."
                    )

        process_button.on_click(on_run_button_clicked)

        control = ipyleaflet.WidgetControl(widget=hbox, position="topright")
        self.add(control)

        # Add the collapsible output widget to the map in the bottom-left corner (updated position)
        output_box = widgets.HBox([output_toggle, collapsible_output_area])
        output_control = ipyleaflet.WidgetControl(
            widget=output_box, position="bottomright"
        )
        self.add(output_control)

    def add_gbif_polygons(self, summarized_poly):
        """Add polygons from a GeoDataFrame to ipyleaflet, with hover tooltips."""

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
        """Style function that applies color based on 'category'."""
        category = feature["properties"].get("category", "core")
        color = self.master_category_colors.get(category, "#3b75af")  # Fallback color
        return {"fillColor": color, "color": color, "weight": 2, "fillOpacity": 0.7}

    def handle_hover(self, tooltip, hover_control):
        """Handle hover event to show tooltip."""

        def inner(feature, **kwargs):
            # Update the tooltip with feature info
            category_value = feature["properties"].get("category", "N/A").title()
            tooltip.value = f"<b>Category:</b> {category_value}"
            tooltip.layout.visibility = "visible"

            # Show the tooltip control
            self.add_control(hover_control)

        return inner

    def handle_mouseout(self, tooltip, hover_control):
        """Handle mouseout event to hide tooltip."""

        def inner(_, content, buffers):
            event_type = content.get("type", "")
            if event_type == "mouseout":
                tooltip.value = ""
                tooltip.layout.visibility = "hidden"
                self.remove_control(hover_control)

        return inner

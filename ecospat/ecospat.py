"""This module provides a custom Map class that extends ipyleaflet.Map to visualize range edge dynamics."""

import os
import ipyleaflet
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
import matplotlib.pyplot as plt
import random
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry import mapping, box
from shapely.wkt import loads
from pygbif import species, occurrences
import json
import pandas as pd
from scipy.stats import linregress
from scipy.spatial.distance import pdist
from shapely.geometry import box
from geopy.distance import geodesic
from rasterio.transform import from_origin
from ipyleaflet import Marker, Popup
import rasterio
import ipywidgets as widgets
from .references_data import REFERENCES
import requests
from io import BytesIO
from scipy.spatial.distance import cdist
from ipyleaflet import GeoData


class HistoricalMap(ipyleaflet.Map):
    def __init__(self, center=[20, 0], zoom=2, height="600px", **kwargs):

        super().__init__(center=center, zoom=zoom, **kwargs)
        self.layout.height = height
        self.scroll_wheel_zoom = True
        self.github_historic_url = (
            "https://raw.githubusercontent.com/wpetry/USTreeAtlas/main/geojson"
        )
        self.github_state_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/10m_cultural"
        self.gdfs = {}
        self.references = REFERENCES

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

    def add_basemap_gui(self, options=None, position="topright"):
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
                self.layers = self.layers[:-2]
                self.add_basemap(change["new"])

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


class GBIF_Map(HistoricalMap):
    def __init__(self, center=[20, 0], zoom=2, height="600px", **kwargs):
        super().__init__(center=center, zoom=zoom, **kwargs)
        self.layout.height = height
        self.scroll_wheel_zoom = True

    def add_gbif_polygons(self, polygons_gdf):
        """Add polygons from a GeoDataFrame to the ipyleaflet map."""
        if not isinstance(polygons_gdf, gpd.GeoDataFrame):
            raise TypeError("Input must be a GeoDataFrame.")
        if "geometry" not in polygons_gdf:
            raise ValueError("GeoDataFrame must have a 'geometry' column.")

        gbif_polygons = GeoData(
            geo_dataframe=polygons_gdf,
            style={"color": "blue", "opacity": 1, "weight": 2, "fillOpacity": 0.4},
        )

        self.add_layer(gbif_polygons)

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

    def add_basemap_gui(self, options=None, position="topright"):
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
                self.layers = self.layers[:-2]
                self.add_basemap(change["new"])

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

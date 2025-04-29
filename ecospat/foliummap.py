"""This module provides a custom Map class that extends folium.Map"""

import folium
import folium.plugins
import requests
import os
import tempfile
import geopandas as gpd
import shutil
from localtileserver import get_folium_tile_layer, TileClient


class Map(folium.Map):
    """A custom Map class that extends folium.Map."""

    def __init__(self, center=(0, 0), zoom=2, tiles="OpenStreetMap", **kwargs):
        """Initializes the Map object.

        Args:
            center (tuple, optional): The initial center of the map as (latitude, longitude). Defaults to (0, 0).
            zoom (int, optional): The initial zoom level of the map. Defaults to 2.
            tiles (str, optional): The tile layer to use for the map. Defaults to "OpenStreetMap".
                Available options:
                    - "OpenStreetMap": Standard street map.
                    - "Esri.WorldImagery": Satellite imagery from Esri.
                    - "Esri.WorldTerrain": Terrain map from Esri.
                    - "Esri.WorldStreetMap": Street map from Esri.
                    - "CartoDB.Positron": A light and minimalist map style.
                    - "CartoDB.DarkMatter": A dark-themed map style.

            **kwargs: Additional keyword arguments for the folium.Map class.
        """
        super().__init__(location=center, zoom_start=zoom, tiles=tiles, **kwargs)

    def add_basemap(self, basemap):
        """Add a basemap to the map using folium's TileLayer.

        Args:
            basemap (str): The name of the basemap to add.
        """
        # Folium built-in tile layers
        builtin_tiles = [
            "OpenStreetMap",
            "OpenTopoMap",
            "Esri.WorldImagery",
            "Esri.WorldTerrain",
            "CartoDB Positron",
            "CartoDB Dark_Matter",
        ]

        if basemap in builtin_tiles:
            folium.TileLayer(basemap, name=basemap).add_to(self)

        else:
            custom_tiles = {
                "OpenTopoMap": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                "Esri.WorldImagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            }

            if basemap in custom_tiles:
                folium.TileLayer(
                    tiles=custom_tiles[basemap], attr="Custom Attribution", name=basemap
                ).add_to(self)
            else:
                raise ValueError(f"Basemap '{basemap}' is not available.")

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
            **kwargs: Additional keyword arguments for the folium.GeoJson layer.

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

        geojson = folium.GeoJson(data=geojson, **kwargs)
        geojson.add_to(self)

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
        """Adds a shapefile from a URL to the map using Folium.

        This function downloads the shapefile components (.shp, .shx, .dbf) from the specified URL, stores them
        in a temporary directory, reads the shapefile using Geopandas, converts it to GeoJSON format, and
        then adds it to the Folium map. If the shapefile's coordinate reference system (CRS) is not set, it assumes
        the CRS to be EPSG:4326 (WGS84).

        Args:
            url (str): The URL pointing to the shapefile's location. The URL should be a raw GitHub link to
                        the shapefile components (e.g., ".shp", ".shx", ".dbf").
            **kwargs: Additional keyword arguments to pass to the `GeoJson` method for styling and
                        configuring the GeoJSON layer on the Folium map.
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

            folium.GeoJson(geojson, **kwargs).add_to(self)

            shutil.rmtree(temp_dir)

        except Exception as e:
            print(f"Error loading shapefile: {e}")

    def add_gdf(self, gdf, **kwargs):
        """Adds a GeoDataFrame to the map.

        Args:
            gdf (geopandas.GeoDataFrame): The GeoDataFrame to add.
            **kwargs: Additional keyword arguments for the GeoJSON layer.
        """
        gdf = gdf.to_crs(epsg=4326)
        geojson = gdf.__geo_interface__
        self.add_geojson(geojson, **kwargs)

    def add_vector(self, data, **kwargs):
        """Adds vector data to the map.

        Args:
            data (str, geopandas.GeoDataFrame, or dict): The vector data. Can be a file path, GeoDataFrame, or GeoJSON dictionary.
            **kwargs: Additional keyword arguments for the GeoJSON layer.

        Raises:
            ValueError: If the data type is invalid.
        """
        import geopandas as gpd

        if isinstance(data, str):
            gdf = gpd.read_file(data)
            self.add_gdf(gdf, **kwargs)
        elif isinstance(data, gpd.GeoDataFrame):
            self.add_gdf(data, **kwargs)
        elif isinstance(data, dict):
            self.add_geojson(data, **kwargs)
        else:
            raise ValueError("Invalid data type")

    def add_layer_control(self):
        """Adds a layer control widget to the map."""
        folium.LayerControl().add_to(self)

    def add_split_map(
        self,
        left,
        right="cartodbpositron",
        name_left="Left Raster",
        name_right="Right Raster",
        colormap_left=None,
        colormap_right=None,
        opacity_left=1.0,
        opacity_right=1.0,
        **kwargs,
    ):
        """
        Adds a split map with one or both sides displaying a raster GeoTIFF, with independent colormaps.

        Args:
            left (str or TileClient): Left map layer (Tile URL, basemap name, or GeoTIFF path).
            right (str or TileClient): Right map layer (Tile URL, basemap name, or GeoTIFF path).
            name_left (str, optional): Name for the left raster layer. Defaults to "Left Raster".
            name_right (str, optional): Name for the right raster layer. Defaults to "Right Raster".
            colormap_left (str, optional): Colormap for the left raster. Defaults to None.
            colormap_right (str, optional): Colormap for the right raster. Defaults to None.
            opacity_left (float, optional): Opacity of the left raster. Defaults to 1.0.
            opacity_right (float, optional): Opacity of the right raster. Defaults to 1.0.
            **kwargs: Additional arguments for the tile layers.

        Returns:
            None
        """

        # Convert left layer if it's a raster file/URL
        if isinstance(left, str) and left.endswith(".tif"):
            client_left = TileClient(left)
            left_layer = get_folium_tile_layer(
                client_left,
                name=name_left,
                colormap=colormap_left,
                opacity=opacity_left,
                **kwargs,
            )
        else:
            left_layer = folium.TileLayer(left, overlay=True, **kwargs)

        # Convert right layer if it's a raster file/URL
        if isinstance(right, str) and right.endswith(".tif"):
            client_right = TileClient(right)
            right_layer = get_folium_tile_layer(
                client_right,
                name=name_right,
                colormap=colormap_right,
                opacity=opacity_right,
                **kwargs,
            )
        else:
            right_layer = folium.TileLayer(right, overlay=True, **kwargs)

        # Add layers to the map
        left_layer.add_to(self)
        right_layer.add_to(self)

        # Create split-screen effect
        split_map = folium.plugins.SideBySideLayers(left_layer, right_layer)
        split_map.add_to(self)

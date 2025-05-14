# Welcome to ecospat

<div align="center">

  <a href="https://pypi.python.org/pypi/ecospat">
    <img src="https://img.shields.io/pypi/v/ecospat.svg" alt="PyPI version"/>
  </a>

  <br/>

  <a href="https://raw.githubusercontent.com/anytko/ecospat/master/logo.png">
    <img src="https://raw.githubusercontent.com/anytko/ecospat/master/logo.png" alt="logo" width="150"/>
  </a>

  <br/><br/>

  <strong>A python package to characterize the range dynamics and shifts of North American tree species.</strong>

</div>

-   GitHub Repo: <https://github.com/anytko/ecospat>
-   Documentation: <https://anytko.github.io/ecospat>
-   PyPI: <https://pypi.org/project/ecospat/>
-   Ecospat tutorials on YouTube: <a href="https://youtu.be/Lc7Zh47KA8w" target="_blank">An introduction to ecospat</a>
-   Free software: <a href="https://opensource.org/license/MIT" target="_blank">MIT License</a>

## Introduction & Statement of Need
**Ecospat** is a Python package for the interactive mapping and characterization of range edges and their predicted persistence. Species ranges are noncontiguous and comprised of separate populations. We can characterize these populations into different range edges based on their latitudinal positions.
- Leading Edge: Populations north of the core
- Core: Largest, most central populations
- Trailing Edge: Populations south of the core
- Relict (latitudinal or longitudinal): Highly disconnected populations south of the trailing edge or very far east or west of the range

If we understand how these edges are moving, we can also infer biologically important characteristics of these populations. Under climate change, species are expected to move northward to track their climate envelopes. Using this model of Positive/all-together movement, the leading edge is expected to demonstrate low genetic and functional diversity, while the trailing edge gains genetic and functional diversity. However, not all range movements are equal under climate change or disturbance; other patterns of movement such as negative movement, stability, pull-apart patterns, and reabsorption to the core exist and affect the genetic and functional diversity of populations. At present, there are no widely adopted software implementations for characterizing range dynamics.

Using the historical ranges of over 670 North American tree species and modern GBIF data, **ecospat** categorizes the range edges of species, northward movement of ranges, and changes in population density over time to identify range patterns and create a predicted persistence raster to be used in species distribution models and further research.

## Features

-   Maps and identifies edges of historical and contemporary ranges for over 600 tree species.
-   Calculates the northward rate of movement, change in population density through time, average temperature, precipitation, and elevation of range edges.
-   Assigns a range movement pattern (i.e. Moving together, Pulling apart, Stability, or Reabsorption)
-   Generates a predicted persistence raster that can be downloaded and used in further analyses.


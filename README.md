# Welcome to ecospat

<div align="center">

  <a href="https://pypi.python.org/pypi/ecospat">
    <img src="https://img.shields.io/pypi/v/ecospat.svg" alt="PyPI version"/>
  </a>

  <br/>

  <a href="https://raw.githubusercontent.com/anytko/ecospat/master/images/logo.png">
    <img src="https://raw.githubusercontent.com/anytko/ecospat/master/images/logo.png" alt="logo" width="150"/>
  </a>

  <br/><br/>

  <strong>A python package that uses GBIF data to categorize the range edges of species through time to understand patterns of range movement, population dynamics, and individual persistence.</strong>

</div>

-   GitHub Repo: <https://github.com/anytko/ecospat>
-   Documentation: <https://anytko.github.io/ecospat>
-   PyPI: <https://pypi.org/project/ecospat/>
-   Ecospat tutorials on YouTube: <a href="https://youtu.be/Lc7Zh47KA8w" target="_blank">An introduction to ecospat</a>
-   Free software: <a href="https://opensource.org/license/MIT" target="_blank">MIT License</a>

## Introduction & Statement of Need
**Ecospat** is a Python package and accompanying webapp for the interactive mapping and characterization of range edges, the identification of range and population dynamics within and across edges, and the predicted propagule pressure and persistence of individuals.

Species ranges are often noncontiguous and comprised of disjunct populations. We can characterize these populations into different range edges based on their latitudinal positions.
- Leading Edge: Populations north of the core
- Core: Largest, most central populations representing a core zone of admixture
- Trailing Edge: Populations south of the core
- Relict (latitudinal or longitudinal): Highly disconnected populations south of the trailing edge or eastern/western isolates

We expect that species are moving northward to track their climate envelopes; however, under climate change, populations have demonstrated a wide variety of range movement dynamics - including moving north or southward together, pulling apart, reabsorbing into the core zone of admixture, and remaining stable. Not only are species' ranges moving, but individuals within and across range edges are also moving, resulting in differential population dynamics.

#### Therefore, if we can identify
1. range edges
2. range movement patterns
3. population dynamics within and across range edges

We can better understand how species have responded to past climate change and infer their potential for persistence at individual, population, community, and ecosystem levels. For instance, populations across a species’ range may gain or lose relative importance for maintaining ecosystem services and functions depending on their abundance and the persistence of their individuals.

At present, there are no widely adopted software implementations for characterizing range edges or their dynamics. However, occurrence data spanning both small and large spatial and temporal scales makes this possible.

Using the historical ranges of over 670 North American tree species, historical GBIF data, and modern GBIF data, **_ecospat_** categorizes the range edges of species, northward movement of ranges, and changes in population density over time to identify range patterns, generate a propagule pressure raster, and calculate the predicted persistence of individuals through time to connect community science to community conservation.

## Features

-   Maps and identifies historical and contemporary range edges of species.
-   Calculates the northward rate of movement, change in population density through time, average temperature, precipitation, and elevation of range edges.
-   Assigns a range movement pattern (i.e. Moving together, Pulling apart, Stability, or Reabsorption)
-   Generates a propagule pressure raster that can be downloaded and used in further analyses.
-   Predicts the one and five year persistence of individuals and assigns them to a risk decile based on predicted persistence.
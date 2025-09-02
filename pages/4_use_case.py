import solara
from solara import component

from pathlib import Path


@solara.component
def Page():
    solara.Style(Path("/Users/alivianytko/Desktop/ecospat/solara/assets/custom.css"))

    with solara.Column(align="center", gap="1rem"):
        # Title (centered, bold)
        solara.Markdown("# **Community Science to Community Conservation**")

        # Subtitle
        solara.Markdown(
            "##_A use-case demonstrating how ecospat can be used to develop local conservation priorities and grass-roots management._",
            style={"text-align": "center"},
        )

        solara.Markdown(
            "##Sarah, Amy, John, and Michael are avid naturalists and community scientists in their neighborhood in Halifax.",
            style={"text-align": "center"},
        )

        community_scientists = "images/community_scientists.png"
        halifax_map = "images/Halifax_map.jpg"

        with solara.Row(gap="2rem"):  # horizontally aligned
            solara.Image(community_scientists, width="400px")
            solara.Image(halifax_map, width="400px")

        solara.Markdown(
            "###Recently, they’ve noticed a decline in mature conifers across the city and nearby areas.",
            style={"text-align": "center"},
        )

        solara.Markdown(
            "###They check GBIF to identify common confier species across Nova Scotia.",
            style={"text-align": "center"},
        )

        with solara.Row(gap="2rem"):  # horizontally aligned
            with solara.Column():
                solara.Markdown(
                    "**_Abies balsamea_ (Balsam Fir)**", style={"text-align": "center"}
                )
                solara.Image("images/abies_balsamea_gbif.jpg", width="400px")

            with solara.Column():
                solara.Markdown(
                    "**_Picea rubens_ (Red Spruce)**", style={"text-align": "center"}
                )
                solara.Image("images/picea_rubens_gbif.jpg", width="400px")

            with solara.Column():
                solara.Markdown(
                    "**_Pinus strobus_ (White Pine)**", style={"text-align": "center"}
                )
                solara.Image("images/pinus_strobus_gbif.jpg", width="400px")

        solara.Markdown(
            "_Abies balsamea_, _Picea rubens_, and _Pinus strobus_ are widely distributed across Nova Scotia and are all important to ecosystem services and functions such as carbon storage, nutrient cycling, and habitat provision.",
            style={"text-align": "left"},
        )

        solara.Markdown(
            "However, the group has limited resources and needs more information about the predicted persistence and dynamics of these species to focus their efforts.",
            style={"text-align": ";eft"},
        )

        solara.Markdown("###They search _ecospat_ for:", style={"text-align": "center"})
        solara.Markdown(
            """
        - **_Abies balsamea_** - 5000 Occurrences - 10% Yearly Mortality
        - **_Picea rubens_** - 5000 Occurrences - 10% Yearly Mortality
        - **_Pinus strobus_** - 5000 Occurrences - [12%](https://www.sciencedirect.com/science/article/abs/pii/S0378112717315645) Yearly Mortality
        """,
            style={"text-align": "center"},
        )

        with solara.Column():
            solara.Markdown(
                "**Individual Persistence:** _Abies balsamea_  \n**Range Dynamic:** Pulling Apart",
                style={"text-align": "center"},
            )
            solara.Image("images/abies_balsamea_persistence.jpg", width="800px")

            solara.Markdown(
                "**Individual Persistence:** _Picea rubens_  \n**Range Dynamic:** Stability",
                style={"text-align": "center"},
            )
            solara.Image("images/picea_rubens_persistence.jpg", width="800px")

            solara.Markdown(
                "**Individual Persistence:** _Pinus strobus_  \n**Range Dynamic:** Positive Moving Together",
                style={"text-align": "center"},
            )
            solara.Image("images/pinus_strobus_persistence.jpg", width="800px")

        solara.Markdown(
            "### Based on the range edge, movement, and individual persistence, the group determines they should focus on _Pinus strobus_ (White Pine), "
            "as it is in the leading edge and has the most individuals at risk.",
            style={"text-align": "center"},
        )

        solara.Image("images/pinus_strobus_close.jpg", width="800px")

        with solara.Column(
            style={
                "text-align": "left",
                "padding-bottom": "3vh",
                "margin-bottom": "50px",
            }
        ):
            solara.Markdown(
                "Although _Pinus strobus_ is found near the city center, individuals with lower predicted persistence are also present in the Blue Mountain–Birch Cove Lakes Wilderness Area. "
                "The group decides to focus their efforts here, monitoring existing individuals for **mortality, insects, and disease**, and working with local authorities to **promote planting and spread of White Pine**.",
                style={"text-align": "left"},
            )

            solara.Markdown(
                "By documenting and managing White Pine in this area, their observations will help better inform predictions of individual persistence over time.",
                style={"text-align": "left"},
            )

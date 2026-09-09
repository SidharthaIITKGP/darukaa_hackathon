"""The curated causal edge set and graph builder.

Section A edges carry published effect sizes and must not be adjusted.
Section B edges are mechanistic: own wide-interval estimates that connect
the graph end to end, backed only by assessment-report direction, never by
an invented number.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from src.graph.nodes import INTERVENTIONS, STATE_VARIABLES, is_intervention, is_state_variable
from src.graph.schemas import (
    CausalEdge,
    Conditions,
    Distribution,
    EffectMetric,
    EvidenceRef,
    EvidenceStrength,
    Confidence,
    validate_source_ids,
)

DERIVED_GRAPH_PATH = "data/derived/causal_graph.json"

EDGES: list[CausalEdge] = []

# ============================= SECTION A: QUANTIFIED EDGES =============================

EDGES.append(
    CausalEdge(
        source="legume_cover_crop",
        target="soil_organic_carbon",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.049, ci_high=0.13, ci_level=0.95, point=0.073),
        lag_years=(2, 4),
        strength=EvidenceStrength.META_ANALYSIS,
        confidence=Confidence.MODERATE,
        contested=True,
        contested_note=(
            "Joshi 2023 reports +7.3% (95% CI 4.9-9.6%, n=61 studies); McClelland 2021 "
            "reports +12% (+1.11 Mg C/ha, 40 publications, 181 observations) in temperate "
            "systems. Interval widened to 4.9-13% to span both rather than averaging."
        ),
        evidence=[
            EvidenceRef(
                source_id="Joshi_2023_covercrops_SOC",
                role="primary",
                note=(
                    "Pooled 0-15cm estimate. Joshi also reports tillage subgroups: "
                    "+8.6-33.7% at 0-15cm under conventional tillage versus +0.3-10.5% "
                    "under no-tillage. The graph models the pooled effect because tillage "
                    "state is not represented as a node, so the no-till subgroup's "
                    "condition (no-tillage) can never be gated and was removed as a "
                    "separate edge rather than left firing unconditionally."
                ),
            ),
            EvidenceRef(source_id="McClelland_2020_covercrops_SOC", role="corroborating"),
        ],
        conditions=Conditions(land_use=["cropland"]),
        mechanism=(
            "Legume cover crops fix atmospheric nitrogen and exude labile carbon compounds "
            "from their roots, feeding soil microbial biomass that stabilises new organic "
            "matter. Their above and below-ground residues add fresh carbon input beyond "
            "what the cash crop alone contributes. Joshi 2023 reports +8.6-33.7% at 0-15cm "
            "under conventional tillage versus +0.3-10.5% under no-tillage; the graph "
            "models the pooled effect since tillage state is not a node here."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="non_legume_cover_crop",
        target="soil_organic_carbon",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.03, ci_high=0.10, ci_level=0.95),
        lag_years=(2, 4),
        strength=EvidenceStrength.META_ANALYSIS,
        confidence=Confidence.LOW,
        conditions=Conditions(land_use=["cropland"]),
        evidence=[
            EvidenceRef(source_id="Joshi_2023_covercrops_SOC", role="primary"),
            EvidenceRef(source_id="McClelland_2020_covercrops_SOC", role="corroborating"),
        ],
        mechanism=(
            "Non-legume cover crops (grasses, brassicas) add above and below-ground biomass "
            "carbon in the same way as legume cover crops, but without symbiotic nitrogen "
            "fixation the carbon input is lower and less readily stabilised, giving a "
            "narrower and lower expected gain than the legume case."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="residue_retention",
        target="soil_organic_carbon",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.0008, ci_high=0.0030, ci_level=0.95),
        lag_years=(1, 20),
        strength=EvidenceStrength.MULTI_SITE,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2022_GSOCseq", role="primary")],
        mechanism=(
            "GSOCseq models three carbon-input scenarios above business-as-usual: SSM1 (+5% "
            "input, low residue retention intensity), SSM2 (+10%, medium), and SSM3 (+20%, "
            "high), yielding annual global sequestration of 0.14/0.29/0.57 Pg C/yr, equivalent "
            "to 0.08/0.15/0.30% per year. The interval spans this low-to-high input range as "
            "a per-year rate rather than a single scenario."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="farmyard_manure",
        target="soil_organic_carbon",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.0008, ci_high=0.0030, ci_level=0.95),
        lag_years=(1, 20),
        strength=EvidenceStrength.MULTI_SITE,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2022_GSOCseq", role="primary")],
        mechanism=(
            "Manure application increases carbon input above business-as-usual in the same "
            "way residue retention does; GSOCseq does not distinguish input source, so the "
            "same SSM1-SSM3 low/medium/high input-intensity basis and per-year rate applies."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="alley_cropping",
        target="species_richness",
        sign="+",
        metric=EffectMetric.LRR,
        effect=Distribution(family="normal", ci_low=-0.10, ci_high=0.45, ci_level=0.95),
        lag_years=(3, 10),
        strength=EvidenceStrength.META_ANALYSIS,
        confidence=Confidence.LOW,
        contested=True,
        contested_note=(
            "GCB 2025 global summary reports biodiversity and abundance improved across "
            "major taxa. Mupepele 2021 finds no unequivocal effect in European agroforestry. "
            "Boinot 2022 critiques Mupepele: species richness alone is an insufficient "
            "biodiversity measure, and at least 11 of 28 included studies lacked adequate "
            "controls. Encoded as a wide interval spanning no effect."
        ),
        evidence=[
            EvidenceRef(source_id="GCB_2025_agroforestry_multifunctionality", role="primary"),
            EvidenceRef(source_id="Mupepele_2021_agroforestry_biodiv", role="contradicting"),
            EvidenceRef(source_id="Boinot_2022_agroforestry_critique", role="critique"),
        ],
        mechanism=(
            "Tree rows add structural and floral resource diversity that can support more "
            "species, but the effect is contested in the literature and depends heavily on "
            "system age, tree species mix, and study methodology."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="pollinator_abundance",
        target="crop_yield",
        sign="+",
        metric=EffectMetric.LRR,
        effect=Distribution(family="lognormal", ci_low=0.05, ci_high=0.30, ci_level=0.95),
        lag_years=(0, 1),
        strength=EvidenceStrength.META_ANALYSIS,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="Woodcock_2019_pollinator_yield", role="primary")],
        conditions=Conditions(land_use=["cropland"]),
        mechanism=(
            "Both functional diversity and abundance of pollinators independently enhance "
            "pollination service delivery, and the yield effect applies to insect-pollinated "
            "crops only, not to wind- or self-pollinated crops."
        ),
    )
)

# ============================= SECTION B: MECHANISTIC EDGES =============================

EDGES.append(
    CausalEdge(
        source="soil_organic_carbon",
        target="aggregate_stability",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.05, ci_high=0.2),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2020_soil_biodiversity", role="primary")],
        mechanism=(
            "Organic matter binds mineral particles into aggregates via microbial glues and "
            "root/fungal filaments, so higher soil carbon increases resistance of aggregates "
            "to slaking and mechanical breakdown."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="aggregate_stability",
        target="infiltration_rate",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.05, ci_high=0.2),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        mechanism=(
            "Stable aggregates preserve macropores at the soil surface instead of sealing "
            "under raindrop impact, so water continues to enter the profile rather than "
            "ponding and running off."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="infiltration_rate",
        target="plant_available_water",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.05, ci_high=0.2),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        mechanism=(
            "Faster infiltration diverts a larger share of rainfall into the root-zone "
            "profile instead of surface runoff, increasing the water stored for plant uptake."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="plant_available_water",
        target="canopy_cover",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPCC_2019_SRCCL_Ch6", role="primary")],
        mechanism=(
            "Greater plant-available water relieves drought stress and supports fuller leaf "
            "expansion and canopy closure across the growing season."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="canopy_cover",
        target="structural_heterogeneity",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 5),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPBES_2018_LDR", role="primary")],
        mechanism=(
            "Denser, taller canopy adds vertical strata and patchiness to the vegetation "
            "structure that a bare or uniformly short crop stand lacks."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="structural_heterogeneity",
        target="pollinator_abundance",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPBES_2018_LDR", role="primary")],
        mechanism=(
            "Structurally varied vegetation offers more nesting sites and a longer sequence "
            "of flowering resources across the season, supporting a larger resident "
            "pollinator community."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="soil_organic_carbon",
        target="microbial_biomass_carbon",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.05, ci_high=0.2),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2020_soil_biodiversity", role="primary")],
        mechanism=(
            "Soil organic carbon is the substrate microbes metabolise, so a larger carbon "
            "pool supports a larger standing microbial biomass."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="microbial_biomass_carbon",
        target="nutrient_cycling_rate",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.03, ci_high=0.12),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2020_soil_biodiversity", role="primary")],
        mechanism=(
            "Microbial biomass is the engine that mineralises organic matter into "
            "plant-available forms; a larger microbial pool turns over nutrients faster."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="nutrient_cycling_rate",
        target="nitrogen_availability",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.03, ci_high=0.12),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2020_soil_biodiversity", role="primary")],
        mechanism=(
            "Faster mineralisation of organic nitrogen pools directly increases the "
            "plant-available nitrogen fraction in the root zone."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="nitrogen_availability",
        target="crop_yield",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.05, ci_high=0.2),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        mechanism=(
            "Nitrogen is the most commonly yield-limiting nutrient in most cropping systems, "
            "so higher availability relieves a limiting constraint on growth."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="contour_bunding",
        target="infiltration_rate",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.05, ci_high=0.2),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        conditions=Conditions(slope_pct=(2, 60)),
        mechanism=(
            "Contour bunds intercept overland flow and pond it behind the bund, extending "
            "the residence time water has to infiltrate instead of running off downslope. "
            "The effect only applies where there is meaningful slope for runoff to occur."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="contour_bunding",
        target="erosion_rate",
        sign="-",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.05, ci_high=0.2),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[
            EvidenceRef(
                source_id="FAO_2017_VGSSM",
                role="primary",
                note=(
                    "Supports direction only (bunding reduces erosion). The 20-60% "
                    "reduction interval is an own estimate, not a published figure."
                ),
            )
        ],
        conditions=Conditions(slope_pct=(2, 60)),
        mechanism=(
            "By slowing overland flow velocity and trapping sediment behind the bund, "
            "contour bunding reduces the volume of soil detached and transported off the "
            "field."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="hedgerow_planting",
        target="structural_heterogeneity",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(2, 5),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPBES_2018_LDR", role="primary")],
        mechanism=(
            "Woody hedgerows add a permanent vertical structural element and edge habitat "
            "that a monoculture field margin lacks."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="hedgerow_planting",
        target="habitat_connectivity",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(2, 5),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPBES_2018_LDR", role="primary")],
        mechanism=(
            "Hedgerows act as linear corridors linking otherwise isolated patches of "
            "semi-natural habitat, allowing species to move between them."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="habitat_connectivity",
        target="species_richness",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(2, 7),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPBES_2018_LDR", role="primary")],
        mechanism=(
            "Better-connected habitat patches support larger, more resilient populations and "
            "allow recolonisation after local extinction, increasing the number of species "
            "an area can sustain."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="reduced_tillage",
        target="aggregate_stability",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.03, ci_high=0.12),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        mechanism=(
            "Reduced mechanical disturbance leaves fungal hyphal networks and existing "
            "aggregates intact rather than physically shattering them each pass."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="reduced_tillage",
        target="microbial_biomass_carbon",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.03, ci_high=0.12),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2020_soil_biodiversity", role="primary")],
        mechanism=(
            "Less disturbance reduces oxidation of soil organic matter and physical "
            "disruption of microbial habitat, letting microbial biomass build up over time."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="mulching",
        target="soil_moisture_retention",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.03, ci_high=0.12),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        conditions=Conditions(rainfall_mm=(200, 1200)),
        mechanism=(
            "A surface mulch layer shades and shelters the soil surface from direct solar "
            "radiation and wind, reducing evaporative water loss. Biomass-based, so gated on "
            "rainfall being sufficient to produce mulch material but not so high that "
            "moisture retention is not limiting."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="soil_moisture_retention",
        target="plant_available_water",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.03, ci_high=0.12),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        mechanism=(
            "Water retained in the profile rather than lost to evaporation directly adds to "
            "the pool available for root uptake."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="rotational_grazing",
        target="canopy_cover",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPCC_2019_SRCCL_Ch6", role="primary")],
        conditions=Conditions(land_use=["grazing_land"]),
        mechanism=(
            "Planned rest periods between grazing bouts let vegetation regrow canopy and "
            "root reserves before the next defoliation, increasing average standing cover "
            "compared to continuous grazing."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="farm_pond",
        target="groundwater_recharge",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        mechanism=(
            "A farm pond captures runoff that would otherwise leave the catchment and holds "
            "it in contact with the soil, where it can percolate downward and recharge the "
            "water table."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="groundwater_recharge",
        target="plant_available_water",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        mechanism=(
            "A shallower or more replenished water table raises capillary rise into the "
            "root zone, supplementing water available to plants between rain events."
        ),
    )
)

# --- Tradeoff edges: negative-sign edges converging on nodes also reached positively ---

EDGES.append(
    CausalEdge(
        source="alley_cropping",
        target="crop_yield",
        sign="-",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 5),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPCC_2019_SRCCL_Ch6", role="primary")],
        conditions=Conditions(rainfall_mm=(0, 600)),
        mechanism=(
            "Tree rows compete with the adjacent crop for water and light, an effect the "
            "IPCC response-option tables list as an adverse side effect of agroforestry. "
            "This competition dominates in low-rainfall settings where water is already "
            "limiting, unlike higher-rainfall systems where the tradeoff is often offset by "
            "microclimate benefits."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="alley_cropping",
        target="plant_available_water",
        sign="-",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(2, 5),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPCC_2019_SRCCL_Ch6", role="primary")],
        conditions=Conditions(climate_zone=["semi_arid", "arid"]),
        mechanism=(
            "The tree-crop competition that the alley_cropping -> crop_yield edge records "
            "is competition for water in the first place, so it acts on the water pool "
            "directly and not only on yield. Alley tree rows transpire from the same "
            "shallow root-zone store the crop draws on, which in semi-arid settings with "
            "limited recharge is a net withdrawal. Encoded as its own edge because a "
            "tradeoff that lands on plant_available_water is what tells a water-limited "
            "site that this intervention works against its binding constraint; recording "
            "only the yield consequence hides that."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="boundary_tree_planting",
        target="plant_available_water",
        sign="-",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(2, 5),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPCC_2019_SRCCL_Ch6", role="primary")],
        conditions=Conditions(climate_zone=["semi_arid", "arid"]),
        mechanism=(
            "Boundary trees draw on the same shallow soil water pool as adjacent crops "
            "through lateral root spread; in semi-arid settings where recharge is limited "
            "this net withdrawal outweighs any shading or windbreak benefit to soil "
            "moisture."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="no_tillage",
        target="nitrogen_availability",
        sign="-",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPCC_2022_AR6_WG3_Ch7", role="primary")],
        mechanism=(
            "Without incorporation, surface residues decompose and mineralise more slowly "
            "and residue carbon can immobilise available nitrogen in the short term, though "
            "this reverses as soil biology adapts over subsequent seasons."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="compost_application",
        target="soil_ph",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="normal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        mechanism=(
            "Organic amendments increase soil buffering capacity and release basic cations "
            "as they decompose, moderating pH toward neutral over successive applications. "
            "Direction depends on baseline pH; represented here as a small relative shift, "
            "not a precise pH-unit change."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="soil_organic_carbon",
        target="mycorrhizal_colonisation",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="FAO_2020_soil_biodiversity", role="primary")],
        mechanism=(
            "Higher soil organic carbon supports a richer soil food web and more stable "
            "root-zone habitat, favouring establishment of mycorrhizal symbioses."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="soil_organic_carbon",
        target="soil_fauna_abundance",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="FAO_2020_soil_biodiversity", role="primary")],
        mechanism=(
            "Organic carbon is the food resource base for earthworms and other soil "
            "macrofauna, so a larger carbon pool supports a larger fauna population."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="soil_fauna_abundance",
        target="nutrient_cycling_rate",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="FAO_2020_soil_biodiversity", role="primary")],
        mechanism=(
            "Earthworms and other macrofauna physically fragment and incorporate organic "
            "residues into the soil, accelerating the rate at which microbes can mineralise "
            "them."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="structural_heterogeneity",
        target="natural_enemy_abundance",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPBES_2018_LDR", role="primary")],
        mechanism=(
            "Structurally diverse vegetation provides overwintering refugia and alternative "
            "prey for predatory and parasitoid arthropods, supporting a larger resident "
            "natural-enemy community."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="natural_enemy_abundance",
        target="crop_yield",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPBES_2018_LDR", role="primary")],
        mechanism=(
            "Predators and parasitoids suppress crop pest populations, reducing pest-driven "
            "yield loss relative to a field with fewer natural enemies present."
        ),
    )
)

# --- Wiring for interventions that would otherwise have no outgoing edges ---

EDGES.append(
    CausalEdge(
        source="check_dam",
        target="groundwater_recharge",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        mechanism=(
            "A check dam slows ephemeral stream flow and ponds water behind the structure, "
            "increasing the time and wetted area over which water can percolate down to the "
            "aquifer."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="contour_trenching",
        target="infiltration_rate",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        conditions=Conditions(slope_pct=(2, 60)),
        mechanism=(
            "Trenches dug along the contour intercept overland flow and hold it in place "
            "long enough to infiltrate rather than run off, similar in principle to contour "
            "bunding but with greater storage volume per unit length."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="vetiver_grass_strips",
        target="erosion_rate",
        sign="-",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        conditions=Conditions(slope_pct=(2, 60)),
        mechanism=(
            "Dense, fibrous vetiver root mats and stiff above-ground stems form a living "
            "barrier along the contour that slows runoff velocity and traps sediment, "
            "reducing the soil mass that leaves the field."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="intercropping",
        target="soil_organic_carbon",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 4),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPCC_2022_AR6_WG3_Ch7", role="primary")],
        conditions=Conditions(land_use=["cropland"]),
        mechanism=(
            "Growing two crop species together increases total root biomass and ground "
            "cover relative to a monoculture of either alone, adding more carbon input to "
            "the soil over the season."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="crop_rotation_diversification",
        target="natural_enemy_abundance",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPBES_2018_LDR", role="primary")],
        mechanism=(
            "Rotating through a wider set of crop species interrupts pest life cycles tied "
            "to a single host crop and diversifies the resource base available to "
            "generalist predators and parasitoids across seasons."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="grazing_exclosure",
        target="canopy_cover",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPCC_2019_SRCCL_Ch6", role="primary")],
        conditions=Conditions(land_use=["grazing_land"]),
        mechanism=(
            "Removing grazing pressure entirely allows vegetation to regrow and canopy to "
            "close without the recurring defoliation that even a rotational grazing regime "
            "still imposes."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="integrated_nutrient_management",
        target="nitrogen_availability",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="FAO_2017_VGSSM", role="primary")],
        mechanism=(
            "Combining organic and inorganic nutrient sources synchronises nutrient release "
            "with crop demand more closely than either source alone, increasing nitrogen "
            "actually available for uptake rather than lost to leaching or volatilisation."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="agroforestry_silvopasture",
        target="canopy_cover",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(2, 6),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPCC_2019_SRCCL_Ch6", role="primary")],
        conditions=Conditions(land_use=["grazing_land"]),
        mechanism=(
            "Integrating trees into grazing land adds a woody canopy layer above the "
            "herbaceous layer, increasing total canopy cover beyond what pasture alone "
            "provides."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="biochar_application",
        target="soil_organic_carbon",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 3),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="IPCC_2022_AR6_WG3_Ch7", role="primary")],
        mechanism=(
            "Biochar is a highly recalcitrant form of carbon that resists microbial "
            "decomposition for centuries, so its addition directly and durably raises the "
            "measured soil carbon pool."
        ),
    )
)


def build_graph() -> nx.MultiDiGraph:
    """Build the causal graph. Multi-edge because some node pairs carry more
    than one distinct published estimate (e.g. legume_cover_crop -> SOC)."""
    for edge in EDGES:
        is_intervention(edge.source) if edge.source in INTERVENTIONS else is_state_variable(edge.source)
        is_intervention(edge.target) if edge.target in INTERVENTIONS else is_state_variable(edge.target)

    g = nx.MultiDiGraph()
    for name in STATE_VARIABLES:
        g.add_node(name, kind="state_variable", **STATE_VARIABLES[name])
    for name in INTERVENTIONS:
        g.add_node(name, kind="intervention", **INTERVENTIONS[name])

    for edge in EDGES:
        g.add_edge(edge.source, edge.target, edge=edge)
    return g


def save_graph(path: str = DERIVED_GRAPH_PATH) -> None:
    g = build_graph()
    data = nx.node_link_data(g, edges="edges")
    for link in data["edges"]:
        link["edge"] = json.loads(link["edge"].model_dump_json())
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    validate_source_ids(EDGES, "sources.yaml")
    graph = build_graph()
    save_graph()
    print(f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    print(f"quantified edges: {sum(1 for e in EDGES if e.strength == EvidenceStrength.META_ANALYSIS)}")
    print(f"mechanistic edges: {sum(1 for e in EDGES if e.strength == EvidenceStrength.MECHANISTIC)}")
    print(f"contested edges: {sum(1 for e in EDGES if e.contested)}")
    print(f"negative edges: {sum(1 for e in EDGES if e.sign == '-')}")
    print(f"saved graph to {DERIVED_GRAPH_PATH}")

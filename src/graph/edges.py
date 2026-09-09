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

# ============================== KNOWN MODELLING GAPS ==============================
#
# Gaps identified and deliberately left open. Recorded here because an
# unrecorded gap is indistinguishable from an oversight.
#
# 1. Cover crops carry no water-competition edge, while alley cropping does.
#    alley_cropping -> plant_available_water and boundary_tree_planting ->
#    plant_available_water both carry negative edges gated to semi-arid and
#    arid zones, representing trees drawing on the same shallow root-zone
#    store the crop uses. A cover crop transpires and competes for that same
#    water, so the equivalent edges (legume_cover_crop and
#    non_legume_cover_crop -> plant_available_water, negative, gated to
#    semi-arid and arid) should exist and do not.
#
#    The asymmetry means the graph likely OVERSTATES cover crops in dry
#    settings, particularly below about 300mm where the rainfall precondition
#    on the carbon edges attenuates their benefit but nothing debits their
#    water cost. On the semi-arid Deccan demo site this is the difference
#    between a cover crop and a water-harvesting structure heading the
#    recommendation.
#
#    Deferred rather than added, for two reasons. No published effect size
#    was available: the corpus covers tree-crop competition (IPCC 2019 SRCCL
#    Ch. 6 response-option tables) but not cover-crop water use as a
#    quantified adverse effect, so the edge would have been an own estimate at
#    MECHANISTIC strength. And it would have been decisive rather than
#    marginal, flipping the headline recommendation on the flagship demo site.
#    Changing what the system recommends on the strength of an unpublished
#    guess is a worse trade than leaving a known asymmetry documented. Add the
#    edge when a published estimate exists to anchor it.

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
        conditions=Conditions(land_use=["cropland"], rainfall_mm=(300, 3000), slope_pct=(0, 8)),
        mechanism=(
            "Legume cover crops fix atmospheric nitrogen and exude labile carbon compounds "
            "from their roots, feeding soil microbial biomass that stabilises new organic "
            "matter. Their above and below-ground residues add fresh carbon input beyond "
            "what the cash crop alone contributes. Joshi 2023 reports +8.6-33.7% at 0-15cm "
            "under conventional tillage versus +0.3-10.5% under no-tillage; the graph "
            "models the pooled effect since tillage state is not a node here. Gated on "
            "rainfall 300-3000mm/yr: the whole effect runs through cover-crop biomass, and "
            "below roughly 300mm establishment is unreliable enough that there is often no "
            "biomass to speak of. The bound is an own judgement on establishment risk, not "
            "a published threshold. Also gated on slope 0-8%: above 8% the topsoil holding "
            "this carbon is exported faster than cover-crop residue accumulates it, which is "
            "the same threshold the erosion diagnosis uses, and the source meta-analyses "
            "measured predominantly gentle arable land. An own judgement, not a published "
            "threshold."
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
        conditions=Conditions(land_use=["cropland"], rainfall_mm=(300, 3000), slope_pct=(0, 8)),
        evidence=[
            EvidenceRef(source_id="Joshi_2023_covercrops_SOC", role="primary"),
            EvidenceRef(source_id="McClelland_2020_covercrops_SOC", role="corroborating"),
        ],
        mechanism=(
            "Non-legume cover crops (grasses, brassicas) add above and below-ground biomass "
            "carbon in the same way as legume cover crops, but without symbiotic nitrogen "
            "fixation the carbon input is lower and less readily stabilised, giving a "
            "narrower and lower expected gain than the legume case. Gated on rainfall "
            "300-3000mm/yr for the same establishment reason as the legume edge, and on "
            "slope 0-8% for the same carbon-export reason. Own judgements rather than "
            "published thresholds."
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
        conditions=Conditions(rainfall_mm=(300, 3000)),
        mechanism=(
            "Tree rows add structural and floral resource diversity that can support more "
            "species, but the effect is contested in the literature and depends heavily on "
            "system age, tree species mix, and study methodology. Gated on rainfall "
            "300-3000mm/yr: the resource diversity is the tree rows themselves, and below "
            "roughly 300mm their establishment without irrigation is unreliable. An own "
            "judgement on establishment risk, not a published threshold."
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
        conditions=Conditions(rainfall_mm=(0, 1200)),
        mechanism=(
            "Greater plant-available water relieves drought stress and supports fuller leaf "
            "expansion and canopy closure across the growing season. Gated on rainfall up "
            "to 1200mm/yr: the mechanism is relief of a water shortage, so it only fires "
            "where water is what limits the canopy. Above roughly 1200mm light and nutrient "
            "supply bind instead, and adding root-zone water buys little extra cover. This "
            "is the ceiling that stops water-holding interventions dominating humid sites. "
            "An own judgement on where the constraint changes hands, not a published "
            "threshold."
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
        conditions=Conditions(ph=(5.0, 8.0)),
        mechanism=(
            "Soil organic carbon is the substrate microbes metabolise, so a larger carbon "
            "pool supports a larger standing microbial biomass. Gated on pH 5.0-8.0: "
            "microbial activity falls away sharply outside that band, so on a strongly acid "
            "or calcareous alkaline soil the extra substrate is not metabolised into "
            "biomass at anything like the same rate. The band is an own judgement on where "
            "activity holds up, not a published threshold. This is the ONLY pH gate on the "
            "microbial and nutrient-cycling chain: the downstream links describe the same "
            "pH-sensitive activity, so gating each of them would apply one constraint three "
            "times over and zero the chain instead of attenuating it."
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
            "plant-available forms; a larger microbial pool turns over nutrients faster. "
            "Deliberately NOT pH gated, although the mineralisation it describes is pH "
            "sensitive: that one fact is already gated upstream at soil_organic_carbon -> "
            "microbial_biomass_carbon, and these edges are in series, so gating it here as "
            "well would apply a single constraint twice and collapse the chain to nothing "
            "rather than attenuate it."
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
            "plant-available nitrogen fraction in the root zone. Not pH gated, for the same "
            "reason as the microbial_biomass_carbon -> nutrient_cycling_rate edge: the pH "
            "sensitivity of this chain is gated once at its head, and repeating it at every "
            "link would multiply one constraint by the length of the chain."
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
        conditions=Conditions(slope_pct=(2, 25)),
        mechanism=(
            "Contour bunds intercept overland flow and pond it behind the bund, extending "
            "the residence time water has to infiltrate instead of running off downslope. "
            "The effect only applies where there is meaningful slope for runoff to occur. "
            "Gated on slope 2-25%: below 2% there is little overland flow to intercept and "
            "the bund does much less, and above 25% an earthen bund is unstable and tends "
            "to breach rather than pond. Both bounds are own judgements on where the "
            "structure works, not published thresholds."
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
        conditions=Conditions(slope_pct=(2, 25)),
        mechanism=(
            "By slowing overland flow velocity and trapping sediment behind the bund, "
            "contour bunding reduces the volume of soil detached and transported off the "
            "field. Gated on slope 2-25%: below 2% there is little water erosion to "
            "prevent, and above 25% the bund itself is unstable. Own judgements, not "
            "published thresholds."
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
        conditions=Conditions(rainfall_mm=(300, 3000)),
        mechanism=(
            "Woody hedgerows add a permanent vertical structural element and edge habitat "
            "that a monoculture field margin lacks. Gated on rainfall 300-3000mm/yr: the "
            "structure is living woody biomass, and below roughly 300mm hedge establishment "
            "without irrigation is unreliable. An own judgement, not a published threshold."
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
        conditions=Conditions(rainfall_mm=(300, 3000)),
        mechanism=(
            "Hedgerows act as linear corridors linking otherwise isolated patches of "
            "semi-natural habitat, allowing species to move between them. Gated on rainfall "
            "300-3000mm/yr on the same establishment grounds as the structural-heterogeneity "
            "edge: a hedge that does not establish is not a corridor. An own judgement, not "
            "a published threshold."
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
        conditions=Conditions(slope_pct=(0, 15)),
        mechanism=(
            "A farm pond captures runoff that would otherwise leave the catchment and holds "
            "it in contact with the soil, where it can percolate downward and recharge the "
            "water table. Gated on slope 0-15%: unlike a bund a pond needs no gradient to "
            "work, so there is no lower bound, but above roughly 15% the excavation and "
            "embankment needed to hold a useful volume stop being practical. An own "
            "judgement, not a published threshold."
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
        conditions=Conditions(slope_pct=(2, 30)),
        mechanism=(
            "A check dam slows ephemeral stream flow and ponds water behind the structure, "
            "increasing the time and wetted area over which water can percolate down to the "
            "aquifer. Gated on slope 2-30%: it needs a drainage line with enough gradient to "
            "concentrate flow, and above roughly 30% flows are energetic enough that a small "
            "structure is undercut rather than ponding. Own judgements, not published "
            "thresholds."
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
        conditions=Conditions(slope_pct=(2, 25)),
        mechanism=(
            "Trenches dug along the contour intercept overland flow and hold it in place "
            "long enough to infiltrate rather than run off, similar in principle to contour "
            "bunding but with greater storage volume per unit length. Gated on slope 2-25%: "
            "below 2% there is little overland flow to intercept, and above 25% trench "
            "spoil and side walls are unstable. Own judgements, not published thresholds."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="contour_trenching",
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
                    "Supports direction only (contour trenching reduces erosion). The "
                    "interval is an own estimate, not a published figure."
                ),
            )
        ],
        conditions=Conditions(slope_pct=(2, 25)),
        mechanism=(
            "A contour trench breaks the slope into shorter runs, so overland flow is "
            "intercepted before it accumulates the volume and velocity that detach soil, "
            "and the sediment it already carries settles in the trench instead of leaving "
            "the field. Gated on slope 2-25% for the same reasons as the infiltration edge: "
            "little erosion to prevent below 2%, unstable trench walls above 25%. Own "
            "judgements, not published thresholds."
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
        conditions=Conditions(slope_pct=(3, 40)),
        mechanism=(
            "Dense, fibrous vetiver root mats and stiff above-ground stems form a living "
            "barrier along the contour that slows runoff velocity and traps sediment, "
            "reducing the soil mass that leaves the field. Gated on slope 3-40%: below 3% "
            "there is little water erosion for a barrier to prevent, while the upper bound "
            "is set higher than for earthworks because a living hedge holds on ground where "
            "an earthen bund would breach. Own judgements, not published thresholds."
        ),
    )
)

# The three edges below carry vetiver beyond erosion control. A live contour
# hedge is not only a sediment barrier: it changes infiltration, soil
# structure and field vegetation structure as well, and an intervention
# represented by its erosion edge alone gets justified on one variable, which
# is below the brief's floor of three. All three are mechanistic own
# estimates. No meta-analysis in this corpus gives a pooled effect size for
# vetiver on any of them, so the direction is what these assert and the
# interval is deliberately weak. Each is gated on the same slope band as the
# erosion edge, because all three mechanisms run on intercepted overland
# flow, which is what a slope produces.

EDGES.append(
    CausalEdge(
        source="vetiver_grass_strips",
        target="infiltration_rate",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(0, 2),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[
            EvidenceRef(
                source_id="FAO_2017_VGSSM",
                role="primary",
                note=(
                    "Supports direction only (vegetative barriers slow runoff and increase "
                    "infiltration opportunity time). The interval is an own estimate, not a "
                    "published figure."
                ),
            )
        ],
        conditions=Conditions(slope_pct=(3, 40)),
        mechanism=(
            "A stiff grass hedge along the contour slows overland flow and ponds it "
            "shallowly on the upslope side, so water spends longer in contact with the "
            "surface and a larger share of it enters the profile instead of leaving the "
            "field. Gated on slope 3-40%, the same band as the erosion edge: the mechanism "
            "is interception of overland flow, and below 3% there is little of it to "
            "intercept. Own judgement, not a published threshold."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="vetiver_grass_strips",
        target="aggregate_stability",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 4),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[
            EvidenceRef(
                source_id="FAO_2017_VGSSM",
                role="primary",
                note=(
                    "Supports direction only (dense perennial root systems bind soil and "
                    "vegetative barriers retain fines on the field). The interval is an own "
                    "estimate, not a published figure."
                ),
            )
        ],
        conditions=Conditions(slope_pct=(3, 40)),
        mechanism=(
            "Dense perennial vetiver roots bind soil particles and exude organic compounds "
            "that hold aggregates together, while the hedge traps the fine, carbon-rich "
            "particles that runoff would otherwise carry away first. Keeping those fines on "
            "the field leaves the material aggregates are built from where it can be "
            "rebuilt. Slower than the erosion effect at 1-4 years, because it depends on "
            "root establishment rather than on the barrier standing up. Gated on slope "
            "3-40%. Own judgement, not a published threshold."
        ),
    )
)

EDGES.append(
    CausalEdge(
        source="vetiver_grass_strips",
        target="structural_heterogeneity",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.01, ci_high=0.08),
        lag_years=(1, 4),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[
            EvidenceRef(
                source_id="IPBES_2018_LDR",
                role="primary",
                note=(
                    "Supports direction only (permanent vegetation strips add structural "
                    "variation to otherwise uniform cropland). The interval is an own "
                    "estimate, not a published figure."
                ),
            )
        ],
        conditions=Conditions(slope_pct=(3, 40)),
        mechanism=(
            "Permanent grass strips introduce undisturbed, structurally distinct vegetation "
            "into a field that is otherwise uniform and tilled on one cycle, adding "
            "within-field variation in vegetation height and density and cover that "
            "persists across the fallow. Smaller in effect than a woody hedgerow, which "
            "adds a vertical layer a grass strip does not. Gated on slope 3-40%, since the "
            "strips are sited where contour barriers are worth installing. Own judgement, "
            "not a published threshold."
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
        conditions=Conditions(land_use=["cropland"], rainfall_mm=(300, 3000), slope_pct=(0, 8)),
        mechanism=(
            "Growing two crop species together increases total root biomass and ground "
            "cover relative to a monoculture of either alone, adding more carbon input to "
            "the soil over the season. Gated on rainfall 300-3000mm/yr: the added carbon is "
            "the companion crop's biomass, and below roughly 300mm two crops compete for "
            "water that will not support both. Also gated on slope 0-8%, above which that "
            "carbon is exported faster than it accumulates. Own judgements, not published "
            "thresholds."
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

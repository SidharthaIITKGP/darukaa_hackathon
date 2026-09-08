"""Node registry for the causal graph.

Two kinds of nodes: state variables (measurable soil/ecosystem/yield
quantities) and interventions (management practices a farmer can adopt).
No edges here, only the vocabulary edges are allowed to reference.
"""

from __future__ import annotations

from typing import Literal, TypedDict

EffortTier = Literal["low", "medium", "high"]


class StateVariableInfo(TypedDict):
    unit: str
    description: str


class InterventionInfo(TypedDict):
    category: str
    effort: EffortTier
    description: str


STATE_VARIABLES: dict[str, StateVariableInfo] = {
    "soil_organic_carbon": {
        "unit": "%",
        "description": "Organic carbon fraction of soil mass, the primary indicator of soil health.",
    },
    "soil_ph": {
        "unit": "pH",
        "description": "Soil acidity/alkalinity, governs nutrient availability and microbial activity.",
    },
    "aggregate_stability": {
        "unit": "index",
        "description": "Resistance of soil aggregates to breakdown under wetting and mechanical stress.",
    },
    "infiltration_rate": {
        "unit": "mm/hr",
        "description": "Rate at which water enters the soil surface, governs runoff versus storage.",
    },
    "plant_available_water": {
        "unit": "mm",
        "description": "Water held in the root zone between field capacity and wilting point.",
    },
    "soil_moisture_retention": {
        "unit": "%",
        "description": "Fraction of applied water retained in the soil profile over time.",
    },
    "microbial_biomass_carbon": {
        "unit": "mg/kg",
        "description": "Carbon held in the living soil microbial community, a labile nutrient pool.",
    },
    "mycorrhizal_colonisation": {
        "unit": "%",
        "description": "Fraction of root length colonised by mycorrhizal fungi, aiding nutrient and water uptake.",
    },
    "soil_fauna_abundance": {
        "unit": "index",
        "description": "Abundance of earthworms, arthropods and other soil macrofauna.",
    },
    "nutrient_cycling_rate": {
        "unit": "index",
        "description": "Rate at which organic matter is mineralised into plant-available nutrients.",
    },
    "nitrogen_availability": {
        "unit": "kg/ha",
        "description": "Plant-available nitrogen in the root zone.",
    },
    "erosion_rate": {
        "unit": "t/ha/yr",
        "description": "Rate of topsoil loss to water and wind erosion.",
    },
    "canopy_cover": {
        "unit": "%",
        "description": "Fraction of ground shaded by vegetation canopy.",
    },
    "structural_heterogeneity": {
        "unit": "index",
        "description": "Vertical and horizontal variety of vegetation structure across the landscape.",
    },
    "habitat_connectivity": {
        "unit": "index",
        "description": "Degree to which semi-natural habitat patches are linked across the landscape.",
    },
    "pollinator_abundance": {
        "unit": "index",
        "description": "Abundance of insect pollinators active in and around the farmed area.",
    },
    "natural_enemy_abundance": {
        "unit": "index",
        "description": "Abundance of predators and parasitoids that suppress crop pests.",
    },
    "species_richness": {
        "unit": "count",
        "description": "Number of distinct species observed across a taxonomic group of interest.",
    },
    "crop_yield": {
        "unit": "t/ha",
        "description": "Harvested crop mass per unit area.",
    },
    "groundwater_recharge": {
        "unit": "mm/yr",
        "description": "Rate at which water percolates past the root zone to replenish groundwater.",
    },
}

INTERVENTIONS: dict[str, InterventionInfo] = {
    "legume_cover_crop": {
        "category": "cover_cropping",
        "effort": "medium",
        "description": "Nitrogen-fixing cover crop grown between or alongside cash crops.",
    },
    "non_legume_cover_crop": {
        "category": "cover_cropping",
        "effort": "medium",
        "description": "Non-nitrogen-fixing cover crop (e.g. grasses, brassicas) grown between cash crops.",
    },
    "crop_rotation_diversification": {
        "category": "rotation",
        "effort": "low",
        "description": "Widening the sequence of crop species grown in rotation on a field.",
    },
    "intercropping": {
        "category": "rotation",
        "effort": "medium",
        "description": "Growing two or more crop species simultaneously on the same field.",
    },
    "alley_cropping": {
        "category": "agroforestry",
        "effort": "high",
        "description": "Rows of trees or shrubs interplanted with arable crops in the alleys between them.",
    },
    "boundary_tree_planting": {
        "category": "agroforestry",
        "effort": "medium",
        "description": "Trees planted along field boundaries rather than within the cropped area.",
    },
    "hedgerow_planting": {
        "category": "agroforestry",
        "effort": "medium",
        "description": "Linear woody shrub/tree strips planted along field margins.",
    },
    "reduced_tillage": {
        "category": "tillage",
        "effort": "low",
        "description": "Tillage limited to partial soil disturbance rather than full inversion.",
    },
    "no_tillage": {
        "category": "tillage",
        "effort": "low",
        "description": "Direct seeding with no mechanical soil disturbance between crops.",
    },
    "residue_retention": {
        "category": "residue_management",
        "effort": "low",
        "description": "Leaving crop residues on the field after harvest instead of removing or burning them.",
    },
    "mulching": {
        "category": "residue_management",
        "effort": "low",
        "description": "Applying organic or inorganic material over the soil surface to conserve moisture.",
    },
    "farmyard_manure": {
        "category": "organic_amendment",
        "effort": "medium",
        "description": "Application of composted or raw livestock manure as a soil amendment.",
    },
    "compost_application": {
        "category": "organic_amendment",
        "effort": "medium",
        "description": "Application of decomposed plant/organic waste compost as a soil amendment.",
    },
    "biochar_application": {
        "category": "organic_amendment",
        "effort": "high",
        "description": "Application of pyrolysed organic matter as a stable soil carbon amendment.",
    },
    "contour_bunding": {
        "category": "water_soil_conservation",
        "effort": "high",
        "description": "Earthen bunds built along contour lines to slow and capture surface runoff.",
    },
    "contour_trenching": {
        "category": "water_soil_conservation",
        "effort": "high",
        "description": "Trenches dug along contour lines to intercept runoff and promote infiltration.",
    },
    "farm_pond": {
        "category": "water_soil_conservation",
        "effort": "high",
        "description": "On-farm excavated pond that captures and stores runoff for later use.",
    },
    "check_dam": {
        "category": "water_soil_conservation",
        "effort": "high",
        "description": "Small barrier built across a drainage line to slow flow and trap sediment.",
    },
    "vetiver_grass_strips": {
        "category": "water_soil_conservation",
        "effort": "low",
        "description": "Dense vetiver grass hedgerows planted along contours to slow runoff and trap sediment.",
    },
    "rotational_grazing": {
        "category": "grazing_management",
        "effort": "medium",
        "description": "Livestock moved between paddocks on a planned rotation to allow vegetation recovery.",
    },
    "grazing_exclosure": {
        "category": "grazing_management",
        "effort": "low",
        "description": "Fencing livestock out of an area to allow vegetation to recover undisturbed.",
    },
    "integrated_nutrient_management": {
        "category": "nutrient_management",
        "effort": "medium",
        "description": "Combining organic and inorganic nutrient sources to match crop demand.",
    },
    "agroforestry_silvopasture": {
        "category": "agroforestry",
        "effort": "high",
        "description": "Integration of trees with livestock grazing on the same land unit.",
    },
}


def is_intervention(name: str) -> bool:
    if name not in INTERVENTIONS:
        raise KeyError(f"{name!r} is not a registered intervention")
    return True


def is_state_variable(name: str) -> bool:
    if name not in STATE_VARIABLES:
        raise KeyError(f"{name!r} is not a registered state variable")
    return True


def node_unit(name: str) -> str:
    if name in STATE_VARIABLES:
        return STATE_VARIABLES[name]["unit"]
    raise KeyError(f"{name!r} is not a registered state variable")

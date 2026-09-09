"""Twelve test sites, each chosen to exercise a different reasoning path.

The set is built to span degradation syndromes rather than to be a
representative sample of Indian farmland. Two sites that both come down to
"soil carbon is low" would test the same branch twice, so each site here is
the one case that reaches a distinct part of the system: a different branch
of limiting_factor, a different set of edge preconditions, or a different
failure mode of the input itself.

Expectations are the acceptable SET, not one answer. Where the correct
answer is genuinely uncertain the expectation is None and the note says why.
Nothing here is set to whatever the system currently returns: the
expectations were written from the site description, and where the system
disagrees with one the eval reports a miss rather than the expectation being
moved.

Flag names are the vocabulary in src.eval.metrics.FLAG_DETECTORS. A flag in
must_flag is a thing the response has to say out loud about this site.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.graph.schemas import Confidence, Measurement, Provenance, SiteState


def _m(value: float, unit: str, confidence: Confidence = Confidence.MODERATE) -> Measurement:
    return Measurement(
        value=value, unit=unit, provenance=Provenance.USER_STATED, confidence=confidence
    )


def _band(band: str, confidence: Confidence = Confidence.MODERATE) -> Measurement:
    return Measurement(band=band, provenance=Provenance.USER_STATED, confidence=confidence)


class EvalSite(BaseModel):
    site: SiteState
    expected_limiting_factor: str | None
    expected_top_intervention_class: list[str] | None
    must_flag: list[str]
    notes: str


# Intervention groups referred to below. Named here so the acceptable sets
# read as classes of practice rather than as lists of node names, and so two
# sites that should accept the same class cannot drift apart.
_WATER_HARVESTING = [
    "contour_bunding",
    "contour_trenching",
    "farm_pond",
    "check_dam",
    "vetiver_grass_strips",
    "mulching",
    "residue_retention",
]
_EROSION_CONTROL = [
    "contour_bunding",
    "contour_trenching",
    "vetiver_grass_strips",
    "check_dam",
    "alley_cropping",
    "hedgerow_planting",
]
_CARBON_BUILDING = [
    "farmyard_manure",
    "compost_application",
    "biochar_application",
    "residue_retention",
    "legume_cover_crop",
    "non_legume_cover_crop",
    "intercropping",
    "reduced_tillage",
    "no_tillage",
    "crop_rotation_diversification",
]
_HABITAT_STRUCTURE = [
    "hedgerow_planting",
    "boundary_tree_planting",
    "alley_cropping",
    "agroforestry_silvopasture",
    "intercropping",
    "crop_rotation_diversification",
]


def semi_arid_low_carbon_cropland() -> EvalSite:
    return EvalSite(
        site=SiteState(
            site_id="eval_semi_arid_low_carbon",
            soil_organic_carbon_pct=_m(0.35, "%"),
            annual_rainfall_mm=_m(340, "mm"),
            ph=_m(8.1, "pH"),
            slope_pct=_m(5, "%"),
            land_use=_band("cropland", Confidence.HIGH),
            crop="wheat",
        ),
        expected_limiting_factor="plant_available_water",
        expected_top_intervention_class=_WATER_HARVESTING,
        must_flag=["extrapolation"],
        notes=(
            "Deccan-like. Under 500mm with soil carbon under 1%, water binds before "
            "carbon: a biomass-based carbon practice cannot establish without moisture "
            "to grow the biomass. The cover-crop evidence in this corpus is scoped to "
            "temperate corn rotations, so applying it on a semi-arid Indian site is an "
            "extrapolation and has to be said."
        ),
    )


def steep_humid_cropland() -> EvalSite:
    return EvalSite(
        site=SiteState(
            site_id="eval_steep_humid",
            soil_organic_carbon_pct=_m(1.20, "%"),
            annual_rainfall_mm=_m(2100, "mm"),
            ph=_m(6.2, "pH"),
            slope_pct=_m(12, "%"),
            land_use=_band("cropland", Confidence.HIGH),
            crop="rice",
        ),
        expected_limiting_factor="erosion_rate",
        expected_top_intervention_class=_EROSION_CONTROL,
        must_flag=[],
        notes=(
            "Western Ghats-like. Steep ground under heavy rainfall loses topsoil faster "
            "than it forms it, so anything added to the profile leaves downslope before "
            "it accumulates. Erosion control has to precede fertility building."
        ),
    )


def flat_sub_humid_cropland() -> EvalSite:
    return EvalSite(
        site=SiteState(
            site_id="eval_flat_sub_humid",
            soil_organic_carbon_pct=_m(0.55, "%"),
            annual_rainfall_mm=_m(620, "mm"),
            ph=_m(7.6, "pH"),
            slope_pct=_m(1, "%"),
            land_use=_band("cropland", Confidence.HIGH),
            crop="wheat",
        ),
        expected_limiting_factor="soil_organic_carbon",
        expected_top_intervention_class=_CARBON_BUILDING,
        must_flag=["default_diagnosis"],
        notes=(
            "Indo-Gangetic-like. Nothing crosses a hard threshold: 0.55% carbon is above "
            "the 0.5% floor, rainfall is adequate, the ground is flat and pH is inside "
            "the band. Carbon is still the weakest link, but it is reached by elimination "
            "rather than by a crossed threshold, and the response has to say which of "
            "those two it is doing."
        ),
    )


def arid_grazing_land() -> EvalSite:
    return EvalSite(
        site=SiteState(
            site_id="eval_arid_grazing",
            soil_organic_carbon_pct=_m(0.45, "%"),
            annual_rainfall_mm=_m(200, "mm"),
            ph=_m(8.0, "pH"),
            slope_pct=_m(3, "%"),
            land_use=_band("grazing_land", Confidence.HIGH),
        ),
        expected_limiting_factor="plant_available_water",
        expected_top_intervention_class=_WATER_HARVESTING
        + ["rotational_grazing", "grazing_exclosure", "agroforestry_silvopasture"],
        must_flag=[],
        notes=(
            "200mm is arid rather than semi-arid, and the land is grazed rather than "
            "cropped, so every cropland precondition in the graph should be unmet here. "
            "The point of the site is whether land use actually reaches the ranking, not "
            "just the diagnosis. Grazing management is admitted to the acceptable set "
            "alongside water harvesting because destocking is the standard first move on "
            "arid rangeland and the two are not competing answers. "
            "This site originally required an extrapolation flag and no longer does, "
            "which is worth stating rather than quietly dropping: the recommendations it "
            "produces rest on globally scoped FAO guidance, so there is no "
            "temperate-scoped evidence in the output for a scope check to fire on. The "
            "gap that made the requirement tempting is real and unaddressed, though. "
            "Every meta-analysis in this corpus was gathered on cropland, and nothing in "
            "the system flags cropland-derived evidence applied to rangeland, because "
            "scope checking reads climate and crop and not land use."
        ),
    )


def saline_alkaline_cropland() -> EvalSite:
    return EvalSite(
        site=SiteState(
            site_id="eval_alkaline",
            soil_organic_carbon_pct=_m(0.90, "%"),
            annual_rainfall_mm=_m(550, "mm"),
            ph=_m(8.8, "pH"),
            slope_pct=_m(2, "%"),
            land_use=_band("cropland", Confidence.HIGH),
            crop="wheat",
        ),
        expected_limiting_factor="soil_ph",
        expected_top_intervention_class=None,
        must_flag=["no_tier_1"],
        notes=(
            "pH 8.8 is outside the band in which phosphorus and micronutrients stay "
            "available, so alkalinity binds. No acceptable intervention set is given "
            "because the graph has exactly one edge into soil_ph and it raises pH, which "
            "is the wrong direction here: gypsum, elemental sulphur and acidifying "
            "amendments are not nodes in this graph at all. The honest expectation is "
            "that the system reports it cannot address the constraint, so no_tier_1 is "
            "required. Naming a top intervention would be scoring the system against a "
            "gap it should be admitting."
        ),
    )


def acid_humid_cropland() -> EvalSite:
    return EvalSite(
        site=SiteState(
            site_id="eval_acid_humid",
            soil_organic_carbon_pct=_m(1.10, "%"),
            annual_rainfall_mm=_m(1800, "mm"),
            ph=_m(5.2, "pH"),
            slope_pct=_m(3, "%"),
            land_use=_band("cropland", Confidence.HIGH),
            crop="rice",
        ),
        expected_limiting_factor="soil_ph",
        expected_top_intervention_class=["compost_application", "farmyard_manure"],
        must_flag=[],
        notes=(
            "pH 5.2 under 1800mm: acid enough to impair nutrient availability directly, "
            "and on flat ground so the erosion branch must not fire despite the rainfall. "
            "The pair with the alkaline site is the test that matters here, because pH "
            "has no fixed improvement direction and the two sites need opposite answers "
            "from the same variable."
        ),
    )


def high_carbon_temperate_analogue() -> EvalSite:
    return EvalSite(
        site=SiteState(
            site_id="eval_high_carbon",
            soil_organic_carbon_pct=_m(2.10, "%"),
            annual_rainfall_mm=_m(900, "mm"),
            ph=_m(6.6, "pH"),
            slope_pct=_m(2, "%"),
            land_use=_band("cropland", Confidence.HIGH),
            crop="barley",
        ),
        expected_limiting_factor="species_richness",
        expected_top_intervention_class=_HABITAT_STRUCTURE,
        must_flag=[],
        notes=(
            "Soil and water are both adequate, which is the case the other eleven sites "
            "do not cover. Where neither carbon nor moisture binds, biodiversity is "
            "constrained by habitat structure rather than by soil fertility, and a system "
            "that recommends manure here has a fixed opinion rather than a diagnosis."
        ),
    )


def fragmented_mosaic() -> EvalSite:
    return EvalSite(
        site=SiteState(
            site_id="eval_fragmented",
            soil_organic_carbon_pct=_m(1.20, "%"),
            annual_rainfall_mm=_m(1100, "mm"),
            ph=_m(6.5, "pH"),
            slope_pct=_m(2, "%"),
            land_use=_band("cropland", Confidence.HIGH),
            edge_density=_band("high", Confidence.HIGH),
            largest_patch_index=_m(8.0, "%"),
            crop="maize",
        ),
        expected_limiting_factor="habitat_connectivity",
        expected_top_intervention_class=["hedgerow_planting", "boundary_tree_planting"],
        must_flag=[],
        notes=(
            "High edge density with a small largest patch is fragmentation: the patches "
            "are there but they are not linked, so species cannot move between them "
            "however good the soil in each one is. Soil and water are deliberately "
            "adequate so that nothing upstream of connectivity can claim the diagnosis."
        ),
    )


def sparse_rainfall_only() -> EvalSite:
    return EvalSite(
        site=SiteState(
            site_id="eval_sparse_rainfall",
            annual_rainfall_mm=_m(340, "mm", Confidence.LOW),
        ),
        expected_limiting_factor=None,
        expected_top_intervention_class=None,
        must_flag=["sparse_input", "default_diagnosis"],
        notes=(
            "Rainfall alone does not identify a binding constraint. 340mm makes water a "
            "candidate, but with no carbon, pH or slope reading the site could equally be "
            "eroding or alkaline, so no expectation is set: the correct behaviour is to "
            "say what is missing and to ask, not to name a constraint. Scoring a "
            "diagnosis here would reward a guess."
        ),
    )


def sparse_coordinates_only() -> EvalSite:
    return EvalSite(
        site=SiteState(site_id="eval_sparse_coords", lat=17.85, lon=75.42),
        expected_limiting_factor=None,
        expected_top_intervention_class=None,
        must_flag=[],
        notes=(
            "Coordinates and nothing else. These are the committed cache coordinates, so "
            "the acquisition path runs offline and fills soil and climate from SoilGrids "
            "and NASA POWER. What is under test is whether the system reaches a diagnosis "
            "from a point on a map at all, and whether the acquired values are reported "
            "as map predictions rather than as measurements. No expectation is set for "
            "the constraint itself because it depends on what the APIs return, and "
            "pinning it here would be pinning the cache rather than the reasoning."
        ),
    )


def empty_site() -> EvalSite:
    return EvalSite(
        site=SiteState(site_id="eval_empty"),
        expected_limiting_factor=None,
        expected_top_intervention_class=None,
        must_flag=["sparse_input", "asks_clarifying_question", "default_diagnosis"],
        notes=(
            "Nothing but an identifier. There is no correct diagnosis, so the only "
            "correct behaviour is to ask for the measurement whose answer would change "
            "the advice most and to be explicit that anything said meanwhile rests on no "
            "measurements. A confident recommendation here is the failure mode this site "
            "exists to catch."
        ),
    )


def contradictory_input() -> EvalSite:
    return EvalSite(
        site=SiteState(
            site_id="eval_contradictory",
            soil_organic_carbon_pct=_m(2.50, "%"),
            annual_rainfall_mm=_m(180, "mm"),
            ph=_m(7.4, "pH"),
            slope_pct=_m(2, "%"),
            land_use=_band("cropland", Confidence.HIGH),
        ),
        expected_limiting_factor=None,
        expected_top_intervention_class=None,
        must_flag=["implausible_input"],
        notes=(
            "2.5% soil organic carbon under 180mm of rainfall is not a site, it is a "
            "typo. Arid systems do not fix enough carbon to hold that stock and do not "
            "have the moisture to stabilise it, so one of the two numbers is wrong and "
            "there is no way to tell which. The correct response is to say the pairing is "
            "implausible and ask which figure to trust. No diagnosis expectation is set, "
            "because every diagnosis from these inputs is reasoning from a number that "
            "should not have been accepted."
        ),
    )


def eval_sites() -> list[EvalSite]:
    """The twelve sites, in the order the summary reports them."""
    return [
        semi_arid_low_carbon_cropland(),
        steep_humid_cropland(),
        flat_sub_humid_cropland(),
        arid_grazing_land(),
        saline_alkaline_cropland(),
        acid_humid_cropland(),
        high_carbon_temperate_analogue(),
        fragmented_mosaic(),
        sparse_rainfall_only(),
        sparse_coordinates_only(),
        empty_site(),
        contradictory_input(),
    ]

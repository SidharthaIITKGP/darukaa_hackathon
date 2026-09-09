"""Core data contracts for the causal graph.

Pydantic v2 models only. No graph construction, no propagation logic here.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, field_validator, model_validator

_Z_BY_CI_LEVEL = {0.90: 1.6448536269514722, 0.95: 1.959963984540054}


class Distribution(BaseModel):
    """An effect size as a distribution, never a point value."""

    family: Literal["lognormal", "normal", "triangular"]
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    point: float | None = None

    @field_validator("ci_level")
    @classmethod
    def _validate_ci_level(cls, v: float) -> float:
        if v not in (0.90, 0.95):
            raise ValueError("ci_level must be 0.90 or 0.95")
        return v

    @model_validator(mode="after")
    def _validate_ci_order(self) -> "Distribution":
        if not self.ci_low < self.ci_high:
            raise ValueError("ci_low must be < ci_high")
        return self

    @model_validator(mode="after")
    def _validate_ci_low_positive(self) -> "Distribution":
        if self.family != "normal" and self.ci_low <= 0:
            raise ValueError("ci_low must be > 0 unless family is 'normal'")
        return self

    def spans_zero(self) -> bool:
        """True when the interval contains zero, meaning the direction of the
        effect is unresolved rather than merely imprecise. Only possible for
        the normal family: the others require ci_low > 0, and a negative-sign
        edge carries its direction in `sign` with a positive magnitude here.
        """
        return self.ci_low < 0 < self.ci_high

    def params(self) -> dict[str, float]:
        """Derive distribution parameters from the published interval."""
        z = _Z_BY_CI_LEVEL[self.ci_level]
        if self.family == "lognormal":
            if self.ci_low <= 0:
                raise ValueError("lognormal requires ci_low > 0")
            if self.point is not None:
                mu = math.log(self.point)
                sigma = max(math.log(self.ci_high) - mu, mu - math.log(self.ci_low)) / z
            else:
                mu = math.log(math.sqrt(self.ci_low * self.ci_high))
                sigma = (math.log(self.ci_high) - math.log(self.ci_low)) / (2 * z)
            return {"mu": mu, "sigma": sigma}
        if self.family == "normal":
            mean = (self.ci_low + self.ci_high) / 2
            sigma = (self.ci_high - self.ci_low) / (2 * z)
            return {"mean": mean, "sigma": sigma}
        # triangular
        mode = self.point if self.point is not None else (self.ci_low + self.ci_high) / 2
        return {"low": self.ci_low, "mode": mode, "high": self.ci_high}

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        p = self.params()
        if self.family == "lognormal":
            return rng.lognormal(mean=p["mu"], sigma=p["sigma"], size=n)
        if self.family == "normal":
            return rng.normal(loc=p["mean"], scale=p["sigma"], size=n)
        return rng.triangular(left=p["low"], mode=p["mode"], right=p["high"], size=n)


class EffectMetric(str, Enum):
    LRR = "LRR"
    PERCENT_CHANGE = "percent_change"
    HEDGES_D = "HedgesD"
    HEDGES_G = "HedgesG"
    COHENS_D = "CohensD"
    ABSOLUTE = "absolute"


def to_relative_change(value: float, metric: EffectMetric) -> float:
    """Convert a metric value to a proportional change.

    Standardised mean differences (Hedges' d/g, Cohen's d) are not
    proportional changes and must not be silently coerced.
    """
    if metric == EffectMetric.LRR:
        return math.exp(value) - 1
    if metric == EffectMetric.PERCENT_CHANGE:
        return value
    if metric in (EffectMetric.HEDGES_D, EffectMetric.HEDGES_G, EffectMetric.COHENS_D):
        raise NotImplementedError(
            f"{metric.value} is a standardised mean difference, not a proportional change"
        )
    raise NotImplementedError(f"no relative-change conversion defined for {metric.value}")


class Confidence(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class Provenance(str, Enum):
    USER_STATED = "user_stated"
    API_SOILGRIDS = "api_soilgrids"
    API_NASA_POWER = "api_nasa_power"
    API_GBIF = "api_gbif"
    RASTER_GSOCSEQ = "raster_gsocseq"
    COMPUTED = "computed"
    INFERRED_DEFAULT = "inferred_default"


class Measurement(BaseModel):
    """One observed variable. Value may be a number or a qualitative band."""

    value: float | None = None
    band: str | None = None
    unit: str | None = None
    provenance: Provenance
    confidence: Confidence
    uncertainty: float | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _validate_value_xor_band(self) -> "Measurement":
        if (self.value is None) == (self.band is None):
            raise ValueError("exactly one of value or band must be set")
        return self


class SiteState(BaseModel):
    site_id: str
    lat: float | None = None
    lon: float | None = None
    soil_organic_carbon_pct: Measurement | None = None
    ph: Measurement | None = None
    clay_pct: Measurement | None = None
    sand_pct: Measurement | None = None
    nitrogen: Measurement | None = None
    bulk_density: Measurement | None = None
    annual_rainfall_mm: Measurement | None = None
    mean_temperature_c: Measurement | None = None
    land_use: Measurement | None = None
    crop: str | None = None
    slope_pct: Measurement | None = None
    observed_species_richness: Measurement | None = None
    edge_density: Measurement | None = None
    largest_patch_index: Measurement | None = None

    def _measurement_field_names(self) -> list[str]:
        return [
            name
            for name, field in self.model_fields.items()
            if field.annotation == (Measurement | None)
        ]

    def known(self) -> list[str]:
        return [name for name in self._measurement_field_names() if getattr(self, name) is not None]

    def missing(self) -> list[str]:
        return [name for name in self._measurement_field_names() if getattr(self, name) is None]

    def climate_zone(self) -> str | None:
        if self.annual_rainfall_mm is None or self.annual_rainfall_mm.value is None:
            return None
        rainfall = self.annual_rainfall_mm.value
        if rainfall < 250:
            return "arid"
        if rainfall < 500:
            return "semi_arid"
        if rainfall < 1000:
            return "sub_humid"
        return "humid"


# Thresholds for the implausibility checks below. Modelling assumptions, not
# published constants, and deliberately loose: the job is to catch a figure
# that is almost certainly a transcription error, not to police the edges of
# what soils do. A site that trips one of these may still be real, which is
# why the output is a question and never a rejection.
IMPLAUSIBLE_SOC_PCT = 1.5
IMPLAUSIBLE_SOC_RAINFALL_MM = 300.0
IMPLAUSIBLE_ACID_PH = 5.5
IMPLAUSIBLE_ACID_RAINFALL_MM = 400.0
# A landscape-scale species count above which "high richness" is the claim
# being made. Coarse by necessity, since richness has no natural scale
# without a taxon and a survey protocol, so a qualitative band is preferred
# where the site carries one.
IMPLAUSIBLE_RICHNESS_COUNT = 100.0
IMPLAUSIBLE_EDGE_DENSITY = 0.7
IMPLAUSIBLE_LARGEST_PATCH_INDEX = 20.0

_HIGH_BANDS = ("high", "very high")
_LOW_BANDS = ("low", "very low")


def _numeric(measurement: "Measurement | None") -> float | None:
    return measurement.value if measurement is not None else None


def _is_high(measurement: "Measurement | None", threshold: float) -> bool:
    if measurement is None:
        return False
    if measurement.band is not None:
        return measurement.band.lower() in _HIGH_BANDS
    return measurement.value is not None and measurement.value > threshold


def _is_low(measurement: "Measurement | None", threshold: float) -> bool:
    if measurement is None:
        return False
    if measurement.band is not None:
        return measurement.band.lower() in _LOW_BANDS
    return measurement.value is not None and measurement.value < threshold


def implausible_combinations(site: SiteState) -> list[str]:
    """Physically inconsistent pairs of measurements, phrased as questions.

    A site can be internally contradictory while every field in it is
    individually valid, and a system that reasons about sites has to notice.
    2.5% soil organic carbon under 180mm of rainfall is the standing example:
    both numbers are ordinary on their own, and together they describe a
    place that does not exist without irrigation or an amendment history.

    Each returned string is a question rather than a verdict, because the
    combination is unusual rather than impossible and the person who stood in
    the field knows things this system does not. Silence would be worse than
    either: it would mean reasoning downstream of a number that should have
    been checked first.

    Returns an empty list when nothing conflicts, which is the normal case.
    """
    notes: list[str] = []

    soc = _numeric(site.soil_organic_carbon_pct)
    rainfall = _numeric(site.annual_rainfall_mm)
    ph = _numeric(site.ph)

    if (
        soc is not None
        and rainfall is not None
        and soc > IMPLAUSIBLE_SOC_PCT
        and rainfall < IMPLAUSIBLE_SOC_RAINFALL_MM
    ):
        notes.append(
            f"A soil organic carbon of {soc:g}% under {rainfall:g}mm annual rainfall is "
            f"unusual, since carbon accrual at that level normally requires more biomass "
            f"production than that rainfall supports. Is the site irrigated, does it "
            f"waterlog seasonally, or has it had heavy organic amendment? Or was that "
            f"figure from a different depth or a different plot?"
        )

    if (
        ph is not None
        and rainfall is not None
        and ph < IMPLAUSIBLE_ACID_PH
        and rainfall < IMPLAUSIBLE_ACID_RAINFALL_MM
    ):
        notes.append(
            f"A pH of {ph:g} under {rainfall:g}mm annual rainfall is unusual. Acidification "
            f"is normally driven by base cations leaching out of the profile, which takes "
            f"more water than falls here, so dry soils tend to run neutral to alkaline. Is "
            f"there an acidifying input such as heavy ammonium fertiliser or mine spoil, or "
            f"could the reading be from a different plot?"
        )

    if (
        _is_high(site.observed_species_richness, IMPLAUSIBLE_RICHNESS_COUNT)
        and _is_high(site.edge_density, IMPLAUSIBLE_EDGE_DENSITY)
        and _is_low(site.largest_patch_index, IMPLAUSIBLE_LARGEST_PATCH_INDEX)
    ):
        notes.append(
            "High species richness in a severely fragmented mosaic, with high edge density "
            "and no large remaining patch, is unusual and warrants verification. Fragmented "
            "landscapes usually lose interior and area-sensitive species first. Does the "
            "richness figure come from a survey on this land, or from records pooled over a "
            "wider area that includes habitat the site itself does not have?"
        )

    return notes


class Conditions(BaseModel):
    """Preconditions under which an edge holds. None means unconstrained."""

    rainfall_mm: tuple[float, float] | None = None
    soc_pct: tuple[float, float] | None = None
    ph: tuple[float, float] | None = None
    clay_pct: tuple[float, float] | None = None
    slope_pct: tuple[float, float] | None = None
    land_use: list[str] | None = None
    climate_zone: list[str] | None = None

    def satisfaction(self, site: SiteState) -> float:
        """Fraction of specified conditions the site meets.

        Unknown site values count as 0.5. Returns 1.0 if no conditions are
        specified.
        """
        checks: list[tuple[str, Measurement | None]] = [
            ("rainfall_mm", site.annual_rainfall_mm),
            ("soc_pct", site.soil_organic_carbon_pct),
            ("ph", site.ph),
            ("clay_pct", site.clay_pct),
            ("slope_pct", site.slope_pct),
        ]

        scores: list[float] = []
        for field_name, measurement in checks:
            bounds = getattr(self, field_name)
            if bounds is None:
                continue
            if measurement is None or measurement.value is None:
                scores.append(0.5)
                continue
            low, high = bounds
            scores.append(1.0 if low <= measurement.value <= high else 0.0)

        if self.land_use is not None:
            if site.land_use is None or site.land_use.band is None:
                scores.append(0.5)
            else:
                scores.append(1.0 if site.land_use.band in self.land_use else 0.0)

        if self.climate_zone is not None:
            zone = site.climate_zone()
            if zone is None:
                scores.append(0.5)
            else:
                scores.append(1.0 if zone in self.climate_zone else 0.0)

        if not scores:
            return 1.0
        return sum(scores) / len(scores)


class EvidenceStrength(str, Enum):
    META_ANALYSIS = "meta_analysis"
    MULTI_SITE = "multi_site"
    SINGLE_SITE = "single_site"
    MECHANISTIC = "mechanistic"
    EXPERT = "expert"


class EvidenceRef(BaseModel):
    """A citation attached to an edge, plus what that citation actually does
    for the edge.

    role is required and must be chosen deliberately, because counting
    references is not the same as counting agreement:
      primary       this source establishes the relationship.
      corroborating an independent source agreeing on DIRECTION. It may
                    disagree on magnitude; that belongs in the interval.
      contradicting a source finding no effect, or the opposite direction.
      critique      a methodological comment on another study, not an
                    independent measurement of the relationship.
    Only primary and corroborating refs count as agreement. A critique of a
    contradicting study is not corroboration of the effect.
    """

    source_id: str
    role: Literal["primary", "corroborating", "contradicting", "critique"]
    pages: list[int] | None = None
    chunk_ids: list[str] | None = None
    quote: str | None = None
    note: str | None = None

    @field_validator("quote")
    @classmethod
    def _validate_quote_length(cls, v: str | None) -> str | None:
        if v is not None and len(v.split()) > 25:
            raise ValueError("quote must be 25 words or fewer")
        return v


class CausalEdge(BaseModel):
    source: str
    target: str
    sign: Literal["+", "-"]
    effect: Distribution
    metric: EffectMetric
    lag_years: tuple[float, float]
    conditions: Conditions = Conditions()
    evidence: list[EvidenceRef]
    strength: EvidenceStrength
    confidence: Confidence
    mechanism: str
    contested: bool = False
    contested_note: str | None = None

    @field_validator("evidence")
    @classmethod
    def _validate_evidence_nonempty(cls, v: list[EvidenceRef]) -> list[EvidenceRef]:
        if not v:
            raise ValueError("evidence must be non-empty")
        return v

    @field_validator("lag_years")
    @classmethod
    def _validate_lag_order(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not v[0] <= v[1]:
            raise ValueError("lag_years[0] must be <= lag_years[1]")
        return v

    @model_validator(mode="after")
    def _validate_contested_note(self) -> "CausalEdge":
        if self.contested and not self.contested_note:
            raise ValueError("contested_note is required when contested is True")
        return self

    @model_validator(mode="after")
    def _validate_sign_magnitude(self) -> "CausalEdge":
        if self.sign == "-" and self.effect.ci_low < 0:
            raise ValueError(
                "effect.ci_low must be a positive magnitude when sign is '-'; "
                "direction belongs in sign alone, not in the interval"
            )
        return self


def validate_source_ids(edges: list[CausalEdge], sources_yaml_path: str) -> None:
    """Raise ValueError listing any source_id not registered in sources.yaml."""
    with open(sources_yaml_path) as f:
        registry = yaml.safe_load(f)
    registered = set(registry.get("sources", {}).keys())

    unregistered: set[str] = set()
    for edge in edges:
        for ref in edge.evidence:
            if ref.source_id not in registered:
                unregistered.add(ref.source_id)

    if unregistered:
        raise ValueError(
            f"unregistered source_id(s) not in {sources_yaml_path}: {sorted(unregistered)}"
        )

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

    def params(self) -> dict[str, float]:
        """Derive distribution parameters from the published interval."""
        z = _Z_BY_CI_LEVEL[self.ci_level]
        if self.family == "lognormal":
            if self.ci_low <= 0:
                raise ValueError("lognormal requires ci_low > 0")
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
    source_id: str
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

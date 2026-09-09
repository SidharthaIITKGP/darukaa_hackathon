import math

import numpy as np
import pytest
from pydantic import ValidationError

from src.graph.schemas import (
    CausalEdge,
    Conditions,
    Confidence,
    Distribution,
    EffectMetric,
    EvidenceRef,
    EvidenceStrength,
    Measurement,
    Provenance,
    SiteState,
    to_relative_change,
    validate_source_ids,
)


def test_distribution_lognormal_params_and_sample():
    dist = Distribution(family="lognormal", ci_low=0.049, ci_high=0.096)
    params = dist.params()
    # mu = log(sqrt(ci_low * ci_high)) = log(sqrt(0.049*0.096)) ~= -2.6797.
    # The prompt's worked example (-3.0387) is inconsistent with its own
    # median target below: exp(-3.0387) ~= 0.0479, not 0.0686. exp(-2.6797)
    # ~= 0.0686 does match, confirming -2.6797 is the correct value.
    assert params["mu"] == pytest.approx(-2.6797, abs=1e-3)

    rng = np.random.default_rng(42)
    samples = dist.sample(10000, rng)
    median = np.median(samples)
    assert median == pytest.approx(0.0686, rel=0.05)


def test_to_relative_change_lrr():
    assert to_relative_change(0.18, EffectMetric.LRR) == pytest.approx(0.1972, abs=1e-3)


def test_to_relative_change_hedges_d_raises():
    with pytest.raises(NotImplementedError):
        to_relative_change(0.5, EffectMetric.HEDGES_D)


def test_measurement_both_value_and_band_raises():
    with pytest.raises(ValidationError):
        Measurement(
            value=1.0,
            band="high",
            provenance=Provenance.USER_STATED,
            confidence=Confidence.HIGH,
        )


def test_measurement_neither_value_nor_band_raises():
    with pytest.raises(ValidationError):
        Measurement(provenance=Provenance.USER_STATED, confidence=Confidence.HIGH)


def test_site_state_climate_zone_semi_arid():
    site = SiteState(
        site_id="site1",
        annual_rainfall_mm=Measurement(
            value=340, provenance=Provenance.USER_STATED, confidence=Confidence.HIGH
        ),
    )
    assert site.climate_zone() == "semi_arid"


def test_conditions_satisfaction_partial_unknown():
    site = SiteState(
        site_id="site1",
        annual_rainfall_mm=Measurement(
            value=340, provenance=Provenance.USER_STATED, confidence=Confidence.HIGH
        ),
    )
    conditions = Conditions(rainfall_mm=(250, 500), land_use=["cropland"])
    assert conditions.satisfaction(site) == pytest.approx(0.75)


def _valid_edge_kwargs(**overrides):
    kwargs = dict(
        source="legume_cover_crop",
        target="soil_organic_carbon",
        sign="+",
        effect=Distribution(family="lognormal", ci_low=0.049, ci_high=0.096),
        metric=EffectMetric.LRR,
        lag_years=(1.0, 3.0),
        evidence=[EvidenceRef(source_id="Joshi_2023_covercrops_SOC", role="primary")],
        strength=EvidenceStrength.META_ANALYSIS,
        confidence=Confidence.MODERATE,
        mechanism="Cover crop residue inputs increase soil carbon.",
    )
    kwargs.update(overrides)
    return kwargs


def test_causal_edge_empty_evidence_raises():
    with pytest.raises(ValidationError):
        CausalEdge(**_valid_edge_kwargs(evidence=[]))


def test_causal_edge_contested_without_note_raises():
    with pytest.raises(ValidationError):
        CausalEdge(**_valid_edge_kwargs(contested=True))


def test_validate_source_ids_fabricated_raises():
    edge = CausalEdge(
        **_valid_edge_kwargs(evidence=[EvidenceRef(source_id="Fabricated_2024", role="primary")])
    )
    with pytest.raises(ValueError):
        validate_source_ids([edge], "sources.yaml")

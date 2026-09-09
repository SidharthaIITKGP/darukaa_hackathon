"""End-to-end vertical slice: site state in, diagnosed recommendations out.

No retrieval, no agent loop, no interface. This module exists to prove that
the curated causal graph plus the Monte Carlo propagation engine already
produce the output shape the brief asks for: a diagnosis, a ranked set of
interventions, a quantified effect per environmental variable with an
interval and a time horizon, a confidence statement on two axes, real
citations, caveats, and sequencing.

The formatting lives in src.agents.render, which the agent graph also uses,
so the conversational path and this standalone path cannot drift apart.
Every number printed comes from src.graph.propagate. Every citation string
is read verbatim from sources.yaml by source_id.
"""

from __future__ import annotations

from src.agents.render import render_report
from src.graph.edges import build_graph
from src.graph.propagate import limiting_factor, rank_interventions
from src.graph.schemas import Confidence, Measurement, Provenance, SiteState


def run(site: SiteState, top_n: int = 4) -> str:
    """Diagnose a site and render the top_n interventions as a text report."""
    graph = build_graph()
    limiting_var, limiting_why = limiting_factor(site)
    ranked = rank_interventions(graph, site)
    return render_report(graph, site, ranked, limiting_var, limiting_why, top_n=top_n)


def _deccan_semiarid() -> SiteState:
    return SiteState(
        site_id="deccan_semiarid",
        lat=17.85,
        lon=75.42,
        soil_organic_carbon_pct=Measurement(
            value=0.35, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            value=340, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        ph=Measurement(
            value=8.1, unit="pH", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
        slope_pct=Measurement(
            value=5, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
    )


def _indo_gangetic() -> SiteState:
    return SiteState(
        site_id="indo_gangetic",
        lat=29.15,
        lon=76.32,
        soil_organic_carbon_pct=Measurement(
            value=0.55, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            value=620, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        ph=Measurement(
            value=7.6, unit="pH", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
        slope_pct=Measurement(
            value=1, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
    )


def _western_ghats() -> SiteState:
    return SiteState(
        site_id="western_ghats",
        lat=15.60,
        lon=74.05,
        soil_organic_carbon_pct=Measurement(
            value=1.20, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            value=2100, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        ph=Measurement(
            value=6.2, unit="pH", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
        slope_pct=Measurement(
            value=12, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
    )


DEMO_SITES: dict[str, SiteState] = {
    "deccan_semiarid": _deccan_semiarid(),
    "indo_gangetic": _indo_gangetic(),
    "western_ghats": _western_ghats(),
}


if __name__ == "__main__":
    print(run(DEMO_SITES["deccan_semiarid"], top_n=4))
    print()
    for site_id in ("indo_gangetic", "western_ghats"):
        print(run(DEMO_SITES[site_id], top_n=1))
        print()

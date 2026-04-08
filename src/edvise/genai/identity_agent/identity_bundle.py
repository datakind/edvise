"""
Single handoff bundle for IdentityAgent LLM outputs (pass 1 grain + pass 2 term).

Schema Mapping Agent does **not** consume this type directly; it consumes the frozen
enriched schema contract validated by :class:`EnrichedSchemaContractForSMA`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from edvise.genai.identity_agent.grain_inference.schemas import InstitutionGrainContract
from edvise.genai.identity_agent.term_normalization.schemas import InstitutionTermContract


class InstitutionIdentityContract(BaseModel):
    """
    One institution, one artifact: grain (pass 1) and term (pass 2) envelopes together.

    Use when persisting or passing IdentityAgent results between steps; build the SMA-facing
    frozen contract via :func:`~edvise.genai.identity_agent.execution.contract_builder.build_schema_contract_from_grain_contracts`
    and enrichment helpers.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    institution_id: str
    grain: InstitutionGrainContract
    term: InstitutionTermContract

    @model_validator(mode="after")
    def _envelopes_match(self) -> "InstitutionIdentityContract":
        if self.grain.institution_id != self.institution_id:
            raise ValueError(
                f"grain.institution_id {self.grain.institution_id!r} != "
                f"institution_id {self.institution_id!r}"
            )
        if self.term.institution_id != self.institution_id:
            raise ValueError(
                f"term.institution_id {self.term.institution_id!r} != "
                f"institution_id {self.institution_id!r}"
            )
        g_keys = set(self.grain.datasets.keys())
        t_keys = set(self.term.datasets.keys())
        if g_keys != t_keys:
            raise ValueError(
                "grain.datasets and term.datasets must have the same keys; "
                f"grain={sorted(g_keys)} term={sorted(t_keys)}"
            )
        return self


def institution_identity_contract_from_parts(
    grain: InstitutionGrainContract,
    term: InstitutionTermContract,
) -> InstitutionIdentityContract:
    """Build :class:`InstitutionIdentityContract` when pass-1 and pass-2 share ``institution_id``."""
    if grain.institution_id != term.institution_id:
        raise ValueError(
            f"grain institution_id {grain.institution_id!r} != "
            f"term institution_id {term.institution_id!r}"
        )
    return InstitutionIdentityContract(
        institution_id=grain.institution_id,
        grain=grain,
        term=term,
    )


__all__ = [
    "InstitutionIdentityContract",
    "institution_identity_contract_from_parts",
]

"""
SEC filings — shallow references only (no XBRL or full 10-K parsing here).

Intended flow later:
  1. For each symbol, resolve CIK (SEC company id).
  2. Query EDGAR submissions (or a vendor) for recent 10-K / 10-Q / 8-K metadata.
  3. Return FilingRef rows (form, date, accession, URL) like we do below.

The LLM (or offline fallback) only sees these references — not the full filing text —
unless you add a separate "fetch filing body" step later.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

from stockbot.models import FilingRef


def fetch_filings_refs(
    symbols: Iterable[str],
    as_of: date | None = None,
) -> list[FilingRef]:
    """
    Stub: one placeholder filing per symbol so prompts and audits have a filings section.

    Replace with real EDGAR/submissions JSON when you are ready; keep returning FilingRef
    so the rest of the pipeline stays unchanged.
    """
    as_of = as_of or date.today()
    out: list[FilingRef] = []
    for sym in symbols:
        sym = sym.upper()
        # EDGAR search URL is a handy human link until you store real accession URLs.
        search = f"https://www.sec.gov/edgar/search/#/q={sym}"
        out.append(
            FilingRef(
                symbol=sym,
                form_type="10-Q",
                filed_at=as_of - timedelta(days=45),
                accession=f"0000000000-{sym}-STUB",
                url=search,
            )
        )
    return out

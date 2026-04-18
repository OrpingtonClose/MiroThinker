# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Government & legal intelligence tools for the Strands research agent.

Accesses publicly available but systematically under-reported government
databases. These contain suppressed drug trial results, adverse events
that pharma companies minimize, corporate fraud buried in SEC filings,
and unsealed court documents from pharmaceutical/environmental lawsuits.

Sources:
  1. ClinicalTrials.gov — graveyard of suppressed drug trials (v2 API, free)
  2. OpenFDA — FAERS adverse events, drug recalls, device incidents (free)
  3. CourtListener/RECAP — unsealed court documents, PACER for free (free)
  4. SEC EDGAR — corporate filings, insider trading, fraud in footnotes (free)
  5. ICIJ Offshore Leaks — Panama/Paradise/Pandora Papers entities (free CSV)

All free APIs — no keys required.
"""

from __future__ import annotations

import json
import logging
import os

from strands import tool

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ClinicalTrials.gov v2 API — suppressed drug trial data
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_clinical_trials(
    query: str,
    status: str = "",
    phase: str = "",
    max_results: int = 10,
) -> str:
    """Search ClinicalTrials.gov for clinical trials. Free, no API key.

    The graveyard of suppressed drug trials. Pharma companies are legally
    required to register trials but often bury negative results here by
    never publishing them in journals. Contains terminated trials, trials
    with unreported results, and trials with suspicious status changes.

    Args:
        query: Search query (drug name, condition, sponsor, etc.).
        status: Filter by status: "COMPLETED", "TERMINATED", "SUSPENDED",
                "WITHDRAWN", "NOT_YET_RECRUITING", "RECRUITING",
                "ACTIVE_NOT_RECRUITING". Leave empty for all.
        phase: Filter by phase: "PHASE1", "PHASE2", "PHASE3", "PHASE4",
               "EARLY_PHASE1", "NA". Leave empty for all.
        max_results: Maximum results (default 10, max 100).

    Returns:
        Formatted list of clinical trials with status, sponsor, and results.
    """
    import httpx

    params: dict = {
        "query.term": query,
        "pageSize": min(max_results, 100),
        "format": "json",
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,StartDate,"
                  "CompletionDate,LeadSponsorName,EnrollmentCount,"
                  "BriefSummary,ResultsFirstSubmitDate,HasResults,"
                  "WhyStopped,Condition,InterventionName",
        "sort": "LastUpdatePostDate:desc",
    }
    if status:
        params["query.term"] += f" AND AREA[OverallStatus]{status}"
    if phase:
        params["filter.advanced"] = f"AREA[Phase]{phase}"

    try:
        resp = httpx.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params=params,
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] ClinicalTrials.gov search failed: {exc}"

    studies = data.get("studies", [])
    if not studies:
        return f"No clinical trials found for: {query}"

    formatted = [f"**Clinical trials for: {query}** ({len(studies)} results)\n"]
    for i, study in enumerate(studies[:max_results], 1):
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        design = proto.get("designModule", {})
        desc = proto.get("descriptionModule", {})
        results_mod = study.get("resultsSection", {})

        nct_id = ident.get("nctId", "")
        title = ident.get("briefTitle", "Unknown")
        overall_status = status_mod.get("overallStatus", "")
        start_date = status_mod.get("startDateStruct", {}).get("date", "")
        completion = status_mod.get("completionDateStruct", {}).get("date", "")
        why_stopped = status_mod.get("whyStopped", "")

        sponsor = ""
        lead_sponsor = sponsor_mod.get("leadSponsor", {})
        if lead_sponsor:
            sponsor = lead_sponsor.get("name", "")

        phases = design.get("phases", [])
        phase_str = ", ".join(phases) if phases else "N/A"

        enrollment = design.get("enrollmentInfo", {}).get("count", "")
        has_results = bool(results_mod)
        brief = (desc.get("briefSummary") or "")[:200]

        # Conditions and interventions
        conditions_mod = proto.get("conditionsModule", {})
        conditions = conditions_mod.get("conditions", [])
        conditions_str = ", ".join(conditions[:3]) if conditions else ""

        interventions_mod = proto.get("armsInterventionsModule", {})
        interventions = interventions_mod.get("interventions", [])
        intervention_names = [iv.get("name", "") for iv in interventions[:3]]
        interventions_str = ", ".join(intervention_names) if intervention_names else ""

        status_flag = ""
        if overall_status in ("TERMINATED", "SUSPENDED", "WITHDRAWN"):
            status_flag = "⚠ "
        results_flag = " [HAS RESULTS]" if has_results else " [NO RESULTS POSTED]"

        formatted.append(
            f"  {i}. {status_flag}**{title}** ({nct_id})\n"
            f"     Status: {overall_status}{results_flag}\n"
            f"     Phase: {phase_str} | Enrollment: {enrollment}\n"
            f"     Sponsor: {sponsor}\n"
            f"     Dates: {start_date} → {completion}\n"
            + (f"     Conditions: {conditions_str}\n" if conditions_str else "")
            + (f"     Interventions: {interventions_str}\n" if interventions_str else "")
            + (f"     WHY STOPPED: {why_stopped}\n" if why_stopped else "")
            + (f"     {brief}..." if brief else "")
        )

    return "\n\n".join(formatted)


@tool
def get_trial_results(nct_id: str) -> str:
    """Get full results for a specific clinical trial by NCT ID.

    Args:
        nct_id: ClinicalTrials.gov identifier (e.g. "NCT04280705").

    Returns:
        Full trial details including results if posted.
    """
    import httpx

    try:
        resp = httpx.get(
            f"https://clinicaltrials.gov/api/v2/studies/{nct_id}",
            params={"format": "json"},
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        resp.raise_for_status()
        study = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Failed to get trial {nct_id}: {exc}"

    proto = study.get("protocolSection", {})
    results = study.get("resultsSection", {})
    ident = proto.get("identificationModule", {})
    desc = proto.get("descriptionModule", {})
    status_mod = proto.get("statusModule", {})
    eligibility = proto.get("eligibilityModule", {})

    title = ident.get("briefTitle", "Unknown")
    official_title = ident.get("officialTitle", "")
    summary = desc.get("briefSummary", "")
    detailed = desc.get("detailedDescription", "")
    overall_status = status_mod.get("overallStatus", "")
    why_stopped = status_mod.get("whyStopped", "")

    output = [
        f"**{title}** ({nct_id})\n",
        f"Official title: {official_title}\n" if official_title else "",
        f"Status: {overall_status}\n",
    ]
    if why_stopped:
        output.append(f"**WHY STOPPED:** {why_stopped}\n")
    if summary:
        output.append(f"\n**Summary:**\n{summary}\n")
    if detailed:
        output.append(f"\n**Detailed Description:**\n{detailed[:1000]}\n")

    # Results section
    if results:
        participant_flow = results.get("participantFlowModule", {})
        baseline = results.get("baselineCharacteristicsModule", {})
        outcome_measures = results.get("outcomeMeasuresModule", {})
        adverse_events = results.get("adverseEventsModule", {})

        output.append("\n--- RESULTS ---\n")

        if participant_flow:
            groups = participant_flow.get("groups", [])
            if groups:
                output.append("**Participant Flow:**")
                for g in groups[:5]:
                    output.append(f"  - {g.get('title', '')}: {g.get('description', '')[:100]}")

        if outcome_measures:
            measures = outcome_measures.get("outcomeMeasures", [])
            if measures:
                output.append("\n**Outcome Measures:**")
                for m in measures[:5]:
                    output.append(
                        f"  - {m.get('title', '')}: {m.get('description', '')[:200]}"
                    )

        if adverse_events:
            output.append("\n**Adverse Events:**")
            freq = adverse_events.get("frequencyThreshold", "")
            output.append(f"  Reporting threshold: {freq}%")
            serious = adverse_events.get("seriousEvents", [])
            other = adverse_events.get("otherEvents", [])
            output.append(f"  Serious events reported: {len(serious)}")
            output.append(f"  Other events reported: {len(other)}")
    else:
        output.append("\n**No results posted for this trial.**")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════
# OpenFDA — Adverse events, drug recalls, device incidents
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_fda_adverse_events(
    drug_name: str = "",
    reaction: str = "",
    max_results: int = 10,
    serious: bool = False,
) -> str:
    """Search FDA Adverse Event Reporting System (FAERS). Free, no key needed.

    Contains reports of adverse drug reactions that pharma companies are
    legally required to submit but systematically under-report. Includes
    deaths, hospitalizations, and life-threatening events.

    Args:
        drug_name: Drug brand or generic name to search.
        reaction: Adverse reaction to search for (e.g. "death", "liver failure").
        max_results: Maximum results (default 10, max 100).
        serious: If True, only return serious events (death, hospitalization, etc.).

    Returns:
        Formatted list of adverse event reports.
    """
    import httpx

    search_terms = []
    if drug_name:
        search_terms.append(f'patient.drug.medicinalproduct:"{drug_name}"')
    if reaction:
        search_terms.append(f'patient.reaction.reactionmeddrapt:"{reaction}"')
    if serious:
        search_terms.append("serious:1")

    if not search_terms:
        return "[TOOL_ERROR] Provide at least drug_name or reaction."

    search_query = "+AND+".join(search_terms)

    try:
        resp = httpx.get(
            "https://api.fda.gov/drug/event.json",
            params={
                "search": search_query,
                "limit": min(max_results, 100),
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] OpenFDA adverse events search failed: {exc}"

    results = data.get("results", [])
    total = data.get("meta", {}).get("results", {}).get("total", 0)

    if not results:
        return f"No FDA adverse events found for: {drug_name or reaction}"

    formatted = [
        f"**FDA Adverse Events** (showing {len(results)} of {total:,} total)\n"
        f"Drug: {drug_name or 'any'} | Reaction: {reaction or 'any'}\n"
    ]

    for i, event in enumerate(results[:max_results], 1):
        report_date = event.get("receiptdate", "")
        if report_date and len(report_date) == 8:
            report_date = f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:]}"

        is_serious = event.get("serious", 0)
        death = event.get("seriousnessdeath", 0)
        hospitalized = event.get("seriousnesshospitalization", 0)
        life_threat = event.get("seriousnesslifethreatening", 0)

        severity_flags = []
        if death:
            severity_flags.append("DEATH")
        if life_threat:
            severity_flags.append("LIFE-THREATENING")
        if hospitalized:
            severity_flags.append("HOSPITALIZED")
        severity_str = " | ".join(severity_flags) if severity_flags else "non-serious"

        # Patient info
        patient = event.get("patient", {})
        age = patient.get("patientagegroup", "")
        sex = {"1": "Male", "2": "Female"}.get(str(patient.get("patientsex", "")), "")

        # Drugs
        drugs = patient.get("drug", [])
        drug_names = [d.get("medicinalproduct", "") for d in drugs[:5] if d.get("medicinalproduct")]

        # Reactions
        reactions = patient.get("reaction", [])
        reaction_names = [r.get("reactionmeddrapt", "") for r in reactions[:5]]

        formatted.append(
            f"  {i}. **{severity_str}** (reported: {report_date})\n"
            f"     Patient: {sex} {age}\n"
            f"     Drugs: {', '.join(drug_names)}\n"
            f"     Reactions: {', '.join(reaction_names)}"
        )

    return "\n\n".join(formatted)


@tool
def search_fda_recalls(
    query: str,
    max_results: int = 10,
) -> str:
    """Search FDA drug recalls and enforcement actions. Free, no key needed.

    Args:
        query: Search query (drug name, company, reason).
        max_results: Maximum results (default 10).

    Returns:
        Formatted list of FDA recalls with reasons and classifications.
    """
    import httpx

    try:
        resp = httpx.get(
            "https://api.fda.gov/drug/enforcement.json",
            params={
                "search": query,
                "limit": min(max_results, 100),
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] FDA recalls search failed: {exc}"

    results = data.get("results", [])
    if not results:
        return f"No FDA recalls found for: {query}"

    formatted = [f"**FDA Drug Recalls for: {query}** ({len(results)} results)\n"]
    for i, recall in enumerate(results[:max_results], 1):
        classification = recall.get("classification", "")
        reason = recall.get("reason_for_recall", "")
        product = recall.get("product_description", "")[:200]
        company = recall.get("recalling_firm", "")
        date = recall.get("report_date", "")
        status = recall.get("status", "")
        voluntary = recall.get("voluntary_mandated", "")

        class_flag = ""
        if "I" in classification:
            class_flag = "⚠ "  # Class I = most serious

        formatted.append(
            f"  {i}. {class_flag}**{classification}** — {company}\n"
            f"     Product: {product}\n"
            f"     Reason: {reason}\n"
            f"     Date: {date} | Status: {status} | {voluntary}"
        )

    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# CourtListener / RECAP — Free PACER court documents
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_court_opinions(
    query: str,
    court: str = "",
    max_results: int = 10,
) -> str:
    """Search CourtListener for court opinions and legal documents. Free.

    Provides free access to PACER documents (normally $0.10/page).
    Contains unsealed pharmaceutical lawsuits, environmental violations,
    patent disputes, and government overreach cases.

    Args:
        query: Search query (case name, topic, party name).
        court: Court abbreviation filter (e.g. "scotus", "ca9", "nysd").
               Leave empty for all courts.
        max_results: Maximum results (default 10).

    Returns:
        Formatted list of court opinions with citations and links.
    """
    import httpx

    params: dict = {
        "q": query,
        "page_size": min(max_results, 20),
        "order_by": "score desc",
    }
    if court:
        params["court"] = court

    courtlistener_token = os.environ.get("COURTLISTENER_API_TOKEN", "")
    headers: dict = {"User-Agent": "MiroThinker/1.0 (research agent)"}
    if courtlistener_token:
        headers["Authorization"] = f"Token {courtlistener_token}"

    try:
        resp = httpx.get(
            "https://www.courtlistener.com/api/rest/v4/search/",
            params=params,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] CourtListener search failed: {exc}"

    results = data.get("results", [])
    if not results:
        return f"No court opinions found for: {query}"

    formatted = [f"**Court opinions for: {query}** ({len(results)} results)\n"]
    for i, case in enumerate(results[:max_results], 1):
        case_name = case.get("caseName", case.get("case_name", "Unknown"))
        court_name = case.get("court", "")
        date_filed = case.get("dateFiled", case.get("date_filed", ""))
        citation = case.get("citation", [""])
        if isinstance(citation, list):
            citation = citation[0] if citation else ""
        snippet = case.get("snippet", "")[:300]
        # Clean HTML from snippet
        import re
        snippet = re.sub(r"<[^>]+>", "", snippet).strip()
        absolute_url = case.get("absolute_url", "")
        url = f"https://www.courtlistener.com{absolute_url}" if absolute_url else ""

        formatted.append(
            f"  {i}. **{case_name}**\n"
            f"     Court: {court_name} | Filed: {date_filed}\n"
            + (f"     Citation: {citation}\n" if citation else "")
            + (f"     URL: {url}\n" if url else "")
            + (f"     {snippet}..." if snippet else "")
        )

    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# SEC EDGAR — Corporate filings, insider trading, fraud
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_sec_filings(
    query: str = "",
    company: str = "",
    filing_type: str = "",
    max_results: int = 10,
) -> str:
    """Search SEC EDGAR for corporate filings. Free, no key needed.

    Contains insider trading disclosures, fraud buried in footnotes,
    corporate governance failures, and financial restatements. Requires
    User-Agent with email per SEC policy.

    Args:
        query: Full-text search query.
        company: Company name or CIK number.
        filing_type: Filing type (e.g. "10-K", "10-Q", "8-K", "DEF 14A",
                     "4" for insider trading). Leave empty for all.
        max_results: Maximum results (default 10).

    Returns:
        Formatted list of SEC filings with links to full documents.
    """
    import httpx

    # SEC requires User-Agent with contact email
    headers = {
        "User-Agent": "MiroThinker research-agent support@miromind.ai",
        "Accept": "application/json",
    }

    # Use EDGAR full-text search API
    params: dict = {
        "q": query or company,
        "dateRange": "custom",
        "startdt": "2010-01-01",
        "enddt": "2026-12-31",
        "forms": filing_type,
    }

    try:
        resp = httpx.get(
            "https://efts.sec.gov/LATEST/search-index",
            params=params,
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            # Fallback to EDGAR full text search
            resp = httpx.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q": query or company,
                    "forms": filing_type,
                },
                headers=headers,
                timeout=30,
            )
    except Exception:
        pass

    # Use the company filings endpoint if company specified
    if company:
        try:
            # Search for company CIK
            resp2 = httpx.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": company, "forms": filing_type or "10-K"},
                headers=headers,
                timeout=30,
            )
        except Exception:
            pass

    # Fallback: use the EDGAR full-text search system
    try:
        search_url = "https://efts.sec.gov/LATEST/search-index"
        search_params = {
            "q": f'"{query}"' if query else company,
            "forms": filing_type,
            "from": 0,
            "size": min(max_results, 50),
        }
        resp = httpx.get(
            search_url,
            params=search_params,
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                formatted = [f"**SEC EDGAR filings** ({len(hits)} results)\n"]
                for i, hit in enumerate(hits[:max_results], 1):
                    source = hit.get("_source", {})
                    file_date = source.get("file_date", "")
                    display_names = source.get("display_names", [])
                    entity_name = display_names[0] if display_names else ""
                    form_type = source.get("form_type", "")
                    file_num = source.get("file_num", "")

                    formatted.append(
                        f"  {i}. **{entity_name}** — {form_type}\n"
                        f"     Filed: {file_date} | File #: {file_num}"
                    )
                return "\n\n".join(formatted)
    except Exception:
        pass

    # Final fallback: use company tickers endpoint
    try:
        resp = httpx.get(
            "https://www.sec.gov/cgi-bin/browse-edgar",
            params={
                "action": "getcompany",
                "company": company or query,
                "type": filing_type or "10-K",
                "dateb": "",
                "owner": "include",
                "count": str(max_results),
                "search_text": "",
                "action": "getcompany",
                "output": "atom",
            },
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            return (
                f"**SEC EDGAR search results for: {company or query}**\n\n"
                f"Direct search URL: https://www.sec.gov/cgi-bin/browse-edgar?"
                f"company={company or query}&CIK=&type={filing_type or '10-K'}&dateb=&owner=include&count={max_results}&search_text=&action=getcompany\n\n"
                f"EDGAR Full-Text Search: https://efts.sec.gov/LATEST/search-index?q={query or company}&forms={filing_type}"
            )
    except Exception as exc:
        return f"[TOOL_ERROR] SEC EDGAR search failed: {exc}"

    return (
        f"**SEC EDGAR — Direct Access Links**\n\n"
        f"Full-text search: https://efts.sec.gov/LATEST/search-index?q={query or company}\n"
        f"Company search: https://www.sec.gov/cgi-bin/browse-edgar?company={company or query}&type={filing_type}\n"
        f"EDGAR search UI: https://efts.sec.gov/LATEST/search-index?q={query or company}&forms={filing_type}\n\n"
        f"Filing types: 10-K (annual), 10-Q (quarterly), 8-K (events), "
        f"DEF 14A (proxy), 4 (insider trading), 13F (institutional holdings)"
    )


# ═══════════════════════════════════════════════════════════════════════
# ICIJ Offshore Leaks — Panama/Paradise/Pandora Papers
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_offshore_leaks(
    query: str,
    max_results: int = 10,
) -> str:
    """Search ICIJ Offshore Leaks database. Free, no key needed.

    810,000+ offshore entities from Panama Papers, Paradise Papers,
    Pandora Papers, and Offshore Leaks. Exposes shell companies, tax
    havens, and hidden wealth structures.

    Args:
        query: Search query (person name, company name, country).
        max_results: Maximum results (default 10).

    Returns:
        Matching offshore entities with jurisdictions and connections.
    """
    import httpx

    try:
        resp = httpx.get(
            "https://offshoreleaks.icij.org/api/v1/search",
            params={
                "q": query,
                "limit": min(max_results, 100),
            },
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data if isinstance(data, list) else data.get("results", data.get("data", []))
        else:
            # The ICIJ API may not have a public search endpoint
            # Provide the bulk download URLs instead
            return (
                f"**ICIJ Offshore Leaks — Search for: {query}**\n\n"
                f"Web search: https://offshoreleaks.icij.org/search?q={query}\n\n"
                f"Bulk data downloads (CSV):\n"
                f"  - Entities: https://offshoreleaks-data.icij.org/offshoreleaks/csv/csv_entities.csv.zip\n"
                f"  - Officers: https://offshoreleaks-data.icij.org/offshoreleaks/csv/csv_officers.csv.zip\n"
                f"  - Intermediaries: https://offshoreleaks-data.icij.org/offshoreleaks/csv/csv_intermediaries.csv.zip\n"
                f"  - Addresses: https://offshoreleaks-data.icij.org/offshoreleaks/csv/csv_addresses.csv.zip\n"
                f"  - Edges (relationships): https://offshoreleaks-data.icij.org/offshoreleaks/csv/csv_edges.csv.zip\n\n"
                f"Load into DuckDB:\n"
                f"  CREATE TABLE entities AS SELECT * FROM read_csv_auto('csv_entities.csv');\n"
                f"  SELECT * FROM entities WHERE name ILIKE '%{query}%';"
            )
    except Exception:
        return (
            f"**ICIJ Offshore Leaks — Search for: {query}**\n\n"
            f"Web search: https://offshoreleaks.icij.org/search?q={query}\n\n"
            f"Bulk data downloads (CSV):\n"
            f"  - Entities: https://offshoreleaks-data.icij.org/offshoreleaks/csv/csv_entities.csv.zip\n"
            f"  - Officers: https://offshoreleaks-data.icij.org/offshoreleaks/csv/csv_officers.csv.zip\n"
            f"  - Intermediaries: https://offshoreleaks-data.icij.org/offshoreleaks/csv/csv_intermediaries.csv.zip\n"
            f"  - Addresses: https://offshoreleaks-data.icij.org/offshoreleaks/csv/csv_addresses.csv.zip\n"
            f"  - Edges: https://offshoreleaks-data.icij.org/offshoreleaks/csv/csv_edges.csv.zip\n\n"
            f"Load into DuckDB:\n"
            f"  CREATE TABLE entities AS SELECT * FROM read_csv_auto('csv_entities.csv');\n"
            f"  SELECT * FROM entities WHERE name ILIKE '%{query}%';"
        )

    if not results:
        return f"No offshore entities found for: {query}"

    formatted = [f"**ICIJ Offshore Leaks for: {query}** ({len(results)} results)\n"]
    for i, entity in enumerate(results[:max_results], 1):
        name = entity.get("name", entity.get("entity", "Unknown"))
        jurisdiction = entity.get("jurisdiction", entity.get("jurisdiction_description", ""))
        country = entity.get("country_codes", entity.get("countries", ""))
        source = entity.get("sourceID", entity.get("source", ""))
        node_id = entity.get("node_id", entity.get("id", ""))
        entity_type = entity.get("entity_type", entity.get("type", ""))

        formatted.append(
            f"  {i}. **{name}**\n"
            f"     Type: {entity_type} | Jurisdiction: {jurisdiction}\n"
            f"     Country: {country} | Source: {source}\n"
            + (f"     Link: https://offshoreleaks.icij.org/nodes/{node_id}\n" if node_id else "")
        )

    return "\n\n".join(formatted)


# ── Tool registry ─────────────────────────────────────────────────────

GOVERNMENT_TOOLS = [
    search_clinical_trials,
    get_trial_results,
    search_fda_adverse_events,
    search_fda_recalls,
    search_court_opinions,
    search_sec_filings,
    search_offshore_leaks,
]

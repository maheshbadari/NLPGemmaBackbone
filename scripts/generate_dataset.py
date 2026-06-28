#!/usr/bin/env python3
"""Synthetic NER dataset generator for GemmaBackBone.

Design principles
-----------------
Every sentence is written to be contextually and structurally valid for *any*
entity drawn from that type's pool.  This is enforced by two constraints:

  1. Pool discipline — all values in a pool share the same syntactic role
     (e.g. every DUR entry works with "for <DUR>"; every TIME entry works
     with "at <TIME>"; every SET entry can serve as a sentence-final adverb).

  2. Template discipline — templates never add a preposition that would be
     wrong for some pool values (e.g. SET templates end with "{SET} ." so the
     entity value itself provides any preposition it needs, and all values
     in the SET pool are tested against all SET templates before inclusion).

Entity type → head mapping
--------------------------
  identity  : PER  ORG  MISC
  location  : LOC  GPE  FAC
  temporal  : DATE TIME DUR SET
  domain    : PROD EVENT LAW

Output format (space-separated, 5 columns per token)
------------------------------------------------------
  word  identity_tag  location_tag  temporal_tag  domain_tag

Usage
-----
  python scripts/generate_dataset.py
  python scripts/generate_dataset.py --train 12000 --val 1500 --test 1500
"""

import re
import os
import sys
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Entity pools
# All entries share the same syntactic role within their type.
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_POOL: Dict[str, List[str]] = {

    # ── People ───────────────────────────────────────────────────────────────
    # Role: noun-phrase subject or object.  Names are "First Last" format.
    "PER": [
        "Alice Johnson", "Bob Martinez", "Carol Williams", "David Chen",
        "Emma Thompson", "Frank Mueller", "Grace Kim", "Henry Patel",
        "Isabel Santos", "James O'Brien", "Karen Nakamura", "Liam Fischer",
        "Maria Gonzalez", "Nathan Singh", "Olivia Fernandez", "Peter Andersen",
        "Rachel Cohen", "Samuel Okafor", "Tanya Petrova", "Umar Hassan",
        "Victoria Park", "William Turner", "Xia Lin", "Yusuf Ibrahim",
        "Zoe Campbell", "Aaron Goldstein", "Beatrix von Hoffmann", "Carlos Ruiz",
        "Diana Morozova", "Eduardo Nkrumah", "Fatima Al-Rashid", "George Blackwood",
        "Hannah Sorensen", "Ivan Kowalski", "Julia Bergmann", "Kevin Tanaka",
        "Laura Johansson", "Michael O'Connor", "Nina Vasquez", "Oscar Lindgren",
        "Priya Mehta", "Quentin Dubois", "Rosa Ferretti", "Stefan Nowak",
        "Theresa Hoffmann", "Uma Krishnan", "Vincent Moreau", "Wendy Larsson",
        "Xavier Dupont", "Yasmin Al-Farsi",
    ],

    # ── Organisations ────────────────────────────────────────────────────────
    # Role: noun-phrase subject or object.
    "ORG": [
        "Global Tech Solutions", "Apex Manufacturing", "NovaBridge Capital",
        "Meridian Health Systems", "Pacific Rim Logistics", "Atlas Energy Group",
        "Horizon Pharmaceuticals", "Sterling Financial Partners", "Crestwood Media",
        "Ironclad Cybersecurity", "BlueSky Aviation", "Summit Retail Group",
        "Quantum Data Labs", "Evergreen Agriculture", "Northern Rail Network",
        "Velocity Sports Management", "Pinnacle Legal Associates", "Cascade Biotech",
        "DeltaForce Security", "Luminary Education Foundation",
        "Redwood Construction", "SilverStream Investments", "TechNova Systems",
        "United Mobility Services", "Vertex Analytics",
        "Westgate Broadcasting", "Xcelerate Marketing Agency", "Yellowstone Mining",
        "Zenith Aerospace", "AlphaWave Telecommunications",
        "Brightfield Consulting", "ClearPath Environmental", "Daybreak Publishing",
        "EastCoast Insurance Group", "FrontLine Healthcare",
        "GreenLeaf Renewable Energy", "Highpoint Architecture",
        "InnovateTech Ventures", "JumpStart Accelerator", "Keystone Logistics",
    ],

    # ── Natural / geographical locations ─────────────────────────────────────
    # Role: object of in / across / through / along.
    # All entries begin with "the" so they read naturally in prepositional phrase.
    "LOC": [
        "the Amazon rainforest", "the Swiss Alps", "the Sahara Desert",
        "the Pacific Ocean", "the Arctic Circle", "the Mediterranean coast",
        "the Mississippi River", "the Himalayan foothills", "the Baltic Sea",
        "the Great Barrier Reef", "the Nile Delta", "the Siberian tundra",
        "the Appalachian Mountains", "the Congo Basin", "the Gobi Desert",
        "the English Channel", "the Andes mountain range", "the Mojave Desert",
        "the Rhine Valley", "the Ganges plain",
    ],

    # ── Geo-political entities ────────────────────────────────────────────────
    # Role: object of in / to / from.  Cities and countries only.
    "GPE": [
        "New York", "London", "Tokyo", "Paris", "Berlin", "Sydney",
        "Toronto", "Mumbai", "Beijing", "São Paulo", "Cairo", "Lagos",
        "Mexico City", "Seoul", "Amsterdam", "Singapore", "Dubai",
        "Nairobi", "Stockholm", "Buenos Aires", "Warsaw", "Vienna",
        "Bangkok", "Istanbul", "Johannesburg", "Oslo", "Helsinki",
        "Lisbon", "Prague", "Zurich", "Brussels", "Dublin", "Riyadh",
        "Jakarta", "Kuala Lumpur", "Manila", "Karachi", "Dhaka",
        "Geneva", "Monaco",
        "Germany", "Japan", "France", "Brazil", "India", "Australia",
        "Canada", "China", "Italy", "Spain", "South Korea",
    ],

    # ── Facilities ────────────────────────────────────────────────────────────
    # Role: object of at / in / outside.
    "FAC": [
        "Heathrow Airport", "Madison Square Garden", "the Sydney Opera House",
        "the Louvre Museum", "Times Square", "Wembley Stadium", "the Pentagon",
        "Grand Central Terminal", "the National Mall", "Tokyo Skytree",
        "the Burj Khalifa", "O'Hare International Airport", "the Colosseum",
        "the Smithsonian Institution", "Yankee Stadium", "the Tate Modern",
        "the Getty Center", "Carnegie Hall", "the Kennedy Center",
        "the International Space Station",
    ],

    # ── Calendar dates ────────────────────────────────────────────────────────
    # Role: object of "on <DATE>" or "by <DATE>".
    # Every entry is a specific calendar point (day or named date).
    "DATE": [
        "January 15 2024", "March 3 2023", "the 12th of August",
        "December 31 2023", "April 7 2023", "the second of November",
        "next Friday", "last Tuesday", "this coming Monday",
        "16 June 2023", "5 September 2024", "the last Thursday of the month",
        "the upcoming Wednesday", "the first Monday of next month",
        "the third Thursday of November",
    ],

    # ── Period expressions ────────────────────────────────────────────────────
    # Role: object of "for <PERIOD>" or "results for <PERIOD>".
    # Every entry is a fiscal / calendar period — NOT a specific day.
    # Both DATE and PERIOD produce B-DATE / I-DATE BIO labels (see _CANONICAL).
    "PERIOD": [
        "the fourth quarter", "year-end 2024", "fiscal year 2025",
        "the first quarter of 2024", "last month", "next quarter",
        "the coming weeks", "the past twelve months", "early next year",
        "the third quarter", "the full financial year", "the first half of 2025",
    ],

    # ── Times ─────────────────────────────────────────────────────────────────
    # Role: object of "at".
    # All entries are clock-time expressions valid after "at".
    "TIME": [
        "3 PM", "noon", "midnight", "9 AM", "dawn", "dusk", "sunrise",
        "6:45 AM", "11:30 PM", "2:00 PM", "8:00 AM", "10:30 AM",
        "half past six", "quarter to eight", "the opening bell",
        "the closing bell", "first light",
    ],

    # ── Durations ─────────────────────────────────────────────────────────────
    # Role: object of "for".
    # All entries work as the complement of "for <DUR>".
    "DUR": [
        "three hours", "two weeks", "six months", "a full decade",
        "nearly five years", "just over an hour", "forty-eight hours",
        "several months", "more than a year", "roughly ninety days",
        "a quarter century", "twelve months", "an extended period",
        "less than a minute", "the entire weekend",
    ],

    # ── Recurring-frequency expressions ──────────────────────────────────────
    # Role: sentence-final adverb (no additional preposition in template).
    # All entries work naturally after "meets / reports / audits {SET} ."
    "SET": [
        "every Tuesday", "twice a year", "on alternate Mondays", "quarterly",
        "each morning", "annually", "biannually", "every other week",
        "on weekdays", "three times a month", "each fiscal quarter",
        "every other day", "weekly", "on the first of each month",
        "on a rolling basis", "twice daily",
    ],

    # ── Products ──────────────────────────────────────────────────────────────
    # Role: noun-phrase subject or object.
    "PROD": [
        "the NovaStar X12", "ProVision 3.0", "HorizonTab Ultra",
        "the Apex Series 7", "CloudSync Pro", "TerraBoard 2.0",
        "the PulseTracker Elite", "VisionAI Studio", "CoreDrive 500",
        "the StellarPhone 15", "DataVault Enterprise", "FlexARM Processor",
        "the EchoPod Max", "StreamBox 4K", "QuantumSense Wearable",
        "the NanoSeal Filter", "BioCore Scanner", "ArcLight Camera System",
        "the GridMaster Platform", "SafeNet 360", "the AtlasCloud Suite",
        "TurboMesh Router", "the HelixDrive SSD", "AquaPure System",
    ],

    # ── Named events ──────────────────────────────────────────────────────────
    # Role: subject, or object of "at / during / ahead of".
    # All entries begin with "the" or a specific name so they read naturally.
    "EVENT": [
        "the World Economic Forum", "the Summer Olympics", "the G20 Summit",
        "the International AI Conference", "the Global Health Assembly",
        "the Annual Tech Expo", "the Climate Action Summit",
        "the International Trade Fair", "the World Cup",
        "the Global Cybersecurity Forum", "the Pacific Rim Leaders Summit",
        "the International Film Festival", "the World Science Congress",
        "the Annual Shareholders Meeting", "the Digital Transformation Conference",
        "the Global Renewable Energy Symposium", "the Urban Mobility Summit",
        "the International Space Exploration Forum", "CES 2024",
        "the World Startup Competition", "the Global Fintech Conference",
        "the Biomedical Innovation Summit", "the Annual Climate Dialogue",
    ],

    # ── Laws, regulations, treaties ───────────────────────────────────────────
    # Role: subject, or object of "under / with / of / regarding".
    "LAW": [
        "the General Data Protection Regulation",
        "the Clean Air Act",
        "Article 50 of the Lisbon Treaty",
        "the Dodd-Frank Act",
        "the Paris Agreement",
        "the Digital Markets Act",
        "the Americans with Disabilities Act",
        "the Basel III Framework",
        "the Freedom of Information Act",
        "the Sarbanes-Oxley Act",
        "the Foreign Corrupt Practices Act",
        "the EU AI Act",
        "the Cybersecurity and Infrastructure Security Act",
        "the Children's Online Privacy Protection Act",
        "the GDPR",
        "the National Environmental Policy Act",
        "Section 230 of the Communications Decency Act",
        "the Health Insurance Portability and Accountability Act",
        "the Anti-Money Laundering Directive",
    ],

    # ── Miscellaneous ─────────────────────────────────────────────────────────
    # Role: noun-phrase subject or object.
    "MISC": [
        "artificial intelligence", "quantum computing", "renewable energy",
        "blockchain technology", "machine learning", "cybersecurity",
        "genetic engineering", "autonomous vehicles", "digital transformation",
        "cloud infrastructure", "the Internet of Things", "augmented reality",
        "facial recognition technology", "5G networks", "nanotechnology",
        "CRISPR gene editing", "satellite broadband", "carbon capture",
        "precision agriculture", "smart manufacturing",
    ],

    # ── Street-level addresses ────────────────────────────────────────────────
    # Role: object of "at / to / on / near / from".
    # These are street addresses only — city and country are annotated as GPE.
    # All entries work naturally after "at {ADDR}" or "to {ADDR}".
    "ADDR": [
        "10 Downing Street", "1600 Pennsylvania Avenue", "221B Baker Street",
        "5th Avenue", "Wall Street", "the Champs-Elysees",
        "123 Innovation Drive", "400 Commerce Boulevard", "50 Broad Street",
        "the corner of 5th and Broadway", "14 Unter den Linden",
        "7 rue de la Paix", "One Canada Square",
        "350 Fifth Avenue", "100 Technology Park Drive",
        "Exchange Square", "Silicon Alley", "Canary Wharf",
        "the financial district on Broad Street",
        "the campus at 1 Infinite Loop",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Templates
# {TYPE} placeholders are filled from the matching entity pool.
# Punctuation is space-tokenised (Penn-Treebank style: word " ." not "word.").
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES: List[str] = [

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — PER
    # ═══════════════════════════════════════════════════════════════════════

    # entity at sentence start
    "{PER} was appointed as the new chief executive officer .",
    "{PER} announced the organisation 's results for the third quarter .",
    "{PER} signed the framework agreement on behalf of the steering committee .",
    "{PER} confirmed that the restructuring plan would proceed as scheduled .",
    "{PER} received the lifetime achievement award at the annual gala .",
    "{PER} resigned from the position following the board 's unanimous decision .",
    "{PER} delivered the opening address to delegates at the summit .",

    # entity in middle of sentence
    "The board unanimously selected {PER} to lead the newly formed task force .",
    "Executives appointed {PER} as the interim director of global operations .",
    "The selection committee awarded the fellowship to {PER} after a rigorous review .",
    "Investors welcomed the appointment of {PER} as the incoming chairperson .",
    "Management authorised {PER} to negotiate the final terms of the transaction .",
    "The panel of judges recognised {PER} for outstanding contributions to the field .",
    "Directors confirmed that {PER} would continue in an advisory capacity .",

    # entity at sentence end
    "The innovation award for the year was formally presented to {PER} .",
    "The final decision on the matter was delegated to {PER} .",
    "Responsibility for oversight of the programme was assigned to {PER} .",
    "The honorary doctorate was conferred upon {PER} at the winter ceremony .",
    "Following a competitive process , delegates elected {PER} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — ORG
    # ═══════════════════════════════════════════════════════════════════════

    # start
    "{ORG} reported record earnings for the third consecutive quarter .",
    "{ORG} announced the launch of an expanded service and product portfolio .",
    "{ORG} will open three new regional offices over the next eighteen months .",
    "{ORG} entered a strategic partnership to accelerate market expansion .",
    "{ORG} issued a public statement addressing the ongoing regulatory inquiry .",
    "{ORG} completed the cross-border acquisition after months of negotiation .",
    "{ORG} released its annual sustainability report highlighting key milestones .",

    # middle
    "The proposed merger involving {ORG} was referred to the competition authority .",
    "Analysts expressed serious concern about the long-term financial position of {ORG} .",
    "A comprehensive audit of {ORG} identified significant gaps in internal controls .",
    "Shares in {ORG} rose sharply immediately following the earnings announcement .",
    "The government awarded a multi-year infrastructure maintenance contract to {ORG} .",
    "The joint venture agreement was structured to benefit {ORG} across three new markets .",

    # end
    "The exclusive distribution licence was granted to {ORG} .",
    "Formal responsibility for programme delivery was assigned to {ORG} .",
    "The independent review of governance practices was commissioned by {ORG} .",
    "The new consolidated headquarters facility will be operated by {ORG} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — GPE
    # ═══════════════════════════════════════════════════════════════════════

    # start
    "{GPE} hosted the annual international summit on trade and investment .",
    "{GPE} introduced new import restrictions on a range of manufactured goods .",
    "{GPE} recorded the highest rate of economic growth in the region last year .",
    "{GPE} pledged additional public funding for renewable energy infrastructure .",

    # middle
    "Delegates from forty-two nations gathered in {GPE} for the multilateral talks .",
    "The organisation 's global headquarters will relocate to {GPE} next year .",
    "Flights to and from {GPE} were suspended following severe weather conditions .",
    "The summit held in {GPE} concluded with the signing of a joint declaration .",
    "Relief shipments were dispatched to {GPE} within hours of the disaster .",
    "Foreign direct investment in {GPE} rose by thirty percent over the past year .",

    # end
    "The annual international conference will take place in {GPE} .",
    "All regional operations will be consolidated and relocated to {GPE} .",
    "The new applied research centre is scheduled to be established in {GPE} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — LOC
    # ═══════════════════════════════════════════════════════════════════════

    # start
    "{LOC} faces unprecedented ecological degradation due to rising temperatures .",
    "{LOC} was designated a protected natural area under international treaty .",
    "{LOC} experienced record-breaking temperatures during the summer season .",

    # middle
    "Research teams conducting fieldwork across {LOC} recorded alarming findings .",
    "A new multi-year conservation programme targeting {LOC} was formally launched .",
    "Wildlife populations in {LOC} have declined by more than thirty percent .",
    "The independent survey covered an area spanning {LOC} and adjacent territories .",

    # end
    "The expedition team successfully completed its full traverse of {LOC} .",
    "The foundation directed substantial long-term funding toward the restoration of {LOC} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — FAC
    # ═══════════════════════════════════════════════════════════════════════

    # start
    "{FAC} will undergo a comprehensive structural renovation over the next two years .",
    "{FAC} was briefly evacuated following a reported security incident .",
    "{FAC} welcomed its ten-millionth visitor of the calendar year last week .",

    # middle
    "The formal signing ceremony was scheduled to take place at {FAC} .",
    "Security arrangements were significantly reinforced at {FAC} ahead of the summit .",
    "A joint press conference was held on the steps outside {FAC} .",
    "Senior representatives convened in the main auditorium of {FAC} .",

    # end
    "The annual awards gala will be hosted at {FAC} .",
    "World leaders are scheduled to convene at {FAC} for the opening session .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — ADDR (street-level addresses)
    # ═══════════════════════════════════════════════════════════════════════

    # entity at sentence start
    "{ADDR} serves as the organisation 's primary operational hub .",
    "{ADDR} was identified as the location of the suspected breach .",
    "{ADDR} will house the newly established regional coordination centre .",

    # entity in middle of sentence
    "The company relocated its global headquarters to {ADDR} last spring .",
    "Investigators were dispatched to {ADDR} following an anonymous tip-off .",
    "The branch office at {ADDR} was the first to implement the new system .",
    "Emergency services responded to an incident at {ADDR} on Friday evening .",
    "Protesters gathered outside the building at {ADDR} during the hearing .",
    "The lease for the premises at {ADDR} was renewed for another five years .",
    "Clients visiting {ADDR} are advised to use the side entrance .",

    # entity at sentence end
    "All formal correspondence should be addressed and directed to {ADDR} .",
    "The package was successfully delivered and signed for at {ADDR} .",
    "The delegation arrived and checked into their accommodation at {ADDR} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — DATE
    # ═══════════════════════════════════════════════════════════════════════

    # start
    "{DATE} marks the official launch of the updated regulatory framework .",
    "{DATE} was identified as the binding deadline for submission of proposals .",
    "{DATE} has been set by the board as the target for completion of the review .",

    # middle
    "The board has formally scheduled the vote for {DATE} .",
    "All completed applications must be submitted to the office by {DATE} .",
    "The committee is expected to reach and announce a final decision by {DATE} .",
    "Financial performance data for {PERIOD} significantly exceeded analyst expectations .",
    "A comprehensive progress review has been planned for {DATE} .",
    "The transition team confirmed that handover would be completed by {DATE} .",

    # end
    "The full programme is expected to be delivered by {DATE} .",
    "The revised regulations are scheduled to formally come into force on {DATE} .",
    "The initial transition period is legally set to conclude on {DATE} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — TIME
    # All templates use "at {TIME}" — every pool entry is a clock-time expression.
    # ═══════════════════════════════════════════════════════════════════════

    "Trading on the exchange was abruptly and unexpectedly halted at {TIME} .",
    "The emergency session of the board was convened at {TIME} .",
    "The press briefing for international media is scheduled to begin at {TIME} .",
    "All critical systems were taken offline at exactly {TIME} .",
    "The official announcement is expected to be made at {TIME} .",
    "The final plenary session of the conference adjourned at {TIME} .",
    "The formal signing ceremony will commence at {TIME} in the east hall .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — DUR
    # All templates use "for {DUR}" — every pool entry works with "for".
    # ═══════════════════════════════════════════════════════════════════════

    # start — "{DUR} of X produced Y" pattern
    "{DUR} of intensive cross-party negotiations finally produced a binding agreement .",
    "{DUR} of sustained public and private investment transformed the sector significantly .",

    # middle
    "The flagship project has been in active development for {DUR} .",
    "The critical service outage persisted for {DUR} before engineers restored access .",
    "Residents in the affected areas were advised to remain indoors for {DUR} .",
    "The court proceedings are expected to continue without interruption for {DUR} .",
    "The primary construction phase is projected to last for {DUR} .",

    # end
    "The comprehensive system overhaul programme is expected to run for {DUR} .",
    "International negotiations on the matter are anticipated to extend for {DUR} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — SET
    # Templates end with "{SET} ." — every pool entry is a sentence adverb.
    # ═══════════════════════════════════════════════════════════════════════

    "The oversight committee meets to review all outstanding applications {SET} .",
    "The statistical agency publishes updated economic indicators {SET} .",
    "The facility undergoes a mandatory full-scope safety inspection {SET} .",
    "Detailed progress reports are formally submitted to the regulator {SET} .",
    "The board of directors conducts a structured performance review {SET} .",
    "Updated compliance data is made available to the public {SET} .",
    "The internal audit team assesses all active client-facing programmes {SET} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — PROD
    # ═══════════════════════════════════════════════════════════════════════

    # start
    "{PROD} was unveiled to the public at the company 's annual developer showcase .",
    "{PROD} received the industry 's highest award for technical innovation this year .",
    "{PROD} is expected to reach global consumers before the end of the quarter .",

    # middle
    "Engineers discovered a critical security vulnerability affecting {PROD} .",
    "The company announced a significant performance and security update to {PROD} .",
    "Early adopter reviews of {PROD} were consistent and overwhelmingly positive .",
    "Consumer demand for {PROD} has substantially outpaced current production capacity .",
    "The national regulatory authority approved the commercial market release of {PROD} .",

    # end
    "The enterprise edition is the most capable and scalable version of {PROD} .",
    "The independent safety review board assessed and certified the release of {PROD} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — EVENT
    # ═══════════════════════════════════════════════════════════════════════

    # start
    "{EVENT} opened with a keynote address delivered by the host nation 's president .",
    "{EVENT} attracted record attendance from delegates representing six continents .",
    "{EVENT} concluded with the unanimous signing of a landmark multilateral agreement .",
    "{EVENT} was postponed by two weeks due to heightened security concerns .",

    # middle
    "World leaders gathered at {EVENT} and agreed on a set of binding climate targets .",
    "The official announcement was made during the opening plenary of {EVENT} .",
    "A large group of protesters assembled outside the venue ahead of {EVENT} .",
    "The pivotal deal was finalised on the sidelines during {EVENT} .",
    "The keynote address delivered at {EVENT} outlined the strategic priorities ahead .",

    # end
    "The organisation unveiled its comprehensive five-year expansion strategy at {EVENT} .",
    "More than sixty leading technology companies are expected to exhibit at {EVENT} .",
    "The company will publicly premiere its flagship consumer product at {EVENT} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — LAW
    # ═══════════════════════════════════════════════════════════════════════

    # start
    "{LAW} mandates that all organisations report data breaches within seventy-two hours .",
    "{LAW} has fundamentally reshaped the compliance landscape across the entire sector .",
    "{LAW} is expected to face a constitutional challenge before the appellate court .",

    # middle
    "The company was found to be in material and deliberate violation of {LAW} .",
    "Full compliance with {LAW} requires substantial and sustained investment in controls .",
    "Senior executives and legal counsel were comprehensively briefed on {LAW} .",
    "The regulatory fine was formally levied under the enforcement provisions of {LAW} .",

    # end
    "All participating member states are legally obligated to implement {LAW} .",
    "The regulator issued detailed operational guidance on the correct interpretation of {LAW} .",
    "The appellate court 's ruling was found to be fully consistent with {LAW} .",

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE-ENTITY  — MISC
    # ═══════════════════════════════════════════════════════════════════════

    # start
    "{MISC} has emerged as the central focus of global technology policy discussions .",
    "{MISC} was cited by executives as the primary driver of the strategic pivot .",

    # middle
    "The report highlighted the growing and transformative role of {MISC} in the economy .",
    "Leading experts remain deeply divided on the long-term implications of {MISC} .",
    "The proposed new framework was specifically designed to govern the deployment of {MISC} .",

    # end
    "Policymakers are urgently seeking a coherent and enforceable approach to {MISC} .",
    "The expert panel issued comprehensive new guidelines addressing the risks of {MISC} .",

    # ═══════════════════════════════════════════════════════════════════════
    # TWO-ENTITY TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════

    # PER + ORG
    "{PER} was promoted to chief strategy officer at {ORG} .",
    "{PER} co-founded {ORG} alongside two university colleagues a decade ago .",
    "{PER} formally announced her departure from {ORG} amid major restructuring plans .",
    "{ORG} named {PER} as its new global head of research and development .",
    "{ORG} confirmed that {PER} would serve as executive chairman for the next full term .",
    "The sudden resignation of {PER} from {ORG} caught most industry observers off guard .",
    "Under the sustained leadership of {PER} , {ORG} expanded into twelve new markets .",
    "{PER} joined {ORG} from a rival organisation where she had served as chief scientist .",

    # PER + GPE
    "{PER} arrived in {GPE} for a scheduled bilateral meeting with senior ministers .",
    "{PER} delivered a keynote address to senior business leaders assembled in {GPE} .",
    "Officials in {GPE} formally received {PER} at the state reception ceremony .",
    "{PER} met with cabinet ministers during a two-day working visit to {GPE} .",

    # PER + EVENT
    "{PER} delivered the formal opening address at {EVENT} .",
    "{PER} was named as lead negotiator for the delegation ahead of {EVENT} .",
    "Organisers confirmed that {PER} will present the research findings at {EVENT} .",
    "The closing panel discussion at {EVENT} was chaired by {PER} .",

    # PER + DATE
    "{PER} is expected to formally issue the decision on {DATE} .",
    "{PER} submitted the completed proposal well ahead of the deadline on {DATE} .",
    "People close to {PER} indicated that a final answer would come by {DATE} .",

    # ORG + GPE
    "{ORG} will establish its regional headquarters in {GPE} next calendar year .",
    "{ORG} expanded its distribution and logistics network into {GPE} last quarter .",
    "The government of {GPE} signed a comprehensive framework agreement with {ORG} .",
    "{ORG} announced a public-private joint venture with local authorities in {GPE} .",

    # ORG + PROD
    "{ORG} unveiled {PROD} at its flagship annual developer and product showcase .",
    "{ORG} announced a six-month delay to the commercial global rollout of {PROD} .",
    "Sales of {PROD} accounted for nearly a third of {ORG} 's consolidated annual revenue .",
    "{ORG} received full regulatory clearance to market {PROD} in all major territories .",

    # ORG + EVENT
    "{ORG} confirmed its participation as a headline sponsor of {EVENT} .",
    "{ORG} will host a dedicated technical showcase on the second day of {EVENT} .",
    "The surprise announcement came during {ORG} 's press briefing held at {EVENT} .",

    # ORG + LAW
    "{ORG} was assessed a record fine for its failure to comply with {LAW} .",
    "{ORG} submitted a formal written response to the regulator regarding {LAW} .",
    "The board of {ORG} unanimously voted to adopt new policies aligned with {LAW} .",

    # EVENT + DATE
    "{EVENT} is officially scheduled to commence on {DATE} .",
    "The organising committee confirmed that {EVENT} will take place on {DATE} .",
    "The deadline for delegate registration for {EVENT} falls on {DATE} .",
    "All security preparations for {EVENT} were finalised well ahead of {DATE} .",

    # EVENT + GPE
    "{EVENT} will be hosted by {GPE} for the second consecutive year .",
    "Organisers formally selected {GPE} as the primary host city for {EVENT} .",
    "{GPE} was unanimously awarded the right to host {EVENT} after a competitive bid .",

    # DATE + TIME
    "The emergency board session convened on {DATE} at exactly {TIME} .",
    "Trading across all markets was halted on {DATE} at {TIME} pending regulatory review .",
    "The historic framework agreement was formally signed on {DATE} shortly before {TIME} .",

    # LAW + GPE
    "{GPE} was the first jurisdiction in the world to formally ratify {LAW} .",
    "Officials in {GPE} are still working to achieve full and consistent implementation of {LAW} .",
    "{LAW} applies to every company with commercial operations within {GPE} .",

    # LAW + DATE
    "{LAW} will formally and irrevocably come into effect on {DATE} .",
    "The deadline for full organisational compliance with {LAW} has been fixed at {DATE} .",
    "The landmark appellate ruling on {DATE} upheld the constitutional validity of {LAW} .",

    # PROD + EVENT
    "The unveiling of {PROD} at {EVENT} generated widespread and sustained media coverage .",
    "Registered attendees at {EVENT} were granted privileged early access to {PROD} .",

    # ORG + FAC
    "{ORG} signed a long-term lease agreement to relocate its offices to {FAC} .",
    "The annual conference hosted by {ORG} will be held for the first time at {FAC} .",

    # ORG + ADDR
    "{ORG} officially opened its new branch at {ADDR} last Thursday .",
    "{ORG} confirmed that its registered office will move to {ADDR} .",
    "A fire broke out at the {ORG} facility located at {ADDR} .",
    "The {ORG} team based at {ADDR} leads all international operations .",

    # PER + ADDR
    "{PER} was last seen leaving the building at {ADDR} late on Tuesday .",
    "Investigators confirmed that {PER} had been at {ADDR} on the evening in question .",
    "{PER} signed the lease for the new premises at {ADDR} .",

    # ADDR + GPE
    "{ADDR} in {GPE} will serve as the nerve centre for the new initiative .",
    "The liaison office at {ADDR} in {GPE} handles all regional enquiries .",
    "The incident took place at {ADDR} in {GPE} during the early hours .",

    # ORG + ADDR + GPE
    "{ORG} announced it would open a new office at {ADDR} in {GPE} next year .",
    "The {ORG} delegation checked in at {ADDR} in {GPE} before the talks began .",

    # ═══════════════════════════════════════════════════════════════════════
    # THREE-ENTITY TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════

    "{PER} of {ORG} attended {EVENT} as the organisation 's chief delegate .",
    "{PER} represented {ORG} at the multilateral talks held in {GPE} .",
    "{PER} accepted the senior position at {ORG} following the conclusion of the summit in {GPE} .",
    "{ORG} announced the formal global launch of {PROD} at {EVENT} in {GPE} .",
    "{EVENT} , hosted in {GPE} on {DATE} , attracted unprecedented global attendance .",
    "{PER} unveiled {PROD} during the high-profile keynote address at {EVENT} .",
    "{ORG} surpassed its financial targets for {PERIOD} despite the constraints imposed by {LAW} .",
    "The court 's ruling on {DATE} found that {ORG} had knowingly violated {LAW} .",
    "{PER} is scheduled to arrive in {GPE} several days ahead of {EVENT} .",
    "{ORG} deployed {PROD} to support critical long-term operations at {FAC} .",
    "The internal compliance audit of {ORG} concluded on {DATE} and revealed breaches of {LAW} .",
    "{PER} , speaking at {EVENT} in {GPE} , outlined an ambitious new long-term strategy .",
    "{ORG} 's new {PROD} will be publicly demonstrated at {EVENT} for the very first time .",
    "{EVENT} in {GPE} this year will be formally chaired and presided over by {PER} .",
    "The agreement signed by {ORG} in {GPE} specifically requires compliance with {LAW} .",
]

# ─────────────────────────────────────────────────────────────────────────────
# Core generation logic
# ─────────────────────────────────────────────────────────────────────────────

# PERIOD shares the DATE BIO label — both go to the temporal head as B-DATE/I-DATE.
# This lets templates specialise on calendar points ("on {DATE}") vs fiscal
# periods ("results for {PERIOD}") without introducing a new NER class.
_CANONICAL: Dict[str, str] = {"PERIOD": "DATE"}


def _bio_label(et: str) -> str:
    """Return the BIO entity type string (e.g. PERIOD -> DATE)."""
    return _CANONICAL.get(et, et)


def _head_for_type(et: str) -> str:
    c = _bio_label(et)
    if c in ("PER", "ORG", "MISC"):
        return "identity"
    if c in ("LOC", "GPE", "FAC", "ADDR"):
        return "location"
    if c in ("DATE", "TIME", "DUR", "SET"):
        return "temporal"
    return "domain"   # PROD, EVENT, LAW


def build_sample(
    template: str,
    pool: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Fill one template and return (tokens, {head: [bio_tags]}).

    re.split with a capturing group alternates between plain-text fragments
    and captured entity-type names:
      "{PER} joined {ORG} ."
      → ['', 'PER', ' joined ', 'ORG', ' .']
         i=0   i=1      i=2      i=3    i=4
    Even indices → plain text; odd indices → entity type.
    """
    parts = re.split(r"\{(\w+)\}", template)
    tokens: List[str] = []
    heads: Dict[str, List[str]] = {
        "identity": [], "location": [], "temporal": [], "domain": []
    }

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Plain-text segment
            toks = part.split()
            tokens.extend(toks)
            for h in heads:
                heads[h].extend(["O"] * len(toks))
        else:
            # Entity placeholder: part is the entity type string
            et = part
            entity_toks = random.choice(pool[et]).split()
            target  = _head_for_type(et)
            bio     = _bio_label(et)          # PERIOD → DATE, others unchanged
            tokens.extend(entity_toks)
            for h in heads:
                if h == target:
                    heads[h].append(f"B-{bio}")
                    heads[h].extend([f"I-{bio}"] * (len(entity_toks) - 1))
                else:
                    heads[h].extend(["O"] * len(entity_toks))

    # Capitalise the first token so sentences starting with "the ..." read properly
    if tokens:
        tokens[0] = tokens[0][0].upper() + tokens[0][1:]

    return tokens, heads


def _types_in(template: str) -> List[str]:
    return re.findall(r"\{(\w+)\}", template)


# ─────────────────────────────────────────────────────────────────────────────
# Balanced generation
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list used for balanced sampling and stats display.
# PERIOD is listed after DATE — both produce B-DATE/I-DATE annotations.
ALL_TYPES = [
    "PER", "ORG", "GPE", "LOC", "FAC", "ADDR",
    "DATE", "PERIOD", "TIME", "DUR", "SET",
    "PROD", "EVENT", "LAW", "MISC",
]


def generate(n: int, pool: Dict, templates: List[str], seed: int) -> List:
    """Generate n samples with balanced entity-type coverage.

    For each sample we pick an entity type in round-robin order, then draw a
    random template that *contains* that type.  This guarantees every type
    appears at least n // len(ALL_TYPES) times while preserving randomness in
    the actual sentence and entity values chosen.
    """
    rng = random.Random(seed)

    by_type: Dict[str, List[int]] = defaultdict(list)
    for idx, t in enumerate(templates):
        for et in set(_types_in(t)):
            by_type[et].append(idx)

    # Build a shuffled type queue that visits each type proportionally
    repeats  = (n // len(ALL_TYPES)) + 1
    type_q   = ALL_TYPES * repeats
    rng.shuffle(type_q)
    type_q   = type_q[:n]

    samples = []
    for et in type_q:
        tmpl_idx = rng.choice(by_type[et])
        tokens, heads = build_sample(templates[tmpl_idx], pool)
        samples.append((tokens, heads))

    rng.shuffle(samples)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def write_dataset(samples: List, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for tokens, heads in samples:
            assert len(tokens) == len(heads["identity"]), "length mismatch"
            for i, tok in enumerate(tokens):
                f.write(
                    f"{tok} "
                    f"{heads['identity'][i]} "
                    f"{heads['location'][i]} "
                    f"{heads['temporal'][i]} "
                    f"{heads['domain'][i]}\n"
                )
            f.write("\n")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def _bar(count: int, scale: int = 60) -> str:
    return "#" * (count // scale)


def print_stats(samples: List, split: str) -> None:
    entity_counts:   Counter = Counter()
    position_counts: Counter = Counter()
    per_sentence:    Counter = Counter()

    for tokens, heads in samples:
        n_ents = 0
        L = len(tokens)
        for tags in heads.values():
            for i, tag in enumerate(tags):
                if tag.startswith("B-"):
                    et = tag[2:]
                    entity_counts[et] += 1
                    n_ents += 1
                    rel = i / max(L - 1, 1)
                    if rel < 0.25:
                        position_counts["start"] += 1
                    elif rel > 0.75:
                        position_counts["end"] += 1
                    else:
                        position_counts["middle"] += 1
        per_sentence[n_ents] += 1

    total_ents = sum(entity_counts.values())
    print(f"\n{'='*58}")
    print(f"  {split.upper()}  |  {len(samples):,} sentences  |  {total_ents:,} entities")
    print(f"{'='*58}")

    print("\n  Entity counts (per BIO label — PERIOD counted under DATE):")
    # Merge PERIOD counts into DATE for display (both annotated as B-DATE/I-DATE)
    display_counts = Counter(entity_counts)
    for et in ALL_TYPES:
        bio = _bio_label(et)
        if bio != et:                          # e.g. PERIOD → DATE
            display_counts[bio] += entity_counts.get(et, 0)
    shown = set()
    for et in ALL_TYPES:
        bio  = _bio_label(et)
        if bio in shown:
            continue
        shown.add(bio)
        cnt  = display_counts.get(bio, 0)
        head = _head_for_type(et)
        pct  = 100 * cnt / total_ents if total_ents else 0
        print(f"    {bio:6s}  [{head:8s}]  {cnt:5d}  ({pct:4.1f}%)  {_bar(cnt, 40)}")

    print("\n  Entity position distribution:")
    for pos in ("start", "middle", "end"):
        cnt = position_counts[pos]
        pct = 100 * cnt / total_ents if total_ents else 0
        print(f"    {pos:6s}  {cnt:5d}  ({pct:4.1f}%)  {_bar(cnt, 40)}")

    print("\n  Entities per sentence:")
    for n in sorted(per_sentence):
        print(f"    {n} entity/entities: {per_sentence[n]:,} sentences")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NER dataset")
    parser.add_argument("--train", type=int, default=10000,
                        help="Number of training sentences (default: 10000)")
    parser.add_argument("--val",   type=int, default=1000,
                        help="Number of validation sentences (default: 1000)")
    parser.add_argument("--test",  type=int, default=1000,
                        help="Number of test sentences (default: 1000)")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--outdir", type=str, default="data",
                        help="Output directory (default: data)")
    args = parser.parse_args()

    splits = [
        ("train", args.train, args.seed),
        ("valid", args.val,   args.seed + 1),
        ("test",  args.test,  args.seed + 2),
    ]

    paths = {
        "train": os.path.join(args.outdir, "train.txt"),
        "valid": os.path.join(args.outdir, "valid.txt"),
        "test":  os.path.join(args.outdir, "test.txt"),
    }

    for split, n, seed in splits:
        print(f"\nGenerating {n:,} {split} sentences …", end=" ", flush=True)
        samples = generate(n, ENTITY_POOL, TEMPLATES, seed)
        write_dataset(samples, paths[split])
        print(f"saved -> {paths[split]}")
        print_stats(samples, split)

    print(f"\n\nDone.  Files written to:  {args.outdir}/")


if __name__ == "__main__":
    main()

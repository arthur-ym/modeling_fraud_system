"""Interpretable names for IEEE-CIS Fraud Detection columns.

Vesta masked most field meanings. Only rename columns whose values (or the
official Kaggle writeup) make the semantics clear. Everything else stays
under a family prefix so you do not invent a false story.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd

Confidence = Literal["high", "medium", "low"]
TableName = Literal["transaction", "identity"]


@dataclass(frozen=True)
class ColumnInfo:
    original: str
    renamed: str
    table: TableName
    family: str
    information: str
    meaning: str
    evidence: str
    confidence: Confidence
    categorical: bool = False


def _c_features() -> list[ColumnInfo]:
    return [
        ColumnInfo(
            f"C{i}",
            f"entity_count_{i:02d}",
            "transaction",
            "entity counts",
            "How often related entities (card, address, email, device) appear together.",
            "Masked counting features. Official example: how many addresses are tied to the payment card. Do not treat C1 as a specific count.",
            "Always present, non-negative, highly skewed integers.",
            "low",
        )
        for i in range(1, 15)
    ]


def _d_features() -> list[ColumnInfo]:
    known: dict[int, tuple[str, str, str, Confidence]] = {
        1: (
            "days_since_card_first_seen",
            "Days from a reference event on this card (often first seen). Near 0 for new cards.",
            "Integer days 0–639; D2 is usually missing when D1 is 0.",
            "medium",
        ),
        9: (
            "transaction_time_of_day",
            "Time of day as a fraction of 24h (hour / 24).",
            "24 distinct values in [0, 1); tracks the fractional part of D8.",
            "high",
        ),
    }
    rows: list[ColumnInfo] = []
    for i in range(1, 16):
        if i in known:
            renamed, meaning, evidence, conf = known[i]
        else:
            renamed = f"timedelta_days_{i:02d}"
            meaning = (
                "Days between this transaction and a previous event for some entity "
                "(card, email, address, device). A few columns allow small negatives "
                "(clock / timezone artifacts)."
            )
            evidence = "Official: timedelta such as days since previous transaction."
            conf = "low"
        rows.append(
            ColumnInfo(
                f"D{i}",
                renamed,
                "transaction",
                "recency / timedelta",
                "How long since a related event (card age, last purchase, etc.).",
                meaning,
                evidence,
                conf,
            )
        )
    return rows


def _m_features() -> list[ColumnInfo]:
    rows = [
        ColumnInfo(
            "M1",
            "match_name_or_address_1",
            "transaction",
            "match flags",
            "Whether purchaser details match the card / address file.",
            "Boolean match (T only in the sample; almost never F). Same missingness as M2/M3; ProductCD=W only.",
            "Official: match such as names on card and address. Observed T/F; missing outside W.",
            "medium",
            True,
        ),
        ColumnInfo(
            "M2",
            "match_name_or_address_2",
            "transaction",
            "match flags",
            "Second name/address match check.",
            "T/F match. Same missingness as M1/M3; ProductCD=W only.",
            "Co-missing with M1/M3; T/F.",
            "medium",
            True,
        ),
        ColumnInfo(
            "M3",
            "match_name_or_address_3",
            "transaction",
            "match flags",
            "Third name/address match check.",
            "T/F match. Same missingness as M1/M2; ProductCD=W only.",
            "Co-missing with M1/M2; T/F.",
            "medium",
            True,
        ),
        ColumnInfo(
            "M4",
            "match_type_code",
            "transaction",
            "match flags",
            "Match type / verification outcome code.",
            "Not T/F. Values M0/M1/M2. Present mainly for ProductCD W and C.",
            "3-level code; different missingness from M1–M3.",
            "medium",
            True,
        ),
    ]
    for i in range(5, 10):
        rows.append(
            ColumnInfo(
                f"M{i}",
                f"match_flag_{i}",
                "transaction",
                "match flags",
                "Additional match between cardholder and transaction details.",
                "T/F. M6 is almost always filled on ProductCD=W. M7–M9 are sparse and share missingness.",
                "Official match family; T/F values.",
                "medium",
                True,
            )
        )
    return rows


def _v_features() -> list[ColumnInfo]:
    return [
        ColumnInfo(
            f"V{i}",
            f"vesta_engineered_{i:03d}",
            "transaction",
            "Vesta engineered",
            "Pre-built ranking, counting, and entity-relation scores from Vesta.",
            "Keep as a block. Individual V columns are not documented and many are redundant.",
            "Official: ranking, counting, and other entity relations. 339 columns.",
            "low",
        )
        for i in range(1, 340)
    ]


TRANSACTION: list[ColumnInfo] = [
    ColumnInfo(
        "TransactionID",
        "transaction_id",
        "transaction",
        "keys",
        "Join key to the identity table.",
        "Unique transaction identifier.",
        "Official join key.",
        "high",
    ),
    ColumnInfo(
        "isFraud",
        "is_fraud",
        "transaction",
        "label",
        "Competition target.",
        "1 if Vesta labeled the payment as fraud.",
        "Official label.",
        "high",
    ),
    ColumnInfo(
        "TransactionDT",
        "seconds_from_reference",
        "transaction",
        "time",
        "When the payment happened, relative to a hidden start date.",
        "Seconds since a fixed reference (not a Unix timestamp). Train starts at 86400 (1 day).",
        "Official: timedelta from a reference datetime.",
        "high",
    ),
    ColumnInfo(
        "TransactionAmt",
        "amount_usd",
        "transaction",
        "payment",
        "How much was charged.",
        "Transaction amount in USD.",
        "Official.",
        "high",
    ),
    ColumnInfo(
        "ProductCD",
        "product_channel",
        "transaction",
        "payment",
        "What kind of purchase / acceptance channel this is.",
        "W ≈ card-present / retail (has billing addr, dist1, M matches; no recipient email). "
        "C ≈ card-not-present / cross-border e-comm (addr missing ~95%, has recipient email, ~12% fraud). "
        "H/R/S ≈ digital / remote products (no dist1; H and R have identity rows more often).",
        "5 codes W/H/C/R/S; missingness of addr/dist/M/R_email splits cleanly by code.",
        "high",
        True,
    ),
    ColumnInfo(
        "card1",
        "card_id",
        "transaction",
        "payment card",
        "Which payment card was used (anonymized).",
        "Highest-cardinality card token. Closest thing to a card account id.",
        "6k+ unique ints, never missing.",
        "high",
    ),
    ColumnInfo(
        "card2",
        "card_issuer",
        "transaction",
        "payment card",
        "Issuing bank / BIN grouping.",
        "Medium-cardinality numeric code (~500 values). Likely issuer or BIN fragment.",
        "Official: issue bank among card1–card6. ~500 distinct values.",
        "medium",
    ),
    ColumnInfo(
        "card3",
        "card_issue_country",
        "transaction",
        "payment card",
        "Country that issued the card.",
        "150 dominates (US-like). 185 is the main foreign code and lines up with ProductCD=C.",
        "Official: country. ~70 codes; 150 ≈ 89% of rows.",
        "high",
    ),
    ColumnInfo(
        "card4",
        "card_network",
        "transaction",
        "payment card",
        "Card scheme.",
        "visa / mastercard / american express / discover.",
        "Raw string values.",
        "high",
        True,
    ),
    ColumnInfo(
        "card5",
        "card_bin_or_product",
        "transaction",
        "payment card",
        "Card product / BIN category (classic vs platinum, etc.).",
        "Low-cardinality numeric (~85). Not the same as card2.",
        "Official: card category. Distinct from card2 cardinality.",
        "medium",
    ),
    ColumnInfo(
        "card6",
        "card_funding_type",
        "transaction",
        "payment card",
        "How the card is funded.",
        "debit / credit (rare: charge card, debit or credit).",
        "Raw string values.",
        "high",
        True,
    ),
    ColumnInfo(
        "addr1",
        "billing_region",
        "transaction",
        "location",
        "Billing ZIP / region (anonymized).",
        "Numeric region code (~260 values). Missing with addr2; almost never present for ProductCD=C.",
        "Official address field; ZIP-like cardinality.",
        "high",
        True,
    ),
    ColumnInfo(
        "addr2",
        "billing_country",
        "transaction",
        "location",
        "Billing country.",
        "87 is the US-like majority (~96% of non-null). Missing with addr1.",
        "Official address; country-like cardinality (tens of codes).",
        "high",
        True,
    ),
    ColumnInfo(
        "dist1",
        "distance_billing_to_counterparty",
        "transaction",
        "location",
        "Distance between billing address and a W-channel counterparty (merchant / shipping).",
        "Only populated for ProductCD=W. Lots of 0–10 values (local) plus a long tail.",
        "100% missing except ProductCD=W.",
        "medium",
    ),
    ColumnInfo(
        "dist2",
        "distance_alt_addresses",
        "transaction",
        "location",
        "Distance for digital / remote products (not W).",
        "Never filled on W; sparse on C/H/R/S.",
        "Complementary missingness vs dist1.",
        "medium",
    ),
    ColumnInfo(
        "P_emaildomain",
        "purchaser_email_domain",
        "transaction",
        "identity (email)",
        "Purchaser email provider.",
        "gmail.com, yahoo.com, anonymous.com, hotmail.com, …",
        "Official.",
        "high",
        True,
    ),
    ColumnInfo(
        "R_emaildomain",
        "recipient_email_domain",
        "transaction",
        "identity (email)",
        "Recipient email provider (digital goods / person-to-person).",
        "Almost never present on ProductCD=W; nearly always present on C and R.",
        "Official; ProductCD split confirms recipient vs purchaser.",
        "high",
        True,
    ),
    *_c_features(),
    *_d_features(),
    *_m_features(),
    *_v_features(),
]


IDENTITY: list[ColumnInfo] = [
    ColumnInfo(
        "TransactionID",
        "transaction_id",
        "identity",
        "keys",
        "Join key. Only ~24% of train transactions have an identity row.",
        "Same id as the transaction table.",
        "Official.",
        "high",
    ),
    ColumnInfo(
        "id_01",
        "identity_risk_score",
        "identity",
        "identity scores",
        "Device / identity risk score from Vesta partners.",
        "Always ≤ 0 (0 is best). Typical values −5 / −10; −100 is worst.",
        "Range [−100, 0]; never missing on identity rows.",
        "medium",
    ),
    ColumnInfo(
        "id_02",
        "device_or_session_id",
        "identity",
        "identity scores",
        "Numeric device / session token.",
        "Almost unique per row (~27k distinct in 30k). Looks like an anonymized device id, not a score.",
        "High cardinality, range ~1k–1e6.",
        "medium",
    ),
    *[
        ColumnInfo(
            f"id_{i:02d}",
            f"identity_score_{i:02d}",
            "identity",
            "identity scores",
            "Masked numeric identity / device score.",
            "Small integer scores. id_07/id_08 are ~96% missing and travel with the proxy block (id_21–id_27).",
            "Official: identity table is IP/ISP/proxy and UA/browser/OS.",
            "low",
        )
        for i in (3, 4, 5, 6, 7, 8, 9, 10)
    ],
    ColumnInfo(
        "id_11",
        "device_trust_pct",
        "identity",
        "identity scores",
        "Device recognition / trust percentage.",
        "Almost always 100; floor is 90.",
        "Range [90, 100].",
        "medium",
    ),
    ColumnInfo(
        "id_12",
        "device_seen_before",
        "identity",
        "lookup flags",
        "Whether this device/id was already in the partner database.",
        "Found / NotFound.",
        "Raw strings.",
        "high",
        True,
    ),
    ColumnInfo(
        "id_13",
        "device_os_group",
        "identity",
        "network / device codes",
        "Masked OS or device-group code.",
        "Integer code (~40 values). Not human-readable.",
        "Low-cardinality int next to OS fields.",
        "low",
        True,
    ),
    ColumnInfo(
        "id_14",
        "timezone_offset_minutes",
        "identity",
        "network / device",
        "Browser timezone offset from UTC.",
        "−300 = UTC−5, −360 = UTC−6, −480 = UTC−8. Matches US timezones.",
        "Values are multiples of 60 minutes.",
        "high",
    ),
    ColumnInfo(
        "id_15",
        "identity_match_status",
        "identity",
        "lookup flags",
        "Did the current identity match a known profile.",
        "Found / New / Unknown.",
        "Raw strings.",
        "high",
        True,
    ),
    ColumnInfo(
        "id_16",
        "identity_in_database",
        "identity",
        "lookup flags",
        "Second Found/NotFound lookup (related to id_15).",
        "Found / NotFound. Found lines up with id_15=Found.",
        "Raw strings; tracks id_15.",
        "medium",
        True,
    ),
    ColumnInfo(
        "id_17",
        "ip_or_browser_country",
        "identity",
        "network / device codes",
        "Country associated with IP or browser locale.",
        "166 and 225 dominate. Same style of numeric country code as card3/addr2.",
        "Country-like cardinality; two large modes.",
        "medium",
        True,
    ),
    ColumnInfo(
        "id_18",
        "connection_type_code",
        "identity",
        "network / device codes",
        "Masked connection / partner code.",
        "Low cardinality, 69% missing.",
        "13 distinct ints.",
        "low",
        True,
    ),
    ColumnInfo(
        "id_19",
        "browser_or_isp_code",
        "identity",
        "network / device codes",
        "Masked browser or ISP identifier.",
        "~400 numeric codes. Not readable.",
        "Medium cardinality ints.",
        "low",
        True,
    ),
    ColumnInfo(
        "id_20",
        "isp_or_proxy_code",
        "identity",
        "network / device codes",
        "Masked ISP / egress identifier.",
        "~200 numeric codes.",
        "Medium cardinality ints.",
        "low",
        True,
    ),
    *[
        ColumnInfo(
            f"id_{i}",
            f"proxy_attr_{i}",
            "identity",
            "proxy block",
            "Proxy / anonymizer attributes (same rows as id_23).",
            "96% missing. Filled only when a proxy was detected.",
            "Missingness matches id_23 exactly.",
            "low",
            True,
        )
        for i in (21, 22, 24, 25, 26)
    ],
    ColumnInfo(
        "id_23",
        "ip_proxy_type",
        "identity",
        "proxy block",
        "Whether the IP is a known proxy and how hidden it is.",
        "IP_PROXY:TRANSPARENT / ANONYMOUS / HIDDEN. Sparse (~4% filled).",
        "Raw strings.",
        "high",
        True,
    ),
    ColumnInfo(
        "id_27",
        "proxy_in_database",
        "identity",
        "proxy block",
        "Proxy listing lookup.",
        "Found / NotFound; same missingness as id_23.",
        "Raw strings; co-missing with proxy block.",
        "high",
        True,
    ),
    ColumnInfo(
        "id_28",
        "device_match_status",
        "identity",
        "lookup flags",
        "Device seen as Found vs New this session.",
        "Found / New.",
        "Raw strings.",
        "high",
        True,
    ),
    ColumnInfo(
        "id_29",
        "device_in_database",
        "identity",
        "lookup flags",
        "Device Found/NotFound counterpart to id_28.",
        "Found / NotFound.",
        "Raw strings.",
        "high",
        True,
    ),
    ColumnInfo(
        "id_30",
        "os_name_version",
        "identity",
        "digital signature",
        "Operating system and version (user agent).",
        "Windows 10, iOS 11.2.1, Android 7.0, Mac OS X 10_12_6, …",
        "Raw OS strings.",
        "high",
        True,
    ),
    ColumnInfo(
        "id_31",
        "browser_name_version",
        "identity",
        "digital signature",
        "Browser and version (user agent).",
        "chrome 63.0, mobile safari 11.0, ie 11.0 for desktop, …",
        "Raw browser strings.",
        "high",
        True,
    ),
    ColumnInfo(
        "id_32",
        "screen_color_depth",
        "identity",
        "digital signature",
        "Display color depth in bits.",
        "24 / 32 / 16 (very rare 0).",
        "Classic color-depth values.",
        "high",
    ),
    ColumnInfo(
        "id_33",
        "screen_resolution",
        "identity",
        "digital signature",
        "Screen width × height.",
        "Strings like 1920x1080, 1334x750.",
        "WxH pattern.",
        "high",
        True,
    ),
    ColumnInfo(
        "id_34",
        "device_match_grade",
        "identity",
        "lookup flags",
        "How strongly the device matched a known fingerprint.",
        "match_status:2 / 1 / 0 / −1.",
        "Raw match_status strings.",
        "high",
        True,
    ),
    *[
        ColumnInfo(
            f"id_{i}",
            f"device_flag_{i}",
            "identity",
            "digital signature flags",
            "Masked T/F device or browser capability (cookies, JS, flash, …).",
            "T/F with the same missingness as DeviceType. Specific capability is not published.",
            "Boolean; official list says id_12–id_38 are categorical identity fields.",
            "low",
            True,
        )
        for i in (35, 36, 37, 38)
    ],
    ColumnInfo(
        "DeviceType",
        "device_type",
        "identity",
        "digital signature",
        "Desktop vs mobile.",
        "desktop / mobile.",
        "Official + raw values.",
        "high",
        True,
    ),
    ColumnInfo(
        "DeviceInfo",
        "device_info",
        "identity",
        "digital signature",
        "Raw device / UA leftover string.",
        "Windows, iOS Device, MacOS, Trident/7.0, Samsung SM-* build strings.",
        "Official.",
        "high",
        True,
    ),
]

ALL_COLUMNS: list[ColumnInfo] = [*TRANSACTION, *IDENTITY]

_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def column_frame(table: TableName | None = None) -> pd.DataFrame:
    rows = ALL_COLUMNS
    if table:
        rows = [c for c in rows if c.table == table]
    return pd.DataFrame([c.__dict__ for c in rows])


def rename_map(
    table: TableName | None = None,
    *,
    min_confidence: Confidence = "medium",
    include_families: Iterable[str] | None = None,
) -> dict[str, str]:
    """Map original -> interpretable name.

    ``min_confidence='high'`` only renames fields we can read from values.
    ``'medium'`` also renames well-supported inferences (issuer, distances, …).
    ``'low'`` also prefixes C/D/V/masked-id blocks — useful for grouping, not semantics.
    """
    allowed = {k for k, r in _CONFIDENCE_RANK.items() if r <= _CONFIDENCE_RANK[min_confidence]}
    families = set(include_families) if include_families else None
    out: dict[str, str] = {}
    for col in ALL_COLUMNS:
        if table and col.table != table:
            continue
        if col.confidence not in allowed:
            continue
        if families is not None and col.family not in families:
            continue
        # Identity and transaction both have TransactionID -> transaction_id
        out[col.original] = col.renamed
    return out


def rename_ieee_columns(
    df: pd.DataFrame,
    table: TableName,
    *,
    min_confidence: Confidence = "medium",
) -> pd.DataFrame:
    """Return a copy with interpretable names. Unknown columns are left as-is."""
    return df.rename(columns=rename_map(table, min_confidence=min_confidence))


def official_categoricals(table: TableName) -> list[str]:
    if table == "transaction":
        return [
            "ProductCD",
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "P_emaildomain",
            "R_emaildomain",
            *[f"M{i}" for i in range(1, 10)],
        ]
    return ["DeviceType", "DeviceInfo", *[f"id_{i}" for i in range(12, 39)]]

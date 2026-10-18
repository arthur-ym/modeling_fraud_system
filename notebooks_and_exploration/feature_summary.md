# IEEE-CIS feature summary

Reference for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) tables used in this project.

The target is **`isFraud`**: probability that an online payment is fraudulent. Data split into **transaction** and **identity**, joined on **`TransactionID`**. Not every transaction has identity (~24% of train).

The main documentation published **families**, not a pairwise dictionary. Two layers below:

1. **Official** — information that was confirmed by the dataset onwner.
2. **Our reading** — names and meanings we are willing to use after looking at values and missingness. Confidence is `high` / `medium` / `low`. We rename `high` only. Keep `C*`, most `D*`, `V*`, and most `id_*` codes as family prefixes — do not invent “C1 = address count”.

The machine-readable map lives in [`ieee_column_dictionary.py`](ieee_column_dictionary.py) (`rename_ieee_columns()`).

---

## Files

| File | Role |
| --- | --- |
| `train_transaction.csv` / `test_transaction.csv` | Payment, card, address, counts, timedeltas, match flags, Vesta `V*` |
| `train_identity.csv` / `test_identity.csv` | Device / network / digital signature. Test uses `id-01` style names; normalize to `id_01`. |

**`TransactionDT`** is a timedelta from a hidden reference datetime, **not** a Unix timestamp. Train starts at `86400` (one day).

---

## Official description (Vesta)

### Transaction table

| Family | Columns | Official meaning |
| --- | --- | --- |
| Time | `TransactionDT` | Timedelta from a given reference datetime |
| Payment | `TransactionAmt` | Amount in USD |
| Payment | `ProductCD` | Product code for each transaction |
| Card | `card1`–`card6` | Card type, category, issue bank, country, etc. |
| Location | `addr1`, `addr2` | Address |
| Location | `dist1`, `dist2` | Distance |
| Email | `P_emaildomain`, `R_emaildomain` | Purchaser and recipient email domain |
| Counts | `C1`–`C14` | Counting (e.g. how many addresses tied to the card). **Masked.** |
| Timedelta | `D1`–`D15` | Timedelta (e.g. days since previous transaction) |
| Match | `M1`–`M9` | Match (e.g. names on card and address) |
| Engineered | `V1`–`V339` | Vesta ranking, counting, entity relations |

**Categorical (transaction):** `ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`.

### Identity table

Identity is network connection (IP, ISP, proxy, …) and digital signature (UA / browser / OS / version, …) collected by Vesta and partners. Field names are masked; no pairwise dictionary.

**Categorical (identity):** `DeviceType`, `DeviceInfo`, `id_12`–`id_38`.

---

## How to read our names

| Confidence | Use as |
| --- | --- |
| **high** | Values or the official writeup make the meaning obvious. Safe to rename. |
| **medium** | Strong inference from cardinality / missingness / co-occurrence. Rename, but treat as an assumption. |
| **low** | Family only. Do not assign a specific story to `C3` or `V127`. |

`ProductCD` is a **channel**, not a SKU. On train:

| Code | Working interpretation | Typical pattern |
| --- | --- | --- |
| **W** | Card-present / retail | Billing addr, `dist1`, M matches; almost never `R_emaildomain` |
| **C** | Card-not-present / cross-border e-comm | `addr` missing ~95%, has recipient email, ~12% fraud |
| **H / R / S** | Digital / remote | No `dist1`; H and R have identity rows more often |

---

## Transaction features (interpretable)

### Keys, label, time, payment

| Original | Renamed | Sample | Family | What it is | Notes | Conf. |
| --- | --- | --- | --- | --- | --- | --- |
| `TransactionID` | `transaction_id` | 2987000, 2987001 | keys | Join key to identity | Unique transaction id | high |
| `isFraud` | `is_fraud` | 0, 1 | label | Competition target | 1 if Vesta labeled the payment as fraud | high |
| `TransactionDT` | `seconds_from_reference` | 86400, 86401 | time | When the payment happened, relative to a hidden start | Seconds since a fixed reference. Not Unix time. Train starts at 86400 | high |
| `TransactionAmt` | `amount_usd` | 68.5, 29, 59 | payment | How much was charged | USD | high |
| `ProductCD` | `product_channel` | W, H, C | payment | Purchase / acceptance channel | See table above | high |

Test min minus train max is exactly 30 days. TransactionID also does not overlap (train ends at 3,577,539; test starts at 3,663,549).

So this is a time split, not a random split: ~6 months train, 1 month blank, ~6 months test.
### Payment card

| Original | Renamed | Sample | What it is | Notes | Conf. |
| --- | --- | --- | --- | --- | --- |
| `card1` | `card_id` | 13926, 2755, 4663 | Anonymized card token | Highest cardinality. Closest thing to a card account id | high |
| `card2` | `card_issuer` | 404, 490, 567 | Issuing bank / BIN grouping | ~500 numeric codes. Likely issuer or BIN fragment | medium |
| `card3` | `card_issue_country` | 150, 117, 185 | Country that issued the card | 150 dominates (US-like). 185 is the main foreign code and lines up with `ProductCD=C` | high |
| `card4` | `card_network` | visa, mastercard | Card scheme | visa / mastercard / american express / discover | high |
| `card5` | `card_bin_or_product` | 142, 102, 166 | Card product / BIN category | ~85 codes. Not the same as `card2` | medium |
| `card6` | `card_funding_type` | credit, debit | How the card is funded | debit / credit (rare: charge card, debit or credit) | high |

### Location and email

| Original | Renamed | Sample | What it is | Notes | Conf. |
| --- | --- | --- | --- | --- | --- |
| `addr1` | `billing_region` | 315, 325, 330 | Billing ZIP / region (anonymized) | ~260 codes. Missing with `addr2`; almost never present for `ProductCD=C` | high |
| `addr2` | `billing_country` | 87, 96 | Billing country | 87 is the US-like majority (~96% of non-null) | high |
| `dist1` | `distance_billing_to_counterparty` | 19, 287, 36 | Distance billing ↔ W-channel counterparty (merchant / shipping) | Only on `ProductCD=W`. Many 0–10 (local) plus a long tail | medium |
| `dist2` | `distance_alt_addresses` | 30, 98, 149 | Distance for digital / remote products | Never filled on W; sparse on C/H/R/S | medium |
| `P_emaildomain` | `purchaser_email_domain` | gmail.com, yahoo.com | Purchaser email provider | gmail, yahoo, anonymous.com, hotmail, … | high |
| `R_emaildomain` | `recipient_email_domain` | gmail.com, hotmail.com | Recipient email (digital goods / P2P) | Almost never on W; nearly always on C and R | high |

### Recency (`D*`) we actually name

The rest of `D1`–`D15` stay as masked timedeltas (see [Masked families](#masked-families)).

| Original | Renamed | Sample | Meaning | Conf. |
| --- | --- | --- | --- | --- |
| `D1` | `days_since_card_first_seen` | 14, 0, 112 | Days from a reference event on this card (often first seen). Near 0 for new cards | medium |
| `D9` | `transaction_time_of_day` | 0, 0.0417, 0.0833 | Time of day as a fraction of 24h (`hour / 24`) | high |

### Match flags (`M1`–`M9`)

Official: match such as names on card and address. All **medium** — we know they are match checks, not which field each one compares.

| Original | Renamed | Sample | Notes |
| --- | --- | --- | --- |
| `M1` | `match_name_or_address_1` | T | Almost never F. Same missingness as M2/M3; **W only** |
| `M2` | `match_name_or_address_2` | T, F | Same missingness as M1/M3; **W only** |
| `M3` | `match_name_or_address_3` | T, F | Same missingness as M1/M2; **W only** |
| `M4` | `match_type_code` | M0, M1, M2 | Not T/F. Present mainly on W and C |
| `M5` | `match_flag_5` | F, T | T/F |
| `M6` | `match_flag_6` | T, F | Almost always filled on W |
| `M7` | `match_flag_7` | F, T | Sparse; shares missingness with M8/M9 |
| `M8` | `match_flag_8` | F, T | Sparse; shares missingness with M7/M9 |
| `M9` | `match_flag_9` | F, T | Sparse; shares missingness with M7/M8 |

---

## Identity features (interpretable)

Only ~24% of train payments have a matching identity row. W (in-person) never does; C / H / R / S usually do.

| Original | Renamed | Sample | Family | What it is | Notes | Conf. |
| --- | --- | --- | --- | --- | --- | --- |
| `TransactionID` | `transaction_id` | 2987004, … | keys | Join key | Same id as the transaction table | high |
| `id_01` | `identity_risk_score` | 0, −5, −15 | identity scores | Device / identity risk from partners | Always ≤ 0 (0 is best). Typical −5 / −10; −100 is worst | medium |
| `id_02` | `device_or_session_id` | 70787, 98945 | identity scores | Numeric device / session token | Almost unique per row. Looks like an anonymized device id, not a score | medium |
| `id_11` | `device_trust_pct` | 100, 93.75 | identity scores | Device recognition / trust % | Almost always 100; floor is 90 | medium |
| `id_12` | `device_seen_before` | NotFound, Found | lookup flags | Device/id already in the partner DB | Found / NotFound | high |
| `id_14` | `timezone_offset_minutes` | −480, −300, −360 | network / device | Browser TZ offset from UTC | −300 = UTC−5, −360 = UTC−6, −480 = UTC−8 | high |
| `id_15` | `identity_match_status` | New, Found, Unknown | lookup flags | Current identity vs known profile | Found / New / Unknown | high |
| `id_16` | `identity_in_database` | NotFound, Found | lookup flags | Second Found/NotFound (related to `id_15`) | Found lines up with `id_15=Found` | medium |
| `id_17` | `ip_or_browser_country` | 166, 121, 225 | network / device codes | Country of IP or browser locale | 166 and 225 dominate. Same style as `card3` / `addr2` | medium |
| `id_23` | `ip_proxy_type` | IP_PROXY:TRANSPARENT, … | proxy block | Known proxy and how hidden | TRANSPARENT / ANONYMOUS / HIDDEN. Sparse (~4% filled) | high |
| `id_27` | `proxy_in_database` | Found, NotFound | proxy block | Proxy listing lookup | Same missingness as `id_23` | high |
| `id_28` | `device_match_status` | New, Found | lookup flags | Device Found vs New this session | Found / New | high |
| `id_29` | `device_in_database` | NotFound, Found | lookup flags | Device Found/NotFound counterpart to `id_28` | Found / NotFound | high |
| `id_30` | `os_name_version` | Android 7.0, iOS 11.1.2 | digital signature | OS and version (UA) | Windows 10, iOS, Android, Mac OS X, … | high |
| `id_31` | `browser_name_version` | chrome 62.0, mobile safari 11.0 | digital signature | Browser and version | chrome, mobile safari, ie 11.0, … | high |
| `id_32` | `screen_color_depth` | 32, 24, 16 | digital signature | Display color depth (bits) | 24 / 32 / 16 (very rare 0) | high |
| `id_33` | `screen_resolution` | 1920x1080, 1334x750 | digital signature | Screen width × height | Strings like `2220x1080` | high |
| `id_34` | `device_match_grade` | match_status:2, :1, :0 | lookup flags | How strongly the device matched a known fingerprint | match_status 2 / 1 / 0 / −1 | high |
| `DeviceType` | `device_type` | mobile, desktop | digital signature | Desktop vs mobile | desktop / mobile | high |
| `DeviceInfo` | `device_info` | Windows, iOS Device, SAMSUNG SM-… | digital signature | Raw device / UA leftover | Windows, MacOS, Trident/7.0, Samsung build strings | high |

---

## Masked families

Do not invent column-level meanings. Use them as blocks (aggregates, PCA, missingness indicators).

| Family | Columns | Official | Why they stay masked |
| --- | --- | --- | --- |
| Entity counts | `C1`–`C14` | Counting, e.g. addresses on the card | Always present, non-negative, highly skewed. No pairwise labels |
| Other timedeltas | `D2`–`D8`, `D10`–`D15` | Days since a previous event for some entity | A few allow small negatives (clock / TZ artifacts). Only `D1` and `D9` are named |
| Vesta engineered | `V1`–`V339` | Ranking, counting, entity relations | 339 columns, many redundant. Keep as a block |
| Other identity scores | `id_03`–`id_10` | Masked numeric scores | Small integers. `id_07`/`id_08` ~96% missing with the proxy block |
| Device/OS group | `id_13` | — | ~40 integer codes, not readable |
| Connection / ISP codes | `id_18`, `id_19`, `id_20` | — | Low/medium cardinality ints; `id_18` ~69% missing |
| Proxy attributes | `id_21`, `id_22`, `id_24`, `id_25`, `id_26` | — | 96% missing; filled only when a proxy was detected (`id_23` block) |
| Device flags | `id_35`–`id_38` | Categorical identity | T/F with the same missingness as `DeviceType`. Specific capability (cookies, JS, …) is unpublished |

In the dictionary these are prefixed (`entity_count_01`, `timedelta_days_02`, `vesta_engineered_001`, …) only when `min_confidence="low"`.

---

## Information types (for modeling)

| Kind | Columns | What you know |
| --- | --- | --- |
| Payment & card | `TransactionAmt`, `ProductCD`, `card1`–`card6` | Amount, channel, card token, issuer, issue country, network, BIN/product, debit vs credit |
| Location & email | `addr1/2`, `dist1/2`, `P_/R_emaildomain` | Billing region/country, distance (W vs digital), purchaser vs recipient email |
| Entity history | `C1`–`C14`, `D1`–`D15`, `M1`–`M9`, `V1`–`V339` | Counts, day recency (`D9` = time of day), match flags, Vesta graph scores |
| Device / network | identity table (~24% of train) | OS, browser, screen, timezone, seen-before, IP country, sparse proxy block |
| Label / time / key | `isFraud`, `TransactionDT`, `TransactionID` | Target; seconds from a hidden start; join key |

Identity coverage is not random: it tracks `ProductCD` (more H/R/C, almost none on in-person W). Missing identity is itself a feature, not MCAR.

---

## Related notebooks

- [`EDA/Download and Initial_EDA.ipynb`](EDA/Download%20and%20Initial_EDA.ipynb) — download, join rates, column dictionary
- [`EDA/Joining and etc.ipynb`](EDA/Joining%20and%20etc.ipynb) — identity join vs `ProductCD` missingness
- [`EDA/Extensive_EDA.ipynb`](EDA/Extensive_EDA.ipynb) — distributions, amounts, card / C / id plots

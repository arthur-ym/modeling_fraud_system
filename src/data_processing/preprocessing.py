"""First-pass cleaning for the joined IEEE transaction + identity frame."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from src.run_time_configuration import BaseConfigParams, last_day_of_month

# Reference date for the transaction datetime.
TRANSACTION_DT_ORIGIN = "2017-12-01"

# Mapping of original column names to new names.
COLUMN_RENAME: dict[str, str] = {
    "TransactionID": "transaction_id",
    "isFraud": "is_fraud",
    "TransactionDT": "seconds_from_reference",
    "TransactionAmt": "amount_usd",
    "ProductCD": "product_channel",
    "card3": "card_issue_country",
    "card4": "card_network",
    "card6": "card_funding_type",
    "addr1": "billing_region",
    "addr2": "billing_country",
    "P_emaildomain": "purchaser_email_domain",
    "R_emaildomain": "recipient_email_domain",
    "D9": "transaction_time_of_day",
    "id_12": "device_seen_before",
    "id_14": "timezone_offset_minutes",
    "id_15": "identity_match_status",
    "id_23": "ip_proxy_type",
    "id_27": "proxy_in_database",
    "id_28": "device_match_status",
    "id_29": "device_in_database",
    "id_30": "os_name_version",
    "id_31": "browser_name_version",
    "id_32": "screen_color_depth",
    "id_33": "screen_resolution",
    "id_34": "device_match_grade",
    "DeviceType": "device_type",
    "DeviceInfo": "device_info",
}

# Vesta uses this domain as a placeholder, not a real mailbox provider.
PLACEHOLDER_EMAIL_DOMAINS = {"anonymous.com"}

# card5 is the BIN-like code. It stays under its original name (medium confidence).
AMOUNT_GROUP_COLUMNS = ("card5", "product_channel", "card_network")

# Categorical columns whose training fraud rate is a feature.
# card1, card2, and card5 keep their original names.
TARGET_RATE_SOURCE_COLUMNS: tuple[str, ...] = (
    "product_channel",
    "card1",
    "card2",
    "card5",
    "card_issue_country",
    "billing_region",
    "purchaser_email_domain",
    "recipient_email_domain",
)

# Pairs joined before the split, then target-encoded with the singles.
# card1 is paired with the channel. Billing region is paired with the channel
# and with the issuing country. The two email domains are paired with each other.
CAT_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("card1", "product_channel"),
    ("card2", "card5"),
    ("card2", "product_channel"),
    ("card5", "product_channel"),
    ("card5", "card_issue_country"),
    ("card_issue_country", "product_channel"),
    ("billing_region", "product_channel"),
    ("billing_region", "card_issue_country"),
    ("purchaser_email_domain", "product_channel"),
    ("recipient_email_domain", "product_channel"),
    ("purchaser_email_domain", "recipient_email_domain"),
)

# Same groups as the extensive EDA. Anything else that is filled is "other".
EMAIL_PROVIDER_GROUPS: dict[str, set[str]] = {
    "Google": {"gmail.com", "gmail"},
    "Yahoo Mail": {
        "yahoo.com",
        "yahoo.com.mx",
        "yahoo.co.uk",
        "yahoo.co.jp",
        "yahoo.de",
        "yahoo.fr",
        "yahoo.es",
    },
    "Microsoft": {
        "hotmail.com",
        "outlook.com",
        "msn.com",
        "live.com.mx",
        "hotmail.es",
        "hotmail.co.uk",
        "hotmail.de",
        "outlook.es",
        "live.com",
        "live.fr",
        "hotmail.fr",
    },
}

# T/F match checks that actually vary. M1 is almost never F. Meaningful on ProductCD W.
MATCH_FAILURE_COLUMNS = ("M2", "M3", "M5", "M6", "M7", "M8", "M9")

# card3 value 150 is the US-like majority. dist1 values 0–10 read as local.
US_LIKE_CARD_COUNTRY = 150
LOCAL_BILLING_DISTANCE = 10

# Past-only stats on card_id. Each pair is (name after rename, raw name).
# Mean and std of amount and of the day-deltas the 0.96 kernel grouped by client.
CARD_HISTORY_MEAN_STD_COLUMNS: tuple[tuple[str, str | None], ...] = (
    ("amount_usd", "TransactionAmt"),
    ("D4", None),
    ("transaction_time_of_day", "D9"),
    ("D10", None),
    ("D15", None),
)
# Mean of the count columns except C3, and of the match flags.
CARD_HISTORY_MEAN_COLUMNS: tuple[tuple[str, str | None], ...] = tuple(
    (f"C{index}", None) for index in range(1, 15) if index != 3
) + tuple((f"M{index}", None) for index in range(1, 10))
# C14 also gets a past standard deviation.
CARD_HISTORY_STD_COLUMNS: tuple[tuple[str, str | None], ...] = (("C14", None),)
# How many distinct values this client has already shown.
CARD_HISTORY_NUNIQUE_COLUMNS: tuple[tuple[str, str | None], ...] = (
    ("purchaser_email_domain", "P_emaildomain"),
    ("dist1", None),
    ("id_02", None),
    ("amount_cents", "cents"),
    ("C13", None),
    ("V314", None),
    ("V127", None),
    ("V136", None),
    ("V309", None),
    ("V307", None),
    ("V320", None),
)

# Inside one V missingness block, a column this correlated with an earlier one is a copy.
V_CORRELATION_MAX = 0.75

def normalize_ieee_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df

def rename_columns(
    df: pd.DataFrame,
    column_rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rename columns whose meaning is already settled.

    ``column_rename`` defaults to ``COLUMN_RENAME``. Pass another dict to
    extend or replace that map without editing the function.
    """
    mapping = COLUMN_RENAME if column_rename is None else column_rename
    return df.rename(columns=mapping)


def add_transaction_datetime(
    df: pd.DataFrame,
    seconds_column: str = "seconds_from_reference",
    origin: str = TRANSACTION_DT_ORIGIN,
) -> pd.DataFrame:
    """Turn the relative seconds column into a datetime plus calendar parts.

    Same anchor as the EDA notebook (``2017-12-01``). ``weekdays`` follows
    pandas ``dayofweek`` (Monday = 0, Sunday = 6). ``flag_is_weekend`` is
    Saturday or Sunday.
    """
    if seconds_column not in df.columns and "TransactionDT" in df.columns:
        seconds_column = "TransactionDT"
    if seconds_column not in df.columns:
        raise KeyError(
            f"Expected '{seconds_column}' or 'TransactionDT' to build the transaction datetime."
        )

    out = df.copy()
    out["transaction_datetime"] = pd.Timestamp(origin) + pd.to_timedelta(
        out[seconds_column], unit="s"
    )
    out["weekdays"] = out["transaction_datetime"].dt.dayofweek
    out["hours"] = out["transaction_datetime"].dt.hour
    out["flag_is_weekend"] = out["weekdays"].ge(5)
    return out


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def _datetime_int(series: pd.Series) -> tuple[np.ndarray, int]:
    """Unix time as int64, plus how many of those units are in one second."""
    timestamps = pd.to_datetime(series, utc=True)
    unit = getattr(timestamps.dtype, "unit", "ns")
    per_second = int(pd.Timedelta(seconds=1) / pd.Timedelta(1, unit=unit))
    return timestamps.astype("int64").to_numpy(), per_second


def _column_name(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise KeyError(f"Expected '{preferred}' or '{fallback}'.")


def add_card_id(
    df: pd.DataFrame,
    card_column: str = "card1",
    region_column: str = "billing_region",
    days_column: str = "D1",
    seconds_column: str = "seconds_from_reference",
) -> pd.DataFrame:
    """Client fingerprint used in place of the raw card token.

    ``card1`` is shared by many clients. ``D1`` is days since that card was
    first seen, so it changes every day and cannot be part of an id. The
    first-seen day stays fixed:

    ``D1n = floor(seconds_from_reference / 86400) - D1``

    ``card_id`` is ``card1``, billing region (``addr1``), and ``D1n`` joined
    with underscores. A missing piece is the string ``nan``.
    """
    card_column = _column_name(df, card_column, "card1")
    region_column = _column_name(df, region_column, "addr1")
    seconds_column = _column_name(df, seconds_column, "TransactionDT")
    _require_columns(df, (card_column, region_column, days_column, seconds_column))

    out = df.copy()
    seconds = pd.to_numeric(out[seconds_column], errors="coerce")
    days_since_first_seen = pd.to_numeric(out[days_column], errors="coerce")
    first_seen_day = np.floor(seconds / 86400) - days_since_first_seen
    out["card_id"] = (
        out[card_column].astype(str)
        + "_"
        + out[region_column].astype(str)
        + "_"
        + first_seen_day.astype(str)
    )
    return out


def add_same_card_window_features(
    df: pd.DataFrame,
    card_column: str = "card_id",
    time_column: str = "transaction_datetime",
    window: str = "10min",
) -> pd.DataFrame:
    """Activity for one ``card_id`` in the previous 10 minutes.

    ``card_id`` is the client fingerprint from ``add_card_id``, not the raw
    ``card1`` token. Looks backward only, so it can run before the
    train/validation/test split. Adds two columns:

    - ``card_txn_count_10min``: how many transactions this ``card_id`` has in the
      window, including the current row. An isolated payment is 1.
    - ``card_txn_delta_10min``: seconds from the earliest other transaction with
      the same ``card_id`` in that window back to now. Missing when the count is 1.
    """
    _require_columns(df, (card_column, time_column))
    out = df.copy()
    count = np.ones(len(out), dtype=np.int32)
    delta = np.full(len(out), np.nan)

    time_int, per_second = _datetime_int(out[time_column])
    window_units = int(pd.Timedelta(window) / pd.Timedelta(seconds=1) * per_second)
    card_codes, _ = pd.factorize(out[card_column], use_na_sentinel=True)
    valid_time = pd.to_datetime(out[time_column], utc=True).notna().to_numpy()
    valid_idx = np.flatnonzero((card_codes >= 0) & valid_time)
    if len(valid_idx) == 0:
        out["card_txn_count_10min"] = count
        out["card_txn_delta_10min"] = delta
        return out

    order = valid_idx[np.lexsort((time_int[valid_idx], card_codes[valid_idx]))]
    ordered_cards = card_codes[order]
    ordered_time = time_int[order]
    group_ends = np.flatnonzero(ordered_cards[1:] != ordered_cards[:-1]) + 1
    starts = np.r_[0, group_ends]
    ends = np.r_[group_ends, len(order)]

    for start, end in zip(starts, ends):
        times = ordered_time[start:end]
        left = np.searchsorted(times, times - window_units, side="left")
        positions = np.arange(end - start)
        counts = positions - left + 1
        span_seconds = (times - times[left]) / per_second
        count[order[start:end]] = counts
        delta[order[start:end]] = np.where(counts > 1, span_seconds, np.nan)

    out["card_txn_count_10min"] = count
    out["card_txn_delta_10min"] = delta
    return out


def _email_domain_is_valid(series: pd.Series) -> pd.Series:
    domain = series.astype("string").str.strip().str.lower()
    return (
        domain.notna()
        & domain.str.contains(".", regex=False)
        & ~domain.isin(PLACEHOLDER_EMAIL_DOMAINS)
    )


def add_email_validity_flags(
    df: pd.DataFrame,
    purchaser_column: str = "purchaser_email_domain",
    recipient_column: str = "recipient_email_domain",
) -> pd.DataFrame:
    """Whether the purchaser and recipient domains look like real mailbox providers.

    A domain is valid when it is present, contains a dot, and is not
    ``anonymous.com`` (Vesta's placeholder). This is a fixed rule, not a
    frequency learned from training. Adds ``flag_purchaser_email_valid`` and
    ``flag_recipient_email_valid``.
    """
    _require_columns(df, (purchaser_column, recipient_column))
    out = df.copy()
    out["flag_purchaser_email_valid"] = _email_domain_is_valid(out[purchaser_column])
    out["flag_recipient_email_valid"] = _email_domain_is_valid(out[recipient_column])
    return out


def _email_provider(series: pd.Series) -> pd.Series:
    domain = series.astype("string").str.strip().str.lower()
    provider = pd.Series("other", index=series.index, dtype="string")
    provider = provider.mask(domain.isna() | domain.eq(""), "missing")
    for name, domains in EMAIL_PROVIDER_GROUPS.items():
        provider = provider.mask(domain.isin(domains), name)
    return provider


def add_email_provider_features(
    df: pd.DataFrame,
    purchaser_column: str = "purchaser_email_domain",
    recipient_column: str = "recipient_email_domain",
) -> pd.DataFrame:
    """Group mailbox domains, and flag when purchaser and recipient differ.

    ``purchaser_email_provider`` and ``recipient_email_provider`` are Google,
    Yahoo Mail, Microsoft, other, or missing. ``flag_email_domains_differ`` is
    true only when both domains are filled and not equal. If either side is
    missing the flag stays missing, so in-person rows are not labeled as a match.
    """
    _require_columns(df, (purchaser_column, recipient_column))
    out = df.copy()
    purchaser = out[purchaser_column].astype("string").str.strip().str.lower()
    recipient = out[recipient_column].astype("string").str.strip().str.lower()
    both_present = purchaser.notna() & purchaser.ne("") & recipient.notna() & recipient.ne("")
    out["purchaser_email_provider"] = _email_provider(out[purchaser_column])
    out["recipient_email_provider"] = _email_provider(out[recipient_column])
    out["flag_email_domains_differ"] = (
        purchaser.ne(recipient).astype("boolean").mask(~both_present)
    )
    return out


def _boolean_when_known(known: pd.Series, flag: pd.Series) -> pd.Series:
    """True/False where ``known`` is true, missing everywhere else."""
    return flag.astype("boolean").mask(~known.fillna(False))


def add_foreign_card_flag(
    df: pd.DataFrame,
    country_column: str = "card_issue_country",
    us_like_code: int = US_LIKE_CARD_COUNTRY,
) -> pd.DataFrame:
    """True when the card was issued outside the US-like country code 150.

    Missing issuing country stays missing. ``185`` is the main foreign code
    and lines up with the cross-border channel.
    """
    _require_columns(df, (country_column,))
    country = pd.to_numeric(df[country_column], errors="coerce")
    out = df.copy()
    out["flag_foreign_card"] = _boolean_when_known(country.notna(), country.ne(us_like_code))
    return out


def add_new_card_flag(df: pd.DataFrame, days_column: str = "D1") -> pd.DataFrame:
    """True when ``D1`` is 0: this card has not been seen before this payment.

    ``D1`` stays under its original name. A missing ``D1`` stays missing.
    """
    _require_columns(df, (days_column,))
    days = pd.to_numeric(df[days_column], errors="coerce")
    out = df.copy()
    out["flag_new_card"] = _boolean_when_known(days.notna(), days.eq(0))
    return out


def add_match_failure_count(
    df: pd.DataFrame,
    match_columns: tuple[str, ...] = MATCH_FAILURE_COLUMNS,
) -> pd.DataFrame:
    """How many card/address match checks came back F.

    Counts ``F`` in M2, M3, and M5–M9. Those checks are filled on in-person
    payments. If every one of them is missing, ``n_match_failures`` stays
    missing instead of zero.
    """
    _require_columns(df, match_columns)
    block = df.loc[:, list(match_columns)].apply(
        lambda column: column.astype("string").str.strip().str.upper()
    )
    filled = block.notna() & block.ne("")
    any_present = filled.any(axis=1)
    failures = block.eq("F").fillna(False).sum(axis=1)
    out = df.copy()
    out["n_match_failures"] = failures.astype("Int64").mask(~any_present)
    return out


def add_known_proxy_flag(
    df: pd.DataFrame,
    proxy_column: str = "ip_proxy_type",
    identity_column: str = "_has_identity",
) -> pd.DataFrame:
    """True when the identity row says the IP is a known proxy.

    Uses ``ip_proxy_type`` (id_23). Rows with no identity stay missing. An
    identity row with an empty proxy field is false.
    """
    out = df.copy()
    if proxy_column not in out.columns:
        out["flag_known_proxy"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        return out
    proxy_text = out[proxy_column].astype("string").str.strip()
    proxy_present = proxy_text.notna() & proxy_text.ne("") & proxy_text.str.lower().ne("<na>")
    if identity_column in out.columns:
        has_identity = out[identity_column].fillna(False).astype(bool)
    else:
        has_identity = proxy_present
    out["flag_known_proxy"] = _boolean_when_known(has_identity, proxy_present)
    return out


def add_amount_shape(df: pd.DataFrame, amount_column: str = "amount_usd") -> pd.DataFrame:
    """Compress the skewed dollar amount and keep the cents.

    ``log_amount`` is ``log1p(amount_usd)``. ``amount_cents`` is the fractional
    part (``49.99`` → ``0.99``). Round dollar amounts and odd cents are
    different patterns, and fraud tickets sit a bit higher on W, H, and R.
    """
    _require_columns(df, (amount_column,))
    amount = pd.to_numeric(df[amount_column], errors="coerce")
    usable = amount.ge(0)
    out = df.copy()
    out["log_amount"] = np.where(usable, np.log1p(amount.clip(lower=0)), np.nan)
    out["amount_cents"] = np.where(usable, np.mod(amount.fillna(0), 1), np.nan)
    return out


def add_far_billing_flag(
    df: pd.DataFrame,
    distance_column: str = "dist1",
    local_max: float = LOCAL_BILLING_DISTANCE,
) -> pd.DataFrame:
    """True when billing-to-counterparty distance is beyond the local mass.

    ``dist1`` is filled on in-person payments only. Most values are 0–10.
    ``flag_far_billing`` is true above that. Missing distance stays missing.
    """
    _require_columns(df, (distance_column,))
    distance = pd.to_numeric(df[distance_column], errors="coerce")
    out = df.copy()
    out["flag_far_billing"] = _boolean_when_known(distance.notna(), distance.gt(local_max))
    return out


def _combined_column_name(columns: tuple[str, ...]) -> str:
    return "_x_".join(columns)


TARGET_RATE_COLUMNS: tuple[str, ...] = TARGET_RATE_SOURCE_COLUMNS + tuple(
    _combined_column_name(combo) for combo in CAT_COMBINATIONS
)


def _group_tokens(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    tokens = pd.DataFrame(index=df.index)
    for column in columns:
        values = df[column]
        missing = values.isna()
        if pd.api.types.is_numeric_dtype(values):
            as_token = values.round(0).astype("Int64").astype("string")
        else:
            as_token = values.astype("string").str.strip().str.lower()
        tokens[column] = as_token.mask(missing, "__missing__")
    return tokens


class CategoricalTargetEncoder(BaseEstimator, TransformerMixin):
    """Fraud rate of each categorical column, fit on the training target.

    ``columns`` defaults to ``TARGET_RATE_COLUMNS``: the single categoricals
    and the joined pairs from ``combine_cat_columns``. Each column adds
    ``{column}_target_rate``. ``fit_transform`` on the training rows uses
    cross-fitting, so a training row is not encoded with its own label.
    Validation and test use the rates learned from the full training set.
    An unseen level gets the overall training fraud rate. Missing values are
    their own level (``__missing__``).
    """

    def __init__(
        self,
        columns: tuple[str, ...] = TARGET_RATE_COLUMNS,
        smooth: str | float = "auto",
        cv: int = 5,
    ):
        self.columns = columns
        self.smooth = smooth
        self.cv = cv

    def _encoder(self) -> TargetEncoder:
        return TargetEncoder(target_type="binary", smooth=self.smooth, cv=self.cv)

    def _feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        _require_columns(X, self.columns)
        return _group_tokens(X, self.columns).astype("object")

    def _write_rates(self, X: pd.DataFrame, encoded: np.ndarray) -> pd.DataFrame:
        if encoded.ndim == 1:
            encoded = encoded.reshape(-1, 1)
        out = X.copy()
        for i, column in enumerate(self.columns):
            out[f"{column}_target_rate"] = encoded[:, i]
        return out

    def fit(self, X, y=None):
        self.encoder_ = self._encoder()
        self.encoder_.fit(self._feature_frame(X), y)
        return self

    def transform(self, X):
        encoded = np.asarray(self.encoder_.transform(self._feature_frame(X)))
        return self._write_rates(X, encoded)

    def fit_transform(self, X, y=None, **fit_params):
        self.encoder_ = self._encoder()
        encoded = np.asarray(
            self.encoder_.fit_transform(self._feature_frame(X), y)
        )
        return self._write_rates(X, encoded)


class AmountOverGroupAverage(BaseEstimator, TransformerMixin):
    """Transaction amount relative to the usual amount for that card and product.

    Adds ``amount_over_group_avg``: ``amount_usd`` divided by the training-set
    mean of ``amount_usd`` inside ``card5`` (BIN) + ``product_channel`` +
    ``card_network``. A value above 1 is larger than that group's usual ticket.
    Groups that never appear in training, and groups whose mean is missing,
    use the overall training average.
    """

    def __init__(
        self,
        amount_column: str = "amount_usd",
        group_columns: tuple[str, ...] = AMOUNT_GROUP_COLUMNS,
        out_column: str = "amount_over_group_avg",
    ):
        self.amount_column = amount_column
        self.group_columns = group_columns
        self.out_column = out_column

    def fit(self, X, y=None):
        _require_columns(X, (self.amount_column, *self.group_columns))
        keys = _group_tokens(X, self.group_columns)
        amounts = pd.to_numeric(X[self.amount_column], errors="coerce")
        self.global_mean_ = float(amounts.mean())
        grouped = keys.copy()
        grouped["_amount"] = amounts.to_numpy()
        self.group_means_ = (
            grouped.groupby(list(self.group_columns), dropna=False)["_amount"]
            .mean()
            .rename("_group_mean")
            .reset_index()
        )
        return self

    def transform(self, X):
        _require_columns(X, (self.amount_column, *self.group_columns))
        keys = _group_tokens(X, self.group_columns)
        merged = keys.reset_index(drop=True).merge(
            self.group_means_,
            on=list(self.group_columns),
            how="left",
        )
        group_mean = merged["_group_mean"].fillna(self.global_mean_).to_numpy()
        amount = pd.to_numeric(X[self.amount_column], errors="coerce").to_numpy()
        out = X.copy()
        out[self.out_column] = amount / group_mean
        return out


class CommonEmailDomainFlags(BaseEstimator, TransformerMixin):
    """Whether each email domain is one of the domains buyers actually use.

    Fit on training rows only. A domain is common when it accounts for at least
    ``min_share`` (default 1%) of the non-null values in that column. Adds
    ``flag_purchaser_email_common`` and ``flag_recipient_email_common``.
    ``anonymous.com`` can be common and still fail the validity flag: frequency
    and "looks like a real provider" are separate questions.
    """

    def __init__(
        self,
        columns: tuple[str, ...] = (
            "purchaser_email_domain",
            "recipient_email_domain",
        ),
        min_share: float = 0.01,
    ):
        self.columns = columns
        self.min_share = min_share

    def fit(self, X, y=None):
        _require_columns(X, self.columns)
        self.common_domains_ = {}
        self.out_columns_ = {}
        for column in self.columns:
            domains = X[column].astype("string").str.strip().str.lower().dropna()
            share = domains.value_counts(normalize=True)
            self.common_domains_[column] = set(share[share >= self.min_share].index)
            self.out_columns_[column] = _common_email_column(column)
        return self

    def transform(self, X):
        _require_columns(X, self.columns)
        out = X.copy()
        for column, out_column in self.out_columns_.items():
            domains = out[column].astype("string").str.strip().str.lower()
            out[out_column] = domains.isin(self.common_domains_[column]).fillna(False)
        return out


def _common_email_column(column: str) -> str:
    if column == "purchaser_email_domain":
        return "flag_purchaser_email_common"
    if column == "recipient_email_domain":
        return "flag_recipient_email_common"
    return f"flag_{column}_common"


def combine_cat_columns(
    df: pd.DataFrame,
    combinations: tuple[tuple[str, ...], ...] = CAT_COMBINATIONS,
) -> pd.DataFrame:
    """Join categorical columns into one token per combination.

    Each piece uses the same rules as the group averages: numeric codes are
    rounded, text is stripped and lowercased, and a missing piece is
    ``__missing__``. Pieces are joined with ``|``. The new column is named
    with ``_x_`` between the source names (``card5_x_product_channel``).

    This is a fixed rewrite, not a rate learned from training, so it runs
    before the split. The processing pipeline target-encodes the joined columns.
    """
    needed = tuple(dict.fromkeys(column for combo in combinations for column in combo))
    _require_columns(df, needed)
    tokens = _group_tokens(df, needed)
    out = df.copy()
    for combo in combinations:
        joined = tokens[combo[0]]
        for column in combo[1:]:
            joined = joined + "|" + tokens[column]
        out[_combined_column_name(combo)] = joined
    return out


def _resolve_source(
    df: pd.DataFrame,
    preferred: str,
    fallback: str | None,
) -> str | None:
    if preferred in df.columns:
        return preferred
    if fallback is not None and fallback in df.columns:
        return fallback
    return None


def _valid_client_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows whose card1, billing region, and D1 are all present.

    ``card_id`` is still filled when one of those is missing, but the missing
    piece becomes the string ``nan`` and would glue unrelated clients together.
    """
    region_column = (
        "billing_region"
        if "billing_region" in df.columns
        else "addr1"
        if "addr1" in df.columns
        else None
    )
    if "card1" in df.columns and region_column is not None and "D1" in df.columns:
        days = pd.to_numeric(df["D1"], errors="coerce")
        return (
            df["card1"].notna() & df[region_column].notna() & days.notna()
        ).to_numpy()
    if "card_id" in df.columns:
        return df["card_id"].notna().to_numpy()
    return np.zeros(len(df), dtype=bool)


def _as_model_number(series: pd.Series) -> np.ndarray:
    """Float values for a past mean. Match flags T/F become 1/0."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    text = series.astype("string").str.strip().str.upper()
    blank = text.isna() | text.eq("") | text.eq("<NA>")
    if bool((blank | text.isin(["T", "F", "TRUE", "FALSE"])).all()):
        mapped = text.map({"T": 1.0, "F": 0.0, "TRUE": 1.0, "FALSE": 0.0})
        return mapped.to_numpy(dtype=np.float64)
    codes, _ = pd.factorize(text.mask(blank), use_na_sentinel=True)
    values = codes.astype(np.float64)
    values[codes < 0] = np.nan
    return values


def _factor_codes(series: pd.Series) -> np.ndarray:
    """Integer codes for a past nunique. Missing is -1."""
    codes, _ = pd.factorize(series, use_na_sentinel=True)
    return codes.astype(np.int64, copy=False)


def _month_codes(datetimes: pd.Series) -> np.ndarray:
    stamps = pd.to_datetime(datetimes, utc=True)
    codes = np.full(len(stamps), -1, dtype=np.int64)
    valid = stamps.notna().to_numpy()
    if valid.any():
        years = stamps.dt.year.to_numpy()
        months = stamps.dt.month.to_numpy()
        codes[valid] = years[valid] * 12 + months[valid]
    return codes


def _history_value_columns(
    df: pd.DataFrame,
) -> list[tuple[str, set[str], np.ndarray]]:
    merged: dict[str, set[str]] = {}
    arrays: dict[str, np.ndarray] = {}
    order: list[str] = []
    batches = (
        (CARD_HISTORY_MEAN_STD_COLUMNS, ("mean", "std")),
        (CARD_HISTORY_MEAN_COLUMNS, ("mean",)),
        (CARD_HISTORY_STD_COLUMNS, ("std",)),
    )
    for columns, stats in batches:
        for preferred, fallback in columns:
            source = _resolve_source(df, preferred, fallback)
            if source is None:
                continue
            if source not in merged:
                merged[source] = set()
                order.append(source)
                arrays[source] = _as_model_number(df[source])
            merged[source].update(stats)
    return [(source, merged[source], arrays[source]) for source in order]


def _history_nunique_columns(df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    found: list[tuple[str, np.ndarray]] = []
    for preferred, fallback in CARD_HISTORY_NUNIQUE_COLUMNS:
        source = _resolve_source(df, preferred, fallback)
        if source is None:
            continue
        found.append((source, _factor_codes(df[source])))
    return found


def _past_mean_std_by_card(
    card_codes: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample std of earlier rows in a card, in the order given.

    ``values`` is already sorted by card, then time. The current row is left
    out of its own mean and std.
    """
    n = len(values)
    mean = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return mean, std
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    grouped = pd.Series(filled).groupby(card_codes, sort=False)
    cumulative = grouped.cumsum().to_numpy()
    cumulative_sq = (
        pd.Series(filled * filled).groupby(card_codes, sort=False).cumsum().to_numpy()
    )
    cumulative_count = (
        pd.Series(valid.astype(np.float64)).groupby(card_codes, sort=False).cumsum().to_numpy()
    )
    previous_sum = cumulative - filled
    previous_sq = cumulative_sq - filled * filled
    previous_count = cumulative_count - valid.astype(np.float64)
    enough_for_mean = previous_count >= 1
    mean[enough_for_mean] = previous_sum[enough_for_mean] / previous_count[enough_for_mean]
    enough_for_std = previous_count >= 2
    variance_num = previous_sq[enough_for_std] - (
        previous_sum[enough_for_std] ** 2
    ) / previous_count[enough_for_std]
    std[enough_for_std] = np.sqrt(
        np.maximum(variance_num / (previous_count[enough_for_std] - 1.0), 0.0)
    )
    return mean, std


def _past_nunique_by_card(card_codes: np.ndarray, value_codes: np.ndarray) -> np.ndarray:
    """How many distinct non-missing codes appeared on earlier rows of this card.

    ``value_codes`` below 0 are missing and do not count. The current row is
    left out. The first row of a card is NaN.
    """
    n = len(card_codes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    pair = pd.DataFrame({"card": card_codes, "value": value_codes})
    first_time = (
        pair.groupby(["card", "value"], sort=False).cumcount().eq(0).to_numpy()
    )
    first_time = first_time & (value_codes >= 0)
    seen = (
        pd.Series(first_time.astype(np.float64))
        .groupby(card_codes, sort=False)
        .cumsum()
        .to_numpy()
    )
    past = seen - first_time.astype(np.float64)
    new_card = np.empty(n, dtype=bool)
    new_card[0] = True
    if n > 1:
        new_card[1:] = card_codes[1:] != card_codes[:-1]
    past[new_card] = np.nan
    out[:] = past
    return out


def _ordered_client_rows(
    df: pd.DataFrame,
    card_column: str,
    time_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions sorted by card_id then time, and the card code of each.

    Equal timestamps keep the original row order, so the earlier row is the
    past of the later one.
    """
    valid_client = _valid_client_mask(df)
    valid_time = pd.to_datetime(df[time_column], utc=True).notna().to_numpy()
    valid_card = df[card_column].notna().to_numpy()
    valid_idx = np.flatnonzero(valid_client & valid_time & valid_card)
    if len(valid_idx) == 0:
        return valid_idx, np.empty(0, dtype=np.int64)
    time_int, _ = _datetime_int(df[time_column])
    card_codes, _ = pd.factorize(df[card_column], use_na_sentinel=True)
    order = valid_idx[
        np.lexsort((valid_idx, time_int[valid_idx], card_codes[valid_idx]))
    ]
    return order, card_codes[order]


def add_outsider15_flag(
    df: pd.DataFrame,
    d1_column: str = "D1",
    d15_column: str = "D15",
) -> pd.DataFrame:
    """1 when D1 and D15 differ by more than 3 days, else 0.

    Both columns are day counts from some past event. A gap larger than 3 days
    means this row does not sit on one client's timeline. The 0.96 kernel calls
    the flag ``outsider15``. A missing D1 or D15 stays missing. Stored as float
    so numeric feature selection keeps it.
    """
    _require_columns(df, (d1_column, d15_column))
    d1 = pd.to_numeric(df[d1_column], errors="coerce")
    d15 = pd.to_numeric(df[d15_column], errors="coerce")
    known = d1.notna() & d15.notna()
    out = df.copy()
    out["flag_outsider15"] = (d1 - d15).abs().gt(3).astype(np.float64).where(known)
    return out


def add_card_history_features(
    df: pd.DataFrame,
    card_column: str = "card_id",
    time_column: str = "transaction_datetime",
) -> pd.DataFrame:
    """Past-only aggregates for one ``card_id``.

    Each row sees earlier transactions of that client and not later ones, and
    not other clients. The current row is left out of its own mean, std, and
    nunique. ``card_id`` stays on the frame; these columns are what the model
    should use.

    Adds, when the source column exists:

    - ``card_prior_txn_count``: how many earlier payments this client has.
    - ``card_{column}_mean`` and ``card_{column}_std`` for amount, D4, time of
      day (D9), D10, and D15.
    - ``card_{column}_mean`` for C1–C14 except C3, and for M1–M9. T/F match
      flags are 1/0 before the mean. ``card_C14_std`` is included.
    - ``card_{column}_nunique`` for purchaser email, dist1, id_02, cents, C13,
      and V127, V136, V307, V309, V314, V320.
    - ``card_month_nunique``: distinct calendar months already seen.

    The first payment of a client has a prior count of 0 and missing mean, std,
    and nunique. A row with no real ``card_id`` (missing card, region, or D1)
    is missing on every one of these columns.
    """
    _require_columns(df, (card_column, time_column))
    value_columns = _history_value_columns(df)
    nunique_columns = _history_nunique_columns(df)
    order, ordered_cards = _ordered_client_rows(df, card_column, time_column)
    n = len(df)
    extra: dict[str, np.ndarray] = {}

    prior = np.full(n, np.nan, dtype=np.float64)
    if len(order):
        prior[order] = (
            pd.Series(np.zeros(len(order)))
            .groupby(ordered_cards, sort=False)
            .cumcount()
            .to_numpy(dtype=np.float64)
        )
    extra["card_prior_txn_count"] = prior

    for source, stats, values in value_columns:
        if len(order):
            mean, std = _past_mean_std_by_card(ordered_cards, values[order])
        else:
            mean = std = np.empty(0, dtype=np.float64)
        if "mean" in stats:
            column_values = np.full(n, np.nan, dtype=np.float64)
            if len(order):
                column_values[order] = mean
            extra[f"card_{source}_mean"] = column_values
        if "std" in stats:
            column_values = np.full(n, np.nan, dtype=np.float64)
            if len(order):
                column_values[order] = std
            extra[f"card_{source}_std"] = column_values

    for source, codes in nunique_columns:
        column_values = np.full(n, np.nan, dtype=np.float64)
        if len(order):
            column_values[order] = _past_nunique_by_card(ordered_cards, codes[order])
        extra[f"card_{source}_nunique"] = column_values

    month_values = np.full(n, np.nan, dtype=np.float64)
    if len(order):
        month_values[order] = _past_nunique_by_card(
            ordered_cards, _month_codes(df[time_column])[order]
        )
    extra["card_month_nunique"] = month_values

    overlap = [name for name in extra if name in df.columns]
    base = df.drop(columns=overlap) if overlap else df
    return pd.concat([base, pd.DataFrame(extra, index=df.index)], axis=1)


def _v_columns(df: pd.DataFrame) -> list[str]:
    names = [
        column
        for column in df.columns
        if isinstance(column, str) and column.startswith("V") and column[1:].isdigit()
    ]
    return sorted(names, key=lambda name: int(name[1:]))


def _is_low_variation(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce")
    return int(values.nunique(dropna=True)) <= 1


def redundant_v_columns(
    df: pd.DataFrame,
    threshold: float = V_CORRELATION_MAX,
) -> list[str]:
    """V columns that copy another V column, or that never vary.

    Columns missing on exactly the same rows form one block. Inside a block,
    walk V1, V2, … and drop a column when its absolute correlation with an
    already kept column is at least ``threshold`` (default 0.75). Columns in
    different blocks are not compared. A constant or all-missing column is
    dropped too.
    """
    columns = _v_columns(df)
    if not columns:
        return []

    groups: list[tuple[np.ndarray, list[str]]] = []
    for column in columns:
        mask = df[column].isna().to_numpy(dtype=bool, copy=True)
        placed = False
        for group_mask, group_columns in groups:
            if group_mask.shape == mask.shape and np.array_equal(group_mask, mask):
                group_columns.append(column)
                placed = True
                break
        if not placed:
            groups.append((mask, [column]))

    dropped: list[str] = []
    for mask, group_columns in groups:
        usable = [column for column in group_columns if not _is_low_variation(df[column])]
        dropped.extend(column for column in group_columns if column not in usable)
        if len(usable) < 2 or int((~mask).sum()) < 2:
            continue
        block = (
            df.iloc[np.flatnonzero(~mask)][usable]
            .apply(pd.to_numeric, errors="coerce")
            .corr()
            .abs()
        )
        kept: list[str] = []
        for column in usable:
            redundant = False
            for other in kept:
                correlation = block.at[column, other]
                if np.isfinite(correlation) and correlation >= threshold:
                    redundant = True
                    break
            if redundant:
                dropped.append(column)
            else:
                kept.append(column)
    return sorted(set(dropped), key=lambda name: int(name[1:]))


def drop_redundant_v_columns(
    df: pd.DataFrame,
    threshold: float = V_CORRELATION_MAX,
) -> pd.DataFrame:
    """Drop V columns that copy another V column, or that never vary.

    The drop list is chosen from ``df`` itself. ``run_preprocessing`` calls
    this after the card-history features, so a V column used as a history
    source can still be counted and then removed.
    """
    drop = [
        column
        for column in redundant_v_columns(df, threshold=threshold)
        if column in df.columns
    ]
    print("len(drop):", len(drop))
    if not drop:
        return df
    return df.drop(columns=drop)


def build_processing_pipeline() -> Pipeline:
    """Pipeline of features whose statistics have to be learned on the training set.

    Fit it on train, then transform validation and test. Steps:

    - ``categorical_target_encoding``: fraud rate of each column in
      ``TARGET_RATE_COLUMNS`` (the single categoricals and the joined pairs
      from ``combine_cat_columns``). Training rows are cross-fitted. An unseen
      level gets the overall training fraud rate.
    - ``amount_over_group_avg``: amount divided by the training average for
      BIN + product + card network, with the overall average as fallback.
    - ``common_email_domains``: purchaser and recipient domains that are at
      least 1% of non-null training rows.
    """
    return Pipeline(
        steps=[
            ("categorical_target_encoding", CategoricalTargetEncoder()),
            ("amount_over_group_avg", AmountOverGroupAverage()),
            ("common_email_domains", CommonEmailDomainFlags()),
        ]
    )


def split_train_validation_test(
    df: pd.DataFrame,
    runtime_config: BaseConfigParams,
    target_column: str = "is_fraud",
    date_column: str = "transaction_datetime",
    validation_split: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split like ``ModelTrainingWorkflow.split_data``, using dates from ``runtime_config``.

    Rows between ``training_start_date`` and ``training_end_date`` are a stratified
    random split into train and validation. Rows between ``test_start_date`` and
    ``test_end_date`` are the test set. End dates include the full last day.
    """
    missing = [name for name in (date_column, target_column) if name not in df.columns]
    if missing:
        raise KeyError(f"Missing columns required to split: {missing}")
    if runtime_config.test_start_date is None or runtime_config.test_end_date is None:
        raise ValueError("runtime_config is missing test_start_date or test_end_date")

    frame = df.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], utc=True)

    training_start = pd.to_datetime(runtime_config.training_start_date, utc=True)
    training_end = pd.to_datetime(
        last_day_of_month(runtime_config.training_end_date), utc=True
    ) + timedelta(hours=23, minutes=59, seconds=59)
    test_start = pd.to_datetime(runtime_config.test_start_date, utc=True)
    test_end = pd.to_datetime(
        last_day_of_month(runtime_config.test_end_date), utc=True
    ) + timedelta(hours=23, minutes=59, seconds=59)

    train_val = frame.loc[
        (frame[date_column] >= training_start) & (frame[date_column] <= training_end)
    ]
    df_test = frame.loc[
        (frame[date_column] >= test_start) & (frame[date_column] <= test_end)
    ].copy()
    if train_val.empty:
        raise ValueError(
            f"No rows in the training window {training_start} to {training_end}."
        )
    if df_test.empty:
        raise ValueError(f"No rows in the test window {test_start} to {test_end}.")

    df_train, df_validation = train_test_split(
        train_val,
        test_size=validation_split,
        random_state=random_state,
        stratify=train_val[target_column],
    )
    return df_train, df_validation, df_test


def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning and features that do not need a training-set average.

    Runs before the time split. ``card_id`` is the client fingerprint, and the
    10-minute velocity is counted on that id, looking backward only.
    ``add_card_history_features`` adds the past-only client means, stds, and
    nunique counts, and ``add_outsider15_flag`` marks rows whose D1 and D15
    disagree. The other flags are fixed rules from the EDA: email provider and
    mismatch, foreign card, new card, failed match checks, known proxy, amount
    shape, and far-from-billing distance. ``combine_cat_columns`` joins the
    categorical pairs that the processing pipeline then target-encodes.
    ``drop_redundant_v_columns`` then removes V columns that copy another V
    column in this frame, or that never vary.

    """
    df = normalize_ieee_columns(df)
    df = rename_columns(df)
    df = drop_redundant_v_columns(df)
    df = add_transaction_datetime(df)
    df = add_card_id(df)
    df = add_same_card_window_features(df)
    df = add_email_validity_flags(df)
    df = add_email_provider_features(df)
    df = add_foreign_card_flag(df)
    df = add_new_card_flag(df)
    df = add_match_failure_count(df)
    df = add_known_proxy_flag(df)
    df = add_amount_shape(df)
    df = add_far_billing_flag(df)
    df = combine_cat_columns(df)
    df = add_outsider15_flag(df)
    df = add_card_history_features(df)
    return df


def run_processing_pipeline(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
    target_column: str = "is_fraud",
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Fit train-only averages and apply them to validation and test.

    The sklearn pipeline adds a ``_target_rate`` column for each name in
    ``TARGET_RATE_COLUMNS``, ``amount_over_group_avg`` (amount / mean amount of
    card5 + product + network), and the common-domain flags. Training rows are
    cross-fitted for the target rates.
    """
    pipeline = build_processing_pipeline()
    df_train_out = pipeline.fit_transform(df_train, df_train[target_column])
    df_validation_out = (
        None if df_validation is None else pipeline.transform(df_validation)
    )
    df_test_out = None if df_test is None else pipeline.transform(df_test)
    return pipeline, df_train_out, df_validation_out, df_test_out
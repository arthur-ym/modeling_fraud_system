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
    "card1": "card_id",
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


def add_same_card_window_features(
    df: pd.DataFrame,
    card_column: str = "card_id",
    time_column: str = "transaction_datetime",
    window: str = "10min",
) -> pd.DataFrame:
    """Same-card activity in the previous 10 minutes.

    Looks backward only, so it can run before the train/validation/test split.
    Adds two columns:

    - ``card_txn_count_10min``: how many transactions this ``card_id`` has in the
      window, including the current row. An isolated payment is 1.
    - ``card_txn_delta_10min``: seconds from the earliest other same-card
      transaction in that window back to now. Missing when the count is 1.
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


class ProductChannelTargetEncoder(BaseEstimator, TransformerMixin):
    """Fraud rate of the product channel, fit on the training target.

    Adds ``product_channel_target_rate`` (ProductCD). ``fit_transform`` on the
    training rows uses cross-fitting, so a training row is not encoded with its
    own label. Validation and test use the rates learned from the full training
    set. An unseen channel gets the overall training fraud rate.
    """

    def __init__(
        self,
        column: str = "product_channel",
        out_column: str = "product_channel_target_rate",
        smooth: str | float = "auto",
        cv: int = 5,
    ):
        self.column = column
        self.out_column = out_column
        self.smooth = smooth
        self.cv = cv

    def _encoder(self) -> TargetEncoder:
        return TargetEncoder(target_type="binary", smooth=self.smooth, cv=self.cv)

    def _product_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        _require_columns(X, (self.column,))
        return pd.DataFrame({self.column: X[self.column].astype("object").to_numpy()})

    def fit(self, X, y=None):
        self.encoder_ = self._encoder()
        self.encoder_.fit(self._product_frame(X), y)
        return self

    def transform(self, X):
        encoded = np.asarray(self.encoder_.transform(self._product_frame(X))).ravel()
        out = X.copy()
        out[self.out_column] = encoded
        return out

    def fit_transform(self, X, y=None, **fit_params):
        self.encoder_ = self._encoder()
        encoded = np.asarray(
            self.encoder_.fit_transform(self._product_frame(X), y)
        ).ravel()
        out = X.copy()
        out[self.out_column] = encoded
        return out


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


def build_processing_pipeline() -> Pipeline:
    """Pipeline of features whose statistics have to be learned on the training set.

    Fit it on train, then transform validation and test. Steps:

    - ``product_target_encoding``: fraud rate of ``product_channel``.
    - ``amount_over_group_avg``: amount divided by the training average for
      BIN + product + card network, with the overall average as fallback.
    - ``common_email_domains``: purchaser and recipient domains that are at
      least 1% of non-null training rows.
    """
    return Pipeline(
        steps=[
            ("product_target_encoding", ProductChannelTargetEncoder()),
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

    Runs before the time split. Card velocity only looks backward in time.
    The other flags are fixed rules from the EDA: email provider and mismatch,
    foreign card, new card, failed match checks, known proxy, amount shape,
    and far-from-billing distance.
    """
    df = normalize_ieee_columns(df)
    df = rename_columns(df)
    df = add_transaction_datetime(df)
    df = add_same_card_window_features(df)
    df = add_email_validity_flags(df)
    df = add_email_provider_features(df)
    df = add_foreign_card_flag(df)
    df = add_new_card_flag(df)
    df = add_match_failure_count(df)
    df = add_known_proxy_flag(df)
    df = add_amount_shape(df)
    df = add_far_billing_flag(df)
    return df


def run_processing_pipeline(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
    target_column: str = "is_fraud",
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Fit train-only averages and apply them to validation and test.

    The sklearn pipeline adds ``product_channel_target_rate``,
    ``amount_over_group_avg`` (amount / mean amount of card5 + product + network),
    and the common-domain flags. Training rows are cross-fitted for the target rate.
    """
    pipeline = build_processing_pipeline()
    df_train_out = pipeline.fit_transform(df_train, df_train[target_column])
    df_validation_out = (
        None if df_validation is None else pipeline.transform(df_validation)
    )
    df_test_out = None if df_test is None else pipeline.transform(df_test)
    return pipeline, df_train_out, df_validation_out, df_test_out
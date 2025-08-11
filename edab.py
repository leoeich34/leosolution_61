from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


DATA_PATH = "/transaction_fraud_data.parquet"
FX_PATH   = "/historical_currency_exchange.parquet"
OUT_DIR = Path("./eda_output")


# ========= имена столбцов =========
AMOUNT_USD_PRIORITY = ["amount_usd", "amount_in_usd", "usd_amount", "transaction_amount_usd"]
AMOUNT_PRIORITY     = ["amount", "transaction_amount", "amt", "value"]
CURRENCY_PRIORITY   = ["currency", "txn_currency", "transaction_currency", "currency_code"]
TIME_PRIORITY       = ["transaction_datetime", "transaction_time", "event_time",
                       "timestamp", "datetime", "date_time", "tx_time", "txn_time", "created_at", "date"]

CLIENT_PRIORITY     = ["customer_id", "client_id", "user_id", "account_id", "cardholder_id"]
CITY_PRIORITY       = ["city", "merchant_city", "billing_city", "customer_city", "transaction_city"]
VENDOR_PRIORITY     = ["vendor", "merchant", "merchant_name", "vendor_name", "vendor_id", "merchant_id"]
VCAT_PRIORITY       = ["vendor_category", "merchant_category", "category"]
VTYPE_PRIORITY      = ["vendor_type", "merchant_type", "type"]
CARD_PRIORITY       = ["card_type"]
FRAUD_COL           = "is_fraud"
HR_VENDOR           = "is_high_risk_vendor"


# ========= утилиты =========
# 1. метод pick(), возвращающий имя столбца из списка кандидатов, в котором что-то имеется
def pick(names: List[str], candidates: List[str]) -> Optional[str]:
    low = {n.lower(): n for n in names}
    for c in candidates:
        if c in low:
            return low[c]
    return None

#2. метод maybe_cents_divider(), эвристически определяющий суммы в центах
def maybe_cents_divider(sample: pd.Series) -> float:
    vals = pd.to_numeric(sample, errors="coerce").dropna()
    if vals.empty:
        return 1.0
    q50 = vals.quantile(0.50)
    q95 = vals.quantile(0.95)
    return 100.0 if (q50 > 5_000 and q95 > 50_000) else 1.0

#3. метод to_bool() для приведения входящих типов в bool-type
def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    num = pd.to_numeric(series, errors="coerce")
    if not num.isna().all():
        return num.fillna(0).astype(int).astype(bool)
    return series.astype(str).str.lower().isin(["1", "true", "t", "yes"])

#4. метод load_fx_long(), считывающий таблицу курсов и приводящий ее в long-формат
def load_fx_long(path: str) -> pd.DataFrame:
    fx = pd.read_parquet(path)
    fx = fx.copy()
    fx["date"] = pd.to_datetime(fx["date"]).dt.date
    fx_long = fx.melt(id_vars=["date"], var_name="currency", value_name="rate")
    fx_long["currency"] = fx_long["currency"].astype(str).str.upper()
    fx_long["rate"] = pd.to_numeric(fx_long["rate"], errors="coerce")
    return fx_long.dropna(subset=["rate"])

#5. метод to_usd(), формирующий единую серию сумм в долларах по всему файлу
def to_usd(
    pf: pq.ParquetFile,
    amount_usd_col: Optional[str],
    amount_col: Optional[str],
    currency_col: Optional[str],
    time_col: Optional[str],
    fx_long: Optional[pd.DataFrame],
) -> pd.Series:
    vals = []
    cents_divider = None

    for rg in range(pf.num_row_groups):
        need = [amount_usd_col] if amount_usd_col else [amount_col, currency_col, time_col]
        df = pf.read_row_group(rg, columns=need).to_pandas()

        if amount_usd_col:
            v = pd.to_numeric(df[amount_usd_col], errors="coerce").dropna()
        else:
            # этап 1 - готовим параметры (date, currency, amount)
            dt = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.tz_localize(None).dt.date
            curr = df[currency_col].astype(str).str.upper()
            amt  = pd.to_numeric(df[amount_col], errors="coerce")
            tmp = pd.DataFrame({"date": dt, "currency": curr, "amount": amt}).dropna(subset=["date", "currency", "amount"])
            if tmp.empty:
                continue

            # этап 2 - для начала производим точное слияние по паре (date,currency)
            merged = tmp.merge(fx_long, on=["date", "currency"], how="left")

            # этап 3 - для пропусков выполняем asof-merge (по ближайшему ключу) назад по дате в разрезе каждой валюты
            if merged["rate"].isna().any():
                parts = []
                for code, part in merged.groupby("currency", group_keys=False):
                    ref = fx_long[fx_long["currency"] == code].sort_values("date")
                    if ref.empty:
                        parts.append(part.assign(rate=pd.NA))
                        continue
                    sub = part.sort_values("date")
                    sub2 = pd.merge_asof(sub, ref, on="date", by="currency", direction="backward")
                    parts.append(sub2)
                merged = pd.concat(parts, ignore_index=True)

            merged = merged.dropna(subset=["rate"])

            # этап 4 - теперь предполагаем rate, равный единице валюты за 1 USD → USD = amount / rate
            v = (merged["amount"] / merged["rate"]).dropna()

        if v.empty:
            continue

        if cents_divider is None:
            cents_divider = maybe_cents_divider(v)

        vals.append(v / cents_divider)

    if not vals:
        return pd.Series([], dtype="float64")

    return pd.concat(vals, ignore_index=True)

#6. метод save_bar() для сохранения bar-чарта в PNG-картинку
def save_bar(series: pd.Series, title: str, fname: str, top: int = 20) -> None:
    plt.figure()
    series.sort_values(ascending=False).head(top).plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / fname)
    plt.close()

#7. метод сохранения гистограммы распределения
def save_hist(series: pd.Series, title: str, fname: str, bins: int = 24) -> None:
    plt.figure()
    series.plot(kind="hist", bins=bins)
    plt.title(title)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / fname)
    plt.close()


# ========= основная функция =========
def main() -> None:
    """
    Здесь выполняются следующие действия:
      1) находит нужные имена столбцов,
      2) делает overview/пропуски,
      3) строит отчёты по времени,
      4) приводит суммы к USD и считает базовую статистику,
      5) считает fraud rate по срезам и статистики чеков по городам,
      6) и сохраняет всё в ./eda_output/.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(DATA_PATH)
    names = [f.name for f in pf.schema_arrow]

    amount_usd_col = pick(names, AMOUNT_USD_PRIORITY)
    amount_col     = pick(names, AMOUNT_PRIORITY)
    currency_col   = pick(names, CURRENCY_PRIORITY)
    time_col       = pick(names, TIME_PRIORITY)
    client_col     = pick(names, CLIENT_PRIORITY)
    city_col       = pick(names, CITY_PRIORITY)
    vendor_col     = pick(names, VENDOR_PRIORITY)
    vcat_col       = pick(names, VCAT_PRIORITY)
    vtype_col      = pick(names, VTYPE_PRIORITY)
    card_col       = pick(names, CARD_PRIORITY)
    fraud_col      = FRAUD_COL if FRAUD_COL in names else None
    hr_col         = HR_VENDOR if HR_VENDOR in names else None

    fx_long = None
    if amount_usd_col is None and all(x is not None for x in [amount_col, currency_col, time_col]):
        fx_long = load_fx_long(FX_PATH)

    # ----- общий сбор и все пропуски  -----
    first = pf.read_row_group(0).to_pandas()
    overview = {
        "rows_total_estimate": sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)),
        "cols": len(names),
        "has_amount_usd": bool(amount_usd_col),
        "time_col_detected": time_col or "",
        "client_col_detected": client_col or "",
        "vendor_col_detected": vendor_col or "",
        "city_col_detected": city_col or "",
        "fraud_col_detected": fraud_col or "",
    }
    pd.DataFrame([overview]).to_csv(OUT_DIR / "summary_overview.csv", index=False)
    first.isna().mean().sort_values(ascending=False).rename("missing_share").to_csv(OUT_DIR / "missingness.csv")

    # ----- время -----
    if time_col:
        times: List[pd.Series] = []
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=[time_col]).to_pandas()[time_col]
            t = pd.to_datetime(t, errors="coerce", utc=True).dropna()
            if not t.empty:
                times.append(t.dt.tz_localize(None))
        if times:
            ts = pd.concat(times, ignore_index=True)
            pd.Series(
                {"min_time": ts.min(), "max_time": ts.max(), "days_span": (ts.max() - ts.min()).days}
            ).to_csv(OUT_DIR / "time_range.csv")
            hours = ts.dt.hour.value_counts()
            hours.to_csv(OUT_DIR / "tx_count_by_hour.csv")
            save_bar(hours, "Transactions by hour of day", "time_hist_by_hour.png", top=24)

    # ----- суммы в USD -----
    usd_all = to_usd(pf, amount_usd_col, amount_col, currency_col, time_col, fx_long)
    if not usd_all.empty:
        pd.Series(
            {
                "count": usd_all.shape[0],
                "mean": usd_all.mean(),
                "median": usd_all.median(),
                "std": usd_all.std(ddof=1) if usd_all.shape[0] > 1 else 0.0,
                "p95": usd_all.quantile(0.95),
                "p99": usd_all.quantile(0.99),
            }
        ).to_csv(OUT_DIR / "amount_usd_basic_stats.csv")
        # важно! картинку рисуем по «обрезанным» 99% значений, чтобы хвост не портил шкалу
        save_hist(usd_all.clip(upper=usd_all.quantile(0.99)), "USD amounts (clipped 99%)", "amount_usd_hist.png", bins=50)

    # ----- fraud rate по срезам -----
    if fraud_col:
        totals = {"rows": 0, "fraud_1": 0}
        by_cols = {
            "vendor_category": vcat_col,
            "vendor_type": vtype_col,
            "city": city_col,
            "card_type": card_col,
            "is_high_risk_vendor": hr_col,
        }
        agg_frames = {k: [] for k in by_cols if by_cols[k] is not None}

        for rg in range(pf.num_row_groups):
            cols = [c for c in [fraud_col, vcat_col, vtype_col, city_col, card_col, hr_col] if c is not None]
            if not cols:
                break
            df = pf.read_row_group(rg, columns=cols).to_pandas()
            f = to_bool(df[fraud_col])
            totals["rows"] += len(f)
            totals["fraud_1"] += int(f.sum())

            for name, col in by_cols.items():
                if col is None:
                    continue
                tmp = pd.DataFrame({name: df[col].astype(str), "is_fraud": f.astype(int)})
                # сразу задаём имена агрегатов
                gr = (
                    tmp.groupby(name, as_index=False)
                       .agg(n=("is_fraud", "size"), fraud_rate=("is_fraud", "mean"))
                )
                agg_frames[name].append(gr)

        # склейка всех row groups и сохранение
        for name, parts in agg_frames.items():
            if not parts:
                continue
            full = pd.concat(parts, ignore_index=True)
            # если один и тот же ключ встретился в разных чанках — суммируем объёмы и усредняем rate взвешенно
            full = full.groupby(name, as_index=False).agg(
                n=("n", "sum"),
                fraud_rate=("fraud_rate", "mean")  # либо подставить взвешенное среднее
            )
            full.sort_values(["fraud_rate", "n"], ascending=[False, False], inplace=True)
            full.to_csv(OUT_DIR / f"fraud_rate_by_{name}.csv", index=False)
            save_bar(full.set_index(name)["fraud_rate"].head(20), f"Fraud rate by {name} (top 20)", f"fraud_rate_bars_{name}.png", top=20)

        pd.Series(totals).to_csv(OUT_DIR / "fraud_overall.csv")

    # ----- статистики чеков по городам -----
    if not usd_all.empty and city_col:
        sums: List[pd.DataFrame] = []
        for rg in range(pf.num_row_groups):
            cols = [city_col] + ([AMOUNT_USD_PRIORITY[0]] if amount_usd_col else [amount_col, currency_col, time_col])
            df = pf.read_row_group(rg, columns=cols).to_pandas()
            if amount_usd_col:
                df["_usd"] = pd.to_numeric(df[amount_usd_col], errors="coerce")
            else:
                dt = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.tz_localize(None).dt.date
                curr = df[currency_col].astype(str).str.upper()
                amt  = pd.to_numeric(df[amount_col], errors="coerce")
                tmp = pd.DataFrame({"date": dt, "currency": curr, "amount": amt}).dropna(subset=["date", "currency", "amount"])
                if tmp.empty:
                    continue
                merged = tmp.merge(load_fx_long(FX_PATH), on=["date", "currency"], how="left").dropna(subset=["rate"])
                df["_usd"] = merged["amount"] / merged["rate"]
            sub = df[[city_col, "_usd"]].dropna()
            if sub.empty:
                continue
            sums.append(sub)

        if sums:
            aa = pd.concat(sums, ignore_index=True)
            grp = aa.groupby(city_col)["_usd"].agg(["count", "mean", "median", "std"]).sort_values("mean", ascending=False)
            grp.to_csv(OUT_DIR / "amount_stats_by_city.csv")

    print(f"[EDA] Готов, файлы записаны в: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
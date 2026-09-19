from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    return {
        "production": pd.read_csv(RAW_DIR / "production_log.csv"),
        "downtime": pd.read_csv(RAW_DIR / "downtime_log.csv"),
        "maintenance": pd.read_csv(RAW_DIR / "maintenance_log.csv"),
        "machines": pd.read_csv(RAW_DIR / "machine_master.csv"),
        "products": pd.read_csv(RAW_DIR / "product_master.csv"),
        "targets": pd.read_csv(RAW_DIR / "production_targets.csv"),
    }


def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    return df


def standardize_text(series):
    return series.astype("string").str.strip()


def normalize_boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return series

    mapped = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    return mapped


quality_checks = []


def add_check(check_name, passed, details):
    quality_checks.append(
        {
            "check_name": check_name,
            "status": "PASS" if bool(passed) else "FAIL",
            "details": str(details),
        }
    )


data = load_data()

production = standardize_columns(data["production"])
downtime = standardize_columns(data["downtime"])
maintenance = standardize_columns(data["maintenance"])
machines = standardize_columns(data["machines"])
products = standardize_columns(data["products"])
targets = standardize_columns(data["targets"])

print("\n--------------------------------")
print("RAW DATA LOADED")
print("--------------------------------")
print(f"Production rows: {len(production):,}")
print(f"Downtime rows: {len(downtime):,}")
print(f"Maintenance rows: {len(maintenance):,}")
print(f"Machines: {len(machines):,}")
print(f"Products: {len(products):,}")
print(f"Targets: {len(targets):,}")


production["timestamp"] = pd.to_datetime(
    production["timestamp"],
    errors="coerce",
)
production["date"] = pd.to_datetime(
    production["date"],
    errors="coerce",
)

downtime["start_time"] = pd.to_datetime(
    downtime["start_time"],
    errors="coerce",
)
downtime["end_time"] = pd.to_datetime(
    downtime["end_time"],
    errors="coerce",
)

maintenance["timestamp"] = pd.to_datetime(
    maintenance["timestamp"],
    errors="coerce",
)

targets["date"] = pd.to_datetime(
    targets["date"],
    errors="coerce",
)

production["station_id"] = (
    standardize_text(production["station_id"]).str.upper()
)
production["product_id"] = (
    standardize_text(production["product_id"]).str.upper()
)
production["shift"] = (
    standardize_text(production["shift"]).str.upper()
)

downtime["station_id"] = (
    standardize_text(downtime["station_id"]).str.upper()
)
downtime["product_id"] = (
    standardize_text(downtime["product_id"]).str.upper()
)
downtime["shift"] = (
    standardize_text(downtime["shift"]).str.upper()
)

maintenance["station_id"] = (
    standardize_text(maintenance["station_id"]).str.upper()
)

machines["station_id"] = (
    standardize_text(machines["station_id"]).str.upper()
)

products["product_id"] = (
    standardize_text(products["product_id"]).str.upper()
)

targets["product_id"] = (
    standardize_text(targets["product_id"]).str.upper()
)
targets["shift"] = (
    standardize_text(targets["shift"]).str.upper()
)

if "planned_flag" in downtime.columns:
    downtime["planned_flag"] = normalize_boolean(
        downtime["planned_flag"]
    )

if "maintenance_required" in downtime.columns:
    downtime["maintenance_required"] = normalize_boolean(
        downtime["maintenance_required"]
    )

if "successful_restart" in maintenance.columns:
    maintenance["successful_restart"] = normalize_boolean(
        maintenance["successful_restart"]
    )


duplicate_production_ids = production["record_id"].duplicated().sum()
add_check(
    "Unique production record IDs",
    duplicate_production_ids == 0,
    f"Duplicates: {duplicate_production_ids}",
)

duplicate_downtime_ids = downtime["downtime_id"].duplicated().sum()
add_check(
    "Unique downtime IDs",
    duplicate_downtime_ids == 0,
    f"Duplicates: {duplicate_downtime_ids}",
)

duplicate_maintenance_ids = (
    maintenance["maintenance_id"].duplicated().sum()
)
add_check(
    "Unique maintenance IDs",
    duplicate_maintenance_ids == 0,
    f"Duplicates: {duplicate_maintenance_ids}",
)

duplicate_machine_ids = machines["station_id"].duplicated().sum()
add_check(
    "Unique machine master IDs",
    duplicate_machine_ids == 0,
    f"Duplicates: {duplicate_machine_ids}",
)

duplicate_product_ids = products["product_id"].duplicated().sum()
add_check(
    "Unique product master IDs",
    duplicate_product_ids == 0,
    f"Duplicates: {duplicate_product_ids}",
)


quantity_errors = (
    (
        production["good_quantity"]
        + production["rejected_quantity"]
    )
    != production["total_quantity"]
).sum()

add_check(
    "Production quantity reconciliation",
    quantity_errors == 0,
    f"Invalid rows: {quantity_errors}",
)


time_valid = np.isclose(
    production["runtime_minutes"]
    + production["downtime_minutes"],
    production["planned_minutes"],
    atol=0.02,
)

time_errors = (~time_valid).sum()

add_check(
    "Production time reconciliation",
    time_errors == 0,
    f"Invalid rows: {time_errors}",
)


production_numeric_columns = [
    "planned_minutes",
    "downtime_minutes",
    "runtime_minutes",
    "ideal_cycle_sec",
    "actual_cycle_sec",
    "total_quantity",
    "good_quantity",
    "rejected_quantity",
]

negative_values = (
    production[production_numeric_columns] < 0
).sum().sum()

add_check(
    "No negative production values",
    negative_values == 0,
    f"Negative values: {negative_values}",
)


valid_stations = set(machines["station_id"].dropna())

invalid_production_stations = (
    ~production["station_id"].isin(valid_stations)
).sum()

add_check(
    "Valid production station IDs",
    invalid_production_stations == 0,
    f"Invalid rows: {invalid_production_stations}",
)

invalid_downtime_stations = (
    ~downtime["station_id"].isin(valid_stations)
).sum()

add_check(
    "Valid downtime station IDs",
    invalid_downtime_stations == 0,
    f"Invalid rows: {invalid_downtime_stations}",
)

invalid_maintenance_stations = (
    ~maintenance["station_id"].isin(valid_stations)
).sum()

add_check(
    "Valid maintenance station IDs",
    invalid_maintenance_stations == 0,
    f"Invalid rows: {invalid_maintenance_stations}",
)


valid_products = set(products["product_id"].dropna())

invalid_production_products = (
    ~production["product_id"].isin(valid_products)
).sum()

add_check(
    "Valid production product IDs",
    invalid_production_products == 0,
    f"Invalid rows: {invalid_production_products}",
)

invalid_downtime_products = (
    ~downtime["product_id"].isin(valid_products)
).sum()

add_check(
    "Valid downtime product IDs",
    invalid_downtime_products == 0,
    f"Invalid rows: {invalid_downtime_products}",
)

invalid_target_products = (
    ~targets["product_id"].isin(valid_products)
).sum()

add_check(
    "Valid target product IDs",
    invalid_target_products == 0,
    f"Invalid rows: {invalid_target_products}",
)


valid_shifts = {"A", "B", "C"}

invalid_production_shifts = (
    ~production["shift"].isin(valid_shifts)
).sum()

add_check(
    "Valid production shifts",
    invalid_production_shifts == 0,
    f"Invalid rows: {invalid_production_shifts}",
)

invalid_downtime_shifts = (
    ~downtime["shift"].isin(valid_shifts)
).sum()

add_check(
    "Valid downtime shifts",
    invalid_downtime_shifts == 0,
    f"Invalid rows: {invalid_downtime_shifts}",
)

invalid_target_shifts = (
    ~targets["shift"].isin(valid_shifts)
).sum()

add_check(
    "Valid target shifts",
    invalid_target_shifts == 0,
    f"Invalid rows: {invalid_target_shifts}",
)


required_production_columns = [
    "record_id",
    "timestamp",
    "date",
    "shift",
    "station_id",
    "product_id",
    "planned_minutes",
    "runtime_minutes",
    "downtime_minutes",
    "ideal_cycle_sec",
    "actual_cycle_sec",
    "total_quantity",
    "good_quantity",
    "rejected_quantity",
]

missing_critical_production = (
    production[required_production_columns]
    .isna()
    .sum()
    .sum()
)

add_check(
    "No missing critical production values",
    missing_critical_production == 0,
    f"Missing values: {missing_critical_production}",
)


required_downtime_columns = [
    "downtime_id",
    "station_id",
    "start_time",
    "end_time",
    "duration_min",
    "shift",
    "category",
    "failure_type",
    "planned_flag",
    "product_id",
]

missing_critical_downtime = (
    downtime[required_downtime_columns]
    .isna()
    .sum()
    .sum()
)

add_check(
    "No missing critical downtime values",
    missing_critical_downtime == 0,
    f"Missing values: {missing_critical_downtime}",
)


block_counts = (
    production
    .groupby(
        [
            "date",
            "shift",
            "station_id",
        ]
    )
    .size()
)

invalid_block_groups = (block_counts != 30).sum()

add_check(
    "30 production blocks per station-shift",
    invalid_block_groups == 0,
    f"Invalid groups: {invalid_block_groups}",
)


products_per_shift = (
    production
    .groupby(
        [
            "date",
            "shift",
        ]
    )["product_id"]
    .nunique()
)

invalid_product_shifts = (
    products_per_shift != 1
).sum()

add_check(
    "One scheduled product per shift",
    invalid_product_shifts == 0,
    f"Invalid shifts: {invalid_product_shifts}",
)


downtime["calculated_duration_min"] = (
    downtime["end_time"]
    - downtime["start_time"]
).dt.total_seconds() / 60

duration_match = np.isclose(
    downtime["duration_min"],
    downtime["calculated_duration_min"],
    atol=0.02,
    equal_nan=False,
)

duration_errors = (~duration_match).sum()

add_check(
    "Downtime duration reconciliation",
    duration_errors == 0,
    f"Invalid events: {duration_errors}",
)

invalid_downtime_duration = (
    downtime["duration_min"] <= 0
).sum()

add_check(
    "Positive downtime duration",
    invalid_downtime_duration == 0,
    f"Invalid events: {invalid_downtime_duration}",
)


production_downtime_total = (
    production["downtime_minutes"].sum()
)

event_downtime_total = (
    downtime["duration_min"].sum()
)

downtime_reconciled = np.isclose(
    production_downtime_total,
    event_downtime_total,
    atol=0.10,
)

add_check(
    "Production vs downtime-event reconciliation",
    downtime_reconciled,
    (
        f"Production: {production_downtime_total:.2f} min | "
        f"Downtime log: {event_downtime_total:.2f} min"
    ),
)


production_downtime_ids = set(
    production["downtime_id"].dropna()
)

event_downtime_ids = set(
    downtime["downtime_id"].dropna()
)

missing_event_links = (
    production_downtime_ids - event_downtime_ids
)

unused_event_links = (
    event_downtime_ids - production_downtime_ids
)

add_check(
    "Production downtime IDs exist in downtime log",
    len(missing_event_links) == 0,
    f"Missing IDs: {len(missing_event_links)}",
)

add_check(
    "Downtime events link to production records",
    len(unused_event_links) == 0,
    f"Unlinked events: {len(unused_event_links)}",
)


duplicate_targets = (
    targets
    .duplicated(
        subset=[
            "date",
            "shift",
        ]
    )
    .sum()
)

add_check(
    "One production target per shift",
    duplicate_targets == 0,
    f"Duplicates: {duplicate_targets}",
)

actual_schedule = (
    production[
        [
            "date",
            "shift",
            "product_id",
        ]
    ]
    .drop_duplicates()
)

target_check = targets.merge(
    actual_schedule,
    on=[
        "date",
        "shift",
        "product_id",
    ],
    how="inner",
)

target_alignment = (
    len(target_check) == len(targets)
    and len(targets) == len(actual_schedule)
)

add_check(
    "Target product matches production schedule",
    target_alignment,
    (
        f"Matched: {len(target_check)} / "
        f"{len(targets)} targets"
    ),
)


valid_maintenance_types = {
    "Preventive",
    "Corrective",
}

invalid_maintenance_types = (
    ~maintenance["type"].isin(valid_maintenance_types)
).sum()

add_check(
    "Valid maintenance types",
    invalid_maintenance_types == 0,
    f"Invalid rows: {invalid_maintenance_types}",
)

invalid_maintenance_duration = (
    maintenance["duration_min"] <= 0
).sum()

add_check(
    "Positive maintenance duration",
    invalid_maintenance_duration == 0,
    f"Invalid rows: {invalid_maintenance_duration}",
)


production["year"] = production["timestamp"].dt.year
production["month"] = production["timestamp"].dt.month
production["month_name"] = production["timestamp"].dt.month_name()
production["week"] = (
    production["timestamp"]
    .dt.isocalendar()
    .week
    .astype("Int64")
)
production["day_name"] = production["timestamp"].dt.day_name()
production["hour"] = production["timestamp"].dt.hour

production["availability_numerator_min"] = (
    production["runtime_minutes"]
)
production["availability_denominator_min"] = (
    production["planned_minutes"]
)

production["performance_numerator_sec"] = (
    production["ideal_cycle_sec"]
    * production["total_quantity"]
)
production["performance_denominator_sec"] = (
    production["runtime_minutes"] * 60
)

production["quality_numerator_units"] = (
    production["good_quantity"]
)
production["quality_denominator_units"] = (
    production["total_quantity"]
)

production["scrap_rate"] = np.where(
    production["total_quantity"] > 0,
    production["rejected_quantity"]
    / production["total_quantity"],
    0.0,
)

production["cycle_variance_pct"] = np.where(
    production["ideal_cycle_sec"] > 0,
    (
        production["actual_cycle_sec"]
        - production["ideal_cycle_sec"]
    )
    / production["ideal_cycle_sec"],
    0.0,
)

production["has_downtime"] = (
    production["downtime_minutes"] > 0
)

production["is_planned_maintenance"] = (
    production["machine_status"]
    == "Planned Maintenance"
)


downtime["downtime_class"] = np.where(
    downtime["planned_flag"].fillna(False),
    "Planned",
    "Unplanned",
)

downtime["date"] = downtime["start_time"].dt.normalize()
downtime["year"] = downtime["start_time"].dt.year
downtime["month"] = downtime["start_time"].dt.month
downtime["month_name"] = downtime["start_time"].dt.month_name()
downtime["week"] = (
    downtime["start_time"]
    .dt.isocalendar()
    .week
    .astype("Int64")
)
downtime["day_name"] = downtime["start_time"].dt.day_name()
downtime["hour"] = downtime["start_time"].dt.hour


maintenance["date"] = maintenance["timestamp"].dt.normalize()
maintenance["year"] = maintenance["timestamp"].dt.year
maintenance["month"] = maintenance["timestamp"].dt.month
maintenance["month_name"] = maintenance["timestamp"].dt.month_name()
maintenance["week"] = (
    maintenance["timestamp"]
    .dt.isocalendar()
    .week
    .astype("Int64")
)
maintenance["day_name"] = maintenance["timestamp"].dt.day_name()


quality_report = pd.DataFrame(quality_checks)

print("\n--------------------------------")
print("DATA QUALITY REPORT")
print("--------------------------------")
print(quality_report.to_string(index=False))

quality_report.to_csv(
    PROCESSED_DIR / "data_quality_report.csv",
    index=False,
)

failed_checks = quality_report[
    quality_report["status"] == "FAIL"
]

if not failed_checks.empty:
    print("\n--------------------------------")
    print("DATA QUALITY CHECKS FAILED")
    print("--------------------------------")
    print(failed_checks.to_string(index=False))
    raise ValueError(
        "Processed files were not created because "
        "one or more data-quality checks failed."
    )


production.to_csv(
    PROCESSED_DIR / "production_clean.csv",
    index=False,
)

downtime.to_csv(
    PROCESSED_DIR / "downtime_clean.csv",
    index=False,
)

maintenance.to_csv(
    PROCESSED_DIR / "maintenance_clean.csv",
    index=False,
)

machines.to_csv(
    PROCESSED_DIR / "machine_master.csv",
    index=False,
)

products.to_csv(
    PROCESSED_DIR / "product_master.csv",
    index=False,
)

targets.to_csv(
    PROCESSED_DIR / "production_targets_clean.csv",
    index=False,
)


print("\n--------------------------------")
print("DATA CLEANING COMPLETE")
print("--------------------------------")
print(f"Processed production rows: {len(production):,}")
print(f"Processed downtime events: {len(downtime):,}")
print(f"Processed maintenance events: {len(maintenance):,}")
print(f"Quality checks passed: {len(quality_report):,}")
print(f"Output folder: {PROCESSED_DIR}")

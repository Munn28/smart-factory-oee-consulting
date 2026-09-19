from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "kpis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAILURE_CATEGORIES = {"Equipment Failure", "Sensor/Control"}


def safe_divide(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def weighted_average(values, weights):
    valid = values.notna() & weights.notna() & (weights > 0)

    if not valid.any():
        return np.nan

    return np.average(
        values[valid],
        weights=weights[valid],
    )


def calculate_production_kpis(df):
    planned_min = df["planned_minutes"].sum()
    runtime_min = df["runtime_minutes"].sum()
    downtime_min = df["downtime_minutes"].sum()

    total_qty = df["total_quantity"].sum()
    good_qty = df["good_quantity"].sum()
    rejected_qty = df["rejected_quantity"].sum()

    availability = safe_divide(
        runtime_min,
        planned_min,
    )

    performance_numerator = (
        df["performance_numerator_sec"].sum()
    )

    performance_denominator = (
        df["performance_denominator_sec"].sum()
    )

    performance = safe_divide(
        performance_numerator,
        performance_denominator,
    )

    quality = safe_divide(
        good_qty,
        total_qty,
    )

    oee = (
        availability
        * performance
        * quality
        if not any(
            pd.isna(x)
            for x in [
                availability,
                performance,
                quality,
            ]
        )
        else np.nan
    )

    scrap_rate = safe_divide(
        rejected_qty,
        total_qty,
    )

    throughput = safe_divide(
        good_qty,
        planned_min / 60,
    )

    actual_cycle_sec = weighted_average(
        df["actual_cycle_sec"],
        df["total_quantity"],
    )

    theoretical_capacity = (
        (
            df["planned_minutes"] * 60
        )
        / df["ideal_cycle_sec"]
    ).sum()

    capacity_utilization = safe_divide(
        total_qty,
        theoretical_capacity,
    )

    return {
        "planned_minutes": round(planned_min, 2),
        "runtime_minutes": round(runtime_min, 2),
        "downtime_minutes": round(downtime_min, 2),
        "total_quantity": int(total_qty),
        "good_quantity": int(good_qty),
        "rejected_quantity": int(rejected_qty),
        "availability": availability,
        "performance": performance,
        "quality": quality,
        "oee": oee,
        "scrap_rate": scrap_rate,
        "throughput_units_per_planned_hour": throughput,
        "weighted_actual_cycle_sec": actual_cycle_sec,
        "theoretical_capacity_units": theoretical_capacity,
        "capacity_utilization": capacity_utilization,
    }


def calculate_grouped_kpis(df, group_columns):
    rows = []

    for keys, group in df.groupby(
        group_columns,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        result = dict(zip(group_columns, keys))
        result.update(calculate_production_kpis(group))
        rows.append(result)

    return pd.DataFrame(rows)


def calculate_reliability(downtime, production):
    failures = downtime[
        (~downtime["planned_flag"])
        & downtime["category"].isin(
            FAILURE_CATEGORIES
        )
    ].copy()

    runtime_by_station = (
        production
        .groupby("station_id")[
            "runtime_minutes"
        ]
        .sum()
    )

    failure_summary = (
        failures
        .groupby("station_id")
        .agg(
            failure_count=(
                "downtime_id",
                "count",
            ),
            repair_minutes=(
                "duration_min",
                "sum",
            ),
        )
    )

    reliability = (
        pd.DataFrame(
            {
                "runtime_minutes":
                    runtime_by_station
            }
        )
        .join(
            failure_summary,
            how="left",
        )
        .fillna(
            {
                "failure_count": 0,
                "repair_minutes": 0,
            }
        )
        .reset_index()
    )

    reliability["failure_count"] = (
        reliability[
            "failure_count"
        ].astype(int)
    )

    reliability["mtbf_hours"] = np.where(
        reliability["failure_count"] > 0,
        (
            reliability["runtime_minutes"]
            / 60
        )
        / reliability["failure_count"],
        np.nan,
    )

    reliability["mttr_minutes"] = np.where(
        reliability["failure_count"] > 0,
        reliability["repair_minutes"]
        / reliability["failure_count"],
        np.nan,
    )

    return reliability


def calculate_target_attainment(
    production,
    targets,
):
    station_output = (
        production
        .groupby(
            [
                "date",
                "shift",
                "product_id",
                "station_id",
            ],
            as_index=False,
        )
        .agg(
            total_quantity=(
                "total_quantity",
                "sum",
            ),
            good_quantity=(
                "good_quantity",
                "sum",
            ),
            rejected_quantity=(
                "rejected_quantity",
                "sum",
            ),
            planned_minutes=(
                "planned_minutes",
                "sum",
            ),
        )
    )

    constraint_index = (
        station_output
        .groupby(
            [
                "date",
                "shift",
                "product_id",
            ]
        )["good_quantity"]
        .idxmin()
    )

    line_output = (
        station_output
        .loc[constraint_index]
        .rename(
            columns={
                "station_id":
                    "constraint_station_id",
                "total_quantity":
                    "actual_total_output",
                "good_quantity":
                    "actual_good_output",
                "rejected_quantity":
                    "actual_rejected_output",
            }
        )
        .reset_index(drop=True)
    )

    target_attainment = targets.merge(
        line_output,
        on=[
            "date",
            "shift",
            "product_id",
        ],
        how="left",
        validate="one_to_one",
    )

    target_attainment[
        "output_attainment"
    ] = (
        target_attainment[
            "actual_good_output"
        ]
        / target_attainment[
            "target_good_output"
        ]
    )

    target_attainment[
        "output_gap_units"
    ] = (
        target_attainment[
            "actual_good_output"
        ]
        - target_attainment[
            "target_good_output"
        ]
    )

    target_attainment[
        "takt_time_sec"
    ] = (
        target_attainment[
            "planned_minutes"
        ]
        * 60
        / target_attainment[
            "target_good_output"
        ]
    )

    target_attainment[
        "actual_throughput_units_per_hour"
    ] = (
        target_attainment[
            "actual_good_output"
        ]
        / (
            target_attainment[
                "planned_minutes"
            ]
            / 60
        )
    )

    return target_attainment


def calculate_factory_reliability(
    reliability_by_station,
):
    failure_count = (
        reliability_by_station[
            "failure_count"
        ].sum()
    )

    repair_minutes = (
        reliability_by_station[
            "repair_minutes"
        ].sum()
    )

    runtime_minutes = (
        reliability_by_station[
            "runtime_minutes"
        ].sum()
    )

    return {
        "functional_failure_count":
            int(failure_count),
        "mtbf_hours":
            safe_divide(
                runtime_minutes / 60,
                failure_count,
            ),
        "mttr_minutes":
            safe_divide(
                repair_minutes,
                failure_count,
            ),
    }


production = pd.read_csv(
    PROCESSED_DIR / "production_clean.csv",
    parse_dates=["timestamp", "date"],
)

downtime = pd.read_csv(
    PROCESSED_DIR / "downtime_clean.csv",
    parse_dates=[
        "start_time",
        "end_time",
        "date",
    ],
)

maintenance = pd.read_csv(
    PROCESSED_DIR / "maintenance_clean.csv",
    parse_dates=[
        "timestamp",
        "date",
    ],
)

machines = pd.read_csv(
    PROCESSED_DIR / "machine_master.csv"
)

products = pd.read_csv(
    PROCESSED_DIR / "product_master.csv"
)

targets = pd.read_csv(
    PROCESSED_DIR
    / "production_targets_clean.csv",
    parse_dates=["date"],
)


factory_kpis = calculate_production_kpis(
    production
)

station_kpis = calculate_grouped_kpis(
    production,
    ["station_id"],
)

station_kpis = station_kpis.merge(
    machines[
        [
            "station_id",
            "station_name",
            "station_type",
        ]
    ],
    on="station_id",
    how="left",
)

shift_kpis = calculate_grouped_kpis(
    production,
    ["shift"],
)

product_kpis = calculate_grouped_kpis(
    production,
    ["product_id"],
)

product_kpis = product_kpis.merge(
    products[
        [
            "product_id",
            "product_name",
        ]
    ],
    on="product_id",
    how="left",
)

daily_kpis = calculate_grouped_kpis(
    production,
    ["date"],
)

station_shift_kpis = calculate_grouped_kpis(
    production,
    [
        "station_id",
        "shift",
    ],
)

station_product_kpis = calculate_grouped_kpis(
    production,
    [
        "station_id",
        "product_id",
    ],
)

reliability_by_station = (
    calculate_reliability(
        downtime,
        production,
    )
)

reliability_by_station = (
    reliability_by_station.merge(
        machines[
            [
                "station_id",
                "station_name",
            ]
        ],
        on="station_id",
        how="left",
    )
)

factory_reliability = (
    calculate_factory_reliability(
        reliability_by_station
    )
)

target_attainment = (
    calculate_target_attainment(
        production,
        targets,
    )
)

line_summary = {
    "finished_good_output":
        int(
            target_attainment[
                "actual_good_output"
            ].sum()
        ),
    "target_good_output":
        int(
            target_attainment[
                "target_good_output"
            ].sum()
        ),
    "output_attainment":
        safe_divide(
            target_attainment[
                "actual_good_output"
            ].sum(),
            target_attainment[
                "target_good_output"
            ].sum(),
        ),
    "output_gap_units":
        int(
            target_attainment[
                "actual_good_output"
            ].sum()
            - target_attainment[
                "target_good_output"
            ].sum()
        ),
    "average_takt_time_sec":
        weighted_average(
            target_attainment[
                "takt_time_sec"
            ],
            target_attainment[
                "target_good_output"
            ],
        ),
    "finished_goods_scrap_rate":
        safe_divide(
            target_attainment[
                "actual_rejected_output"
            ].sum(),
            target_attainment[
                "actual_total_output"
            ].sum(),
        ),
    "finished_goods_throughput_per_hour":
        safe_divide(
            target_attainment[
                "actual_good_output"
            ].sum(),
            target_attainment[
                "planned_minutes"
            ].sum()
            / 60,
        ),
}


factory_summary = {
    **factory_kpis,
    **factory_reliability,
    **line_summary,
}

factory_summary_df = pd.DataFrame(
    [factory_summary]
)


station_kpis.to_csv(
    OUTPUT_DIR / "kpi_by_station.csv",
    index=False,
)

shift_kpis.to_csv(
    OUTPUT_DIR / "kpi_by_shift.csv",
    index=False,
)

product_kpis.to_csv(
    OUTPUT_DIR / "kpi_by_product.csv",
    index=False,
)

daily_kpis.to_csv(
    OUTPUT_DIR / "kpi_daily.csv",
    index=False,
)

station_shift_kpis.to_csv(
    OUTPUT_DIR
    / "kpi_by_station_shift.csv",
    index=False,
)

station_product_kpis.to_csv(
    OUTPUT_DIR
    / "kpi_by_station_product.csv",
    index=False,
)

reliability_by_station.to_csv(
    OUTPUT_DIR
    / "reliability_by_station.csv",
    index=False,
)

target_attainment.to_csv(
    OUTPUT_DIR
    / "target_attainment.csv",
    index=False,
)

factory_summary_df.to_csv(
    OUTPUT_DIR
    / "factory_summary.csv",
    index=False,
)


percent_columns = [
    "availability",
    "performance",
    "quality",
    "oee",
    "scrap_rate",
    "capacity_utilization",
]

display_station = station_kpis.copy()

for column in percent_columns:
    if column in display_station.columns:
        display_station[column] = (
            display_station[column]
            * 100
        ).round(2)

print("\n--------------------------------")
print("FACTORY KPI SUMMARY")
print("--------------------------------")
print(
    f"OEE: "
    f"{factory_kpis['oee'] * 100:.2f}%"
)
print(
    f"Availability: "
    f"{factory_kpis['availability'] * 100:.2f}%"
)
print(
    f"Performance: "
    f"{factory_kpis['performance'] * 100:.2f}%"
)
print(
    f"Quality: "
    f"{factory_kpis['quality'] * 100:.2f}%"
)
print(
    f"MTBF: "
    f"{factory_reliability['mtbf_hours']:.2f} h"
)
print(
    f"MTTR: "
    f"{factory_reliability['mttr_minutes']:.2f} min"
)
print(
    f"Finished good output: "
    f"{line_summary['finished_good_output']:,}"
)
print(
    f"Target attainment: "
    f"{line_summary['output_attainment'] * 100:.2f}%"
)
print(
    f"Finished-goods throughput: "
    f"{line_summary['finished_goods_throughput_per_hour']:.2f} units/h"
)

print("\n--------------------------------")
print("STATION KPI SUMMARY")
print("--------------------------------")
print(
    display_station[
        [
            "station_id",
            "station_name",
            "availability",
            "performance",
            "quality",
            "oee",
            "scrap_rate",
            "capacity_utilization",
        ]
    ].to_string(index=False)
)

print("\n--------------------------------")
print("KPI CALCULATION COMPLETE")
print("--------------------------------")
print(f"Output folder: {OUTPUT_DIR}")

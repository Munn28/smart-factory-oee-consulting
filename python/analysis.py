from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
KPI_DIR = PROCESSED_DIR / "kpis"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"
CHART_DIR = PROJECT_ROOT / "screenshots" / "day5"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


production = pd.read_csv(
    PROCESSED_DIR / "production_clean.csv",
    parse_dates=["timestamp", "date"],
)

downtime = pd.read_csv(
    PROCESSED_DIR / "downtime_clean.csv",
    parse_dates=["start_time", "end_time", "date"],
)

machines = pd.read_csv(
    PROCESSED_DIR / "machine_master.csv"
)

station_kpis = pd.read_csv(
    KPI_DIR / "kpi_by_station.csv"
)

shift_kpis = pd.read_csv(
    KPI_DIR / "kpi_by_shift.csv"
)

product_kpis = pd.read_csv(
    KPI_DIR / "kpi_by_product.csv"
)

daily_kpis = pd.read_csv(
    KPI_DIR / "kpi_daily.csv",
    parse_dates=["date"],
)

station_shift_kpis = pd.read_csv(
    KPI_DIR / "kpi_by_station_shift.csv"
)

reliability = pd.read_csv(
    KPI_DIR / "reliability_by_station.csv"
)

target_attainment = pd.read_csv(
    KPI_DIR / "target_attainment.csv",
    parse_dates=["date"],
)


station_lookup = machines[
    ["station_id", "station_name"]
].drop_duplicates()


# Downtime by station
downtime_by_station = (
    downtime
    .groupby("station_id", as_index=False)
    .agg(
        downtime_minutes=("duration_min", "sum"),
        downtime_events=("downtime_id", "count"),
        average_event_duration_min=("duration_min", "mean"),
    )
    .merge(
        station_lookup,
        on="station_id",
        how="left",
    )
    .sort_values(
        "downtime_minutes",
        ascending=False,
    )
)

total_downtime = downtime_by_station[
    "downtime_minutes"
].sum()

downtime_by_station[
    "downtime_share"
] = (
    downtime_by_station["downtime_minutes"]
    / total_downtime
)

downtime_by_station.to_csv(
    ANALYSIS_DIR / "downtime_by_station.csv",
    index=False,
)


# Planned vs unplanned downtime
planned_vs_unplanned = (
    downtime
    .groupby(
        ["station_id", "downtime_class"],
        as_index=False,
    )
    .agg(
        downtime_minutes=("duration_min", "sum"),
        downtime_events=("downtime_id", "count"),
    )
    .merge(
        station_lookup,
        on="station_id",
        how="left",
    )
)

planned_vs_unplanned.to_csv(
    ANALYSIS_DIR / "planned_vs_unplanned.csv",
    index=False,
)


# Unplanned downtime Pareto
unplanned = downtime[
    downtime["downtime_class"] == "Unplanned"
].copy()

downtime_by_category = (
    unplanned
    .groupby("category", as_index=False)
    .agg(
        downtime_minutes=("duration_min", "sum"),
        event_count=("downtime_id", "count"),
        average_duration_min=("duration_min", "mean"),
    )
    .sort_values(
        "downtime_minutes",
        ascending=False,
    )
)

category_total = downtime_by_category[
    "downtime_minutes"
].sum()

downtime_by_category[
    "downtime_share"
] = (
    downtime_by_category["downtime_minutes"]
    / category_total
)

downtime_by_category[
    "cumulative_share"
] = downtime_by_category[
    "downtime_share"
].cumsum()

downtime_by_category.to_csv(
    ANALYSIS_DIR / "downtime_by_category.csv",
    index=False,
)


failure_pareto = (
    unplanned
    .groupby(
        ["failure_type", "category"],
        as_index=False,
    )
    .agg(
        downtime_minutes=("duration_min", "sum"),
        event_count=("downtime_id", "count"),
        average_duration_min=("duration_min", "mean"),
    )
    .sort_values(
        "downtime_minutes",
        ascending=False,
    )
)

failure_total = failure_pareto[
    "downtime_minutes"
].sum()

failure_pareto[
    "downtime_share"
] = (
    failure_pareto["downtime_minutes"]
    / failure_total
)

failure_pareto[
    "cumulative_share"
] = failure_pareto[
    "downtime_share"
].cumsum()

failure_pareto.to_csv(
    ANALYSIS_DIR / "downtime_pareto_failure_type.csv",
    index=False,
)


# Failure frequency and severity
failure_frequency_severity = (
    unplanned
    .groupby(
        [
            "station_id",
            "failure_type",
            "category",
        ],
        as_index=False,
    )
    .agg(
        event_count=("downtime_id", "count"),
        total_downtime_min=("duration_min", "sum"),
        average_duration_min=("duration_min", "mean"),
        max_duration_min=("duration_min", "max"),
    )
    .merge(
        station_lookup,
        on="station_id",
        how="left",
    )
    .sort_values(
        "total_downtime_min",
        ascending=False,
    )
)

failure_frequency_severity.to_csv(
    ANALYSIS_DIR / "failure_frequency_severity.csv",
    index=False,
)


# Bottleneck / constraint frequency
if "constraint_station_id" not in target_attainment.columns:
    raise ValueError(
        "constraint_station_id is missing from target_attainment.csv. "
        "Run the corrected Day 4 calculate_kpis.py first."
    )

constraint_frequency = (
    target_attainment
    .groupby(
        "constraint_station_id",
        as_index=False,
    )
    .agg(
        constraint_shifts=("target_id", "count"),
        average_output_attainment=("output_attainment", "mean"),
        total_output_gap_units=("output_gap_units", "sum"),
    )
    .rename(
        columns={
            "constraint_station_id": "station_id"
        }
    )
    .merge(
        station_lookup,
        on="station_id",
        how="left",
    )
)

total_shifts = len(target_attainment)

constraint_frequency[
    "constraint_share"
] = (
    constraint_frequency["constraint_shifts"]
    / total_shifts
)

constraint_frequency = constraint_frequency.sort_values(
    "constraint_shifts",
    ascending=False,
)

constraint_frequency.to_csv(
    ANALYSIS_DIR / "constraint_frequency.csv",
    index=False,
)


# Shift comparison
shift_summary = shift_kpis[
    [
        "shift",
        "availability",
        "performance",
        "quality",
        "oee",
        "scrap_rate",
        "throughput_units_per_planned_hour",
        "capacity_utilization",
    ]
].copy()

shift_summary.to_csv(
    ANALYSIS_DIR / "shift_summary.csv",
    index=False,
)


# Station-shift comparison
station_shift_summary = (
    station_shift_kpis
    .merge(
        station_lookup,
        on="station_id",
        how="left",
    )
    .sort_values(
        ["station_id", "shift"]
    )
)

station_shift_summary.to_csv(
    ANALYSIS_DIR / "station_shift_summary.csv",
    index=False,
)


# Product comparison
product_summary = product_kpis[
    [
        "product_id",
        "product_name",
        "availability",
        "performance",
        "quality",
        "oee",
        "scrap_rate",
        "throughput_units_per_planned_hour",
        "capacity_utilization",
    ]
].copy()

product_summary.to_csv(
    ANALYSIS_DIR / "product_summary.csv",
    index=False,
)


# Daily production trends
daily_summary = daily_kpis[
    [
        "date",
        "availability",
        "performance",
        "quality",
        "oee",
        "downtime_minutes",
        "good_quantity",
        "rejected_quantity",
        "scrap_rate",
    ]
].copy()

daily_summary.to_csv(
    ANALYSIS_DIR / "daily_trends.csv",
    index=False,
)


# Root-cause candidate table
root_cause_candidates = (
    failure_frequency_severity
    .head(15)
    .copy()
)

root_cause_candidates[
    "downtime_share_of_unplanned"
] = (
    root_cause_candidates[
        "total_downtime_min"
    ]
    / unplanned["duration_min"].sum()
)

root_cause_candidates[
    "candidate_reason"
] = np.select(
    [
        (
            root_cause_candidates["event_count"]
            >= root_cause_candidates[
                "event_count"
            ].median()
        )
        & (
            root_cause_candidates[
                "average_duration_min"
            ]
            >= root_cause_candidates[
                "average_duration_min"
            ].median()
        ),
        root_cause_candidates["event_count"]
        >= root_cause_candidates[
            "event_count"
        ].median(),
        root_cause_candidates[
            "average_duration_min"
        ]
        >= root_cause_candidates[
            "average_duration_min"
        ].median(),
    ],
    [
        "High frequency and high severity",
        "High frequency",
        "High severity",
    ],
    default="Moderate impact",
)

root_cause_candidates.to_csv(
    ANALYSIS_DIR / "root_cause_candidates.csv",
    index=False,
)


# Chart 1: OEE by station
plot_station = station_kpis.copy()

if "station_name" not in plot_station.columns:
    plot_station = plot_station.merge(
        station_lookup,
        on="station_id",
        how="left",
    )

plt.figure(figsize=(9, 5))
plt.bar(
    plot_station["station_name"],
    plot_station["oee"] * 100,
)
plt.ylabel("OEE (%)")
plt.title("OEE by Station")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "01_oee_by_station.png",
    dpi=200,
)
plt.close()


# Chart 2: Downtime Pareto
pareto_plot = failure_pareto.head(10).copy()

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(
    pareto_plot["failure_type"],
    pareto_plot["downtime_minutes"],
)
ax1.set_ylabel("Downtime (min)")
ax1.set_title("Top 10 Unplanned Downtime Causes")
ax1.tick_params(axis="x", rotation=35)

ax2 = ax1.twinx()
ax2.plot(
    pareto_plot["failure_type"],
    pareto_plot["cumulative_share"] * 100,
    marker="o",
)
ax2.set_ylabel("Cumulative Share (%)")
ax2.set_ylim(0, 110)

fig.tight_layout()
fig.savefig(
    CHART_DIR / "02_downtime_pareto.png",
    dpi=200,
)
plt.close(fig)


# Chart 3: Downtime by station
plt.figure(figsize=(9, 5))
plt.bar(
    downtime_by_station["station_name"],
    downtime_by_station["downtime_minutes"],
)
plt.ylabel("Downtime (min)")
plt.title("Total Downtime by Station")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "03_downtime_by_station.png",
    dpi=200,
)
plt.close()


# Chart 4: OEE by shift
plt.figure(figsize=(7, 5))
plt.bar(
    shift_summary["shift"],
    shift_summary["oee"] * 100,
)
plt.ylabel("OEE (%)")
plt.xlabel("Shift")
plt.title("OEE by Shift")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "04_oee_by_shift.png",
    dpi=200,
)
plt.close()


# Chart 5: Constraint frequency
plt.figure(figsize=(9, 5))
plt.bar(
    constraint_frequency["station_name"],
    constraint_frequency["constraint_share"] * 100,
)
plt.ylabel("Constraint Share of Shifts (%)")
plt.title("Line Constraint Frequency by Station")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "05_constraint_frequency.png",
    dpi=200,
)
plt.close()


# Chart 6: Daily OEE trend
plt.figure(figsize=(10, 5))
plt.plot(
    daily_summary["date"],
    daily_summary["oee"] * 100,
    marker="o",
)
plt.ylabel("OEE (%)")
plt.xlabel("Date")
plt.title("Daily OEE Trend")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "06_daily_oee_trend.png",
    dpi=200,
)
plt.close()


# Chart 7: Failure frequency vs severity
plt.figure(figsize=(9, 6))
plt.scatter(
    failure_frequency_severity["event_count"],
    failure_frequency_severity[
        "average_duration_min"
    ],
)

for row in failure_frequency_severity.head(8).itertuples():
    plt.annotate(
        row.failure_type,
        (
            row.event_count,
            row.average_duration_min,
        ),
        fontsize=8,
    )

plt.xlabel("Failure Event Count")
plt.ylabel("Average Event Duration (min)")
plt.title("Failure Frequency vs Severity")
plt.tight_layout()
plt.savefig(
    CHART_DIR
    / "07_failure_frequency_vs_severity.png",
    dpi=200,
)
plt.close()


top_station = downtime_by_station.iloc[0]
top_failure = failure_pareto.iloc[0]
top_constraint = constraint_frequency.iloc[0]
lowest_oee_station = plot_station.loc[
    plot_station["oee"].idxmin()
]
lowest_availability_station = plot_station.loc[
    plot_station["availability"].idxmin()
]
lowest_performance_station = plot_station.loc[
    plot_station["performance"].idxmin()
]
lowest_quality_station = plot_station.loc[
    plot_station["quality"].idxmin()
]

print("\n--------------------------------")
print("DAY 5 ANALYSIS SUMMARY")
print("--------------------------------")
print(
    f"Highest downtime station: "
    f"{top_station['station_id']} - "
    f"{top_station['station_name']} "
    f"({top_station['downtime_minutes']:.2f} min)"
)
print(
    f"Top unplanned downtime cause: "
    f"{top_failure['failure_type']} "
    f"({top_failure['downtime_minutes']:.2f} min, "
    f"{top_failure['downtime_share'] * 100:.2f}% "
    f"of unplanned downtime)"
)
print(
    f"Most frequent line constraint: "
    f"{top_constraint['station_id']} - "
    f"{top_constraint['station_name']} "
    f"({top_constraint['constraint_share'] * 100:.2f}% "
    f"of shifts)"
)
print(
    f"Lowest OEE: "
    f"{lowest_oee_station['station_id']} - "
    f"{lowest_oee_station['station_name']} "
    f"({lowest_oee_station['oee'] * 100:.2f}%)"
)
print(
    f"Lowest availability: "
    f"{lowest_availability_station['station_id']} - "
    f"{lowest_availability_station['station_name']} "
    f"({lowest_availability_station['availability'] * 100:.2f}%)"
)
print(
    f"Lowest performance: "
    f"{lowest_performance_station['station_id']} - "
    f"{lowest_performance_station['station_name']} "
    f"({lowest_performance_station['performance'] * 100:.2f}%)"
)
print(
    f"Lowest quality: "
    f"{lowest_quality_station['station_id']} - "
    f"{lowest_quality_station['station_name']} "
    f"({lowest_quality_station['quality'] * 100:.2f}%)"
)

print("\n--------------------------------")
print("DAY 5 ANALYSIS COMPLETE")
print("--------------------------------")
print(f"Analysis tables: {ANALYSIS_DIR}")
print(f"Charts: {CHART_DIR}")

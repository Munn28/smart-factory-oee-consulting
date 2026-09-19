from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
KPI_DIR = PROCESSED_DIR / "kpis"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"
ROOT_CAUSE_DIR = PROCESSED_DIR / "root_cause"
CHART_DIR = PROJECT_ROOT / "screenshots" / "day6"

ROOT_CAUSE_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_STATION = "ST04"
SECONDARY_STATION = "ST02"
SENSOR_FAILURE = "Part Presence Sensor"
WORKING_DAYS_PER_YEAR = 250
OBSERVATION_WORKING_DAYS = 30


production = pd.read_csv(
    PROCESSED_DIR / "production_clean.csv",
    parse_dates=["timestamp", "date"],
)

downtime = pd.read_csv(
    PROCESSED_DIR / "downtime_clean.csv",
    parse_dates=["start_time", "end_time", "date"],
)

station_kpis = pd.read_csv(
    KPI_DIR / "kpi_by_station.csv"
)

station_shift_kpis = pd.read_csv(
    KPI_DIR / "kpi_by_station_shift.csv"
)

station_product_kpis = pd.read_csv(
    KPI_DIR / "kpi_by_station_product.csv"
)

target_attainment = pd.read_csv(
    KPI_DIR / "target_attainment.csv",
    parse_dates=["date"],
)

machines = pd.read_csv(
    PROCESSED_DIR / "machine_master.csv"
)

products = pd.read_csv(
    PROCESSED_DIR / "product_master.csv"
)


station_lookup = machines[
    ["station_id", "station_name"]
].drop_duplicates()

product_lookup = products[
    ["product_id", "product_name"]
].drop_duplicates()


# ST04 constraint and performance deep dive
st04_kpi = station_kpis[
    station_kpis["station_id"] == PRIMARY_STATION
].copy()

st04_shift = (
    station_shift_kpis[
        station_shift_kpis["station_id"] == PRIMARY_STATION
    ]
    .copy()
    .sort_values("oee")
)

st04_shift.to_csv(
    ROOT_CAUSE_DIR / "st04_shift_analysis.csv",
    index=False,
)

st04_product = (
    station_product_kpis[
        station_product_kpis["station_id"] == PRIMARY_STATION
    ]
    .merge(
        product_lookup,
        on="product_id",
        how="left",
    )
    .sort_values("performance")
)

st04_product.to_csv(
    ROOT_CAUSE_DIR / "st04_product_analysis.csv",
    index=False,
)


st04_production = production[
    production["station_id"] == PRIMARY_STATION
].copy()

st04_cycle_by_shift = (
    st04_production
    .groupby("shift", as_index=False)
    .agg(
        observations=("record_id", "count"),
        mean_ideal_cycle_sec=("ideal_cycle_sec", "mean"),
        mean_actual_cycle_sec=("actual_cycle_sec", "mean"),
        median_actual_cycle_sec=("actual_cycle_sec", "median"),
        p90_actual_cycle_sec=(
            "actual_cycle_sec",
            lambda x: x.quantile(0.90),
        ),
        mean_cycle_variance_pct=("cycle_variance_pct", "mean"),
        good_quantity=("good_quantity", "sum"),
        downtime_minutes=("downtime_minutes", "sum"),
    )
    .sort_values(
        "mean_actual_cycle_sec",
        ascending=False,
    )
)

st04_cycle_by_shift.to_csv(
    ROOT_CAUSE_DIR / "st04_cycle_by_shift.csv",
    index=False,
)


st04_cycle_by_product = (
    st04_production
    .groupby("product_id", as_index=False)
    .agg(
        observations=("record_id", "count"),
        mean_ideal_cycle_sec=("ideal_cycle_sec", "mean"),
        mean_actual_cycle_sec=("actual_cycle_sec", "mean"),
        median_actual_cycle_sec=("actual_cycle_sec", "median"),
        p90_actual_cycle_sec=(
            "actual_cycle_sec",
            lambda x: x.quantile(0.90),
        ),
        mean_cycle_variance_pct=("cycle_variance_pct", "mean"),
        good_quantity=("good_quantity", "sum"),
        downtime_minutes=("downtime_minutes", "sum"),
    )
    .merge(
        product_lookup,
        on="product_id",
        how="left",
    )
    .sort_values(
        "mean_actual_cycle_sec",
        ascending=False,
    )
)

st04_cycle_by_product.to_csv(
    ROOT_CAUSE_DIR / "st04_cycle_by_product.csv",
    index=False,
)


st04_downtime = downtime[
    (downtime["station_id"] == PRIMARY_STATION)
    & (downtime["downtime_class"] == "Unplanned")
].copy()

st04_downtime_breakdown = (
    st04_downtime
    .groupby(
        ["category", "failure_type"],
        as_index=False,
    )
    .agg(
        event_count=("downtime_id", "count"),
        downtime_minutes=("duration_min", "sum"),
        average_duration_min=("duration_min", "mean"),
        max_duration_min=("duration_min", "max"),
    )
    .sort_values(
        "downtime_minutes",
        ascending=False,
    )
)

st04_unplanned_total = st04_downtime[
    "duration_min"
].sum()

st04_downtime_breakdown[
    "share_of_st04_unplanned"
] = np.where(
    st04_unplanned_total > 0,
    st04_downtime_breakdown["downtime_minutes"]
    / st04_unplanned_total,
    np.nan,
)

st04_downtime_breakdown.to_csv(
    ROOT_CAUSE_DIR / "st04_downtime_breakdown.csv",
    index=False,
)


total_shifts = len(target_attainment)

st04_constraint_shifts = (
    target_attainment["constraint_station_id"]
    == PRIMARY_STATION
).sum()

st04_constraint_share = (
    st04_constraint_shifts / total_shifts
    if total_shifts > 0
    else np.nan
)


# ST02 Part Presence Sensor deep dive
st02_unplanned = downtime[
    (downtime["station_id"] == SECONDARY_STATION)
    & (downtime["downtime_class"] == "Unplanned")
].copy()

sensor_events = st02_unplanned[
    st02_unplanned["failure_type"] == SENSOR_FAILURE
].copy()

sensor_total_minutes = sensor_events[
    "duration_min"
].sum()

st02_unplanned_minutes = st02_unplanned[
    "duration_min"
].sum()

total_unplanned_minutes = downtime[
    downtime["downtime_class"] == "Unplanned"
]["duration_min"].sum()


sensor_summary = pd.DataFrame(
    [
        {
            "station_id": SECONDARY_STATION,
            "failure_type": SENSOR_FAILURE,
            "event_count": len(sensor_events),
            "total_downtime_min": sensor_total_minutes,
            "average_duration_min": sensor_events[
                "duration_min"
            ].mean(),
            "median_duration_min": sensor_events[
                "duration_min"
            ].median(),
            "max_duration_min": sensor_events[
                "duration_min"
            ].max(),
            "share_of_st02_unplanned":
                sensor_total_minutes / st02_unplanned_minutes,
            "share_of_total_unplanned":
                sensor_total_minutes / total_unplanned_minutes,
        }
    ]
)

sensor_summary.to_csv(
    ROOT_CAUSE_DIR / "st02_sensor_summary.csv",
    index=False,
)


sensor_by_shift = (
    sensor_events
    .groupby("shift", as_index=False)
    .agg(
        event_count=("downtime_id", "count"),
        downtime_minutes=("duration_min", "sum"),
        average_duration_min=("duration_min", "mean"),
    )
)

sensor_by_shift[
    "downtime_share"
] = (
    sensor_by_shift["downtime_minutes"]
    / sensor_total_minutes
)

sensor_by_shift.to_csv(
    ROOT_CAUSE_DIR / "st02_sensor_by_shift.csv",
    index=False,
)


sensor_by_product = (
    sensor_events
    .groupby("product_id", as_index=False)
    .agg(
        event_count=("downtime_id", "count"),
        downtime_minutes=("duration_min", "sum"),
        average_duration_min=("duration_min", "mean"),
    )
    .merge(
        product_lookup,
        on="product_id",
        how="left",
    )
)

sensor_by_product[
    "downtime_share"
] = (
    sensor_by_product["downtime_minutes"]
    / sensor_total_minutes
)

sensor_by_product.to_csv(
    ROOT_CAUSE_DIR / "st02_sensor_by_product.csv",
    index=False,
)


sensor_daily = (
    sensor_events
    .groupby("date", as_index=False)
    .agg(
        event_count=("downtime_id", "count"),
        downtime_minutes=("duration_min", "sum"),
    )
    .sort_values("date")
)

sensor_daily.to_csv(
    ROOT_CAUSE_DIR / "st02_sensor_daily.csv",
    index=False,
)


# Recoverable downtime scenarios
annualization_factor = (
    WORKING_DAYS_PER_YEAR
    / OBSERVATION_WORKING_DAYS
)

scenarios = []

for reduction in [0.20, 0.30, 0.50]:
    recovered_observation_min = (
        sensor_total_minutes * reduction
    )

    recovered_annual_hours = (
        recovered_observation_min
        * annualization_factor
        / 60
    )

    scenarios.append(
        {
            "scenario":
                f"{int(reduction * 100)}% sensor downtime reduction",
            "reduction_pct": reduction,
            "recovered_minutes_30_day_period":
                recovered_observation_min,
            "annualized_recovered_hours":
                recovered_annual_hours,
            "assumption":
                (
                    "Annualized from the simulated 30-working-day "
                    "observation period using 250 working days/year."
                ),
        }
    )

recovery_scenarios = pd.DataFrame(scenarios)

recovery_scenarios.to_csv(
    ROOT_CAUSE_DIR
    / "sensor_recovery_scenarios.csv",
    index=False,
)


# Evidence and hypotheses
worst_st04_shift = st04_shift.iloc[0]
slowest_st04_shift = st04_cycle_by_shift.iloc[0]
slowest_st04_product = st04_cycle_by_product.iloc[0]
top_st04_downtime = (
    st04_downtime_breakdown.iloc[0]
    if not st04_downtime_breakdown.empty
    else None
)

st04_five_why_rows = [
    {
        "level": 1,
        "why_question":
            "Why is line throughput constrained?",
        "answer":
            (
                f"{PRIMARY_STATION} is the line constraint in "
                f"{st04_constraint_share * 100:.2f}% of shifts."
            ),
        "evidence_status": "Supported by data",
    },
    {
        "level": 2,
        "why_question":
            "Why does ST04 produce less than the rest of the line?",
        "answer":
            (
                f"ST04 has the lowest station performance at "
                f"{st04_kpi.iloc[0]['performance'] * 100:.2f}% "
                f"and OEE of "
                f"{st04_kpi.iloc[0]['oee'] * 100:.2f}%."
            ),
        "evidence_status": "Supported by data",
    },
    {
        "level": 3,
        "why_question":
            "Where is the cycle-time loss most visible?",
        "answer":
            (
                f"Shift {slowest_st04_shift['shift']} has the highest "
                f"mean actual cycle time at "
                f"{slowest_st04_shift['mean_actual_cycle_sec']:.2f} sec."
            ),
        "evidence_status": "Supported by data",
    },
    {
        "level": 4,
        "why_question":
            "What operational mechanism could be increasing cycle time?",
        "answer":
            (
                "Possible contributors include manual work-content variation, "
                "functional-test duration, fixture handling, changeover, or "
                "material waiting."
            ),
        "evidence_status":
            "Hypothesis - requires process observation",
    },
    {
        "level": 5,
        "why_question":
            "What should be validated on the shop floor?",
        "answer":
            (
                "Perform a time study by work element, compare shifts, review "
                "test sequence duration, and observe fixture/material waiting."
            ),
        "evidence_status":
            "Validation action",
    },
]

st04_five_why = pd.DataFrame(
    st04_five_why_rows
)

st04_five_why.to_csv(
    ROOT_CAUSE_DIR / "st04_5why.csv",
    index=False,
)


sensor_event_count = int(
    sensor_summary.iloc[0]["event_count"]
)

sensor_avg_duration = (
    sensor_summary.iloc[0]["average_duration_min"]
)

sensor_share_st02 = (
    sensor_summary.iloc[0]["share_of_st02_unplanned"]
)

st02_five_why_rows = [
    {
        "level": 1,
        "why_question":
            "Why is ST02 availability lower than the other stations?",
        "answer":
            (
                "ST02 has the highest total downtime and the lowest "
                "availability among the five stations."
            ),
        "evidence_status": "Supported by data",
    },
    {
        "level": 2,
        "why_question":
            "Which failure contributes most to ST02 unplanned downtime?",
        "answer":
            (
                f"{SENSOR_FAILURE} caused "
                f"{sensor_total_minutes:.2f} minutes across "
                f"{sensor_event_count} events."
            ),
        "evidence_status": "Supported by data",
    },
    {
        "level": 3,
        "why_question":
            "Is the sensor problem driven by frequency, recovery time, or both?",
        "answer":
            (
                f"The average event duration is "
                f"{sensor_avg_duration:.2f} minutes and the failure accounts "
                f"for {sensor_share_st02 * 100:.2f}% of ST02 unplanned downtime."
            ),
        "evidence_status": "Supported by data",
    },
    {
        "level": 4,
        "why_question":
            "Why might the part-presence signal be failing?",
        "answer":
            (
                "Possible causes include sensor alignment, mounting stability, "
                "contamination, wiring/connectors, product geometry, or PLC "
                "signal handling."
            ),
        "evidence_status":
            "Hypothesis - requires physical validation",
    },
    {
        "level": 5,
        "why_question":
            "What should be validated before selecting a solution?",
        "answer":
            (
                "Inspect mounting/alignment, review fault codes and PLC input "
                "history, check wiring/connectors, compare product variants, "
                "and observe operator recovery steps."
            ),
        "evidence_status":
            "Validation action",
    },
]

st02_five_why = pd.DataFrame(
    st02_five_why_rows
)

st02_five_why.to_csv(
    ROOT_CAUSE_DIR / "st02_sensor_5why.csv",
    index=False,
)


ishikawa = pd.DataFrame(
    [
        {
            "problem": "ST04 throughput constraint",
            "category": "Machine",
            "candidate_cause":
                "Fixture handling or test equipment cycle delay",
            "status": "Hypothesis",
        },
        {
            "problem": "ST04 throughput constraint",
            "category": "Method",
            "candidate_cause":
                "Work sequence or standard-work imbalance",
            "status": "Hypothesis",
        },
        {
            "problem": "ST04 throughput constraint",
            "category": "Manpower",
            "candidate_cause":
                "Shift-dependent manual work variation",
            "status": "Hypothesis",
        },
        {
            "problem": "ST04 throughput constraint",
            "category": "Material",
            "candidate_cause":
                "Component waiting or replenishment delays",
            "status": "Hypothesis",
        },
        {
            "problem": "ST04 throughput constraint",
            "category": "Measurement",
            "candidate_cause":
                "Functional-test sequence duration or retest behavior",
            "status": "Hypothesis",
        },
        {
            "problem": "ST04 throughput constraint",
            "category": "Environment",
            "candidate_cause":
                "Shift-specific operating conditions",
            "status": "Hypothesis",
        },
        {
            "problem": "ST02 Part Presence Sensor downtime",
            "category": "Machine",
            "candidate_cause":
                "Sensor alignment, mounting, or component degradation",
            "status": "Hypothesis",
        },
        {
            "problem": "ST02 Part Presence Sensor downtime",
            "category": "Method",
            "candidate_cause":
                "Inconsistent reset and recovery procedure",
            "status": "Hypothesis",
        },
        {
            "problem": "ST02 Part Presence Sensor downtime",
            "category": "Manpower",
            "candidate_cause":
                "Different operator troubleshooting practices",
            "status": "Hypothesis",
        },
        {
            "problem": "ST02 Part Presence Sensor downtime",
            "category": "Material",
            "candidate_cause":
                "Product geometry or surface affecting detection",
            "status": "Hypothesis",
        },
        {
            "problem": "ST02 Part Presence Sensor downtime",
            "category": "Measurement",
            "candidate_cause":
                "PLC input filtering, threshold, or diagnostic visibility",
            "status": "Hypothesis",
        },
        {
            "problem": "ST02 Part Presence Sensor downtime",
            "category": "Environment",
            "candidate_cause":
                "Contamination, vibration, or ambient conditions",
            "status": "Hypothesis",
        },
    ]
)

ishikawa.to_csv(
    ROOT_CAUSE_DIR / "ishikawa_hypotheses.csv",
    index=False,
)


improvements = pd.DataFrame(
    [
        {
            "problem": "ST04 throughput constraint",
            "initiative":
                "Work-element time study and line balancing",
            "expected_mechanism":
                "Identify non-value-added time and rebalance manual work.",
            "impact": "High",
            "effort": "Low-Medium",
            "validation_needed":
                "Observe cycle by work element across all shifts.",
        },
        {
            "problem": "ST04 throughput constraint",
            "initiative":
                "Functional-test and fixture cycle optimization",
            "expected_mechanism":
                "Reduce fixed process time at the dominant constraint.",
            "impact": "High",
            "effort": "Medium",
            "validation_needed":
                "Measure fixture and test sub-cycle durations.",
        },
        {
            "problem": "ST04 throughput constraint",
            "initiative":
                "Shift-specific standard-work review",
            "expected_mechanism":
                "Reduce cycle variation between shifts.",
            "impact": "Medium",
            "effort": "Low",
            "validation_needed":
                f"Investigate Shift {slowest_st04_shift['shift']} first.",
        },
        {
            "problem": "ST02 Part Presence Sensor downtime",
            "initiative":
                "Sensor inspection and reliability action plan",
            "expected_mechanism":
                "Reduce recurrence of the dominant sensor failure.",
            "impact": "High",
            "effort": "Low-Medium",
            "validation_needed":
                "Inspect alignment, mounting, contamination, wiring and wear.",
        },
        {
            "problem": "ST02 Part Presence Sensor downtime",
            "initiative":
                "Standard robot/sensor recovery procedure",
            "expected_mechanism":
                "Reduce fault recovery duration and escalation time.",
            "impact": "Medium",
            "effort": "Low",
            "validation_needed":
                "Observe current recovery steps by shift/operator.",
        },
        {
            "problem": "ST02 Part Presence Sensor downtime",
            "initiative":
                "PLC alarm and diagnostic improvement",
            "expected_mechanism":
                "Make the fault source and recovery action visible faster.",
            "impact": "Medium-High",
            "effort": "Medium",
            "validation_needed":
                "Review PLC input state, alarm history and fault-code detail.",
        },
    ]
)

improvements.to_csv(
    ROOT_CAUSE_DIR / "improvement_opportunities.csv",
    index=False,
)


# Charts
plt.figure(figsize=(8, 5))
plt.bar(
    st04_cycle_by_shift["shift"],
    st04_cycle_by_shift["mean_actual_cycle_sec"],
)
plt.ylabel("Mean Actual Cycle Time (sec)")
plt.xlabel("Shift")
plt.title("ST04 Mean Actual Cycle Time by Shift")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "01_st04_cycle_by_shift.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(9, 5))
plt.bar(
    st04_cycle_by_product["product_name"],
    st04_cycle_by_product["mean_actual_cycle_sec"],
)
plt.ylabel("Mean Actual Cycle Time (sec)")
plt.title("ST04 Mean Actual Cycle Time by Product")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "02_st04_cycle_by_product.png",
    dpi=200,
)
plt.close()


top_st04_losses = st04_downtime_breakdown.head(8)

plt.figure(figsize=(10, 5))
plt.bar(
    top_st04_losses["failure_type"],
    top_st04_losses["downtime_minutes"],
)
plt.ylabel("Downtime (min)")
plt.title("ST04 Unplanned Downtime Causes")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "03_st04_downtime_causes.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(7, 5))
plt.bar(
    sensor_by_shift["shift"],
    sensor_by_shift["downtime_minutes"],
)
plt.ylabel("Downtime (min)")
plt.xlabel("Shift")
plt.title("ST02 Part Presence Sensor Downtime by Shift")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "04_st02_sensor_by_shift.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(9, 5))
plt.bar(
    sensor_by_product["product_name"],
    sensor_by_product["downtime_minutes"],
)
plt.ylabel("Downtime (min)")
plt.title("ST02 Part Presence Sensor Downtime by Product")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "05_st02_sensor_by_product.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(
    sensor_daily["date"],
    sensor_daily["downtime_minutes"],
    marker="o",
)
plt.ylabel("Downtime (min)")
plt.xlabel("Date")
plt.title("ST02 Part Presence Sensor Downtime Trend")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(
    CHART_DIR / "06_st02_sensor_daily_trend.png",
    dpi=200,
)
plt.close()


print("\n--------------------------------")
print("DAY 6 ROOT-CAUSE SUMMARY")
print("--------------------------------")
print(
    f"ST04 constraint share: "
    f"{st04_constraint_share * 100:.2f}% of shifts"
)
print(
    f"ST04 OEE: "
    f"{st04_kpi.iloc[0]['oee'] * 100:.2f}%"
)
print(
    f"ST04 performance: "
    f"{st04_kpi.iloc[0]['performance'] * 100:.2f}%"
)
print(
    f"Slowest ST04 shift by mean cycle time: "
    f"{slowest_st04_shift['shift']} "
    f"({slowest_st04_shift['mean_actual_cycle_sec']:.2f} sec)"
)
print(
    f"Slowest ST04 product by mean cycle time: "
    f"{slowest_st04_product['product_id']} - "
    f"{slowest_st04_product['product_name']} "
    f"({slowest_st04_product['mean_actual_cycle_sec']:.2f} sec)"
)

if top_st04_downtime is not None:
    print(
        f"Top ST04 unplanned downtime cause: "
        f"{top_st04_downtime['failure_type']} "
        f"({top_st04_downtime['downtime_minutes']:.2f} min)"
    )

print(
    f"ST02 Part Presence Sensor events: "
    f"{sensor_event_count}"
)
print(
    f"ST02 Part Presence Sensor downtime: "
    f"{sensor_total_minutes:.2f} min"
)
print(
    f"Share of ST02 unplanned downtime: "
    f"{sensor_share_st02 * 100:.2f}%"
)
print(
    f"Average Part Presence Sensor event: "
    f"{sensor_avg_duration:.2f} min"
)

print("\n--------------------------------")
print("DAY 6 ROOT-CAUSE ANALYSIS COMPLETE")
print("--------------------------------")
print(f"Root-cause tables: {ROOT_CAUSE_DIR}")
print(f"Charts: {CHART_DIR}")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
KPI_DIR = PROCESSED_DIR / "kpis"
BUSINESS_CASE_DIR = PROJECT_ROOT / "business_case"
CHART_DIR = PROJECT_ROOT / "screenshots" / "day7"

BUSINESS_CASE_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_STATION = "ST04"
SECONDARY_STATION = "ST02"
SENSOR_FAILURE = "Part Presence Sensor"

OBSERVATION_WORKING_DAYS = 30
ANNUAL_WORKING_DAYS = 250
ANNUALIZATION_FACTOR = (
    ANNUAL_WORKING_DAYS
    / OBSERVATION_WORKING_DAYS
)

VALUE_PER_ADDITIONAL_GOOD_UNIT_EUR = 20.0
CAPACITY_REALIZATION_FACTOR = 0.50
YEAR1_RAMP_FACTOR = 0.75
MAINTENANCE_SUPPORT_COST_PER_HOUR_EUR = 60.0


production = pd.read_csv(
    PROCESSED_DIR / "production_clean.csv",
    parse_dates=["timestamp", "date"],
)

downtime = pd.read_csv(
    PROCESSED_DIR / "downtime_clean.csv",
    parse_dates=["start_time", "end_time", "date"],
)

target_attainment = pd.read_csv(
    KPI_DIR / "target_attainment.csv",
    parse_dates=["date"],
)

machines = pd.read_csv(
    PROCESSED_DIR / "machine_master.csv"
)


def safe_divide(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def calculate_roi(annual_benefit, implementation_cost):
    if implementation_cost <= 0:
        return np.nan

    return (
        (annual_benefit - implementation_cost)
        / implementation_cost
    )


def calculate_payback_months(
    annual_benefit,
    implementation_cost,
):
    if annual_benefit <= 0:
        return np.nan

    return (
        implementation_cost
        / annual_benefit
        * 12
    )


financial_assumptions = pd.DataFrame(
    [
        {
            "assumption":
                "Observation working days",
            "value":
                OBSERVATION_WORKING_DAYS,
            "unit":
                "days",
            "basis":
                "Synthetic project design",
        },
        {
            "assumption":
                "Annual working days",
            "value":
                ANNUAL_WORKING_DAYS,
            "unit":
                "days/year",
            "basis":
                "Simulated business-case assumption",
        },
        {
            "assumption":
                "Economic value per additional good actuator",
            "value":
                VALUE_PER_ADDITIONAL_GOOD_UNIT_EUR,
            "unit":
                "EUR/unit",
            "basis":
                "Simulated contribution-value assumption",
        },
        {
            "assumption":
                "Capacity realization factor",
            "value":
                CAPACITY_REALIZATION_FACTOR,
            "unit":
                "share",
            "basis":
                "Simulated assumption: only part of theoretical capacity becomes sellable output",
        },
        {
            "assumption":
                "Year-1 ramp factor",
            "value":
                YEAR1_RAMP_FACTOR,
            "unit":
                "share",
            "basis":
                "Simulated implementation ramp-up assumption",
        },
        {
            "assumption":
                "Maintenance/support value per recovered hour",
            "value":
                MAINTENANCE_SUPPORT_COST_PER_HOUR_EUR,
            "unit":
                "EUR/hour",
            "basis":
                "Simulated labor/support-cost assumption",
        },
    ]
)

financial_assumptions.to_csv(
    BUSINESS_CASE_DIR / "financial_assumptions.csv",
    index=False,
)


station_shift = (
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
        runtime_minutes=(
            "runtime_minutes",
            "sum",
        ),
        planned_minutes=(
            "planned_minutes",
            "sum",
        ),
        weighted_cycle_numerator=(
            "actual_cycle_sec",
            lambda x: 0.0,
        ),
    )
)

cycle_weighted = (
    production
    .assign(
        cycle_weighted_value=(
            production["actual_cycle_sec"]
            * production["total_quantity"]
        )
    )
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
        cycle_weighted_sum=(
            "cycle_weighted_value",
            "sum",
        ),
        cycle_weight_units=(
            "total_quantity",
            "sum",
        ),
        mean_ideal_cycle_sec=(
            "ideal_cycle_sec",
            "mean",
        ),
    )
)

station_shift = station_shift.drop(
    columns=["weighted_cycle_numerator"]
).merge(
    cycle_weighted,
    on=[
        "date",
        "shift",
        "product_id",
        "station_id",
    ],
    how="left",
    validate="one_to_one",
)

station_shift[
    "weighted_actual_cycle_sec"
] = np.where(
    station_shift["cycle_weight_units"] > 0,
    station_shift["cycle_weighted_sum"]
    / station_shift["cycle_weight_units"],
    station_shift["mean_ideal_cycle_sec"],
)

station_shift[
    "quality_yield"
] = np.where(
    station_shift["total_quantity"] > 0,
    station_shift["good_quantity"]
    / station_shift["total_quantity"],
    1.0,
)


def current_line_output_table():
    rows = []

    for (
        date,
        shift,
        product_id,
    ), group in station_shift.groupby(
        ["date", "shift", "product_id"]
    ):
        ordered = group.sort_values(
            "good_quantity"
        )

        current_line_output = (
            ordered["good_quantity"].iloc[0]
        )

        second_lowest_output = (
            ordered["good_quantity"].iloc[1]
            if len(ordered) > 1
            else current_line_output
        )

        constraint_station = (
            ordered["station_id"].iloc[0]
        )

        rows.append(
            {
                "date": date,
                "shift": shift,
                "product_id": product_id,
                "constraint_station_id":
                    constraint_station,
                "current_line_output":
                    current_line_output,
                "second_lowest_station_output":
                    second_lowest_output,
                "constraint_gap_units":
                    max(
                        second_lowest_output
                        - current_line_output,
                        0,
                    ),
            }
        )

    return pd.DataFrame(rows)


baseline_line = current_line_output_table()

st04_constraint_gap = baseline_line[
    baseline_line["constraint_station_id"]
    == PRIMARY_STATION
]["constraint_gap_units"].sum()

st04_annual_opportunity_units = (
    st04_constraint_gap
    * ANNUALIZATION_FACTOR
)

st04_annual_opportunity_value = (
    st04_annual_opportunity_units
    * CAPACITY_REALIZATION_FACTOR
    * YEAR1_RAMP_FACTOR
    * VALUE_PER_ADDITIONAL_GOOD_UNIT_EUR
)


def simulate_st04_cycle_reduction(
    cycle_reduction_pct,
):
    scenario_rows = []

    for (
        date,
        shift,
        product_id,
    ), group in station_shift.groupby(
        ["date", "shift", "product_id"]
    ):
        outputs = group[
            [
                "station_id",
                "good_quantity",
            ]
        ].copy()

        outputs["good_quantity"] = (
            outputs["good_quantity"].astype(float)
        )

        st04_mask = (
            outputs["station_id"]
            == PRIMARY_STATION
        )

        outputs.loc[
            st04_mask,
            "good_quantity",
        ] = (
            outputs.loc[
                st04_mask,
                "good_quantity",
            ]
            / (1 - cycle_reduction_pct)
        )

        new_line_output = (
            outputs["good_quantity"].min()
        )

        current_output = baseline_line.loc[
            (
                (baseline_line["date"] == date)
                & (baseline_line["shift"] == shift)
                & (
                    baseline_line["product_id"]
                    == product_id
                )
            ),
            "current_line_output",
        ].iloc[0]

        scenario_rows.append(
            {
                "date": date,
                "shift": shift,
                "product_id": product_id,
                "current_line_output":
                    current_output,
                "scenario_line_output":
                    new_line_output,
                "additional_good_units":
                    max(
                        new_line_output
                        - current_output,
                        0,
                    ),
            }
        )

    return pd.DataFrame(scenario_rows)


st04_scenario_inputs = [
    {
        "scenario":
            "Work-element study + standard work",
        "cycle_reduction_pct":
            0.03,
        "implementation_cost_eur":
            15000.0,
        "implementation_horizon":
            "0-3 months",
        "confidence":
            "Medium-High",
    },
    {
        "scenario":
            "Functional-test / fixture optimization",
        "cycle_reduction_pct":
            0.05,
        "implementation_cost_eur":
            35000.0,
        "implementation_horizon":
            "3-6 months",
        "confidence":
            "Medium",
    },
    {
        "scenario":
            "Combined ST04 optimization package",
        "cycle_reduction_pct":
            0.08,
        "implementation_cost_eur":
            60000.0,
        "implementation_horizon":
            "3-9 months",
        "confidence":
            "Medium",
    },
]

st04_business_rows = []

for scenario in st04_scenario_inputs:
    detail = simulate_st04_cycle_reduction(
        scenario["cycle_reduction_pct"]
    )

    additional_units_30d = (
        detail["additional_good_units"].sum()
    )

    annual_additional_units = (
        additional_units_30d
        * ANNUALIZATION_FACTOR
    )

    realized_annual_units = (
        annual_additional_units
        * CAPACITY_REALIZATION_FACTOR
        * YEAR1_RAMP_FACTOR
    )

    annual_benefit = (
        realized_annual_units
        * VALUE_PER_ADDITIONAL_GOOD_UNIT_EUR
    )

    cost = scenario[
        "implementation_cost_eur"
    ]

    st04_business_rows.append(
        {
            **scenario,
            "additional_good_units_30d":
                additional_units_30d,
            "annual_additional_good_units":
                annual_additional_units,
            "realized_year1_good_units":
                realized_annual_units,
            "annual_financial_benefit_eur":
                annual_benefit,
            "year1_roi":
                calculate_roi(
                    annual_benefit,
                    cost,
                ),
            "payback_months":
                calculate_payback_months(
                    annual_benefit,
                    cost,
                ),
            "current_annual_constraint_opportunity_eur":
                st04_annual_opportunity_value,
            "financial_basis":
                (
                    "Additional line output is capped by the "
                    "next-lowest station output. Unit value is "
                    "a simulated assumption."
                ),
        }
    )

st04_business_case = pd.DataFrame(
    st04_business_rows
)

st04_business_case.to_csv(
    BUSINESS_CASE_DIR
    / "st04_business_case.csv",
    index=False,
)


sensor_events = downtime[
    (
        downtime["station_id"]
        == SECONDARY_STATION
    )
    & (
        downtime["failure_type"]
        == SENSOR_FAILURE
    )
    & (
        downtime["downtime_class"]
        == "Unplanned"
    )
].copy()

sensor_by_shift = (
    sensor_events
    .groupby(
        ["date", "shift", "product_id"],
        as_index=False,
    )
    .agg(
        sensor_downtime_min=(
            "duration_min",
            "sum",
        ),
        sensor_events=(
            "downtime_id",
            "count",
        ),
    )
)


def simulate_st02_sensor_reduction(
    downtime_reduction_pct,
):
    scenario = station_shift.copy()

    scenario = scenario.merge(
        sensor_by_shift,
        on=[
            "date",
            "shift",
            "product_id",
        ],
        how="left",
    )

    scenario[
        "sensor_downtime_min"
    ] = (
        scenario[
            "sensor_downtime_min"
        ].fillna(0)
    )

    scenario[
        "sensor_events"
    ] = (
        scenario[
            "sensor_events"
        ].fillna(0)
    )

    scenario[
        "scenario_good_quantity"
    ] = scenario["good_quantity"].astype(float)

    st02_mask = (
        scenario["station_id"]
        == SECONDARY_STATION
    )

    recovered_minutes = (
        scenario.loc[
            st02_mask,
            "sensor_downtime_min",
        ]
        * downtime_reduction_pct
    )

    extra_total_units = (
        recovered_minutes
        * 60
        / scenario.loc[
            st02_mask,
            "weighted_actual_cycle_sec",
        ]
    )

    extra_good_units = (
        extra_total_units
        * scenario.loc[
            st02_mask,
            "quality_yield",
        ]
    )

    scenario.loc[
        st02_mask,
        "scenario_good_quantity",
    ] = (
        scenario.loc[
            st02_mask,
            "good_quantity",
        ]
        + extra_good_units
    )

    rows = []

    for (
        date,
        shift,
        product_id,
    ), group in scenario.groupby(
        ["date", "shift", "product_id"]
    ):
        current_output = baseline_line.loc[
            (
                (baseline_line["date"] == date)
                & (baseline_line["shift"] == shift)
                & (
                    baseline_line["product_id"]
                    == product_id
                )
            ),
            "current_line_output",
        ].iloc[0]

        new_line_output = (
            group[
                "scenario_good_quantity"
            ].min()
        )

        rows.append(
            {
                "date": date,
                "shift": shift,
                "product_id": product_id,
                "current_line_output":
                    current_output,
                "scenario_line_output":
                    new_line_output,
                "additional_good_units":
                    max(
                        new_line_output
                        - current_output,
                        0,
                    ),
            }
        )

    detail = pd.DataFrame(rows)

    recovered_sensor_minutes_30d = (
        sensor_events["duration_min"].sum()
        * downtime_reduction_pct
    )

    return (
        detail,
        recovered_sensor_minutes_30d,
    )


st02_scenario_inputs = [
    {
        "scenario":
            "Basic sensor inspection + standard recovery",
        "downtime_reduction_pct":
            0.20,
        "implementation_cost_eur":
            3000.0,
        "implementation_horizon":
            "0-2 months",
        "confidence":
            "High",
    },
    {
        "scenario":
            "Sensor reliability action plan",
        "downtime_reduction_pct":
            0.30,
        "implementation_cost_eur":
            5500.0,
        "implementation_horizon":
            "0-3 months",
        "confidence":
            "Medium-High",
    },
    {
        "scenario":
            "Sensor + PLC diagnostics improvement",
        "downtime_reduction_pct":
            0.50,
        "implementation_cost_eur":
            9000.0,
        "implementation_horizon":
            "2-6 months",
        "confidence":
            "Medium",
    },
]

st02_business_rows = []

for scenario in st02_scenario_inputs:
    (
        detail,
        recovered_minutes_30d,
    ) = simulate_st02_sensor_reduction(
        scenario["downtime_reduction_pct"]
    )

    additional_units_30d = (
        detail["additional_good_units"].sum()
    )

    annual_additional_units = (
        additional_units_30d
        * ANNUALIZATION_FACTOR
    )

    annual_recovered_hours = (
        recovered_minutes_30d
        * ANNUALIZATION_FACTOR
        / 60
    )

    realized_annual_units = (
        annual_additional_units
        * CAPACITY_REALIZATION_FACTOR
        * YEAR1_RAMP_FACTOR
    )

    annual_output_value = (
        realized_annual_units
        * VALUE_PER_ADDITIONAL_GOOD_UNIT_EUR
    )

    annual_support_value = (
        annual_recovered_hours
        * MAINTENANCE_SUPPORT_COST_PER_HOUR_EUR
    )

    annual_benefit = (
        annual_output_value
        + annual_support_value
    )

    cost = scenario[
        "implementation_cost_eur"
    ]

    st02_business_rows.append(
        {
            **scenario,
            "recovered_sensor_minutes_30d":
                recovered_minutes_30d,
            "annual_recovered_sensor_hours":
                annual_recovered_hours,
            "additional_good_units_30d":
                additional_units_30d,
            "annual_additional_good_units":
                annual_additional_units,
            "realized_year1_good_units":
                realized_annual_units,
            "annual_output_value_eur":
                annual_output_value,
            "annual_support_value_eur":
                annual_support_value,
            "annual_financial_benefit_eur":
                annual_benefit,
            "year1_roi":
                calculate_roi(
                    annual_benefit,
                    cost,
                ),
            "payback_months":
                calculate_payback_months(
                    annual_benefit,
                    cost,
                ),
            "financial_basis":
                (
                    "Recovered ST02 time increases finished-goods "
                    "output only when ST02 becomes the active line "
                    "constraint. Support value uses a simulated "
                    "EUR/hour assumption."
                ),
        }
    )

st02_business_case = pd.DataFrame(
    st02_business_rows
)

st02_business_case.to_csv(
    BUSINESS_CASE_DIR
    / "st02_sensor_business_case.csv",
    index=False,
)


priority_rows = []

for row in st04_business_case.itertuples():
    priority_rows.append(
        {
            "problem":
                "ST04 throughput constraint",
            "initiative":
                row.scenario,
            "annual_benefit_eur":
                row.annual_financial_benefit_eur,
            "implementation_cost_eur":
                row.implementation_cost_eur,
            "payback_months":
                row.payback_months,
            "confidence":
                row.confidence,
            "implementation_horizon":
                row.implementation_horizon,
        }
    )

for row in st02_business_case.itertuples():
    priority_rows.append(
        {
            "problem":
                "ST02 Part Presence Sensor downtime",
            "initiative":
                row.scenario,
            "annual_benefit_eur":
                row.annual_financial_benefit_eur,
            "implementation_cost_eur":
                row.implementation_cost_eur,
            "payback_months":
                row.payback_months,
            "confidence":
                row.confidence,
            "implementation_horizon":
                row.implementation_horizon,
        }
    )

prioritization = pd.DataFrame(
    priority_rows
)

confidence_score_map = {
    "High": 5,
    "Medium-High": 4,
    "Medium": 3,
    "Medium-Low": 2,
    "Low": 1,
}

prioritization[
    "confidence_score"
] = prioritization[
    "confidence"
].map(confidence_score_map)

max_benefit = prioritization[
    "annual_benefit_eur"
].max()

prioritization[
    "benefit_score"
] = np.where(
    max_benefit > 0,
    (
        prioritization[
            "annual_benefit_eur"
        ]
        / max_benefit
        * 5
    ),
    0,
)

prioritization[
    "payback_score"
] = np.select(
    [
        prioritization["payback_months"] <= 6,
        prioritization["payback_months"] <= 12,
        prioritization["payback_months"] <= 24,
    ],
    [
        5,
        4,
        3,
    ],
    default=2,
)

max_cost = prioritization[
    "implementation_cost_eur"
].max()

prioritization[
    "cost_score"
] = np.where(
    max_cost > 0,
    (
        1
        - (
            prioritization[
                "implementation_cost_eur"
            ]
            / max_cost
        )
    )
    * 4
    + 1,
    1,
)

prioritization[
    "priority_score"
] = (
    prioritization[
        "benefit_score"
    ] * 0.40
    + prioritization[
        "payback_score"
    ] * 0.25
    + prioritization[
        "confidence_score"
    ] * 0.20
    + prioritization[
        "cost_score"
    ] * 0.15
)

prioritization = prioritization.sort_values(
    "priority_score",
    ascending=False,
).reset_index(drop=True)

prioritization[
    "priority_rank"
] = (
    prioritization.index + 1
)

prioritization.to_csv(
    BUSINESS_CASE_DIR
    / "improvement_prioritization.csv",
    index=False,
)


best_st04 = st04_business_case.loc[
    st04_business_case[
        "year1_roi"
    ].idxmax()
]

best_st02 = st02_business_case.loc[
    st02_business_case[
        "year1_roi"
    ].idxmax()
]

executive_business_case = pd.DataFrame(
    [
        {
            "case":
                "ST04 throughput improvement",
            "recommended_scenario":
                best_st04["scenario"],
            "implementation_cost_eur":
                best_st04[
                    "implementation_cost_eur"
                ],
            "annual_benefit_eur":
                best_st04[
                    "annual_financial_benefit_eur"
                ],
            "year1_roi":
                best_st04["year1_roi"],
            "payback_months":
                best_st04["payback_months"],
            "annual_additional_good_units":
                best_st04[
                    "annual_additional_good_units"
                ],
            "main_business_reason":
                (
                    "ST04 constrains line output in nearly all "
                    "observed shifts, so cycle-time reduction can "
                    "translate directly into finished-goods capacity."
                ),
        },
        {
            "case":
                "ST02 sensor reliability improvement",
            "recommended_scenario":
                best_st02["scenario"],
            "implementation_cost_eur":
                best_st02[
                    "implementation_cost_eur"
                ],
            "annual_benefit_eur":
                best_st02[
                    "annual_financial_benefit_eur"
                ],
            "year1_roi":
                best_st02["year1_roi"],
            "payback_months":
                best_st02["payback_months"],
            "annual_additional_good_units":
                best_st02[
                    "annual_additional_good_units"
                ],
            "main_business_reason":
                (
                    "The Part Presence Sensor is a concentrated "
                    "reliability loss, but throughput benefit is "
                    "limited unless ST02 becomes the active line "
                    "constraint."
                ),
        },
    ]
)

executive_business_case.to_csv(
    BUSINESS_CASE_DIR
    / "executive_business_case.csv",
    index=False,
)


# Charts
plt.figure(figsize=(10, 6))
plt.scatter(
    prioritization[
        "implementation_cost_eur"
    ],
    prioritization[
        "annual_benefit_eur"
    ],
    s=90,
)

for row in prioritization.itertuples():
    plt.annotate(
        f"{row.priority_rank}. {row.initiative}",
        (
            row.implementation_cost_eur,
            row.annual_benefit_eur,
        ),
        fontsize=8,
    )

plt.xlabel("Implementation Cost (EUR)")
plt.ylabel("Estimated Annual Benefit (EUR)")
plt.title("Improvement Prioritization: Cost vs Annual Benefit")
plt.tight_layout()
plt.savefig(
    CHART_DIR
    / "01_improvement_prioritization.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(9, 5))
plt.bar(
    st04_business_case["scenario"],
    st04_business_case[
        "annual_financial_benefit_eur"
    ],
)
plt.ylabel("Estimated Annual Benefit (EUR)")
plt.title("ST04 Improvement Scenarios")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    CHART_DIR
    / "02_st04_business_case.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(9, 5))
plt.bar(
    st02_business_case["scenario"],
    st02_business_case[
        "annual_financial_benefit_eur"
    ],
)
plt.ylabel("Estimated Annual Benefit (EUR)")
plt.title("ST02 Sensor Improvement Scenarios")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    CHART_DIR
    / "03_st02_business_case.png",
    dpi=200,
)
plt.close()


print("\n--------------------------------")
print("DAY 7 BUSINESS CASE SUMMARY")
print("--------------------------------")
print(
    f"Simulated contribution value per realized good unit: "
    f"EUR {VALUE_PER_ADDITIONAL_GOOD_UNIT_EUR:.2f}"
)
print(
    f"Capacity realization factor: "
    f"{CAPACITY_REALIZATION_FACTOR * 100:.0f}%"
)
print(
    f"Year-1 ramp factor: "
    f"{YEAR1_RAMP_FACTOR * 100:.0f}%"
)
print(
    f"Estimated annual ST04 constraint opportunity: "
    f"EUR {st04_annual_opportunity_value:,.2f}"
)

print("\nBest ST04 scenario:")
print(
    f"  {best_st04['scenario']}"
)
print(
    f"  Annual additional good units: "
    f"{best_st04['annual_additional_good_units']:,.1f}"
)
print(
    f"  Annual benefit: "
    f"EUR {best_st04['annual_financial_benefit_eur']:,.2f}"
)
print(
    f"  Implementation cost: "
    f"EUR {best_st04['implementation_cost_eur']:,.2f}"
)
print(
    f"  Year-1 ROI: "
    f"{best_st04['year1_roi'] * 100:.2f}%"
)
print(
    f"  Payback: "
    f"{best_st04['payback_months']:.2f} months"
)

print("\nBest ST02 scenario:")
print(
    f"  {best_st02['scenario']}"
)
print(
    f"  Annual recovered sensor hours: "
    f"{best_st02['annual_recovered_sensor_hours']:,.1f}"
)
print(
    f"  Annual additional good units: "
    f"{best_st02['annual_additional_good_units']:,.1f}"
)
print(
    f"  Annual benefit: "
    f"EUR {best_st02['annual_financial_benefit_eur']:,.2f}"
)
print(
    f"  Implementation cost: "
    f"EUR {best_st02['implementation_cost_eur']:,.2f}"
)
print(
    f"  Year-1 ROI: "
    f"{best_st02['year1_roi'] * 100:.2f}%"
)
print(
    f"  Payback: "
    f"{best_st02['payback_months']:.2f} months"
)

print("\nTop prioritized initiative:")
top_priority = prioritization.iloc[0]
print(
    f"  Rank 1: {top_priority['initiative']}"
)
print(
    f"  Problem: {top_priority['problem']}"
)
print(
    f"  Priority score: "
    f"{top_priority['priority_score']:.2f} / 5"
)

print("\n--------------------------------")
print("DAY 7 BUSINESS CASE COMPLETE")
print("--------------------------------")
print(f"Business-case files: {BUSINESS_CASE_DIR}")
print(f"Charts: {CHART_DIR}")

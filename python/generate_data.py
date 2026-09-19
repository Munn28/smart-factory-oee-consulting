from pathlib import Path
import numpy as np
import pandas as pd

# Project configuration

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

START_DATE = pd.Timestamp("2026-01-05")

PRODUCTION_DAYS = 30

BLOCK_MINUTES = 15
PLANNED_MINUTES_PER_SHIFT = 450
BLOCKS_PER_SHIFT = PLANNED_MINUTES_PER_SHIFT // BLOCK_MINUTES

machines = pd.DataFrame(
    [
        {
            "station_id": "ST01",
            "station_name": "CNC Machining",
            "station_type": "CNC",
            "ideal_cycle_sec": 66,
            "base_scrap_rate": 0.018,
        },
        {
            "station_id": "ST02",
            "station_name": "Robot Handling",
            "station_type": "Industrial Robot",
            "ideal_cycle_sec": 60,
            "base_scrap_rate": 0.008,
        },
        {
            "station_id": "ST03",
            "station_name": "Quality Inspection",
            "station_type": "Inspection",
            "ideal_cycle_sec": 63,
            "base_scrap_rate": 0.025,
        },
        {
            "station_id": "ST04",
            "station_name": "Assembly & Testing",
            "station_type": "Assembly",
            "ideal_cycle_sec": 72,
            "base_scrap_rate": 0.022,
        },
        {
            "station_id": "ST05",
            "station_name": "Packaging",
            "station_type": "Packaging",
            "ideal_cycle_sec": 64,
            "base_scrap_rate": 0.006,
        },
    ]
)

machines.to_csv(RAW_DATA_DIR / "machine_master.csv", index=False)

products = pd.DataFrame(
    [
        {
            "product_id": "ACT-100",
            "product_name": "Standard Actuator",
            "production_mix": 0.50,
            "cycle_factor": 0.95,
            "quality_factor": 0.90,
        },
        {
            "product_id": "ACT-200",
            "product_name": "Medium-Duty Actuator",
            "production_mix": 0.30,
            "cycle_factor": 1.00,
            "quality_factor": 1.00,
        },
        {
            "product_id": "ACT-300",
            "product_name": "Heavy-Duty Actuator",
            "production_mix": 0.20,
            "cycle_factor": 1.10,
            "quality_factor": 1.25,
        },
    ]
)

products.to_csv(RAW_DATA_DIR / "product_master.csv", index=False)

SHIFTS = {
    "A": {
        "start_hour": 6,
        "speed_factor": 1.00,
        "quality_factor": 1.00,
        "failure_factor": 1.00,
        "repair_factor": 1.00,
    },
    "B": {
        "start_hour": 14,
        "speed_factor": 1.02,
        "quality_factor": 1.05,
        "failure_factor": 1.05,
        "repair_factor": 1.05,
    },
    "C": {
        "start_hour": 22,
        "speed_factor": 1.06,
        "quality_factor": 1.12,
        "failure_factor": 1.15,
        "repair_factor": 1.15,
    },
}

def generate_working_days(start_date, number_of_days):
    dates = []
    current_date = start_date

    while len(dates) < number_of_days:

        if current_date.weekday() < 5:
            dates.append(current_date)

        current_date += pd.Timedelta(days=1)

    return dates

production_dates = generate_working_days(
    START_DATE,
    PRODUCTION_DAYS
)

def choose_product():
    return np.random.choice(
        products["product_id"],
        p=products["production_mix"]
    )

FAILURE_LIBRARY = {

    "ST01": [
        ("Equipment Failure", "Spindle Fault", 0.25),
        ("Tooling", "Tool Wear", 0.30),
        ("Sensor/Control", "Coolant Sensor Alarm", 0.15),
        ("Tooling", "Tool Change", 0.20),
        ("Micro Stop", "Minor CNC Stop", 0.10),
    ],

    "ST02": [
        ("Equipment Failure", "Robot Position Fault", 0.30),
        ("Equipment Failure", "Gripper Fault", 0.20),
        ("Sensor/Control", "Part Presence Sensor", 0.22),
        ("Sensor/Control", "Communication Fault", 0.13),
        ("Micro Stop", "Robot Micro Stop", 0.15),
    ],

    "ST03": [
        ("Sensor/Control", "Vision System Fault", 0.30),
        ("Sensor/Control", "Measurement Sensor Fault", 0.25),
        ("Quality", "Inspection Calibration", 0.20),
        ("Micro Stop", "Inspection Micro Stop", 0.25),
    ],

    "ST04": [
        ("Material", "Component Shortage", 0.25),
        ("Equipment Failure", "Fixture Fault", 0.20),
        ("Quality", "Functional Test Failure", 0.20),
        ("Changeover", "Product Changeover", 0.20),
        ("Micro Stop", "Assembly Micro Stop", 0.15),
    ],

    "ST05": [
        ("Equipment Failure", "Packaging Jam", 0.25),
        ("Sensor/Control", "Label Sensor Fault", 0.25),
        ("Material", "Packaging Material Shortage", 0.20),
        ("Micro Stop", "Packaging Micro Stop", 0.30),
    ],
}

BASE_FAILURE_PROBABILITY = {
    "ST01": 0.035,
    "ST02": 0.070,
    "ST03": 0.040,
    "ST04": 0.045,
    "ST05": 0.025,
}

def generate_failure(station_id, shift_name):

    failure_options = FAILURE_LIBRARY[station_id]

    probabilities = [
        failure[2]
        for failure in failure_options
    ]

    index = np.random.choice(
        len(failure_options),
        p=probabilities
    )

    category, failure_type, _ = failure_options[index]

    repair_factor = SHIFTS[shift_name]["repair_factor"]

    if category == "Micro Stop":
        duration = np.random.uniform(1, 4)

    elif category == "Changeover":
        duration = np.random.uniform(10, 30)

    elif category == "Tooling":
        duration = np.random.uniform(5, 20)

    elif category == "Material":
        duration = np.random.uniform(5, 25)

    else:
        duration = np.random.uniform(5, 35)

    duration *= repair_factor

    return category, failure_type, round(duration, 2)

def planned_maintenance_occurs(station_id):

    probabilities = {
        "ST01": 0.008,
        "ST02": 0.005,
        "ST03": 0.005,
        "ST04": 0.006,
        "ST05": 0.004,
    }

    return np.random.random() < probabilities[station_id]

production_records = []
downtime_records = []
maintenance_records = []

production_record_counter = 1
downtime_counter = 1
maintenance_counter = 1

for production_date in production_dates:

    for shift_name, shift_config in SHIFTS.items():

        shift_start = production_date + pd.Timedelta(
            hours=shift_config["start_hour"]
        )

        # Shift C finishes next day naturally because timestamps
        # continue beyond midnight.

        shift_product = choose_product()

        product_row = products.loc[
            products["product_id"] == shift_product
        ].iloc[0]

        product_cycle_factor = product_row["cycle_factor"]
        product_quality_factor = product_row["quality_factor"]

        for station in machines.itertuples():

            station_id = station.station_id

            base_cycle = station.ideal_cycle_sec

            for block in range(BLOCKS_PER_SHIFT):

                timestamp = shift_start + pd.Timedelta(
                    minutes=block * BLOCK_MINUTES
                )

                planned_minutes = BLOCK_MINUTES

                downtime_minutes = 0.0

                machine_status = "Running"

                failure_category = None
                failure_type = None

                current_downtime_id = None

                # ----------------------------------------
                # UNPLANNED FAILURE
                # ----------------------------------------

                failure_probability = (
                    BASE_FAILURE_PROBABILITY[station_id]
                    * shift_config["failure_factor"]
                )

                if np.random.random() < failure_probability:

                    (
                        failure_category,
                        failure_type,
                        failure_duration,
                    ) = generate_failure(
                        station_id,
                        shift_name
                    )

                    effective_failure_duration = min(
                        failure_duration,
                        BLOCK_MINUTES * 0.90
                    )

                    downtime_minutes = effective_failure_duration
                    machine_status = "Downtime"
                    current_downtime_id = f"DT{downtime_counter:06d}"

                    failure_start = timestamp + pd.Timedelta(
                        minutes=np.random.uniform(0, 5)
                    )

                    failure_end = failure_start + pd.Timedelta(
                        minutes=effective_failure_duration
                    )

                    downtime_records.append(
                        {
                            "downtime_id": current_downtime_id,
                            "station_id": station_id,
                            "start_time": failure_start,
                            "end_time": failure_end,
                            "duration_min": round(
                                effective_failure_duration,
                                2
                            ),
                            "shift": shift_name,
                            "category": failure_category,
                            "failure_type": failure_type,
                            "planned_flag": False,
                            "maintenance_required": (
                                effective_failure_duration >= 8
                            ),
                            "product_id": shift_product,
                        }
                    )

                    downtime_counter += 1

                    if effective_failure_duration >= 8:

                        maintenance_records.append(
                            {
                                "maintenance_id":
                                    f"MT{maintenance_counter:06d}",
                                "station_id": station_id,
                                "timestamp": failure_start,
                                "type": "Corrective",
                                "failure_type": failure_type,
                                "duration_min": round(
                                    effective_failure_duration,
                                    2
                                ),
                                "component": failure_type,
                                "action_taken":
                                    "Inspect, reset or replace affected component",
                                "successful_restart": True,
                            }
                        )

                        maintenance_counter += 1

                # ----------------------------------------
                # PLANNED MAINTENANCE
                # ----------------------------------------

                elif planned_maintenance_occurs(station_id):

                    maintenance_duration = min(
                        np.random.uniform(5, 12),
                        BLOCK_MINUTES * 0.90
                    )

                    downtime_minutes = maintenance_duration
                    machine_status = "Planned Maintenance"
                    current_downtime_id = f"DT{downtime_counter:06d}"

                    downtime_records.append(
                        {
                            "downtime_id": current_downtime_id,
                            "station_id": station_id,
                            "start_time": timestamp,
                            "end_time": timestamp + pd.Timedelta(
                                minutes=maintenance_duration
                            ),
                            "duration_min": round(
                                maintenance_duration,
                                2
                            ),
                            "shift": shift_name,
                            "category": "Maintenance",
                            "failure_type": "Planned Maintenance",
                            "planned_flag": True,
                            "maintenance_required": True,
                            "product_id": shift_product,
                        }
                    )

                    downtime_counter += 1

                    maintenance_records.append(
                        {
                            "maintenance_id":
                                f"MT{maintenance_counter:06d}",
                            "station_id": station_id,
                            "timestamp": timestamp,
                            "type": "Preventive",
                            "failure_type": None,
                            "duration_min": round(
                                maintenance_duration,
                                2
                            ),
                            "component": "Scheduled Inspection",
                            "action_taken":
                                "Preventive inspection and servicing",
                            "successful_restart": True,
                        }
                    )

                    maintenance_counter += 1

                runtime_minutes = (
                    planned_minutes - downtime_minutes
                )

                runtime_minutes = max(
                    0,
                    runtime_minutes
                )

                normal_variation = np.random.normal(
                    loc=1.0,
                    scale=0.025
                )

                actual_cycle_sec = (
                    base_cycle
                    * product_cycle_factor
                    * shift_config["speed_factor"]
                    * normal_variation
                )

                actual_cycle_sec = max(
                    actual_cycle_sec,
                    base_cycle * 0.95
                )

                if (
                    station_id == "ST04"
                    and shift_name == "C"
                ):
                    actual_cycle_sec *= 1.07

                runtime_seconds = runtime_minutes * 60

                if runtime_seconds > 0:

                    total_quantity = int(
                        runtime_seconds
                        / actual_cycle_sec
                    )

                else:

                    total_quantity = 0

                scrap_probability = (
                    station.base_scrap_rate
                    * product_quality_factor
                    * shift_config["quality_factor"]
                )

                if station_id == "ST03":
                    scrap_probability *= 1.15

                if (
                    station_id == "ST04"
                    and shift_name == "C"
                ):
                    scrap_probability *= 1.20

                rejected_quantity = np.random.binomial(
                    total_quantity,
                    min(scrap_probability, 0.15)
                )

                good_quantity = (
                    total_quantity
                    - rejected_quantity
                )

                operator_team = {
                    "A": "Team-A",
                    "B": "Team-B",
                    "C": "Team-C",
                }[shift_name]

                production_records.append(
                    {
                        "record_id":
                            f"PR{production_record_counter:07d}",
                        "timestamp": timestamp,
                        "date": production_date.date(),
                        "shift": shift_name,
                        "station_id": station_id,
                        "product_id": shift_product,
                        "planned_minutes":
                            round(planned_minutes, 2),
                        "downtime_minutes":
                            round(downtime_minutes, 2),
                        "runtime_minutes":
                            round(runtime_minutes, 2),
                        "ideal_cycle_sec":
                            round(
                                base_cycle
                                * product_cycle_factor,
                                2
                            ),
                        "actual_cycle_sec":
                            round(actual_cycle_sec, 2),
                        "total_quantity":
                            total_quantity,
                        "good_quantity":
                            good_quantity,
                        "rejected_quantity":
                            rejected_quantity,
                        "machine_status":
                            machine_status,
                        "operator_team":
                            operator_team,
                        "downtime_id":
                            current_downtime_id,
                    }
                )

                production_record_counter += 1

production_df = pd.DataFrame(
    production_records
)

downtime_df = pd.DataFrame(
    downtime_records
)

maintenance_df = pd.DataFrame(
    maintenance_records
)

target_records = []
target_counter = 1

shift_schedule = (
    production_df[
        ["date", "shift", "product_id"]
    ]
    .drop_duplicates()
    .sort_values(["date", "shift"])
)

for row in shift_schedule.itertuples(index=False):

    product_row = products.loc[
        products["product_id"] == row.product_id
    ].iloc[0]

    bottleneck_cycle = (
        machines["ideal_cycle_sec"].max()
        * product_row["cycle_factor"]
    )

    theoretical_output = (
        PLANNED_MINUTES_PER_SHIFT * 60
    ) / bottleneck_cycle

    target_output = int(
        theoretical_output * 0.88
    )

    target_records.append(
        {
            "target_id":
                f"TG{target_counter:05d}",
            "date":
                row.date,
            "shift":
                row.shift,
            "product_id":
                row.product_id,
            "target_good_output":
                target_output,
        }
    )

    target_counter += 1

targets_df = pd.DataFrame(
    target_records
)

production_df.to_csv(
    RAW_DATA_DIR / "production_log.csv",
    index=False
)

downtime_df.to_csv(
    RAW_DATA_DIR / "downtime_log.csv",
    index=False
)

maintenance_df.to_csv(
    RAW_DATA_DIR / "maintenance_log.csv",
    index=False
)

targets_df.to_csv(
    RAW_DATA_DIR / "production_targets.csv",
    index=False
)

print("\n--------------------------------")
print("DATA GENERATION COMPLETE")
print("--------------------------------")

print(
    f"Production records: {len(production_df):,}"
)

print(
    f"Downtime events: {len(downtime_df):,}"
)

print(
    f"Maintenance events: {len(maintenance_df):,}"
)

print(
    f"Production targets: {len(targets_df):,}"
)

quantity_check = (
    production_df["good_quantity"]
    + production_df["rejected_quantity"]
    == production_df["total_quantity"]
).all()

time_check = np.isclose(
    production_df["runtime_minutes"]
    + production_df["downtime_minutes"],
    production_df["planned_minutes"],
    atol=0.02
).all()

print(f"Quantity consistency: {quantity_check}")
print(f"Time consistency: {time_check}")

print("\nDowntime by station:")

print(
    downtime_df.groupby("station_id")[
        "duration_min"
    ]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop failure types:")

print(
    downtime_df.groupby("failure_type")[
        "duration_min"
    ]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nProduction by shift:")

print(
    production_df.groupby("shift")[
        "good_quantity"
    ]
    .sum()
)
# Smart Factory Production Performance & OEE Improvement

## Overview

This portfolio project simulates a small manufacturing consulting engagement for a fictional industrial manufacturer, **NexaMotion Components GmbH**.

The objective is to analyze production performance, identify OEE losses, determine bottlenecks and reliability issues, and develop improvement recommendations supported by a simulated financial business case.

All production, maintenance, failure, and financial data used in this project are synthetic.

## Production Process

Raw Material  
→ **ST01 — CNC Machining**  
→ **ST02 — Robot Handling**  
→ **ST03 — Quality Inspection**  
→ **ST04 — Assembly & Testing**  
→ **ST05 — Packaging**

Products:

- ACT-100 — Standard Actuator
- ACT-200 — Medium-Duty Actuator
- ACT-300 — Heavy-Duty Actuator

## Tools

- Python
- Pandas
- NumPy
- Matplotlib
- Excel
- Power BI
- DAX
- Git / GitHub

## Analysis Scope

The project includes:

- Synthetic manufacturing data generation
- Data cleaning and validation
- OEE calculation
- Availability, Performance, and Quality analysis
- MTBF and MTTR analysis
- Throughput and target-attainment analysis
- Scrap and rejection analysis
- Downtime Pareto analysis
- Bottleneck analysis
- Shift and product comparison
- Root-cause analysis
- Improvement prioritization
- ROI and payback estimation
- Power BI dashboard
- Excel management workbook

## Performance Baseline

| KPI | Result |
|---|---:|
| OEE | 88.30% |
| Availability | 96.39% |
| Performance | 93.21% |
| Quality | 98.28% |
| MTBF | 9.74 h |
| MTTR | 12.45 min |

## Key Findings

### ST04 — Assembly & Testing

- Lowest station OEE: **85.46%**
- Lowest station Performance: **91.48%**
- Line constraint in **97.78% of shifts**
- Slowest shift: **Shift C — 80.58 sec mean cycle time**
- Slowest product: **ACT-300 — 82.23 sec mean cycle time**
- Largest ST04 unplanned downtime cause: **Functional Test Failure — 439.60 min**

**Interpretation:** ST04 is the primary production throughput constraint.

### ST02 — Robot Handling

- Highest total downtime: **1,993.35 min**
- Lowest Availability: **95.08%**
- Part Presence Sensor downtime: **635.46 min**
- Sensor events: **49**
- Sensor share of ST02 unplanned downtime: **33.07%**
- Average sensor event duration: **12.97 min**

**Interpretation:** ST02 is the main reliability loss area, but improving it alone does not increase finished-goods output while ST04 remains constrained.

### ST03 — Quality Inspection

- Lowest Quality: **96.81%**
- Scrap / rejection rate: **3.19%**

**Interpretation:** ST03 is the main quality-loss area and requires further process validation before implementing corrective actions.

## Recommended Improvement Strategy

### Phase 1 — Improve ST04 Throughput

- Work-element time study
- Standard work optimization
- Shift C comparison
- ACT-300 cycle-time analysis
- Functional-test cycle review
- Fixture and material-waiting analysis

### Phase 2 — Improve ST02 Reliability

- Inspect sensor alignment and mounting
- Review wiring, contamination, and detection conditions
- Improve PLC diagnostics
- Standardize recovery procedures
- Reduce failure recurrence and recovery time

## Simulated Financial Business Case

| Metric | Result |
|---|---:|
| Implementation cost | €15,000 |
| Estimated Year-1 benefit | €54,807.99 |
| Year-1 ROI | 265.39% |
| Payback period | 3.28 months |
| Theoretical additional annual capacity | 7,307.7 units |

Financial values are simulated assumptions created for portfolio demonstration.

## Repository Structure

```text
smart-factory-oee/
│
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── python/
│   ├── generate_data.py
│   ├── clean_data.py
│   ├── calculate_kpis.py
│   ├── analysis.py
│   ├── root_cause_analysis.py
│   └── business_case.py
├── powerbi/
├── excel/
├── business_case/
├── screenshots/
├── diagrams/
├── documentation/
└── presentation/
```

## Run the Project

```bash
python -m venv .venv
```

Windows:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis pipeline:

```bash
python python/generate_data.py
python python/clean_data.py
python python/calculate_kpis.py
python python/analysis.py
python python/root_cause_analysis.py
python python/business_case.py
```

## Project Note

This project is designed for portfolio demonstration in Manufacturing Consulting, Smart Factory, Industry 4.0, Automation Consulting, Digital Manufacturing, and related roles. No confidential industrial data is used.

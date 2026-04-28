# Parameter Reference

Quick reference for all parameters used across the M-S-O pipeline.

| Symbol | Name | Value | Unit | Stage / Notes |
|--------|------|-------|------|---------------|
| H | Hospitals | 12 | count | All stages |
| B | Blood banks | 3 | count | All stages |
| T | Time windows | 24 | hours | All stages |
| K | Blood types | 4 | count | S, O |
| D_train | Training days | 6 | days | M |
| D_test | Test days | 1 | day | M |
| s | Seasonal period | 24 | hours | M (SARIMA) |
| T_exp | Expiration window | 15 | min | S |
| c | Drone fleet size | 8 | drones | S, O |
| v | Drone speed | 50 | km/h | S, O |
| beta | Battery drain | 1.5 | %/km | S |
| B_min | Min battery | 30 | % | S |
| r_charge | Recharge rate | 20 | %/min | S |
| Q_max | Max payload | 4 | units | S, O |
| t_s | Service time (hospital) | 3 | min | S, O |
| t_b | Loading time (bank) | 2 | min | S |
| Delta_t | Dispatch cycle | 30 | sec | S |
| I_0 | Initial inventory | 50/type/bank | units | S, O |
| C_fleet | Fleet throughput cap | 24 | missions/hr | O |
| R | Replications | 20 | per policy | S, Eval |
| alpha | Significance level | 0.05 | — | Eval |
| noise_std | Demand noise SD | 0.15 | req/hr | M (data gen) |
| scale[h] | Hospital demand scale | U(0.6, 1.4) | — | M (data gen) |

## Blood Type Distribution

| Type | Probability |
|------|-------------|
| O- | 0.10 |
| O+ | 0.40 |
| A+ | 0.35 |
| B+ | 0.15 |

## Priority Class Distribution

| Priority | Value | Probability |
|----------|-------|-------------|
| NORMAL | 1 | 0.50 |
| URGENT | 2 | 0.35 |
| EMERGENCY | 3 | 0.15 |

## Base Lambda Profile (requests/hr)

| Hour | Rate | Period |
|------|------|--------|
| 0-5 | 0.3 | Overnight (low) |
| 6 | 0.8 | Morning ramp |
| 7 | 1.2 | Morning ramp |
| 8 | 1.5 | Morning ramp |
| 9 | 1.4 | Morning ramp |
| 10 | 1.3 | Morning ramp |
| 11 | 1.6 | Morning ramp |
| 12 | 1.8 | Peak |
| 13 | 2.0 | Peak |
| 14 | 1.9 | Peak |
| 15 | 1.7 | Peak |
| 16 | 1.5 | Peak |
| 17 | 1.4 | Peak |
| 18 | 1.2 | Evening |
| 19 | 1.0 | Evening |
| 20 | 0.8 | Evening |
| 21 | 0.6 | Evening |
| 22 | 0.5 | Evening |
| 23 | 0.4 | Evening |

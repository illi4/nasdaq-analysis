# Investment Strategy Simulator

## Overview

This Python script simulates different investment strategies based on market pullbacks using historical financial data. It allows users to define various configurations in a `config.yaml` file, specifying how much capital to allocate when the market dips by certain percentages from its recent peak. The simulation runs over a specified date range, calculates performance metrics for each strategy, and compares them against standard Dollar-Cost Averaging (DCA) approaches.

## Features

* **Configurable Strategies:** Define multiple investment scenarios in `config.yaml`, specifying monthly investment amounts, analysis periods, and pullback thresholds with corresponding capital allocation percentages.
* **Pullback-Based Buying:** Simulates buying decisions based on market pullbacks (percentage drops from the most recent peak high).
* **Flexible Allocation:** Strategies can allocate percentages of *available* cash across different pullback levels. The total allocation can be 100% or less, allowing for implicit cash reserves.
* **Cash Accumulation & Interest:** If no buy conditions are met, the monthly investment accumulates as cash. An optional annual interest rate can be applied to uninvested cash.
* **DCA Comparison:** Automatically includes two baseline DCA strategies for comparison:
    * Investing the full monthly amount on the first trading day of the month.
    * Investing half the monthly amount on the first trading day and half on the trading day nearest the 15th.
* **Historical Data:** Uses a CSV file containing historical market data (Date, Close/Last, High, Low).
* **Performance Analysis:** Calculates key metrics for each strategy, including:
    * Total Capital Provided
    * Total Cash Invested
    * Final Portfolio Value (Investments + Remaining Cash)
    * Overall Growth (%)
    * Compound Annual Growth Rate (CAGR) (%)
    * Weighted Average Purchase Price
* **Detailed Reporting:** Generates an Excel spreadsheet (`.xlsx`) with:
    * A 'Summary' sheet comparing all simulated strategies, sorted by performance.
    * Individual sheets for each strategy (including DCA) showing month-by-month details like cash flow, investments made, units held, and portfolio value.
* **Logging:** Creates detailed log files for each simulation run in a `logs/` directory for debugging and tracking.

## Folder Structure

```
your-project-root/
├── investment_simulator.py  # Main Python script
├── config.yaml              # Configuration file for simulations
├── requirements.txt         # Python dependencies
├── data/
│   └── data.csv             # Historical market data file
├── results/
│   └── simulation_results.xlsx # Output Excel file (created after running)
└── logs/
    ├── <simulation_name>.log   # Log file for each simulation (created after running)
    └── ...
```

## Prerequisites

You need Python 3 installed. The required libraries are listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
```

## Requirements.txt

```
pandas
PyYAML
numpy
xlsxwriter
```

## Configuration (config.yaml)

The `config.yaml` file controls the simulation parameters:

- `monthly_investment`: The fixed amount of cash added each month for potential investment.
- `start_date` / `end_date`: The period for the simulation (YYYY-MM-DD format).
- `data_file`: Path to the historical data CSV file.
- `output_file`: Path where the results Excel file will be saved.
- `date_column` / `price_column` / `high_column` / `low_column`: Names of the corresponding columns in your data.csv file. `price_column` (Close/Last) is used for valuation, `low_column` is used for triggering buy orders, and `high_column` is used for tracking peaks.
- `log_level`: Set logging verbosity ('INFO' or 'DEBUG').
- `cash_interest_rate_annual`: Annual interest rate (as a percentage, e.g., 3.0 for 3%) applied monthly to uninvested cash balance. Set to 0 if no interest is desired.
- `simulations`: A dictionary where each key is a unique name for a simulation scenario.
  - `description`: A brief description of the strategy.
  - `allocations`: A dictionary defining the core logic:
    - key: The pullback percentage (e.g., 10 for a 10% drop from the peak).
    - value: The percentage of currently available cash to invest when that pullback threshold is hit for the first time since the last peak (e.g., 50 for 50%). The script ensures a specific pullback level triggers only once per peak.

### Example Allocation:

```yaml
simulations:
  buy_the_dips_moderate:
    description: "Allocate moderately (Adjusted %: 30, 40, 30 of initial)"
    # Original: 5: 30, 10: 40, 15: 30
    allocations:
      5: 30.00   # Invest 30% of initial available cash
      10: 57.14  # Invest 40% of initial available cash (40 / (100-30))
      15: 100.00 # Invest 30% of initial available cash (30 / (100-30-40))
```

## Data (data/data.csv)

The script requires a CSV file containing historical market data. It must include columns for the date and the daily closing price, high price, and low price. Ensure the column names in the CSV match those specified in config.yaml.

- **Date Format**: Ensure the date column can be parsed by pandas (e.g., YYYY-MM-DD, MM/DD/YYYY).
- **Numeric Prices**: Price/High/Low columns should contain numeric values. The script attempts to clean common currency symbols like '$' and ','.

## Usage

1. Configure your settings and strategies in `config.yaml`.
2. Ensure your historical data is correctly formatted in the location specified by `data_file` in the config.
3. Run the script from your terminal in the project's root directory:

```bash
python investment_simulator.py
```

4. Check the `results/` folder for the output Excel file and the `logs/` folder for detailed logs.

## Output (results/simulation_results.xlsx)

The script generates an Excel file (e.g., `simulation_results.xlsx`) containing:

- **Summary Sheet**:
  - Lists all configured scenarios plus the two DCA strategies.
  - Provides key performance metrics for each (Total Capital, Total Invested, Final Value, Growth %, CAGR %, etc.).
  - Rows are sorted by 'Overall Growth (%)' in descending order.

- **Detailed Monthly Sheets**:
  - One sheet per simulation scenario (named after the scenario key in `config.yaml`, plus sheets for `DCA_Monthly_Start` and `DCA_BiMonthly_1_15`).
  - Shows month-end snapshots including cash added, interest earned, cash invested, units bought, cumulative holdings, market price, value of investments, cash on hand, and total portfolio value.

## Logging

Detailed logs for each simulation scenario are saved in the `logs/` directory. These are useful for understanding the specific buy decisions and cash flows during the simulation, especially when using the DEBUG log level.

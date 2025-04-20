import pandas as pd
import yaml
import os
import logging
import logging.handlers # For file logging
# xlsxwriter is used implicitly by pandas when engine='xlsxwriter' is set
# No direct import needed unless using advanced features beyond formatting.

# --- Root Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# --- Configuration Loading ---
def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file."""
    logging.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        required_keys = ['date_column', 'price_column', 'high_column', 'low_column']
        if not all(key in config for key in required_keys):
            missing = [key for key in required_keys if key not in config]
            logging.error(f"Error: Missing required configuration keys: {missing}. Please add 'high_column' and 'low_column' to config.yaml.")
            return None
        config.setdefault('log_level', 'INFO')
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Error: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return None

# --- Data Loading and Preparation ---
def load_and_prepare_data(config):
    """Loads and prepares the historical market data."""
    data_file = config['data_file']
    date_col = config['date_column']
    price_col = config['price_column']
    high_col = config['high_column']
    low_col = config['low_column']
    start_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])

    logging.info(f"Loading data from {data_file}")
    try:
        df = pd.read_csv(data_file, parse_dates=[date_col], usecols=[date_col, price_col, high_col, low_col])
        df = df.rename(columns={
            date_col: 'Date', price_col: 'Close', high_col: 'High', low_col: 'Low'
        })
        df = df.sort_values(by='Date').reset_index(drop=True)

        for col in ['Close', 'High', 'Low']:
             if df[col].dtype == 'object':
                 df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
             df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=['Date', 'Close', 'High', 'Low'], inplace=True)
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

        if df.empty:
            logging.error(f"No valid data found for the specified date range: {start_date.date()} to {end_date.date()}")
            return None

        df['PeakHigh'] = df['High'].expanding().max()
        df.set_index('Date', inplace=True)

        logging.info(f"Data loaded and prepared. Shape: {df.shape}. Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {data_file}")
        return None
    except KeyError as e:
         logging.error(f"Error: Column mapping issue. Could not find required columns based on config ('{config['date_column']}', '{config['price_column']}', '{config['high_column']}', '{config['low_column']}') in {data_file}. Original error detail: {e}")
         return None
    except ValueError as e:
         logging.error(f"Error converting price data to numeric. Check for non-numeric values in columns '{config['price_column']}', '{config['high_column']}', '{config['low_column']}'. Original error detail: {e}")
         return None
    except Exception as e:
        logging.error(f"Error processing data file: {e}")
        return None

# --- Simulation Execution ---
def run_simulation(config, data):
    """Runs the investment simulation with proper Excel number formatting."""
    if data is None:
        logging.error("Simulation cannot run without data.")
        return None

    monthly_investment = config['monthly_investment']
    start_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])
    simulations = config['simulations']
    output_file = config['output_file']
    log_level_str = config.get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Per-simulation logs will be saved in '{log_dir}/' directory.")

    all_results_summary = []
    all_monthly_details = {}

    for sim_name, sim_config in simulations.items():
        # --- Setup Logger ---
        sim_logger = logging.getLogger(sim_name)
        sim_logger.setLevel(log_level)
        for handler in sim_logger.handlers[:]:
            sim_logger.removeHandler(handler)
            handler.close()
        log_file_path = os.path.join(log_dir, f"{sim_name}.log")
        file_handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        sim_logger.addHandler(file_handler)
        sim_logger.propagate = False

        sim_logger.info(f"--- Starting Simulation: {sim_name} ---")
        # (Logging simulation details...)
        sim_logger.info(f"Description: {sim_config['description']}")
        sim_logger.info(f"Allocations: {sim_config['allocations']}")
        sim_logger.info(f"Monthly Investment: ${monthly_investment}")
        sim_logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")


        allocations = sim_config['allocations']
        sorted_thresholds = sorted(allocations.keys())
        cash_available = 0.0
        total_units_held = 0.0
        total_cash_invested = 0.0
        investments = []
        monthly_log = []
        last_peak_high = 0
        triggered_pullbacks_since_peak = set()

        # --- Simulation Loop ---
        for month_start in pd.date_range(start_date, end_date, freq='MS'):
            month_end = month_start + pd.offsets.MonthEnd(0)
            if month_end > end_date: month_end = end_date

            cash_available += monthly_investment
            cash_at_month_start = cash_available
            sim_logger.debug(f"Month Start {month_start.strftime('%Y-%m')}: Added ${monthly_investment}, Cash Available: ${cash_at_month_start:,.2f}")

            month_data = data.loc[month_start:month_end]
            cash_invested_this_month = 0.0
            units_bought_this_month = 0.0
            purchase_details_this_month = []

            if not month_data.empty:
                for date, row in month_data.iterrows():
                    current_high = row['High']
                    current_low = row['Low']
                    current_peak_high = row['PeakHigh']

                    if current_peak_high > last_peak_high:
                        if last_peak_high > 0:
                             sim_logger.debug(f"{date.date()}: New peak detected: {current_peak_high:.2f} (was {last_peak_high:.2f}). Resetting triggers.")
                        last_peak_high = current_peak_high
                        triggered_pullbacks_since_peak = set()

                    for threshold in sorted_thresholds:
                        if last_peak_high <= 0: continue
                        pullback_target_price = last_peak_high * (1 - threshold / 100.0)

                        if current_low <= pullback_target_price and threshold not in triggered_pullbacks_since_peak:
                            allocation_pct = allocations[threshold] / 100.0
                            amount_to_invest = cash_available * allocation_pct

                            if amount_to_invest > 0 and current_low > 0:
                                units_bought = amount_to_invest / current_low
                                purchase_price = current_low
                                cash_available -= amount_to_invest
                                total_units_held += units_bought
                                total_cash_invested += amount_to_invest
                                cash_invested_this_month += amount_to_invest
                                units_bought_this_month += units_bought
                                purchase_details_this_month.append({'date': date, 'price': purchase_price, 'units': units_bought, 'amount': amount_to_invest, 'threshold': threshold})
                                investments.append({'date': date, 'units': units_bought, 'price': purchase_price, 'amount': amount_to_invest})
                                triggered_pullbacks_since_peak.add(threshold)
                                sim_logger.debug(f"{date.date()}: Buy triggered! Low ({current_low:.2f}) <= Target ({pullback_target_price:.2f}) for {threshold}% pullback from Peak High ({last_peak_high:.2f}). Investing ${amount_to_invest:.2f} ({allocation_pct*100}%) at ${purchase_price:.2f}. Cash left: ${cash_available:.2f}")
                            elif current_low <= 0:
                                sim_logger.warning(f"{date.date()}: Skipping buy for {threshold}% pullback trigger because Low price is zero or negative ({current_low}).")

            # --- Month End Calculations & Logging ---
            final_close_price_month_end = data['Close'].asof(month_end)
            if pd.isna(final_close_price_month_end):
                sim_logger.warning(f"Could not find closing price on or before month end {month_end.strftime('%Y-%m-%d')} using asof(). Using 0 price for valuation this month.")
                final_close_price_month_end = 0

            value_of_investments = total_units_held * final_close_price_month_end
            total_portfolio_value = value_of_investments + cash_available
            avg_purchase_price_month = (sum(p['amount'] for p in purchase_details_this_month) /
                                       sum(p['units'] for p in purchase_details_this_month)) if units_bought_this_month > 0 else 0

            monthly_log_entry = {
                'MonthEnd': month_end.strftime('%Y-%m-%d'), # Keep date as string for Excel clarity
                'CashAdded': monthly_investment,
                'CashAtMonthStart': cash_at_month_start,
                'CashInvestedThisMonth': cash_invested_this_month,
                'AvgPurchasePriceThisMonth': avg_purchase_price_month,
                'UnitsBoughtThisMonth': units_bought_this_month,
                'CumulativeUnitsHeld': total_units_held,
                'CumulativeCashInvested': total_cash_invested,
                'MarketClosePriceMonthEnd': final_close_price_month_end,
                'ValueInvestmentsMonthEnd': value_of_investments,
                'CashOnHandMonthEnd': cash_available,
                'TotalPortfolioValueMonthEnd': total_portfolio_value
            }
            monthly_log.append(monthly_log_entry)
            sim_logger.info(f"Month End {month_end.strftime('%Y-%m-%d')}: Portfolio Value: ${total_portfolio_value:,.2f} (Investments: ${value_of_investments:,.2f}, Cash: ${cash_available:,.2f})")
            sim_logger.debug(f"Month End Details: {monthly_log_entry}")

        # --- End of Simulation Calculation ---
        final_date = data.index[-1]
        final_close_price = data.loc[final_date]['Close']
        final_investment_value = total_units_held * final_close_price
        final_portfolio_value = final_investment_value + cash_available
        total_months = len(pd.date_range(start_date, end_date, freq='MS'))
        total_capital_provided = total_months * monthly_investment

        # Calculate growth as a decimal (e.g., 0.15 for 15%) for Excel % formatting
        percent_growth_decimal = ((final_portfolio_value / total_capital_provided) - 1) if total_capital_provided > 0 else 0

        weighted_avg_purchase_price = (total_cash_invested / total_units_held) if total_units_held > 0 else 0

        summary = {
            'Scenario': sim_name, 'Description': sim_config['description'],
            'Total Capital Provided': total_capital_provided,
            'Total Cash Invested': total_cash_invested,
            'Final Investment Value': final_investment_value,
            'Final Cash Remaining': cash_available,
            'Final Total Portfolio Value': final_portfolio_value,
            'Overall Growth (%)': percent_growth_decimal, # Store as decimal
            'Total Units Purchased': total_units_held,
            'Weighted Avg Purchase Price': weighted_avg_purchase_price,
            'Final Market Close Price': final_close_price
        }
        all_results_summary.append(summary)
        # Convert monthly log list to DataFrame here before storing
        all_monthly_details[sim_name] = pd.DataFrame(monthly_log)

        sim_logger.info(f"--- Simulation {sim_name} Complete ---")
        sim_logger.info(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        sim_logger.info(f"Total Capital Provided: ${total_capital_provided:,.2f}")
        # Log the percentage correctly formatted
        sim_logger.info(f"Overall Growth: {percent_growth_decimal:.2%}")
        sim_logger.info(f"Weighted Avg Purchase Price: ${weighted_avg_purchase_price:.2f}")

        file_handler.close()

    # --- Write results to Excel using xlsxwriter engine ---
    logging.info(f"Writing results to {output_file} using xlsxwriter engine for number formatting.")
    try:
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # --- Write Summary Sheet ---
            summary_df = pd.DataFrame(all_results_summary)
            summary_sheet_name = 'Summary'
            summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)

            # Get workbook and worksheet objects
            workbook = writer.book
            summary_worksheet = writer.sheets[summary_sheet_name]

            # Define formats
            currency_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            units_format = workbook.add_format({'num_format': '#,##0.0000'}) # 4 decimal places for units
            header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})


            # Apply formatting to Summary sheet columns
            # Helper to get column index (A=0, B=1, ...)
            summary_col_map = {name: i for i, name in enumerate(summary_df.columns)}

            summary_currency_cols = ['Total Capital Provided', 'Total Cash Invested', 'Final Investment Value', 'Final Cash Remaining', 'Final Total Portfolio Value', 'Weighted Avg Purchase Price', 'Final Market Close Price']
            summary_unit_cols = ['Total Units Purchased']
            summary_percent_cols = ['Overall Growth (%)']

            for col_name in summary_currency_cols:
                if col_name in summary_col_map:
                    summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 18, currency_format) # Adjust width (18) as needed
            for col_name in summary_unit_cols:
                 if col_name in summary_col_map:
                    summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 18, units_format)
            for col_name in summary_percent_cols:
                 if col_name in summary_col_map:
                    summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 12, percent_format)

            # Apply header format to summary sheet
            for col_num, value in enumerate(summary_df.columns.values):
                summary_worksheet.write(0, col_num, value, header_format)


            # --- Write Detailed Monthly Sheets ---
            for sim_name, monthly_df in all_monthly_details.items():
                safe_sheet_name = "".join(c for c in sim_name if c.isalnum() or c in (' ', '_', '-'))[:31]
                monthly_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

                # Get worksheet object for the monthly sheet
                monthly_worksheet = writer.sheets[safe_sheet_name]
                monthly_col_map = {name: i for i, name in enumerate(monthly_df.columns)}

                # Define columns for formatting in monthly sheets
                monthly_currency_cols = ['CashAdded', 'CashAtMonthStart', 'CashInvestedThisMonth', 'AvgPurchasePriceThisMonth', 'CumulativeCashInvested', 'MarketClosePriceMonthEnd', 'ValueInvestmentsMonthEnd', 'CashOnHandMonthEnd', 'TotalPortfolioValueMonthEnd']
                monthly_units_cols = ['UnitsBoughtThisMonth', 'CumulativeUnitsHeld']

                # Apply formats to monthly sheet columns
                for col_name in monthly_currency_cols:
                     if col_name in monthly_col_map:
                        monthly_worksheet.set_column(monthly_col_map[col_name], monthly_col_map[col_name], 18, currency_format)
                for col_name in monthly_units_cols:
                     if col_name in monthly_col_map:
                        monthly_worksheet.set_column(monthly_col_map[col_name], monthly_col_map[col_name], 18, units_format)

                # Apply header format to monthly sheet
                for col_num, value in enumerate(monthly_df.columns.values):
                    monthly_worksheet.write(0, col_num, value, header_format)


        logging.info("Results successfully written to Excel with number formatting.")
    except ImportError:
        logging.error("Error: 'xlsxwriter' engine required but not installed. Please run: pip install xlsxwriter")
        # Fallback or re-raise might be needed here if xlsxwriter is critical
    except Exception as e:
        logging.error(f"Error writing results to Excel: {e}")

    return pd.DataFrame(all_results_summary) # Return summary still useful


# --- Main Execution ---
if __name__ == "__main__":
    config = load_config()
    if config:
        data = load_and_prepare_data(config)
        if data is not None:
            results = run_simulation(config, data)
            if results is not None:
                logging.info("\n--- Overall Simulation Summary ---")
                # print(results.to_string(index=False)) # Raw data before formatting
            else:
                logging.error("Simulation run failed.")
        else:
            logging.error("Data loading failed. Aborting simulation.")
    else:
        logging.error("Configuration loading failed. Aborting.")


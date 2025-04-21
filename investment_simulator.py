import pandas as pd
import yaml
import os
import logging
import logging.handlers # For file logging
import numpy as np # For CAGR calculation

# --- Root Logger Setup ---
# Configure the root logger (optional, but good practice)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # Log to console

# --- Configuration Loading ---
def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file."""
    logging.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Validate required keys
        required_keys = ['monthly_investment', 'start_date', 'end_date', 'data_file',
                         'output_file', 'date_column', 'price_column', 'high_column', 'low_column',
                         'simulations']
        if not all(key in config for key in required_keys):
            missing = [key for key in required_keys if key not in config]
            logging.error(f"Error: Missing required configuration keys: {missing}.")
            return None
        # Set defaults for optional keys
        config.setdefault('log_level', 'INFO')
        config.setdefault('cash_interest_rate_annual', 0.0) # Default interest rate is 0
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
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
        # Load only necessary columns
        df = pd.read_csv(data_file, parse_dates=[date_col], usecols=[date_col, price_col, high_col, low_col])
        # Rename columns for consistency
        df = df.rename(columns={
            date_col: 'Date', price_col: 'Close', high_col: 'High', low_col: 'Low'
        })
        df = df.sort_values(by='Date').reset_index(drop=True)

        # Clean and convert price columns to numeric
        for col in ['Close', 'High', 'Low']:
             if df[col].dtype == 'object':
                 # Remove common currency symbols and commas
                 df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
             df[col] = pd.to_numeric(df[col], errors='coerce') # Convert, invalid parsing will be NaT

        # Drop rows with missing essential data
        df.dropna(subset=['Date', 'Close', 'High', 'Low'], inplace=True)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True) # Ensure chronological order

        # Filter data based on config dates, considering available data range
        data_start_date = df.index.min()
        data_end_date = df.index.max()
        effective_start = max(start_date, data_start_date)
        effective_end = min(end_date, data_end_date)

        df = df.loc[effective_start:effective_end].copy() # Use .copy() to avoid SettingWithCopyWarning

        if df.empty:
            logging.error(f"No valid data found for the specified date range: {effective_start.date()} to {effective_end.date()}")
            return None

        # Calculate expanding peak high for pullback calculations
        df['PeakHigh'] = df['High'].expanding().max()

        logging.info(f"Data loaded and prepared. Shape: {df.shape}. Date range: {df.index.min().date()} to {df.index.max().date()}. Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {data_file}")
        return None
    except KeyError as e:
         logging.error(f"Error: Column mapping issue. Check 'date_column', 'price_column', 'high_column', 'low_column' in config. Original error detail: {e}")
         return None
    except ValueError as e:
         logging.error(f"Error converting price data to numeric. Check data format in CSV. Original error detail: {e}")
         return None
    except Exception as e:
        logging.error(f"Error processing data file: {e}")
        return None

# --- CAGR Calculation Helper ---
def calculate_cagr(end_value, start_value, years):
    """Calculates Compound Annual Growth Rate (CAGR)."""
    if years <= 0 or start_value <= 0 or end_value <= 0:
        return 0.0 # Avoid division by zero or invalid inputs
    # Ensure float division and handle potential type issues
    try:
        return (float(end_value) / float(start_value)) ** (1.0 / years) - 1.0
    except (ValueError, TypeError):
        logging.warning(f"Could not calculate CAGR for end={end_value}, start={start_value}, years={years}")
        return 0.0

# --- Helper to find next trading day ---
def find_next_trading_day(target_date, data_index):
    """Finds the first trading day index on or after target_date within the provided index."""
    # Ensure data_index is sorted, which it should be from load_and_prepare_data
    if not data_index.is_monotonic_increasing:
        logging.warning("Data index provided to find_next_trading_day is not sorted. Sorting it.")
        data_index = data_index.sort_values()

    try:
        # searchsorted finds insertion point; if target_date exists, it points to it.
        # If target_date doesn't exist, it points to the next valid date's position.
        idx_pos = data_index.searchsorted(target_date)
        if idx_pos < len(data_index):
            return data_index[idx_pos] # Return the actual date index
        else:
            # Target date is after the last date in the index
            return None
    except Exception as e:
        logging.error(f"Error finding next trading day near {target_date}: {e}")
        return None

# --- Simulation Execution ---
def run_simulation(config, data):
    """Runs investment simulations including cash interest, config scenarios and two DCA variants."""
    if data is None or data.empty:
        logging.error("Simulation cannot run without valid data.")
        return None

    # --- Simulation Setup ---
    monthly_investment = config['monthly_investment']
    actual_start_date = data.index.min()
    actual_end_date = data.index.max()
    simulations = config.get('simulations', {})
    output_file = config['output_file']
    log_level_str = config.get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid

    # Cash interest setup
    cash_interest_rate_annual = config.get('cash_interest_rate_annual', 0.0)
    monthly_interest_rate = cash_interest_rate_annual / 100.0 / 12.0
    logging.info(f"Applying monthly interest rate to uninvested cash: {monthly_interest_rate:.4%}")

    # Ensure logs directory exists
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Per-simulation logs will be saved in '{log_dir}/' directory.")

    # --- Data Structures for Results ---
    all_results_summary = []
    all_monthly_details = {}
    all_transaction_details = {} # NEW: Store individual buy transactions per simulation

    # --- Calculate overall simulation period details ---
    num_years = (actual_end_date - actual_start_date).days / 365.25
    if num_years <= 0: num_years = np.nan # Use NaN for invalid duration

    market_start_price = data['Close'].iloc[0]
    market_end_price = data['Close'].iloc[-1]

    logging.info(f"Simulation Period: {actual_start_date.date()} to {actual_end_date.date()} ({num_years:.2f} years)")
    logging.info(f"Market Start Price ({actual_start_date.date()}): {market_start_price:,.2f}")
    logging.info(f"Market End Price ({actual_end_date.date()}): {market_end_price:,.2f}")

    # --- Run Config-Based Simulations ---
    for sim_name, sim_config in simulations.items():
        # --- Per-Simulation Logger Setup ---
        sim_logger = logging.getLogger(sim_name)
        sim_logger.setLevel(log_level)
        # Remove existing handlers to avoid duplicate logs if script is re-run
        for handler in sim_logger.handlers[:]:
            sim_logger.removeHandler(handler)
            handler.close()
        # Add file handler for this simulation
        log_file_path = os.path.join(log_dir, f"{sim_name}.log")
        try:
            file_handler = logging.FileHandler(log_file_path, mode='w') # Overwrite log file each run
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            sim_logger.addHandler(file_handler)
            sim_logger.propagate = False # Prevent logs from propagating to root logger
            sim_logger.info(f"--- Starting Simulation: {sim_name} ---")
            sim_logger.info(f"Description: {sim_config.get('description', 'N/A')}")
            sim_logger.info(f"Allocations: {sim_config.get('allocations', {})}")
        except Exception as e:
            logging.error(f"Failed to set up logger for simulation {sim_name}: {e}")
            continue # Skip this simulation if logging fails

        # --- Simulation State Variables ---
        allocations = sim_config.get('allocations', {})
        if not isinstance(allocations, dict):
             sim_logger.error("Invalid 'allocations' format in config. Skipping simulation.")
             file_handler.close()
             continue
        sorted_thresholds = sorted(allocations.keys()) # Buy at smaller pullbacks first if hit on same day
        cash_available = 0.0
        total_units_held = 0.0
        total_cash_invested = 0.0
        num_buys = 0
        monthly_log = [] # Stores summary data for each month
        all_purchases_log = [] # NEW: Stores every individual purchase detail for this sim
        last_peak_high = 0 # Track the peak high since the start or last reset
        triggered_pullbacks_since_peak = set() # Track which thresholds have been triggered since the last peak

        # --- Simulation Loop (Iterate through months) ---
        for month_start_dt in pd.date_range(actual_start_date, actual_end_date, freq='MS'): # 'MS' for Month Start
            # Find the actual first and last trading days within the current month in our data
            try:
                current_month_start_actual = data.index[data.index >= month_start_dt].min()
                month_end_target = month_start_dt + pd.offsets.MonthEnd(0)
                current_month_end_actual = data.index[data.index <= month_end_target].max()
                # Ensure the end date doesn't exceed the overall simulation end date
                current_month_end_actual = min(current_month_end_actual, actual_end_date)
            except IndexError:
                sim_logger.warning(f"No trading data found for month starting {month_start_dt.strftime('%Y-%m')}. Skipping month.")
                continue # Skip to the next month if no data exists

            interest_earned_this_month = 0.0
            # Apply interest *before* adding monthly investment
            if cash_available > 0 and monthly_interest_rate > 0:
                interest_earned_this_month = cash_available * monthly_interest_rate
                cash_available += interest_earned_this_month
                sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Interest Earned: ${interest_earned_this_month:,.2f}, Cash Now: ${cash_available:,.2f}")

            # Add monthly investment amount
            cash_added_this_month = 0.0
            if month_start_dt >= actual_start_date: # Only add if within the simulation period
                 cash_available += monthly_investment
                 cash_added_this_month = monthly_investment
                 cash_at_month_start_recorded = cash_available # Record cash *after* adding interest and monthly amount
                 sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Added ${monthly_investment}, Cash Available: ${cash_at_month_start_recorded:,.2f}")
            else:
                 cash_at_month_start_recorded = cash_available # Record cash after interest if start date is mid-month

            # Get data for the current month
            month_data = data.loc[current_month_start_actual:current_month_end_actual]
            cash_invested_this_month = 0.0
            units_bought_this_month = 0.0
            # purchase_details_this_month = [] # No longer needed monthly, use all_purchases_log

            # --- Daily Buy Logic Loop (Within the month) ---
            if not month_data.empty:
                for date, row in month_data.iterrows():
                    current_high = row['High']
                    current_low = row['Low']
                    current_peak_high = row['PeakHigh'] # Use the precalculated expanding peak

                    # Check for new peak high
                    if current_peak_high > last_peak_high:
                        if last_peak_high > 0: # Avoid logging on the very first peak
                            sim_logger.debug(f"{date.date()}: New peak detected: {current_peak_high:.2f} (Previous: {last_peak_high:.2f}). Resetting pullback triggers.")
                        last_peak_high = current_peak_high
                        triggered_pullbacks_since_peak = set() # Reset triggers on new peak

                    # Check pullback thresholds
                    for threshold in sorted_thresholds:
                        if last_peak_high <= 0: continue # Cannot calculate pullback without a peak

                        pullback_target_price = last_peak_high * (1 - threshold / 100.0)

                        # Check if low price hits the target AND this threshold hasn't triggered since the last peak
                        if current_low <= pullback_target_price and threshold not in triggered_pullbacks_since_peak:
                            allocation_pct = allocations[threshold] / 100.0
                            amount_to_invest = cash_available * allocation_pct

                            # Execute buy if amount is significant and price is valid
                            if amount_to_invest >= 0.01 and current_low > 0: # Use low price as buy price
                                purchase_price = current_low
                                units_bought = amount_to_invest / purchase_price

                                # Update state
                                cash_available -= amount_to_invest
                                total_units_held += units_bought
                                total_cash_invested += amount_to_invest
                                cash_invested_this_month += amount_to_invest
                                units_bought_this_month += units_bought
                                num_buys += 1

                                # Log the transaction details
                                transaction_log = {
                                    'date': date,
                                    'price': purchase_price,
                                    'units': units_bought,
                                    'amount': amount_to_invest,
                                    'threshold': threshold # Record the trigger threshold
                                }
                                all_purchases_log.append(transaction_log) # Add to the overall list

                                # Mark this threshold as triggered for the current peak
                                triggered_pullbacks_since_peak.add(threshold)

                                # *** ENHANCED LOG MESSAGE ***
                                sim_logger.debug(f"{date.date()}: Buy triggered at {threshold}% pullback! Investing ${amount_to_invest:.2f} to buy {units_bought:.4f} units @ ${purchase_price:.2f}. Cash left: ${cash_available:.2f}")

                            elif amount_to_invest < 0.01:
                                sim_logger.debug(f"{date.date()}: Skipping buy for {threshold}% pullback - calculated investment amount (${amount_to_invest:.2f}) too small.")
                            elif current_low <= 0:
                                sim_logger.warning(f"{date.date()}: Skipping buy for {threshold}% pullback - Low price (${current_low:.2f}) is zero or negative.")

            # --- Month End Calculations & Logging ---
            # Get the closing price on the last actual trading day of the month
            final_close_price_month_end = data['Close'].asof(current_month_end_actual)
            if pd.isna(final_close_price_month_end): final_close_price_month_end = 0 # Handle potential NaN

            value_of_investments = total_units_held * final_close_price_month_end
            total_portfolio_value = value_of_investments + cash_available
            # Calculate average purchase price *for this month only*
            avg_purchase_price_month = (cash_invested_this_month / units_bought_this_month) if units_bought_this_month > 0 else 0

            # Create log entry for the monthly summary sheet
            monthly_log_entry = {
                'MonthEnd': current_month_end_actual.strftime('%Y-%m-%d'),
                'CashAdded': cash_added_this_month,
                'InterestEarnedThisMonth': interest_earned_this_month,
                'CashAtMonthStart': cash_at_month_start_recorded, # Cash after adding interest and monthly amount
                'CashInvestedThisMonth': cash_invested_this_month,
                'AvgPurchasePriceThisMonth': avg_purchase_price_month, # Avg price of buys *this month*
                'UnitsBoughtThisMonth': units_bought_this_month,
                'CumulativeUnitsHeld': total_units_held,
                'CumulativeCashInvested': total_cash_invested,
                'MarketClosePriceMonthEnd': final_close_price_month_end,
                'ValueInvestmentsMonthEnd': value_of_investments,
                'CashOnHandMonthEnd': cash_available,
                'TotalPortfolioValueMonthEnd': total_portfolio_value
            }
            monthly_log.append(monthly_log_entry)
            sim_logger.info(f"Month End {current_month_end_actual.strftime('%Y-%m-%d')}: Portfolio Value: ${total_portfolio_value:,.2f} (Investments: ${value_of_investments:,.2f}, Cash: ${cash_available:,.2f})")
            sim_logger.debug(f"Month End Details: Units={total_units_held:.4f}, CashInvested={total_cash_invested:.2f}")

        # --- End of Simulation Calculation (Config Scenario) ---
        final_close_price = market_end_price # Use the overall market end price
        final_investment_value = total_units_held * final_close_price
        final_portfolio_value = final_investment_value + cash_available
        # Calculate total capital provided accurately based on months simulated
        total_months_simulated = len(pd.date_range(start=actual_start_date, end=actual_end_date, freq='MS'))
        total_capital_provided = total_months_simulated * monthly_investment
        # Calculate metrics
        scenario_cagr = calculate_cagr(final_portfolio_value, total_capital_provided, num_years)
        avg_buy_size = total_cash_invested / num_buys if num_buys > 0 else 0.0
        weighted_avg_purchase_price = (total_cash_invested / total_units_held) if total_units_held > 0 else 0
        overall_growth_pct = ((final_portfolio_value / total_capital_provided) - 1.0) if total_capital_provided > 0 else 0.0

        summary = {
            'Scenario': sim_name,
            'Description': sim_config.get('description', 'N/A'),
            'Total Capital Provided': total_capital_provided,
            'Total Cash Invested': total_cash_invested,
            'Final Investment Value': final_investment_value,
            'Final Cash Remaining': cash_available,
            'Final Total Portfolio Value': final_portfolio_value,
            'Number of Buys': num_buys,
            'Average Buy Size ($)': avg_buy_size,
            'Overall Growth (%)': overall_growth_pct,
            'Scenario Annualized Growth (%)': scenario_cagr,
            'Total Units Purchased': total_units_held,
            'Weighted Avg Purchase Price': weighted_avg_purchase_price,
            'Final Market Close Price': final_close_price
        }
        all_results_summary.append(summary)
        all_monthly_details[sim_name] = pd.DataFrame(monthly_log)
        all_transaction_details[sim_name] = pd.DataFrame(all_purchases_log) # Store transaction details DF

        # Log final results for this simulation
        sim_logger.info(f"--- Simulation {sim_name} Complete ---")
        sim_logger.info(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        sim_logger.info(f"Total Capital Provided: ${total_capital_provided:,.2f}")
        sim_logger.info(f"Overall Growth: {overall_growth_pct:.2%}")
        sim_logger.info(f"CAGR: {scenario_cagr:.2%}")
        sim_logger.info(f"Total Units: {total_units_held:.4f}, Avg Price: ${weighted_avg_purchase_price:.2f}")
        sim_logger.info(f"Number of Buys: {num_buys}")
        file_handler.close() # Close the log file handler


    # --- Run DCA Scenario 1: Monthly Start ---
    dca1_sim_name = "DCA_Monthly_Start"
    dca1_description = f"Invest fixed amount (${monthly_investment}) on first trading day of each month."
    # (DCA1 Logger setup - similar to above)
    sim_logger = logging.getLogger(dca1_sim_name)
    sim_logger.setLevel(log_level)
    for handler in sim_logger.handlers[:]: sim_logger.removeHandler(handler); handler.close()
    log_file_path = os.path.join(log_dir, f"{dca1_sim_name}.log")
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        sim_logger.addHandler(file_handler)
        sim_logger.propagate = False
        sim_logger.info(f"--- Starting Simulation: {dca1_sim_name} ---")
        sim_logger.info(dca1_description)
    except Exception as e:
        logging.error(f"Failed to set up logger for simulation {dca1_sim_name}: {e}")
        # Decide if you want to stop entirely or just skip this DCA sim
        # For now, we'll log error and continue, results might be incomplete

    # --- DCA1 State Variables ---
    cash_available = 0.0
    total_units_held = 0.0
    total_cash_invested = 0.0
    num_buys = 0
    monthly_log = []
    all_purchases_log_dca1 = [] # NEW: Store transaction details for DCA1

    # --- Simulation Loop (DCA Monthly Start) ---
    for month_start_dt in pd.date_range(actual_start_date, actual_end_date, freq='MS'):
        try:
            current_month_start_actual = data.index[data.index >= month_start_dt].min()
            month_end_target = month_start_dt + pd.offsets.MonthEnd(0)
            current_month_end_actual = data.index[data.index <= month_end_target].max()
            current_month_end_actual = min(current_month_end_actual, actual_end_date)
        except IndexError:
            sim_logger.warning(f"No trading data found for month starting {month_start_dt.strftime('%Y-%m')}. Skipping month.")
            continue

        interest_earned_this_month = 0.0
        if cash_available > 0 and monthly_interest_rate > 0:
            interest_earned_this_month = cash_available * monthly_interest_rate
            cash_available += interest_earned_this_month
            # sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Interest Earned: ${interest_earned_this_month:,.2f}, Cash Now: ${cash_available:,.2f}") # Optional debug

        cash_added_this_month = 0.0
        if month_start_dt >= actual_start_date:
             # For DCA, we assume the investment amount is available *before* the buy attempt
             cash_available += monthly_investment
             cash_added_this_month = monthly_investment
             cash_at_month_start_recorded = cash_available
             # sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Added ${monthly_investment}, Cash Available: ${cash_at_month_start_recorded:,.2f}") # Optional debug
        else:
             cash_at_month_start_recorded = cash_available

        cash_invested_this_month = 0.0
        units_bought_this_month = 0.0
        # purchase_details_this_month = [] # Not needed

        # --- Execute Buy on First Trading Day ---
        try:
            buy_date = current_month_start_actual
            buy_price = data.loc[buy_date, 'Close'] # Buy at the closing price of the first day
            amount_to_invest_this_month = monthly_investment # Target investment

            if amount_to_invest_this_month >= 0.01 and buy_price > 0:
                # Invest the monthly amount, but not more than available cash
                actual_investment = min(amount_to_invest_this_month, cash_available)
                if actual_investment < amount_to_invest_this_month:
                    sim_logger.warning(f"{buy_date.date()}: Insufficient cash (${cash_available:.2f}) for full DCA investment (${amount_to_invest_this_month:.2f}). Investing available amount.")

                if actual_investment >= 0.01: # Ensure we invest something
                    units_bought = actual_investment / buy_price
                    cash_available -= actual_investment
                    total_units_held += units_bought
                    total_cash_invested += actual_investment
                    cash_invested_this_month += actual_investment
                    units_bought_this_month += units_bought
                    num_buys += 1

                    # Log the transaction
                    transaction_log = {
                        'date': buy_date,
                        'price': buy_price,
                        'units': units_bought,
                        'amount': actual_investment,
                        'threshold': 'DCA_Start' # Identifier for this type of buy
                    }
                    all_purchases_log_dca1.append(transaction_log)

                    sim_logger.debug(f"{buy_date.date()}: DCA Buy executed. Investing ${actual_investment:.2f} @ ${buy_price:.2f}. Units: {units_bought:.4f}. Cash left: ${cash_available:.2f}")
                else:
                    sim_logger.debug(f"{buy_date.date()}: Skipping DCA buy, effective investment amount is too small or zero.")
            elif buy_price <= 0:
                sim_logger.warning(f"{buy_date.date()}: Skipping DCA buy, market price (${buy_price:.2f}) is zero or negative.")
            elif amount_to_invest_this_month < 0.01:
                 sim_logger.debug(f"{buy_date.date()}: Skipping DCA buy, monthly investment amount is too small.")

        except KeyError:
            sim_logger.error(f"Could not find price data for DCA buy date {buy_date.date()}. Skipping buy for this month.")
        except Exception as e:
            sim_logger.error(f"Error during DCA buy execution near {month_start_dt.date()}: {e}")

        # --- Month End Calculations & Logging (DCA Monthly Start) ---
        final_close_price_month_end = data['Close'].asof(current_month_end_actual)
        if pd.isna(final_close_price_month_end): final_close_price_month_end = 0
        value_of_investments = total_units_held * final_close_price_month_end
        total_portfolio_value = value_of_investments + cash_available
        avg_purchase_price_month = (cash_invested_this_month / units_bought_this_month) if units_bought_this_month > 0 else 0

        monthly_log_entry = {
             'MonthEnd': current_month_end_actual.strftime('%Y-%m-%d'),
             'CashAdded': cash_added_this_month,
             'InterestEarnedThisMonth': interest_earned_this_month,
             'CashAtMonthStart': cash_at_month_start_recorded,
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
        sim_logger.info(f"Month End {current_month_end_actual.strftime('%Y-%m-%d')}: Portfolio Value: ${total_portfolio_value:,.2f}")

    # --- End of Simulation Calculation (DCA Monthly Start) ---
    final_close_price = market_end_price
    final_investment_value = total_units_held * final_close_price
    final_portfolio_value = final_investment_value + cash_available
    total_months_simulated = len(pd.date_range(start=actual_start_date, end=actual_end_date, freq='MS'))
    total_capital_provided = total_months_simulated * monthly_investment
    scenario_cagr = calculate_cagr(final_portfolio_value, total_capital_provided, num_years)
    avg_buy_size = total_cash_invested / num_buys if num_buys > 0 else 0.0
    weighted_avg_purchase_price = (total_cash_invested / total_units_held) if total_units_held > 0 else 0
    overall_growth_pct = ((final_portfolio_value / total_capital_provided) - 1.0) if total_capital_provided > 0 else 0.0

    summary = {
        'Scenario': dca1_sim_name, 'Description': dca1_description,
        'Total Capital Provided': total_capital_provided, 'Total Cash Invested': total_cash_invested,
        'Final Investment Value': final_investment_value, 'Final Cash Remaining': cash_available,
        'Final Total Portfolio Value': final_portfolio_value,
        'Number of Buys': num_buys, 'Average Buy Size ($)': avg_buy_size,
        'Overall Growth (%)': overall_growth_pct,
        'Scenario Annualized Growth (%)': scenario_cagr,
        'Total Units Purchased': total_units_held, 'Weighted Avg Purchase Price': weighted_avg_purchase_price,
        'Final Market Close Price': final_close_price
    }
    all_results_summary.append(summary)
    all_monthly_details[dca1_sim_name] = pd.DataFrame(monthly_log)
    all_transaction_details[dca1_sim_name] = pd.DataFrame(all_purchases_log_dca1) # Store transaction details

    sim_logger.info(f"--- Simulation {dca1_sim_name} Complete ---")
    sim_logger.info(f"Final Portfolio Value: ${final_portfolio_value:,.2f}") # etc.
    if 'file_handler' in locals() and file_handler: file_handler.close()


    # --- Run DCA Scenario 2: Bi-Monthly 1st and 15th ---
    dca2_sim_name = "DCA_BiMonthly_1_15"
    investment_per_period = monthly_investment / 2.0
    dca2_description = f"Invest half (${investment_per_period:.2f}) on ~1st & ~15th trading day of month."
    # (DCA2 Logger setup - similar to above)
    sim_logger = logging.getLogger(dca2_sim_name)
    sim_logger.setLevel(log_level)
    for handler in sim_logger.handlers[:]: sim_logger.removeHandler(handler); handler.close()
    log_file_path = os.path.join(log_dir, f"{dca2_sim_name}.log")
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        sim_logger.addHandler(file_handler)
        sim_logger.propagate = False
        sim_logger.info(f"--- Starting Simulation: {dca2_sim_name} ---")
        sim_logger.info(dca2_description)
    except Exception as e:
        logging.error(f"Failed to set up logger for simulation {dca2_sim_name}: {e}")

    # --- DCA2 State Variables ---
    cash_available = 0.0
    total_units_held = 0.0
    total_cash_invested = 0.0
    num_buys = 0
    monthly_log = []
    all_purchases_log_dca2 = [] # NEW: Store transaction details for DCA2

    # --- Simulation Loop (DCA Bi-Monthly) ---
    for month_start_dt in pd.date_range(actual_start_date, actual_end_date, freq='MS'):
        try:
            current_month_start_actual = data.index[data.index >= month_start_dt].min()
            month_end_target = month_start_dt + pd.offsets.MonthEnd(0)
            current_month_end_actual = data.index[data.index <= month_end_target].max()
            current_month_end_actual = min(current_month_end_actual, actual_end_date)
            # Get the index for the current month's data
            month_data_index = data.loc[current_month_start_actual:current_month_end_actual].index
        except IndexError:
            sim_logger.warning(f"No trading data found for month starting {month_start_dt.strftime('%Y-%m')}. Skipping month.")
            continue

        interest_earned_this_month = 0.0
        if cash_available > 0 and monthly_interest_rate > 0:
            interest_earned_this_month = cash_available * monthly_interest_rate
            cash_available += interest_earned_this_month
            # sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Interest Earned: ${interest_earned_this_month:,.2f}, Cash Now: ${cash_available:,.2f}")

        cash_added_this_month = 0.0
        if month_start_dt >= actual_start_date:
             cash_available += monthly_investment # Add full monthly amount at start
             cash_added_this_month = monthly_investment
             cash_at_month_start_recorded = cash_available
             # sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Added ${monthly_investment}, Cash Available: ${cash_at_month_start_recorded:,.2f}")
        else:
             cash_at_month_start_recorded = cash_available

        cash_invested_this_month = 0.0
        units_bought_this_month = 0.0
        # purchase_details_this_month = [] # Not needed

        # --- Try to buy near 1st ---
        target_date_1 = month_start_dt
        buy_date_1 = find_next_trading_day(target_date_1, month_data_index) # Search within the month's index
        buy_executed_1 = False # Flag to track if first buy happened

        if buy_date_1:
            try:
                buy_price_1 = data.loc[buy_date_1, 'Close']
                if investment_per_period >= 0.01 and buy_price_1 > 0:
                    actual_investment_1 = min(investment_per_period, cash_available)
                    if actual_investment_1 < investment_per_period and actual_investment_1 >= 0.01:
                        sim_logger.warning(f"{buy_date_1.date()}: Insufficient cash for full 1st DCA buy ({investment_per_period:.2f}). Investing available ${actual_investment_1:.2f}.")

                    if actual_investment_1 >= 0.01:
                        units_bought = actual_investment_1 / buy_price_1
                        cash_available -= actual_investment_1
                        total_units_held += units_bought
                        total_cash_invested += actual_investment_1
                        cash_invested_this_month += actual_investment_1
                        units_bought_this_month += units_bought
                        num_buys += 1

                        transaction_log = {'date': buy_date_1, 'price': buy_price_1, 'units': units_bought, 'amount': actual_investment_1, 'threshold': 'DCA_1st'}
                        all_purchases_log_dca2.append(transaction_log)

                        sim_logger.debug(f"{buy_date_1.date()}: DCA Buy (1st) executed. Investing ${actual_investment_1:.2f} @ ${buy_price_1:.2f}. Units: {units_bought:.4f}. Cash left: ${cash_available:.2f}")
                        buy_executed_1 = True
                    else: sim_logger.debug(f"{buy_date_1.date()}: Skipping 1st DCA buy, amount too small or zero.")
                elif buy_price_1 <= 0: sim_logger.warning(f"{buy_date_1.date()}: Skipping 1st DCA buy, market price <= 0.")
                elif investment_per_period < 0.01: sim_logger.debug(f"{buy_date_1.date()}: Skipping 1st DCA buy, investment per period too small.")
            except KeyError: sim_logger.error(f"Could not find price data for DCA buy date {buy_date_1.date()}. Skipping 1st buy.")
            except Exception as e: sim_logger.error(f"Error during 1st DCA buy execution near {buy_date_1.date()}: {e}")
        else: sim_logger.debug(f"Could not find trading day for 1st DCA buy in {month_start_dt.strftime('%Y-%m')}.")

        # --- Try to buy near 15th ---
        target_date_15 = pd.Timestamp(year=month_start_dt.year, month=month_start_dt.month, day=15)
        # Ensure the target date isn't before the actual start of the month's data
        target_date_15 = max(target_date_15, current_month_start_actual)
        buy_date_15 = find_next_trading_day(target_date_15, month_data_index) # Search within the month's index

        # Execute 15th buy if a valid date is found AND it's not the same day as the 1st buy (unless 1st buy failed)
        if buy_date_15 and (not buy_executed_1 or buy_date_15 != buy_date_1):
            try:
                buy_price_15 = data.loc[buy_date_15, 'Close']
                if investment_per_period >= 0.01 and buy_price_15 > 0:
                    actual_investment_15 = min(investment_per_period, cash_available)
                    if actual_investment_15 < investment_per_period and actual_investment_15 >= 0.01:
                        sim_logger.warning(f"{buy_date_15.date()}: Insufficient cash for full 15th DCA buy ({investment_per_period:.2f}). Investing available ${actual_investment_15:.2f}.")

                    if actual_investment_15 >= 0.01:
                        units_bought = actual_investment_15 / buy_price_15
                        cash_available -= actual_investment_15
                        total_units_held += units_bought
                        total_cash_invested += actual_investment_15
                        cash_invested_this_month += actual_investment_15
                        units_bought_this_month += units_bought
                        num_buys += 1

                        transaction_log = {'date': buy_date_15, 'price': buy_price_15, 'units': units_bought, 'amount': actual_investment_15, 'threshold': 'DCA_15th'}
                        all_purchases_log_dca2.append(transaction_log)

                        sim_logger.debug(f"{buy_date_15.date()}: DCA Buy (15th) executed. Investing ${actual_investment_15:.2f} @ ${buy_price_15:.2f}. Units: {units_bought:.4f}. Cash left: ${cash_available:.2f}")
                    else: sim_logger.debug(f"{buy_date_15.date()}: Skipping 15th DCA buy, amount too small or zero.")
                elif buy_price_15 <= 0: sim_logger.warning(f"{buy_date_15.date()}: Skipping 15th DCA buy, market price <= 0.")
                elif investment_per_period < 0.01: sim_logger.debug(f"{buy_date_15.date()}: Skipping 15th DCA buy, investment per period too small.")
            except KeyError: sim_logger.error(f"Could not find price data for DCA buy date {buy_date_15.date()}. Skipping 15th buy.")
            except Exception as e: sim_logger.error(f"Error during 15th DCA buy execution near {buy_date_15.date()}: {e}")
        elif buy_date_15 and buy_executed_1 and buy_date_15 == buy_date_1:
             sim_logger.debug(f"Skipping 15th DCA buy for {month_start_dt.strftime('%Y-%m')} as it falls on the same day as the 1st buy ({buy_date_15.date()}).")
        elif not buy_date_15:
             sim_logger.debug(f"Could not find trading day for 15th DCA buy in {month_start_dt.strftime('%Y-%m')}.")

        # --- Month End Calculations & Logging (DCA Bi-Monthly) ---
        final_close_price_month_end = data['Close'].asof(current_month_end_actual)
        if pd.isna(final_close_price_month_end): final_close_price_month_end = 0
        value_of_investments = total_units_held * final_close_price_month_end
        total_portfolio_value = value_of_investments + cash_available
        avg_purchase_price_month = (cash_invested_this_month / units_bought_this_month) if units_bought_this_month > 0 else 0

        monthly_log_entry = {
             'MonthEnd': current_month_end_actual.strftime('%Y-%m-%d'),
             'CashAdded': cash_added_this_month,
             'InterestEarnedThisMonth': interest_earned_this_month,
             'CashAtMonthStart': cash_at_month_start_recorded,
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
        sim_logger.info(f"Month End {current_month_end_actual.strftime('%Y-%m-%d')}: Portfolio Value: ${total_portfolio_value:,.2f}")

    # --- End of Simulation Calculation (DCA Bi-Monthly) ---
    final_close_price = market_end_price
    final_investment_value = total_units_held * final_close_price
    final_portfolio_value = final_investment_value + cash_available
    total_months_simulated = len(pd.date_range(start=actual_start_date, end=actual_end_date, freq='MS'))
    total_capital_provided = total_months_simulated * monthly_investment
    scenario_cagr = calculate_cagr(final_portfolio_value, total_capital_provided, num_years)
    avg_buy_size = total_cash_invested / num_buys if num_buys > 0 else 0.0
    weighted_avg_purchase_price = (total_cash_invested / total_units_held) if total_units_held > 0 else 0
    overall_growth_pct = ((final_portfolio_value / total_capital_provided) - 1.0) if total_capital_provided > 0 else 0.0

    summary = {
        'Scenario': dca2_sim_name, 'Description': dca2_description,
        'Total Capital Provided': total_capital_provided, 'Total Cash Invested': total_cash_invested,
        'Final Investment Value': final_investment_value, 'Final Cash Remaining': cash_available,
        'Final Total Portfolio Value': final_portfolio_value,
        'Number of Buys': num_buys, 'Average Buy Size ($)': avg_buy_size,
        'Overall Growth (%)': overall_growth_pct,
        'Scenario Annualized Growth (%)': scenario_cagr,
        'Total Units Purchased': total_units_held, 'Weighted Avg Purchase Price': weighted_avg_purchase_price,
        'Final Market Close Price': final_close_price
    }
    all_results_summary.append(summary)
    all_monthly_details[dca2_sim_name] = pd.DataFrame(monthly_log)
    all_transaction_details[dca2_sim_name] = pd.DataFrame(all_purchases_log_dca2) # Store transaction details

    sim_logger.info(f"--- Simulation {dca2_sim_name} Complete ---")
    sim_logger.info(f"Final Portfolio Value: ${final_portfolio_value:,.2f}") # etc.
    if 'file_handler' in locals() and file_handler: file_handler.close()


    # --- Write results to Excel using xlsxwriter engine ---
    logging.info(f"Writing results to {output_file} using xlsxwriter engine for number formatting.")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir: # Check if path includes a directory
            os.makedirs(output_dir, exist_ok=True)

        with pd.ExcelWriter(output_file, engine='xlsxwriter',
                            datetime_format='yyyy-mm-dd', # Set default date format
                            date_format='yyyy-mm-dd') as writer:
            # --- Write Summary Sheet ---
            summary_df = pd.DataFrame(all_results_summary)
            # Sort by performance (handle potential missing column)
            if 'Overall Growth (%)' in summary_df.columns:
                 summary_df.sort_values(by='Overall Growth (%)', ascending=False, inplace=True)
            else: logging.warning("Could not sort summary results by 'Overall Growth (%)'.")

            summary_sheet_name = 'Summary'
            # Define column order for summary sheet
            summary_cols_order = [
                'Scenario', 'Description', 'Total Capital Provided', 'Total Cash Invested',
                'Final Investment Value', 'Final Cash Remaining', 'Final Total Portfolio Value',
                'Number of Buys', 'Average Buy Size ($)',
                'Total Units Purchased', 'Weighted Avg Purchase Price',
                'Overall Growth (%)', 'Scenario Annualized Growth (%)',
                'Final Market Close Price'
            ]
            # Filter out any columns that might not exist (e.g., if a sim failed)
            summary_cols_order = [col for col in summary_cols_order if col in summary_df.columns]
            summary_df = summary_df[summary_cols_order] # Reorder columns
            summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)

            # --- Apply Formatting to Summary Sheet ---
            workbook = writer.book
            summary_worksheet = writer.sheets[summary_sheet_name]

            # Define formats
            currency_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            units_format = workbook.add_format({'num_format': '#,##0.0000'}) # 4 decimal places for units
            integer_format = workbook.add_format({'num_format': '#,##0'})
            header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1, 'align': 'center'})

            # Map column names to indices for easy formatting application
            summary_col_map = {name: i for i, name in enumerate(summary_df.columns)}

            # Define which columns get which format
            summary_currency_cols = ['Total Capital Provided', 'Total Cash Invested', 'Final Investment Value', 'Final Cash Remaining', 'Final Total Portfolio Value', 'Average Buy Size ($)', 'Weighted Avg Purchase Price', 'Final Market Close Price']
            summary_unit_cols = ['Total Units Purchased']
            summary_percent_cols = ['Overall Growth (%)', 'Scenario Annualized Growth (%)']
            summary_integer_cols = ['Number of Buys']

            # Apply formats to columns
            for col_name in summary_currency_cols:
                if col_name in summary_col_map: summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 18, currency_format)
            for col_name in summary_unit_cols:
                 if col_name in summary_col_map: summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 18, units_format)
            for col_name in summary_percent_cols:
                 if col_name in summary_col_map: summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 15, percent_format)
            for col_name in summary_integer_cols:
                 if col_name in summary_col_map: summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 12, integer_format)

            # Format description column
            if 'Description' in summary_col_map: summary_worksheet.set_column(summary_col_map['Description'], summary_col_map['Description'], 30)
            if 'Scenario' in summary_col_map: summary_worksheet.set_column(summary_col_map['Scenario'], summary_col_map['Scenario'], 25)


            # Apply header format
            for col_num, value in enumerate(summary_df.columns.values):
                summary_worksheet.write(0, col_num, value, header_format)
            summary_worksheet.freeze_panes(1, 0) # Freeze header row


            # --- Write Detailed Monthly Sheets ---
            for sim_name, monthly_df in all_monthly_details.items():
                # Sanitize sheet name (max 31 chars, no invalid chars)
                safe_sheet_name = "".join(c for c in sim_name if c.isalnum() or c in (' ', '_', '-'))[:31]
                monthly_sheet_name = f"{safe_sheet_name}_Monthly"

                if isinstance(monthly_df, pd.DataFrame) and not monthly_df.empty:
                     # Add InterestEarnedThisMonth if it exists, otherwise add as 0
                     if 'InterestEarnedThisMonth' not in monthly_df.columns:
                         monthly_df['InterestEarnedThisMonth'] = 0.0

                     # Define column order for monthly sheet
                     monthly_cols_order = [
                         'MonthEnd', 'CashAdded', 'InterestEarnedThisMonth', 'CashAtMonthStart',
                         'CashInvestedThisMonth', 'AvgPurchasePriceThisMonth', 'UnitsBoughtThisMonth',
                         'CumulativeUnitsHeld', 'CumulativeCashInvested', 'MarketClosePriceMonthEnd',
                         'ValueInvestmentsMonthEnd', 'CashOnHandMonthEnd', 'TotalPortfolioValueMonthEnd'
                     ]
                     # Filter and reorder
                     monthly_cols_order = [col for col in monthly_cols_order if col in monthly_df.columns]
                     monthly_df = monthly_df[monthly_cols_order]

                     monthly_df.to_excel(writer, sheet_name=monthly_sheet_name, index=False)
                     monthly_worksheet = writer.sheets[monthly_sheet_name]
                     monthly_col_map = {name: i for i, name in enumerate(monthly_df.columns)}

                     # Define formats for monthly sheet
                     monthly_currency_cols = ['CashAdded', 'InterestEarnedThisMonth', 'CashAtMonthStart', 'CashInvestedThisMonth', 'AvgPurchasePriceThisMonth', 'CumulativeCashInvested', 'MarketClosePriceMonthEnd', 'ValueInvestmentsMonthEnd', 'CashOnHandMonthEnd', 'TotalPortfolioValueMonthEnd']
                     monthly_units_cols = ['UnitsBoughtThisMonth', 'CumulativeUnitsHeld']
                     monthly_date_cols = ['MonthEnd']

                     # Apply formats
                     for col_name in monthly_currency_cols:
                          if col_name in monthly_col_map: monthly_worksheet.set_column(monthly_col_map[col_name], monthly_col_map[col_name], 18, currency_format)
                     for col_name in monthly_units_cols:
                          if col_name in monthly_col_map: monthly_worksheet.set_column(monthly_col_map[col_name], monthly_col_map[col_name], 18, units_format)
                     # Date format is handled by ExcelWriter setting, but can set width
                     for col_name in monthly_date_cols:
                          if col_name in monthly_col_map: monthly_worksheet.set_column(monthly_col_map[col_name], monthly_col_map[col_name], 12)


                     # Apply header format
                     for col_num, value in enumerate(monthly_df.columns.values):
                         monthly_worksheet.write(0, col_num, value, header_format)
                     monthly_worksheet.freeze_panes(1, 0) # Freeze header row
                else:
                     logging.warning(f"Skipping monthly sheet for '{sim_name}' because data is empty or not a DataFrame.")

            # --- Write Detailed Transaction Sheets --- NEW SECTION ---
            for sim_name, transactions_df in all_transaction_details.items():
                safe_sheet_name = "".join(c for c in sim_name if c.isalnum() or c in (' ', '_', '-'))[:31]
                transactions_sheet_name = f"{safe_sheet_name}_Buys"

                if isinstance(transactions_df, pd.DataFrame) and not transactions_df.empty:
                    # Define column order
                    trans_cols_order = ['date', 'price', 'units', 'amount', 'threshold']
                    trans_cols_order = [col for col in trans_cols_order if col in transactions_df.columns]
                    transactions_df = transactions_df[trans_cols_order]

                    transactions_df.to_excel(writer, sheet_name=transactions_sheet_name, index=False)
                    trans_worksheet = writer.sheets[transactions_sheet_name]
                    trans_col_map = {name: i for i, name in enumerate(transactions_df.columns)}

                    # Define formats for transaction sheet
                    trans_currency_cols = ['price', 'amount']
                    trans_units_cols = ['units']
                    trans_date_cols = ['date']
                    trans_other_cols = ['threshold'] # e.g., percentage or DCA type

                    # Apply formats
                    for col_name in trans_currency_cols:
                        if col_name in trans_col_map: trans_worksheet.set_column(trans_col_map[col_name], trans_col_map[col_name], 15, currency_format)
                    for col_name in trans_units_cols:
                        if col_name in trans_col_map: trans_worksheet.set_column(trans_col_map[col_name], trans_col_map[col_name], 18, units_format)
                    for col_name in trans_date_cols:
                        if col_name in trans_col_map: trans_worksheet.set_column(trans_col_map[col_name], trans_col_map[col_name], 12) # Date format handled by writer
                    for col_name in trans_other_cols:
                        if col_name in trans_col_map: trans_worksheet.set_column(trans_col_map[col_name], trans_col_map[col_name], 10) # Adjust width as needed

                    # Apply header format
                    for col_num, value in enumerate(transactions_df.columns.values):
                        trans_worksheet.write(0, col_num, value, header_format)
                    trans_worksheet.freeze_panes(1, 0) # Freeze header row
                else:
                    logging.info(f"No buy transactions recorded for '{sim_name}'. Skipping transaction detail sheet.")


        logging.info(f"Results successfully written to {output_file}")
    except ImportError:
        logging.error("Error: 'xlsxwriter' engine required but not installed. Please run: pip install xlsxwriter")
        # Optionally, write to CSV as a fallback?
        # summary_df.to_csv(output_file.replace('.xlsx', '_summary.csv'), index=False)
        # logging.info(f"Results summary written to CSV as fallback.")
    except Exception as e:
        logging.error(f"Error writing results to Excel: {e}")
        import traceback
        logging.error(traceback.format_exc()) # Log full traceback for debugging

    return pd.DataFrame(all_results_summary)


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting Investment Simulator Script")
    config = load_config()
    if config:
        data = load_and_prepare_data(config)
        if data is not None:
            results_summary_df = run_simulation(config, data)
            if results_summary_df is not None and not results_summary_df.empty:
                logging.info("\n--- Overall Simulation Summary ---")
                # Display summary in a readable format in the console log
                try:
                    # Select key columns for console output
                    cols_to_print = ['Scenario', 'Final Total Portfolio Value', 'Overall Growth (%)', 'Scenario Annualized Growth (%)', 'Number of Buys', 'Weighted Avg Purchase Price']
                    cols_to_print = [col for col in cols_to_print if col in results_summary_df.columns]
                    print(results_summary_df[cols_to_print].to_string(index=False, float_format='{:,.2f}'.format)) # Basic formatting for console
                except Exception as e:
                    logging.warning(f"Could not print formatted summary to console: {e}")
                    # print(results_summary_df.to_string(index=False)) # Fallback to default print
                logging.info(f"Detailed results saved to: {config['output_file']}")
                logging.info(f"Individual simulation logs saved in: {os.path.abspath('logs/')}")
            elif results_summary_df is not None and results_summary_df.empty:
                 logging.warning("Simulation completed, but no summary results were generated (possibly no simulations defined or all failed).")
            else:
                logging.error("Simulation run failed to produce results.")
        else:
            logging.error("Data loading and preparation failed. Aborting simulation.")
    else:
        logging.error("Configuration loading failed. Aborting.")
    logging.info("Investment Simulator Script Finished")

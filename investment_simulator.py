import pandas as pd
import yaml
import os
import logging
import logging.handlers # For file logging
import numpy as np # For CAGR calculation

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
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        data_start_date = df.index.min()
        data_end_date = df.index.max()
        effective_start = max(start_date, data_start_date)
        effective_end = min(end_date, data_end_date)

        df = df.loc[effective_start:effective_end].copy()

        if df.empty:
            logging.error(f"No valid data found for the specified date range: {effective_start.date()} to {effective_end.date()}")
            return None

        df['PeakHigh'] = df['High'].expanding().max()

        logging.info(f"Data loaded and prepared. Shape: {df.shape}. Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {data_file}")
        return None
    except KeyError as e:
         logging.error(f"Error: Column mapping issue. Original error detail: {e}")
         return None
    except ValueError as e:
         logging.error(f"Error converting price data to numeric. Original error detail: {e}")
         return None
    except Exception as e:
        logging.error(f"Error processing data file: {e}")
        return None

# --- CAGR Calculation Helper ---
def calculate_cagr(end_value, start_value, years):
    """Calculates Compound Annual Growth Rate (CAGR) for a strategy."""
    if years <= 0 or start_value <= 0 or end_value <= 0:
        return 0.0
    return (float(end_value) / float(start_value)) ** (1.0 / years) - 1.0

# --- Helper to find next trading day ---
def find_next_trading_day(target_date, data_index):
    """Finds the first trading day index on or after target_date."""
    try:
        idx_pos = data_index.searchsorted(target_date)
        if idx_pos < len(data_index):
            return data_index[idx_pos]
        else: return None
    except Exception: return None

# --- Simulation Execution ---
def run_simulation(config, data):
    """Runs investment simulations including cash interest, config scenarios and two DCA variants."""
    if data is None or data.empty:
        logging.error("Simulation cannot run without valid data.")
        return None

    monthly_investment = config['monthly_investment']
    actual_start_date = data.index.min()
    actual_end_date = data.index.max()
    simulations = config.get('simulations', {})
    output_file = config['output_file']
    log_level_str = config.get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    # Get cash interest rate
    cash_interest_rate_annual = config.get('cash_interest_rate_annual', 0.0)
    monthly_interest_rate = cash_interest_rate_annual / 100.0 / 12.0
    logging.info(f"Applying monthly interest rate to uninvested cash: {monthly_interest_rate:.4%}")


    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Per-simulation logs will be saved in '{log_dir}/' directory.")

    all_results_summary = []
    all_monthly_details = {}

    # --- Calculate overall simulation period details ---
    num_years = (actual_end_date - actual_start_date).days / 365.25
    if num_years <= 0: num_years = np.nan

    market_start_price = data['Close'].iloc[0]
    market_end_price = data['Close'].iloc[-1]

    logging.info(f"Simulation Period: {actual_start_date.date()} to {actual_end_date.date()} ({num_years:.2f} years)")
    logging.info(f"Market Start Price ({actual_start_date.date()}): {market_start_price:,.2f}")
    logging.info(f"Market End Price ({actual_end_date.date()}): {market_end_price:,.2f}")

    # --- Run Config-Based Simulations ---
    for sim_name, sim_config in simulations.items():
        # (Logger setup)
        sim_logger = logging.getLogger(sim_name)
        sim_logger.setLevel(log_level)
        for handler in sim_logger.handlers[:]: sim_logger.removeHandler(handler); handler.close()
        log_file_path = os.path.join(log_dir, f"{sim_name}.log")
        file_handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        sim_logger.addHandler(file_handler)
        sim_logger.propagate = False
        sim_logger.info(f"--- Starting Simulation: {sim_name} ---") # etc...

        allocations = sim_config['allocations']
        sorted_thresholds = sorted(allocations.keys())
        cash_available = 0.0
        total_units_held = 0.0
        total_cash_invested = 0.0
        num_buys = 0
        investments = []
        monthly_log = []
        last_peak_high = 0
        triggered_pullbacks_since_peak = set()

        # --- Simulation Loop (Config Scenario) ---
        for month_start_dt in pd.date_range(actual_start_date, actual_end_date, freq='MS'):
            current_month_start_actual = data.index[data.index >= month_start_dt].min()
            month_end_target = month_start_dt + pd.offsets.MonthEnd(0)
            current_month_end_actual = data.index[data.index <= month_end_target].max()
            current_month_end_actual = min(current_month_end_actual, actual_end_date)

            interest_earned_this_month = 0.0
            # Apply interest *before* adding monthly investment
            if cash_available > 0 and monthly_interest_rate > 0:
                interest_earned_this_month = cash_available * monthly_interest_rate
                cash_available += interest_earned_this_month
                sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Interest Earned: ${interest_earned_this_month:,.2f}, Cash Now: ${cash_available:,.2f}")

            # Add monthly investment
            if month_start_dt >= actual_start_date:
                 cash_available += monthly_investment
                 cash_at_month_start = cash_available # Capture cash *after* adding interest and monthly amount
                 sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Added ${monthly_investment}, Cash Available: ${cash_at_month_start:,.2f}")
            else:
                 cash_at_month_start = cash_available # Capture cash after interest if start date is mid-month

            month_data = data.loc[current_month_start_actual:current_month_end_actual]
            cash_invested_this_month = 0.0
            units_bought_this_month = 0.0
            purchase_details_this_month = []

            # (Buy logic loop remains the same)
            if not month_data.empty:
                for date, row in month_data.iterrows():
                    current_high = row['High']
                    current_low = row['Low']
                    current_peak_high = row['PeakHigh']
                    if current_peak_high > last_peak_high:
                        if last_peak_high > 0: sim_logger.debug(f"{date.date()}: New peak: {current_peak_high:.2f}. Resetting triggers.")
                        last_peak_high = current_peak_high
                        triggered_pullbacks_since_peak = set()
                    for threshold in sorted_thresholds:
                        if last_peak_high <= 0: continue
                        pullback_target_price = last_peak_high * (1 - threshold / 100.0)
                        if current_low <= pullback_target_price and threshold not in triggered_pullbacks_since_peak:
                            allocation_pct = allocations[threshold] / 100.0
                            amount_to_invest = cash_available * allocation_pct
                            if amount_to_invest > 0.01 and current_low > 0:
                                units_bought = amount_to_invest / current_low
                                purchase_price = current_low
                                cash_available -= amount_to_invest
                                total_units_held += units_bought
                                total_cash_invested += amount_to_invest
                                cash_invested_this_month += amount_to_invest
                                units_bought_this_month += units_bought
                                num_buys += 1
                                purchase_details_this_month.append({'date': date, 'price': purchase_price, 'units': units_bought, 'amount': amount_to_invest, 'threshold': threshold})
                                investments.append({'date': date, 'units': units_bought, 'price': purchase_price, 'amount': amount_to_invest})
                                triggered_pullbacks_since_peak.add(threshold)
                                sim_logger.debug(f"{date.date()}: Buy triggered! Investing ${amount_to_invest:.2f}. Cash left: ${cash_available:.2f}")
                            elif current_low <= 0: sim_logger.warning(f"{date.date()}: Skipping buy for {threshold}% - Low price <= 0.")

            # --- Month End Calculations & Logging ---
            final_close_price_month_end = data['Close'].asof(current_month_end_actual)
            if pd.isna(final_close_price_month_end): final_close_price_month_end = 0
            value_of_investments = total_units_held * final_close_price_month_end
            total_portfolio_value = value_of_investments + cash_available
            avg_purchase_price_month = (sum(p['amount'] for p in purchase_details_this_month) / sum(p['units'] for p in purchase_details_this_month)) if units_bought_this_month > 0 else 0

            monthly_log_entry = {
                'MonthEnd': current_month_end_actual.strftime('%Y-%m-%d'),
                'CashAdded': monthly_investment if month_start_dt >= actual_start_date else 0,
                'InterestEarnedThisMonth': interest_earned_this_month, # New column
                'CashAtMonthStart': cash_at_month_start, # Note: This now includes interest earned before adding capital
                'CashInvestedThisMonth': cash_invested_this_month, 'AvgPurchasePriceThisMonth': avg_purchase_price_month,
                'UnitsBoughtThisMonth': units_bought_this_month, 'CumulativeUnitsHeld': total_units_held,
                'CumulativeCashInvested': total_cash_invested, 'MarketClosePriceMonthEnd': final_close_price_month_end,
                'ValueInvestmentsMonthEnd': value_of_investments, 'CashOnHandMonthEnd': cash_available,
                'TotalPortfolioValueMonthEnd': total_portfolio_value
            }
            monthly_log.append(monthly_log_entry)
            sim_logger.info(f"Month End {current_month_end_actual.strftime('%Y-%m-%d')}: Portfolio Value: ${total_portfolio_value:,.2f}")
            sim_logger.debug(f"Month End Details: {monthly_log_entry}")

        # --- End of Simulation Calculation (Config Scenario) ---
        final_close_price = market_end_price
        final_investment_value = total_units_held * final_close_price
        final_portfolio_value = final_investment_value + cash_available
        total_months_simulated = len(pd.date_range(start=actual_start_date, end=actual_end_date, freq='MS'))
        total_capital_provided = total_months_simulated * monthly_investment
        scenario_cagr = calculate_cagr(final_portfolio_value, total_capital_provided, num_years) if num_years > 0 else 0.0
        avg_buy_size = total_cash_invested / num_buys if num_buys > 0 else 0.0
        weighted_avg_purchase_price = (total_cash_invested / total_units_held) if total_units_held > 0 else 0
        overall_growth_pct = ((final_portfolio_value / total_capital_provided) - 1) if total_capital_provided > 0 else 0.0

        summary = { # Market CAGR removed
            'Scenario': sim_name, 'Description': sim_config['description'],
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
        all_monthly_details[sim_name] = pd.DataFrame(monthly_log)
        # (Logging final results)
        sim_logger.info(f"--- Simulation {sim_name} Complete ---") # etc...
        file_handler.close()


    # --- Run DCA Scenario 1: Monthly Start ---
    dca1_sim_name = "DCA_Monthly_Start"
    dca1_description = f"Invest fixed amount (${monthly_investment}) on first trading day of each month."
    # (DCA1 Logger setup)
    sim_logger = logging.getLogger(dca1_sim_name)
    sim_logger.setLevel(log_level)
    for handler in sim_logger.handlers[:]: sim_logger.removeHandler(handler); handler.close()
    log_file_path = os.path.join(log_dir, f"{dca1_sim_name}.log")
    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    sim_logger.addHandler(file_handler)
    sim_logger.propagate = False
    sim_logger.info(f"--- Starting Simulation: {dca1_sim_name} ---") # etc...

    cash_available = 0.0
    total_units_held = 0.0
    total_cash_invested = 0.0
    num_buys = 0
    investments = []
    monthly_log = []

    # --- Simulation Loop (DCA Monthly Start) ---
    for month_start_dt in pd.date_range(actual_start_date, actual_end_date, freq='MS'):
        current_month_start_actual = data.index[data.index >= month_start_dt].min()
        month_end_target = month_start_dt + pd.offsets.MonthEnd(0)
        current_month_end_actual = data.index[data.index <= month_end_target].max()
        current_month_end_actual = min(current_month_end_actual, actual_end_date)

        interest_earned_this_month = 0.0
        # Apply interest *before* adding monthly investment
        if cash_available > 0 and monthly_interest_rate > 0:
            interest_earned_this_month = cash_available * monthly_interest_rate
            cash_available += interest_earned_this_month
            sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Interest Earned: ${interest_earned_this_month:,.2f}, Cash Now: ${cash_available:,.2f}")

        # Add monthly investment
        if month_start_dt >= actual_start_date:
             cash_available += monthly_investment
             cash_at_month_start = cash_available
             sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Added ${monthly_investment}, Cash Available: ${cash_at_month_start:,.2f}")
        else:
             cash_at_month_start = cash_available

        amount_to_invest_this_month = monthly_investment
        cash_invested_this_month = 0.0
        units_bought_this_month = 0.0
        purchase_details_this_month = []

        try: # Find and execute buy on first trading day
            buy_date = current_month_start_actual
            buy_price = data.loc[buy_date, 'Close']
            if amount_to_invest_this_month > 0.01 and buy_price > 0:
                actual_investment = min(amount_to_invest_this_month, cash_available)
                if actual_investment < amount_to_invest_this_month: sim_logger.warning(f"{buy_date.date()}: Insufficient cash for DCA.")
                if actual_investment > 0.01:
                    units_bought = actual_investment / buy_price
                    cash_available -= actual_investment
                    total_units_held += units_bought
                    total_cash_invested += actual_investment
                    cash_invested_this_month += actual_investment
                    units_bought_this_month += units_bought
                    num_buys += 1
                    purchase_details_this_month.append({'date': buy_date, 'price': buy_price, 'units': units_bought, 'amount': actual_investment, 'threshold': 'DCA'})
                    investments.append({'date': buy_date, 'units': units_bought, 'price': buy_price, 'amount': actual_investment})
                    sim_logger.debug(f"{buy_date.date()}: DCA Buy executed. Investing ${actual_investment:.2f} at ${buy_price:.2f}.")
                else: sim_logger.debug(f"{buy_date.date()}: Skipping DCA buy, amount too small.")
            elif buy_price <= 0: sim_logger.warning(f"{buy_date.date()}: Skipping DCA buy, market price <= 0.")
        except Exception as e: sim_logger.error(f"Error during DCA buy execution near {month_start_dt.date()}: {e}")

        # --- Month End Calculations & Logging (DCA Monthly Start) ---
        final_close_price_month_end = data['Close'].asof(current_month_end_actual)
        if pd.isna(final_close_price_month_end): final_close_price_month_end = 0
        value_of_investments = total_units_held * final_close_price_month_end
        total_portfolio_value = value_of_investments + cash_available
        avg_purchase_price_month = (sum(p['amount'] for p in purchase_details_this_month) / sum(p['units'] for p in purchase_details_this_month)) if units_bought_this_month > 0 else 0
        monthly_log_entry = {
             'MonthEnd': current_month_end_actual.strftime('%Y-%m-%d'),
             'CashAdded': monthly_investment if month_start_dt >= actual_start_date else 0,
             'InterestEarnedThisMonth': interest_earned_this_month, # New column
             'CashAtMonthStart': cash_at_month_start,
             'CashInvestedThisMonth': cash_invested_this_month, 'AvgPurchasePriceThisMonth': avg_purchase_price_month,
             'UnitsBoughtThisMonth': units_bought_this_month, 'CumulativeUnitsHeld': total_units_held, 'CumulativeCashInvested': total_cash_invested,
             'MarketClosePriceMonthEnd': final_close_price_month_end, 'ValueInvestmentsMonthEnd': value_of_investments, 'CashOnHandMonthEnd': cash_available,
             'TotalPortfolioValueMonthEnd': total_portfolio_value
        }
        monthly_log.append(monthly_log_entry)
        sim_logger.info(f"Month End {current_month_end_actual.strftime('%Y-%m-%d')}: Portfolio Value: ${total_portfolio_value:,.2f}")
        sim_logger.debug(f"Month End Details: {monthly_log_entry}")

    # --- End of Simulation Calculation (DCA Monthly Start) ---
    final_close_price = market_end_price
    final_investment_value = total_units_held * final_close_price
    final_portfolio_value = final_investment_value + cash_available
    total_months_simulated = len(pd.date_range(start=actual_start_date, end=actual_end_date, freq='MS'))
    total_capital_provided = total_months_simulated * monthly_investment
    scenario_cagr = calculate_cagr(final_portfolio_value, total_capital_provided, num_years) if num_years > 0 else 0.0
    avg_buy_size = total_cash_invested / num_buys if num_buys > 0 else 0.0
    weighted_avg_purchase_price = (total_cash_invested / total_units_held) if total_units_held > 0 else 0
    overall_growth_pct = ((final_portfolio_value / total_capital_provided) - 1) if total_capital_provided > 0 else 0.0

    summary = { # Market CAGR removed
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
    sim_logger.info(f"--- Simulation {dca1_sim_name} Complete ---") # etc...
    file_handler.close()


    # --- Run DCA Scenario 2: Bi-Monthly 1st and 15th ---
    dca2_sim_name = "DCA_BiMonthly_1_15"
    dca2_description = f"Invest half (${monthly_investment/2:.2f}) on 1st & 15th trading day of month."
    # (DCA2 Logger setup)
    sim_logger = logging.getLogger(dca2_sim_name)
    sim_logger.setLevel(log_level)
    for handler in sim_logger.handlers[:]: sim_logger.removeHandler(handler); handler.close()
    log_file_path = os.path.join(log_dir, f"{dca2_sim_name}.log")
    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    sim_logger.addHandler(file_handler)
    sim_logger.propagate = False
    sim_logger.info(f"--- Starting Simulation: {dca2_sim_name} ---") # etc...

    cash_available = 0.0
    total_units_held = 0.0
    total_cash_invested = 0.0
    num_buys = 0
    investments = []
    monthly_log = []

    # --- Simulation Loop (DCA Bi-Monthly) ---
    investment_per_period = monthly_investment / 2.0

    for month_start_dt in pd.date_range(actual_start_date, actual_end_date, freq='MS'):
        current_month_start_actual = data.index[data.index >= month_start_dt].min()
        month_end_target = month_start_dt + pd.offsets.MonthEnd(0)
        current_month_end_actual = data.index[data.index <= month_end_target].max()
        current_month_end_actual = min(current_month_end_actual, actual_end_date)

        interest_earned_this_month = 0.0
        # Apply interest *before* adding monthly investment
        if cash_available > 0 and monthly_interest_rate > 0:
            interest_earned_this_month = cash_available * monthly_interest_rate
            cash_available += interest_earned_this_month
            sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Interest Earned: ${interest_earned_this_month:,.2f}, Cash Now: ${cash_available:,.2f}")

        # Add monthly investment
        if month_start_dt >= actual_start_date:
             cash_available += monthly_investment
             cash_at_month_start = cash_available
             sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')}: Added ${monthly_investment}, Cash Available: ${cash_at_month_start:,.2f}")
        else:
             cash_at_month_start = cash_available

        cash_invested_this_month = 0.0
        units_bought_this_month = 0.0
        purchase_details_this_month = []

        # --- Try to buy near 1st ---
        target_date_1 = month_start_dt
        buy_date_1 = find_next_trading_day(target_date_1, data.loc[current_month_start_actual:current_month_end_actual].index)
        buy_executed_1 = False # Flag to track if first buy happened

        if buy_date_1:
            buy_price_1 = data.loc[buy_date_1, 'Close']
            if investment_per_period > 0.01 and buy_price_1 > 0:
                actual_investment_1 = min(investment_per_period, cash_available)
                if actual_investment_1 < investment_per_period and actual_investment_1 > 0.01: sim_logger.warning(f"{buy_date_1.date()}: Insufficient cash for full 1st DCA buy.")
                if actual_investment_1 > 0.01:
                    units_bought = actual_investment_1 / buy_price_1
                    cash_available -= actual_investment_1
                    total_units_held += units_bought
                    total_cash_invested += actual_investment_1
                    cash_invested_this_month += actual_investment_1
                    units_bought_this_month += units_bought
                    num_buys += 1
                    purchase_details_this_month.append({'date': buy_date_1, 'price': buy_price_1, 'units': units_bought, 'amount': actual_investment_1, 'threshold': 'DCA_1'})
                    investments.append({'date': buy_date_1, 'units': units_bought, 'price': buy_price_1, 'amount': actual_investment_1})
                    sim_logger.debug(f"{buy_date_1.date()}: DCA Buy (1st) executed. Investing ${actual_investment_1:.2f} at ${buy_price_1:.2f}.")
                    buy_executed_1 = True
                else: sim_logger.debug(f"{buy_date_1.date()}: Skipping 1st DCA buy, amount too small or zero.")
            elif buy_price_1 <= 0: sim_logger.warning(f"{buy_date_1.date()}: Skipping 1st DCA buy, market price <= 0.")
        else: sim_logger.debug(f"Could not find trading day for 1st DCA buy in {month_start_dt.strftime('%Y-%m')}.")

        # --- Try to buy near 15th ---
        target_date_15 = pd.Timestamp(year=month_start_dt.year, month=month_start_dt.month, day=15)
        target_date_15 = max(target_date_15, current_month_start_actual)
        buy_date_15 = find_next_trading_day(target_date_15, data.loc[current_month_start_actual:current_month_end_actual].index)

        if buy_date_15 and (not buy_executed_1 or buy_date_15 != buy_date_1): # Ensure different day if first buy happened
            buy_price_15 = data.loc[buy_date_15, 'Close']
            if investment_per_period > 0.01 and buy_price_15 > 0:
                actual_investment_15 = min(investment_per_period, cash_available)
                if actual_investment_15 < investment_per_period and actual_investment_15 > 0.01: sim_logger.warning(f"{buy_date_15.date()}: Insufficient cash for full 15th DCA buy.")
                if actual_investment_15 > 0.01:
                    units_bought = actual_investment_15 / buy_price_15
                    cash_available -= actual_investment_15
                    total_units_held += units_bought
                    total_cash_invested += actual_investment_15
                    cash_invested_this_month += actual_investment_15
                    units_bought_this_month += units_bought
                    num_buys += 1
                    purchase_details_this_month.append({'date': buy_date_15, 'price': buy_price_15, 'units': units_bought, 'amount': actual_investment_15, 'threshold': 'DCA_15'})
                    investments.append({'date': buy_date_15, 'units': units_bought, 'price': buy_price_15, 'amount': actual_investment_15})
                    sim_logger.debug(f"{buy_date_15.date()}: DCA Buy (15th) executed. Investing ${actual_investment_15:.2f} at ${buy_price_15:.2f}.")
                else: sim_logger.debug(f"{buy_date_15.date()}: Skipping 15th DCA buy, amount too small or zero.")
            elif buy_price_15 <= 0: sim_logger.warning(f"{buy_date_15.date()}: Skipping 15th DCA buy, market price <= 0.")
        elif buy_date_15 and buy_executed_1 and buy_date_15 == buy_date_1:
             sim_logger.debug(f"Skipping 15th DCA buy for {month_start_dt.strftime('%Y-%m')} as it falls on the same day as the 1st buy ({buy_date_15.date()}).")
        elif not buy_date_15:
             sim_logger.debug(f"Could not find trading day for 15th DCA buy in {month_start_dt.strftime('%Y-%m')}.")

        # --- Month End Calculations & Logging (DCA Bi-Monthly) ---
        final_close_price_month_end = data['Close'].asof(current_month_end_actual)
        if pd.isna(final_close_price_month_end): final_close_price_month_end = 0
        value_of_investments = total_units_held * final_close_price_month_end
        total_portfolio_value = value_of_investments + cash_available
        avg_purchase_price_month = (sum(p['amount'] for p in purchase_details_this_month) / sum(p['units'] for p in purchase_details_this_month)) if units_bought_this_month > 0 else 0
        monthly_log_entry = {
             'MonthEnd': current_month_end_actual.strftime('%Y-%m-%d'),
             'CashAdded': monthly_investment if month_start_dt >= actual_start_date else 0,
             'InterestEarnedThisMonth': interest_earned_this_month, # New column
             'CashAtMonthStart': cash_at_month_start,
             'CashInvestedThisMonth': cash_invested_this_month, 'AvgPurchasePriceThisMonth': avg_purchase_price_month,
             'UnitsBoughtThisMonth': units_bought_this_month, 'CumulativeUnitsHeld': total_units_held, 'CumulativeCashInvested': total_cash_invested,
             'MarketClosePriceMonthEnd': final_close_price_month_end, 'ValueInvestmentsMonthEnd': value_of_investments, 'CashOnHandMonthEnd': cash_available,
             'TotalPortfolioValueMonthEnd': total_portfolio_value
        }
        monthly_log.append(monthly_log_entry)
        sim_logger.info(f"Month End {current_month_end_actual.strftime('%Y-%m-%d')}: Portfolio Value: ${total_portfolio_value:,.2f}")
        sim_logger.debug(f"Month End Details: {monthly_log_entry}")

    # --- End of Simulation Calculation (DCA Bi-Monthly) ---
    final_close_price = market_end_price
    final_investment_value = total_units_held * final_close_price
    final_portfolio_value = final_investment_value + cash_available
    total_months_simulated = len(pd.date_range(start=actual_start_date, end=actual_end_date, freq='MS'))
    total_capital_provided = total_months_simulated * monthly_investment
    scenario_cagr = calculate_cagr(final_portfolio_value, total_capital_provided, num_years) if num_years > 0 else 0.0
    avg_buy_size = total_cash_invested / num_buys if num_buys > 0 else 0.0
    weighted_avg_purchase_price = (total_cash_invested / total_units_held) if total_units_held > 0 else 0
    overall_growth_pct = ((final_portfolio_value / total_capital_provided) - 1) if total_capital_provided > 0 else 0.0

    summary = { # Market CAGR removed
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
    sim_logger.info(f"--- Simulation {dca2_sim_name} Complete ---") # etc...
    file_handler.close()


    # --- Write results to Excel using xlsxwriter engine ---
    logging.info(f"Writing results to {output_file} using xlsxwriter engine for number formatting.")
    try:
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # --- Write Summary Sheet ---
            summary_df = pd.DataFrame(all_results_summary)
            if 'Overall Growth (%)' in summary_df.columns:
                 summary_df.sort_values(by='Overall Growth (%)', ascending=False, inplace=True)
            else: logging.warning("Could not sort summary results.")

            summary_sheet_name = 'Summary'
            summary_cols_order = [ # Market CAGR removed
                'Scenario', 'Description', 'Total Capital Provided', 'Total Cash Invested',
                'Final Investment Value', 'Final Cash Remaining', 'Final Total Portfolio Value',
                'Number of Buys', 'Average Buy Size ($)', 'Overall Growth (%)',
                'Scenario Annualized Growth (%)',
                'Total Units Purchased', 'Weighted Avg Purchase Price', 'Final Market Close Price'
            ]
            summary_cols_order = [col for col in summary_cols_order if col in summary_df.columns]
            summary_df = summary_df[summary_cols_order]
            summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)

            workbook = writer.book
            summary_worksheet = writer.sheets[summary_sheet_name]

            # (Format definitions)
            currency_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            units_format = workbook.add_format({'num_format': '#,##0.0000'})
            integer_format = workbook.add_format({'num_format': '#,##0'})
            header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})

            summary_col_map = {name: i for i, name in enumerate(summary_df.columns)}
            # (Updated format lists - Market CAGR removed)
            summary_currency_cols = ['Total Capital Provided', 'Total Cash Invested', 'Final Investment Value', 'Final Cash Remaining', 'Final Total Portfolio Value', 'Average Buy Size ($)', 'Weighted Avg Purchase Price', 'Final Market Close Price']
            summary_unit_cols = ['Total Units Purchased']
            summary_percent_cols = ['Overall Growth (%)', 'Scenario Annualized Growth (%)']
            summary_integer_cols = ['Number of Buys']

            # (Applying formats - Market CAGR removed)
            for col_name in summary_currency_cols:
                if col_name in summary_col_map: summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 18, currency_format)
            for col_name in summary_unit_cols:
                 if col_name in summary_col_map: summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 18, units_format)
            for col_name in summary_percent_cols:
                 if col_name in summary_col_map: summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 15, percent_format)
            for col_name in summary_integer_cols:
                 if col_name in summary_col_map: summary_worksheet.set_column(summary_col_map[col_name], summary_col_map[col_name], 12, integer_format)

            for col_num, value in enumerate(summary_df.columns.values): summary_worksheet.write(0, col_num, value, header_format)

            # --- Write Detailed Monthly Sheets (including both DCAs) ---
            for sim_name, monthly_df in all_monthly_details.items():
                safe_sheet_name = "".join(c for c in sim_name if c.isalnum() or c in (' ', '_', '-'))[:31]
                if isinstance(monthly_df, pd.DataFrame):
                     # Add InterestEarnedThisMonth if it exists
                     if 'InterestEarnedThisMonth' not in monthly_df.columns:
                         monthly_df['InterestEarnedThisMonth'] = 0.0 # Add column if missing (e.g., older runs)

                     # Ensure correct column order including Interest
                     monthly_cols_order = [
                         'MonthEnd', 'CashAdded', 'InterestEarnedThisMonth', 'CashAtMonthStart',
                         'CashInvestedThisMonth', 'AvgPurchasePriceThisMonth', 'UnitsBoughtThisMonth',
                         'CumulativeUnitsHeld', 'CumulativeCashInvested', 'MarketClosePriceMonthEnd',
                         'ValueInvestmentsMonthEnd', 'CashOnHandMonthEnd', 'TotalPortfolioValueMonthEnd'
                     ]
                     monthly_cols_order = [col for col in monthly_cols_order if col in monthly_df.columns]
                     monthly_df = monthly_df[monthly_cols_order] # Apply order

                     monthly_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                     monthly_worksheet = writer.sheets[safe_sheet_name]
                     monthly_col_map = {name: i for i, name in enumerate(monthly_df.columns)}

                     # Add InterestEarnedThisMonth to currency formatting
                     monthly_currency_cols = ['CashAdded', 'InterestEarnedThisMonth', 'CashAtMonthStart', 'CashInvestedThisMonth', 'AvgPurchasePriceThisMonth', 'CumulativeCashInvested', 'MarketClosePriceMonthEnd', 'ValueInvestmentsMonthEnd', 'CashOnHandMonthEnd', 'TotalPortfolioValueMonthEnd']
                     monthly_units_cols = ['UnitsBoughtThisMonth', 'CumulativeUnitsHeld']

                     for col_name in monthly_currency_cols:
                          if col_name in monthly_col_map: monthly_worksheet.set_column(monthly_col_map[col_name], monthly_col_map[col_name], 18, currency_format)
                     for col_name in monthly_units_cols:
                          if col_name in monthly_col_map: monthly_worksheet.set_column(monthly_col_map[col_name], monthly_col_map[col_name], 18, units_format)

                     for col_num, value in enumerate(monthly_df.columns.values): monthly_worksheet.write(0, col_num, value, header_format)
                else:
                     logging.warning(f"Skipping sheet '{safe_sheet_name}' because data is not a DataFrame.")

        logging.info("Results successfully written to Excel with number formatting and sorted summary.")
    except ImportError:
        logging.error("Error: 'xlsxwriter' engine required but not installed. Please run: pip install xlsxwriter")
    except Exception as e:
        logging.error(f"Error writing results to Excel: {e}")

    return pd.DataFrame(all_results_summary)


# --- Main Execution ---
if __name__ == "__main__":
    config = load_config()
    if config:
        data = load_and_prepare_data(config)
        if data is not None:
            results = run_simulation(config, data)
            if results is not None:
                logging.info("\n--- Overall Simulation Summary (DataFrame before sorting/writing) ---")
                # print(results.to_string(index=False))
            else:
                logging.error("Simulation run failed.")
        else:
            logging.error("Data loading failed. Aborting simulation.")
    else:
        logging.error("Configuration loading failed. Aborting.")


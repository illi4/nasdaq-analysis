import pandas as pd
import yaml
import os
import logging
import logging.handlers # For file logging
import numpy as np # For CAGR calculation
import traceback # For detailed error logging

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
        required_keys = ['monthly_investment', 'start_date', 'end_date', 'data_file',
                         'output_file', 'date_column', 'price_column', 'high_column', 'low_column',
                         'simulations']
        if not all(key in config for key in required_keys):
            missing = [key for key in required_keys if key not in config]
            logging.error(f"Error: Missing required configuration keys: {missing}.")
            return None
        config.setdefault('log_level', 'INFO')
        config.setdefault('cash_interest_rate_annual', 0.0)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Error: Config file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
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
        df = df.rename(columns={date_col: 'Date', price_col: 'Close', high_col: 'High', low_col: 'Low'})
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
            logging.error(f"No valid data for date range: {effective_start.date()} to {effective_end.date()}")
            return None

        df['PeakHigh'] = df['High'].expanding().max()
        logging.info(f"Data loaded. Shape: {df.shape}. Range: {df.index.min().date()} to {df.index.max().date()}.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Data file not found: {data_file}")
        return None
    except KeyError as e:
         logging.error(f"Error: Column mapping issue (check config): {e}")
         return None
    except ValueError as e:
         logging.error(f"Error converting price data to numeric: {e}")
         return None
    except Exception as e:
        logging.error(f"Error processing data file: {e}")
        return None

# --- CAGR Calculation Helper ---
def calculate_cagr(end_value, start_value, years):
    """Calculates Compound Annual Growth Rate (CAGR)."""
    if years <= 0 or start_value <= 0 or end_value <= 0: return 0.0
    try:
        return (float(end_value) / float(start_value)) ** (1.0 / years) - 1.0
    except (ValueError, TypeError):
        logging.warning(f"Could not calculate CAGR for end={end_value}, start={start_value}, years={years}")
        return 0.0

# --- Helper to find next trading day ---
def find_next_trading_day(target_date, data_index):
    """Finds the first trading day index on or after target_date."""
    if not data_index.is_monotonic_increasing:
        logging.warning("Data index not sorted. Sorting.")
        data_index = data_index.sort_values()
    try:
        idx_pos = data_index.searchsorted(target_date)
        return data_index[idx_pos] if idx_pos < len(data_index) else None
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
    actual_start_date = data.index.min() # Actual start date based on data
    actual_end_date = data.index.max()   # Actual end date based on data
    simulations = config.get('simulations', {})
    output_file = config['output_file']
    log_level_str = config.get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    cash_interest_rate_annual = config.get('cash_interest_rate_annual', 0.0)
    monthly_interest_rate = cash_interest_rate_annual / 100.0 / 12.0
    logging.info(f"Monthly interest rate: {monthly_interest_rate:.4%}")

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Per-simulation logs in: '{log_dir}/'")

    all_results_summary = []
    all_monthly_details = {}
    all_transaction_details = {}

    num_years = (actual_end_date - actual_start_date).days / 365.25
    if num_years <= 0: num_years = np.nan

    market_start_price = data['Close'].iloc[0]
    market_end_price = data['Close'].iloc[-1]
    logging.info(f"Sim Period: {actual_start_date.date()} to {actual_end_date.date()} ({num_years:.2f} yrs)")
    logging.info(f"Market Start Price: {market_start_price:,.2f}, End Price: {market_end_price:,.2f}")

    # --- Run Config-Based Simulations ---
    for sim_name, sim_config in simulations.items():
        sim_logger = logging.getLogger(sim_name)
        sim_logger.setLevel(log_level)
        for handler in sim_logger.handlers[:]: sim_logger.removeHandler(handler); handler.close()
        log_file_path = os.path.join(log_dir, f"{sim_name}.log")
        file_handler = None
        try:
            file_handler = logging.FileHandler(log_file_path, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            sim_logger.addHandler(file_handler)
            sim_logger.propagate = False
            sim_logger.info(f"--- Starting Sim: {sim_name} ---")
            sim_logger.info(f"Desc: {sim_config.get('description', 'N/A')}")
            sim_logger.info(f"Alloc: {sim_config.get('allocations', {})}")

            allocations = sim_config.get('allocations', {})
            if not isinstance(allocations, dict):
                 sim_logger.error("Invalid 'allocations'. Skipping.")
                 continue
            sorted_thresholds = sorted(allocations.keys())
            cash_available, total_units_held, total_cash_invested, num_buys = 0.0, 0.0, 0.0, 0
            monthly_log, all_purchases_log = [], []
            last_peak_high = 0
            triggered_pullbacks_since_peak = set()

            for month_start_dt in pd.date_range(actual_start_date, actual_end_date, freq='MS'):
                try:
                    current_month_start_actual = data.index[data.index >= month_start_dt].min()
                    month_end_target = month_start_dt + pd.offsets.MonthEnd(0)
                    current_month_end_actual = data.index[data.index <= month_end_target].max()
                    current_month_end_actual = min(current_month_end_actual, actual_end_date)
                except IndexError:
                    sim_logger.warning(f"No data for month {month_start_dt.strftime('%Y-%m')}. Skip.")
                    continue

                interest_earned_this_month = 0.0
                if cash_available > 0 and monthly_interest_rate > 0:
                    interest_earned_this_month = cash_available * monthly_interest_rate
                    cash_available += interest_earned_this_month
                    sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')} Interest: ${interest_earned_this_month:,.2f}")

                cash_added_this_month = 0.0
                if month_start_dt >= actual_start_date:
                     cash_available += monthly_investment
                     cash_added_this_month = monthly_investment
                     cash_at_month_start_recorded = cash_available
                     sim_logger.debug(f"Month Start {month_start_dt.strftime('%Y-%m')} Added ${monthly_investment}, Cash: ${cash_at_month_start_recorded:,.2f}")
                else:
                     cash_at_month_start_recorded = cash_available

                month_data = data.loc[current_month_start_actual:current_month_end_actual]
                cash_invested_this_month, units_bought_this_month = 0.0, 0.0

                if not month_data.empty:
                    for date, row in month_data.iterrows():
                        current_low, current_peak_high = row['Low'], row['PeakHigh']
                        if current_peak_high > last_peak_high:
                            if last_peak_high > 0: sim_logger.debug(f"{date.date()}: New peak {current_peak_high:.2f}. Reset.")
                            last_peak_high = current_peak_high
                            triggered_pullbacks_since_peak = set()

                        for threshold in sorted_thresholds:
                            if last_peak_high <= 0: continue
                            pullback_target_price = last_peak_high * (1 - threshold / 100.0)
                            if current_low <= pullback_target_price and threshold not in triggered_pullbacks_since_peak:
                                allocation_pct = allocations[threshold] / 100.0
                                amount_to_invest = cash_available * allocation_pct
                                if amount_to_invest >= 0.01 and current_low > 0:
                                    units_bought = amount_to_invest / current_low
                                    cash_available -= amount_to_invest
                                    total_units_held += units_bought
                                    total_cash_invested += amount_to_invest
                                    cash_invested_this_month += amount_to_invest
                                    units_bought_this_month += units_bought
                                    num_buys += 1
                                    all_purchases_log.append({'date': date, 'price': current_low, 'units': units_bought, 'amount': amount_to_invest, 'threshold': threshold})
                                    triggered_pullbacks_since_peak.add(threshold)
                                    sim_logger.debug(f"{date.date()}: Buy @ {threshold}%! Invest ${amount_to_invest:.2f}, buy {units_bought:.4f} @ ${current_low:.2f}. Left ${cash_available:.2f}")
                                elif amount_to_invest < 0.01: sim_logger.debug(f"{date.date()}: Skip {threshold}% - amt ${amount_to_invest:.2f} small.")
                                elif current_low <= 0: sim_logger.warning(f"{date.date()}: Skip {threshold}% - Low ${current_low:.2f} <= 0.")

                final_close = data['Close'].asof(current_month_end_actual)
                if pd.isna(final_close): final_close = 0
                value_invest = total_units_held * final_close
                total_value = value_invest + cash_available
                avg_price_month = (cash_invested_this_month / units_bought_this_month) if units_bought_this_month > 0 else 0

                monthly_log.append({
                    'MonthEnd': current_month_end_actual.strftime('%Y-%m-%d'), 'CashAdded': cash_added_this_month,
                    'InterestEarnedThisMonth': interest_earned_this_month, 'CashAtMonthStart': cash_at_month_start_recorded,
                    'CashInvestedThisMonth': cash_invested_this_month, 'AvgPurchasePriceThisMonth': avg_price_month,
                    'UnitsBoughtThisMonth': units_bought_this_month, 'CumulativeUnitsHeld': total_units_held,
                    'CumulativeCashInvested': total_cash_invested, 'MarketClosePriceMonthEnd': final_close,
                    'ValueInvestmentsMonthEnd': value_invest, 'CashOnHandMonthEnd': cash_available,
                    'TotalPortfolioValueMonthEnd': total_value
                })
                sim_logger.info(f"Month End {current_month_end_actual.strftime('%Y-%m-%d')}: Port Value ${total_value:,.2f}")
                sim_logger.debug(f"Month Details: Units={total_units_held:.4f}, Invested=${total_cash_invested:.2f}")

            final_inv_val = total_units_held * market_end_price
            final_port_val = final_inv_val + cash_available
            total_months = len(pd.date_range(start=actual_start_date, end=actual_end_date, freq='MS'))
            total_capital = total_months * monthly_investment
            cagr = calculate_cagr(final_port_val, total_capital, num_years)
            avg_buy = total_cash_invested / num_buys if num_buys > 0 else 0.0
            w_avg_price = (total_cash_invested / total_units_held) if total_units_held > 0 else 0
            growth_pct = ((final_port_val / total_capital) - 1.0) if total_capital > 0 else 0.0

            summary = {
                'Scenario': sim_name,
                'Description': sim_config.get('description', 'N/A'),
                'Sim Start Date': actual_start_date.strftime('%Y-%m-%d'),
                'Sim End Date': actual_end_date.strftime('%Y-%m-%d'),
                'Total Capital Provided': total_capital,
                'Total Cash Invested': total_cash_invested,
                'Final Investment Value': final_inv_val,
                'Final Cash Remaining': cash_available,
                'Final Total Portfolio Value': final_port_val,
                'Number of Buys': num_buys,
                'Average Buy Size ($)': avg_buy,
                'Overall Growth (%)': growth_pct,
                'Scenario Annualized Growth (%)': cagr,
                'Total Units Purchased': total_units_held,
                'Weighted Avg Purchase Price': w_avg_price,
                'Final Market Close Price': market_end_price
            }
            all_results_summary.append(summary)
            all_monthly_details[sim_name] = pd.DataFrame(monthly_log)
            all_transaction_details[sim_name] = pd.DataFrame(all_purchases_log)
            sim_logger.info(f"--- Sim {sim_name} Complete --- Final Value: ${final_port_val:,.2f}")

        except Exception as e:
            sim_logger.error(f"!!! ERROR in sim {sim_name}: {e} !!!")
            sim_logger.error(traceback.format_exc())
            logging.error(f"Sim '{sim_name}' failed. See logs/{sim_name}.log. Skip results.")
        finally:
            if file_handler: file_handler.close()


    # --- Run DCA Scenario 1: Monthly Start ---
    dca1_sim_name = "DCA_Monthly_Start"
    dca1_desc = f"Invest ${monthly_investment} on first trading day/month."
    sim_logger = logging.getLogger(dca1_sim_name)
    sim_logger.setLevel(log_level)
    for handler in sim_logger.handlers[:]: sim_logger.removeHandler(handler); handler.close()
    log_file_path = os.path.join(log_dir, f"{dca1_sim_name}.log")
    file_handler = None
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter); sim_logger.addHandler(file_handler); sim_logger.propagate = False
        sim_logger.info(f"--- Starting Sim: {dca1_sim_name} ---"); sim_logger.info(dca1_desc)

        cash_available, total_units_held, total_cash_invested, num_buys = 0.0, 0.0, 0.0, 0
        monthly_log, all_purchases_log_dca1 = [], []

        for month_start_dt in pd.date_range(actual_start_date, actual_end_date, freq='MS'):
            try:
                current_month_start_actual = data.index[data.index >= month_start_dt].min()
                month_end_target = month_start_dt + pd.offsets.MonthEnd(0)
                current_month_end_actual = data.index[data.index <= month_end_target].max()
                current_month_end_actual = min(current_month_end_actual, actual_end_date)
            except IndexError: sim_logger.warning(f"No data month {month_start_dt.strftime('%Y-%m')}. Skip."); continue

            interest_earned = cash_available * monthly_interest_rate if cash_available > 0 else 0.0
            cash_available += interest_earned
            cash_added = 0.0
            if month_start_dt >= actual_start_date: cash_available += monthly_investment; cash_added = monthly_investment
            cash_start_rec = cash_available
            cash_invested_month, units_bought_month = 0.0, 0.0

            try:
                buy_date = current_month_start_actual
                buy_price = data.loc[buy_date, 'Close']
                if monthly_investment >= 0.01 and buy_price > 0:
                    actual_inv = min(monthly_investment, cash_available)
                    if actual_inv < monthly_investment: sim_logger.warning(f"{buy_date.date()}: Insufficient cash for full DCA. Invest ${actual_inv:.2f}.")
                    if actual_inv >= 0.01:
                        units = actual_inv / buy_price
                        cash_available -= actual_inv; total_units_held += units; total_cash_invested += actual_inv
                        cash_invested_month += actual_inv; units_bought_month += units; num_buys += 1
                        all_purchases_log_dca1.append({'date': buy_date, 'price': buy_price, 'units': units, 'amount': actual_inv, 'threshold': 'DCA_Start'})
                        sim_logger.debug(f"{buy_date.date()}: DCA Buy. Invest ${actual_inv:.2f} @ ${buy_price:.2f}. Units: {units:.4f}.")
                    else: sim_logger.debug(f"{buy_date.date()}: Skip DCA - amt small.")
                elif buy_price <= 0: sim_logger.warning(f"{buy_date.date()}: Skip DCA - price <= 0.")
                elif monthly_investment < 0.01: sim_logger.debug(f"{buy_date.date()}: Skip DCA - monthly amt small.")
            except KeyError: sim_logger.error(f"No price data for {buy_date.date()}. Skip buy.")
            except Exception as e: sim_logger.error(f"Error in DCA buy near {month_start_dt.date()}: {e}")

            final_close = data['Close'].asof(current_month_end_actual)
            if pd.isna(final_close): final_close = 0
            value_invest = total_units_held * final_close; total_value = value_invest + cash_available
            avg_price_month = (cash_invested_month / units_bought_month) if units_bought_month > 0 else 0
            monthly_log.append({
                 'MonthEnd': current_month_end_actual.strftime('%Y-%m-%d'), 'CashAdded': cash_added, 'InterestEarnedThisMonth': interest_earned,
                 'CashAtMonthStart': cash_start_rec, 'CashInvestedThisMonth': cash_invested_month, 'AvgPurchasePriceThisMonth': avg_price_month,
                 'UnitsBoughtThisMonth': units_bought_month, 'CumulativeUnitsHeld': total_units_held, 'CumulativeCashInvested': total_cash_invested,
                 'MarketClosePriceMonthEnd': final_close, 'ValueInvestmentsMonthEnd': value_invest, 'CashOnHandMonthEnd': cash_available,
                 'TotalPortfolioValueMonthEnd': total_value
            })
            sim_logger.info(f"Month End {current_month_end_actual.strftime('%Y-%m-%d')}: Port Value ${total_value:,.2f}")

        final_inv_val = total_units_held * market_end_price; final_port_val = final_inv_val + cash_available
        total_months = len(pd.date_range(start=actual_start_date, end=actual_end_date, freq='MS'))
        total_capital = total_months * monthly_investment
        cagr = calculate_cagr(final_port_val, total_capital, num_years)
        avg_buy = total_cash_invested / num_buys if num_buys > 0 else 0.0
        w_avg_price = (total_cash_invested / total_units_held) if total_units_held > 0 else 0
        growth_pct = ((final_port_val / total_capital) - 1.0) if total_capital > 0 else 0.0
        summary = {
            'Scenario': dca1_sim_name,
            'Description': dca1_desc,
            'Sim Start Date': actual_start_date.strftime('%Y-%m-%d'),
            'Sim End Date': actual_end_date.strftime('%Y-%m-%d'),
            'Total Capital Provided': total_capital,
            'Total Cash Invested': total_cash_invested,
            'Final Investment Value': final_inv_val,
            'Final Cash Remaining': cash_available,
            'Final Total Portfolio Value': final_port_val,
            'Number of Buys': num_buys,
            'Average Buy Size ($)': avg_buy,
            'Overall Growth (%)': growth_pct,
            'Scenario Annualized Growth (%)': cagr,
            'Total Units Purchased': total_units_held,
            'Weighted Avg Purchase Price': w_avg_price,
            'Final Market Close Price': market_end_price
        }
        all_results_summary.append(summary); all_monthly_details[dca1_sim_name] = pd.DataFrame(monthly_log)
        all_transaction_details[dca1_sim_name] = pd.DataFrame(all_purchases_log_dca1)
        sim_logger.info(f"--- Sim {dca1_sim_name} Complete --- Final Value: ${final_port_val:,.2f}")

    except Exception as e:
        sim_logger.error(f"!!! ERROR in sim {dca1_sim_name}: {e} !!!"); sim_logger.error(traceback.format_exc())
        logging.error(f"Sim '{dca1_sim_name}' failed. See logs/{dca1_sim_name}.log. Skip results.")
    finally:
        if file_handler: file_handler.close()


    # --- Run DCA Scenario 2: Bi-Monthly 1st and 15th ---
    dca2_sim_name = "DCA_BiMonthly_1_15"
    inv_per_period = monthly_investment / 2.0
    dca2_desc = f"Invest ${inv_per_period:.2f} on ~1st & ~15th day/month."
    sim_logger = logging.getLogger(dca2_sim_name)
    sim_logger.setLevel(log_level)
    for handler in sim_logger.handlers[:]: sim_logger.removeHandler(handler); handler.close()
    log_file_path = os.path.join(log_dir, f"{dca2_sim_name}.log")
    file_handler = None
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter); sim_logger.addHandler(file_handler); sim_logger.propagate = False
        sim_logger.info(f"--- Starting Sim: {dca2_sim_name} ---"); sim_logger.info(dca2_desc)

        cash_available, total_units_held, total_cash_invested, num_buys = 0.0, 0.0, 0.0, 0
        monthly_log, all_purchases_log_dca2 = [], []

        for month_start_dt in pd.date_range(actual_start_date, actual_end_date, freq='MS'):
            try:
                current_month_start_actual = data.index[data.index >= month_start_dt].min()
                month_end_target = month_start_dt + pd.offsets.MonthEnd(0)
                current_month_end_actual = data.index[data.index <= month_end_target].max()
                current_month_end_actual = min(current_month_end_actual, actual_end_date)
                month_data_index = data.loc[current_month_start_actual:current_month_end_actual].index
            except IndexError: sim_logger.warning(f"No data month {month_start_dt.strftime('%Y-%m')}. Skip."); continue

            interest_earned = cash_available * monthly_interest_rate if cash_available > 0 else 0.0
            cash_available += interest_earned
            cash_added = 0.0
            if month_start_dt >= actual_start_date: cash_available += monthly_investment; cash_added = monthly_investment
            cash_start_rec = cash_available
            cash_invested_month, units_bought_month = 0.0, 0.0

            buy_date_1 = find_next_trading_day(month_start_dt, month_data_index)
            buy_exec_1 = False
            if buy_date_1:
                try:
                    buy_price_1 = data.loc[buy_date_1, 'Close']
                    if inv_per_period >= 0.01 and buy_price_1 > 0:
                        actual_inv_1 = min(inv_per_period, cash_available)
                        if actual_inv_1 < inv_per_period: sim_logger.warning(f"{buy_date_1.date()}: Insufficient cash for 1st DCA. Invest ${actual_inv_1:.2f}.")
                        if actual_inv_1 >= 0.01:
                            units = actual_inv_1 / buy_price_1
                            cash_available -= actual_inv_1; total_units_held += units; total_cash_invested += actual_inv_1
                            cash_invested_month += actual_inv_1; units_bought_month += units; num_buys += 1
                            all_purchases_log_dca2.append({'date': buy_date_1, 'price': buy_price_1, 'units': units, 'amount': actual_inv_1, 'threshold': 'DCA_1st'})
                            sim_logger.debug(f"{buy_date_1.date()}: DCA Buy (1st). Invest ${actual_inv_1:.2f} @ ${buy_price_1:.2f}. Units: {units:.4f}.")
                            buy_exec_1 = True
                        else: sim_logger.debug(f"{buy_date_1.date()}: Skip 1st DCA - amt small.")
                    elif buy_price_1 <= 0: sim_logger.warning(f"{buy_date_1.date()}: Skip 1st DCA - price <= 0.")
                    elif inv_per_period < 0.01: sim_logger.debug(f"{buy_date_1.date()}: Skip 1st DCA - period amt small.")
                except KeyError: sim_logger.error(f"No price data {buy_date_1.date()}. Skip 1st buy.")
                except Exception as e: sim_logger.error(f"Error 1st DCA near {buy_date_1.date()}: {e}")
            else: sim_logger.debug(f"No trading day for 1st DCA in {month_start_dt.strftime('%Y-%m')}.")

            target_15 = pd.Timestamp(year=month_start_dt.year, month=month_start_dt.month, day=15)
            target_15 = max(target_15, current_month_start_actual)
            buy_date_15 = find_next_trading_day(target_15, month_data_index)
            if buy_date_15 and (not buy_exec_1 or buy_date_15 != buy_date_1):
                try:
                    buy_price_15 = data.loc[buy_date_15, 'Close']
                    if inv_per_period >= 0.01 and buy_price_15 > 0:
                        actual_inv_15 = min(inv_per_period, cash_available)
                        if actual_inv_15 < inv_per_period: sim_logger.warning(f"{buy_date_15.date()}: Insufficient cash for 15th DCA. Invest ${actual_inv_15:.2f}.")
                        if actual_inv_15 >= 0.01:
                            units = actual_inv_15 / buy_price_15
                            cash_available -= actual_inv_15; total_units_held += units; total_cash_invested += actual_inv_15
                            cash_invested_month += actual_inv_15; units_bought_month += units; num_buys += 1
                            all_purchases_log_dca2.append({'date': buy_date_15, 'price': buy_price_15, 'units': units, 'amount': actual_inv_15, 'threshold': 'DCA_15th'})
                            sim_logger.debug(f"{buy_date_15.date()}: DCA Buy (15th). Invest ${actual_inv_15:.2f} @ ${buy_price_15:.2f}. Units: {units:.4f}.")
                        else: sim_logger.debug(f"{buy_date_15.date()}: Skip 15th DCA - amt small.")
                    elif buy_price_15 <= 0: sim_logger.warning(f"{buy_date_15.date()}: Skip 15th DCA - price <= 0.")
                    elif inv_per_period < 0.01: sim_logger.debug(f"{buy_date_15.date()}: Skip 15th DCA - period amt small.")
                except KeyError: sim_logger.error(f"No price data {buy_date_15.date()}. Skip 15th buy.")
                except Exception as e: sim_logger.error(f"Error 15th DCA near {buy_date_15.date()}: {e}")
            elif buy_date_15 and buy_exec_1 and buy_date_15 == buy_date_1: sim_logger.debug(f"Skip 15th DCA - same day as 1st ({buy_date_15.date()}).")
            elif not buy_date_15: sim_logger.debug(f"No trading day for 15th DCA in {month_start_dt.strftime('%Y-%m')}.")

            final_close = data['Close'].asof(current_month_end_actual)
            if pd.isna(final_close): final_close = 0
            value_invest = total_units_held * final_close; total_value = value_invest + cash_available
            avg_price_month = (cash_invested_month / units_bought_month) if units_bought_month > 0 else 0
            monthly_log.append({
                 'MonthEnd': current_month_end_actual.strftime('%Y-%m-%d'), 'CashAdded': cash_added, 'InterestEarnedThisMonth': interest_earned,
                 'CashAtMonthStart': cash_start_rec, 'CashInvestedThisMonth': cash_invested_month, 'AvgPurchasePriceThisMonth': avg_price_month,
                 'UnitsBoughtThisMonth': units_bought_month, 'CumulativeUnitsHeld': total_units_held, 'CumulativeCashInvested': total_cash_invested,
                 'MarketClosePriceMonthEnd': final_close, 'ValueInvestmentsMonthEnd': value_invest, 'CashOnHandMonthEnd': cash_available,
                 'TotalPortfolioValueMonthEnd': total_value
            })
            sim_logger.info(f"Month End {current_month_end_actual.strftime('%Y-%m-%d')}: Port Value ${total_value:,.2f}")

        final_inv_val = total_units_held * market_end_price; final_port_val = final_inv_val + cash_available
        total_months = len(pd.date_range(start=actual_start_date, end=actual_end_date, freq='MS'))
        total_capital = total_months * monthly_investment
        cagr = calculate_cagr(final_port_val, total_capital, num_years)
        avg_buy = total_cash_invested / num_buys if num_buys > 0 else 0.0
        w_avg_price = (total_cash_invested / total_units_held) if total_units_held > 0 else 0
        growth_pct = ((final_port_val / total_capital) - 1.0) if total_capital > 0 else 0.0
        summary = {
            'Scenario': dca2_sim_name,
            'Description': dca2_desc,
            'Sim Start Date': actual_start_date.strftime('%Y-%m-%d'),
            'Sim End Date': actual_end_date.strftime('%Y-%m-%d'),
            'Total Capital Provided': total_capital,
            'Total Cash Invested': total_cash_invested,
            'Final Investment Value': final_inv_val,
            'Final Cash Remaining': cash_available,
            'Final Total Portfolio Value': final_port_val,
            'Number of Buys': num_buys,
            'Average Buy Size ($)': avg_buy,
            'Overall Growth (%)': growth_pct,
            'Scenario Annualized Growth (%)': cagr,
            'Total Units Purchased': total_units_held,
            'Weighted Avg Purchase Price': w_avg_price,
            'Final Market Close Price': market_end_price
        }
        all_results_summary.append(summary); all_monthly_details[dca2_sim_name] = pd.DataFrame(monthly_log)
        all_transaction_details[dca2_sim_name] = pd.DataFrame(all_purchases_log_dca2)
        sim_logger.info(f"--- Sim {dca2_sim_name} Complete --- Final Value: ${final_port_val:,.2f}")

    except Exception as e:
        sim_logger.error(f"!!! ERROR in sim {dca2_sim_name}: {e} !!!"); sim_logger.error(traceback.format_exc())
        logging.error(f"Sim '{dca2_sim_name}' failed. See logs/{dca2_sim_name}.log. Skip results.")
    finally:
        if file_handler: file_handler.close()


    # --- Write results to Excel using xlsxwriter engine ---
    logging.info(f"Writing results to {output_file} using xlsxwriter engine.")
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        if not all_results_summary:
            logging.warning("No results to write. Check logs.")
            return None

        with pd.ExcelWriter(output_file, engine='xlsxwriter',
                            datetime_format='yyyy-mm-dd', date_format='yyyy-mm-dd') as writer:
            # --- Write Summary Sheet ---
            summary_df = pd.DataFrame(all_results_summary)
            if 'Overall Growth (%)' in summary_df.columns:
                 summary_df.sort_values(by='Overall Growth (%)', ascending=False, inplace=True)
            else: logging.warning("Cannot sort summary by 'Overall Growth (%)'.")
            summary_sheet_name = 'Summary'

            # *** REORDERED SUMMARY COLUMNS ***
            summary_cols_order = [
                'Scenario', 'Description', 'Sim Start Date', 'Sim End Date',
                'Overall Growth (%)', 'Scenario Annualized Growth (%)', # MOVED Growth columns here
                'Total Capital Provided', 'Total Cash Invested',
                'Final Investment Value', 'Final Cash Remaining', 'Final Total Portfolio Value',
                'Number of Buys', 'Average Buy Size ($)', 'Total Units Purchased',
                'Weighted Avg Purchase Price',
                'Final Market Close Price'
            ]
            # *** END REORDER ***

            summary_cols_order = [col for col in summary_cols_order if col in summary_df.columns]
            summary_df = summary_df[summary_cols_order]
            summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)

            workbook = writer.book; summary_worksheet = writer.sheets[summary_sheet_name]
            curr_fmt = workbook.add_format({'num_format': '$#,##0.00'})
            pct_fmt = workbook.add_format({'num_format': '0.00%'})
            unit_fmt = workbook.add_format({'num_format': '#,##0.0000'})
            int_fmt = workbook.add_format({'num_format': '#,##0'})
            date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            hdr_fmt = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1, 'align': 'center'})
            col_map = {name: i for i, name in enumerate(summary_df.columns)}
            curr_cols = ['Total Capital Provided', 'Total Cash Invested', 'Final Investment Value', 'Final Cash Remaining', 'Final Total Portfolio Value', 'Average Buy Size ($)', 'Weighted Avg Purchase Price', 'Final Market Close Price']
            unit_cols = ['Total Units Purchased']
            pct_cols = ['Overall Growth (%)', 'Scenario Annualized Growth (%)'] # These remain percentage formatted
            int_cols = ['Number of Buys']
            date_cols = ['Sim Start Date', 'Sim End Date']

            for col in curr_cols:
                if col in col_map: summary_worksheet.set_column(col_map[col], col_map[col], 18, curr_fmt)
            for col in unit_cols:
                 if col in col_map: summary_worksheet.set_column(col_map[col], col_map[col], 18, unit_fmt)
            for col in pct_cols:
                 if col in col_map: summary_worksheet.set_column(col_map[col], col_map[col], 15, pct_fmt) # Apply percentage format
            for col in int_cols:
                 if col in col_map: summary_worksheet.set_column(col_map[col], col_map[col], 12, int_fmt)
            for col in date_cols:
                 if col in col_map: summary_worksheet.set_column(col_map[col], col_map[col], 12, date_fmt)
            if 'Description' in col_map: summary_worksheet.set_column(col_map['Description'], col_map['Description'], 30)
            if 'Scenario' in col_map: summary_worksheet.set_column(col_map['Scenario'], col_map['Scenario'], 25)

            for i, val in enumerate(summary_df.columns.values): summary_worksheet.write(0, i, val, hdr_fmt)
            summary_worksheet.freeze_panes(1, 0)

            # --- Write Detailed Monthly Sheets ---
            logging.info(f"Writing monthly sheets for {len(all_monthly_details)} scenarios...")
            for sim_name, monthly_df in all_monthly_details.items():
                try:
                    safe_base_name = "".join(c for c in sim_name if c.isalnum() or c in (' ', '_', '-'))
                    suffix = "_Monthly"
                    max_base_len = 31 - len(suffix)
                    truncated_base = safe_base_name[:max_base_len]
                    monthly_sheet_name = f"{truncated_base}{suffix}"

                    logging.debug(f"Attempting monthly sheet: {monthly_sheet_name}")
                    if isinstance(monthly_df, pd.DataFrame) and not monthly_df.empty:
                         if 'InterestEarnedThisMonth' not in monthly_df.columns: monthly_df['InterestEarnedThisMonth'] = 0.0
                         cols_order = [
                             'MonthEnd', 'CashAdded', 'InterestEarnedThisMonth', 'CashAtMonthStart', 'CashInvestedThisMonth',
                             'AvgPurchasePriceThisMonth', 'UnitsBoughtThisMonth', 'CumulativeUnitsHeld', 'CumulativeCashInvested',
                             'MarketClosePriceMonthEnd', 'ValueInvestmentsMonthEnd', 'CashOnHandMonthEnd', 'TotalPortfolioValueMonthEnd'
                         ]
                         cols_order = [col for col in cols_order if col in monthly_df.columns]
                         monthly_df = monthly_df[cols_order]
                         monthly_df.to_excel(writer, sheet_name=monthly_sheet_name, index=False)

                         worksheet = writer.sheets[monthly_sheet_name]
                         col_map_m = {name: i for i, name in enumerate(monthly_df.columns)}
                         curr_cols_m = ['CashAdded', 'InterestEarnedThisMonth', 'CashAtMonthStart', 'CashInvestedThisMonth', 'AvgPurchasePriceThisMonth', 'CumulativeCashInvested', 'MarketClosePriceMonthEnd', 'ValueInvestmentsMonthEnd', 'CashOnHandMonthEnd', 'TotalPortfolioValueMonthEnd']
                         unit_cols_m = ['UnitsBoughtThisMonth', 'CumulativeUnitsHeld']
                         date_cols_m = ['MonthEnd']
                         for col in curr_cols_m:
                              if col in col_map_m: worksheet.set_column(col_map_m[col], col_map_m[col], 18, curr_fmt)
                         for col in unit_cols_m:
                              if col in col_map_m: worksheet.set_column(col_map_m[col], col_map_m[col], 18, unit_fmt)
                         for col in date_cols_m:
                              if col in col_map_m: worksheet.set_column(col_map_m[col], col_map_m[col], 12)
                         for i, val in enumerate(monthly_df.columns.values): worksheet.write(0, i, val, hdr_fmt)
                         worksheet.freeze_panes(1, 0)
                         logging.debug(f"OK monthly sheet: {monthly_sheet_name}")
                    else:
                         logging.warning(f"Skip monthly sheet '{sim_name}' - empty/invalid.")
                except Exception as e_sheet:
                    logging.error(f"!!! FAILED sheet '{monthly_sheet_name}' for sim '{sim_name}': {e_sheet} !!!")
                    logging.error(traceback.format_exc())

            # --- Write Detailed Transaction Sheets ---
            logging.info(f"Writing transaction sheets for {len(all_transaction_details)} scenarios...")
            for sim_name, transactions_df in all_transaction_details.items():
                try:
                    safe_base_name = "".join(c for c in sim_name if c.isalnum() or c in (' ', '_', '-'))
                    suffix = "_Buys"
                    max_base_len = 31 - len(suffix)
                    truncated_base = safe_base_name[:max_base_len]
                    transactions_sheet_name = f"{truncated_base}{suffix}"

                    logging.debug(f"Attempting transaction sheet: {transactions_sheet_name}")
                    if isinstance(transactions_df, pd.DataFrame) and not transactions_df.empty:
                        cols_order = ['date', 'price', 'units', 'amount', 'threshold']
                        cols_order = [col for col in cols_order if col in transactions_df.columns]
                        transactions_df = transactions_df[cols_order]
                        transactions_df.to_excel(writer, sheet_name=transactions_sheet_name, index=False)

                        worksheet = writer.sheets[transactions_sheet_name]
                        col_map_t = {name: i for i, name in enumerate(transactions_df.columns)}
                        curr_cols_t = ['price', 'amount']; unit_cols_t = ['units']; date_cols_t = ['date']; other_cols_t = ['threshold']
                        for col in curr_cols_t:
                            if col in col_map_t: worksheet.set_column(col_map_t[col], col_map_t[col], 15, curr_fmt)
                        for col in unit_cols_t:
                            if col in col_map_t: worksheet.set_column(col_map_t[col], col_map_t[col], 18, unit_fmt)
                        for col in date_cols_t:
                            if col in col_map_t: worksheet.set_column(col_map_t[col], col_map_t[col], 12)
                        for col in other_cols_t:
                            if col in col_map_t: worksheet.set_column(col_map_t[col], col_map_t[col], 10)
                        for i, val in enumerate(transactions_df.columns.values): worksheet.write(0, i, val, hdr_fmt)
                        worksheet.freeze_panes(1, 0)
                        logging.debug(f"OK transaction sheet: {transactions_sheet_name}")
                    else:
                        logging.info(f"No buys for '{sim_name}'. Skip sheet.")
                except Exception as e_sheet:
                    logging.error(f"!!! FAILED sheet '{transactions_sheet_name}' for sim '{sim_name}': {e_sheet} !!!")
                    logging.error(traceback.format_exc())

        logging.info(f"Results written to {output_file}")
    except ImportError:
        logging.error("Error: 'xlsxwriter' not installed. pip install xlsxwriter")
    except Exception as e:
        logging.error(f"Error writing results to Excel: {e}")
        logging.error(traceback.format_exc())

    return pd.DataFrame(all_results_summary)


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting Investment Simulator Script")
    config = load_config()
    if config:
        data = load_and_prepare_data(config)
        if data is not None:
            results_summary_df = run_simulation(config, data)
            if isinstance(results_summary_df, pd.DataFrame) and not results_summary_df.empty:
                logging.info("\n--- Overall Simulation Summary ---")
                try:
                    cols_to_print = ['Scenario', 'Final Total Portfolio Value', 'Overall Growth (%)', 'Scenario Annualized Growth (%)', 'Number of Buys', 'Weighted Avg Purchase Price']
                    cols_to_print = [col for col in cols_to_print if col in results_summary_df.columns]
                    print(results_summary_df[cols_to_print].to_string(index=False, float_format='{:,.2f}'.format))
                except Exception as e:
                    logging.warning(f"Could not print formatted summary: {e}")
                logging.info(f"Detailed results saved to: {config['output_file']}")
                logging.info(f"Logs saved in: {os.path.abspath('logs/')}")
            elif isinstance(results_summary_df, pd.DataFrame) and results_summary_df.empty:
                 logging.warning("Sim completed, no summary results (all sims may have failed). Check logs.")
            elif results_summary_df is None:
                 logging.error("Sim run failed to produce results summary. Check logs.")
            else:
                 logging.error("Sim run returned unexpected type.")
        else:
            logging.error("Data loading failed. Abort.")
    else:
        logging.error("Config loading failed. Abort.")
    logging.info("Investment Simulator Script Finished")

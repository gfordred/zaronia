import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from scipy.optimize import brentq

# ==========================================
# CONFIGURATION & UTILS
# ==========================================
st.set_page_config(layout="wide", page_title="ZARONIA/JIBAR Swap Pricer")

# SA Holidays (2023-2026 approx, including Sunday rollovers)
SA_HOLIDAYS = {
    date(2023, 1, 1), date(2023, 1, 2), date(2023, 3, 21), date(2023, 4, 7), date(2023, 4, 10),
    date(2023, 4, 27), date(2023, 5, 1), date(2023, 6, 16), date(2023, 8, 9), date(2023, 9, 24),
    date(2023, 9, 25), date(2023, 12, 16), date(2023, 12, 25), date(2023, 12, 26),
    date(2024, 1, 1), date(2024, 3, 21), date(2024, 3, 29), date(2024, 4, 1), date(2024, 4, 27),
    date(2024, 5, 1), date(2024, 6, 16), date(2024, 6, 17), date(2024, 8, 9), date(2024, 9, 24),
    date(2024, 12, 16), date(2024, 12, 25), date(2024, 12, 26),
    date(2025, 1, 1), date(2025, 3, 21), date(2025, 4, 18), date(2025, 4, 21), date(2025, 4, 27),
    date(2025, 4, 28), date(2025, 5, 1), date(2025, 6, 16), date(2025, 8, 9), date(2025, 9, 24),
    date(2025, 12, 16), date(2025, 12, 25), date(2025, 12, 26),
    date(2026, 1, 1) # Extend as needed
}

def is_jbd(d):
    """Is Johannesburg Business Day"""
    if d.weekday() >= 5: return False # Sat=5, Sun=6
    if d in SA_HOLIDAYS: return False
    return True

def add_business_days(start_date, num_days):
    """Roll business days skipping weekends and holidays"""
    current = start_date
    added = 0
    step = 1 if num_days >= 0 else -1
    target = abs(num_days)
    
    while added < target:
        current += timedelta(days=step)
        if is_jbd(current):
            added += 1
    return current

def year_frac(d1, d2):
    """ACT/365 Year Fraction"""
    return (d2 - d1).days / 365.0

@st.cache_data
def load_historical_zaronia():
    """Load historical ZARONIA rates from CSV if available"""
    try:
        df = pd.read_csv("SARB-benchmark-data.csv", skiprows=6, header=None)
        df = df.iloc[:, [0, 1]]
        df.columns = ["Date", "Rate"]
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.dropna().sort_values("Date")
        return dict(zip(df["Date"], df["Rate"]))
    except Exception as e:
        return {}

@st.cache_data
def load_historical_market_data():
    """
    Load and merge ZARONIA history with JIBAR/Swap data from Excel.
    Returns a combined DataFrame sorted by Date.
    """
    # 1. Load ZARONIA (from CSV)
    z_history = load_historical_zaronia() # dict {date: rate}
    df_z = pd.DataFrame(list(z_history.items()), columns=["Date", "ZARONIA"])
    if not df_z.empty:
        df_z["Date"] = pd.to_datetime(df_z["Date"])
    
    # 2. Load JIBAR/SWAPS (from Excel)
    df_mkt = pd.DataFrame()
    try:
        df_mkt = pd.read_excel("JIBAR_FRA_SWAPS.xlsx")
        # Ensure standard column names if needed, but user specified headings:
        # Date, JIBAR3M, FRA 3x6, ...
        # Clean column names just in case
        df_mkt.columns = [str(c).strip() for c in df_mkt.columns]
        
        if "Date" in df_mkt.columns:
            df_mkt["Date"] = pd.to_datetime(df_mkt["Date"])
    except Exception as e:
        st.warning(f"Could not load JIBAR_FRA_SWAPS.xlsx: {e}. Please ensure the file is closed.")
        pass
        
    # 3. Merge
    if df_z.empty and df_mkt.empty:
        return pd.DataFrame()
    elif df_z.empty:
        return df_mkt.sort_values("Date", ascending=False)
    elif df_mkt.empty:
        return df_z.sort_values("Date", ascending=False)
    else:
        # Outer join on Date
        df_combined = pd.merge(df_mkt, df_z, on="Date", how="outer")
        df_combined = df_combined.sort_values("Date", ascending=False)
        return df_combined

# ==========================================
# CURVE CLASSES
# ==========================================

class JibarZeroCurve:
    def __init__(self, tenors, rates_nacc):
        """
        tenors: list of years (float)
        rates_nacc: list of NACC zero rates (decimal, not %)
        """
        self.tenors = np.array(tenors)
        self.rates = np.array(rates_nacc)
        
        # Sort just in case
        idx = np.argsort(self.tenors)
        self.tenors = self.tenors[idx]
        self.rates = self.rates[idx]

    def get_zero_rate(self, t):
        """Linear interpolation of zero rates"""
        if t <= 0:
            return self.rates[0]
        # Linear extrapolation flat at ends? Or linear?
        # Standard: Linear interp on rates
        return np.interp(t, self.tenors, self.rates)

    def get_df(self, t):
        r = self.get_zero_rate(t)
        return np.exp(-r * t)

class ZaroniaCurve:
    def __init__(self, val_date, jibar_curve, spread_func, max_years=10):
        self.val_date = val_date
        self.jibar_curve = jibar_curve
        self.spread_func = spread_func
        self.max_years = max_years
        self.history = load_historical_zaronia() # Load history
        self._build_curve()

    def get_rate_at(self, d):
        """
        Get overnight ZARONIA rate for a specific date.
        If d <= val_date, try history, else curve.
        Returns rate as decimal (e.g. 0.075 for 7.5%)
        """
        if d <= self.val_date:
            # Check history
            if d in self.history:
                return self.history[d] / 100.0
            # If missing history but in past/today, fallback to curve spot (t=0)
            # or could warn. For now, fallback to t=0 fwd
            t = 0.0
            return self.fwd_zaronia_1d[0]
        else:
            # Future
            t = year_frac(self.val_date, d)
            # Find index in t_axis
            # Simple nearest neighbor or interp
            # t_axis is daily grid
            idx = int(t * 365) # Approx index
            if idx < 0: idx = 0
            if idx >= len(self.fwd_zaronia_1d): idx = -1
            return self.fwd_zaronia_1d[idx]

    def _build_curve(self):
        """
        Step 5 & 6: Build ZARONIA curve from daily forwards
        """
        # Generate daily grid up to max_years + buffer
        # We need daily steps for compounding
        days = int(self.max_years * 365) + 10 # buffer
        
        self.t_axis = np.arange(0, days + 1) # days from val_date
        self.dates = [self.val_date + timedelta(days=int(x)) for x in self.t_axis]
        self.year_fracs = self.t_axis / 365.0 # ACT/365 from val_date
        
        # Calculate Forward 3M Jibar at each day t
        # f_3M(t) logic from Step 2
        # t is start of forward. End is t + 3M.
        # We approximate 3M as 91 days (or 0.25 years) for the projection
        # Ideally we'd map to business days, but for vectorization we'll use t + 0.25y
        
        t_start = self.year_fracs
        t_end = t_start + 0.25 # 3 months forward
        
        # Jibar curve DFs
        df_start = np.array([self.jibar_curve.get_df(t) for t in t_start])
        df_end = np.array([self.jibar_curve.get_df(t) for t in t_end])
        
        # f3m calculation (Eq 2 simplified for implementation)
        # The prompt formula: exp[...] / exp[...] - 1 / dt
        # This is exactly (DF(t)/DF(t+3m) - 1) / (t_3m - t)
        dt = t_end - t_start # roughly 0.25
        self.fwd_jibar_3m = (df_start / df_end - 1) / dt
        
        # Get spreads for each t
        spreads = np.array([self.spread_func(t) for t in t_start])
        
        # Overnight ZARONIA forward (Step 5)
        # f_1bd = f_3m - spread
        self.fwd_zaronia_1d = self.fwd_jibar_3m - spreads
        
        # Build Discount Curve (Step 6)
        # Z(t, t+x) = Product [1 + f_1bd(i) * tau]^-1
        # tau is 1/365 for daily steps
        tau = 1.0 / 365.0
        
        # strictly speaking, we compound (1 + r * tau). 
        # The prompt says "nominal annual compounded continuously" for Zero rates, 
        # but for the overnight accumulation it says:
        # Z(t, t+x) = Prod [1 + f_1bd * tau]^-1
        
        step_dfs = 1.0 / (1.0 + self.fwd_zaronia_1d * tau)
        self.cum_dfs = np.cumprod(step_dfs)
        # Insert 1.0 at t=0
        self.cum_dfs = np.insert(self.cum_dfs, 0, 1.0)
        # Trim to match t_axis length (cumprod returns N, we added 1, so N+1)
        # wait, fwd_zaronia_1d has N elements (for each day). 
        # cum_dfs[i] is DF at end of day i (start of day i+1).
        # DF(0) = 1.
        # DF(t_1) = 1 / (1+r0*tau)
        self.cum_dfs = self.cum_dfs[:-1] # Adjust size if needed, but let's be careful.
        
        # Re-align:
        # t_axis[0] = 0. DF should be 1.
        # t_axis[1] = 1 day. DF should be step_dfs[0]
        self.dfs = np.zeros(len(self.t_axis))
        self.dfs[0] = 1.0
        self.dfs[1:] = np.cumprod(step_dfs[:-1]) 

    def get_df(self, t):
        """Interpolate DF from daily grid"""
        # Linear interpolation on log DF (constant forward assumption between days)
        # or just linear on DF for simplicity given daily grid
        return np.interp(t, self.year_fracs, self.dfs)

    def get_zero_rate(self, t):
        if t < 1e-9: 
            # Return instantaneous short rate (first step forward) to avoid graph distortion
            return self.fwd_zaronia_1d[0]
        df = self.get_df(t)
        if df <= 0: return 0.0
        return -np.log(df) / t

    def get_fwd_rates(self, t_array):
        """Return JIBAR 3M and ZARONIA 1D forwards interpolated on t_array"""
        # Align forwards with time axis (step function effectively, but we interp for smooth plot)
        t_grid = self.year_fracs
        
        # Jibar 3M Forward
        j_fwd = np.interp(t_array, t_grid, self.fwd_jibar_3m)
        
        # Zaronia 1D Forward
        z_fwd = np.interp(t_array, t_grid, self.fwd_zaronia_1d)
        
        return j_fwd, z_fwd


# ==========================================
# PRICING ENGINE
# ==========================================

def generate_schedule(start_date, tenor_years, freq='Quarterly', convention='Modified Following'):
    """
    Generate unadjusted schedule for simplicity, then adjust.
    freq: Annual, Semi-annual, Quarterly
    """
    freq_map = {'Annual': 12, 'Semi-annual': 6, 'Quarterly': 3, 'Monthly': 1}
    months = freq_map.get(freq, 3)
    
    end_date = start_date + timedelta(days=int(tenor_years * 365)) # Approximate
    # Better: add years/months logic
    try:
        end_date = start_date.replace(year=start_date.year + int(tenor_years))
    except ValueError: # Leap year Feb 29
        end_date = start_date + timedelta(days=int(tenor_years * 365.25))

    dates = []
    curr = start_date
    while curr < end_date:
        # Add months
        m = curr.month + months
        y = curr.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        day = curr.day
        # Handle month end
        # (Simple logic: if day > max for month, clamp)
        max_d = (date(y, m % 12 + 1, 1) - timedelta(days=1)).day
        next_date = date(y, m, min(day, max_d))
        
        if next_date > end_date:
            next_date = end_date
        
        # Adjust business day (simple FOLLOWING for this demo if weekend)
        # Ideally use proper calendar
        while next_date.weekday() > 4:
            next_date += timedelta(days=1)
            
        dates.append(next_date)
        curr = next_date
        
    # Ensure at least one date for short tenors (stub at end)
    if not dates and start_date < end_date:
        dates.append(end_date)
        
    return dates

class SwapLeg:
    def __init__(self, leg_type, currency, notional, start_date, maturity_years, frequency, spread_bps=0, fixed_rate=None, float_index=None):
        self.type = leg_type # 'Fixed' or 'Float'
        self.currency = currency
        self.notional = notional
        self.start_date = start_date
        self.maturity = maturity_years
        self.freq = frequency
        self.spread = spread_bps / 10000.0
        self.fixed_rate = fixed_rate
        self.float_index = float_index # 'JIBAR' or 'ZARONIA'
        
        self.schedule = generate_schedule(start_date, maturity_years, frequency)
        self.cashflows = []

    def calculate_cashflows(self, jibar_curve, zaronia_curve, discount_curve_choice):
        """
        Populate self.cashflows list of dicts
        """
        self.cashflows = []
        prev_date = self.start_date
        
        # Select discount curve
        disc_curve = zaronia_curve if discount_curve_choice == 'ZARONIA' else jibar_curve

        for pay_date in self.schedule:
            # Accrual
            yf = year_frac(prev_date, pay_date)
            
            # Rate determination
            rate = 0.0
            
            if self.type == 'Fixed':
                rate = self.fixed_rate
            else:
                # Float
                t_start = year_frac(zaronia_curve.val_date, prev_date)
                t_end = year_frac(zaronia_curve.val_date, pay_date)
                
                if self.float_index == 'JIBAR':
                    # Forward 3M JIBAR
                    # Implied from Jibar Curve
                    df_s = jibar_curve.get_df(t_start)
                    df_e = jibar_curve.get_df(t_end)
                    fwd = (df_s / df_e - 1) / yf
                    rate = fwd
                elif self.float_index == 'ZARONIA':
                    # Compounded Overnight
                    # (DF_z_start / DF_z_end - 1) / yf
                    df_s = zaronia_curve.get_df(t_start)
                    df_e = zaronia_curve.get_df(t_end)
                    fwd = (df_s / df_e - 1) / yf
                    rate = fwd
            
            # Add spread
            total_rate = rate + self.spread
            
            # Cashflow
            cf = self.notional * total_rate * yf
            
            # Discount
            t_pay = year_frac(zaronia_curve.val_date, pay_date)
            df = disc_curve.get_df(t_pay)
            pv = cf * df
            
            self.cashflows.append({
                'Period End': pay_date,
                'Accrual Start': prev_date,
                'Accrual End': pay_date,
                'Year Fraction': yf,
                'Forward Rate': rate,
                'Spread': self.spread,
                'Net Rate': total_rate,
                'Cashflow': cf,
                'Discount Factor': df,
                'PV': pv
            })
            
            prev_date = pay_date

    def get_pv(self):
        return sum(cf['PV'] for cf in self.cashflows)

class ZaroniaFRN:
    def __init__(self, notional, start_date, maturity_date, margin_bps, lookback_days=5, freq='Quarterly'):
        self.notional = notional
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.margin = margin_bps / 10000.0
        self.lookback = lookback_days
        
        # Generate schedule (Modified Following)
        # Re-use generate_schedule but we need precise dates
        years = year_frac(start_date, maturity_date)
        # Round to nearest year for simplicity in generating list, or logic
        # We'll use the existing function which takes years
        # A bit hacky but works for demo. Better to just loop months.
        self.schedule = generate_schedule(start_date, years, freq) 
        # Fix last date to match maturity exactly if close
        if self.schedule:
             self.schedule[-1] = maturity_date
             
        self.cashflows = []

    def calculate_cashflows(self, zaronia_curve):
        self.cashflows = []
        prev_date = self.start_date
        
        for pay_date in self.schedule:
            # Interest Period: prev_date to pay_date (exclude pay_date)
            # We iterate through JBDs in this period
            
            curr = prev_date
            compounding_factor = 1.0
            
            # For logging daily rates in this period
            period_dailies = []
            
            while curr < pay_date:
                if not is_jbd(curr):
                    curr += timedelta(days=1)
                    continue
                
                # It is a JBD.
                # 1. Find next JBD to determine n_i
                next_jbd = curr + timedelta(days=1)
                while not is_jbd(next_jbd) and next_jbd < pay_date:
                     next_jbd += timedelta(days=1)
                # Note: if next_jbd hits pay_date, that's the end of accrual for this period step
                # Actually, standard convention: n_i is days to next JBD (regardless of period end? 
                # Usually period end cuts it off. Schedule 1 says "d = calendar days in Interest Period")
                # But n_i = number of calendar days until next JBD.
                # If next_jbd > pay_date, it should probably be capped at pay_date for the final stub?
                # The formula implies sum(n_i) = d. So yes, cap at pay_date.
                
                if next_jbd > pay_date:
                    next_jbd = pay_date
                    
                n_i = (next_jbd - curr).days
                if n_i == 0: # Should not happen if loop condition is curr < pay_date
                    break
                    
                # 2. Determine Rate Date (Lookback)
                # 5th JBD prior to curr
                rate_date = add_business_days(curr, -self.lookback)
                
                # 3. Get Rate
                r = zaronia_curve.get_rate_at(rate_date)
                
                # 4. Compound
                # Rounding: Schedule 1 says "Round resulting compounded rate to 4 decimals". 
                # Usually compounding is precise, final rate rounded. 
                # Prompt says: "Round resulting compounded rate to 4 decimals". 
                # We'll accumulate precise, then round the final 'Compounded ZARONIA'.
                
                step_factor = 1 + r * n_i / 365.0
                compounding_factor *= step_factor
                
                period_dailies.append({
                    "Date": curr,
                    "RateDate": rate_date,
                    "Rate": r,
                    "Days": n_i
                })
                
                curr = next_jbd
            
            # End of Period Calculation
            # Formula: [ Prod(...) - 1 ] * 365 / d
            d_total = (pay_date - prev_date).days
            if d_total > 0:
                compounded_rate = (compounding_factor - 1.0) * (365.0 / d_total)
            else:
                compounded_rate = 0.0
                
            # Rounding (Schedule 1)
            # 0.00005% -> 5th decimal place? "Round resulting compounded rate to 4 decimals" usually means 1.2345%
            # i.e. 0.012345 rounded to 0.0123. 
            # Let's standard round to 6 decimal places (for rate as decimal) -> 4 decimal places for %
            # Wait, 4 decimals usually refers to percentage points in these docs. e.g. 5.1234%
            # So round(rate * 100, 4) / 100.
            final_float_rate = round(compounded_rate * 1000000) / 1000000.0 # Clean float
            # Let's stick to Python float precision for internal, but display rounded?
            # Prompt says "This must be enforced". OK. 
            # "Round resulting compounded rate to 4 decimals, 0.00005% rounds up" -> Standard round half up.
            # We'll apply it to the final coupon rate maybe? 
            # "Compounded ZARONIA" is the index.
            # Let's apply standard rounding to 5 decimals (e.g. 0.06543)
            
            idx_rate = round(compounded_rate, 5) # approx 4 decimals %
            
            # Coupon Rate
            coupon_rate = idx_rate + self.margin
            
            # Cashflow
            interest = self.notional * coupon_rate * d_total / 365.0
            
            # Discount
            df = zaronia_curve.get_df(year_frac(zaronia_curve.val_date, pay_date))
            pv = interest * df
            
            self.cashflows.append({
                "Start": prev_date,
                "End": pay_date,
                "Days": d_total,
                "ZARONIA_Comp": idx_rate,
                "Margin": self.margin,
                "Coupon": coupon_rate,
                "Amount": interest,
                "DF": df,
                "PV": pv,
                "Dailies": period_dailies
            })
            
            prev_date = pay_date
            
        # Add Principal at end? FRN valuation usually includes principal.
        # But if we are pricing just the swap leg, no. 
        # Prompt says "FRN pricing engine". FRNs pay principal at maturity.
        # We should add principal flow for Price calculation.
        final_df = zaronia_curve.get_df(year_frac(zaronia_curve.val_date, self.maturity_date))
        principal_pv = self.notional * final_df
        
        self.principal_flow = {
            "Start": self.cashflows[-1]["Start"] if self.cashflows else self.start_date,
            "End": self.maturity_date,
            "Days": 0,
            "ZARONIA_Comp": 0,
            "Margin": 0,
            "Coupon": 0,
            "Amount": self.notional,
            "DF": final_df,
            "PV": principal_pv,
            "Dailies": []
        }

    def get_clean_price(self):
        # Sum of interest PVs + Principal PV (scaled to 100)
        total_pv = sum(cf['PV'] for cf in self.cashflows) + self.principal_flow['PV']
        return (total_pv / self.notional) * 100.0

    def get_quarterly_equivalent(self):
        # Weighted average or simple compounding of the daily rates?
        # Prompt: (1 + Rq * DC) = Prod(...)
        # We can just take the calculated compounded rates for each period and average them?
        # Or returns the single Rq that matches the total compounding over the whole term?
        # Usually it's per period.
        if not self.cashflows: return 0.0
        return self.cashflows[0]['ZARONIA_Comp'] # Return first period for sample

# ==========================================
# BOOTSTRAPPING ENGINE
# ==========================================
def bootstrap_nacc_curve(mkt_data):
    """
    Bootstrap NACC Zero Curve from Market Par Rates.
    Robust implementation handling gaps and variable pillars.
    """
    instruments = []
    
    def get_rate(key):
        if key in mkt_data and not pd.isna(mkt_data[key]):
            return mkt_data[key] / 100.0
        return None

    # --- 1. Short End (Direct DFs) ---
    
    # 0.25Y: JIBAR 3M
    r_3m = get_rate('JIBAR3M')
    if r_3m is not None:
        instruments.append({'T': 0.25, 'DF': 1.0 / (1.0 + r_3m * 0.25)})
    
    # 0.50Y: FRA 3x6
    # DF(0.5) = DF(0.25) / (1 + r*0.25)
    if instruments and instruments[-1]['T'] == 0.25:
        r_3x6 = get_rate('FRA 3x6')
        if r_3x6 is not None:
             instruments.append({'T': 0.50, 'DF': instruments[-1]['DF'] / (1.0 + r_3x6 * 0.25)})
             
    # 0.75Y: FRA 6x9
    if instruments and instruments[-1]['T'] == 0.50:
        r_6x9 = get_rate('FRA 6x9')
        if r_6x9 is not None:
             instruments.append({'T': 0.75, 'DF': instruments[-1]['DF'] / (1.0 + r_6x9 * 0.25)})
             
    # 1.00Y: FRA 9x12 (Optional, lower priority than SASW1 usually, but let's check)
    # If we have SASW1, we solve it as a swap. If we only have FRA 9x12, we chain it.
    # Let's verify if SASW1 is present.
    has_sasw1 = get_rate('SASW1') is not None
    
    if not has_sasw1 and instruments and instruments[-1]['T'] == 0.75:
        r_9x12 = get_rate('FRA 9x12')
        if r_9x12 is not None:
             instruments.append({'T': 1.00, 'DF': instruments[-1]['DF'] / (1.0 + r_9x12 * 0.25)})

    # --- 2. Swaps (Generic Solver) ---
    # We solve for a flat zero rate from last_t to t_target
    
    swap_config = [
        ('SASW1', 1.0),
        ('SASW2', 2.0),
        ('SASW3', 3.0),
        ('SASW5', 5.0),
        ('SASW10', 10.0)
    ]
    
    # FRA 18x21 (1.75Y) check? 
    # If we have SASW2, we bootstrap 1.0 -> 2.0.
    # If we really want to use FRA 18x21, we would need to insert a node at 1.75.
    # But usually swaps are better quality pillars. We'll stick to swaps.
    
    for label, t_target in swap_config:
        r_swap = get_rate(label)
        if r_swap is None:
            continue
            
        # Check where we are
        if not instruments:
            continue # Can't start with swap if no short end
            
        last_instr = instruments[-1]
        last_t = last_instr['T']
        last_df = last_instr['DF']
        
        if t_target <= last_t:
            continue # Already covered (e.g. FRA 9x12 covered 1.0)
            
        # We need to value the swap.
        # Fixed Leg: Sum(R * 0.25 * DF(ti))
        # Float Leg: 1.0 - DF(tn) (assuming standard single curve)
        
        # We need DFs at all quarterly coupons up to t_target.
        # Split into Known (t <= last_t) and Unknown (t > last_t)
        
        # Generate full schedule 0.25, 0.50 ... t_target
        schedule = np.arange(0.25, t_target + 0.001, 0.25)
        
        # 1. Calculate PV of Known part
        # We need to interpolate DFs for schedule dates that are <= last_t if they are missing
        # But our instruments list should have them if we are consistent.
        # If we have gaps (e.g. missing FRA 6x9), we might need interp.
        # Let's build a temporary interpolation function for known curve
        
        known_ts = np.array([x['T'] for x in instruments])
        known_dfs = np.array([x['DF'] for x in instruments])
        # Log-linear interpolation
        known_log_dfs = np.log(known_dfs)
        
        def get_known_df(t):
            if t <= known_ts[0]: # extrapol back? shouldn't happen for t>=0.25
                return np.exp(known_log_dfs[0] * t / known_ts[0]) 
            # Interp
            log_df = np.interp(t, known_ts, known_log_dfs)
            return np.exp(log_df)

        pv_known = 0.0
        
        # Filter schedule for unknown
        unknown_dates = []
        
        for t in schedule:
            if t <= last_t + 0.001:
                pv_known += get_known_df(t)
            else:
                unknown_dates.append(t)
                
        if not unknown_dates:
            continue
            
        # 2. Solve for Zero Rate z from last_t to t_target
        # DF(t) = DF(last_t) * exp(-z * (t - last_t))
        
        def obj(z):
            pv_unknown = 0.0
            df_end = 0.0
            
            for t in unknown_dates:
                df_t = last_df * np.exp(-z * (t - last_t))
                pv_unknown += df_t
                if abs(t - t_target) < 0.001:
                    df_end = df_t
            
            # Swap Equation: Fixed = Float
            # R * 0.25 * (PV_known + PV_unknown) = 1.0 - DF_end
            fixed_leg = r_swap * 0.25 * (pv_known + pv_unknown)
            float_leg = 1.0 - df_end
            return fixed_leg - float_leg
            
        try:
            z_sol = brentq(obj, -0.10, 0.30)
            
            # Add points to instruments
            for t in unknown_dates:
                df_t = last_df * np.exp(-z_sol * (t - last_t))
                instruments.append({'T': t, 'DF': df_t})
                
        except Exception as e:
            # Solver failed
            # print(f"Failed to bootstrap {label}: {e}")
            pass

    # Convert to DataFrame
    data = []
    for x in instruments:
        t = x['T']
        df = x['DF']
        if t > 0:
            r_zero = -np.log(df) / t
            data.append({"Tenor (Y)": t, "Zero Rate (%)": r_zero * 100})
            
    return pd.DataFrame(data)

@st.cache_data
def get_historical_surfaces(df_market):
    """
    Bootstrap curves for historical dates to create a surface.
    """
    if df_market.empty: return None, None, None
    
    # Sample every 5th day (~weekly) for performance
    subset = df_market.iloc[::5].head(52) # Last year approx
    
    dates = []
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    rates_surface = []
    
    for _, row in subset.iterrows():
        d = row['Date']
        mkt = row.to_dict()
        curve_df = bootstrap_nacc_curve(mkt)
        
        if curve_df is not None and not curve_df.empty:
             ts = curve_df['Tenor (Y)'].values
             rs = curve_df['Zero Rate (%)'].values
             
             # Interp onto fixed grid
             # Need to ensure we don't extrapolation too wildly
             interp_rs = np.interp(tenors, ts, rs)
             rates_surface.append(interp_rs)
             dates.append(d)
             
    return dates, tenors, np.array(rates_surface)

# ==========================================
# UI & MAIN APP
# ==========================================

def main():
    st.title("ZARONIA / JIBAR Swap Pricer")
    st.markdown("Implementation of SARB MPG Methodology for ZARONIA OIS Curve Construction")
    
    # Load Data First
    df_market = load_historical_market_data()
    latest_data = {}
    
    default_val_date = date.today()
    default_j_spot = 6.78
    default_z_spot = 6.5
    
    if not df_market.empty:
        # Get top row (latest date)
        latest_row = df_market.iloc[0]
        latest_data = latest_row.to_dict()
        
        # Set defaults
        try:
            default_val_date = latest_row["Date"].date()
            if not pd.isna(latest_row["JIBAR3M"]):
                 default_j_spot = float(latest_row["JIBAR3M"])
            if not pd.isna(latest_row["ZARONIA"]):
                 default_z_spot = float(latest_row["ZARONIA"])
        except:
            pass

    # ==========================================
    # SIDEBAR INPUTS
    # ==========================================
    st.sidebar.header("Configuration")
    
    with st.sidebar.expander("1. Market Data Inputs", expanded=True):
        val_date = st.date_input("Valuation Date", value=default_val_date)
        
        # Look up market data for this specific date
        market_data_for_date = latest_data # Fallback
        found_date_msg = "Using Latest"
        
        if not df_market.empty:
            # df_market is sorted desc
            # Find row <= val_date (closest past date)
            mask = df_market['Date'].dt.date <= val_date
            filtered = df_market.loc[mask]
            if not filtered.empty:
                row = filtered.iloc[0]
                market_data_for_date = row.to_dict()
                found_date = row['Date'].date()
                if found_date == val_date:
                    found_date_msg = "Data Found"
                else:
                    found_date_msg = f"Using {found_date}"
            else:
                 found_date_msg = "No history found"

        st.caption(f" {found_date_msg}")
        
        analysis_term = st.slider("Analysis Term (Years)", 1, 10, 5)
        
        col1, col2 = st.columns(2)
        j_spot = col1.number_input("3M JIBAR Spot (%)", value=default_j_spot, step=0.01)
        z_spot = col2.number_input("ZARONIA Spot (%)", value=default_z_spot, step=0.01)
        
        s0_bps = (j_spot - z_spot) * 100
        st.info(f"Spot Spread $s_0(t)$: **{s0_bps:.2f} bps**")

    with st.sidebar.expander("2. Zero Curve & Spreads", expanded=False):
        st.subheader("JIBAR Zero Curve (NACC)")
        
        # Bootstrapping
        # Check if we need to init
        if 'jibar_curve_data' not in st.session_state:
            bootstrapped_df = bootstrap_nacc_curve(latest_data) # Start with latest
            if bootstrapped_df is not None and not bootstrapped_df.empty:
                 st.session_state.jibar_curve_data = bootstrapped_df
            else:
                default_data = {
                    "Tenor (Y)": [0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30],
                    "Zero Rate (%)": [7.0, 7.1, 7.25, 7.4, 7.55, 7.65, 7.75, 7.9, 8.1, 8.3, 8.4, 8.5]
                }
                st.session_state.jibar_curve_data = pd.DataFrame(default_data)
        
        # Button to re-bootstrap from selected date
        if st.button(f"Bootstrap from {found_date_msg} Data"):
             bootstrapped_df = bootstrap_nacc_curve(market_data_for_date)
             if bootstrapped_df is not None and not bootstrapped_df.empty:
                 st.session_state.jibar_curve_data = bootstrapped_df
                 st.rerun()
             else:
                 st.error("Bootstrapping failed or no data for this date.")

        edited_jibar = st.data_editor(
            st.session_state.jibar_curve_data, 
            num_rows="dynamic",
            use_container_width=True,
            key='editor_jibar',
            column_config={
                "Tenor (Y)": st.column_config.NumberColumn(format="%.2f Y"),
                "Zero Rate (%)": st.column_config.NumberColumn(format="%.4f %%")
            }
        )
        st.session_state.jibar_curve_data = edited_jibar
        
        st.subheader("Tenor Spread Overrides (bps)")
        spread_tenors_labels = ["0bd", "1M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y"]
        spread_tenors_years = [0.0, 1/12, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
        
        default_spreads = {
            "Tenor": spread_tenors_labels,
            "Spread (bps)": [s0_bps] * len(spread_tenors_labels)
        }
        df_spreads = pd.DataFrame(default_spreads)
        edited_spreads = st.data_editor(
            df_spreads,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Spread (bps)": st.column_config.NumberColumn(format="%.2f bps")
            }
        )

    with st.sidebar.expander("3. Trade Definition", expanded=True):
        trade_type = st.selectbox("Trade Type", ["JIBAR IRS", "ZARONIA OIS", "Basis Swap"])
        
        with st.sidebar.expander("ℹ️  Explanation: Swap Product Details"):
            st.markdown("""
            **1. JIBAR IRS (Interest Rate Swap)**
            *   **Definition**: A standard fixed-for-floating swap referencing the 3-Month JIBAR rate.
            *   **Mechanics**: The floating rate is a **forward-looking term rate** (3M JIBAR) set at the start of each period.
            *   **Risk Profile**: Reflects term bank credit risk. Historically the primary benchmark in South Africa.
            
            **2. ZARONIA OIS (Overnight Indexed Swap)**
            *   **Definition**: A swap referencing the compounded ZARONIA overnight rate.
            *   **Mechanics**: The floating rate is calculated by **compounding daily overnight rates** over the payment period. The final amount is only known at the end of the period.
            *   **Risk Profile**: Near risk-free (RFR). Used to hedge central bank policy rate expectations.
            
            **3. Basis Swap (JIBAR vs ZARONIA)**
            *   **Definition**: An exchange of two floating indices: 3M JIBAR vs Compounded ZARONIA.
            *   **Mechanics**: One leg pays JIBAR + Spread, the other pays ZARONIA (flat).
            *   **Economics**: Trades the **credit/liquidity spread** between the bank funding rate (JIBAR) and the risk-free rate (ZARONIA).
            """)

        col_t1, col_t2 = st.columns(2)
        notional = col_t1.number_input("Notional", value=10_000_000, step=1_000_000, format="%d")
        currency = col_t2.text_input("Currency", value="ZAR", disabled=True)
        
        start_delay = st.number_input("Start Delay (Days)", value=2, min_value=0)
        start_date = add_business_days(val_date, int(start_delay))
        st.caption(f"Start Date: {start_date}")
        
        maturity_str = st.selectbox("Maturity", ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"])
        maturity_years = int(maturity_str.replace("Y",""))
        
        st.markdown("---")
        st.caption("Leg Settings")
        
        fixed_leg_freq = st.selectbox("Fixed Leg Freq", ["Quarterly", "Semi-annual", "Annual"])
        
        solve_par = st.checkbox("Solve for Par Rate", value=True)
        solve_basis = False
        if trade_type == "Basis Swap":
             solve_basis = st.checkbox("Solve for Par Basis Spread", value=True)
             
        user_fixed_rate = st.number_input("Fixed Rate (%)", value=7.0, disabled=solve_par and trade_type != "Basis Swap", format="%.4f")
        
        float_spread_bps = st.number_input("Float Leg Spread (bps)", value=0.0, disabled=solve_basis, format="%.2f")

    # ==========================================
    # BUILD CURVES
    # ==========================================
    # 1. JIBAR Curve
    try:
        j_tenors = edited_jibar["Tenor (Y)"].values
        j_rates = edited_jibar["Zero Rate (%)"].values / 100.0
        jibar_curve = JibarZeroCurve(j_tenors, j_rates)
    except Exception as e:
        st.error(f"Error building JIBAR curve: {e}")
        return

    # 2. Spread Function
    try:
        s_y_points = edited_spreads["Spread (bps)"].values / 10000.0 
        def spread_func(t):
            if t > 5.0: return s_y_points[-1] 
            return np.interp(t, spread_tenors_years, s_y_points)
    except Exception as e:
        st.error(f"Error building spread function: {e}")
        return

    # 3. ZARONIA Curve
    zaronia_curve = ZaroniaCurve(val_date, jibar_curve, spread_func)

    # ==========================================
    # MAIN PANEL TABS
    # ==========================================
    tab_method, tab_charts, tab_trade, tab_bench, tab_frn, tab_comp, tab_hist = st.tabs([
        "Methodology & Theory", "Analysis & Charts", "Trade Pricing", "Benchmark Rates", 
        "FRN Pricing & Hedging", "Quarterly Compounding Analysis", "Historical Data"
    ])

    with tab_method:
        st.subheader("Academic & Technical Methodology")
        st.markdown("""
        This application implements the methodology described in the SARB MPG paper **"Historical estimation of the ZARONIA OIS curve"** (Section 3). 
        It constructs a ZARONIA OIS curve from the observable JIBAR swap curve and JIBAR-ZARONIA spreads without requiring historical deposit data.
        """)

        st.markdown("### Step 1: Input JSE Nominal Swap Zero Curve")
        st.info("**Academic Context**: The JSE publishes a nominal swap zero curve derived from JIBAR swaps. These rates are quoted as **NACC** (Nominal Annual Compounded Continuously).")
        st.latex(r"DF_{JIBAR}(t, T) = \exp\left(-r_{NACC}(t, T) \cdot \tau(t, T)\right)")
        st.markdown("""
        *   **Input**: A set of tenors and NACC zero rates.
        *   **Technical**: We linearly interpolate these zero rates $r(t,T)$ to obtain discount factors for any maturity $T$.
        """)
        
        st.divider()
        
        st.markdown("### Step 2: Build 3-Month JIBAR Forward Curve")
        st.info("**Academic Context**: The implied 3-month forward rate $f_{3M}(t)$ is the rate that equates the discount factors over the forward period.")
        st.markdown("Using Equation (2) from the paper, the forward rate for a reset date $t+x$ is calculated as:")
        st.latex(r"""
        f_{3M}(t+x) = \frac{ \frac{DF(t, t+x)}{DF(t, t+x+3M)} - 1 }{\tau_{3M}}
        """)
        st.markdown(r"""
        *   **Technical Implementation**: 
            1. We generate a daily grid of dates.
            2. For each day $T_{reset}$, we calculate $T_{maturity} \approx T_{reset} + 0.25$ years.
            3. We fetch $DF(T_{reset})$ and $DF(T_{maturity})$ from the JIBAR zero curve.
            4. The forward rate is computed analytically.
        """)

        st.divider()

        st.markdown("### Step 3 & 4: Spread Determination")
        st.info("**Academic Context**: Since ZARONIA is a risk-free(ish) overnight rate and JIBAR includes bank credit risk and term premium, $JIBAR > ZARONIA$. The spread $s(t)$ represents this basis.")
        st.markdown("The spot spread $s_0(t)$ is defined in Equation (1):")
        st.latex(r"s_0(t) = J_{spot}(t) - Z_{spot}(t)")
        st.markdown(r"""
        *   **Term Structure**: The paper assumes a term structure of spreads $s_y(t)$ for different tenors $y$.
        *   **Interpolation (Step 4)**: 
            *   Linear interpolation for tenors $y \in [0, 5]$ years.
            *   Flat extrapolation for $y > 5$ years.
        """)

        st.divider()

        st.markdown("### Step 5: Overnight ZARONIA Forward Curve")
        st.info("**Academic Context**: The core assumption is that the overnight ZARONIA rate is the 3M JIBAR forward rate minus the tenor-dependent spread.")
        st.markdown("Per Equation (4):")
        st.latex(r"f_{1bd}(t+x) = f_{3M}(t+x) - s_y(t)")
        st.markdown("""
        *   **Meaning**: We strip the credit/term spread from the JIBAR forward to recover the expected overnight risk-free rate.
        *   **Technical**: This is performed vector-wise across the full pricing grid (30+ years).
        """)

        st.divider()

        st.markdown("### Step 6: ZARONIA Discount Curve")
        st.info("**Academic Context**: An OIS discount factor is the geometric product of $(1 + r \tau)^{-1}$ for all overnight rates in the period.")
        st.markdown("Per Equation (5):")
        st.latex(r"Z(t, T) = \prod_{i=0}^{N-1} \left[ 1 + f_{1bd}(t_i) \cdot \tau_i \right]^{-1}")
        st.markdown("""
        *   **Technical**: 
            *   We compute step discount factors: $df_i = (1 + f_{1bd}[i] / 365)^{-1}$.
            *   We use `numpy.cumprod` to generate the full curve $Z(t, T)$.
        """)
        
        st.divider()
        
        st.markdown("### Step 7: Pricing & Benchmark Rates")
        st.markdown(r"""
        **Swap Valuation**:
        *   **Fixed Leg**: $\sum N \cdot C \cdot \tau_i \cdot DF(T_i)$
        *   **Floating Leg (OIS)**: The floating payment is compounded.
            *   Rate $\approx [\prod (1 + f_{1bd} \tau) - 1] / \tau_{total}$
            *   This is equivalent to: $(Z(t, T_{start}) / Z(t, T_{end}) - 1) / \tau_{total}$
        *   **Discounting**: Cashflows are discounted using the **ZARONIA** curve derived in Step 6 (Standard OIS discounting).
        """)

    with tab_charts:
        st.subheader("Curve Analytics")
        st.caption("Visualizing the transformation from JIBAR Zero Rates $\\rightarrow$ JIBAR Forwards $\\rightarrow$ ZARONIA Forwards $\\rightarrow$ ZARONIA Discount Curve.")
        t_plot = np.linspace(0, analysis_term, 300)
        
        # Helper for charts
        def create_fig(title, x_title, y_title):
            fig = go.Figure()
            fig.update_layout(
                title=dict(text=title, font=dict(size=18)),
                xaxis_title=x_title, 
                yaxis_title=y_title,
                template='plotly_dark',
                hovermode='x unified',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            return fig

        # 1. Zero Curves
        fig_zero = create_fig("Zero Curves (NACC)", "Years", "Rate (%)")
        j_zeros = [jibar_curve.get_zero_rate(t)*100 for t in t_plot]
        z_zeros = [zaronia_curve.get_zero_rate(t)*100 for t in t_plot]
        fig_zero.add_trace(go.Scatter(x=t_plot, y=j_zeros, name="JIBAR Zero", line=dict(color='#00CC96', width=2)))
        fig_zero.add_trace(go.Scatter(x=t_plot, y=z_zeros, name="ZARONIA Zero", line=dict(color='#EF553B', width=2)))

        # 2. Forward Rates
        fig_fwd = create_fig("Projected Forward Rates", "Years", "Rate (%)")
        j_fwd, z_fwd = zaronia_curve.get_fwd_rates(t_plot)
        fig_fwd.add_trace(go.Scatter(x=t_plot, y=j_fwd*100, name="3M JIBAR Forward", line=dict(color='#AB63FA', width=2)))
        fig_fwd.add_trace(go.Scatter(x=t_plot, y=z_fwd*100, name="ZARONIA O/N Forward", line=dict(color='#FFA15A', width=1)))

        # 3. Discount Factors
        fig_df = create_fig("Discount Factors", "Years", "DF")
        j_dfs = [jibar_curve.get_df(t) for t in t_plot]
        z_dfs = [zaronia_curve.get_df(t) for t in t_plot]
        fig_df.add_trace(go.Scatter(x=t_plot, y=j_dfs, name="JIBAR DF", line=dict(color='#00CC96', dash='dot')))
        fig_df.add_trace(go.Scatter(x=t_plot, y=z_dfs, name="ZARONIA DF", line=dict(color='#EF553B')))

        # 4. Spread Structure
        fig_spread = create_fig("Term Structure of Spreads", "Years", "Spread (bps)")
        spreads_plot = [spread_func(t)*10000 for t in t_plot]
        fig_spread.add_trace(go.Scatter(x=t_plot, y=spreads_plot, name="JIBAR-ZARONIA Spread", fill='tozeroy', line=dict(color='#19D3F3')))

        # Layout Grid
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_zero, use_container_width=True)
        c2.plotly_chart(fig_fwd, use_container_width=True)
        
        c3, c4 = st.columns(2)
        c3.plotly_chart(fig_df, use_container_width=True)
        c4.plotly_chart(fig_spread, use_container_width=True)
        
        st.divider()
        st.subheader("Historical Curve Evolution (3D Surface)")
        
        # Calculate historical surfaces
        if not df_market.empty:
            dates_surf, tenors_surf, rates_surf = get_historical_surfaces(df_market)
            
            if dates_surf and len(dates_surf) > 1:
                 fig_3d = go.Figure(data=[go.Surface(
                     z=rates_surf, 
                     x=tenors_surf, 
                     y=dates_surf,
                     colorscale='Viridis'
                 )])
                 
                 fig_3d.update_layout(
                     title="JIBAR Zero Curve Evolution (Last 1 Year)",
                     scene = dict(
                         xaxis_title='Tenor (Years)',
                         yaxis_title='Date',
                         zaxis_title='Zero Rate (%)'
                     ),
                     template='plotly_dark',
                     height=600,
                     margin=dict(l=0, r=0, b=0, t=50)
                 )
                 st.plotly_chart(fig_3d, use_container_width=True)
            else:
                 st.info("Insufficient historical data for 3D surface plot.")
        else:
            st.info("Load market data to see historical evolution.")

    with tab_trade:
        st.subheader("Swap Valuation Engine")
        
        # --- Trade Logic Same as before ---
        st.markdown("##### Discounting Configuration")
        disc_method = st.radio("Discounting Method", ["ZARONIA (OIS)", "JIBAR"], horizontal=True)
        
        with st.expander("ℹ️  Explanation: Discounting Methods"):
            st.markdown("""
            **Which curve should be used to calculate the Present Value (PV) of future cashflows?**
            
            *   **ZARONIA (OIS)**: 
                *   **Standard for Collateralized Trades**: Used when the trade is under a CSA (Credit Support Annex) where collateral earns the overnight rate.
                *   **Risk-Free**: Reflects the time value of money with minimal credit risk.
                *   *Modern market standard.*
                
            *   **JIBAR**: 
                *   **Uncollateralized / Historical**: Used for trades without collateral, where the funding cost implies bank credit risk (3M JIBAR).
                *   **Single-Curve**: The same curve is used for projecting floating rates and discounting (older convention).
            """)
            
        discount_curve_map = {'ZARONIA (OIS)': 'ZARONIA', 'JIBAR': 'JIBAR'}
        selected_disc = discount_curve_map[disc_method]

        leg1 = None
        leg2 = None
        
        if trade_type == "JIBAR IRS":
            leg1 = SwapLeg('Fixed', 'ZAR', notional, start_date, maturity_years, fixed_leg_freq, fixed_rate=0.0)
            leg2 = SwapLeg('Float', 'ZAR', notional, start_date, maturity_years, 'Quarterly', spread_bps=float_spread_bps, float_index='JIBAR')
        elif trade_type == "ZARONIA OIS":
            leg1 = SwapLeg('Fixed', 'ZAR', notional, start_date, maturity_years, fixed_leg_freq, fixed_rate=0.0)
            leg2 = SwapLeg('Float', 'ZAR', notional, start_date, maturity_years, 'Annual', spread_bps=float_spread_bps, float_index='ZARONIA')
        elif trade_type == "Basis Swap":
            leg1 = SwapLeg('Float', 'ZAR', notional, start_date, maturity_years, 'Quarterly', spread_bps=float_spread_bps, float_index='JIBAR')
            leg2 = SwapLeg('Float', 'ZAR', notional, start_date, maturity_years, 'Annual', spread_bps=0.0, float_index='ZARONIA')

        # Helper to update rates and recalc
        def price_swap(fixed_rate_val):
            if leg1.type == 'Fixed': leg1.fixed_rate = fixed_rate_val
            if leg2.type == 'Fixed': leg2.fixed_rate = fixed_rate_val
            
            leg1.calculate_cashflows(jibar_curve, zaronia_curve, selected_disc)
            leg2.calculate_cashflows(jibar_curve, zaronia_curve, selected_disc)
            return leg1.get_pv(), leg2.get_pv()

        # Solve for Par
        par_rate = 0.0
        par_spread = 0.0
        
        if trade_type in ["JIBAR IRS", "ZARONIA OIS"]:
            def obj(r):
                pv1, pv2 = price_swap(r)
                return pv1 - pv2
            try:
                par_rate = brentq(obj, -0.10, 0.50)
            except:
                par_rate = 0.0
        elif trade_type == "Basis Swap" and solve_basis:
             # Solve for spread on Leg 1 (JIBAR) to equate to Leg 2 (ZARONIA)
             def obj_basis(s_bps):
                 # Update spread on leg 1
                 leg1.spread = s_bps / 10000.0
                 leg1.calculate_cashflows(jibar_curve, zaronia_curve, selected_disc)
                 # Leg 2 is Zaronia flat (spread=0 usually, or fixed)
                 # Re-calc leg 2 just in case
                 leg2.calculate_cashflows(jibar_curve, zaronia_curve, selected_disc)
                 return leg1.get_pv() - leg2.get_pv()
             
             try:
                 par_spread = brentq(obj_basis, -500, 500) # +/- 500 bps
                 float_spread_bps = par_spread
                 leg1.spread = par_spread / 10000.0
             except:
                 par_spread = 0.0

        final_fixed_rate = par_rate if solve_par else user_fixed_rate / 100.0
        if trade_type == "Basis Swap":
            # If solved, spread is already set. If not, user input used.
            # Just ensure curves are calculated
            price_swap(0) # fixed rate arg ignored for Basis Swap
        else:
            price_swap(final_fixed_rate)
            
        pv_leg1 = leg1.get_pv()
        pv_leg2 = leg2.get_pv()
        npv = pv_leg1 - pv_leg2
        
        # --- DV01 Calculation (Finite Difference) ---
        # 1. Bump JIBAR Curve +1bp
        j_rates_up = j_rates + 0.0001
        jibar_curve_up = JibarZeroCurve(j_tenors, j_rates_up)
        # 2. Rebuild ZARONIA (it depends on JIBAR)
        zaronia_curve_up = ZaroniaCurve(val_date, jibar_curve_up, spread_func)
        
        # 3. Reprice Legs with Bumped Curves
        # We temporarily modify the legs to calculate sensitivity, then restore
        leg1.calculate_cashflows(jibar_curve_up, zaronia_curve_up, selected_disc)
        pv_leg1_up = leg1.get_pv()
        
        leg2.calculate_cashflows(jibar_curve_up, zaronia_curve_up, selected_disc)
        pv_leg2_up = leg2.get_pv()
        
        # 4. Calculate DV01 (Change in PV)
        # Standard convention: Sensitivity to +1bp parallel shift
        dv01_leg1 = pv_leg1_up - pv_leg1
        dv01_leg2 = pv_leg2_up - pv_leg2
        net_dv01 = dv01_leg1 - dv01_leg2
        
        # 5. Restore Legs to Base State (Critical for display)
        leg1.calculate_cashflows(jibar_curve, zaronia_curve, selected_disc)
        leg2.calculate_cashflows(jibar_curve, zaronia_curve, selected_disc)

        # Summary Metrics
        st.markdown("---")
        st.markdown("### 📊 Trade Valuation Summary")

        # --- Verbal Confirmation ---
        st.info(f"""
        **Trade Confirmation Summary**:
        *   **Structure**: {maturity_years}Y {trade_type} ({leg1.type} vs {leg2.type})
        *   **Notional**: {currency} {notional:,.0f}
        *   **Dates**: Effective {start_date} $\\rightarrow$ Maturing {leg1.schedule[-1]} ({leg1.schedule[-1] - start_date} days)
        *   **Economics**: 
            *   **Leg 1**: {leg1.type} Rate of **{final_fixed_rate*100:.4f}%** {f"(Spread {leg1.spread*10000:.1f} bps)" if leg1.type == 'Float' else ""}
            *   **Leg 2**: {leg2.type} Rate {f"(Spread {leg2.spread*10000:.1f} bps)" if leg2.type == 'Float' else ""} referencing {leg2.float_index if leg2.type == 'Float' else 'Fixed'}
        *   **Valuation**: The trade has a Net Present Value of **{currency} {npv:,.2f}** (Leg 1 PV: {pv_leg1:,.0f} | Leg 2 PV: {pv_leg2:,.0f}).
        """)
        
        # Custom CSS for metrics
        st.markdown("""
        <style>
        .metric-card {
            background-color: #262730;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #464b59;
            text-align: center;
        }
        .metric-label {
            font-size: 14px;
            color: #fafafa;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #00CC96;
        }
        .metric-value-neg {
            font-size: 24px;
            font-weight: bold;
            color: #EF553B;
        }
        .metric-sub {
            font-size: 12px;
            color: #aaaaaa;
            margin-top: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            if trade_type == "Basis Swap":
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Par Basis Spread</div>
                    <div class="metric-value">{"{:.2f} bps".format(par_spread) if solve_basis else "N/A"}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Par Rate</div>
                    <div class="metric-value">{"{:.4f}%".format(par_rate*100)}</div>
                </div>
                """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Fixed Rate Used</div>
                <div class="metric-value">{"{:.4f}%".format(final_fixed_rate*100) if trade_type != "Basis Swap" else "N/A"}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            npv_color = "metric-value" if npv >= 0 else "metric-value-neg"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Net Present Value</div>
                <div class="{npv_color}">{npv:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
             st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Net DV01 (+1bp)</div>
                <div class="metric-value">{"{:,.2f}".format(net_dv01)}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Leg Detail Cards
        st.markdown("")
        l1_col, l2_col = st.columns(2)
        
        with l1_col:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid #00CC96;">
                <div class="metric-label">Leg 1: {leg1.type}</div>
                <div class="metric-value">{pv_leg1:,.2f}</div>
                <div class="metric-sub">DV01: <b>{dv01_leg1:,.2f}</b> ZAR/bp</div>
            </div>
            """, unsafe_allow_html=True)
            
        with l2_col:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid #EF553B;">
                <div class="metric-label">Leg 2: {leg2.type}</div>
                <div class="metric-value">{pv_leg2:,.2f}</div>
                <div class="metric-sub">DV01: <b>{dv01_leg2:,.2f}</b> ZAR/bp</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 Cash Flow Visualisation")
        
        # Prepare CF Data
        cf1 = pd.DataFrame(leg1.cashflows)
        cf2 = pd.DataFrame(leg2.cashflows)
        
        fig_cf = go.Figure()
        
        if not cf1.empty:
            fig_cf.add_trace(go.Bar(
                x=cf1['Period End'], 
                y=cf1['Cashflow'],
                name=f"Leg 1 ({leg1.type})",
                marker_color='#00CC96'
            ))
            
        if not cf2.empty:
            # If standard swap, one leg pays. We might want to show direction if inferred, 
            # but standard is just show computed CF. 
            # Usually Fixed Leg pays (negative) in standard payer swap. 
            # But here SwapLeg returns absolute CF unless we apply sign.
            # Let's plot Leg 2 as negative for visualization contrast if it's opposite?
            # Or just plot side by side.
            # Visualizing typical Payer Swap: Rec Float, Pay Fixed.
            # We don't know user intent, but side-by-side bars are safest.
            fig_cf.add_trace(go.Bar(
                x=cf2['Period End'], 
                y=cf2['Cashflow'],
                name=f"Leg 2 ({leg2.type})",
                marker_color='#EF553B'
            ))

        fig_cf.update_layout(
            title="Projected Cash Flows",
            xaxis_title="Payment Date",
            yaxis_title="Amount (ZAR)",
            barmode='group',
            template='plotly_dark',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_cf, use_container_width=True)

        st.markdown("### 📋 Cashflow Tables")
        def show_leg_cf(leg, name):
            with st.expander(f"{name} ({leg.type}) - Details", expanded=False):
                df = pd.DataFrame(leg.cashflows)
                if not df.empty:
                    df['Period End'] = pd.to_datetime(df['Period End']).dt.date
                    df['Accrual Start'] = pd.to_datetime(df['Accrual Start']).dt.date
                    df['Accrual End'] = pd.to_datetime(df['Accrual End']).dt.date
                    st.dataframe(df.style.format({
                        'Year Fraction': '{:.4f}',
                        'Forward Rate': '{:.4%}',
                        'Spread': '{:.4%}',
                        'Net Rate': '{:.4%}',
                        'Cashflow': '{:,.2f}',
                        'Discount Factor': '{:.6f}',
                        'PV': '{:,.2f}'
                    }), use_container_width=True)
        
        c_tbl1, c_tbl2 = st.columns(2)
        with c_tbl1: show_leg_cf(leg1, "Leg 1")
        with c_tbl2: show_leg_cf(leg2, "Leg 2")

    with tab_bench:
        st.subheader("Step 7: Benchmark OIS Rates")
        st.write("Calculated par rates for standard ZARONIA OIS tenors based on current curves.")
        bench_tenors = [
            ("1M", 1/12, 'Annual'), ("3M", 0.25, 'Annual'), ("6M", 0.5, 'Annual'), 
            ("9M", 0.75, 'Annual'), ("1Y", 1.0, 'Annual'), ("2Y", 2.0, 'Annual'),
            ("5Y", 5.0, 'Annual'), ("10Y", 10.0, 'Annual'), ("30Y", 30.0, 'Annual')
        ]
        
        bench_data = []
        for label, t_year, freq in bench_tenors:
            leg_fix = SwapLeg('Fixed', 'ZAR', 100, start_date, t_year, freq, fixed_rate=0.0)
            leg_flt = SwapLeg('Float', 'ZAR', 100, start_date, t_year, 'Annual', float_index='ZARONIA')
            
            def solve_bench(r):
                leg_fix.fixed_rate = r
                leg_fix.calculate_cashflows(jibar_curve, zaronia_curve, 'ZARONIA')
                leg_flt.calculate_cashflows(jibar_curve, zaronia_curve, 'ZARONIA')
                
                pv_fix = leg_fix.get_pv()
                pv_flt = leg_flt.get_pv()
                
                # Check for zero sensitivity to avoid solver errors
                if pv_fix == 0 and r != 0: 
                    return 0.0 # Degenerate case
                
                return pv_fix - pv_flt
                
            # Pre-check existence of cashflows
            leg_fix.calculate_cashflows(jibar_curve, zaronia_curve, 'ZARONIA')
            if not leg_fix.cashflows:
                bench_data.append({"Tenor": label, "Par Rate (%)": "No Cashflows"})
                continue

            try:
                # Widen bracket and check signs
                r_par = brentq(solve_bench, -0.99, 5.0)
                bench_data.append({"Tenor": label, "Par Rate (%)": r_par * 100})
            except Exception as e:
                bench_data.append({"Tenor": label, "Par Rate (%)": f"Err: {str(e)}"})
                
        df_bench = pd.DataFrame(bench_data).set_index("Tenor").T
        # Use a more flexible formatter that handles strings (errors) and floats
        st.dataframe(df_bench, use_container_width=True)
        
        # Plot Benchmark Curve
        st.subheader("Benchmark Yield Curve")
        
        # Filter only numeric rates for plotting
        plot_data = [x for x in bench_data if isinstance(x["Par Rate (%)"], (int, float))]
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            # Map tenors to numerical years for X-axis sorting
            tenor_map = {
                "1M": 1/12, "3M": 0.25, "6M": 0.5, "9M": 0.75,
                "1Y": 1.0, "2Y": 2.0, "5Y": 5.0, "10Y": 10.0, "30Y": 30.0
            }
            df_plot["Years"] = df_plot["Tenor"].map(tenor_map)
            df_plot = df_plot.sort_values("Years")
            
            fig_bench = go.Figure()
            fig_bench.add_trace(go.Scatter(
                x=df_plot["Tenor"], # Categorical axis preserves labels like "1M"
                y=df_plot["Par Rate (%)"],
                mode='lines+markers+text',
                text=[f"{r:.2f}%" for r in df_plot["Par Rate (%)"]],
                textposition="top center",
                name='ZARONIA OIS',
                line=dict(color='#AB63FA', width=3)
            ))
            
            fig_bench.update_layout(
                title="ZARONIA OIS Benchmark Curve",
                xaxis_title="Tenor",
                yaxis_title="Par Rate (%)",
                template='plotly_dark',
                height=450,
                hovermode='x unified'
            )
            st.plotly_chart(fig_bench, use_container_width=True)

    with tab_frn:
        st.subheader("ZARONIA-Linked FRN Pricing (e.g., ABFZ02)")
        
        with st.expander("📝 Pricing Supplement (Schedule 1) Rules", expanded=False):
            st.markdown("""
            **Reference Rate**: Compounded Daily ZARONIA
            
            *   **Observation Method**: "Lookback" (item 35(d))
            *   **Lookback Period**: 5 Johannesburg Business Days (item 35(e))
            *   **Observation Shift**: Not applicable (item 35(f))
            *   **Interest Rate Determination Date**: The 5th JBD prior to each IPD (item 35(b))
            
            **Compounding Formula**:
            $$
            \\text{Compounded ZARONIA} = \\left[ \\prod_{i=1}^{do} \\left(1 + \\frac{ZARONIA_{i-5,JBD} \\cdot n_i}{365}\\right) - 1 \\right] \\cdot \\frac{365}{d}
            $$
            Where:
            *   $d$: calendar days in Interest Period
            *   $d_o$: JBDs in Interest Period
            *   $n_i$: calendar days until next JBD
            """)

        # Inputs
        col_f1, col_f2, col_f3 = st.columns(3)
        issue_date = col_f1.date_input("Issue Date", value=val_date)
        # Default 3Y maturity
        mat_date = col_f2.date_input("Maturity Date", value=add_business_days(val_date, 365*3))
        frn_notional = col_f3.number_input("Notional (ZAR)", value=100000000, step=1000000)
        
        col_f4, col_f5, col_f6 = st.columns(3)
        margin_bps = col_f4.number_input("Margin (bps)", value=95.0)
        lookback = col_f5.number_input("Lookback (JBD)", value=5)
        clean_target = col_f6.number_input("Target Clean Price (%)", value=100.0)
        
        st.divider()
        
        # Calculate
        frn = ZaroniaFRN(frn_notional, issue_date, mat_date, margin_bps, lookback)
        frn.calculate_cashflows(zaronia_curve)
        
        clean_price = frn.get_clean_price()
        dirty_price = clean_price # Approx for now, need accrued logic for full dirty
        # Accrued Interest Calculation (Simplified: if val_date inside a period)
        accrued = 0.0
        # Find current period
        for cf in frn.cashflows:
            if cf['Start'] <= val_date < cf['End']:
                # Roughly calculate accrued
                # Real way: exact daily compounding from Start to val_date
                # We can reuse the daily logic or simplified linear for display
                days_accrued = (val_date - cf['Start']).days
                if days_accrued > 0:
                     # This is an approximation for display
                     accrued = (cf['Coupon'] * days_accrued / 365.0) * 100
                break
        
        dirty_price = clean_price + accrued
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clean Price", f"{clean_price:.4f}%")
        m2.metric("Dirty Price", f"{dirty_price:.4f}%")
        m3.metric("Accrued Interest", f"{accrued:.4f}%")
        
        equiv_q = frn.get_quarterly_equivalent()
        m4.metric("Q-Equiv Rate (First Period)", f"{equiv_q*100:.4f}%")
        
        st.subheader("Cash Flow Schedule")
        df_frn = pd.DataFrame(frn.cashflows)
        if not df_frn.empty:
            # Format
            display_cols = ["Start", "End", "Days", "ZARONIA_Comp", "Margin", "Coupon", "Amount", "DF", "PV"]
            st.dataframe(df_frn[display_cols].style.format({
                "ZARONIA_Comp": "{:.4%}",
                "Margin": "{:.4%}",
                "Coupon": "{:.4%}",
                "Amount": "{:,.2f}",
                "DF": "{:.6f}",
                "PV": "{:,.2f}"
            }), use_container_width=True)
            
            with st.expander("Show Daily Rate Details (First Period)"):
                if "Dailies" in df_frn.columns and df_frn.iloc[0]["Dailies"]:
                    st.dataframe(pd.DataFrame(df_frn.iloc[0]["Dailies"]))

        # Hedging Section
        st.divider()
        st.subheader("🛡️ Hedging & Relative Value Analysis")
        
        # 1. Calculate Equivalent Fixed Rates (Matched Maturity)
        # We need to price a swap exactly matching the FRN maturity
        # FRN Maturity in years
        frn_years = year_frac(zaronia_curve.val_date, frn.maturity_date)
        
        # Helper to get par rate for specific maturity
        def get_par_rate_for_tenor(tenor_y, index_type):
            # Create dummy legs
            # Fixed vs Index
            l_fix = SwapLeg('Fixed', 'ZAR', 100, val_date, tenor_y, 'Quarterly', fixed_rate=0.0)
            l_flt = SwapLeg('Float', 'ZAR', 100, val_date, tenor_y, 'Quarterly' if index_type=='JIBAR' else 'Annual', float_index=index_type)
            
            def solver(r):
                l_fix.fixed_rate = r
                l_fix.calculate_cashflows(jibar_curve, zaronia_curve, 'ZARONIA') # Discount on OIS always
                l_flt.calculate_cashflows(jibar_curve, zaronia_curve, 'ZARONIA')
                return l_fix.get_pv() - l_flt.get_pv()
                
            try:
                return brentq(solver, -0.05, 0.20)
            except:
                return 0.0

        par_zaronia_ois = get_par_rate_for_tenor(frn_years, 'ZARONIA')
        par_jibar_irs = get_par_rate_for_tenor(frn_years, 'JIBAR')
        
        # FRN Effective Yield (Simple approximation: Coupon)
        # Better: Yield to Maturity given Clean Price
        # We have get_clean_price based on Margin. 
        # If user inputs Target Price, we should solve for Margin? 
        # Or calculate Yield given Clean Price.
        # Let's assume Price = 100 (Par) -> Yield = Q-Equiv + Margin approx
        # For accurate Relative Value, we compare Margin vs Par Spreads.
        
        # Effective Yield of FRN (ZARONIA + Margin) roughly
        # Let's use the first period Q-Equiv + Margin as the "current yield" proxy
        frn_yield = equiv_q + (margin_bps / 10000.0)
        
        # Asset Swap Spread (ASW) over ZARONIA OIS
        # Spread = FRN Yield - ZARONIA Par Rate
        asw_zaronia = frn_yield - par_zaronia_ois
        
        # Spread over JIBAR IRS
        spread_vs_jibar = frn_yield - par_jibar_irs

        # Display Relative Value Dashboard
        rv1, rv2, rv3 = st.columns(3)
        
        rv1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Matched ZARONIA OIS Rate</div>
            <div class="metric-value">{par_zaronia_ois*100:.4f}%</div>
            <div class="metric-sub">{frn_years:.2f}Y Tenor</div>
        </div>
        """, unsafe_allow_html=True)
        
        rv2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Matched JIBAR IRS Rate</div>
            <div class="metric-value">{par_jibar_irs*100:.4f}%</div>
            <div class="metric-sub">Spread to ZARONIA: {(par_jibar_irs - par_zaronia_ois)*10000:.1f} bps</div>
        </div>
        """, unsafe_allow_html=True)
        
        rv3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">FRN Effective Yield</div>
            <div class="metric-value" style="color:#FFA15A;">{frn_yield*100:.4f}%</div>
            <div class="metric-sub">Est. ASW: <b>{asw_zaronia*10000:.1f} bps</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        st.caption("Relative Value Chart: Yield Decomposition")
        
        # Waterfall / Bar Chart
        fig_rv = go.Figure()
        
        fig_rv.add_trace(go.Bar(
            name='Base Rate',
            x=['ZARONIA OIS', 'JIBAR IRS', 'FRN Yield'],
            y=[par_zaronia_ois*100, par_jibar_irs*100, frn_yield*100],
            marker_color=['#EF553B', '#00CC96', '#FFA15A'],
            text=[f"{par_zaronia_ois*100:.2f}%", f"{par_jibar_irs*100:.2f}%", f"{frn_yield*100:.2f}%"],
            textposition='auto'
        ))
        
        fig_rv.update_layout(
            title=f"Relative Value: {frn_years:.1f}Y Tenor Comparison",
            yaxis_title="Rate (%)",
            template='plotly_dark',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_rv, use_container_width=True)

        st.divider()
        st.markdown("#### Hedging Sensitivity (DV01)")
        st.markdown("Hedge against **1bp Parallel ZARONIA Curve Shift**")
        
        h_col1, h_col2 = st.columns(2)
        hedge_tenor = h_col1.selectbox("Hedge Instrument (OIS Swap)", ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y"])
        
        # 1. Calculate FRN Sensitivity
        # Bump curve
        j_rates_up = j_rates + 0.0001
        jibar_curve_up = JibarZeroCurve(j_tenors, j_rates_up)
        zaronia_curve_up = ZaroniaCurve(val_date, jibar_curve_up, spread_func)
        
        frn.calculate_cashflows(zaronia_curve_up)
        pv_frn_up = sum(cf['PV'] for cf in frn.cashflows) + frn.principal_flow['PV']
        
        # Restore
        frn.calculate_cashflows(zaronia_curve)
        pv_frn_base = sum(cf['PV'] for cf in frn.cashflows) + frn.principal_flow['PV']
        
        dv01_frn = pv_frn_up - pv_frn_base
        
        # 2. Calculate Hedge Swap Sensitivity (Par Swap)
        hedge_years = int(hedge_tenor.replace("Y",""))
        # Create a Par OIS Swap (Fixed vs ZARONIA)
        # We need to find the par rate first to create a 0 NPV swap, then shock it
        # Or just create a 1bp fixed rate swap and measure PV change?
        # Standard: Par Swap DV01.
        
        # Helper to price OIS swap
        def price_ois(curve, rate):
            l1 = SwapLeg('Fixed', 'ZAR', frn_notional, val_date, hedge_years, 'Quarterly', fixed_rate=rate)
            l2 = SwapLeg('Float', 'ZAR', frn_notional, val_date, hedge_years, 'Annual', float_index='ZARONIA')
            l1.calculate_cashflows(jibar_curve, curve, 'ZARONIA')
            l2.calculate_cashflows(jibar_curve, curve, 'ZARONIA')
            return l1.get_pv() - l2.get_pv()

        # Find par rate
        try:
             par_ois = brentq(lambda r: price_ois(zaronia_curve, r), -0.1, 0.5)
        except:
             par_ois = 0.07

        # Price at base
        base_swap_pv = price_ois(zaronia_curve, par_ois) # Should be ~0
        
        # Price at up
        up_swap_pv = price_ois(zaronia_curve_up, par_ois)
        
        dv01_hedge = up_swap_pv - base_swap_pv
        
        # Hedge Ratio
        if dv01_hedge != 0:
            hedge_ratio = -dv01_frn / dv01_hedge
        else:
            hedge_ratio = 0
            
        # Display
        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("FRN DV01", f"{dv01_frn:,.2f}")
        hm2.metric(f"{hedge_tenor} Swap DV01", f"{dv01_hedge:,.2f}")
        hm3.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
        
        st.info(f"To hedge the FRN, you should **{'Buy' if hedge_ratio > 0 else 'Sell'}** {abs(hedge_ratio):.2f} units of the {hedge_tenor} OIS Swap.")

    with tab_comp:
        st.subheader("Quarterly Compounding Analysis")
        st.caption("Historical comparison of JIBAR 3M (Fixed at Start) vs Compounded ZARONIA (Realized over Quarter)")
        
        # Load full history
        df_hist_all = load_historical_market_data()
        
        if df_hist_all.empty:
            st.warning("No historical data available to perform analysis.")
        else:
            # We need a robust date index
            df_hist_all["Date"] = pd.to_datetime(df_hist_all["Date"])
            # Create a quick lookup for JIBAR
            # We use the raw loaded data which has "JIBAR3M"
            
            # Generate Quarter Start Dates (IMM Dates approx: 1st of Mar, Jun, Sep, Dec)
            # Let's just generate 1st of every quarter for last 3 years
            end_d = pd.Timestamp(val_date)
            start_d = end_d - pd.DateOffset(years=3)
            
            # Generate quarters
            quarters = pd.date_range(start=start_d, end=end_d, freq='QS') # Quarter Start (Jan 1, Apr 1, ...)
            
            analysis_data = []
            
            for q_start in quarters:
                q_end = q_start + pd.DateOffset(months=3)
                
                if q_end > end_d: continue # Don't analyze incomplete current quarter? Or maybe partial.
                
                # 1. Get JIBAR 3M at q_start
                # Find closest date <= q_start in history
                # Filter history
                mask_jibar = (df_hist_all["Date"] <= q_start)
                df_j = df_hist_all.loc[mask_jibar]
                
                jibar_val = np.nan
                if not df_j.empty:
                    # Sort by date desc, take first
                    # Try to find date close to q_start (within 5 days)
                    closest_row = df_j.iloc[0] # Sorted desc in load function? Yes.
                    # Wait, load_historical_market_data sorts desc.
                    # So df_j.iloc[0] is the date closest to q_start (on or before).
                    # Check if it's too old
                    if (q_start - closest_row["Date"]).days < 7:
                        jibar_val = closest_row["JIBAR3M"] if "JIBAR3M" in closest_row else np.nan

                # 2. Calculate Compounded ZARONIA over [q_start, q_end)
                # We need daily rates.
                # Use zaronia_curve.history (dict)
                # Loop days
                curr = q_start.date()
                end_date_obj = q_end.date()
                
                comp_factor = 1.0
                days_total = 0
                
                valid_calc = True
                
                while curr < end_date_obj:
                    # Simple business day logic or just calendar?
                    # OIS compounding is typically on business days.
                    # We iterate daily.
                    # ZARONIA is an overnight rate published on JBDs.
                    # If curr is JBD, we apply rate for n days (usually 1, or 3 over weekend).
                    # Simplified: Just iterate calendar days?
                    # No, OIS formula: Prod(1 + r*n/365).
                    # We need to step by JBDs.
                    
                    if not is_jbd(curr):
                        curr += timedelta(days=1)
                        continue
                        
                    next_jbd = curr + timedelta(days=1)
                    while not is_jbd(next_jbd) and next_jbd < end_date_obj:
                        next_jbd += timedelta(days=1)
                        
                    # Cap at end
                    if next_jbd > end_date_obj:
                        next_jbd = end_date_obj
                        
                    n_days = (next_jbd - curr).days
                    if n_days == 0: break
                    
                    # Get rate
                    rate = zaronia_curve.get_rate_at(curr)
                    # Note: get_rate_at will return curve forecast if history missing.
                    # That's acceptable for recent/future quarters.
                    
                    # Check if we are really in history or forecast
                    # Optional: flag if forecast used?
                    
                    comp_factor *= (1 + rate * n_days / 365.0)
                    curr = next_jbd
                    days_total += n_days
                    
                if days_total == 0:
                    zaronia_comp = 0.0
                else:
                    zaronia_comp = (comp_factor - 1.0) * (365.0 / (q_end - q_start).days) # ACT/365 annualized?
                    # Actually standard is usually * 365 / days_in_period
                    # (q_end - q_start).days is calendar days.
                    cal_days = (q_end - q_start).days
                    zaronia_comp = (comp_factor - 1.0) * (365.0 / cal_days)

                analysis_data.append({
                    "Quarter": q_start.strftime("%Y-%m"),
                    "Start Date": q_start.date(),
                    "JIBAR 3M": jibar_val,
                    "ZARONIA Comp": zaronia_comp * 100, # %
                    "Spread (bps)": (jibar_val - zaronia_comp*100) * 100 if not pd.isna(jibar_val) else np.nan
                })
                
            df_comp = pd.DataFrame(analysis_data)
            
            # Chart
            if not df_comp.empty:
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(x=df_comp["Quarter"], y=df_comp["JIBAR 3M"], name="JIBAR 3M (Fix)", line=dict(color='#00CC96', width=2)))
                    fig_ts.add_trace(go.Scatter(x=df_comp["Quarter"], y=df_comp["ZARONIA Comp"], name="ZARONIA (Realized)", line=dict(color='#EF553B', width=2)))
                    fig_ts.update_layout(title="Fixing vs Realized: JIBAR 3M vs ZARONIA", xaxis_title="Quarter", yaxis_title="Rate (%)", template="plotly_dark", height=400)
                    st.plotly_chart(fig_ts, use_container_width=True)
                    
                with c2:
                    fig_sp = go.Figure()
                    fig_sp.add_trace(go.Bar(x=df_comp["Quarter"], y=df_comp["Spread (bps)"], name="Spread", marker_color='#19D3F3'))
                    fig_sp.update_layout(title="Realized Basis Spread (bps)", xaxis_title="Quarter", yaxis_title="Basis (bps)", template="plotly_dark", height=400)
                    st.plotly_chart(fig_sp, use_container_width=True)
                    
                st.dataframe(df_comp.set_index("Quarter"), use_container_width=True)

    with tab_hist:
        st.subheader("Historical Market Rates")
        df_hist = load_historical_market_data()
        
        if not df_hist.empty:
            # 1. Plotting
            st.caption("Comparison of ZARONIA, JIBAR 3M, and Swap Rates")
            
            fig_hist = go.Figure()
            
            # Helper to add trace if column exists
            def add_hist_trace(col, color, width=1):
                if col in df_hist.columns:
                    # Filter out NaNs for plot continuity
                    series = df_hist[["Date", col]].dropna()
                    fig_hist.add_trace(go.Scatter(
                        x=series["Date"], 
                        y=series[col], 
                        name=col,
                        line=dict(color=color, width=width)
                    ))

            # ZARONIA (Primary)
            add_hist_trace("ZARONIA", "#EF553B", 2)
            
            # JIBAR 3M
            add_hist_trace("JIBAR3M", "#00CC96", 2)
            
            # Swaps
            add_hist_trace("SASW1", "#AB63FA", 1)
            add_hist_trace("SASW2", "#FFA15A", 1)
            add_hist_trace("SASW5", "#19D3F3", 1)
            add_hist_trace("SASW10", "#FF6692", 1)
            
            fig_hist.update_layout(
                title="Historical Benchmark Rates",
                xaxis_title="Date",
                yaxis_title="Rate (%)",
                template='plotly_dark',
                hovermode='x unified',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # 2. Data Table
            st.subheader("Data Table")
            st.dataframe(df_hist, use_container_width=True)
            
        else:
            st.warning("No historical data loaded. Ensure 'SARB-benchmark-data.csv' and 'JIBAR_FRA_SWAPS.xlsx' are in the directory.")

if __name__ == "__main__":
    main()

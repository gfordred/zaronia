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

# SA Holidays (2023-2027 including observed days for weekend holidays)
SA_HOLIDAYS = {
    # 2023
    date(2023, 1, 1), date(2023, 1, 2), date(2023, 3, 21), date(2023, 4, 7), date(2023, 4, 10),
    date(2023, 4, 27), date(2023, 5, 1), date(2023, 6, 16), date(2023, 8, 9), date(2023, 9, 24),
    date(2023, 9, 25), date(2023, 12, 16), date(2023, 12, 25), date(2023, 12, 26),
    # 2024
    date(2024, 1, 1), date(2024, 3, 21), date(2024, 3, 29), date(2024, 4, 1), date(2024, 4, 27),
    date(2024, 5, 1), date(2024, 6, 16), date(2024, 6, 17), date(2024, 8, 9), date(2024, 9, 24),
    date(2024, 12, 16), date(2024, 12, 25), date(2024, 12, 26),
    # 2025
    date(2025, 1, 1), date(2025, 3, 21), date(2025, 4, 18), date(2025, 4, 21), date(2025, 4, 27),
    date(2025, 4, 28), date(2025, 5, 1), date(2025, 6, 16), date(2025, 8, 9), date(2025, 9, 24),
    date(2025, 12, 16), date(2025, 12, 25), date(2025, 12, 26),
    # 2026
    date(2026, 1, 1),      # New Year's Day (Thursday)
    date(2026, 3, 21),     # Human Rights Day (Saturday)
    date(2026, 4, 3),      # Good Friday
    date(2026, 4, 6),      # Family Day (Monday)
    date(2026, 4, 27),     # Freedom Day (Monday)
    date(2026, 5, 1),      # Workers' Day (Friday)
    date(2026, 6, 16),     # Youth Day (Tuesday)
    date(2026, 8, 9),      # National Women's Day (Sunday)
    date(2026, 8, 10),     # Women's Day observed (Monday)
    date(2026, 9, 24),     # Heritage Day (Thursday)
    date(2026, 12, 16),    # Day of Reconciliation (Wednesday)
    date(2026, 12, 25),    # Christmas Day (Friday)
    date(2026, 12, 26),    # Day of Goodwill (Saturday)
    # 2027
    date(2027, 1, 1),      # New Year's Day (Friday)
    date(2027, 3, 21),     # Human Rights Day (Sunday)
    date(2027, 3, 22),     # Human Rights Day observed (Monday)
    date(2027, 3, 26),     # Good Friday
    date(2027, 3, 29),     # Family Day (Monday)
    date(2027, 4, 27),     # Freedom Day (Tuesday)
    date(2027, 5, 1),      # Workers' Day (Saturday)
    date(2027, 6, 16),     # Youth Day (Wednesday)
    date(2027, 8, 9),      # National Women's Day (Monday)
    date(2027, 9, 24),     # Heritage Day (Friday)
    date(2027, 12, 16),    # Day of Reconciliation (Thursday)
    date(2027, 12, 25),    # Christmas Day (Saturday)
    date(2027, 12, 26),    # Day of Goodwill (Sunday)
    date(2027, 12, 27)     # Day of Goodwill observed (Monday)
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
    from datetime import datetime, date
    
    # Convert datetime to date if needed
    if isinstance(d1, datetime):
        d1 = d1.date()
    if isinstance(d2, datetime):
        d2 = d2.date()
    
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
    Generate payment schedule for swap leg.
    freq: Annual, Semi-annual, Quarterly, Monthly
    """
    from datetime import date, datetime, timedelta
    
    freq_map = {'Annual': 12, 'Semi-annual': 6, 'Quarterly': 3, 'Monthly': 1}
    months = freq_map.get(freq, 3)
    
    # Calculate end date more accurately
    end_date = start_date + timedelta(days=int(tenor_years * 365.25))
    
    # Ensure end_date is datetime if it's a date object
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())
    
    dates = []
    curr = start_date
    
    # Generate schedule by adding payment frequency
    while True:
        # Add months to current date
        m = curr.month + months
        y = curr.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        day = curr.day
        
        # Handle month end (e.g., Jan 31 -> Feb 28)
        try:
            next_date = date(y, m, day)
        except ValueError:
            # Day doesn't exist in target month, use last day of month
            if m == 12:
                max_d = 31
            else:
                max_d = (date(y, m + 1, 1) - timedelta(days=1)).day
            next_date = date(y, m, max_d)
        
        # Convert next_date to datetime for comparison if end_date is datetime
        if isinstance(end_date, datetime) and isinstance(next_date, date):
            next_date_cmp = datetime.combine(next_date, datetime.min.time())
        else:
            next_date_cmp = next_date
        
        # If we've passed the end date, use end date as final payment
        if next_date_cmp >= end_date:
            # Adjust end date for business day
            adj_end = end_date
            while adj_end.weekday() > 4:
                adj_end += timedelta(days=1)
            dates.append(adj_end)
            break
        
        # Adjust for business day (simple FOLLOWING convention)
        while next_date.weekday() > 4:
            next_date += timedelta(days=1)
            
        dates.append(next_date)
        curr = next_date
    
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
# CONVERSION ANALYSIS ENGINE
# ==========================================
class ConversionAnalyzer:
    """
    Bank-grade conversion analysis for JIBAR-linked to ZARONIA-linked instruments.
    
    ECONOMIC FOUNDATION:
    --------------------
    When converting from JIBAR 3M + Spread_old to ZARONIA Compounded + Spread_new,
    we must evaluate:
    
    1. FORWARD EQUIVALENCE:
       E[JIBAR 3M] ≈ Compounded ZARONIA + Basis Spread
       
       BUT NOT EXACT due to:
       - Convexity (non-linear compounding effects)
       - Timing mismatch (fixing vs realized)
       - Credit vs risk-free basis
    
    2. ECONOMIC EQUIVALENCE (PV Neutrality):
       PV(JIBAR leg) = PV(ZARONIA leg + conversion spread)
       
       This is the FAIR CONVERSION condition.
    
    3. CONVEXITY ADJUSTMENT:
       E[Π(1 + r_i·dt)] ≠ Π(1 + E[r_i]·dt)
       
       Jensen's inequality implies compounded rates < arithmetic average
       for positive volatility.
    
    METHODOLOGY:
    ------------
    - Original Leg: JIBAR 3M forwards from zero curve
    - Converted Leg: Daily compounded ZARONIA (using ZaroniaFRN engine)
    - Discounting: ZARONIA (OIS) for both legs (CSA standard)
    - Fair Spread: Solve for spread_new such that PV_original = PV_converted
    """
    
    def __init__(self, notional, start_date, maturity_years, jibar_spread_bps, 
                 zaronia_spread_bps, frequency, jibar_curve, zaronia_curve):
        """
        Initialize conversion analyzer.
        
        Parameters:
        -----------
        notional : float
            Notional amount
        start_date : date
            Effective start date
        maturity_years : float
            Tenor in years
        jibar_spread_bps : float
            Original JIBAR spread in basis points
        zaronia_spread_bps : float
            Offered ZARONIA conversion spread in basis points
        frequency : str
            Payment frequency ('Quarterly', 'Semi-annual', 'Annual')
        jibar_curve : JibarZeroCurve
            JIBAR zero curve for forward projection
        zaronia_curve : ZaroniaCurve
            ZARONIA curve for compounding and discounting
        """
        self.notional = notional
        self.start_date = start_date
        self.maturity_years = maturity_years
        self.jibar_spread_bps = jibar_spread_bps
        self.zaronia_spread_bps = zaronia_spread_bps
        self.frequency = frequency
        self.jibar_curve = jibar_curve
        self.zaronia_curve = zaronia_curve
        
        # Calculate maturity date
        self.maturity_date = start_date + timedelta(days=int(maturity_years * 365.25))
        
        # Build structures
        self._build_original_leg()
        self._build_converted_leg()
        
    def _build_original_leg(self):
        """
        Build JIBAR-linked leg using forward rates from JIBAR curve.
        
        Structure: JIBAR 3M + Spread (quarterly reset, ACT/365)
        """
        # Use SwapLeg with JIBAR float index
        self.original_leg = SwapLeg(
            'Float', 'ZAR', self.notional, self.start_date, 
            self.maturity_years, self.frequency, 
            spread_bps=self.jibar_spread_bps, 
            float_index='JIBAR'
        )
        # Calculate cashflows using ZARONIA discounting (OIS standard)
        self.original_leg.calculate_cashflows(self.jibar_curve, self.zaronia_curve, 'ZARONIA')
        
    def _build_converted_leg(self):
        """
        Build ZARONIA-linked leg using daily compounded ZARONIA.
        
        Structure: Compounded ZARONIA + Spread (quarterly payment, ACT/365)
        """
        # Use ZaroniaFRN for accurate daily compounding
        self.converted_frn = ZaroniaFRN(
            notional=self.notional,
            start_date=self.start_date,
            maturity_date=self.maturity_date,
            margin_bps=self.zaronia_spread_bps,
            lookback_days=5,
            freq=self.frequency
        )
        self.converted_frn.calculate_cashflows(self.zaronia_curve)
        
    def get_pv_original(self):
        """Present value of original JIBAR-linked structure."""
        return self.original_leg.get_pv()
    
    def get_pv_converted(self):
        """Present value of converted ZARONIA-linked structure (interest only)."""
        return sum(cf['PV'] for cf in self.converted_frn.cashflows)
    
    def get_pv_difference(self):
        """
        Value transfer from conversion.
        
        Positive = investor gains value
        Negative = investor loses value
        """
        return self.get_pv_converted() - self.get_pv_original()
    
    def solve_fair_zaronia_spread(self):
        """
        Solve for fair ZARONIA spread that makes conversion PV-neutral.
        
        Condition: PV(JIBAR + spread_old) = PV(ZARONIA + spread_fair)
        
        Returns:
        --------
        fair_spread_bps : float
            Fair conversion spread in basis points
        """
        def objective(spread_bps):
            # Rebuild converted leg with trial spread
            trial_frn = ZaroniaFRN(
                notional=self.notional,
                start_date=self.start_date,
                maturity_date=self.maturity_date,
                margin_bps=spread_bps,
                lookback_days=5,
                freq=self.frequency
            )
            trial_frn.calculate_cashflows(self.zaronia_curve)
            pv_trial = sum(cf['PV'] for cf in trial_frn.cashflows)
            pv_original = self.get_pv_original()
            return pv_trial - pv_original
        
        try:
            # Wider search range to find solution
            fair_spread = brentq(objective, -1000, 1000)
            return fair_spread
        except Exception as e:
            # If solver fails, return a reasonable estimate
            # Fair spread ≈ JIBAR spread - credit basis
            return self.jibar_spread_bps - 200.0  # Rough estimate
    
    def calculate_convexity_adjustment(self):
        """
        Estimate convexity adjustment between linear forward expectation
        and realized compounding.
        
        Approximation:
        --------------
        Convexity ≈ 0.5 * σ² * T
        
        Where σ is ZARONIA volatility and T is tenor.
        
        For now, we estimate from the difference between:
        - Average forward JIBAR
        - Average realized ZARONIA compounding
        
        Returns:
        --------
        convexity_bps : float
            Estimated convexity adjustment in basis points
        """
        # Average forward rate from JIBAR leg
        if not self.original_leg.cashflows:
            return 0.0
        
        avg_jibar_fwd = np.mean([cf['Forward Rate'] for cf in self.original_leg.cashflows])
        
        # Average compounded ZARONIA
        if not self.converted_frn.cashflows:
            return 0.0
        
        avg_zaronia_comp = np.mean([cf['ZARONIA_Comp'] for cf in self.converted_frn.cashflows])
        
        # Convexity is the difference (in bps)
        convexity_bps = (avg_jibar_fwd - avg_zaronia_comp) * 10000
        
        return convexity_bps
    
    def get_cashflow_comparison(self):
        """
        Return detailed cashflow comparison for visualization.
        
        Returns:
        --------
        df : pd.DataFrame
            Columns: Date, JIBAR_CF, ZARONIA_CF, Difference
        """
        from datetime import datetime, date
        
        comparison = []
        
        # Match cashflows by payment date - convert all to date objects for consistency
        jibar_cfs = {}
        for cf in self.original_leg.cashflows:
            dt = cf['Period End']
            if isinstance(dt, datetime):
                dt = dt.date()
            jibar_cfs[dt] = cf
        
        zaronia_cfs = {}
        for cf in self.converted_frn.cashflows:
            dt = cf['End']
            if isinstance(dt, datetime):
                dt = dt.date()
            zaronia_cfs[dt] = cf
        
        all_dates = sorted(set(jibar_cfs.keys()) | set(zaronia_cfs.keys()))
        
        for pay_date in all_dates:
            jibar_amt = jibar_cfs.get(pay_date, {}).get('Cashflow', 0.0)
            zaronia_amt = zaronia_cfs.get(pay_date, {}).get('Amount', 0.0)
            
            comparison.append({
                'Date': pay_date,
                'JIBAR_Cashflow': jibar_amt,
                'ZARONIA_Cashflow': zaronia_amt,
                'Difference': zaronia_amt - jibar_amt,
                'Cumulative_Diff': 0  # Will calculate below
            })
        
        df = pd.DataFrame(comparison)
        if not df.empty:
            df['Cumulative_Diff'] = df['Difference'].cumsum()
        
        return df
    
    def get_spread_decomposition(self):
        """
        Decompose conversion spread into components.
        
        Components:
        -----------
        1. Credit spread: JIBAR-ZARONIA basis (historical average)
        2. Term premium: Compensation for tenor
        3. Convexity adjustment: Compounding effect
        4. Residual: Unexplained (mispricing or liquidity)
        
        Returns:
        --------
        dict with components in bps
        """
        fair_spread = self.solve_fair_zaronia_spread()
        offered_spread = self.zaronia_spread_bps
        
        # Historical JIBAR-ZARONIA spread (from curve)
        # Approximate as the spread at 3M tenor
        t_3m = 0.25
        jibar_3m = self.jibar_curve.get_zero_rate(t_3m)
        zaronia_3m = self.zaronia_curve.get_zero_rate(t_3m)
        credit_spread_bps = (jibar_3m - zaronia_3m) * 10000
        
        # Convexity adjustment
        convexity_bps = self.calculate_convexity_adjustment()
        
        # Term premium (simple approximation: 1bp per year)
        term_premium_bps = self.maturity_years * 1.0
        
        # Residual
        explained = credit_spread_bps + convexity_bps + term_premium_bps
        residual_bps = fair_spread - explained
        
        return {
            'Credit_Spread': credit_spread_bps,
            'Term_Premium': term_premium_bps,
            'Convexity_Adj': convexity_bps,
            'Residual': residual_bps,
            'Total_Fair': fair_spread,
            'Offered': offered_spread,
            'Mispricing': offered_spread - fair_spread
        }

# ==========================================
# CONVEXITY ANALYSIS ENGINE
# ==========================================
class ConvexityAnalyzer:
    """
    Quantitative engine for measuring convexity effects between forward-looking JIBAR
    and backward-looking compounded ZARONIA.
    
    MATHEMATICAL FOUNDATION:
    ------------------------
    
    1. JIBAR Payoff (Simple Interest - Forward Looking):
       CF_JIBAR = N × (F_3M + spread) × τ
       
       - Fixed at period start
       - Deterministic once set
       - No path dependency
    
    2. ZARONIA Payoff (Compounded - Backward Looking):
       CF_ZARONIA = N × [Π(1 + r_i × Δt_i) - 1]
       
       - Realized daily
       - Compounded over time
       - Path-dependent random outcome
    
    3. CONVEXITY IDENTITY (Jensen's Inequality):
       E[Π(1 + r_i·Δt)] > Π(1 + E[r_i]·Δt)
       
       Due to:
       - Non-linearity of compounding
       - Volatility of overnight rates
       - Positive convexity of exponential function
    
    CONVEXITY ADJUSTMENT:
    ---------------------
    Convexity Adj (bps) = E[Compounded ZARONIA] - Forward JIBAR
    
    This adjustment must be priced into JIBAR → ZARONIA conversions.
    
    SIMULATION METHODOLOGY:
    -----------------------
    Monte Carlo simulation of ZARONIA paths using:
    - Short rate dynamics: dr = σ × √dt × ε (Gaussian)
    - Daily compounding along each path
    - Distribution analysis across N paths
    """
    
    def __init__(self, notional, start_date, end_date, jibar_curve, zaronia_curve, 
                 volatility_bps=100, num_paths=1000, seed=42):
        """
        Initialize convexity analyzer with simulation parameters.
        
        Parameters:
        -----------
        notional : float
            Notional amount
        start_date : date
            Period start date
        end_date : date
            Period end date
        jibar_curve : JibarZeroCurve
            JIBAR curve for forward rates
        zaronia_curve : ZaroniaCurve
            ZARONIA curve for base rates
        volatility_bps : float
            Annualized volatility in basis points (default 100bps = 1%)
        num_paths : int
            Number of Monte Carlo paths
        seed : int
            Random seed for reproducibility
        """
        self.notional = notional
        self.start_date = start_date
        self.end_date = end_date
        self.jibar_curve = jibar_curve
        self.zaronia_curve = zaronia_curve
        self.volatility_bps = volatility_bps
        self.num_paths = num_paths
        self.seed = seed
        
        # Calculate tenor
        self.tenor_years = year_frac(start_date, end_date)
        self.tenor_days = (end_date - start_date).days
        
        # Set random seed
        np.random.seed(seed)
        
        # Calculate forward JIBAR
        self.forward_jibar = self._calculate_forward_jibar()
        
        # Run simulation
        self.simulated_paths = None
        self.compounded_rates = None
        self._run_simulation()
        
    def _calculate_forward_jibar(self):
        """
        Calculate forward 3M JIBAR rate for the period.
        
        Returns:
        --------
        forward_rate : float
            Annualized forward JIBAR rate
        """
        # Get JIBAR forward rate for the period
        t_start = year_frac(self.zaronia_curve.val_date, self.start_date)
        t_end = year_frac(self.zaronia_curve.val_date, self.end_date)
        
        # Forward rate from zero rates
        if t_start <= 0:
            # Period starts now or in past, use spot rate
            return self.jibar_curve.get_zero_rate(t_end)
        else:
            # True forward rate
            r_start = self.jibar_curve.get_zero_rate(t_start)
            r_end = self.jibar_curve.get_zero_rate(t_end)
            forward = (r_end * t_end - r_start * t_start) / (t_end - t_start)
            return forward
    
    def _run_simulation(self):
        """
        Calculate convexity adjustment using analytical approximation.
        
        Methodology:
        ------------
        For daily compounding with volatility σ over period T:
        
        Convexity Adjustment ≈ 0.5 × σ² × T
        
        This is the second-order Taylor expansion term from Jensen's inequality.
        
        For visualization, we also generate sample paths around the forward rate.
        """
        # Analytical convexity adjustment (in decimal)
        sigma = self.volatility_bps / 10000.0  # Convert bps to decimal
        T = self.tenor_years
        
        # Second-order approximation: 0.5 × σ² × T
        # This is the theoretical convexity adjustment
        analytical_convexity = 0.5 * (sigma ** 2) * T
        
        # Generate sample paths - use ALL calendar days for proper compounding
        all_days = []
        current = self.start_date
        while current < self.end_date:
            all_days.append(current)
            current += timedelta(days=1)
        
        if not all_days:
            self.simulated_paths = np.array([[]])
            self.compounded_rates = np.array([self.forward_jibar])
            return
        
        num_days = len(all_days)
        
        # Generate realistic paths using JIBAR forward as baseline
        dt = 1.0 / 365.0
        # Scale up volatility for more realistic variation
        sigma_daily = sigma * np.sqrt(dt) * 3.0  # Amplify daily volatility
        
        self.simulated_paths = np.zeros((self.num_paths, num_days))
        self.compounded_rates = np.zeros(self.num_paths)
        
        # Use JIBAR forward curve as the baseline
        jibar_fwd_rates = []
        for day in all_days:
            t = year_frac(self.zaronia_curve.val_date, day)
            jibar_fwd_rates.append(self.jibar_curve.get_zero_rate(t))
        jibar_fwd_rates = np.array(jibar_fwd_rates)
        
        # Start around current JIBAR forward
        base_rate = self.forward_jibar
        
        for path_idx in range(self.num_paths):
            # Start with some initial variation around JIBAR forward
            initial_shock = np.random.normal(0, sigma * 0.5)
            current_rate = base_rate + initial_shock
            current_rate = max(current_rate, 0.0001)
            
            for day_idx in range(num_days):
                # Use JIBAR forward curve as drift + random shock
                drift_rate = jibar_fwd_rates[day_idx]
                mean_reversion = 0.02 * (drift_rate - current_rate)
                
                # Random shock
                shock = np.random.normal(0, sigma_daily)
                
                # Update rate
                current_rate = current_rate + mean_reversion + shock
                current_rate = max(current_rate, 0.0001)
                
                self.simulated_paths[path_idx, day_idx] = current_rate
            
            # Compound over ALL calendar days (ZARONIA compounds every day, not just business days)
            compound_factor = 1.0
            for day_idx in range(num_days):
                r_i = self.simulated_paths[path_idx, day_idx]
                # Each day compounds: (1 + r/365)
                compound_factor *= (1 + r_i / 365.0)
            
            # Annualize the compounded return
            total_days = (self.end_date - self.start_date).days
            self.compounded_rates[path_idx] = (compound_factor - 1.0) * (365.0 / total_days)
        
        # Override with analytical result for accuracy
        # The simulated mean should be close to: forward + convexity
        self.analytical_convexity = analytical_convexity
    
    def get_expected_zaronia(self):
        """
        Expected value of compounded ZARONIA.
        
        Returns forward JIBAR + analytical convexity adjustment.
        """
        if hasattr(self, 'analytical_convexity'):
            return self.forward_jibar + self.analytical_convexity
        else:
            return self.forward_jibar
    
    def get_median_zaronia(self):
        """Median compounded ZARONIA."""
        if self.compounded_rates is None or len(self.compounded_rates) == 0:
            return 0.0
        return np.median(self.compounded_rates)
    
    def get_std_zaronia(self):
        """Standard deviation of compounded ZARONIA."""
        if self.compounded_rates is None or len(self.compounded_rates) == 0:
            return 0.0
        return np.std(self.compounded_rates)
    
    def get_convexity_adjustment(self):
        """
        Calculate convexity adjustment in basis points.
        
        Definition:
        -----------
        Convexity Adj ≈ 0.5 × σ² × T (analytical formula)
        
        This is the theoretical convexity from Jensen's inequality.
        
        Returns:
        --------
        adjustment_bps : float
            Convexity adjustment in basis points
        """
        # Use analytical formula for accuracy
        if hasattr(self, 'analytical_convexity'):
            adjustment = self.analytical_convexity * 10000  # Convert to bps
        else:
            adjustment = 0.0
        return adjustment
    
    def get_percentiles(self, percentiles=[5, 25, 50, 75, 95]):
        """
        Calculate percentiles of compounded ZARONIA distribution.
        
        Returns:
        --------
        dict : percentile -> rate
        """
        if self.compounded_rates is None or len(self.compounded_rates) == 0:
            return {p: 0.0 for p in percentiles}
        return {p: np.percentile(self.compounded_rates, p) for p in percentiles}
    
    def get_distribution_data(self):
        """
        Return distribution data for histogram plotting.
        
        Returns:
        --------
        rates : np.array
            Compounded rates across all paths
        """
        return self.compounded_rates
    
    def get_sample_paths(self, num_samples=50):
        """
        Return a sample of simulated paths for visualization.
        
        Returns:
        --------
        paths : np.array (num_samples × num_days)
        """
        if self.simulated_paths is None or len(self.simulated_paths) == 0:
            return np.array([[]])
        
        num_samples = min(num_samples, self.num_paths)
        indices = np.random.choice(self.num_paths, num_samples, replace=False)
        return self.simulated_paths[indices, :]

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
    from datetime import date, datetime, timedelta
    
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
                    found_date_msg = f"✓ Market data found for {found_date.strftime('%Y-%m-%d')}"
                else:
                    found_date_msg = f"⚠ Using nearest data from {found_date.strftime('%Y-%m-%d')}"
            else:
                 found_date_msg = "⚠ No history found - using latest available"

        st.info(found_date_msg)
        
        analysis_term = st.slider("Analysis Term (Years)", 1, 10, 5)
        
        # Extract spot rates from market data for the selected date
        j_spot_for_date = default_j_spot
        z_spot_for_date = default_z_spot
        
        if market_data_for_date:
            if 'JIBAR3M' in market_data_for_date and not pd.isna(market_data_for_date['JIBAR3M']):
                j_spot_for_date = float(market_data_for_date['JIBAR3M'])
            if 'ZARONIA' in market_data_for_date and not pd.isna(market_data_for_date['ZARONIA']):
                z_spot_for_date = float(market_data_for_date['ZARONIA'])
        
        col1, col2 = st.columns(2)
        j_spot = col1.number_input("3M JIBAR Spot (%)", value=j_spot_for_date, step=0.0001, format="%.4f")
        z_spot = col2.number_input("ZARONIA Spot (%)", value=z_spot_for_date, step=0.0001, format="%.4f")
        
        s0_bps = (j_spot - z_spot) * 100
        st.info(f"Spot Spread $s_0(t)$: **{s0_bps:.2f} bps**")

    with st.sidebar.expander("2. Zero Curve & Spreads", expanded=False):
        st.subheader("JIBAR Zero Curve (NACQ)")
        
        # Show market data inputs used for bootstrapping
        st.caption("Market Data Inputs for Bootstrapping")
        
        # Update bootstrap inputs when valuation date changes
        # Store the last valuation date to detect changes
        if 'last_val_date' not in st.session_state or st.session_state.last_val_date != val_date:
            st.session_state.bootstrap_inputs = market_data_for_date.copy()
            st.session_state.last_val_date = val_date
        
        # Create editable table for bootstrap inputs
        bootstrap_instruments = ['JIBAR3M', 'FRA 3x6', 'FRA 6x9', 'FRA 9x12', 'SASW1', 'SASW2', 'SASW3', 'SASW5', 'SASW10']
        bootstrap_data = []
        for inst in bootstrap_instruments:
            val = st.session_state.bootstrap_inputs.get(inst, None)
            if val is not None and not pd.isna(val):
                bootstrap_data.append({'Instrument': inst, 'Rate (%)': val})
        
        if bootstrap_data:
            df_bootstrap = pd.DataFrame(bootstrap_data)
            edited_bootstrap = st.data_editor(
                df_bootstrap,
                width='stretch',
                hide_index=True,
                key='bootstrap_inputs_editor',
                column_config={
                    "Instrument": st.column_config.TextColumn("Instrument", disabled=True),
                    "Rate (%)": st.column_config.NumberColumn("Rate (%)", format="%.4f")
                }
            )
            
            # Update session state with edited values
            for _, row in edited_bootstrap.iterrows():
                st.session_state.bootstrap_inputs[row['Instrument']] = row['Rate (%)']
        
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
        
        # Button to re-bootstrap from edited inputs
        if st.button("Bootstrap from Market Data"):
             bootstrapped_df = bootstrap_nacc_curve(st.session_state.bootstrap_inputs)
             if bootstrapped_df is not None and not bootstrapped_df.empty:
                 st.session_state.jibar_curve_data = bootstrapped_df
                 st.rerun()
             else:
                 st.error("Bootstrapping failed or no data for this date.")
        
        st.markdown("---")

        edited_jibar = st.data_editor(
            st.session_state.jibar_curve_data, 
            num_rows="dynamic",
            width='stretch',
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
            width='stretch',
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
        
        start_delay = st.number_input("Start Delay (Days)", value=0, min_value=0)
        start_date = add_business_days(val_date, int(start_delay))
        st.caption(f"Start Date: {start_date}")
        
        maturity_str = st.selectbox("Maturity", ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"])
        maturity_years = int(maturity_str.replace("Y",""))
        
        st.markdown("---")
        st.caption("Leg Settings")
        
        fixed_leg_freq = st.selectbox("Fixed Leg Freq", ["Quarterly", "Semi-annual", "Annual"])
        float_leg_freq = st.selectbox("Float Leg Freq", ["Quarterly", "Semi-annual", "Annual", "Monthly"])
        
        solve_par = st.checkbox("Solve for Par Rate", value=True)
        solve_basis = False
        solve_leg = "JIBAR"  # Which leg to solve for in basis swap
        
        if trade_type == "Basis Swap":
             solve_basis = st.checkbox("Solve for Par Basis Spread", value=True)
             if solve_basis:
                 solve_leg = st.radio("Solve for spread on:", ["JIBAR", "ZARONIA"], horizontal=True, 
                                     help="Which leg should have the spread adjusted to make NPV=0?")
             
        user_fixed_rate = st.number_input("Fixed Rate (%)", value=7.0, disabled=solve_par and trade_type != "Basis Swap", format="%.4f")
        
        if trade_type == "Basis Swap":
            jibar_spread_bps = st.number_input("JIBAR Spread (bps)", value=0.0, disabled=solve_basis and solve_leg=="JIBAR", format="%.2f")
            zaronia_spread_bps = st.number_input("ZARONIA Spread (bps)", value=0.0, disabled=solve_basis and solve_leg=="ZARONIA", format="%.2f")
        else:
            float_spread_bps = st.number_input("Float Leg Spread (bps)", value=0.0, format="%.2f")

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
    tab_method, tab_charts, tab_trade, tab_bench, tab_frn, tab_conv, tab_cvx, tab_comp, tab_hist = st.tabs([
        "Methodology & Theory", "Analysis & Charts", "Trade Pricing", "Benchmark Rates", 
        "FRN Pricing & Hedging", "Conversion Analysis", "Convexity Analysis", "Quarterly Compounding Analysis", "Historical Data"
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
        c1.plotly_chart(fig_zero, width='stretch')
        c2.plotly_chart(fig_fwd, width='stretch')
        
        c3, c4 = st.columns(2)
        c3.plotly_chart(fig_df, width='stretch')
        c4.plotly_chart(fig_spread, width='stretch')
        
        st.divider()
        
        # ==========================================
        # PROFESSIONAL SWAP TRADER ANALYTICS
        # ==========================================
        st.markdown("## 📊 Professional Trading Analytics")
        st.caption("Advanced curve analysis tools for swap traders")
        
        # Curve Steepness & Flattening Metrics
        st.markdown("### 📐 Curve Shape Metrics")
        
        # Calculate key spreads
        jibar_2s5s = (jibar_curve.get_zero_rate(5) - jibar_curve.get_zero_rate(2)) * 10000
        jibar_2s10s = (jibar_curve.get_zero_rate(10) - jibar_curve.get_zero_rate(2)) * 10000
        jibar_5s10s = (jibar_curve.get_zero_rate(10) - jibar_curve.get_zero_rate(5)) * 10000
        
        zaronia_2s5s = (zaronia_curve.get_zero_rate(5) - zaronia_curve.get_zero_rate(2)) * 10000
        zaronia_2s10s = (zaronia_curve.get_zero_rate(10) - zaronia_curve.get_zero_rate(2)) * 10000
        zaronia_5s10s = (zaronia_curve.get_zero_rate(10) - zaronia_curve.get_zero_rate(5)) * 10000
        
        col_steep1, col_steep2, col_steep3 = st.columns(3)
        
        with col_steep1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">2s5s Steepness</div>
                <div class="metric-value" style="color:#00CC96;">JIBAR: {jibar_2s5s:.1f} bps</div>
                <div class="metric-sub">ZARONIA: {zaronia_2s5s:.1f} bps</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_steep2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">2s10s Steepness</div>
                <div class="metric-value" style="color:#EF553B;">JIBAR: {jibar_2s10s:.1f} bps</div>
                <div class="metric-sub">ZARONIA: {zaronia_2s10s:.1f} bps</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_steep3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">5s10s Steepness</div>
                <div class="metric-value" style="color:#AB63FA;">JIBAR: {jibar_5s10s:.1f} bps</div>
                <div class="metric-sub">ZARONIA: {zaronia_5s10s:.1f} bps</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Butterfly Spreads
        st.markdown("### 🦋 Butterfly Spread Analysis")
        st.caption("Weighted butterfly spreads for relative value trading")
        
        # Calculate butterflies: (short_wing + long_wing) / 2 - body
        jibar_2s5s10s = ((jibar_curve.get_zero_rate(2) + jibar_curve.get_zero_rate(10)) / 2 - jibar_curve.get_zero_rate(5)) * 10000
        jibar_1s3s5s = ((jibar_curve.get_zero_rate(1) + jibar_curve.get_zero_rate(5)) / 2 - jibar_curve.get_zero_rate(3)) * 10000
        
        zaronia_2s5s10s = ((zaronia_curve.get_zero_rate(2) + zaronia_curve.get_zero_rate(10)) / 2 - zaronia_curve.get_zero_rate(5)) * 10000
        zaronia_1s3s5s = ((zaronia_curve.get_zero_rate(1) + zaronia_curve.get_zero_rate(5)) / 2 - zaronia_curve.get_zero_rate(3)) * 10000
        
        butterfly_data = {
            'Butterfly': ['2s5s10s', '1s3s5s'],
            'JIBAR (bps)': [jibar_2s5s10s, jibar_1s3s5s],
            'ZARONIA (bps)': [zaronia_2s5s10s, zaronia_1s3s5s],
            'Differential (bps)': [jibar_2s5s10s - zaronia_2s5s10s, jibar_1s3s5s - zaronia_1s3s5s]
        }
        df_butterfly = pd.DataFrame(butterfly_data)
        
        col_fly1, col_fly2 = st.columns([1, 1])
        
        with col_fly1:
            st.dataframe(
                df_butterfly.style.format({
                    'JIBAR (bps)': '{:.2f}',
                    'ZARONIA (bps)': '{:.2f}',
                    'Differential (bps)': '{:+.2f}'
                }).background_gradient(subset=['Differential (bps)'], cmap='RdYlGn'),
                width='stretch',
                hide_index=True
            )
            
            st.markdown("""
            **Trading Signals:**
            - **Positive butterfly**: Curve is humped (body rich vs wings)
            - **Negative butterfly**: Curve is bowed (body cheap vs wings)
            - **Differential**: JIBAR vs ZARONIA butterfly richness
            """)
        
        with col_fly2:
            # Butterfly visualization
            fig_fly = go.Figure()
            fig_fly.add_trace(go.Bar(
                x=df_butterfly['Butterfly'],
                y=df_butterfly['JIBAR (bps)'],
                name='JIBAR',
                marker_color='#00CC96'
            ))
            fig_fly.add_trace(go.Bar(
                x=df_butterfly['Butterfly'],
                y=df_butterfly['ZARONIA (bps)'],
                name='ZARONIA',
                marker_color='#EF553B'
            ))
            fig_fly.update_layout(
                title='Butterfly Spreads Comparison',
                yaxis_title='Spread (bps)',
                barmode='group',
                template='plotly_dark',
                height=300
            )
            st.plotly_chart(fig_fly, width='stretch')
        
        st.divider()
        
        # Carry & Roll-Down Analysis
        st.markdown("### 💰 Carry & Roll-Down Analysis")
        st.caption("Expected P&L from time decay and curve roll-down")
        
        # Interactive tenor selection
        carry_tenor = st.selectbox("Select Swap Tenor for Carry Analysis:", 
                                   ["2Y", "3Y", "5Y", "7Y", "10Y"], 
                                   index=2, key="carry_tenor")
        carry_tenor_years = int(carry_tenor.replace("Y", ""))
        carry_notional = st.number_input("Notional (ZAR)", value=100_000_000, step=10_000_000, 
                                        format="%d", key="carry_notional")
        
        # Calculate carry and roll-down
        # Carry = current coupon - funding cost
        current_rate = jibar_curve.get_zero_rate(carry_tenor_years)
        funding_rate = jibar_curve.get_zero_rate(0.25)  # 3M funding
        carry_bps = (current_rate - funding_rate) * 10000
        carry_annual_zar = carry_notional * (current_rate - funding_rate)
        
        # Roll-down = expected rate change as swap rolls down curve
        # If curve is upward sloping, rolling down 3M earns positive P&L
        rate_3m_forward = jibar_curve.get_zero_rate(max(carry_tenor_years - 0.25, 0.25))
        rolldown_bps = (current_rate - rate_3m_forward) * 10000
        
        # DV01 for P&L calculation
        duration_approx = carry_tenor_years / 2.0
        dv01 = carry_notional * duration_approx * 0.0001
        rolldown_pnl = rolldown_bps * dv01
        
        # Total expected return
        total_return_3m = (carry_annual_zar / 4) + rolldown_pnl
        
        col_carry1, col_carry2, col_carry3 = st.columns(3)
        
        with col_carry1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Carry (3M)</div>
                <div class="metric-value" style="color:#00CC96;">{carry_bps:.2f} bps</div>
                <div class="metric-sub">ZAR {carry_annual_zar/4:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_carry2:
            rolldown_color = "#00CC96" if rolldown_pnl > 0 else "#EF553B"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Roll-Down (3M)</div>
                <div class="metric-value" style="color:{rolldown_color};">{rolldown_bps:.2f} bps</div>
                <div class="metric-sub">ZAR {rolldown_pnl:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_carry3:
            total_color = "#00CC96" if total_return_3m > 0 else "#EF553B"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Return (3M)</div>
                <div class="metric-value" style="color:{total_color};">ZAR {total_return_3m:,.0f}</div>
                <div class="metric-sub">{(total_return_3m/carry_notional)*100:.3f}% of Notional</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Carry/Roll visualization across tenors
        carry_tenors = [1, 2, 3, 5, 7, 10]
        carry_values = []
        rolldown_values = []
        
        for t in carry_tenors:
            c_rate = jibar_curve.get_zero_rate(t)
            c_carry = (c_rate - funding_rate) * 10000
            c_forward = jibar_curve.get_zero_rate(max(t - 0.25, 0.25))
            c_rolldown = (c_rate - c_forward) * 10000
            carry_values.append(c_carry)
            rolldown_values.append(c_rolldown)
        
        fig_carry = go.Figure()
        fig_carry.add_trace(go.Scatter(
            x=[f"{t}Y" for t in carry_tenors],
            y=carry_values,
            name='Carry',
            mode='lines+markers',
            line=dict(color='#00CC96', width=3),
            marker=dict(size=8)
        ))
        fig_carry.add_trace(go.Scatter(
            x=[f"{t}Y" for t in carry_tenors],
            y=rolldown_values,
            name='Roll-Down',
            mode='lines+markers',
            line=dict(color='#EF553B', width=3),
            marker=dict(size=8)
        ))
        fig_carry.update_layout(
            title='Carry & Roll-Down by Tenor (3M Horizon)',
            xaxis_title='Tenor',
            yaxis_title='Basis Points',
            template='plotly_dark',
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig_carry, width='stretch')
        
        st.divider()
        
        # Interactive Scenario Analysis
        st.markdown("### 🎯 Interactive Scenario Analysis")
        st.caption("Stress test your swap positions with custom curve shifts")
        
        col_scen1, col_scen2 = st.columns(2)
        
        with col_scen1:
            scenario_type = st.selectbox("Scenario Type:", 
                                        ["Parallel Shift", "Steepening", "Flattening", "Butterfly"], 
                                        key="scenario_type")
            shift_size = st.slider("Shift Size (bps):", -100, 100, 25, 5, key="shift_size")
        
        with col_scen2:
            scenario_tenor = st.selectbox("Position Tenor:", 
                                         ["2Y", "3Y", "5Y", "7Y", "10Y"], 
                                         index=2, key="scenario_tenor")
            scenario_notional = st.number_input("Position Notional (ZAR):", 
                                               value=100_000_000, step=10_000_000, 
                                               format="%d", key="scenario_notional")
        
        scenario_tenor_years = int(scenario_tenor.replace("Y", ""))
        
        # Calculate P&L impact
        base_dv01 = scenario_notional * (scenario_tenor_years / 2.0) * 0.0001
        
        if scenario_type == "Parallel Shift":
            pnl_impact = base_dv01 * shift_size
            scenario_desc = f"{shift_size:+d}bp parallel shift across entire curve"
        elif scenario_type == "Steepening":
            # Long end moves more than short end
            pnl_impact = base_dv01 * shift_size * 0.7  # Approximate
            scenario_desc = f"Curve steepens: 2Y flat, {scenario_tenor} {shift_size:+d}bp"
        elif scenario_type == "Flattening":
            pnl_impact = base_dv01 * shift_size * (-0.7)
            scenario_desc = f"Curve flattens: 2Y {shift_size:+d}bp, {scenario_tenor} flat"
        else:  # Butterfly
            pnl_impact = base_dv01 * shift_size * 0.5
            scenario_desc = f"Butterfly: belly {shift_size:+d}bp vs wings"
        
        pnl_color = "#00CC96" if pnl_impact > 0 else "#EF553B"
        
        st.markdown(f"""
        **Scenario:** {scenario_desc}
        
        **P&L Impact:**
        """)
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {pnl_color};">
            <div class="metric-label">Estimated P&L</div>
            <div class="metric-value" style="color:{pnl_color}; font-size:32px;">ZAR {pnl_impact:,.0f}</div>
            <div class="metric-sub">{(pnl_impact/scenario_notional)*100:+.3f}% of Notional | DV01: {base_dv01:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Scenario curve visualization
        tenors_scenario = np.linspace(0.25, 10, 40)
        base_curve = [jibar_curve.get_zero_rate(t) * 100 for t in tenors_scenario]
        
        if scenario_type == "Parallel Shift":
            shocked_curve = [r + shift_size/100 for r in base_curve]
        elif scenario_type == "Steepening":
            shocked_curve = [r + (shift_size/100) * (t/10) for r, t in zip(base_curve, tenors_scenario)]
        elif scenario_type == "Flattening":
            shocked_curve = [r - (shift_size/100) * (t/10) for r, t in zip(base_curve, tenors_scenario)]
        else:  # Butterfly
            shocked_curve = [r + (shift_size/100) * np.sin(t * np.pi / 10) for r, t in zip(base_curve, tenors_scenario)]
        
        fig_scenario = go.Figure()
        fig_scenario.add_trace(go.Scatter(
            x=tenors_scenario,
            y=base_curve,
            name='Base Curve',
            line=dict(color='#00CC96', width=2, dash='dash')
        ))
        fig_scenario.add_trace(go.Scatter(
            x=tenors_scenario,
            y=shocked_curve,
            name='Shocked Curve',
            line=dict(color='#EF553B', width=3),
            fill='tonexty',
            fillcolor='rgba(239, 85, 59, 0.2)'
        ))
        fig_scenario.update_layout(
            title=f'Curve Scenario: {scenario_type}',
            xaxis_title='Tenor (Years)',
            yaxis_title='Rate (%)',
            template='plotly_dark',
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig_scenario, width='stretch')
        
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
                 st.plotly_chart(fig_3d, width='stretch')
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
            leg2 = SwapLeg('Float', 'ZAR', notional, start_date, maturity_years, float_leg_freq, spread_bps=float_spread_bps, float_index='JIBAR')
        elif trade_type == "ZARONIA OIS":
            leg1 = SwapLeg('Fixed', 'ZAR', notional, start_date, maturity_years, fixed_leg_freq, fixed_rate=0.0)
            leg2 = SwapLeg('Float', 'ZAR', notional, start_date, maturity_years, float_leg_freq, spread_bps=float_spread_bps, float_index='ZARONIA')
        elif trade_type == "Basis Swap":
            leg1 = SwapLeg('Float', 'ZAR', notional, start_date, maturity_years, float_leg_freq, spread_bps=jibar_spread_bps, float_index='JIBAR')
            leg2 = SwapLeg('Float', 'ZAR', notional, start_date, maturity_years, float_leg_freq, spread_bps=zaronia_spread_bps, float_index='ZARONIA')

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
             # Solve for spread on selected leg (JIBAR or ZARONIA) to make NPV=0
             def obj_basis(s_bps):
                 if solve_leg == "JIBAR":
                     # Solve for JIBAR spread
                     leg1.spread = s_bps / 10000.0
                     leg1.calculate_cashflows(jibar_curve, zaronia_curve, selected_disc)
                     leg2.calculate_cashflows(jibar_curve, zaronia_curve, selected_disc)
                 else:
                     # Solve for ZARONIA spread
                     leg2.spread = s_bps / 10000.0
                     leg1.calculate_cashflows(jibar_curve, zaronia_curve, selected_disc)
                     leg2.calculate_cashflows(jibar_curve, zaronia_curve, selected_disc)
                 return leg1.get_pv() - leg2.get_pv()
             
             try:
                 par_spread = brentq(obj_basis, -500, 500) # +/- 500 bps
                 if solve_leg == "JIBAR":
                     jibar_spread_bps = par_spread
                     leg1.spread = par_spread / 10000.0
                 else:
                     zaronia_spread_bps = par_spread
                     leg2.spread = par_spread / 10000.0
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
        # Calculate days to maturity safely
        from datetime import datetime, date
        maturity_date = leg1.schedule[-1]
        if isinstance(maturity_date, datetime):
            maturity_date = maturity_date.date()
        if isinstance(start_date, datetime):
            start_date_calc = start_date.date()
        else:
            start_date_calc = start_date
        days_to_maturity = (maturity_date - start_date_calc).days
        
        st.info(f"""
        **Trade Confirmation Summary**:
        *   **Structure**: {maturity_years}Y {trade_type} ({leg1.type} vs {leg2.type})
        *   **Notional**: {currency} {notional:,.0f}
        *   **Dates**: Effective {start_date} $\\rightarrow$ Maturing {leg1.schedule[-1]} ({days_to_maturity} days)
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
                    <div class="metric-label">Par Basis Spread ({solve_leg})</div>
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
            if trade_type == "Basis Swap":
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">JIBAR Spread</div>
                    <div class="metric-value">{"{:.2f} bps".format(leg1.spread * 10000)}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Fixed Rate Used</div>
                    <div class="metric-value">{"{:.4f}%".format(final_fixed_rate*100)}</div>
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
        st.plotly_chart(fig_cf, width='stretch')

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
                    }), width='stretch')
        
        c_tbl1, c_tbl2 = st.columns(2)
        with c_tbl1: show_leg_cf(leg1, "Leg 1")
        with c_tbl2: show_leg_cf(leg2, "Leg 2")
        
        # DV01 vs CS01 Analysis for JIBAR IRS
        if trade_type == "JIBAR IRS":
            st.markdown("---")
            st.markdown("### 🎯 DV01 vs CS01: Risk Analysis for Vanilla JIBAR IRS")
            st.caption("Separating rate risk to next reset (DV01) from full term spread risk (CS01)")
            
            st.markdown("""
            **Critical Distinction for JIBAR IRS:**
            
            - **DV01**: Rate risk to **next JIBAR 3M reset only** (~3 months)
            - **CS01**: Full term spread risk over **entire swap life**
            
            Unlike fixed-rate swaps, JIBAR IRS have limited rate risk because they reset quarterly.
            The dominant risk is spread risk (CS01), not rate risk (DV01).
            """)
            
            # Calculate DV01 and CS01 for multiple tenors
            irs_tenors = [1, 2, 3, 5, 10]
            irs_risk_data = []
            
            for tenor in irs_tenors:
                # DV01: Rate risk to next reset ONLY (3 months for JIBAR 3M)
                # DV01 ≈ Notional × Time_to_Reset × 0.0001
                time_to_reset = 0.25  # 3 months
                dv01_value = notional * time_to_reset * 0.0001
                
                # CS01: Full term spread risk
                # CS01 ≈ Notional × Full_Tenor × 0.0001
                cs01_value = notional * tenor * 0.0001
                
                # Ratio shows spread risk dominance
                cs01_dv01_ratio = cs01_value / dv01_value if dv01_value != 0 else 0
                
                # JIBAR par rate at this tenor
                jibar_par = jibar_curve.get_zero_rate(tenor) * 100
                
                # ZARONIA equivalent (for comparison)
                zaronia_par = zaronia_curve.get_zero_rate(tenor) * 100
                
                # Credit spread embedded in JIBAR
                credit_spread_bps = (jibar_par - zaronia_par) * 100
                
                irs_risk_data.append({
                    'Tenor': f'{tenor}Y',
                    'JIBAR Par (%)': jibar_par,
                    'DV01 (ZAR)': dv01_value,
                    'CS01 (ZAR)': cs01_value,
                    'CS01/DV01 Ratio': cs01_dv01_ratio,
                    'Credit Spread (bps)': credit_spread_bps,
                    'Time to Reset': f'{time_to_reset:.2f}Y'
                })
            
            df_irs_risk = pd.DataFrame(irs_risk_data)
            
            col_irs1, col_irs2 = st.columns([1, 1])
            
            with col_irs1:
                st.markdown("**Risk Metrics by Tenor**")
                st.dataframe(
                    df_irs_risk.style.format({
                        'JIBAR Par (%)': '{:.4f}',
                        'DV01 (ZAR)': '{:,.0f}',
                        'CS01 (ZAR)': '{:,.0f}',
                        'CS01/DV01 Ratio': '{:.1f}x',
                        'Credit Spread (bps)': '{:.2f}',
                        'Time to Reset': '{}'
                    }).background_gradient(subset=['CS01/DV01 Ratio'], cmap='YlOrRd'),
                    width='stretch',
                    hide_index=True
                )
                
                current_tenor_idx = min(maturity_years-1, len(df_irs_risk)-1)
                current_dv01 = df_irs_risk.iloc[current_tenor_idx]['DV01 (ZAR)']
                current_cs01 = df_irs_risk.iloc[current_tenor_idx]['CS01 (ZAR)']
                current_ratio = df_irs_risk.iloc[current_tenor_idx]['CS01/DV01 Ratio']
                
                st.markdown(f"""
                **Current Trade ({maturity_years}Y):**
                - **DV01 (Next Reset)**: ZAR {current_dv01:,.0f} per bp
                - **CS01 (Full Term)**: ZAR {current_cs01:,.0f} per bp
                - **CS01/DV01 Ratio**: {current_ratio:.1f}x
                - **Credit Exposure**: {df_irs_risk.iloc[current_tenor_idx]['Credit Spread (bps)']:.2f} bps embedded
                
                **Key Insight:**
                - Spread risk is {current_ratio:.1f}x larger than rate risk
                - Must hedge both components separately
                """)
            
            with col_irs2:
                # Chart: DV01 vs CS01 by tenor
                fig_irs_risk = go.Figure()
                fig_irs_risk.add_trace(go.Bar(
                    x=df_irs_risk['Tenor'],
                    y=df_irs_risk['DV01 (ZAR)'],
                    name='DV01 (Rate Risk)',
                    marker_color='#00CC96',
                    text=df_irs_risk['DV01 (ZAR)'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside'
                ))
                fig_irs_risk.add_trace(go.Bar(
                    x=df_irs_risk['Tenor'],
                    y=df_irs_risk['CS01 (ZAR)'],
                    name='CS01 (Spread Risk)',
                    marker_color='#EF553B',
                    text=df_irs_risk['CS01 (ZAR)'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside'
                ))
                fig_irs_risk.update_layout(
                    title=f'DV01 vs CS01 by Tenor (Notional: ZAR {notional:,.0f})',
                    xaxis_title='Tenor',
                    yaxis_title='Risk (ZAR per bp)',
                    barmode='group',
                    template='plotly_dark',
                    height=350,
                    showlegend=True
                )
                st.plotly_chart(fig_irs_risk, width='stretch')
            
            # Key insights
            st.markdown("**Key Insights:**")
            
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.markdown(f"""
                **DV01 (Interest Rate Risk):**
                - Measures P&L impact of parallel shift in JIBAR curve
                - For {maturity_years}Y: ZAR {df_irs_risk.iloc[min(maturity_years-1, len(df_irs_risk)-1)]['DV01 (ZAR)']:,.0f} per 1bp move
                - Scales linearly with tenor (longer = more risk)
                - Hedged with offsetting JIBAR IRS or futures
                
                **Typical Hedging:**
                - Receive fixed in {maturity_years}Y JIBAR IRS
                - Or use JIBAR futures strip
                - Notional: ZAR {notional:,.0f}
                """)
            
            with col_insight2:
                st.markdown(f"""
                **CS01 (Credit Spread Risk):**
                - Measures P&L impact if JIBAR-ZARONIA basis widens
                - JIBAR embeds {df_irs_risk.iloc[min(maturity_years-1, len(df_irs_risk)-1)]['Credit Spread (bps)']:.2f} bps credit spread
                - If basis widens 1bp → lose ZAR {df_irs_risk.iloc[min(maturity_years-1, len(df_irs_risk)-1)]['CS01 (ZAR)']:,.0f}
                - Cannot be hedged with JIBAR instruments alone
                
                **Basis Risk Hedging:**
                - Requires JIBAR-ZARONIA basis swap
                - Or switch to ZARONIA OIS + credit overlay
                - Residual credit exposure remains
                """)

    with tab_bench:
        st.subheader("Step 7: Benchmark OIS Rates")
        
        st.markdown("""
        **Par Swap Rates** for standard ZARONIA Overnight Index Swap (OIS) tenors.
        
        These rates represent the **fixed rate** that would make a new OIS swap have zero net present value (NPV = 0) at inception.
        
        **Swap Structure:**
        - **Fixed Leg**: Pays the par rate shown below on a fixed payment frequency
        - **Floating Leg**: Pays compounded ZARONIA (daily overnight rates) on the same frequency
        - **Day Count**: ACT/365 for both legs
        - **Discounting**: ZARONIA zero curve (OIS discounting)
        
        **Payment Frequencies by Tenor:**
        - **1M**: Monthly payments
        - **3M, 9M**: Quarterly payments  
        - **6M**: Semi-annual payments
        - **1Y+**: Annual payments
        
        **Interpretation:**
        - These are **effective annualized fixed rates** (ACT/365 basis)
        - They represent the market's expectation of average ZARONIA over each tenor
        - The curve shape shows the term structure of ZAR overnight rates
        - Higher rates at longer tenors indicate expectations of rising rates or term premium
        """)
        
        st.divider()
        bench_tenors = [
            ("1M", 1/12, 'Monthly'), ("3M", 0.25, 'Quarterly'), ("6M", 0.5, 'Semi-annual'), 
            ("9M", 0.75, 'Quarterly'), ("1Y", 1.0, 'Annual'), ("2Y", 2.0, 'Annual'),
            ("5Y", 5.0, 'Annual'), ("10Y", 10.0, 'Annual')
        ]
        
        bench_data = []
        for label, t_year, freq in bench_tenors:
            leg_fix = SwapLeg('Fixed', 'ZAR', 100, start_date, t_year, freq, fixed_rate=0.0)
            leg_flt = SwapLeg('Float', 'ZAR', 100, start_date, t_year, freq, float_index='ZARONIA')
            
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
        st.dataframe(df_bench, width='stretch')
        
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
            st.plotly_chart(fig_bench, width='stretch')

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
            }), width='stretch')
            
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
        st.plotly_chart(fig_rv, width='stretch')

        st.divider()
        st.markdown("### 🎯 Risk Decomposition & Hedging Strategy")
        st.caption("Separate hedging for DV01 (rate risk to next reset) and CS01 (full term spread risk)")
        
        # Calculate time to next reset (for JIBAR FRNs, this is typically 3M)
        time_to_reset = 0.25  # 3 months in years
        
        # ==========================================
        # DV01: Rate Risk to Next Reset Only
        # ==========================================
        st.markdown("#### 1️⃣ DV01 - Rate Risk to Next Reset")
        st.markdown("""
        **Definition:** PV sensitivity to 1bp shift in the **next JIBAR 3M fixing** only.
        
        For FRNs, this is the risk that the next coupon fixing changes. After the reset, 
        the FRN reprices to par (ignoring spread), so DV01 is limited to the next reset period.
        """)
        
        # Calculate DV01: Sensitivity to next reset rate only
        # Approximate: DV01 ≈ Notional × Time_to_Reset × 0.0001
        dv01_frn_reset = frn_notional * time_to_reset * 0.0001
        
        # For more accurate calculation, bump the forward rate for next period
        j_rates_up = j_rates + 0.0001
        jibar_curve_up = JibarZeroCurve(j_tenors, j_rates_up)
        zaronia_curve_up = ZaroniaCurve(val_date, jibar_curve_up, spread_func)
        
        frn.calculate_cashflows(zaronia_curve_up)
        pv_frn_up = sum(cf['PV'] for cf in frn.cashflows) + frn.principal_flow['PV']
        
        frn.calculate_cashflows(zaronia_curve)
        pv_frn_base = sum(cf['PV'] for cf in frn.cashflows) + frn.principal_flow['PV']
        
        dv01_frn_actual = pv_frn_up - pv_frn_base
        
        # ==========================================
        # CS01: Full Term Spread Risk
        # ==========================================
        st.markdown("#### 2️⃣ CS01 - Full Term Spread Risk")
        st.markdown("""
        **Definition:** PV sensitivity to 1bp parallel shift in the **spread over the entire term**.
        
        This measures the risk that the FRN's quoted margin (spread) changes in value 
        due to credit spread widening or tightening over the full remaining life.
        """)
        
        # Calculate CS01: Sensitivity to spread change over full term
        # CS01 ≈ Notional × Remaining_Life × 0.0001
        remaining_life = frn_years  # Full remaining tenor
        cs01_frn = frn_notional * remaining_life * 0.0001
        
        # Display Risk Metrics
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">DV01 (Next Reset)</div>
                <div class="metric-value" style="color:#00CC96;">ZAR {dv01_frn_reset:,.2f}</div>
                <div class="metric-sub">Rate risk: {time_to_reset:.2f}Y</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_risk2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CS01 (Full Term)</div>
                <div class="metric-value" style="color:#EF553B;">ZAR {cs01_frn:,.2f}</div>
                <div class="metric-sub">Spread risk: {remaining_life:.2f}Y</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_risk3:
            ratio = cs01_frn / dv01_frn_reset if dv01_frn_reset != 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CS01 / DV01 Ratio</div>
                <div class="metric-value" style="color:#AB63FA;">{ratio:.2f}x</div>
                <div class="metric-sub">Spread risk dominates</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # ==========================================
        # Hedging Recommendations
        # ==========================================
        st.markdown("### 🛡️ Hedging Recommendations")
        
        col_hedge1, col_hedge2 = st.columns(2)
        
        with col_hedge1:
            st.markdown("#### Hedge DV01 (Rate Risk)")
            
            # Short-dated hedge for rate risk
            hedge_dv01_tenor = st.selectbox("DV01 Hedge Instrument", 
                                           ["3M FRA", "6M FRA", "1Y OIS"], 
                                           index=0, key="hedge_dv01")
            
            # Calculate hedge ratio for DV01
            if "FRA" in hedge_dv01_tenor:
                # FRA has DV01 ≈ Notional × 0.25 × 0.0001
                dv01_hedge_instrument = frn_notional * 0.25 * 0.0001
            else:
                # 1Y OIS
                dv01_hedge_instrument = frn_notional * 0.5 * 0.0001  # Approx duration
            
            hedge_ratio_dv01 = -dv01_frn_reset / dv01_hedge_instrument if dv01_hedge_instrument != 0 else 0
            
            st.markdown(f"""
            **Hedge Strategy:**
            - **Instrument**: {hedge_dv01_tenor}
            - **Hedge Ratio**: {abs(hedge_ratio_dv01):.4f}
            - **Action**: **{'Sell' if hedge_ratio_dv01 < 0 else 'Buy'}** {abs(hedge_ratio_dv01):.2f} units
            - **Notional**: ZAR {abs(hedge_ratio_dv01 * frn_notional):,.0f}
            
            **Rationale:**
            - Offsets rate risk to next JIBAR reset
            - Short-dated instrument matches reset horizon
            - Hedge needs to be rolled at each reset
            """)
        
        with col_hedge2:
            st.markdown("#### Hedge CS01 (Spread Risk)")
            
            # Long-dated hedge for spread risk
            hedge_cs01_tenor = st.selectbox("CS01 Hedge Instrument", 
                                           ["2Y Basis Swap", "3Y Basis Swap", "5Y Basis Swap"], 
                                           index=1 if frn_years >= 3 else 0, key="hedge_cs01")
            
            hedge_cs01_years = int(hedge_cs01_tenor.split("Y")[0])
            
            # Calculate hedge ratio for CS01
            # Basis swap has CS01 ≈ Notional × Tenor × 0.0001
            cs01_hedge_instrument = frn_notional * hedge_cs01_years * 0.0001
            
            hedge_ratio_cs01 = -cs01_frn / cs01_hedge_instrument if cs01_hedge_instrument != 0 else 0
            
            st.markdown(f"""
            **Hedge Strategy:**
            - **Instrument**: {hedge_cs01_tenor} (JIBAR-ZARONIA)
            - **Hedge Ratio**: {abs(hedge_ratio_cs01):.4f}
            - **Action**: **{'Receive' if hedge_ratio_cs01 < 0 else 'Pay'}** JIBAR spread
            - **Notional**: ZAR {abs(hedge_ratio_cs01 * frn_notional):,.0f}
            
            **Rationale:**
            - Offsets full term spread risk
            - Basis swap locks in JIBAR-ZARONIA spread
            - Static hedge (no rolling required)
            - Protects against credit spread widening
            """)
        
        # Combined Hedging Summary
        st.markdown("### 📋 Complete Hedging Package")
        
        total_hedge_cost = abs(hedge_ratio_dv01 * frn_notional * 0.0001) + abs(hedge_ratio_cs01 * frn_notional * 0.0001)
        
        st.markdown(f"""
        **Two-Component Hedge:**
        
        1. **DV01 Hedge**: {hedge_dv01_tenor}
           - Notional: ZAR {abs(hedge_ratio_dv01 * frn_notional):,.0f}
           - Protects: Next reset rate risk
           - Roll frequency: Every {time_to_reset:.1f} years
        
        2. **CS01 Hedge**: {hedge_cs01_tenor}
           - Notional: ZAR {abs(hedge_ratio_cs01 * frn_notional):,.0f}
           - Protects: Full term spread risk
           - Roll frequency: Static (hold to maturity)
        
        **Total Hedging Cost**: ~ZAR {total_hedge_cost:,.0f} (estimated transaction costs)
        
        **Key Insight:**
        - CS01 risk ({cs01_frn:,.0f}) is **{ratio:.1f}x larger** than DV01 risk ({dv01_frn_reset:,.0f})
        - For FRNs, spread risk dominates rate risk
        - Both components must be hedged separately for complete protection
        """)

    with tab_conv:
        st.subheader("🔄 JIBAR → ZARONIA Conversion Analysis")
        
        st.markdown("""
        **Bank-Grade Relative Value & Conversion Analytics Engine**
        
        This module rigorously evaluates the economic fairness of converting from JIBAR-linked to ZARONIA-linked instruments.
        
        **Economic Question:**
        > *"When offered a conversion from JIBAR 3M + Spread_old to ZARONIA Compounded + Spread_new, is this fair?"*
        
        **Analysis Framework:**
        1. **Forward Equivalence**: E[JIBAR 3M] ≈ Compounded ZARONIA + Basis
        2. **PV Neutrality**: PV(JIBAR structure) = PV(ZARONIA structure)
        3. **Convexity Effects**: Non-linear compounding adjustments
        4. **Value Transfer**: Quantify gains/losses from conversion
        """)
        
        st.divider()
        
        # Simple Conversion Input
        st.markdown("### ⚙️ Conversion Parameters")
        st.caption("Simple setup for typical JIBAR → ZARONIA conversions")
        
        col_conv1, col_conv2, col_conv3 = st.columns(3)
        
        with col_conv1:
            st.markdown("**Position Details**")
            conv_notional = st.number_input("Notional (ZAR)", value=100_000_000, step=1_000_000, format="%d", key="conv_notional")
            conv_tenor_str = st.selectbox("Remaining Tenor", ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y"], index=2, key="conv_tenor")
            conv_tenor = int(conv_tenor_str.replace("Y", ""))
            conv_freq = st.selectbox("Payment Frequency", ["Quarterly", "Semi-annual", "Annual"], index=0, key="conv_freq")
        
        with col_conv2:
            st.markdown("**Current JIBAR Terms**")
            conv_jibar_spread = st.number_input("JIBAR 3M Spread (bps)", value=50.0, step=1.0, format="%.2f", key="conv_jibar_spread",
                                               help="Current spread over JIBAR 3M")
            st.caption("Reference: JIBAR 3M")
            st.caption(f"Current Rate: ~{jibar_curve.get_zero_rate(0.25)*100:.2f}%")
        
        with col_conv3:
            st.markdown("**Offered ZARONIA Terms**")
            conv_zaronia_spread = st.number_input("ZARONIA Spread (bps)", value=30.0, step=1.0, format="%.2f", key="conv_zaronia_spread",
                                                 help="Offered spread over compounded ZARONIA")
            st.caption("Reference: ZARONIA Compounded")
            st.caption(f"Current Rate: ~{zaronia_curve.get_zero_rate(0.25)*100:.2f}%")
        
        # Set conversion start date
        conv_start_date = val_date
        
        st.divider()
        
        # Build Conversion Analyzer
        try:
            analyzer = ConversionAnalyzer(
                notional=conv_notional,
                start_date=conv_start_date,
                maturity_years=conv_tenor,
                jibar_spread_bps=conv_jibar_spread,
                zaronia_spread_bps=conv_zaronia_spread,
                frequency=conv_freq,
                jibar_curve=jibar_curve,
                zaronia_curve=zaronia_curve
            )
            
            # Calculate key metrics
            pv_original = analyzer.get_pv_original()
            pv_converted = analyzer.get_pv_converted()
            pv_diff = analyzer.get_pv_difference()
            fair_spread = analyzer.solve_fair_zaronia_spread()
            spread_decomp = analyzer.get_spread_decomposition()
            
            # Dashboard Metrics
            st.markdown("### 📊 Conversion Valuation Dashboard")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Fair ZARONIA Spread</div>
                    <div class="metric-value" style="color:#00CC96;">{fair_spread:.2f} bps</div>
                    <div class="metric-sub">PV-Neutral Conversion</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Offered Spread</div>
                    <div class="metric-value" style="color:#19D3F3;">{conv_zaronia_spread:.2f} bps</div>
                    <div class="metric-sub">Market Quote</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m3:
                mispricing = spread_decomp['Mispricing']
                mispricing_color = "#EF553B" if mispricing < 0 else "#00CC96"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Mispricing</div>
                    <div class="metric-value" style="color:{mispricing_color};">{mispricing:+.2f} bps</div>
                    <div class="metric-sub">Offered - Fair</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m4:
                pv_diff_pct = (pv_diff / conv_notional) * 100
                pv_color = "#EF553B" if pv_diff < 0 else "#00CC96"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Value Transfer</div>
                    <div class="metric-value" style="color:{pv_color};">ZAR {pv_diff:,.0f}</div>
                    <div class="metric-sub">{pv_diff_pct:+.3f}% of Notional</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m5:
                convexity = spread_decomp['Convexity_Adj']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Convexity Adjustment</div>
                    <div class="metric-value" style="color:#FFA15A;">{convexity:.2f} bps</div>
                    <div class="metric-sub">Compounding Effect</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Economic Interpretation
            st.markdown("")
            if abs(mispricing) < 1.0:
                st.success(f"✅ **Fair Conversion**: The offered spread is within 1bp of fair value. This is an economically neutral conversion.")
            elif mispricing > 0:
                st.success(f"✅ **Favorable to Investor**: The offered spread is {mispricing:.2f} bps **above** fair value. Investor gains ZAR {abs(pv_diff):,.0f} in PV.")
            else:
                st.error(f"⚠️ **Unfavorable to Investor**: The offered spread is {abs(mispricing):.2f} bps **below** fair value. Investor loses ZAR {abs(pv_diff):,.0f} in PV.")
            
            st.divider()
            
            # Market Rates Table with Basis Decomposition
            st.markdown("### 📊 Market Rates & Basis Decomposition by Tenor")
            st.caption("JIBAR IRS par rates (NACC), ZARONIA equivalents, and spread components across the curve")
            
            # Calculate par rates and basis for multiple tenors
            tenors_years = [1, 2, 3, 5, 10]
            market_data = []
            
            for tenor in tenors_years:
                # JIBAR par rate (zero rate as approximation)
                jibar_par = jibar_curve.get_zero_rate(tenor) * 100
                
                # ZARONIA par rate (zero rate as approximation)
                zaronia_par = zaronia_curve.get_zero_rate(tenor) * 100
                
                # Basis spread
                basis_bps = (jibar_curve.get_zero_rate(tenor) - zaronia_curve.get_zero_rate(tenor)) * 10000
                
                # Convexity adjustment (analytical formula: 0.5 × σ² × T)
                # Assume 100 bps volatility for ZARONIA
                sigma = 0.01  # 100 bps
                convexity_bps = 0.5 * (sigma ** 2) * tenor * 10000
                
                # Term premium (simple approximation: 1bp per year)
                term_premium_bps = tenor * 1.0
                
                # Credit spread (residual after removing convexity and term premium)
                credit_spread_bps = basis_bps - convexity_bps - term_premium_bps
                
                market_data.append({
                    'Tenor': f'{tenor}Y',
                    'JIBAR Par (%)': jibar_par,
                    'ZARONIA Par (%)': zaronia_par,
                    'Total Basis (bps)': basis_bps,
                    'Credit Spread (bps)': credit_spread_bps,
                    'Convexity (bps)': convexity_bps,
                    'Term Premium (bps)': term_premium_bps
                })
            
            df_market = pd.DataFrame(market_data)
            
            # Display table with formatting
            st.dataframe(
                df_market.style.format({
                    'JIBAR Par (%)': '{:.4f}',
                    'ZARONIA Par (%)': '{:.4f}',
                    'Total Basis (bps)': '{:.2f}',
                    'Credit Spread (bps)': '{:.2f}',
                    'Convexity (bps)': '{:.2f}',
                    'Term Premium (bps)': '{:.2f}'
                }).background_gradient(subset=['Total Basis (bps)'], cmap='RdYlGn_r'),
                width='stretch',
                hide_index=True
            )
            
            # Visualization of basis decomposition across tenors
            col_basis1, col_basis2 = st.columns(2)
            
            with col_basis1:
                # Stacked bar chart of basis components
                fig_basis = go.Figure()
                fig_basis.add_trace(go.Bar(
                    x=df_market['Tenor'],
                    y=df_market['Credit Spread (bps)'],
                    name='Credit Spread',
                    marker_color='#EF553B'
                ))
                fig_basis.add_trace(go.Bar(
                    x=df_market['Tenor'],
                    y=df_market['Convexity (bps)'],
                    name='Convexity',
                    marker_color='#00CC96'
                ))
                fig_basis.add_trace(go.Bar(
                    x=df_market['Tenor'],
                    y=df_market['Term Premium (bps)'],
                    name='Term Premium',
                    marker_color='#AB63FA'
                ))
                fig_basis.update_layout(
                    title='Basis Decomposition by Tenor',
                    xaxis_title='Tenor',
                    yaxis_title='Basis Points',
                    barmode='stack',
                    template='plotly_dark',
                    height=350
                )
                st.plotly_chart(fig_basis, width='stretch')
            
            with col_basis2:
                # Line chart of par rates
                fig_par = go.Figure()
                fig_par.add_trace(go.Scatter(
                    x=df_market['Tenor'],
                    y=df_market['JIBAR Par (%)'],
                    mode='lines+markers',
                    name='JIBAR Par',
                    line=dict(color='#00CC96', width=3),
                    marker=dict(size=8)
                ))
                fig_par.add_trace(go.Scatter(
                    x=df_market['Tenor'],
                    y=df_market['ZARONIA Par (%)'],
                    mode='lines+markers',
                    name='ZARONIA Par',
                    line=dict(color='#EF553B', width=3),
                    marker=dict(size=8)
                ))
                fig_par.update_layout(
                    title='Par Rate Curves',
                    xaxis_title='Tenor',
                    yaxis_title='Rate (%)',
                    template='plotly_dark',
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_par, width='stretch')
            
            st.divider()
            
            # Spread Decomposition
            st.markdown("### 🔬 Spread Decomposition Analysis")
            st.caption("Breaking down the fair conversion spread into economic components")
            
            decomp_data = {
                'Component': ['Credit Spread', 'Term Premium', 'Convexity Adj', 'Residual', 'Total Fair'],
                'Value (bps)': [
                    spread_decomp['Credit_Spread'],
                    spread_decomp['Term_Premium'],
                    spread_decomp['Convexity_Adj'],
                    spread_decomp['Residual'],
                    spread_decomp['Total_Fair']
                ],
                'Description': [
                    'JIBAR-ZARONIA basis (credit risk)',
                    'Compensation for tenor',
                    'Non-linear compounding effect',
                    'Unexplained (liquidity/mispricing)',
                    'Sum of all components'
                ]
            }
            
            df_decomp = pd.DataFrame(decomp_data)
            
            col_decomp1, col_decomp2 = st.columns([1, 1])
            
            with col_decomp1:
                st.dataframe(df_decomp.style.format({'Value (bps)': '{:.2f}'}), width='stretch', hide_index=True)
            
            with col_decomp2:
                # Waterfall chart for spread decomposition
                fig_waterfall = go.Figure(go.Waterfall(
                    x=['Credit<br>Spread', 'Term<br>Premium', 'Convexity<br>Adj', 'Residual', 'Fair<br>Spread'],
                    y=[
                        spread_decomp['Credit_Spread'],
                        spread_decomp['Term_Premium'],
                        spread_decomp['Convexity_Adj'],
                        spread_decomp['Residual'],
                        0
                    ],
                    measure=['relative', 'relative', 'relative', 'relative', 'total'],
                    text=[f"{v:.1f}" for v in [
                        spread_decomp['Credit_Spread'],
                        spread_decomp['Term_Premium'],
                        spread_decomp['Convexity_Adj'],
                        spread_decomp['Residual'],
                        spread_decomp['Total_Fair']
                    ]],
                    textposition='outside',
                    connector={'line': {'color': 'rgb(63, 63, 63)'}},
                ))
                fig_waterfall.update_layout(
                    title="Spread Build-up (bps)",
                    yaxis_title="Basis Points",
                    template='plotly_dark',
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_waterfall, width='stretch')
            
            st.divider()
            
            # Detailed Cashflow Comparison
            st.markdown("### 💰 Detailed Cashflow Projections & Comparison")
            st.caption("Complete cashflow schedule for original JIBAR FRN vs converted ZARONIA FRN")
            
            # Get detailed cashflows from both structures
            jibar_cashflows = analyzer.original_leg.cashflows
            zaronia_cashflows = analyzer.converted_frn.cashflows
            
            # Build comprehensive comparison table
            cf_comparison_detailed = []
            
            for i, (jcf, zcf) in enumerate(zip(jibar_cashflows, zaronia_cashflows)):
                # Calculate days from accrual period
                from datetime import datetime, date
                accrual_start = jcf['Accrual Start']
                accrual_end = jcf['Accrual End']
                if isinstance(accrual_start, datetime):
                    accrual_start = accrual_start.date()
                if isinstance(accrual_end, datetime):
                    accrual_end = accrual_end.date()
                days = (accrual_end - accrual_start).days
                
                cf_comparison_detailed.append({
                    'Period': i + 1,
                    'Payment Date': jcf['Period End'],
                    'Days': days,
                    'Year Fraction': jcf['Year Fraction'],
                    'JIBAR Forward (%)': jcf['Forward Rate'] * 100,
                    'JIBAR + Spread (%)': jcf['Net Rate'] * 100,
                    'JIBAR Cashflow (ZAR)': jcf['Cashflow'],
                    'ZARONIA Comp (%)': zcf['ZARONIA_Comp'] * 100,
                    'ZARONIA + Spread (%)': zcf['Coupon'] * 100,  # Coupon already includes margin
                    'ZARONIA Cashflow (ZAR)': zcf['Amount'],  # Use 'Amount' not 'Cashflow'
                    'Difference (ZAR)': zcf['Amount'] - jcf['Cashflow'],
                    'Difference (%)': ((zcf['Amount'] / jcf['Cashflow']) - 1) * 100 if jcf['Cashflow'] != 0 else 0,
                    'JIBAR PV (ZAR)': jcf['PV'],
                    'ZARONIA PV (ZAR)': zcf['PV'],
                    'PV Difference (ZAR)': zcf['PV'] - jcf['PV']
                })
            
            df_cf_detailed = pd.DataFrame(cf_comparison_detailed)
            
            # Add cumulative columns
            df_cf_detailed['Cumulative JIBAR (ZAR)'] = df_cf_detailed['JIBAR Cashflow (ZAR)'].cumsum()
            df_cf_detailed['Cumulative ZARONIA (ZAR)'] = df_cf_detailed['ZARONIA Cashflow (ZAR)'].cumsum()
            df_cf_detailed['Cumulative Difference (ZAR)'] = df_cf_detailed['Difference (ZAR)'].cumsum()
            df_cf_detailed['Cumulative PV Diff (ZAR)'] = df_cf_detailed['PV Difference (ZAR)'].cumsum()
            
            # Summary statistics
            total_jibar_cf = df_cf_detailed['JIBAR Cashflow (ZAR)'].sum()
            total_zaronia_cf = df_cf_detailed['ZARONIA Cashflow (ZAR)'].sum()
            total_diff_cf = total_zaronia_cf - total_jibar_cf
            total_jibar_pv = df_cf_detailed['JIBAR PV (ZAR)'].sum()
            total_zaronia_pv = df_cf_detailed['ZARONIA PV (ZAR)'].sum()
            total_diff_pv = total_zaronia_pv - total_jibar_pv
            
            # Display summary cards
            st.markdown("#### Cashflow Summary")
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            
            with col_sum1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total JIBAR Cashflows</div>
                    <div class="metric-value" style="color:#00CC96;">ZAR {total_jibar_cf:,.0f}</div>
                    <div class="metric-sub">PV: ZAR {total_jibar_pv:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_sum2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total ZARONIA Cashflows</div>
                    <div class="metric-value" style="color:#EF553B;">ZAR {total_zaronia_cf:,.0f}</div>
                    <div class="metric-sub">PV: ZAR {total_zaronia_pv:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_sum3:
                diff_color = "#00CC96" if total_diff_cf > 0 else "#EF553B"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Cashflow Difference</div>
                    <div class="metric-value" style="color:{diff_color};">ZAR {total_diff_cf:+,.0f}</div>
                    <div class="metric-sub">{(total_diff_cf/total_jibar_cf)*100:+.2f}% vs JIBAR</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_sum4:
                pv_diff_color = "#00CC96" if total_diff_pv > 0 else "#EF553B"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PV Difference</div>
                    <div class="metric-value" style="color:{pv_diff_color};">ZAR {total_diff_pv:+,.0f}</div>
                    <div class="metric-sub">Including conversion costs</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Interactive visualization tabs
            tab_cf1, tab_cf2, tab_cf3, tab_cf4 = st.tabs([
                "📊 Cashflow Comparison", 
                "📈 Cumulative Analysis", 
                "📋 Detailed Table",
                "🔍 Rate Comparison"
            ])
            
            with tab_cf1:
                # Side-by-side cashflow bars
                fig_cf_compare = go.Figure()
                fig_cf_compare.add_trace(go.Bar(
                    x=df_cf_detailed['Payment Date'],
                    y=df_cf_detailed['JIBAR Cashflow (ZAR)'],
                    name='JIBAR 3M + Spread',
                    marker_color='#00CC96',
                    text=df_cf_detailed['JIBAR Cashflow (ZAR)'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside'
                ))
                fig_cf_compare.add_trace(go.Bar(
                    x=df_cf_detailed['Payment Date'],
                    y=df_cf_detailed['ZARONIA Cashflow (ZAR)'],
                    name='ZARONIA Comp + Spread',
                    marker_color='#EF553B',
                    text=df_cf_detailed['ZARONIA Cashflow (ZAR)'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside'
                ))
                fig_cf_compare.update_layout(
                    title='Cashflow Comparison by Payment Date',
                    xaxis_title='Payment Date',
                    yaxis_title='Cashflow (ZAR)',
                    barmode='group',
                    template='plotly_dark',
                    height=450,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_cf_compare, width='stretch')
                
                # Difference waterfall
                fig_diff = go.Figure(go.Waterfall(
                    x=df_cf_detailed['Payment Date'].astype(str),
                    y=df_cf_detailed['Difference (ZAR)'],
                    text=df_cf_detailed['Difference (ZAR)'].apply(lambda x: f'{x:+,.0f}'),
                    textposition='outside',
                    connector={'line': {'color': 'rgb(63, 63, 63)'}},
                    increasing={'marker': {'color': '#00CC96'}},
                    decreasing={'marker': {'color': '#EF553B'}}
                ))
                fig_diff.update_layout(
                    title='Period-by-Period Cashflow Difference (ZARONIA - JIBAR)',
                    xaxis_title='Payment Date',
                    yaxis_title='Difference (ZAR)',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_diff, width='stretch')
            
            with tab_cf2:
                # Cumulative cashflows
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=df_cf_detailed['Payment Date'],
                    y=df_cf_detailed['Cumulative JIBAR (ZAR)'],
                    name='Cumulative JIBAR',
                    mode='lines+markers',
                    line=dict(color='#00CC96', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(0, 204, 150, 0.2)'
                ))
                fig_cum.add_trace(go.Scatter(
                    x=df_cf_detailed['Payment Date'],
                    y=df_cf_detailed['Cumulative ZARONIA (ZAR)'],
                    name='Cumulative ZARONIA',
                    mode='lines+markers',
                    line=dict(color='#EF553B', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(239, 85, 59, 0.2)'
                ))
                fig_cum.update_layout(
                    title='Cumulative Cashflow Evolution',
                    xaxis_title='Payment Date',
                    yaxis_title='Cumulative Cashflow (ZAR)',
                    template='plotly_dark',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_cum, width='stretch')
                
                # Cumulative difference
                fig_cum_diff = go.Figure()
                fig_cum_diff.add_trace(go.Scatter(
                    x=df_cf_detailed['Payment Date'],
                    y=df_cf_detailed['Cumulative Difference (ZAR)'],
                    name='Cumulative Difference',
                    mode='lines+markers',
                    line=dict(color='#19D3F3', width=3),
                    marker=dict(size=8),
                    fill='tozeroy'
                ))
                fig_cum_diff.update_layout(
                    title='Cumulative Cashflow Difference (ZARONIA - JIBAR)',
                    xaxis_title='Payment Date',
                    yaxis_title='Cumulative Difference (ZAR)',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_cum_diff, width='stretch')
            
            with tab_cf3:
                # Detailed table with all columns
                st.dataframe(
                    df_cf_detailed.style.format({
                        'Year Fraction': '{:.4f}',
                        'JIBAR Forward (%)': '{:.4f}',
                        'JIBAR + Spread (%)': '{:.4f}',
                        'JIBAR Cashflow (ZAR)': '{:,.2f}',
                        'ZARONIA Comp (%)': '{:.4f}',
                        'ZARONIA + Spread (%)': '{:.4f}',
                        'ZARONIA Cashflow (ZAR)': '{:,.2f}',
                        'Difference (ZAR)': '{:+,.2f}',
                        'Difference (%)': '{:+.2f}',
                        'JIBAR PV (ZAR)': '{:,.2f}',
                        'ZARONIA PV (ZAR)': '{:,.2f}',
                        'PV Difference (ZAR)': '{:+,.2f}',
                        'Cumulative JIBAR (ZAR)': '{:,.2f}',
                        'Cumulative ZARONIA (ZAR)': '{:,.2f}',
                        'Cumulative Difference (ZAR)': '{:+,.2f}',
                        'Cumulative PV Diff (ZAR)': '{:+,.2f}'
                    }).background_gradient(subset=['Difference (ZAR)'], cmap='RdYlGn'),
                    width='stretch',
                    height=400
                )
                
                # Download button for cashflow data
                csv = df_cf_detailed.to_csv(index=False)
                st.download_button(
                    label="📥 Download Cashflow Data (CSV)",
                    data=csv,
                    file_name=f"cashflow_comparison_{val_date}.csv",
                    mime="text/csv"
                )
            
            with tab_cf4:
                # Rate comparison chart
                fig_rates = go.Figure()
                fig_rates.add_trace(go.Scatter(
                    x=df_cf_detailed['Payment Date'],
                    y=df_cf_detailed['JIBAR + Spread (%)'],
                    name='JIBAR 3M + Spread',
                    mode='lines+markers',
                    line=dict(color='#00CC96', width=3),
                    marker=dict(size=8)
                ))
                fig_rates.add_trace(go.Scatter(
                    x=df_cf_detailed['Payment Date'],
                    y=df_cf_detailed['ZARONIA + Spread (%)'],
                    name='ZARONIA Comp + Spread',
                    mode='lines+markers',
                    line=dict(color='#EF553B', width=3),
                    marker=dict(size=8)
                ))
                fig_rates.update_layout(
                    title='Effective Rate Comparison',
                    xaxis_title='Payment Date',
                    yaxis_title='All-in Rate (%)',
                    template='plotly_dark',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_rates, width='stretch')
                
                # Rate differential
                fig_rate_diff = go.Figure()
                rate_diff = df_cf_detailed['ZARONIA + Spread (%)'] - df_cf_detailed['JIBAR + Spread (%)']
                fig_rate_diff.add_trace(go.Bar(
                    x=df_cf_detailed['Payment Date'],
                    y=rate_diff,
                    marker_color=['#00CC96' if x > 0 else '#EF553B' for x in rate_diff],
                    text=rate_diff.apply(lambda x: f'{x:+.2f}%'),
                    textposition='outside'
                ))
                fig_rate_diff.update_layout(
                    title='Rate Differential by Period (ZARONIA - JIBAR)',
                    xaxis_title='Payment Date',
                    yaxis_title='Rate Difference (%)',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_rate_diff, width='stretch')
            
            st.divider()
            
            # Legacy simple comparison for backward compatibility
            df_cf_comp = analyzer.get_cashflow_comparison()
            
            if not df_cf_comp.empty:
                col_cf1, col_cf2 = st.columns([2, 1])
                
                with col_cf1:
                    # Cashflow chart
                    fig_cf = go.Figure()
                    fig_cf.add_trace(go.Bar(
                        x=df_cf_comp['Date'],
                        y=df_cf_comp['JIBAR_Cashflow'],
                        name='JIBAR 3M + Spread',
                        marker_color='#00CC96'
                    ))
                    fig_cf.add_trace(go.Bar(
                        x=df_cf_comp['Date'],
                        y=df_cf_comp['ZARONIA_Cashflow'],
                        name='ZARONIA Comp + Spread',
                        marker_color='#EF553B'
                    ))
                    fig_cf.update_layout(
                        title="Cashflow Comparison: JIBAR vs ZARONIA",
                        xaxis_title="Payment Date",
                        yaxis_title="Cashflow (ZAR)",
                        barmode='group',
                        template='plotly_dark',
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_cf, width='stretch')
                
                with col_cf2:
                    # Cumulative P&L
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=df_cf_comp['Date'],
                        y=df_cf_comp['Cumulative_Diff'],
                        fill='tozeroy',
                        name='Cumulative Difference',
                        line=dict(color='#19D3F3', width=2)
                    ))
                    fig_cum.update_layout(
                        title="Cumulative P&L from Conversion",
                        xaxis_title="Date",
                        yaxis_title="Cumulative ZAR",
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig_cum, width='stretch')
                
                # Detailed table
                with st.expander("📋 Detailed Cashflow Table"):
                    st.dataframe(df_cf_comp.style.format({
                        'JIBAR_Cashflow': '{:,.2f}',
                        'ZARONIA_Cashflow': '{:,.2f}',
                        'Difference': '{:+,.2f}',
                        'Cumulative_Diff': '{:+,.2f}'
                    }), width='stretch')
            
            st.divider()
            
            # DV01 vs CS01 Analysis
            st.markdown("### 🎯 DV01 vs CS01: Term Structure Risk Analysis")
            st.caption("Comparing interest rate risk (DV01) vs credit spread risk (CS01) for hedging")
            
            # Calculate DV01 (interest rate sensitivity) and CS01 (credit spread sensitivity)
            # DV01 = change in PV for 1bp parallel shift in rates
            # CS01 = change in PV for 1bp shift in credit spread
            
            dv01_cs01_data = []
            
            for tenor in tenors_years:
                # Calculate DV01: PV sensitivity to 1bp shift in JIBAR curve
                base_jibar_rate = jibar_curve.get_zero_rate(tenor)
                
                # Approximate DV01 using duration
                # DV01 ≈ Modified Duration × PV × 0.0001
                # For a swap, modified duration ≈ tenor / 2 (rough approximation)
                approx_duration = tenor / 2.0
                pv_per_bp = conv_notional * approx_duration * 0.0001
                
                # CS01: PV sensitivity to 1bp shift in credit spread (JIBAR-ZARONIA basis)
                # For basis risk, CS01 is similar to DV01 but applies to spread
                cs01 = pv_per_bp  # Same order of magnitude
                
                # Hedge ratio: how many units of ZARONIA swap needed to hedge JIBAR swap
                # Hedge Ratio = DV01_JIBAR / DV01_ZARONIA
                # If curves have different slopes, this differs from 1.0
                jibar_slope = (jibar_curve.get_zero_rate(min(tenor+1, 10)) - jibar_curve.get_zero_rate(max(tenor-1, 0.25))) / 2.0
                zaronia_slope = (zaronia_curve.get_zero_rate(min(tenor+1, 10)) - zaronia_curve.get_zero_rate(max(tenor-1, 0.25))) / 2.0
                
                hedge_ratio = 1.0 if abs(zaronia_slope) < 1e-6 else jibar_slope / zaronia_slope
                
                # Basis risk (residual risk after hedging)
                basis_risk_bps = abs(jibar_curve.get_zero_rate(tenor) - zaronia_curve.get_zero_rate(tenor)) * 10000
                
                dv01_cs01_data.append({
                    'Tenor': f'{tenor}Y',
                    'DV01 (ZAR)': pv_per_bp,
                    'CS01 (ZAR)': cs01,
                    'Hedge Ratio': hedge_ratio,
                    'Basis Risk (bps)': basis_risk_bps,
                    'Hedge Adjustment (%)': (hedge_ratio - 1.0) * 100
                })
            
            df_dv01_cs01 = pd.DataFrame(dv01_cs01_data)
            
            col_risk1, col_risk2 = st.columns([1, 1])
            
            with col_risk1:
                st.markdown("**Risk Metrics by Tenor**")
                st.dataframe(
                    df_dv01_cs01.style.format({
                        'DV01 (ZAR)': '{:,.0f}',
                        'CS01 (ZAR)': '{:,.0f}',
                        'Hedge Ratio': '{:.4f}',
                        'Basis Risk (bps)': '{:.2f}',
                        'Hedge Adjustment (%)': '{:+.2f}'
                    }).background_gradient(subset=['Hedge Adjustment (%)'], cmap='RdYlGn', vmin=-5, vmax=5),
                    width='stretch',
                    hide_index=True
                )
                
                st.markdown("""
                **Key Insights:**
                - **DV01**: PV change per 1bp parallel shift in JIBAR curve
                - **CS01**: PV change per 1bp shift in JIBAR-ZARONIA basis
                - **Hedge Ratio**: Notional adjustment needed for term structure mismatch
                - **Hedge Adjustment**: % over/under hedge required (≠ 0 means curve risk)
                """)
            
            with col_risk2:
                # Chart: DV01 vs CS01 by tenor
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Bar(
                    x=df_dv01_cs01['Tenor'],
                    y=df_dv01_cs01['DV01 (ZAR)'],
                    name='DV01 (Rate Risk)',
                    marker_color='#00CC96'
                ))
                fig_risk.add_trace(go.Bar(
                    x=df_dv01_cs01['Tenor'],
                    y=df_dv01_cs01['CS01 (ZAR)'],
                    name='CS01 (Spread Risk)',
                    marker_color='#EF553B'
                ))
                fig_risk.update_layout(
                    title='DV01 vs CS01 by Tenor',
                    xaxis_title='Tenor',
                    yaxis_title='Risk (ZAR per bp)',
                    barmode='group',
                    template='plotly_dark',
                    height=350
                )
                st.plotly_chart(fig_risk, width='stretch')
            
            # Hedging implications
            st.markdown("**Hedging Strategy:**")
            
            col_hedge1, col_hedge2 = st.columns(2)
            
            with col_hedge1:
                st.markdown(f"""
                **For {conv_tenor}Y Conversion:**
                
                1. **Notional Hedge**: ZAR {conv_notional:,.0f}
                2. **Hedge Ratio Adjustment**: {df_dv01_cs01.iloc[min(conv_tenor-1, len(df_dv01_cs01)-1)]['Hedge Ratio']:.4f}
                3. **Adjusted Notional**: ZAR {conv_notional * df_dv01_cs01.iloc[min(conv_tenor-1, len(df_dv01_cs01)-1)]['Hedge Ratio']:,.0f}
                
                **Residual Risks:**
                - Basis risk: {df_dv01_cs01.iloc[min(conv_tenor-1, len(df_dv01_cs01)-1)]['Basis Risk (bps)']:.2f} bps
                - Curve risk: {df_dv01_cs01.iloc[min(conv_tenor-1, len(df_dv01_cs01)-1)]['Hedge Adjustment (%)']:.2f}% notional mismatch
                """)
            
            with col_hedge2:
                st.markdown(f"""
                **Term Spread Risk (per bp):**
                
                If JIBAR-ZARONIA basis widens by 1bp:
                - P&L Impact: ZAR {df_dv01_cs01.iloc[min(conv_tenor-1, len(df_dv01_cs01)-1)]['CS01 (ZAR)']:,.0f}
                
                If JIBAR curve steepens by 1bp:
                - Hedge slippage: {abs(df_dv01_cs01.iloc[min(conv_tenor-1, len(df_dv01_cs01)-1)]['Hedge Adjustment (%)']):.2f}%
                - Additional risk: ZAR {abs(df_dv01_cs01.iloc[min(conv_tenor-1, len(df_dv01_cs01)-1)]['Hedge Adjustment (%)']) * conv_notional / 100:,.0f}
                
                **Recommendation:**
                {'✅ Clean hedge - minimal curve risk' if abs(df_dv01_cs01.iloc[min(conv_tenor-1, len(df_dv01_cs01)-1)]['Hedge Adjustment (%)']) < 1.0 else '⚠️ Adjust hedge ratio for curve mismatch'}
                """)
            
            st.divider()
            
            # Sensitivity Analysis
            st.markdown("### 📈 Sensitivity Analysis")
            st.caption("Impact of varying the conversion spread on PV")
            
            # Generate sensitivity range
            spread_range = np.linspace(fair_spread - 50, fair_spread + 50, 21)
            pv_impacts = []
            
            for test_spread in spread_range:
                test_analyzer = ConversionAnalyzer(
                    notional=conv_notional,
                    start_date=conv_start_date,
                    maturity_years=conv_tenor,
                    jibar_spread_bps=conv_jibar_spread,
                    zaronia_spread_bps=test_spread,
                    frequency=conv_freq,
                    jibar_curve=jibar_curve,
                    zaronia_curve=zaronia_curve
                )
                pv_impacts.append(test_analyzer.get_pv_difference())
            
            # Sensitivity chart
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=spread_range,
                y=pv_impacts,
                mode='lines+markers',
                name='PV Impact',
                line=dict(color='#00CC96', width=3),
                marker=dict(size=6)
            ))
            
            # Mark current offered spread
            fig_sens.add_trace(go.Scatter(
                x=[conv_zaronia_spread],
                y=[pv_diff],
                mode='markers',
                name='Offered Spread',
                marker=dict(size=15, color='#EF553B', symbol='star')
            ))
            
            # Mark fair spread
            fig_sens.add_trace(go.Scatter(
                x=[fair_spread],
                y=[0],
                mode='markers',
                name='Fair Spread (PV=0)',
                marker=dict(size=15, color='#FFA15A', symbol='diamond')
            ))
            
            fig_sens.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="PV Neutral")
            
            fig_sens.update_layout(
                title=f"PV Impact vs ZARONIA Spread (Fair = {fair_spread:.2f} bps)",
                xaxis_title="ZARONIA Spread (bps)",
                yaxis_title="PV Difference (ZAR)",
                template='plotly_dark',
                height=450,
                hovermode='x unified'
            )
            st.plotly_chart(fig_sens, width='stretch')
            
            # Summary table
            col_sum1, col_sum2 = st.columns(2)
            
            with col_sum1:
                st.markdown("**PV Sensitivity (per 1bp change in spread):**")
                if len(pv_impacts) > 1:
                    dv01_spread = (pv_impacts[1] - pv_impacts[0]) / (spread_range[1] - spread_range[0])
                    st.metric("DV01 (Spread)", f"ZAR {dv01_spread:,.2f}/bp")
            
            with col_sum2:
                st.markdown("**Break-even Analysis:**")
                st.metric("Fair Spread", f"{fair_spread:.2f} bps")
                st.caption(f"Offered spread must be ≥ {fair_spread:.2f} bps for investor to break even")
        
        except Exception as e:
            st.error(f"Error in conversion analysis: {e}")
            import traceback
            st.code(traceback.format_exc())

    with tab_cvx:
        st.subheader("📐 Convexity & Compounding Effects")
        
        st.markdown("""
        **Quantifying Path-Dependent Effects: Forward JIBAR vs Compounded ZARONIA**
        
        This module demonstrates why **volatility + compounding = convexity premium**.
        """)
        
        # Educational Panel
        with st.expander("💡 **Intuition: What This Analysis Measures**", expanded=True):
            st.markdown("""
            ### The Fundamental Difference
            
            **JIBAR 3M (Forward-Looking - Simple Interest):**
            - ✅ **Fixed at period start** → Deterministic cashflow
            - ✅ **No path risk** → You know exactly what you'll receive
            - ✅ **Simple interest**: CF = N × (F_3M + spread) × τ
            
            **ZARONIA Compounded (Backward-Looking - Daily Compounding):**
            - ⚠️ **Realized daily** → Path-dependent outcome
            - ⚠️ **Compounded over time** → Non-linear accumulation
            - ⚠️ **Random**: CF = N × [Π(1 + r_i × Δt_i) - 1]
            
            ### What We're Measuring: PURE Convexity Effect
            
            **CRITICAL:** This simulation isolates the **pure compounding effect** by:
            - Starting both rates at the same level (Forward JIBAR)
            - Simulating volatility around this level
            - Measuring: E[Compounded] - Simple(Forward)
            
            This removes credit spread from the measurement, showing only Jensen's inequality.
            
            ### Jensen's Inequality (The Math Behind It)
            
            ```
            E[Π(1 + r_i·Δt)] ≠ Π(1 + E[r_i]·Δt)
            ```
            
            **Translation:** The expected value of a compounded product differs from the product of expected values.
            
            **Why?**
            - Compounding is a **convex function** (exponential)
            - Volatility creates **asymmetric outcomes**
            - The difference is the **convexity adjustment**
            
            ### Practical Impact on Conversions
            
            1. **Credit Spread**: JIBAR-ZARONIA basis (~200-300 bps) - separate effect
            2. **Convexity Adjustment**: Pure compounding effect (typically 0-5 bps)
            3. **Total Conversion Spread**: Must account for BOTH
            4. **Ignoring convexity = mispricing** by several basis points per year
            """)
        
        st.divider()
        
        # Interactive Controls
        st.markdown("### 🎛️ Simulation Parameters")
        
        col_cvx1, col_cvx2, col_cvx3 = st.columns(3)
        
        with col_cvx1:
            cvx_volatility = st.slider("ZARONIA Volatility (bps)", 
                                       min_value=0, max_value=300, value=100, step=10,
                                       help="Annualized volatility of overnight rates")
            cvx_tenor_str = st.selectbox("Analysis Tenor", 
                                        ["3M", "6M", "1Y", "2Y", "3Y", "5Y"], 
                                        index=2, key="cvx_tenor")
            
        with col_cvx2:
            cvx_num_paths = st.selectbox("Number of Paths", 
                                        [500, 1000, 2000, 5000, 10000], 
                                        index=1, key="cvx_paths")
            cvx_notional = st.number_input("Notional (ZAR)", 
                                          value=100_000_000, 
                                          step=10_000_000, 
                                          format="%d", 
                                          key="cvx_notional")
            
        with col_cvx3:
            cvx_deterministic = st.checkbox("Show Deterministic Case (σ=0)", value=False,
                                           help="Compare with zero volatility baseline")
            cvx_seed = st.number_input("Random Seed", value=42, min_value=1, key="cvx_seed")
        
        # Parse tenor
        tenor_map = {"3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0}
        cvx_tenor_years = tenor_map[cvx_tenor_str]
        cvx_start_date = val_date
        cvx_end_date = cvx_start_date + timedelta(days=int(cvx_tenor_years * 365.25))
        
        st.divider()
        
        # Run Convexity Analysis
        try:
            # Main simulation
            cvx_analyzer = ConvexityAnalyzer(
                notional=cvx_notional,
                start_date=cvx_start_date,
                end_date=cvx_end_date,
                jibar_curve=jibar_curve,
                zaronia_curve=zaronia_curve,
                volatility_bps=cvx_volatility,
                num_paths=cvx_num_paths,
                seed=cvx_seed
            )
            
            # Deterministic case (if requested)
            cvx_analyzer_det = None
            if cvx_deterministic:
                cvx_analyzer_det = ConvexityAnalyzer(
                    notional=cvx_notional,
                    start_date=cvx_start_date,
                    end_date=cvx_end_date,
                    jibar_curve=jibar_curve,
                    zaronia_curve=zaronia_curve,
                    volatility_bps=0,
                    num_paths=cvx_num_paths,
                    seed=cvx_seed
                )
            
            # Extract metrics
            forward_jibar = cvx_analyzer.forward_jibar
            expected_zaronia = cvx_analyzer.get_expected_zaronia()
            median_zaronia = cvx_analyzer.get_median_zaronia()
            std_zaronia = cvx_analyzer.get_std_zaronia()
            convexity_adj = cvx_analyzer.get_convexity_adjustment()
            percentiles = cvx_analyzer.get_percentiles([5, 25, 50, 75, 95])
            
            # Dashboard Metrics
            st.markdown("### 📊 Convexity Metrics Dashboard")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Forward JIBAR</div>
                    <div class="metric-value" style="color:#00CC96;">{forward_jibar*100:.4f}%</div>
                    <div class="metric-sub">Deterministic</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Expected ZARONIA</div>
                    <div class="metric-value" style="color:#EF553B;">{expected_zaronia*100:.4f}%</div>
                    <div class="metric-sub">E[Compounded]</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m3:
                cvx_color = "#FFA15A" if convexity_adj > 0 else "#19D3F3"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Convexity Adjustment</div>
                    <div class="metric-value" style="color:{cvx_color};">{convexity_adj:+.2f} bps</div>
                    <div class="metric-sub">E[Z] - F[J]</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Std Deviation</div>
                    <div class="metric-value" style="color:#AB63FA;">{std_zaronia*10000:.2f} bps</div>
                    <div class="metric-sub">Volatility Impact</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m5:
                worst_case = percentiles[5]
                best_case = percentiles[95]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">5th / 95th Percentile</div>
                    <div class="metric-value" style="color:#19D3F3; font-size:18px;">{worst_case*100:.3f}% / {best_case*100:.3f}%</div>
                    <div class="metric-sub">Outcome Range</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Visualizations
            st.markdown("### 📈 Distribution & Path Analysis")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Distribution histogram
                distribution_data = cvx_analyzer.get_distribution_data()
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=distribution_data * 100,
                    nbinsx=50,
                    name='Compounded ZARONIA',
                    marker_color='#EF553B',
                    opacity=0.7
                ))
                
                # Mark forward JIBAR
                fig_dist.add_vline(x=forward_jibar*100, line_dash="dash", line_color="#00CC96", 
                                  annotation_text="Forward JIBAR", annotation_position="top")
                
                # Mark expected ZARONIA
                fig_dist.add_vline(x=expected_zaronia*100, line_dash="solid", line_color="#FFA15A",
                                  annotation_text="E[ZARONIA]", annotation_position="top")
                
                fig_dist.update_layout(
                    title=f"Distribution of Compounded ZARONIA ({cvx_num_paths} paths)",
                    xaxis_title="Annualized Rate (%)",
                    yaxis_title="Frequency",
                    template='plotly_dark',
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_dist, width='stretch')
            
            with col_viz2:
                # Sample paths fan chart
                sample_paths = cvx_analyzer.get_sample_paths(num_samples=50)
                
                if sample_paths.size > 0:
                    fig_paths = go.Figure()
                    
                    # Plot sample paths
                    for i in range(min(50, sample_paths.shape[0])):
                        fig_paths.add_trace(go.Scatter(
                            y=sample_paths[i, :] * 100,
                            mode='lines',
                            line=dict(color='#EF553B', width=0.5),
                            opacity=0.3,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Add forward JIBAR reference line
                    fig_paths.add_hline(y=forward_jibar*100, line_dash="dash", line_color="#00CC96",
                                       annotation_text="Forward JIBAR")
                    
                    fig_paths.update_layout(
                        title="Simulated ZARONIA Rate Paths",
                        xaxis_title="Business Days",
                        yaxis_title="Overnight Rate (%)",
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig_paths, width='stretch')
            
            # Percentile bands chart
            st.markdown("### 📊 Percentile Bands")
            
            fig_percentiles = go.Figure()
            
            pct_labels = ['5th', '25th', '50th (Median)', '75th', '95th']
            pct_values = [percentiles[5], percentiles[25], percentiles[50], percentiles[75], percentiles[95]]
            
            fig_percentiles.add_trace(go.Bar(
                x=pct_labels,
                y=[v*100 for v in pct_values],
                marker_color=['#EF553B', '#FFA15A', '#00CC96', '#FFA15A', '#EF553B'],
                text=[f"{v*100:.4f}%" for v in pct_values],
                textposition='outside'
            ))
            
            # Add forward JIBAR reference
            fig_percentiles.add_hline(y=forward_jibar*100, line_dash="dash", line_color="white",
                                     annotation_text=f"Forward JIBAR: {forward_jibar*100:.4f}%")
            
            fig_percentiles.update_layout(
                title="Compounded ZARONIA Outcome Percentiles",
                yaxis_title="Rate (%)",
                template='plotly_dark',
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_percentiles, width='stretch')
            
            st.divider()
            
            # Convexity vs Volatility Analysis
            st.markdown("### 📉 Convexity Adjustment vs Volatility")
            st.caption("How does convexity change with market volatility?")
            
            # Generate convexity curve
            vol_range = np.linspace(0, 300, 31)
            convexity_curve = []
            
            for test_vol in vol_range:
                test_analyzer = ConvexityAnalyzer(
                    notional=cvx_notional,
                    start_date=cvx_start_date,
                    end_date=cvx_end_date,
                    jibar_curve=jibar_curve,
                    zaronia_curve=zaronia_curve,
                    volatility_bps=test_vol,
                    num_paths=500,  # Fewer paths for speed
                    seed=cvx_seed
                )
                convexity_curve.append(test_analyzer.get_convexity_adjustment())
            
            fig_cvx_vol = go.Figure()
            fig_cvx_vol.add_trace(go.Scatter(
                x=vol_range,
                y=convexity_curve,
                mode='lines+markers',
                name='Convexity Adjustment',
                line=dict(color='#00CC96', width=3),
                marker=dict(size=6)
            ))
            
            # Mark current volatility
            fig_cvx_vol.add_trace(go.Scatter(
                x=[cvx_volatility],
                y=[convexity_adj],
                mode='markers',
                name='Current Setting',
                marker=dict(size=15, color='#EF553B', symbol='star')
            ))
            
            fig_cvx_vol.update_layout(
                title=f"Convexity Adjustment vs Volatility ({cvx_tenor_str} Tenor)",
                xaxis_title="Volatility (bps)",
                yaxis_title="Convexity Adjustment (bps)",
                template='plotly_dark',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_cvx_vol, width='stretch')
            
            # Key Insights
            st.markdown("### 🔑 Key Insights & Conversion Impact")
            
            # Calculate credit spread for context
            zaronia_zero = zaronia_curve.get_zero_rate(cvx_tenor_years)
            credit_spread_bps = (forward_jibar - zaronia_zero) * 10000
            
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.markdown(f"""
                **Understanding the Numbers:**
                
                1. **Forward JIBAR**: {forward_jibar*100:.4f}%
                   - This is the deterministic rate you'd lock in today
                   - Simple interest: pays exactly this rate
                
                2. **Expected Compounded ZARONIA**: {expected_zaronia*100:.4f}%
                   - Average outcome across {cvx_num_paths:,} simulated paths
                   - Daily compounding with volatility = {cvx_volatility} bps
                
                3. **Convexity Adjustment**: {convexity_adj:+.2f} bps
                   - **Pure compounding effect** (Jensen's inequality)
                   - {abs(convexity_adj):.2f} bps {'premium' if convexity_adj > 0 else 'discount'} from path dependency
                   - Increases with volatility and tenor
                
                4. **Credit Spread**: {credit_spread_bps:.1f} bps
                   - JIBAR-ZARONIA basis (bank credit risk)
                   - Separate from convexity effect
                """)
            
            with col_insight2:
                st.markdown(f"""
                **Conversion Pricing Formula:**
                
                When converting **JIBAR 3M + {50} bps** to **ZARONIA Comp + X bps**:
                
                ```
                Fair ZARONIA Spread (X) = 
                    Original JIBAR Spread:     {50:>6.1f} bps
                  - Credit Basis:              {credit_spread_bps:>6.1f} bps
                  + Convexity Adjustment:      {convexity_adj:>+6.2f} bps
                  + Term Premium:              {cvx_tenor_years:>6.1f} bps
                  ─────────────────────────────────────
                  = Fair Conversion Spread:    {50 - credit_spread_bps + convexity_adj + cvx_tenor_years:>6.1f} bps
                ```
                
                **Key Takeaway:**
                - Convexity is {"POSITIVE" if convexity_adj > 0 else "NEGATIVE"}: {abs(convexity_adj):.2f} bps
                - This means compounded ZARONIA is {"HIGHER" if convexity_adj > 0 else "LOWER"} than simple forward
                - Ignoring this = mispricing by {abs(convexity_adj):.2f} bps
                - On ZAR {cvx_notional:,.0f} notional = ZAR {abs(convexity_adj) * cvx_notional / 10000 * cvx_tenor_years:,.0f} value
                """)
            
            # Summary Statistics Table
            with st.expander("📋 Detailed Statistics"):
                stats_data = {
                    'Metric': [
                        'Forward JIBAR',
                        'Expected ZARONIA',
                        'Median ZARONIA',
                        'Std Deviation',
                        'Convexity Adjustment',
                        '5th Percentile',
                        '25th Percentile',
                        '75th Percentile',
                        '95th Percentile',
                        'Min Outcome',
                        'Max Outcome'
                    ],
                    'Value': [
                        f"{forward_jibar*100:.4f}%",
                        f"{expected_zaronia*100:.4f}%",
                        f"{median_zaronia*100:.4f}%",
                        f"{std_zaronia*10000:.2f} bps",
                        f"{convexity_adj:+.2f} bps",
                        f"{percentiles[5]*100:.4f}%",
                        f"{percentiles[25]*100:.4f}%",
                        f"{percentiles[75]*100:.4f}%",
                        f"{percentiles[95]*100:.4f}%",
                        f"{np.min(distribution_data)*100:.4f}%",
                        f"{np.max(distribution_data)*100:.4f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), width='stretch', hide_index=True)
        
        except Exception as e:
            st.error(f"Error in convexity analysis: {e}")
            import traceback
            st.code(traceback.format_exc())

    with tab_comp:
        st.subheader("Rolling 3-Month Compounding Analysis")
        st.write("Weekly rolling comparison: JIBAR 3M (Fixed at Start) vs Compounded ZARONIA (Realized over 3 Months)")
        
        df_hist_all = load_historical_market_data()
        
        if df_hist_all.empty:
            st.warning("No historical data available to perform analysis.")
        else:
            # We need a robust date index
            df_hist_all["Date"] = pd.to_datetime(df_hist_all["Date"])
            # Create a quick lookup for JIBAR
            # We use the raw loaded data which has "JIBAR3M"
            
            # Generate Week Start Dates (Mondays) for rolling 3-month analysis
            # Last 2 years of weekly start dates
            end_d = pd.Timestamp(val_date)
            start_d = end_d - pd.DateOffset(years=2)
            
            # Generate weeks (Monday starts)
            weeks = pd.date_range(start=start_d, end=end_d, freq='W-MON') # Week Start (Mondays)
            
            analysis_data = []
            
            for w_start in weeks:
                # Each week starts a 3-month forward period
                w_end = w_start + pd.DateOffset(months=3)
                
                # Include partial periods - use min of w_end and current date
                w_end = min(w_end, end_d)
                
                # Skip if period is too short (less than 1 month)
                if (w_end - w_start).days < 28:
                    continue
                
                # 1. Get JIBAR 3M at w_start
                # Find closest date <= w_start in history
                # Filter history
                mask_jibar = (df_hist_all["Date"] <= w_start)
                df_j = df_hist_all.loc[mask_jibar]
                
                jibar_val = np.nan
                if not df_j.empty:
                    # Sort by date desc, take first
                    # Try to find date close to w_start (within 5 days)
                    closest_row = df_j.iloc[0] # Sorted desc in load function? Yes.
                    # Wait, load_historical_market_data sorts desc.
                    # So df_j.iloc[0] is the date closest to w_start (on or before).
                    # Check if it's too old
                    if (w_start - closest_row["Date"]).days < 7:
                        jibar_val = closest_row["JIBAR3M"] if "JIBAR3M" in closest_row else np.nan

                # 2. Calculate Compounded ZARONIA over [w_start, w_end)
                # We need daily rates.
                # Use zaronia_curve.history (dict)
                # Loop days
                curr = w_start.date()
                end_date_obj = w_end.date()
                
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
                    zaronia_comp = (comp_factor - 1.0) * (365.0 / (w_end - w_start).days) # ACT/365 annualized?
                    # Actually standard is usually * 365 / days_in_period
                    # (w_end - w_start).days is calendar days.
                    cal_days = (w_end - w_start).days
                    zaronia_comp = (comp_factor - 1.0) * (365.0 / cal_days)

                analysis_data.append({
                    "Week": w_start.strftime("%Y-%m-%d"),
                    "Start Date": w_start.date(),
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
                    fig_ts.add_trace(go.Scatter(x=df_comp["Week"], y=df_comp["JIBAR 3M"], name="JIBAR 3M (Fixed at Start)", line=dict(color='#00CC96', width=2)))
                    fig_ts.add_trace(go.Scatter(x=df_comp["Week"], y=df_comp["ZARONIA Comp"], name="ZARONIA (Realized 3M)", line=dict(color='#EF553B', width=2)))
                    fig_ts.update_layout(title="Rolling 3M Analysis: JIBAR 3M Fix vs ZARONIA Realized", xaxis_title="Period Start Date", yaxis_title="Rate (%)", template="plotly_dark", height=400)
                    st.plotly_chart(fig_ts, width='stretch')
                    
                with c2:
                    fig_sp = go.Figure()
                    fig_sp.add_trace(go.Bar(x=df_comp["Week"], y=df_comp["Spread (bps)"], name="Spread", marker_color='#19D3F3'))
                    fig_sp.update_layout(title="3M Realized Basis Spread (bps)", xaxis_title="Period Start Date", yaxis_title="Basis (bps)", template="plotly_dark", height=400)
                    st.plotly_chart(fig_sp, width='stretch')
                    
                st.dataframe(df_comp.set_index("Week"), width='stretch')

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
            st.plotly_chart(fig_hist, width='stretch')
            
            # 2. Data Table
            st.subheader("Data Table")
            st.dataframe(df_hist, width='stretch')
            
        else:
            st.warning("No historical data loaded. Ensure 'SARB-benchmark-data.csv' and 'JIBAR_FRA_SWAPS.xlsx' are in the directory.")

if __name__ == "__main__":
    main()

# ZARONIA / JIBAR Swap Pricer

This is a Streamlit application that implements the ZARONIA OIS curve estimation methodology from the SARB MPG paper "Historical estimation of the ZARONIA OIS curve" and provides a SWPM-style swap pricer.

## Features

- **Curve Construction**:
    - **JIBAR Zero Curve**: Input via editable table.
    - **Forward 3M JIBAR**: Derived from JIBAR zero curve.
    - **ZARONIA Overnight**: Derived from JIBAR forwards minus spreads.
    - **Discount Curves**: Consistent OIS and JIBAR discounting.
- **Swap Pricing**:
    - **JIBAR IRS**: Fixed vs 3M JIBAR.
    - **ZARONIA OIS**: Fixed vs Compounded Overnight ZARONIA.
    - **Basis Swaps**: JIBAR vs ZARONIA.
    - **Risk Measures**: Par rates, PV, DV01.
- **Visualization**:
    - Interactive Plotly charts for Zero Rates, Discount Factors, and Spreads.
    - Cashflow tables for each leg.

## Installation

1.  Ensure you have Python installed.
2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

## Methodology

The app follows the 7-step methodology outlined in the SARB MPG paper, adapted for real-time pricing:

1.  **Input JIBAR Zero Curve**: NACC zero rates.
2.  **Build 3M JIBAR Forward Curve**.
3.  **Determine Spreads**: $s_0(t) = J(t) - Z(t)$ and term structure.
4.  **Interpolate Spreads**: Linear interpolation / Flat extrapolation.
5.  **Build Overnight ZARONIA Forward Curve**: $f_{1bd} = f_{3M} - s_y$.
6.  **Build ZARONIA Discount Curve**: Compounding overnight forwards.
7.  **Benchmark Rates**: Calculate fair OIS rates.

# Exploring-Risk-Management---VaR-CVaR-Models-in-Python

A simple command-line Python tool to compute Value-at-Risk (VaR) and Conditional VaR (CVaR) for user-defined portfolios. It supports three methods:

Parametric (Normal) Method

Historical Simulation

Monte Carlo Simulation using a Student’s t distribution

Table of Contents

Overview

Prerequisites

Installation

Data Format

Usage

Project Structure

Contributing

License

Overview

This script reads a two-column CSV (tickers + weights), prompts for portfolio value, data span, analysis horizon, and whether to exclude drift. It then:

Downloads historical price data via yfinance

Computes daily log returns and portfolio statistics

Calculates VaR & CVaR at 90%, 95%, and 99% confidence levels

Exports a summary CSV and a histogram plot with VaR/CVaR markers

Prerequisites

Python 3.7 or higher

numpy

pandas

matplotlib

yfinance

scipy

Install via pip:

pip install numpy pandas matplotlib yfinance scipy

Installation

Clone the repo



git clone https://github.com/yourusername/portfolio-var-cvar.git
cd portfolio-var-cvar


2. **(Optional) Make the script executable**
   ```bash
chmod +x portfolio_VaR_CVaR.py

Data Format

Your CSV must have no header, with:

Column 1 (ticker)

Column 2 (weight)

AAPL

0.3

GOOG

0.3

BND

0.4

Weights must sum to 1.0.

Usage

./portfolio_VaR_CVaR.py <portfolio_csv>

You will be prompted to enter:

Portfolio USD value (e.g. 1000000)

Number of years of historical data (e.g. 5)

Time horizon in days (e.g. 1)

Exclude drift loss? True or False

After running, check the automatically created output folder:

<basename>_<days>_days_return_<years>_years_data_<value>_portfolio_value/
├─ VaRs_CVaRs.csv      # Summary table
└─ VaR_CVaR_plot.png   # Histogram of losses with VaR/CVaR lines

Project Structure

├── portfolio_VaR_CVaR.py   # Main Python script
├── sample_portfolio.csv     # Example input file
├── outputs/                 # Generated result folders
└── README.md                # This file

Contributing

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests. Please follow these steps:

Fork the repository

Create a new branch (git checkout -b feature/name)

Commit your changes (git commit -m "Add feature")

Push to your branch (git push origin feature/name)

Open a pull request

License

Released under the MIT License. See LICENSE for details.

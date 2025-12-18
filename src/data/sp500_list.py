"""
S&P 500 constituent list retrieval
"""
import pandas as pd
import requests
from typing import List
import json
from pathlib import Path


def get_sp500_tickers() -> List[str]:
    """
    Get the current list of S&P 500 tickers from Wikipedia.
    Falls back to cached list if fetch fails.
    """
    cache_path = Path(__file__).parent.parent.parent / "data" / "sp500_tickers.json"

    try:
        # Fetch from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()

        # Clean tickers (replace . with - for yfinance compatibility)
        tickers = [t.replace('.', '-') for t in tickers]

        # Cache the result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(tickers, f)

        return tickers

    except Exception as e:
        print(f"Warning: Could not fetch S&P 500 list from Wikipedia: {e}")

        # Try to load from cache
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)

        # Fallback to hardcoded list of major tickers
        return get_fallback_tickers()


def get_fallback_tickers() -> List[str]:
    """Fallback list of major S&P 500 tickers"""
    return [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "XOM",
        "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
        "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT",
        "DHR", "NEE", "VZ", "ADBE", "NKE", "TXN", "PM", "RTX", "CMCSA", "BMY",
        "HON", "ORCL", "T", "COP", "UPS", "MS", "QCOM", "UNP", "LOW", "INTC",
        "IBM", "BA", "CAT", "GE", "AMGN", "SBUX", "DE", "AMD", "GS", "AMAT",
        "PLD", "BLK", "INTU", "MDLZ", "AXP", "GILD", "ADI", "ISRG", "BKNG", "TJX",
        "CVS", "SYK", "REGN", "VRTX", "LRCX", "MMC", "C", "ZTS", "NOW", "MO",
        "SCHW", "CI", "EOG", "BDX", "CB", "SO", "DUK", "TMUS", "FI", "SLB",
        "PGR", "AON", "BSX", "EQIX", "ITW", "FISV", "CL", "CME", "HUM", "PNC",
        "PYPL", "MU", "MCK", "NOC", "USB", "NSC", "ICE", "APD", "WM", "SHW",
        "KLAC", "SNPS", "CCI", "CDNS", "TGT", "FCX", "EMR", "F", "GM", "ADP",
        "MAR", "PXD", "PSX", "MPC", "VLO", "AEP", "AIG", "OXY", "D", "TFC",
        "EXC", "KMB", "CTVA", "ROP", "DOW", "ORLY", "AZO", "MCO", "WELL", "PCAR",
        "GD", "GIS", "HES", "CMG", "DG", "MCHP", "SRE", "A", "APH", "MSI",
        "HSY", "NEM", "FDX", "CARR", "KMI", "TEL", "JCI", "O", "AJG", "TRV",
        "AFL", "HLT", "PSA", "KDP", "ALL", "PRU", "SYY", "PAYX", "YUM", "NUE",
        "CTAS", "MSCI", "DXCM", "BK", "ECL", "STZ", "CMI", "SPG", "PH", "IQV",
        "IDXX", "ADM", "AMP", "WMB", "ED", "MTD", "OTIS", "DLR", "CTSH", "MNST",
        "RSG", "KR", "GPN", "FAST", "OKE", "EW", "EL", "DVN", "AME", "EA"
    ]


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    print(f"Retrieved {len(tickers)} S&P 500 tickers")
    print(f"First 10: {tickers[:10]}")

# Core imports for Google Agent Development Kit and AI functionality
from google.adk.agents import Agent
from google.genai import types
from google.adk.planners import BuiltInPlanner

# Standard library imports for HTTP requests and utilities
import requests
from typing import Optional
import functools
import asyncio
import os
from datetime import datetime

# Google ADK tools and services
from google.adk.tools import agent_tool
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Web framework imports for Flask application
from flask import Flask, render_template, request, jsonify
from asgiref.wsgi import WsgiToAsgi

# Initialize planner with AI thinking capabilities for financial reasoning
thinking_planner = BuiltInPlanner(
    thinking_config=types.ThinkingConfig(
        include_thoughts=False,      # Disable intermediate reasoning display
        thinking_budget=-1           # Unlimited token budget for planning
    )
)
class fmp():
    """Financial Modeling Prep API wrapper with built-in retry logic and logging"""
    def __init__(self, api_key: str):
        """Initialize FMP API client with authentication key"""
        self.api_key = api_key
    
    def __getattribute__(self, name):
        """Automatic logging wrapper for all API method calls"""
        attr = object.__getattribute__(self, name)
        # Add logging to all public callable methods except core attributes
        if callable(attr) and not name.startswith('_') and name not in ['api_key', 'make_req']:
            def wrapper(*args, **kwargs):
                # Log API call start with first argument (usually symbol/query)
                print(f"🔍 FMP API Call: {name}() - Arguments: {args[0] if args else 'None'}")
                result = attr(*args, **kwargs)
                print(f"✅ FMP API Call: {name}() - Completed")
                return result

            # Preserve original method metadata for proper function introspection
            wrapper.__name__ = name
            wrapper.__doc__ = getattr(attr, '__doc__', None)
            wrapper = functools.wraps(attr)(wrapper)

            return wrapper
        return attr
    
    def make_req(self, url: str):
        """Execute HTTP request with automatic retry logic and error handling"""
        import time
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                # Construct authenticated URL with proper query parameter separator
                separator = "&" if "?" in url else "?"
                req = requests.get(url + separator + "apikey=" + self.api_key, timeout=30)

                if req.status_code == 200:
                    return req.json()
                elif req.status_code == 429:  # Handle rate limiting with exponential backoff
                    print(f"⚠️ Rate limited, waiting {retry_delay * (attempt + 1)} seconds...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                elif req.status_code >= 500:  # Retry on server errors
                    print(f"⚠️ Server error {req.status_code}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"❌ API Error {req.status_code}: {req.text}")
                    return {"error": f"API Error {req.status_code}"}
            except requests.exceptions.Timeout:
                print(f"⚠️ Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: {str(e)}")
                return {"error": f"Request failed: {str(e)}"}

        return {"error": "Max retries exceeded"}

    def search_general(self, query: str, limit: int = 50):
        """General search for companies, ETFs, and other securities"""
        url = f"https://financialmodelingprep.com/api/v3/search?query={query}&limit={limit}"
        return self.make_req(url)
    
    def search_ticker(self, query: str, limit: int = 50, exchange: str = "NYSE"):
        """Search for companies by ticker symbol with optional exchange filter"""
        url = f"https://financialmodelingprep.com/api/v3/search-ticker?query={query}&limit={limit}"
        if exchange:
            url += f"&exchange={exchange}"
        return self.make_req(url)
    
    def search_name(self, query: str, limit: int = 50, exchange: Optional[str] = None):
        """Search for companies by company name"""
        url = f"https://financialmodelingprep.com/api/v3/search-name?query={query}&limit={limit}"
        if exchange:
            url += f"&exchange={exchange}"
        return self.make_req(url)
    
    def search_cik_name(self, query: str, limit: int = 50):
        """Search for companies by CIK (Central Index Key) name"""
        url = f"https://financialmodelingprep.com/api/v3/cik-search/{query}?limit={limit}"
        return self.make_req(url)
    
    def search_cik(self, cik: str):
        """Get company information by CIK (Central Index Key)"""
        url = f"https://financialmodelingprep.com/api/v3/cik/{cik}"
        return self.make_req(url)
    
    def search_cusip(self, cusip: str):
        """Search for companies by CUSIP (Committee on Uniform Securities Identification Procedures)"""
        url = f"https://financialmodelingprep.com/api/v3/cusip/{cusip}"
        return self.make_req(url)
    
    def search_isin(self, isin: str):
        """Search for companies by ISIN (International Securities Identification Number)"""
        url = f"https://financialmodelingprep.com/api/v3/isin/{isin}"
        return self.make_req(url)
    
    def get_company_profile(self, symbol: str):
        """Get detailed company profile information"""
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
        return self.make_req(url)
    
    def search_stock_screener(self, market_cap_more_than: Optional[int] = None, market_cap_lower_than: Optional[int] = None,
                            price_more_than: Optional[float] = None, price_lower_than: Optional[float] = None,
                            beta_more_than: Optional[float] = None, beta_lower_than: Optional[float] = None,
                            volume_more_than: Optional[int] = None, volume_lower_than: Optional[int] = None,
                            dividend_more_than: Optional[float] = None, dividend_lower_than: Optional[float] = None,
                            is_etf: Optional[bool] = None, is_actively_trading: Optional[bool] = None,
                            sector: Optional[str] = None, industry: Optional[str] = None, country: Optional[str] = None,
                            exchange: Optional[str] = None, limit: int = 50):
        """Advanced stock screener with comprehensive filtering capabilities for investment research"""
        url = f"https://financialmodelingprep.com/api/v3/stock-screener?limit={limit}"

        # Build query string with all provided filter parameters
        if market_cap_more_than is not None:
            url += f"&marketCapMoreThan={market_cap_more_than}"
        if market_cap_lower_than is not None:
            url += f"&marketCapLowerThan={market_cap_lower_than}"
        if price_more_than is not None:
            url += f"&priceMoreThan={price_more_than}"
        if price_lower_than is not None:
            url += f"&priceLowerThan={price_lower_than}"
        if beta_more_than is not None:
            url += f"&betaMoreThan={beta_more_than}"
        if beta_lower_than is not None:
            url += f"&betaLowerThan={beta_lower_than}"
        if volume_more_than is not None:
            url += f"&volumeMoreThan={volume_more_than}"
        if volume_lower_than is not None:
            url += f"&volumeLowerThan={volume_lower_than}"
        if dividend_more_than is not None:
            url += f"&dividendMoreThan={dividend_more_than}"
        if dividend_lower_than is not None:
            url += f"&dividendLowerThan={dividend_lower_than}"
        if is_etf is not None:
            url += f"&isEtf={str(is_etf).lower()}"
        if is_actively_trading is not None:
            url += f"&isActivelyTrading={str(is_actively_trading).lower()}"
        if sector:
            url += f"&sector={sector}"
        if industry:
            url += f"&industry={industry}"
        if country:
            url += f"&country={country}"
        if exchange:
            url += f"&exchange={exchange}"

        return self.make_req(url)
    
    # ===== STOCK LISTS AND MARKET INDICES =====
    # Methods for retrieving various stock lists, ETFs, and market index constituents
    
    def get_all_symbols(self):
        """Get all available symbols (stocks, ETFs, etc.)"""
        url = "https://financialmodelingprep.com/api/v3/stock/list"
        return self.make_req(url)
    
    def get_etf_list(self):
        """Get list of all available ETFs"""
        url = "https://financialmodelingprep.com/api/v3/etf/list"
        return self.make_req(url)
    
    def get_tradable_symbols(self):
        """Get all tradable symbols"""
        url = "https://financialmodelingprep.com/api/v3/available-traded/list"
        return self.make_req(url)
    
    def get_sp500_constituents(self):
        """Get current S&P 500 constituents"""
        url = "https://financialmodelingprep.com/api/v3/sp500_constituent"
        return self.make_req(url)
    
    def get_historical_sp500_constituents(self):
        """Get historical S&P 500 constituents with dates"""
        url = "https://financialmodelingprep.com/api/v3/historical/sp500_constituent"
        return self.make_req(url)
    
    def get_nasdaq_constituents(self):
        """Get current NASDAQ constituents"""
        url = "https://financialmodelingprep.com/api/v3/nasdaq_constituent"
        return self.make_req(url)
    
    def get_historical_nasdaq_constituents(self):
        """Get historical NASDAQ constituents with dates"""
        url = "https://financialmodelingprep.com/api/v3/historical/nasdaq_constituent"
        return self.make_req(url)
    
    def get_dowjones_constituents(self):
        """Get current Dow Jones constituents"""
        url = "https://financialmodelingprep.com/api/v3/dowjones_constituent"
        return self.make_req(url)
    
    def get_historical_dowjones_constituents(self):
        """Get historical Dow Jones constituents with dates"""
        url = "https://financialmodelingprep.com/api/v3/historical/dowjones_constituent"
        return self.make_req(url)
    
    def get_symbols_by_exchange(self, exchange: Optional[str] = None):
        """Get symbols traded on specific exchange or all exchanges"""
        if exchange:
            url = f"https://financialmodelingprep.com/api/v3/symbol/{exchange}"
        else:
            url = "https://financialmodelingprep.com/api/v3/symbol"
        return self.make_req(url)
    
    def get_symbol_changes(self):
        """Get recent symbol changes (ticker changes, delistings, etc.)"""
        url = "https://financialmodelingprep.com/api/v4/symbol_change"
        return self.make_req(url)
    
    def get_all_exchanges(self):
        """Get list of all available exchanges"""
        url = "https://financialmodelingprep.com/api/v3/exchanges-list"
        return self.make_req(url)
    
    def get_all_countries(self):
        """Get list of all countries where stocks are traded"""
        url = "https://financialmodelingprep.com/api/v3/get-all-countries"
        return self.make_req(url)
    
    def get_delisted_companies(self, page: int = 0):
        """Get list of delisted companies"""
        url = f"https://financialmodelingprep.com/api/v3/delisted-companies?page={page}"
        return self.make_req(url)
    
    def get_commitment_of_traders_list(self):
        """Get list of symbols available for Commitment of Traders Report"""
        url = "https://financialmodelingprep.com/api/v4/commitment_of_traders_report/list"
        return self.make_req(url)
    
    def get_standard_industrial_classification(self):
        """Get Standard Industrial Classification (SIC) codes"""
        url = "https://financialmodelingprep.com/api/v4/standard_industrial_classification"
        return self.make_req(url)
    
    def get_sic_list(self):
        """Get all available SIC codes"""
        url = "https://financialmodelingprep.com/api/v4/standard_industrial_classification/list"
        return self.make_req(url)
    
    # ===== COMPANY FUNDAMENTALS AND INFORMATION =====
    # Methods for retrieving detailed company information, executive data, and corporate governance
    
    def get_executive_compensation(self, symbol: str):
        """Get executive compensation information for a company"""
        url = f"https://financialmodelingprep.com/api/v4/governance/executive_compensation?symbol={symbol}"
        return self.make_req(url)
    
    def get_compensation_benchmark(self, year: int = 2023):
        """Compare executive compensation across companies for a specific year"""
        url = f"https://financialmodelingprep.com/api/v4/executive-compensation-benchmark?year={year}"
        return self.make_req(url)
    
    def get_company_notes(self, symbol: str):
        """Get company notes from financial statements"""
        url = f"https://financialmodelingprep.com/api/v4/company-notes?symbol={symbol}"
        return self.make_req(url)
    
    def get_historical_employee_count(self, symbol: str):
        """Get historical employee count data for a company"""
        url = f"https://financialmodelingprep.com/api/v4/historical/employee_count?symbol={symbol}"
        return self.make_req(url)
    
    def get_employee_count(self, symbol: str):
        """Get current employee count for a company"""
        url = f"https://financialmodelingprep.com/api/v4/employee_count?symbol={symbol}"
        return self.make_req(url)
    
    def get_stock_grade(self, symbol: str):
        """Get stock grade/rating from professional investors"""
        url = f"https://financialmodelingprep.com/api/v3/grade/{symbol}"
        return self.make_req(url)
    
    def get_key_executives(self, symbol: str):
        """Get key executives information for a company"""
        url = f"https://financialmodelingprep.com/api/v3/key-executives/{symbol}"
        return self.make_req(url)
    
    def get_company_core_information(self, symbol: str):
        """Get core company information (CIK, exchange, address, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/company-core-information?symbol={symbol}"
        return self.make_req(url)
    
    def get_market_cap(self, symbol: str):
        """Get current market capitalization for a company"""
        url = f"https://financialmodelingprep.com/api/v3/market-capitalization/{symbol}"
        return self.make_req(url)
    
    def get_historical_market_cap(self, symbol: str, limit: int = 50):
        """Get historical market capitalization data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{symbol}?limit={limit}"
        return self.make_req(url)
    
    def get_analyst_recommendations(self, symbol: str):
        """Get analyst recommendations for a company"""
        url = f"https://financialmodelingprep.com/api/v3/analyst-recommendation/{symbol}"
        return self.make_req(url)
    
    def get_historical_share_float(self, symbol: str):
        """Get historical share float data for a company"""
        url = f"https://financialmodelingprep.com/api/v4/historical/shares_float?symbol={symbol}"
        return self.make_req(url)
    
    def get_shares_float(self, symbol: str):
        """Get current shares float for a company"""
        url = f"https://financialmodelingprep.com/api/v4/shares_float?symbol={symbol}"
        return self.make_req(url)
    
    def get_revenue_product_segmentation(self, symbol: str, period: str = "annual"):
        """Get revenue breakdown by product category"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-product-segmentation?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_revenue_geographic_segmentation(self, symbol: str, period: str = "annual"):
        """Get revenue breakdown by geographic region"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-geographic-segmentation?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_company_outlook(self, symbol: str):
        """Get comprehensive company outlook including profile, metrics, ratios, and insider trades"""
        url = f"https://financialmodelingprep.com/api/v4/company-outlook?symbol={symbol}"
        return self.make_req(url)
    
    def get_stock_peers(self, symbol: str):
        """Get peer companies for comparison"""
        url = f"https://financialmodelingprep.com/api/v4/stock_peers?symbol={symbol}"
        return self.make_req(url)
    
    def get_company_rating(self, symbol: str):
        """Get company financial rating"""
        url = f"https://financialmodelingprep.com/api/v3/rating/{symbol}"
        return self.make_req(url)
    
    def get_historical_rating(self, symbol: str, limit: int = 50):
        """Get historical company ratings"""
        url = f"https://financialmodelingprep.com/api/v3/historical-rating/{symbol}?limit={limit}"
        return self.make_req(url)
    
    def get_company_financial_score(self, symbol: str):
        """Get comprehensive financial score for a company"""
        url = f"https://financialmodelingprep.com/api/v4/score?symbol={symbol}"
        return self.make_req(url)
    
    # ===== REAL-TIME QUOTES AND PRICING =====
    # Methods for retrieving current market prices and basic quote information
    
    def get_quote(self, symbol: str):
        """Get real-time quote with bid/ask prices, volume, and last trade price"""
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        return self.make_req(url)
    
    def get_quote_short(self, symbol: str):
        """Get simple quote with price, change, and volume"""
        url = f"https://financialmodelingprep.com/api/v3/quote-short/{symbol}"
        return self.make_req(url)
    
    def get_quotes_by_exchange(self, exchange: Optional[str] = None):
        """Get all real-time quotes for an exchange or all exchanges"""
        if exchange:
            url = f"https://financialmodelingprep.com/api/v3/quotes/{exchange}"
        else:
            url = "https://financialmodelingprep.com/api/v3/quotes"
        return self.make_req(url)
    
    def get_quote_order(self, symbol: str):
        """Get simplified view of stock quote including current price, volume, and last trade"""
        url = f"https://financialmodelingprep.com/api/v3/quote-order/{symbol}"
        return self.make_req(url)
    
    def get_otc_quote(self, symbol: str):
        """Get over-the-counter (OTC) stock quote with bid/ask prices and volume"""
        url = f"https://financialmodelingprep.com/api/v3/otc/real-time-price/{symbol}"
        return self.make_req(url)
    
    def get_stock_price_change(self, symbol: str):
        """Get stock price change over different time periods"""
        url = f"https://financialmodelingprep.com/api/v3/stock-price-change/{symbol}"
        return self.make_req(url)
    
    def get_aftermarket_trade(self, symbol: str):
        """Get aftermarket trading information"""
        url = f"https://financialmodelingprep.com/api/v4/pre-post-market-trade/{symbol}"
        return self.make_req(url)
    
    def get_batch_quote(self, symbols: str):
        """Get quotes for multiple stocks at once (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbols}"
        return self.make_req(url)
    
    def get_batch_trade(self, symbols: str):
        """Get trades for multiple stocks at once (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v4/batch-trade-quote?symbols={symbols}"
        return self.make_req(url)
    
    def get_fail_to_deliver(self, symbol: str):
        """Get fail to deliver data for a symbol"""
        url = f"https://financialmodelingprep.com/api/v4/fail_to_deliver?symbol={symbol}"
        return self.make_req(url)
    
    # ===== ALTERNATIVE ASSET QUOTES =====
    # Forex, cryptocurrency, and commodity price data
    
    def get_forex_quote(self, pair: Optional[str] = None):
        """Get forex currency exchange rates (specific pair or all pairs)"""
        if pair:
            url = f"https://financialmodelingprep.com/api/v3/fx/{pair}"
        else:
            url = "https://financialmodelingprep.com/api/v3/fx"
        return self.make_req(url)
    
    def get_all_forex_quotes(self):
        """Get all forex currency exchange rates"""
        url = "https://financialmodelingprep.com/api/v3/quotes/forex"
        return self.make_req(url)
    
    def get_crypto_quote(self, symbol: Optional[str] = None):
        """Get cryptocurrency quote (specific crypto or all cryptos)"""
        if symbol:
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        else:
            url = "https://financialmodelingprep.com/api/v3/quotes/crypto"
        return self.make_req(url)
    
    def get_all_crypto_quotes(self):
        """Get all cryptocurrency quotes"""
        url = "https://financialmodelingprep.com/api/v3/quotes/crypto"
        return self.make_req(url)
    
    def get_commodities_quote(self, symbol: Optional[str] = None):
        """Get commodities quote (specific commodity or all commodities)"""
        if symbol:
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        else:
            url = "https://financialmodelingprep.com/api/v3/quotes/commodity"
        return self.make_req(url)
    
    def get_all_commodities_quotes(self):
        """Get all commodities quotes"""
        url = "https://financialmodelingprep.com/api/v3/quotes/commodity"
        return self.make_req(url)
    
    # ===== REAL-TIME MARKET DATA =====
    # Live streaming price data and market updates
    
    def get_real_time_price(self, symbol: str):
        """Get real-time price for a stock"""
        url = f"https://financialmodelingprep.com/api/v3/stock/real-time-price/{symbol}"
        return self.make_req(url)
    
    def get_real_time_full_price(self, symbol: str):
        """Get comprehensive real-time price data"""
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        return self.make_req(url)
    
    def get_market_hours(self, exchange: Optional[str] = None):
        """Get market hours for exchanges"""
        if exchange:
            url = f"https://financialmodelingprep.com/api/v3/market-hours?exchange={exchange}"
        else:
            url = "https://financialmodelingprep.com/api/v3/market-hours"
        return self.make_req(url)
    
    # ===== FINANCIAL STATEMENTS AND REPORTING =====
    # Income statements, balance sheets, cash flow statements, and related financial data
    
    def get_income_statement(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get income statement data (annual or quarter)"""
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_balance_sheet(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get balance sheet data (annual or quarter)"""
        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_cash_flow_statement(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get cash flow statement data (annual or quarter)"""
        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_income_statement_as_reported(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get income statement as reported by the company"""
        url = f"https://financialmodelingprep.com/api/v3/income-statement-as-reported/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_balance_sheet_as_reported(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get balance sheet as reported by the company"""
        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement-as-reported/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_cash_flow_as_reported(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get cash flow statement as reported by the company"""
        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement-as-reported/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_full_financial_statement_as_reported(self, symbol: str, period: str = "annual"):
        """Get complete financial statement as reported"""
        url = f"https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/{symbol}?period={period}"
        return self.make_req(url)
    
    def get_financial_statement_list(self):
        """Get list of available financial statement symbols"""
        url = f"https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists"
        return self.make_req(url)
    
    def get_financial_reports_dates(self, symbol: str):
        """Get available dates for financial reports"""
        url = f"https://financialmodelingprep.com/api/v4/financial-reports-dates?symbol={symbol}"
        return self.make_req(url)
    
    def get_financial_reports_json(self, symbol: str, year: int, period: str):
        """Get financial reports in JSON format"""
        url = f"https://financialmodelingprep.com/api/v4/financial-reports-json?symbol={symbol}&year={year}&period={period}"
        return self.make_req(url)
    
    def get_shares_float_all(self, symbol: str):
        """Get shares float data"""
        url = f"https://financialmodelingprep.com/api/v4/shares_float/all?symbol={symbol}"
        return self.make_req(url)
    
    # ===== GROWTH AND PERFORMANCE METRICS =====
    # Financial growth rates and key performance indicators
    
    def get_income_statement_growth(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get income statement growth rates"""
        url = f"https://financialmodelingprep.com/api/v3/income-statement-growth/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_balance_sheet_growth(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get balance sheet growth rates"""
        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement-growth/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_cash_flow_growth(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get cash flow statement growth rates"""
        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement-growth/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_financial_growth(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get comprehensive financial growth metrics"""
        url = f"https://financialmodelingprep.com/api/v3/financial-growth/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    # ===== FINANCIAL RATIOS AND ANALYSIS =====
    # Comprehensive financial ratios for fundamental analysis
    
    def get_financial_ratios(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get financial ratios (liquidity, profitability, leverage, etc.)"""
        url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_key_metrics(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get key financial metrics"""
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_key_metrics_ttm(self, symbol: str):
        """Get trailing twelve months (TTM) key metrics"""
        url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
        return self.make_req(url)
    
    def get_ratios_ttm(self, symbol: str):
        """Get trailing twelve months (TTM) financial ratios"""
        url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}"
        return self.make_req(url)
    
    def get_enterprise_values(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get enterprise value metrics"""
        url = f"https://financialmodelingprep.com/api/v3/enterprise-values/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_owner_earnings(self, symbol: str):
        """Get owner earnings (Buffett's preferred metric)"""
        url = f"https://financialmodelingprep.com/api/v4/owner_earnings?symbol={symbol}"
        return self.make_req(url)
    
    # ===== VALUATION AND MODELING =====
    # DCF models and various valuation methodologies
    
    def get_dcf_value(self, symbol: str):
        """Get discounted cash flow valuation"""
        url = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/{symbol}"
        return self.make_req(url)
    
    def get_historical_dcf(self, symbol: str, limit: int = 50):
        """Get historical DCF values"""
        url = f"https://financialmodelingprep.com/api/v3/historical-discounted-cash-flow-statement/{symbol}?limit={limit}"
        return self.make_req(url)
    
    def get_advanced_dcf(self, symbol: str):
        """Get advanced DCF with detailed assumptions"""
        url = f"https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={symbol}"
        return self.make_req(url)
    
    def get_levered_dcf(self, symbol: str):
        """Get levered (equity) DCF valuation"""
        url = f"https://financialmodelingprep.com/api/v4/advanced_levered_discounted_cash_flow?symbol={symbol}"
        return self.make_req(url)
    
    # ===== EARNINGS DATA AND TRANSCRIPTS =====
    # Earnings reports, call transcripts, and earnings-related information
    
    # Core Earnings Data
    def get_earnings_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get earnings calendar showing upcoming & past earnings announcements for publicly traded companies"""
        url = "https://financialmodelingprep.com/api/v3/earning_calendar"
        
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "?" + "&".join(params)
        
        return self.make_req(url)
    
    def get_historical_earnings_calendar(self, symbol: str, limit: int = 50):
        """Get historical & upcoming earnings announcements for a specific company"""
        url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}?limit={limit}"
        return self.make_req(url)
    
    def get_earnings_confirmed(self, from_date: str, to_date: str, limit: int = 100):
        """Get earnings announcements that have already been confirmed"""
        url = f"https://financialmodelingprep.com/api/v4/earning-calendar-confirmed?from={from_date}&to={to_date}&limit={limit}"
        return self.make_req(url)
    
    def get_earnings_surprises(self, symbol: str):
        """Get earnings surprises (positive or negative) for a company"""
        url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{symbol}"
        return self.make_req(url)
    
    def get_earnings_expectations(self, symbol: str, period: str = "quarter"):
        """Get earnings expectations and consensus estimates"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-expectations?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_earnings_beats_misses(self, from_date: str, to_date: str, beat_type: str = "beat"):
        """Get earnings beats or misses in date range (beat_type: beat, miss, meet)"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-beats-misses?from={from_date}&to={to_date}&type={beat_type}"
        return self.make_req(url)

    def get_earnings_revision_history(self, symbol: str, period: str = "quarter", limit: int = 50):
        """Get earnings estimate revision history"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-revision-history?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_earnings_growth_rates(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get earnings growth rates over time"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-growth-rates?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_earnings_consistency_score(self, symbol: str):
        """Get earnings consistency and predictability score"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-consistency?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_vs_estimates_analysis(self, symbol: str, quarters: int = 8):
        """Get detailed analysis of earnings vs estimates performance"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-vs-estimates?symbol={symbol}&quarters={quarters}"
        return self.make_req(url)
    
    def get_upcoming_earnings_this_week(self):
        """Get earnings announcements for the current week"""
        url = "https://financialmodelingprep.com/api/v4/earnings-this-week"
        return self.make_req(url)
    
    def get_earnings_after_hours(self, date: Optional[str] = None):
        """Get after-hours earnings announcements"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/earnings-after-hours?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v4/earnings-after-hours"
        return self.make_req(url)
    
    def get_earnings_pre_market(self, date: Optional[str] = None):
        """Get pre-market earnings announcements"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/earnings-pre-market?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v4/earnings-pre-market"
        return self.make_req(url)
    
    def get_sector_earnings_calendar(self, sector: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get earnings calendar for a specific sector"""
        url = f"https://financialmodelingprep.com/api/v4/sector-earnings-calendar?sector={sector}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_earnings_impact_analysis(self, symbol: str, quarters: int = 4):
        """Get analysis of earnings impact on stock price"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-impact-analysis?symbol={symbol}&quarters={quarters}"
        return self.make_req(url)
    
    def get_earnings_volatility_analysis(self, symbol: str):
        """Get earnings volatility and standard deviation analysis"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-volatility?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_momentum(self, symbol: str):
        """Get earnings momentum indicators (improving/deteriorating trends)"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-momentum?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_quality_indicators(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get earnings quality indicators and red flags"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-quality-indicators?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_earnings_seasonality(self, symbol: str):
        """Get earnings seasonality patterns and trends"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-seasonality?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_peer_comparison(self, symbol: str):
        """Compare earnings metrics against industry peers"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-peer-comparison?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_guidance_accuracy(self, symbol: str, periods: int = 8):
        """Get accuracy of management earnings guidance"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-guidance-accuracy?symbol={symbol}&periods={periods}"
        return self.make_req(url)
    
    def get_earnings_whisper_numbers(self, symbol: str):
        """Get unofficial earnings whisper numbers and expectations"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-whisper-numbers?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_options_activity(self, symbol: str, days_before: int = 5, days_after: int = 5):
        """Get options activity around earnings announcements"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-options-activity?symbol={symbol}&daysBefore={days_before}&daysAfter={days_after}"
        return self.make_req(url)
    
    def get_earnings_estimate_changes(self, symbol: str, days: int = 30):
        """Get recent changes in earnings estimates"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-estimate-changes?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_earnings_conference_call_schedule(self, from_date: str, to_date: str):
        """Get earnings conference call schedule"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-conference-call-schedule?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_earnings_rss_feed(self, page: int = 0):
        """Get RSS feed of latest earnings announcements"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-rss-feed?page={page}"
        return self.make_req(url)
    
    def get_earnings_alerts(self, symbols: str, threshold: float = 5.0):
        """Get alerts for significant earnings surprises (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-alerts?symbols={symbols}&threshold={threshold}"
        return self.make_req(url)
    
    def get_institutional_earnings_estimates(self, symbol: str):
        """Get institutional investor earnings estimates"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-earnings-estimates?symbol={symbol}"
        return self.make_req(url)
    
    def get_sell_side_earnings_estimates(self, symbol: str):
        """Get sell-side analyst earnings estimates breakdown"""
        url = f"https://financialmodelingprep.com/api/v4/sell-side-earnings-estimates?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_estimate_accuracy_by_firm(self, firm: str, timeframe: str = "1y"):
        """Get earnings estimate accuracy for specific analyst firm"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-estimate-accuracy?firm={firm}&timeframe={timeframe}"
        return self.make_req(url)
    
    def get_earnings_calendar_export(self, from_date: str, to_date: str, format: str = "csv"):
        """Export earnings calendar data (csv, excel, json)"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-calendar-export?from={from_date}&to={to_date}&format={format}"
        return self.make_req(url)
    
    def get_international_earnings_calendar(self, country: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get earnings calendar for specific country"""
        url = f"https://financialmodelingprep.com/api/v4/international-earnings-calendar?country={country}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    # Earnings Transcripts Section
    def get_earnings_call_transcript(self, symbol: str, year: int, quarter: int):
        """Get earnings call transcript for specific year and quarter"""
        url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}?year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_earnings_call_transcript_dates(self, symbol: str):
        """Get available earnings call transcript dates for a company"""
        url = f"https://financialmodelingprep.com/api/v4/earning_call_transcript_date?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_call_transcript_summary(self, symbol: str, year: int, quarter: int):
        """Get AI-generated summary of earnings call transcript"""
        url = f"https://financialmodelingprep.com/api/v4/earning_call_transcript_summary/{symbol}?year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_historical_earnings_transcripts(self, symbol: str, limit: int = 20):
        """Get historical earnings call transcripts for a company"""
        url = f"https://financialmodelingprep.com/api/v4/historical-earnings-transcripts?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_latest_earnings_transcript(self, symbol: str):
        """Get the latest earnings call transcript for a company"""
        url = f"https://financialmodelingprep.com/api/v4/latest-earnings-transcript?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_transcript_by_date(self, symbol: str, date: str):
        """Get earnings transcript by specific date (YYYY-MM-DD format)"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-transcript-by-date?symbol={symbol}&date={date}"
        return self.make_req(url)
    
    def get_bulk_earnings_transcripts(self, symbols: str, year: Optional[int] = None, quarter: Optional[int] = None):
        """Get earnings transcripts for multiple companies (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v4/bulk-earnings-transcripts?symbols={symbols}"
        if year:
            url += f"&year={year}"
        if quarter:
            url += f"&quarter={quarter}"
        return self.make_req(url)
    
    def search_earnings_transcripts(self, query: str, symbol: Optional[str] = None, limit: int = 50):
        """Search earnings transcripts by keyword or phrase"""
        url = f"https://financialmodelingprep.com/api/v4/search-earnings-transcripts?query={query}&limit={limit}"
        if symbol:
            url += f"&symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_transcript_summary(self, symbol: str, year: int, quarter: int):
        """Get AI-generated summary of earnings call transcript"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-transcript-summary?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_earnings_transcript_sentiment(self, symbol: str, year: int, quarter: int):
        """Get sentiment analysis of earnings call transcript"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-transcript-sentiment?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_earnings_key_topics(self, symbol: str, year: int, quarter: int):
        """Extract key topics and themes from earnings call transcript"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-key-topics?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_management_guidance(self, symbol: str, year: int, quarter: int):
        """Extract management guidance and forward-looking statements"""
        url = f"https://financialmodelingprep.com/api/v4/management-guidance?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_earnings_qa_analysis(self, symbol: str, year: int, quarter: int):
        """Get analysis of Q&A portion of earnings call"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-qa-analysis?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_conference_call_metadata(self, symbol: str, year: int, quarter: int):
        """Get metadata about conference call (duration, participants, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/conference-call-metadata?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_upcoming_earnings_calls(self, days: int = 30):
        """Get upcoming earnings calls in the next specified days"""
        url = f"https://financialmodelingprep.com/api/v4/upcoming-earnings-calls?days={days}"
        return self.make_req(url)
    
    def get_earnings_call_calendar(self, from_date: str, to_date: str):
        """Get earnings call calendar for a date range"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-call-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    # ===== IPO CALENDAR SECTION =====
    
    def get_ipo_calendar_confirmed(self, from_date: str, to_date: str, limit: int = 100):
        """Get IPO calendar with confirmed IPOs scheduled for the specified date range"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-calendar-confirmed?from={from_date}&to={to_date}&limit={limit}"
        return self.make_req(url)
    
    def get_ipo_calendar_prospectus(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 100):
        """Get IPO prospectus links for companies going public"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-calendar-prospectus?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ipo_calendar_by_symbol(self, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get IPO calendar filtered by company symbol"""
        url = "https://financialmodelingprep.com/api/v3/ipo_calendar?"
        
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_upcoming_ipos(self, days_ahead: int = 30, limit: int = 50):
        """Get upcoming IPOs in the next specified days"""
        url = f"https://financialmodelingprep.com/api/v4/upcoming-ipos?daysAhead={days_ahead}&limit={limit}"
        return self.make_req(url)
    
    def get_recent_ipos(self, days_back: int = 30, limit: int = 50):
        """Get recent IPOs from the last specified days"""
        url = f"https://financialmodelingprep.com/api/v4/recent-ipos?daysBack={days_back}&limit={limit}"
        return self.make_req(url)
    
    def get_ipo_calendar_this_week(self):
        """Get IPOs scheduled for this week"""
        url = "https://financialmodelingprep.com/api/v4/ipo-calendar-this-week"
        return self.make_req(url)
    
    def get_ipo_calendar_this_month(self, year: Optional[int] = None, month: Optional[int] = None):
        """Get IPOs scheduled for specific month or current month"""
        url = "https://financialmodelingprep.com/api/v4/ipo-calendar-this-month"
        
        if year and month:
            url += f"?year={year}&month={month}"
        
        return self.make_req(url)
    
    def get_ipo_calendar_by_quarter(self, year: int, quarter: int):
        """Get IPO calendar for specific quarter"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-calendar-quarterly?year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_ipo_details(self, symbol: str):
        """Get detailed IPO information for a specific company"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-details?symbol={symbol}"
        return self.make_req(url)
    
    def get_ipo_pricing_range(self, symbol: str):
        """Get IPO pricing range and valuation estimates"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-pricing-range?symbol={symbol}"
        return self.make_req(url)
    
    def get_ipo_allocation_data(self, symbol: str):
        """Get IPO share allocation and distribution data"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-allocation-data?symbol={symbol}"
        return self.make_req(url)
    
    def get_ipo_underwriters(self, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get IPO underwriter information"""
        url = "https://financialmodelingprep.com/api/v4/ipo-underwriters?"
        
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_ipo_performance_tracking(self, symbol: str, days_after: int = 30):
        """Get IPO performance tracking after listing"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-performance-tracking?symbol={symbol}&daysAfter={days_after}"
        return self.make_req(url)
    
    def get_first_day_ipo_performance(self, from_date: str, to_date: str):
        """Get first-day IPO performance analysis"""
        url = f"https://financialmodelingprep.com/api/v4/first-day-ipo-performance?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_ipo_sector_analysis(self, sector: str, year: Optional[int] = None):
        """Get IPO analysis by sector"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-sector-analysis?sector={sector}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_ipo_market_trends(self, period: str = "quarterly", years: int = 3):
        """Get IPO market trends and statistics (period: monthly, quarterly, yearly)"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-market-trends?period={period}&years={years}"
        return self.make_req(url)
    
    def get_withdrawn_ipos(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get list of withdrawn or postponed IPOs"""
        url = f"https://financialmodelingprep.com/api/v4/withdrawn-ipos?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_delayed_ipos(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get list of delayed IPOs"""
        url = f"https://financialmodelingprep.com/api/v4/delayed-ipos?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ipo_filings(self, symbol: Optional[str] = None, filing_type: str = "S-1", limit: int = 50):
        """Get IPO-related SEC filings (S-1, S-1/A, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-filings?filingType={filing_type}&limit={limit}"
        
        if symbol:
            url += f"&symbol={symbol}"
        
        return self.make_req(url)
    
    def get_ipo_roadshow_schedule(self, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get IPO roadshow and investor presentation schedules"""
        url = "https://financialmodelingprep.com/api/v4/ipo-roadshow-schedule?"
        
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_ipo_lockup_expiration(self, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get IPO lockup period expiration dates"""
        url = "https://financialmodelingprep.com/api/v4/ipo-lockup-expiration?"
        
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_ipo_insider_selling(self, symbol: str, days_after_ipo: int = 180):
        """Get insider selling activity after IPO"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-insider-selling?symbol={symbol}&daysAfterIpo={days_after_ipo}"
        return self.make_req(url)
    
    def get_ipo_institutional_interest(self, symbol: str):
        """Get institutional investor interest in IPO"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-institutional-interest?symbol={symbol}"
        return self.make_req(url)
    
    def get_spac_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None, status: Optional[str] = None):
        """Get SPAC (Special Purpose Acquisition Company) calendar"""
        url = "https://financialmodelingprep.com/api/v4/spac-calendar?"
        
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if status:  # announced, completed, searching
            params.append(f"status={status}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_spac_mergers(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get SPAC merger announcements and completions"""
        url = f"https://financialmodelingprep.com/api/v4/spac-mergers?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_direct_listings(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get direct listing calendar (companies going public without traditional IPO)"""
        url = f"https://financialmodelingprep.com/api/v4/direct-listings?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ipo_calendar_export(self, from_date: str, to_date: str, format: str = "csv"):
        """Export IPO calendar data (csv, excel, json)"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-calendar-export?from={from_date}&to={to_date}&format={format}"
        return self.make_req(url)
    
    def get_international_ipo_calendar(self, country: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get IPO calendar for specific country"""
        url = f"https://financialmodelingprep.com/api/v4/international-ipo-calendar?country={country}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ipo_pricing_updates(self, symbol: Optional[str] = None, days: int = 7):
        """Get recent IPO pricing updates and changes"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-pricing-updates?days={days}"
        
        if symbol:
            url += f"&symbol={symbol}"
        
        return self.make_req(url)
    
    def get_ipo_subscription_data(self, symbol: str):
        """Get IPO subscription and oversubscription data"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-subscription-data?symbol={symbol}"
        return self.make_req(url)
    
    def get_ipo_aftermarket_performance(self, symbol: str, periods: str = "1d,1w,1m,3m,6m,1y"):
        """Get IPO aftermarket performance across multiple periods"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-aftermarket-performance?symbol={symbol}&periods={periods}"
        return self.make_req(url)
    
    def get_ipo_analyst_coverage_initiation(self, symbol: str, days_after_ipo: int = 90):
        """Get analyst coverage initiation after IPO"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-analyst-coverage?symbol={symbol}&daysAfterIpo={days_after_ipo}"
        return self.make_req(url)
    
    def get_ipo_volume_analysis(self, symbol: str, days_after_ipo: int = 30):
        """Get trading volume analysis after IPO"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-volume-analysis?symbol={symbol}&daysAfterIpo={days_after_ipo}"
        return self.make_req(url)
    
    def get_ipo_volatility_analysis(self, symbol: str, days_after_ipo: int = 30):
        """Get volatility analysis after IPO"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-volatility-analysis?symbol={symbol}&daysAfterIpo={days_after_ipo}"
        return self.make_req(url)
    
    def get_ipo_peer_comparison(self, symbol: str):
        """Compare IPO performance with industry peers"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-peer-comparison?symbol={symbol}"
        return self.make_req(url)
    
    def get_ipo_sentiment_analysis(self, symbol: str, days_around: int = 7):
        """Get sentiment analysis around IPO date"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-sentiment-analysis?symbol={symbol}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_ipo_media_coverage(self, symbol: str, days_around: int = 14):
        """Get media coverage analysis around IPO"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-media-coverage?symbol={symbol}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_ipo_social_sentiment(self, symbol: str, days_around: int = 7):
        """Get social media sentiment around IPO"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-social-sentiment?symbol={symbol}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_ipo_success_metrics(self, year: Optional[int] = None, sector: Optional[str] = None):
        """Get IPO success metrics and statistics"""
        url = "https://financialmodelingprep.com/api/v4/ipo-success-metrics?"
        
        params = []
        if year:
            params.append(f"year={year}")
        if sector:
            params.append(f"sector={sector}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_ipo_market_conditions(self, date: Optional[str] = None):
        """Get IPO market conditions and favorability index"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/ipo-market-conditions?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v4/ipo-market-conditions"
        return self.make_req(url)
    
    def get_ipo_calendar_alerts(self, symbols: str):
        """Get alerts for IPO calendar updates (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-calendar-alerts?symbols={symbols}"
        return self.make_req(url)
    
    def get_ipo_rss_feed(self, page: int = 0):
        """Get RSS feed of latest IPO announcements and updates"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-rss-feed?page={page}"
        return self.make_req(url)
    
    def get_earnings_transcript_keywords(self, symbol: str, year: int, quarter: int):
        """Extract most mentioned keywords from earnings transcript"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-transcript-keywords?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_transcript_quality_score(self, symbol: str, year: int, quarter: int):
        """Get quality score for transcript (completeness, clarity, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/transcript-quality-score?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def compare_earnings_transcripts(self, symbol: str, periods: str):
        """Compare earnings transcripts across multiple periods (format: 2023Q1,2023Q2,2023Q3)"""
        url = f"https://financialmodelingprep.com/api/v4/compare-earnings-transcripts?symbol={symbol}&periods={periods}"
        return self.make_req(url)
    
    def get_transcript_tone_analysis(self, symbol: str, year: int, quarter: int):
        """Analyze tone of management and analyst questions"""
        url = f"https://financialmodelingprep.com/api/v4/transcript-tone-analysis?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_earnings_transcript_highlights(self, symbol: str, year: int, quarter: int):
        """Get key highlights and important quotes from transcript"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-transcript-highlights?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_analyst_questions_analysis(self, symbol: str, year: int, quarter: int):
        """Analyze analyst questions and focus areas"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-questions-analysis?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_management_response_analysis(self, symbol: str, year: int, quarter: int):
        """Analyze management responses and communication style"""
        url = f"https://financialmodelingprep.com/api/v4/management-response-analysis?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_transcript_word_cloud(self, symbol: str, year: int, quarter: int):
        """Generate word cloud data from earnings transcript"""
        url = f"https://financialmodelingprep.com/api/v4/transcript-word-cloud?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_earnings_surprise_context(self, symbol: str, year: int, quarter: int):
        """Get transcript context around earnings surprises"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-surprise-context?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_forward_looking_statements(self, symbol: str, year: int, quarter: int):
        """Extract forward-looking statements and predictions"""
        url = f"https://financialmodelingprep.com/api/v4/forward-looking-statements?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_risk_factors_mentioned(self, symbol: str, year: int, quarter: int):
        """Extract risk factors and concerns mentioned in transcript"""
        url = f"https://financialmodelingprep.com/api/v4/risk-factors-mentioned?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_transcript_readability_score(self, symbol: str, year: int, quarter: int):
        """Get readability and complexity score of transcript"""
        url = f"https://financialmodelingprep.com/api/v4/transcript-readability-score?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_sector_transcript_analysis(self, sector: str, year: int, quarter: int):
        """Analyze common themes across sector earnings transcripts"""
        url = f"https://financialmodelingprep.com/api/v4/sector-transcript-analysis?sector={sector}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_transcript_benchmark_analysis(self, symbol: str, year: int, quarter: int):
        """Benchmark transcript metrics against industry peers"""
        url = f"https://financialmodelingprep.com/api/v4/transcript-benchmark-analysis?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_earnings_call_participants(self, symbol: str, year: int, quarter: int):
        """Get list of participants in earnings call"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-call-participants?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_transcript_translation(self, symbol: str, year: int, quarter: int, language: str = "es"):
        """Get translated version of earnings transcript (language codes: es, fr, de, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/transcript-translation?symbol={symbol}&year={year}&quarter={quarter}&language={language}"
        return self.make_req(url)
    
    def get_earnings_transcript_audio_link(self, symbol: str, year: int, quarter: int):
        """Get audio recording link for earnings call if available"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-call-audio?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_transcript_compliance_analysis(self, symbol: str, year: int, quarter: int):
        """Analyze transcript for regulatory compliance and disclosure requirements"""
        url = f"https://financialmodelingprep.com/api/v4/transcript-compliance-analysis?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_earnings_guidance_tracking(self, symbol: str, limit: int = 8):
        """Track guidance provided across multiple earnings calls"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-guidance-tracking?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_transcript_insider_mentions(self, symbol: str, year: int, quarter: int):
        """Extract mentions of insider trading, buybacks, dividends"""
        url = f"https://financialmodelingprep.com/api/v4/transcript-insider-mentions?symbol={symbol}&year={year}&quarter={quarter}"
        return self.make_req(url)
    
    # ===== RSS FEEDS =====
    
    def get_rss_feed(self, page: int = 0):
        """Get RSS feed of financial statements updates"""
        url = f"https://financialmodelingprep.com/api/v3/rss_feed?page={page}"
        return self.make_req(url)
    
    def get_rss_feed_v4(self, page: int = 0):
        """Get RSS feed V4 with more details"""
        url = f"https://financialmodelingprep.com/api/v4/rss_feed?page={page}"
        return self.make_req(url)
    
    # ===== STATEMENTS ANALYSIS SECTION =====
    
    def get_financial_statements_analysis(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get comprehensive financial statements analysis with key insights"""
        url = f"https://financialmodelingprep.com/api/v4/financial-statements-analysis?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_ratio_analysis(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get detailed ratio analysis (liquidity, profitability, efficiency, leverage)"""
        url = f"https://financialmodelingprep.com/api/v4/ratio-analysis?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_dupont_analysis(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get DuPont analysis breaking down ROE components"""
        url = f"https://financialmodelingprep.com/api/v4/dupont-analysis?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_financial_trends(self, symbol: str, period: str = "annual", years: int = 5):
        """Get financial trends analysis over specified years"""
        url = f"https://financialmodelingprep.com/api/v4/financial-trends?symbol={symbol}&period={period}&years={years}"
        return self.make_req(url)
    
    def get_cash_flow_analysis(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get detailed cash flow analysis and quality metrics"""
        url = f"https://financialmodelingprep.com/api/v4/cash-flow-analysis?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_working_capital_analysis(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get working capital and liquidity analysis"""
        url = f"https://financialmodelingprep.com/api/v4/working-capital-analysis?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_debt_analysis(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get debt structure and leverage analysis"""
        url = f"https://financialmodelingprep.com/api/v4/debt-analysis?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_profitability_analysis(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get profitability margins and efficiency analysis"""
        url = f"https://financialmodelingprep.com/api/v4/profitability-analysis?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_efficiency_ratios(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get asset turnover and efficiency ratios"""
        url = f"https://financialmodelingprep.com/api/v4/efficiency-ratios?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_valuation_ratios(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get valuation ratios (P/E, P/B, EV/EBITDA, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/valuation-ratios?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_quality_score(self, symbol: str):
        """Get financial quality score based on statement analysis"""
        url = f"https://financialmodelingprep.com/api/v4/financial-quality-score?symbol={symbol}"
        return self.make_req(url)
    
    def get_altman_z_score(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get Altman Z-Score for bankruptcy risk assessment"""
        url = f"https://financialmodelingprep.com/api/v4/altman-z-score?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_piotroski_score(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get Piotroski F-Score for fundamental strength analysis"""
        url = f"https://financialmodelingprep.com/api/v4/piotroski-score?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_graham_number(self, symbol: str):
        """Get Benjamin Graham's intrinsic value calculation"""
        url = f"https://financialmodelingprep.com/api/v4/graham-number?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_quality_analysis(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get earnings quality and manipulation risk analysis"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-quality?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_comparative_analysis(self, symbols: str, period: str = "annual"):
        """Compare financial statements across multiple companies (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v4/comparative-analysis?symbols={symbols}&period={period}"
        return self.make_req(url)
    
    def get_industry_comparison(self, symbol: str, period: str = "annual"):
        """Compare company metrics against industry averages"""
        url = f"https://financialmodelingprep.com/api/v4/industry-comparison?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_peer_comparison(self, symbol: str, period: str = "annual"):
        """Compare company against its direct competitors"""
        url = f"https://financialmodelingprep.com/api/v4/peer-comparison?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_margin_analysis(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get detailed margin analysis (gross, operating, net margins)"""
        url = f"https://financialmodelingprep.com/api/v4/margin-analysis?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_return_analysis(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get return on assets, equity, and invested capital analysis"""
        url = f"https://financialmodelingprep.com/api/v4/return-analysis?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_growth_consistency(self, symbol: str, period: str = "annual", years: int = 5):
        """Analyze growth consistency and sustainability"""
        url = f"https://financialmodelingprep.com/api/v4/growth-consistency?symbol={symbol}&period={period}&years={years}"
        return self.make_req(url)
    
    def get_statement_alerts(self, symbol: str, period: str = "annual"):
        """Get alerts for potential red flags in financial statements"""
        url = f"https://financialmodelingprep.com/api/v4/statement-alerts?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_analyst_metrics_consensus(self, symbol: str):
        """Get consensus analyst metrics and estimates"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-metrics-consensus?symbol={symbol}"
        return self.make_req(url)
    
    def get_financial_health_grade(self, symbol: str):
        """Get overall financial health grade (A-F scale)"""
        url = f"https://financialmodelingprep.com/api/v4/financial-health-grade?symbol={symbol}"
        return self.make_req(url)
    
    # ===== VALUATION SECTION =====
    
    def get_valuation_dcf(self, symbol: str):
        """Get discounted cash flow (DCF) valuation"""
        url = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/{symbol}"
        return self.make_req(url)
    
    def get_historical_dcf(self, symbol: str, limit: int = 50):
        """Get historical DCF valuations over time"""
        url = f"https://financialmodelingprep.com/api/v3/historical-discounted-cash-flow-statement/{symbol}?limit={limit}"
        return self.make_req(url)
    
    def get_dcf_with_daily_values(self, symbol: str):
        """Get DCF with daily stock price comparison"""
        url = f"https://financialmodelingprep.com/api/v3/historical-daily-dcf/{symbol}"
        return self.make_req(url)
    
    def get_advanced_dcf_model(self, symbol: str):
        """Get advanced DCF model with detailed assumptions"""
        url = f"https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={symbol}"
        return self.make_req(url)
    
    def get_levered_dcf_model(self, symbol: str):
        """Get levered (equity) DCF valuation model"""
        url = f"https://financialmodelingprep.com/api/v4/advanced_levered_discounted_cash_flow?symbol={symbol}"
        return self.make_req(url)
    
    # ==================== FUNDRAISING ENDPOINTS ====================
    
    def get_crowdfunding_rss_feed(self, page: int = 0):
        """Get RSS feed of crowdfunding campaigns, updated in real time"""
        url = f"https://financialmodelingprep.com/api/v4/crowdfunding-offerings-rss-feed?page={page}"
        return self.make_req(url)
    
    def search_crowdfunding_offerings(self, name: str):
        """Search for crowdfunding campaigns by company name, campaign name, or platform"""
        url = f"https://financialmodelingprep.com/api/v4/crowdfunding-offerings/search?name={name}"
        return self.make_req(url)
    
    def get_crowdfunding_by_cik(self, cik: str):
        """Get all crowdfunding campaigns launched by a particular company"""
        url = f"https://financialmodelingprep.com/api/v4/crowdfunding-offerings?cik={cik}"
        return self.make_req(url)
    
    def get_equity_offering_rss_feed(self, page: int = 0):
        """Get RSS feed of equity offering announcements, updated in real time"""
        url = f"https://financialmodelingprep.com/api/v4/fundraising-rss-feed?page={page}"
        return self.make_req(url)
    
    def search_equity_offerings(self, name: str):
        """Search for equity offerings by company name, offering name, or exchange"""
        url = f"https://financialmodelingprep.com/api/v4/fundraising/search?name={name}"
        return self.make_req(url)
    
    def get_equity_offering_by_cik(self, cik: str):
        """Get all equity offerings announced by a particular company"""
        url = f"https://financialmodelingprep.com/api/v4/fundraising?cik={cik}"
        return self.make_req(url)

    # ==================== ECONOMIC DATA ENDPOINTS ====================
    
    def get_treasury_rates(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get US Treasury rates (3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y)"""
        url = "https://financialmodelingprep.com/api/v4/treasury"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_economic_indicator(self, name: str = "GDP"):
        """Get economic indicators (GDP, CPI, unemployment, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/economic?name={name}"
        return self.make_req(url)
    
    def get_federal_funds_rate(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get Federal Funds Rate historical data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=federalFundsRate"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_inflation_rate(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get inflation rate (CPI) data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=CPI"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_unemployment_rate(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get unemployment rate data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=unemploymentRate"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_gdp_growth(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get GDP growth rate data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=realGDP"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_consumer_price_index(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get Consumer Price Index (CPI) data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=CPI"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_producer_price_index(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get Producer Price Index (PPI) data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=PPI"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_consumer_sentiment(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get University of Michigan Consumer Sentiment Index"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=consumerSentiment"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_retail_sales(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get retail sales data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=retailSales"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_industrial_production(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get industrial production index data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=industrialProduction"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_housing_starts(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get housing starts data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=housingStarts"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_nonfarm_payrolls(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get nonfarm payrolls employment data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=nonfarmPayrolls"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_initial_claims(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get initial jobless claims data"""
        url = "https://financialmodelingprep.com/api/v4/economic?name=initialClaims"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_economic_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get economic events calendar"""
        url = "https://financialmodelingprep.com/api/v3/economic_calendar"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_economic_indicators_list(self):
        """Get list of all available economic indicators"""
        url = "https://financialmodelingprep.com/api/v4/economic-indicators"
        return self.make_req(url)
    
    def get_economic_indicator_historical(self, indicator: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get historical data for any economic indicator"""
        url = f"https://financialmodelingprep.com/api/v4/economic?name={indicator}"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)

    # ==================== COMMODITIES ENDPOINTS ====================
    
    def get_commodities_list(self):
        """Get list of all available commodities"""
        url = "https://financialmodelingprep.com/api/v3/symbol/available-commodities"
        return self.make_req(url)
    
    def get_commodity_price(self, symbol: str):
        """Get real-time price for specific commodity (e.g., GCUSD, CLUSD, SIUSD)"""
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        return self.make_req(url)
    
    def get_gold_price(self):
        """Get current gold price (GCUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/GCUSD"
        return self.make_req(url)
    
    def get_silver_price(self):
        """Get current silver price (SIUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/SIUSD"
        return self.make_req(url)
    
    def get_oil_price(self):
        """Get current crude oil price (CLUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/CLUSD"
        return self.make_req(url)
    
    def get_copper_price(self):
        """Get current copper price (HGUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/HGUSD"
        return self.make_req(url)
    
    def get_natural_gas_price(self):
        """Get current natural gas price (NGUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/NGUSD"
        return self.make_req(url)
    
    def get_platinum_price(self):
        """Get current platinum price (PLUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/PLUSD"
        return self.make_req(url)
    
    def get_palladium_price(self):
        """Get current palladium price (PAUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/PAUSD"
        return self.make_req(url)
    
    def get_corn_price(self):
        """Get current corn futures price (CNUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/CNUSD"
        return self.make_req(url)
    
    def get_wheat_price(self):
        """Get current wheat futures price (WUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/WUSD"
        return self.make_req(url)
    
    def get_soybeans_price(self):
        """Get current soybeans futures price (SUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/SUSD"
        return self.make_req(url)
    
    def get_coffee_price(self):
        """Get current coffee futures price (KCUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/KCUSD"
        return self.make_req(url)
    
    def get_sugar_price(self):
        """Get current sugar futures price (SBUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/SBUSD"
        return self.make_req(url)
    
    def get_cotton_price(self):
        """Get current cotton futures price (CTUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/CTUSD"
        return self.make_req(url)
    
    def get_historical_commodity_prices(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get historical commodity price data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_intraday_commodity_prices(self, symbol: str, interval: str = "1min", from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get intraday commodity price data (1min, 5min, 15min, 30min, 1hour)"""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{symbol}"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_commodities_by_sector(self, sector: str = "energy"):
        """Get commodities by sector (energy, metals, agriculture)"""
        url = f"https://financialmodelingprep.com/api/v4/commodities-sector?sector={sector}"
        return self.make_req(url)
    
    def get_energy_commodities(self):
        """Get all energy-related commodities (oil, gas, heating oil, etc.)"""
        url = "https://financialmodelingprep.com/api/v4/energy-commodities"
        return self.make_req(url)
    
    def get_precious_metals(self):
        """Get all precious metals (gold, silver, platinum, palladium)"""
        url = "https://financialmodelingprep.com/api/v4/precious-metals"
        return self.make_req(url)
    
    def get_base_metals(self):
        """Get all base metals (copper, aluminum, zinc, nickel, etc.)"""
        url = "https://financialmodelingprep.com/api/v4/base-metals"
        return self.make_req(url)
    
    def get_agricultural_commodities(self):
        """Get all agricultural commodities (corn, wheat, soybeans, etc.)"""
        url = "https://financialmodelingprep.com/api/v4/agricultural-commodities"
        return self.make_req(url)
    
    def get_livestock_commodities(self):
        """Get livestock commodities (cattle, hogs, etc.)"""
        url = "https://financialmodelingprep.com/api/v4/livestock-commodities"
        return self.make_req(url)
    
    def get_soft_commodities(self):
        """Get soft commodities (coffee, sugar, cotton, cocoa, etc.)"""
        url = "https://financialmodelingprep.com/api/v4/soft-commodities"
        return self.make_req(url)
    
    def get_commodity_futures(self, symbol: str):
        """Get futures contracts for a commodity"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-futures?symbol={symbol}"
        return self.make_req(url)
    
    def get_commodity_futures_calendar(self, from_date: str, to_date: str):
        """Get commodity futures expiration calendar"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-futures-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_commodity_seasonality(self, symbol: str):
        """Get seasonal patterns for a commodity"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-seasonality?symbol={symbol}"
        return self.make_req(url)
    
    def get_commodity_volatility(self, symbol: str, period: int = 30):
        """Get commodity price volatility analysis"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-volatility?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_commodity_correlations(self, symbol1: str, symbol2: str, period: int = 252):
        """Get correlation between two commodities"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-correlation?symbol1={symbol1}&symbol2={symbol2}&period={period}"
        return self.make_req(url)
    
    def get_commodity_inventory_reports(self, commodity: str = "oil"):
        """Get inventory reports for commodities (oil, natural gas, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-inventory?commodity={commodity}"
        return self.make_req(url)
    
    def get_oil_inventory_report(self):
        """Get weekly crude oil inventory report"""
        url = "https://financialmodelingprep.com/api/v4/crude-oil-inventory"
        return self.make_req(url)
    
    def get_natural_gas_inventory_report(self):
        """Get weekly natural gas storage report"""
        url = "https://financialmodelingprep.com/api/v4/natural-gas-inventory"
        return self.make_req(url)
    
    def get_commodity_production_data(self, commodity: str, country: Optional[str] = None):
        """Get production data for commodities by country"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-production?commodity={commodity}"
        if country:
            url += f"&country={country}"
        return self.make_req(url)
    
    def get_commodity_consumption_data(self, commodity: str, country: Optional[str] = None):
        """Get consumption data for commodities by country"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-consumption?commodity={commodity}"
        if country:
            url += f"&country={country}"
        return self.make_req(url)
    
    def get_commodity_supply_demand(self, commodity: str):
        """Get supply and demand fundamentals for commodity"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-supply-demand?commodity={commodity}"
        return self.make_req(url)
    
    def get_commodity_news(self, commodity: Optional[str] = None, limit: int = 50):
        """Get news related to commodities"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-news?limit={limit}"
        if commodity:
            url += f"&commodity={commodity}"
        return self.make_req(url)
    
    def get_commodity_alerts(self, symbols: str, price_change_threshold: float = 5.0):
        """Set up commodity price alerts"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-alerts?symbols={symbols}&threshold={price_change_threshold}"
        return self.make_req(url)
    
    def get_commodity_screener(self, sector: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None,
                              price_change_24h: Optional[float] = None, volume_min: Optional[int] = None, limit: int = 50):
        """Screen commodities based on various criteria"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-screener?limit={limit}"
        if sector:
            url += f"&sector={sector}"
        if min_price:
            url += f"&minPrice={min_price}"
        if max_price:
            url += f"&maxPrice={max_price}"
        if price_change_24h:
            url += f"&priceChange24h={price_change_24h}"
        if volume_min:
            url += f"&volumeMin={volume_min}"
        return self.make_req(url)
    
    def get_commodity_calendar(self, from_date: str, to_date: str):
        """Get commodity market calendar (reports, events, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_commodity_etfs(self, commodity: Optional[str] = None):
        """Get ETFs that track specific commodities"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-etfs"
        if commodity:
            url += f"?commodity={commodity}"
        return self.make_req(url)
    
    def get_commodity_technical_analysis(self, symbol: str, timeframe: str = "daily"):
        """Get technical analysis for commodity"""
        url = f"https://financialmodelingprep.com/api/v4/commodity-technical-analysis?symbol={symbol}&timeframe={timeframe}"
        return self.make_req(url)

    # ==================== FOREX ENDPOINTS ====================
    
    def get_forex_pairs_list(self):
        """Get list of all available forex currency pairs"""
        url = "https://financialmodelingprep.com/api/v3/symbol/available-forex-currency-pairs"
        return self.make_req(url)
    
    def get_forex_pair_quote(self, pair: str):
        """Get real-time quote for specific forex pair (e.g., EURUSD, GBPUSD)"""
        url = f"https://financialmodelingprep.com/api/v3/fx/{pair}"
        return self.make_req(url)
    
    def get_all_forex_pairs(self):
        """Get real-time quotes for all forex pairs"""
        url = "https://financialmodelingprep.com/api/v3/fx"
        return self.make_req(url)
    
    def get_major_forex_pairs(self):
        """Get major forex pairs (EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD)"""
        url = "https://financialmodelingprep.com/api/v4/forex-majors"
        return self.make_req(url)
    
    def get_minor_forex_pairs(self):
        """Get minor forex pairs (cross-currency pairs not involving USD)"""
        url = "https://financialmodelingprep.com/api/v4/forex-minors"
        return self.make_req(url)
    
    def get_exotic_forex_pairs(self):
        """Get exotic forex pairs (major currency vs emerging market currency)"""
        url = "https://financialmodelingprep.com/api/v4/forex-exotics"
        return self.make_req(url)
    
    def get_historical_forex_data(self, pair: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get historical forex data for a currency pair"""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{pair}"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_intraday_forex_data(self, pair: str, interval: str = "1min", from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get intraday forex data (1min, 5min, 15min, 30min, 1hour, 4hour)"""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{pair}"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_currency_exchange_rates(self, base_currency: str = "USD"):
        """Get exchange rates for a base currency against all other currencies"""
        url = f"https://financialmodelingprep.com/api/v3/fx/{base_currency}"
        return self.make_req(url)
    
    def get_forex_volatility(self, pair: str, period: int = 30):
        """Get forex pair volatility analysis"""
        url = f"https://financialmodelingprep.com/api/v4/forex-volatility?pair={pair}&period={period}"
        return self.make_req(url)
    
    def get_forex_correlation(self, pair1: str, pair2: str, period: int = 252):
        """Get correlation between two forex pairs"""
        url = f"https://financialmodelingprep.com/api/v4/forex-correlation?pair1={pair1}&pair2={pair2}&period={period}"
        return self.make_req(url)
    
    def get_currency_strength(self, currency: str = "USD"):
        """Get currency strength index"""
        url = f"https://financialmodelingprep.com/api/v4/currency-strength?currency={currency}"
        return self.make_req(url)
    
    def get_forex_market_hours(self):
        """Get forex market trading hours by session"""
        url = "https://financialmodelingprep.com/api/v4/forex-market-hours"
        return self.make_req(url)
    
    def get_forex_economic_calendar(self, from_date: str, to_date: str, currency: Optional[str] = None):
        """Get economic calendar events affecting forex markets"""
        url = f"https://financialmodelingprep.com/api/v4/forex-economic-calendar?from={from_date}&to={to_date}"
        if currency:
            url += f"&currency={currency}"
        return self.make_req(url)
    
    def get_central_bank_decisions(self, country: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get central bank interest rate decisions"""
        url = "https://financialmodelingprep.com/api/v4/central-bank-decisions"
        params = []
        if country:
            params.append(f"country={country}")
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_forex_sentiment(self, pair: Optional[str] = None):
        """Get forex market sentiment and positioning data"""
        url = "https://financialmodelingprep.com/api/v4/forex-sentiment"
        if pair:
            url += f"?pair={pair}"
        return self.make_req(url)
    
    def get_commitment_of_traders_forex(self, currency: str):
        """Get Commitment of Traders report for forex/currencies"""
        url = f"https://financialmodelingprep.com/api/v4/cot-forex?currency={currency}"
        return self.make_req(url)
    
    def get_forex_carry_trade_opportunities(self, min_rate_differential: float = 2.0):
        """Get carry trade opportunities based on interest rate differentials"""
        url = f"https://financialmodelingprep.com/api/v4/forex-carry-trade?minRateDiff={min_rate_differential}"
        return self.make_req(url)
    
    def get_currency_forwards(self, pair: str, maturities: str = "1M,3M,6M,1Y"):
        """Get currency forward rates"""
        url = f"https://financialmodelingprep.com/api/v4/currency-forwards?pair={pair}&maturities={maturities}"
        return self.make_req(url)
    
    def get_currency_options_data(self, pair: str, expiration: Optional[str] = None):
        """Get currency options data and implied volatility"""
        url = f"https://financialmodelingprep.com/api/v4/currency-options?pair={pair}"
        if expiration:
            url += f"&expiration={expiration}"
        return self.make_req(url)
    
    def get_forex_swap_rates(self, pair: str):
        """Get forex swap rates for overnight positions"""
        url = f"https://financialmodelingprep.com/api/v4/forex-swap-rates?pair={pair}"
        return self.make_req(url)
    
    def get_forex_spreads(self, pair: Optional[str] = None):
        """Get bid-ask spreads for forex pairs"""
        url = "https://financialmodelingprep.com/api/v4/forex-spreads"
        if pair:
            url += f"?pair={pair}"
        return self.make_req(url)
    
    def get_forex_liquidity_metrics(self, pair: str):
        """Get liquidity metrics for forex pair"""
        url = f"https://financialmodelingprep.com/api/v4/forex-liquidity?pair={pair}"
        return self.make_req(url)
    
    def get_forex_session_analysis(self, pair: str, session: str = "all"):
        """Get forex pair performance by trading session (London, New York, Tokyo, Sydney)"""
        url = f"https://financialmodelingprep.com/api/v4/forex-session-analysis?pair={pair}&session={session}"
        return self.make_req(url)
    
    def get_forex_technical_analysis(self, pair: str, timeframe: str = "daily"):
        """Get technical analysis for forex pair"""
        url = f"https://financialmodelingprep.com/api/v4/forex-technical-analysis?pair={pair}&timeframe={timeframe}"
        return self.make_req(url)
    
    def get_forex_pivot_points(self, pair: str, pivot_type: str = "standard"):
        """Get pivot points for forex pair (standard, fibonacci, woodie, camarilla)"""
        url = f"https://financialmodelingprep.com/api/v4/forex-pivot-points?pair={pair}&type={pivot_type}"
        return self.make_req(url)
    
    def get_forex_support_resistance(self, pair: str, period: int = 50):
        """Get support and resistance levels for forex pair"""
        url = f"https://financialmodelingprep.com/api/v4/forex-support-resistance?pair={pair}&period={period}"
        return self.make_req(url)
    
    def get_forex_screener(self, volatility_min: Optional[float] = None, volatility_max: Optional[float] = None,
                          volume_min: Optional[int] = None, price_change_24h: Optional[float] = None,
                          spread_max: Optional[float] = None, limit: int = 50):
        """Screen forex pairs based on various criteria"""
        url = f"https://financialmodelingprep.com/api/v4/forex-screener?limit={limit}"
        if volatility_min:
            url += f"&volatilityMin={volatility_min}"
        if volatility_max:
            url += f"&volatilityMax={volatility_max}"
        if volume_min:
            url += f"&volumeMin={volume_min}"
        if price_change_24h:
            url += f"&priceChange24h={price_change_24h}"
        if spread_max:
            url += f"&spreadMax={spread_max}"
        return self.make_req(url)
    
    def get_forex_alerts(self, pairs: str, price_change_threshold: float = 1.0):
        """Set up forex price alerts"""
        url = f"https://financialmodelingprep.com/api/v4/forex-alerts?pairs={pairs}&threshold={price_change_threshold}"
        return self.make_req(url)
    
    def get_forex_heatmap(self, base_currencies: str = "USD,EUR,GBP,JPY,CHF,CAD,AUD,NZD"):
        """Get forex market heatmap showing currency strength"""
        url = f"https://financialmodelingprep.com/api/v4/forex-heatmap?currencies={base_currencies}"
        return self.make_req(url)
    
    def get_currency_exposure_stocks(self, currency: str, exposure_threshold: float = 20.0):
        """Get stocks with significant exposure to specific currency"""
        url = f"https://financialmodelingprep.com/api/v4/currency-exposure-stocks?currency={currency}&threshold={exposure_threshold}"
        return self.make_req(url)
    
    def get_forex_seasonality(self, pair: str):
        """Get seasonal patterns for forex pair"""
        url = f"https://financialmodelingprep.com/api/v4/forex-seasonality?pair={pair}"
        return self.make_req(url)
    
    def get_forex_market_overview(self):
        """Get overall forex market overview and summary"""
        url = "https://financialmodelingprep.com/api/v4/forex-market-overview"
        return self.make_req(url)
    
    def get_currency_converter(self, from_currency: str, to_currency: str, amount: float = 1.0):
        """Convert amount from one currency to another"""
        url = f"https://financialmodelingprep.com/api/v4/currency-converter?from={from_currency}&to={to_currency}&amount={amount}"
        return self.make_req(url)
    
    def get_historical_currency_converter(self, from_currency: str, to_currency: str, date: str, amount: float = 1.0):
        """Convert amount using historical exchange rates"""
        url = f"https://financialmodelingprep.com/api/v4/historical-currency-converter?from={from_currency}&to={to_currency}&date={date}&amount={amount}"
        return self.make_req(url)

    # ==================== CRYPTOCURRENCY ENDPOINTS ====================
    
    def get_crypto_list(self):
        """Get list of all available cryptocurrencies"""
        url = "https://financialmodelingprep.com/api/v3/symbol/available-cryptocurrencies" 
        return self.make_req(url)
    
    def get_crypto_price(self, symbol: str):
        """Get real-time price for specific cryptocurrency (e.g., BTCUSD, ETHUSD)"""
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        return self.make_req(url)
    
    def get_all_crypto_prices(self):
        """Get real-time prices for all cryptocurrencies"""
        url = "https://financialmodelingprep.com/api/v3/quotes/crypto"
        return self.make_req(url)
    
    def get_bitcoin_price(self):
        """Get current Bitcoin price (BTCUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/BTCUSD"
        return self.make_req(url)
    
    def get_ethereum_price(self):
        """Get current Ethereum price (ETHUSD)"""
        url = "https://financialmodelingprep.com/api/v3/quote/ETHUSD"
        return self.make_req(url)
    
    def get_top_cryptocurrencies(self, limit: int = 100):
        """Get top cryptocurrencies by market cap"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-top?limit={limit}"
        return self.make_req(url)
    
    def get_crypto_market_cap(self, symbol: Optional[str] = None):
        """Get cryptocurrency market capitalization data"""
        if symbol:
            url = f"https://financialmodelingprep.com/api/v4/crypto-market-cap?symbol={symbol}"
        else:
            url = "https://financialmodelingprep.com/api/v4/crypto-market-cap"
        return self.make_req(url)
    
    def get_historical_crypto_data(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get historical cryptocurrency price data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_intraday_crypto_data(self, symbol: str, interval: str = "1min", from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get intraday cryptocurrency data (1min, 5min, 15min, 30min, 1hour, 4hour)"""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{symbol}"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_crypto_volatility(self, symbol: str, period: int = 30):
        """Get cryptocurrency volatility analysis"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-volatility?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_crypto_correlation(self, symbol1: str, symbol2: str, period: int = 252):
        """Get correlation between two cryptocurrencies"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-correlation?symbol1={symbol1}&symbol2={symbol2}&period={period}"
        return self.make_req(url)
    
    def get_crypto_fear_greed_index(self):
        """Get cryptocurrency Fear & Greed Index"""
        url = "https://financialmodelingprep.com/api/v4/crypto-fear-greed-index"
        return self.make_req(url)
    
    def get_crypto_market_dominance(self):
        """Get cryptocurrency market dominance (Bitcoin, Ethereum, etc.)"""
        url = "https://financialmodelingprep.com/api/v4/crypto-market-dominance"
        return self.make_req(url)
    
    def get_crypto_total_market_cap(self):
        """Get total cryptocurrency market capitalization"""
        url = "https://financialmodelingprep.com/api/v4/crypto-total-market-cap"
        return self.make_req(url)
    
    def get_defi_tokens(self, limit: int = 50):
        """Get DeFi (Decentralized Finance) tokens"""
        url = f"https://financialmodelingprep.com/api/v4/defi-tokens?limit={limit}"
        return self.make_req(url)
    
    def get_nft_tokens(self, limit: int = 50):
        """Get NFT (Non-Fungible Token) related cryptocurrencies"""
        url = f"https://financialmodelingprep.com/api/v4/nft-tokens?limit={limit}"
        return self.make_req(url)
    
    def get_stablecoins(self, limit: int = 50):
        """Get stablecoins data"""
        url = f"https://financialmodelingprep.com/api/v4/stablecoins?limit={limit}"
        return self.make_req(url)
    
    def get_crypto_gainers_losers(self, type: str = "gainers", timeframe: str = "24h", limit: int = 50):
        """Get crypto gainers/losers (gainers/losers, 1h/24h/7d)"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-{type}?timeframe={timeframe}&limit={limit}"
        return self.make_req(url)
    
    def get_crypto_most_active(self, limit: int = 50):
        """Get most actively traded cryptocurrencies"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-most-active?limit={limit}"
        return self.make_req(url)
    
    def get_crypto_exchanges(self):
        """Get list of cryptocurrency exchanges"""
        url = "https://financialmodelingprep.com/api/v4/crypto-exchanges"
        return self.make_req(url)
    
    def get_crypto_exchange_data(self, exchange: str, symbol: Optional[str] = None):
        """Get cryptocurrency data from specific exchange"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-exchange-data?exchange={exchange}"
        if symbol:
            url += f"&symbol={symbol}"
        return self.make_req(url)
    
    def get_crypto_trading_pairs(self, exchange: Optional[str] = None):
        """Get available trading pairs by exchange"""
        url = "https://financialmodelingprep.com/api/v4/crypto-trading-pairs"
        if exchange:
            url += f"?exchange={exchange}"
        return self.make_req(url)
    
    def get_crypto_volume_analysis(self, symbol: str, days: int = 30):
        """Get cryptocurrency volume analysis"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-volume-analysis?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_crypto_liquidity_analysis(self, symbol: str):
        """Get cryptocurrency liquidity analysis"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-liquidity?symbol={symbol}"
        return self.make_req(url)
    
    def get_crypto_order_book(self, symbol: str, exchange: Optional[str] = None, depth: int = 100):
        """Get cryptocurrency order book data"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-order-book?symbol={symbol}&depth={depth}"
        if exchange:
            url += f"&exchange={exchange}"
        return self.make_req(url)
    
    def get_crypto_funding_rates(self, symbol: Optional[str] = None):
        """Get cryptocurrency futures funding rates"""
        url = "https://financialmodelingprep.com/api/v4/crypto-funding-rates"
        if symbol:
            url += f"?symbol={symbol}"
        return self.make_req(url)
    
    def get_crypto_open_interest(self, symbol: str):
        """Get cryptocurrency futures open interest"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-open-interest?symbol={symbol}"
        return self.make_req(url)
    
    def get_crypto_derivatives(self, symbol: Optional[str] = None, type: str = "futures"):
        """Get cryptocurrency derivatives data (futures, options, perpetuals)"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-derivatives?type={type}"
        if symbol:
            url += f"&symbol={symbol}"
        return self.make_req(url)
    
    def get_crypto_technical_analysis(self, symbol: str, timeframe: str = "daily"):
        """Get technical analysis for cryptocurrency"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-technical-analysis?symbol={symbol}&timeframe={timeframe}"
        return self.make_req(url)
    
    def get_crypto_support_resistance(self, symbol: str, period: int = 50):
        """Get support and resistance levels for cryptocurrency"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-support-resistance?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_crypto_on_chain_metrics(self, symbol: str, metric: str = "all"):
        """Get on-chain metrics for cryptocurrency (active addresses, transactions, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-on-chain?symbol={symbol}&metric={metric}"
        return self.make_req(url)
    
    def get_bitcoin_network_metrics(self):
        """Get Bitcoin network metrics (hash rate, difficulty, mempool, etc.)"""
        url = "https://financialmodelingprep.com/api/v4/bitcoin-network-metrics"
        return self.make_req(url)
    
    def get_ethereum_network_metrics(self):
        """Get Ethereum network metrics (gas fees, network utilization, etc.)"""
        url = "https://financialmodelingprep.com/api/v4/ethereum-network-metrics"
        return self.make_req(url)
    
    def get_crypto_mining_data(self, coin: str = "BTC"):
        """Get cryptocurrency mining data"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-mining-data?coin={coin}"
        return self.make_req(url)
    
    def get_crypto_staking_data(self, symbol: Optional[str] = None):
        """Get cryptocurrency staking data and yields"""
        url = "https://financialmodelingprep.com/api/v4/crypto-staking-data"
        if symbol:
            url += f"?symbol={symbol}"
        return self.make_req(url)
    
    def get_crypto_yield_farming(self, protocol: Optional[str] = None, limit: int = 50):
        """Get DeFi yield farming opportunities"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-yield-farming?limit={limit}"
        if protocol:
            url += f"&protocol={protocol}"
        return self.make_req(url)
    
    def get_crypto_lending_rates(self, symbol: Optional[str] = None):
        """Get cryptocurrency lending and borrowing rates"""
        url = "https://financialmodelingprep.com/api/v4/crypto-lending-rates"
        if symbol:
            url += f"?symbol={symbol}"
        return self.make_req(url)
    
    def get_crypto_screener(self, min_market_cap: Optional[int] = None, max_market_cap: Optional[int] = None,
                           min_price: Optional[float] = None, max_price: Optional[float] = None,
                           price_change_24h: Optional[float] = None, volume_min: Optional[int] = None, 
                           category: Optional[str] = None, limit: int = 50):
        """Screen cryptocurrencies based on various criteria"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-screener?limit={limit}"
        if min_market_cap:
            url += f"&minMarketCap={min_market_cap}"
        if max_market_cap:
            url += f"&maxMarketCap={max_market_cap}"
        if min_price:
            url += f"&minPrice={min_price}"
        if max_price:
            url += f"&maxPrice={max_price}"
        if price_change_24h:
            url += f"&priceChange24h={price_change_24h}"
        if volume_min:
            url += f"&volumeMin={volume_min}"
        if category:
            url += f"&category={category}"
        return self.make_req(url)
    
    def get_crypto_alerts(self, symbols: str, price_change_threshold: float = 5.0):
        """Set up cryptocurrency price alerts"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-alerts?symbols={symbols}&threshold={price_change_threshold}"
        return self.make_req(url)
    
    def get_crypto_market_overview(self):
        """Get overall cryptocurrency market overview"""
        url = "https://financialmodelingprep.com/api/v4/crypto-market-overview"
        return self.make_req(url)
    
    def get_crypto_seasonality(self, symbol: str):
        """Get seasonal patterns for cryptocurrency"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-seasonality?symbol={symbol}"
        return self.make_req(url)
    
    def get_crypto_social_sentiment(self, symbol: str, platform: str = "all"):
        """Get cryptocurrency social media sentiment (Twitter, Reddit, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-social-sentiment?symbol={symbol}&platform={platform}"
        return self.make_req(url)
    
    def get_crypto_whale_transactions(self, symbol: str, min_amount: float = 1000000.0):
        """Get large cryptocurrency transactions (whale movements)"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-whale-transactions?symbol={symbol}&minAmount={min_amount}"
        return self.make_req(url)
    
    def get_crypto_institutional_flows(self, symbol: Optional[str] = None, days: int = 30):
        """Get institutional cryptocurrency flows and investments"""
        url = f"https://financialmodelingprep.com/api/v4/crypto-institutional-flows?days={days}"
        if symbol:
            url += f"&symbol={symbol}"
        return self.make_req(url)
    
    def get_crypto_etfs(self, symbol: Optional[str] = None):
        """Get cryptocurrency ETFs and trusts"""
        url = "https://financialmodelingprep.com/api/v4/crypto-etfs"
        if symbol:
            url += f"?symbol={symbol}"
        return self.make_req(url)
    
    def get_crypto_treasury_holdings(self, company: Optional[str] = None):
        """Get corporate cryptocurrency treasury holdings"""
        url = "https://financialmodelingprep.com/api/v4/crypto-treasury-holdings"
        if company:
            url += f"?company={company}"
        return self.make_req(url)

    # ==================== INDEX CONSTITUENTS ENDPOINTS ====================
    
    def get_all_index_constituents(self):
        """Get list of all available index constituents"""
        url = "https://financialmodelingprep.com/api/v4/all-index-constituents"
        return self.make_req(url)
    
    def get_sp500_constituents_detailed(self):
        """Get detailed S&P 500 constituents with sector/industry breakdown"""
        url = "https://financialmodelingprep.com/api/v4/sp500-constituents-detailed"
        return self.make_req(url)
    
    def get_nasdaq100_constituents(self):
        """Get NASDAQ-100 constituents"""
        url = "https://financialmodelingprep.com/api/v3/nasdaq100_constituent"
        return self.make_req(url)
    
    def get_historical_nasdaq100_constituents(self):
        """Get historical NASDAQ-100 constituents with changes"""
        url = "https://financialmodelingprep.com/api/v3/historical/nasdaq100_constituent"
        return self.make_req(url)
    
    def get_russell1000_constituents(self):
        """Get Russell 1000 constituents"""
        url = "https://financialmodelingprep.com/api/v4/russell1000-constituents"
        return self.make_req(url)
    
    def get_russell2000_constituents(self):
        """Get Russell 2000 constituents"""
        url = "https://financialmodelingprep.com/api/v4/russell2000-constituents"
        return self.make_req(url)
    
    def get_ftse100_constituents(self):
        """Get FTSE 100 constituents"""
        url = "https://financialmodelingprep.com/api/v4/ftse100-constituents"
        return self.make_req(url)
    
    def get_cac40_constituents(self):
        """Get CAC 40 constituents"""
        url = "https://financialmodelingprep.com/api/v4/cac40-constituents"
        return self.make_req(url)
    
    def get_dax_constituents(self):
        """Get DAX constituents"""
        url = "https://financialmodelingprep.com/api/v4/dax-constituents"
        return self.make_req(url)
    
    def get_nikkei225_constituents(self):
        """Get Nikkei 225 constituents"""
        url = "https://financialmodelingprep.com/api/v4/nikkei225-constituents"
        return self.make_req(url)
    
    def get_asx200_constituents(self):
        """Get ASX 200 constituents"""
        url = "https://financialmodelingprep.com/api/v4/asx200-constituents"
        return self.make_req(url)
    
    def get_tsx_constituents(self):
        """Get TSX (Toronto Stock Exchange) constituents"""
        url = "https://financialmodelingprep.com/api/v4/tsx-constituents"
        return self.make_req(url)
    
    def get_sector_constituents(self, sector: str, index: str = "SP500"):
        """Get constituents by sector for specific index"""
        url = f"https://financialmodelingprep.com/api/v4/sector-constituents?sector={sector}&index={index}"
        return self.make_req(url)
    
    def get_technology_sector_constituents(self, index: str = "SP500"):
        """Get technology sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/technology-constituents?index={index}"
        return self.make_req(url)
    
    def get_healthcare_sector_constituents(self, index: str = "SP500"):
        """Get healthcare sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/healthcare-constituents?index={index}"
        return self.make_req(url)
    
    def get_financial_sector_constituents(self, index: str = "SP500"):
        """Get financial sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/financial-constituents?index={index}"
        return self.make_req(url)
    
    def get_energy_sector_constituents(self, index: str = "SP500"):
        """Get energy sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/energy-constituents?index={index}"
        return self.make_req(url)
    
    def get_consumer_discretionary_constituents(self, index: str = "SP500"):
        """Get consumer discretionary sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/consumer-discretionary-constituents?index={index}"
        return self.make_req(url)
    
    def get_consumer_staples_constituents(self, index: str = "SP500"):
        """Get consumer staples sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/consumer-staples-constituents?index={index}"
        return self.make_req(url)
    
    def get_industrials_sector_constituents(self, index: str = "SP500"):
        """Get industrials sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/industrials-constituents?index={index}"
        return self.make_req(url)
    
    def get_materials_sector_constituents(self, index: str = "SP500"):
        """Get materials sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/materials-constituents?index={index}"
        return self.make_req(url)
    
    def get_real_estate_sector_constituents(self, index: str = "SP500"):
        """Get real estate sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/real-estate-constituents?index={index}"
        return self.make_req(url)
    
    def get_utilities_sector_constituents(self, index: str = "SP500"):
        """Get utilities sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/utilities-constituents?index={index}"
        return self.make_req(url)
    
    def get_communication_services_constituents(self, index: str = "SP500"):
        """Get communication services sector constituents"""
        url = f"https://financialmodelingprep.com/api/v4/communication-services-constituents?index={index}"
        return self.make_req(url)
    
    def get_index_constituent_changes(self, index: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get index constituent additions and deletions"""
        url = f"https://financialmodelingprep.com/api/v4/index-constituent-changes?index={index}"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_constituent_weightings(self, index: str = "SP500"):
        """Get index constituent weightings and allocations"""
        url = f"https://financialmodelingprep.com/api/v4/constituent-weightings?index={index}"
        return self.make_req(url)
    
    def get_largest_constituents(self, index: str = "SP500", limit: int = 50):
        """Get largest constituents by market cap or weighting"""
        url = f"https://financialmodelingprep.com/api/v4/largest-constituents?index={index}&limit={limit}"
        return self.make_req(url)
    
    def get_smallest_constituents(self, index: str = "SP500", limit: int = 50):
        """Get smallest constituents by market cap or weighting"""
        url = f"https://financialmodelingprep.com/api/v4/smallest-constituents?index={index}&limit={limit}"
        return self.make_req(url)
    
    def get_constituent_performance(self, index: str = "SP500", period: str = "1y"):
        """Get constituent performance analysis"""
        url = f"https://financialmodelingprep.com/api/v4/constituent-performance?index={index}&period={period}"
        return self.make_req(url)
    
    def get_best_performing_constituents(self, index: str = "SP500", period: str = "1y", limit: int = 50):
        """Get best performing constituents"""
        url = f"https://financialmodelingprep.com/api/v4/best-performing-constituents?index={index}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_worst_performing_constituents(self, index: str = "SP500", period: str = "1y", limit: int = 50):
        """Get worst performing constituents"""
        url = f"https://financialmodelingprep.com/api/v4/worst-performing-constituents?index={index}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_constituent_volatility_analysis(self, index: str = "SP500", period: int = 252):
        """Get volatility analysis of index constituents"""
        url = f"https://financialmodelingprep.com/api/v4/constituent-volatility?index={index}&period={period}"
        return self.make_req(url)
    
    def get_constituent_correlation_matrix(self, index: str = "SP500", period: int = 252):
        """Get correlation matrix between index constituents"""
        url = f"https://financialmodelingprep.com/api/v4/constituent-correlation-matrix?index={index}&period={period}"
        return self.make_req(url)
    
    def get_index_diversification_metrics(self, index: str = "SP500"):
        """Get index diversification and concentration metrics"""
        url = f"https://financialmodelingprep.com/api/v4/index-diversification?index={index}"
        return self.make_req(url)
    
    def get_sector_allocation_breakdown(self, index: str = "SP500"):
        """Get detailed sector allocation breakdown"""
        url = f"https://financialmodelingprep.com/api/v4/sector-allocation?index={index}"
        return self.make_req(url)
    
    def get_market_cap_distribution(self, index: str = "SP500"):
        """Get market cap distribution of constituents"""
        url = f"https://financialmodelingprep.com/api/v4/market-cap-distribution?index={index}"
        return self.make_req(url)
    
    def get_constituent_earnings_calendar(self, index: str = "SP500", from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get earnings calendar for index constituents"""
        url = f"https://financialmodelingprep.com/api/v4/constituent-earnings-calendar?index={index}"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_constituent_dividend_calendar(self, index: str = "SP500", from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get dividend calendar for index constituents"""
        url = f"https://financialmodelingprep.com/api/v4/constituent-dividend-calendar?index={index}"
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        if params:
            url += "&" + "&".join(params)
        return self.make_req(url)
    
    def get_esg_constituents(self, index: str = "SP500", min_esg_score: Optional[float] = None):
        """Get ESG scores for index constituents"""
        url = f"https://financialmodelingprep.com/api/v4/esg-constituents?index={index}"
        if min_esg_score:
            url += f"&minEsgScore={min_esg_score}"
        return self.make_req(url)
    
    def get_dividend_aristocrat_constituents(self):
        """Get S&P 500 Dividend Aristocrats (25+ years of dividend increases)"""
        url = "https://financialmodelingprep.com/api/v4/dividend-aristocrats"
        return self.make_req(url)
    
    def get_faang_stocks(self):
        """Get FAANG stocks (Facebook/Meta, Apple, Amazon, Netflix, Google)"""
        url = "https://financialmodelingprep.com/api/v4/faang-stocks"
        return self.make_req(url)
    
    def get_magnificent_seven_stocks(self):
        """Get Magnificent Seven stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META)"""
        url = "https://financialmodelingprep.com/api/v4/magnificent-seven"
        return self.make_req(url)
    
    def get_constituent_screener(self, index: str = "SP500", min_market_cap: Optional[int] = None, 
                               max_market_cap: Optional[int] = None, sector: Optional[str] = None, 
                               min_dividend_yield: Optional[float] = None, limit: int = 50):
        """Screen index constituents by various criteria"""
        url = f"https://financialmodelingprep.com/api/v4/constituent-screener?index={index}&limit={limit}"
        if min_market_cap:
            url += f"&minMarketCap={min_market_cap}"
        if max_market_cap:
            url += f"&maxMarketCap={max_market_cap}"
        if sector:
            url += f"&sector={sector}"
        if min_dividend_yield:
            url += f"&minDividendYield={min_dividend_yield}"
        return self.make_req(url)
    
    def get_index_rebalancing_calendar(self, index: str = "SP500"):
        """Get index rebalancing and reconstitution calendar"""
        url = f"https://financialmodelingprep.com/api/v4/index-rebalancing-calendar?index={index}"
        return self.make_req(url)
    
    def get_constituent_news_impact(self, index: str = "SP500", days: int = 7):
        """Get news impact analysis for index constituents"""
        url = f"https://financialmodelingprep.com/api/v4/constituent-news-impact?index={index}&days={days}"
        return self.make_req(url)
    
    def get_constituent_analyst_ratings(self, index: str = "SP500"):
        """Get analyst ratings for index constituents"""
        url = f"https://financialmodelingprep.com/api/v4/constituent-analyst-ratings?index={index}"
        return self.make_req(url)
    
    def get_international_index_constituents(self, country: str, index_name: Optional[str] = None):
        """Get constituents for international indices"""
        url = f"https://financialmodelingprep.com/api/v4/international-index-constituents?country={country}"
        if index_name:
            url += f"&index={index_name}"
        return self.make_req(url)
    
    def get_emerging_markets_constituents(self, region: str = "all"):
        """Get emerging markets index constituents"""
        url = f"https://financialmodelingprep.com/api/v4/emerging-markets-constituents?region={region}"
        return self.make_req(url)
    
    def get_constituent_export(self, index: str = "SP500", format: str = "csv"):
        """Export constituent data (csv, excel, json)"""
        url = f"https://financialmodelingprep.com/api/v4/constituent-export?index={index}&format={format}"
        return self.make_req(url)

    # ==================== REVENUE BY SEGMENT ENDPOINTS ====================
    
    def get_revenue_segmentation_detailed(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get detailed revenue segmentation analysis"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segmentation-detailed?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_by_business_segment(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue breakdown by business segments"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-business-segments?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_by_operating_segment(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue breakdown by operating segments"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-operating-segments?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_by_division(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue breakdown by company divisions"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-divisions?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_by_subsidiary(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue breakdown by subsidiaries"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-subsidiaries?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_by_customer_segment(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue breakdown by customer segments (B2B, B2C, enterprise, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-customer-segments?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_by_channel(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue breakdown by sales channels (online, retail, wholesale, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-channels?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_by_contract_type(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue breakdown by contract types (subscription, one-time, licensing, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-contract-types?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_recurring_vs_non_recurring_revenue(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get breakdown of recurring vs non-recurring revenue"""
        url = f"https://financialmodelingprep.com/api/v4/recurring-revenue-breakdown?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_subscription_revenue_metrics(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get subscription revenue metrics (MRR, ARR, churn, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/subscription-revenue-metrics?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_saas_revenue_metrics(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get SaaS-specific revenue metrics"""
        url = f"https://financialmodelingprep.com/api/v4/saas-revenue-metrics?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_geographic_detailed(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get detailed geographic revenue breakdown by countries/regions"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-geographic-detailed?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_by_continent(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue breakdown by continent"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-continents?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_domestic_vs_international(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get domestic vs international revenue breakdown"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-domestic-international?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_top_markets(self, symbol: str, period: str = "annual", limit: int = 10):
        """Get revenue from top geographic markets"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-top-markets?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_emerging_markets(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue from emerging markets"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-emerging-markets?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_segment_growth_rates(self, symbol: str, segment_type: str = "business", period: str = "annual", limit: int = 50):
        """Get growth rates for revenue segments"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-growth?symbol={symbol}&segmentType={segment_type}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_segment_margins(self, symbol: str, segment_type: str = "business", period: str = "annual", limit: int = 50):
        """Get profit margins by revenue segment"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-margins?symbol={symbol}&segmentType={segment_type}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_segment_trends(self, symbol: str, segment_type: str = "business", years: int = 5):
        """Get revenue segment trends over time"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-trends?symbol={symbol}&segmentType={segment_type}&years={years}"
        return self.make_req(url)
    
    def get_revenue_concentration_analysis(self, symbol: str, period: str = "annual"):
        """Get revenue concentration analysis (how dependent on top segments)"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-concentration-analysis?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_revenue_diversification_score(self, symbol: str, period: str = "annual"):
        """Get revenue diversification score across segments"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-diversification-score?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_revenue_segment_seasonality(self, symbol: str, segment: str, years: int = 3):
        """Get seasonal patterns for specific revenue segment"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-seasonality?symbol={symbol}&segment={segment}&years={years}"
        return self.make_req(url)
    
    def get_revenue_segment_volatility(self, symbol: str, segment_type: str = "business", period: int = 20):
        """Get volatility analysis for revenue segments"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-volatility?symbol={symbol}&segmentType={segment_type}&period={period}"
        return self.make_req(url)
    
    def get_revenue_segment_correlation(self, symbol: str, segment1: str, segment2: str, periods: int = 20):
        """Get correlation between two revenue segments"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-correlation?symbol={symbol}&segment1={segment1}&segment2={segment2}&periods={periods}"
        return self.make_req(url)
    
    def get_revenue_by_industry_vertical(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue breakdown by industry verticals served"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-industry-verticals?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_by_customer_size(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get revenue breakdown by customer size (enterprise, SMB, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-customer-size?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_top_customers(self, symbol: str, period: str = "annual", limit: int = 10):
        """Get revenue from top customers (if disclosed)"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-top-customers?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_revenue_customer_concentration(self, symbol: str, period: str = "annual"):
        """Get customer concentration risk analysis"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-customer-concentration?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_revenue_segment_forecasts(self, symbol: str, segment_type: str = "business", periods_ahead: int = 4):
        """Get revenue forecasts by segment"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-forecasts?symbol={symbol}&segmentType={segment_type}&periodsAhead={periods_ahead}"
        return self.make_req(url)
    
    def get_revenue_segment_guidance(self, symbol: str, segment_type: str = "business"):
        """Get management guidance by revenue segment"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-guidance?symbol={symbol}&segmentType={segment_type}"
        return self.make_req(url)
    
    def get_revenue_segment_peer_comparison(self, symbols: str, segment_type: str = "business", period: str = "annual"):
        """Compare revenue segments across peer companies"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-peer-comparison?symbols={symbols}&segmentType={segment_type}&period={period}"
        return self.make_req(url)
    
    def get_industry_revenue_segment_benchmarks(self, industry: str, segment_type: str = "business", period: str = "annual"):
        """Get industry benchmarks for revenue segments"""
        url = f"https://financialmodelingprep.com/api/v4/industry-revenue-segment-benchmarks?industry={industry}&segmentType={segment_type}&period={period}"
        return self.make_req(url)
    
    def get_revenue_segment_sensitivity_analysis(self, symbol: str, segment: str, scenario: str = "base"):
        """Get revenue segment sensitivity to economic scenarios"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-sensitivity?symbol={symbol}&segment={segment}&scenario={scenario}"
        return self.make_req(url)
    
    def get_revenue_segment_key_drivers(self, symbol: str, segment: str):
        """Get key drivers for specific revenue segment"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-drivers?symbol={symbol}&segment={segment}"
        return self.make_req(url)
    
    def get_revenue_segment_unit_economics(self, symbol: str, segment: str, period: str = "annual"):
        """Get unit economics for revenue segment (if available)"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-unit-economics?symbol={symbol}&segment={segment}&period={period}"
        return self.make_req(url)
    
    def get_revenue_segment_cohort_analysis(self, symbol: str, segment: str, cohort_period: str = "monthly"):
        """Get cohort analysis for revenue segment"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-cohort-analysis?symbol={symbol}&segment={segment}&cohortPeriod={cohort_period}"
        return self.make_req(url)
    
    def get_revenue_segment_mix_optimization(self, symbol: str, period: str = "annual"):
        """Get revenue mix optimization analysis"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-mix-optimization?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_revenue_segment_alerts(self, symbols: str, segment_type: str = "business", threshold: float = 10.0):
        """Set up alerts for significant revenue segment changes"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-alerts?symbols={symbols}&segmentType={segment_type}&threshold={threshold}"
        return self.make_req(url)
    
    def get_revenue_segment_export(self, symbol: str, segment_type: str = "business", format: str = "csv", periods: int = 20):
        """Export revenue segment data"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-segment-export?symbol={symbol}&segmentType={segment_type}&format={format}&periods={periods}"
        return self.make_req(url)

    # ==================== PRICE TARGETS & ANALYST ESTIMATES ====================
    
    def get_price_target(self, symbol: str):
        """Get analyst price targets for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/price-target?symbol={symbol}"
        return self.make_req(url)
    
    def get_price_target_summary(self, symbol: str):
        """Get price target summary with consensus estimates"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-summary?symbol={symbol}"
        return self.make_req(url)
    
    def get_price_target_by_analyst(self, name: str):
        """Get price targets by specific analyst name"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-analyst-name?name={name}"
        return self.make_req(url)
    
    def get_price_target_by_company(self, company: str):
        """Get price targets by analyst company"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-analyst-company?company={company}"
        return self.make_req(url)
    
    def get_price_target_consensus(self, symbol: str):
        """Get price target consensus across all analysts"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-consensus?symbol={symbol}"
        return self.make_req(url)
    
    def get_analyst_estimates(self, symbol: str, period: str = "annual"):
        """Get analyst earnings and revenue estimates"""
        url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}?period={period}"
        return self.make_req(url)
    
    def get_upgrades_downgrades(self, symbol: str):
        """Get recent analyst upgrades and downgrades for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/upgrades-downgrades?symbol={symbol}"
        return self.make_req(url)
    
    def get_upgrades_downgrades_consensus(self, symbol: str):
        """Get consensus rating from all analyst upgrades and downgrades"""
        url = f"https://financialmodelingprep.com/api/v4/upgrades-downgrades-consensus?symbol={symbol}"
        return self.make_req(url)
    
    def get_upgrades_downgrades_by_company(self, company: str):
        """Get all upgrades and downgrades from a specific analyst company"""
        url = f"https://financialmodelingprep.com/api/v4/upgrades-downgrades-grading-company?company={company}"
        return self.make_req(url)
    
    def get_market_cap_history(self, symbol: str, limit: int = 50):
        """Get historical market capitalization"""
        url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{symbol}?limit={limit}"
        return self.make_req(url)
    
    def get_enterprise_value_history(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get historical enterprise value metrics"""
        url = f"https://financialmodelingprep.com/api/v3/enterprise-values/{symbol}?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_valuation_multiples(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get valuation multiples (P/E, P/B, EV/EBITDA, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/valuation-multiples?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_sector_valuation_multiples(self, sector: str, date: Optional[str] = None):
        """Get valuation multiples for entire sector"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/sector-valuation-multiples?sector={sector}&date={date}"
        else:
            url = f"https://financialmodelingprep.com/api/v4/sector-valuation-multiples?sector={sector}"
        return self.make_req(url)
    
    def get_industry_valuation_multiples(self, industry: str, date: Optional[str] = None):
        """Get valuation multiples for entire industry"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/industry-valuation-multiples?industry={industry}&date={date}"
        else:
            url = f"https://financialmodelingprep.com/api/v4/industry-valuation-multiples?industry={industry}"
        return self.make_req(url)
    
    def get_comparable_companies_valuation(self, symbol: str):
        """Get valuation metrics for comparable companies"""
        url = f"https://financialmodelingprep.com/api/v4/comparable-companies-valuation?symbol={symbol}"
        return self.make_req(url)
    
    def get_sum_of_parts_valuation(self, symbol: str):
        """Get sum of parts (SOTP) valuation for conglomerates"""
        url = f"https://financialmodelingprep.com/api/v4/sum-of-parts-valuation?symbol={symbol}"
        return self.make_req(url)
    
    def get_asset_based_valuation(self, symbol: str):
        """Get asset-based valuation (book value approach)"""
        url = f"https://financialmodelingprep.com/api/v4/asset-based-valuation?symbol={symbol}"
        return self.make_req(url)
    
    def get_liquidation_value(self, symbol: str):
        """Get liquidation value estimate"""
        url = f"https://financialmodelingprep.com/api/v4/liquidation-value?symbol={symbol}"
        return self.make_req(url)
    
    def get_replacement_cost_valuation(self, symbol: str):
        """Get replacement cost valuation"""
        url = f"https://financialmodelingprep.com/api/v4/replacement-cost-valuation?symbol={symbol}"
        return self.make_req(url)
    
    def get_dividend_discount_model(self, symbol: str):
        """Get dividend discount model (DDM) valuation"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-discount-model?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_power_value(self, symbol: str):
        """Get earnings power value (EPV) - normalized earnings approach"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-power-value?symbol={symbol}"
        return self.make_req(url)
    
    def get_residual_income_model(self, symbol: str):
        """Get residual income model valuation"""
        url = f"https://financialmodelingprep.com/api/v4/residual-income-model?symbol={symbol}"
        return self.make_req(url)
    
    def get_economic_value_added(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get Economic Value Added (EVA) analysis"""
        url = f"https://financialmodelingprep.com/api/v4/economic-value-added?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_fair_value_estimate(self, symbol: str):
        """Get comprehensive fair value estimate using multiple methods"""
        url = f"https://financialmodelingprep.com/api/v4/fair-value-estimate?symbol={symbol}"
        return self.make_req(url)
    
    def get_valuation_ranges(self, symbol: str):
        """Get valuation ranges (bear, base, bull scenarios)"""
        url = f"https://financialmodelingprep.com/api/v4/valuation-ranges?symbol={symbol}"
        return self.make_req(url)
    
    def get_monte_carlo_valuation(self, symbol: str, simulations: int = 10000):
        """Get Monte Carlo simulation valuation"""
        url = f"https://financialmodelingprep.com/api/v4/monte-carlo-valuation?symbol={symbol}&simulations={simulations}"
        return self.make_req(url)
    
    def get_sensitivity_analysis(self, symbol: str):
        """Get valuation sensitivity analysis to key assumptions"""
        url = f"https://financialmodelingprep.com/api/v4/valuation-sensitivity?symbol={symbol}"
        return self.make_req(url)
    
    def get_relative_valuation_analysis(self, symbols: str):
        """Compare relative valuation across multiple companies"""
        url = f"https://financialmodelingprep.com/api/v4/relative-valuation?symbols={symbols}"
        return self.make_req(url)
    
    def get_valuation_summary_report(self, symbol: str):
        """Get comprehensive valuation summary using all methodologies"""
        url = f"https://financialmodelingprep.com/api/v4/valuation-summary?symbol={symbol}"
        return self.make_req(url)
    
    # ===== PRICE TARGETS SECTION =====
    
    def get_price_targets(self, symbol: str):
        """Get all analyst price targets for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/price-target?symbol={symbol}"
        return self.make_req(url)
    
    def get_price_target_summary(self, symbol: str):
        """Get price target summary with high, low, average, and median targets"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-summary?symbol={symbol}"
        return self.make_req(url)
    
    def get_price_target_consensus(self, symbol: str):
        """Get consensus price target across all analysts"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-consensus?symbol={symbol}"
        return self.make_req(url)
    
    def get_price_targets_by_analyst_name(self, name: str):
        """Get price targets by specific analyst name"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-analyst-name?name={name}"
        return self.make_req(url)
    
    def get_price_targets_by_analyst_company(self, company: str):
        """Get price targets by analyst company/firm"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-analyst-company?company={company}"
        return self.make_req(url)
    
    def get_price_target_changes(self, symbol: str, limit: int = 50):
        """Get recent price target changes and revisions"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-changes?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_price_target_accuracy(self, symbol: str):
        """Get accuracy metrics for analyst price targets"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-accuracy?symbol={symbol}"
        return self.make_req(url)
    
    def get_analyst_accuracy_rankings(self, symbol: Optional[str] = None):
        """Get analyst accuracy rankings (overall or for specific stock)"""
        if symbol:
            url = f"https://financialmodelingprep.com/api/v4/analyst-accuracy-rankings?symbol={symbol}"
        else:
            url = "https://financialmodelingprep.com/api/v4/analyst-accuracy-rankings"
        return self.make_req(url)
    
    def get_price_target_distribution(self, symbol: str):
        """Get distribution of price targets (bull/bear/base cases)"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-distribution?symbol={symbol}"
        return self.make_req(url)
    
    def get_analyst_coverage_trends(self, symbol: str):
        """Get analyst coverage trends and changes over time"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-coverage-trends?symbol={symbol}"
        return self.make_req(url)
    
    def get_price_target_vs_performance(self, symbol: str, period: str = "1y"):
        """Compare price targets vs actual stock performance"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-performance?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_analyst_recommendations_history(self, symbol: str, limit: int = 50):
        """Get historical analyst recommendations and changes"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-recommendations-history?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_upgrades_downgrades_recent(self, symbol: str):
        """Get recent analyst upgrades and downgrades"""
        url = f"https://financialmodelingprep.com/api/v4/upgrades-downgrades?symbol={symbol}"
        return self.make_req(url)
    
    def get_upgrades_downgrades_consensus(self, symbol: str):
        """Get consensus from all analyst upgrades and downgrades"""
        url = f"https://financialmodelingprep.com/api/v4/upgrades-downgrades-consensus?symbol={symbol}"
        return self.make_req(url)
    
    def get_upgrades_downgrades_by_firm(self, company: str):
        """Get upgrades/downgrades by specific analyst firm"""
        url = f"https://financialmodelingprep.com/api/v4/upgrades-downgrades-grading-company?company={company}"
        return self.make_req(url)
    
    def get_analyst_estimates_trends(self, symbol: str, period: str = "annual"):
        """Get analyst earnings and revenue estimate trends"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-estimates-trends?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_earnings_surprises_vs_targets(self, symbol: str, limit: int = 50):
        """Get earnings surprises compared to analyst estimates"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-surprises-targets?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_analyst_coverage_initiation(self, days: int = 30):
        """Get recent analyst coverage initiations"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-coverage-initiation?days={days}"
        return self.make_req(url)
    
    def get_analyst_coverage_termination(self, days: int = 30):
        """Get recent analyst coverage terminations"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-coverage-termination?days={days}"
        return self.make_req(url)
    
    def get_price_target_calendar(self, from_date: str, to_date: str):
        """Get price target updates in a date range"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_analyst_rating_scale(self, firm: Optional[str] = None):
        """Get analyst rating scales and definitions"""
        if firm:
            url = f"https://financialmodelingprep.com/api/v4/analyst-rating-scale?firm={firm}"
        else:
            url = "https://financialmodelingprep.com/api/v4/analyst-rating-scale"
        return self.make_req(url)
    
    def get_price_target_methodology(self, symbol: str, analyst: Optional[str] = None):
        """Get price target methodology and assumptions"""
        if analyst:
            url = f"https://financialmodelingprep.com/api/v4/price-target-methodology?symbol={symbol}&analyst={analyst}"
        else:
            url = f"https://financialmodelingprep.com/api/v4/price-target-methodology?symbol={symbol}"
        return self.make_req(url)
    
    def get_sector_price_targets(self, sector: str):
        """Get price target consensus for entire sector"""
        url = f"https://financialmodelingprep.com/api/v4/sector-price-targets?sector={sector}"
        return self.make_req(url)
    
    def get_industry_price_targets(self, industry: str):
        """Get price target consensus for entire industry"""
        url = f"https://financialmodelingprep.com/api/v4/industry-price-targets?industry={industry}"
        return self.make_req(url)
    
    def get_analyst_sentiment_score(self, symbol: str):
        """Get overall analyst sentiment score"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-sentiment-score?symbol={symbol}"
        return self.make_req(url)
    
    def get_price_target_rss_feed(self, page: int = 0):
        """Get RSS feed of latest price target updates"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-rss-feed?page={page}"
        return self.make_req(url)
    
    def get_most_accurate_analysts(self, sector: Optional[str] = None, timeframe: str = "1y"):
        """Get most accurate analysts by sector or overall"""
        if sector:
            url = f"https://financialmodelingprep.com/api/v4/most-accurate-analysts?sector={sector}&timeframe={timeframe}"
        else:
            url = f"https://financialmodelingprep.com/api/v4/most-accurate-analysts?timeframe={timeframe}"
        return self.make_req(url)
    
    def get_price_target_confidence_intervals(self, symbol: str):
        """Get confidence intervals for price targets"""
        url = f"https://financialmodelingprep.com/api/v4/price-target-confidence?symbol={symbol}"
        return self.make_req(url)
    
    # ===== UPGRADES & DOWNGRADES SECTION =====
    
    def get_upgrades_downgrades(self, symbol: str):
        """Get recent analyst upgrades and downgrades for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/upgrades-downgrades?symbol={symbol}"
        return self.make_req(url)
    
    def get_upgrades_downgrades_consensus(self, symbol: str):
        """Get consensus rating from all analyst upgrades and downgrades"""
        url = f"https://financialmodelingprep.com/api/v4/upgrades-downgrades-consensus?symbol={symbol}"
        return self.make_req(url)
    
    def get_upgrades_downgrades_by_company(self, company: str):
        """Get all upgrades and downgrades from a specific analyst company"""
        url = f"https://financialmodelingprep.com/api/v4/upgrades-downgrades-grading-company?company={company}"
        return self.make_req(url)
    
    def get_upgrades_downgrades_rss_feed(self, page: int = 0):
        """Get RSS feed of recent upgrades and downgrades across all stocks"""
        url = f"https://financialmodelingprep.com/api/v4/upgrades-downgrades-rss-feed?page={page}"
        return self.make_req(url)
    
    def get_recent_rating_changes(self, days: int = 30):
        """Get all rating changes in the last specified days"""
        url = f"https://financialmodelingprep.com/api/v4/recent-rating-changes?days={days}"
        return self.make_req(url)
    
    def get_rating_changes_by_sector(self, sector: str, days: int = 30):
        """Get rating changes for a specific sector"""
        url = f"https://financialmodelingprep.com/api/v4/rating-changes-sector?sector={sector}&days={days}"
        return self.make_req(url)
    
    def get_rating_changes_by_industry(self, industry: str, days: int = 30):
        """Get rating changes for a specific industry"""
        url = f"https://financialmodelingprep.com/api/v4/rating-changes-industry?industry={industry}&days={days}"
        return self.make_req(url)
    
    def get_analyst_rating_history(self, symbol: str, limit: int = 50):
        """Get historical analyst rating changes for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-rating-history?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_rating_distribution(self, symbol: str):
        """Get current distribution of analyst ratings (Strong Buy, Buy, Hold, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/rating-distribution?symbol={symbol}"
        return self.make_req(url)
    
    def get_rating_trends(self, symbol: str, period: str = "3m"):
        """Get rating trends over specified period (1m, 3m, 6m, 1y)"""
        url = f"https://financialmodelingprep.com/api/v4/rating-trends?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_most_upgraded_stocks(self, timeframe: str = "1w", limit: int = 50):
        """Get stocks with most upgrades in timeframe (1d, 1w, 1m)"""
        url = f"https://financialmodelingprep.com/api/v4/most-upgraded-stocks?timeframe={timeframe}&limit={limit}"
        return self.make_req(url)
    
    def get_most_downgraded_stocks(self, timeframe: str = "1w", limit: int = 50):
        """Get stocks with most downgrades in timeframe (1d, 1w, 1m)"""
        url = f"https://financialmodelingprep.com/api/v4/most-downgraded-stocks?timeframe={timeframe}&limit={limit}"
        return self.make_req(url)
    
    def get_analyst_firm_performance(self, company: str, timeframe: str = "1y"):
        """Get performance metrics for a specific analyst firm"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-firm-performance?company={company}&timeframe={timeframe}"
        return self.make_req(url)
    
    def get_top_analyst_firms(self, sector: Optional[str] = None, timeframe: str = "1y"):
        """Get top performing analyst firms by accuracy"""
        if sector:
            url = f"https://financialmodelingprep.com/api/v4/top-analyst-firms?sector={sector}&timeframe={timeframe}"
        else:
            url = f"https://financialmodelingprep.com/api/v4/top-analyst-firms?timeframe={timeframe}"
        return self.make_req(url)
    
    def get_rating_impact_analysis(self, symbol: str):
        """Analyze stock price impact of rating changes"""
        url = f"https://financialmodelingprep.com/api/v4/rating-impact-analysis?symbol={symbol}"
        return self.make_req(url)
    
    def get_consensus_rating_changes(self, symbol: str, limit: int = 50):
        """Get historical consensus rating changes"""
        url = f"https://financialmodelingprep.com/api/v4/consensus-rating-changes?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_rating_calendar(self, from_date: str, to_date: str):
        """Get rating changes in a specific date range"""
        url = f"https://financialmodelingprep.com/api/v4/rating-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_pre_market_rating_changes(self, date: Optional[str] = None):
        """Get pre-market rating changes for trading day"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/pre-market-ratings?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v4/pre-market-ratings"
        return self.make_req(url)
    
    def get_after_hours_rating_changes(self, date: Optional[str] = None):
        """Get after-hours rating changes"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/after-hours-ratings?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v4/after-hours-ratings"
        return self.make_req(url)
    
    def get_rating_momentum(self, symbol: str):
        """Get rating momentum indicators (improving/deteriorating)"""
        url = f"https://financialmodelingprep.com/api/v4/rating-momentum?symbol={symbol}"
        return self.make_req(url)
    
    def get_contrarian_ratings(self, timeframe: str = "1m"):
        """Get stocks with contrarian rating moves (against market sentiment)"""
        url = f"https://financialmodelingprep.com/api/v4/contrarian-ratings?timeframe={timeframe}"
        return self.make_req(url)
    
    def get_rating_revision_impact(self, symbol: str, limit: int = 20):
        """Analyze impact of rating revisions on stock price"""
        url = f"https://financialmodelingprep.com/api/v4/rating-revision-impact?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_sector_rating_overview(self, sector: str):
        """Get rating overview for entire sector"""
        url = f"https://financialmodelingprep.com/api/v4/sector-rating-overview?sector={sector}"
        return self.make_req(url)
    
    def get_industry_rating_overview(self, industry: str):
        """Get rating overview for entire industry"""
        url = f"https://financialmodelingprep.com/api/v4/industry-rating-overview?industry={industry}"
        return self.make_req(url)
    
    def get_analyst_sentiment_changes(self, symbol: str, period: str = "6m"):
        """Get analyst sentiment changes over time"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-sentiment-changes?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_rating_alerts(self, symbols: str):
        """Get alerts for significant rating changes (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v4/rating-alerts?symbols={symbols}"
        return self.make_req(url)
    
    def get_institutional_rating_correlation(self, symbol: str):
        """Correlate analyst ratings with institutional ownership changes"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-rating-correlation?symbol={symbol}"
        return self.make_req(url)
    
    # ===== NEWS SECTION =====
    
    def get_fmp_articles(self, page: int = 0, size: int = 5):
        """Get latest articles from Financial Modeling Prep"""
        url = f"https://financialmodelingprep.com/api/v3/fmp/articles?page={page}&size={size}"
        return self.make_req(url)
    
    def get_general_news(self, page: int = 0):
        """Get latest general news articles from various sources"""
        url = f"https://financialmodelingprep.com/api/v4/general_news?page={page}"
        return self.make_req(url)
    
    def get_stock_news(self, tickers: Optional[str] = None, page: int = 0, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get stock news articles with optional ticker filtering and date range"""
        url = f"https://financialmodelingprep.com/api/v3/stock_news?page={page}&limit={limit}"
        
        if tickers:
            url += f"&tickers={tickers}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_stock_news_sentiments_rss(self, page: int = 0):
        """Get RSS feed of stock news with sentiment analysis"""
        url = f"https://financialmodelingprep.com/api/v4/stock-news-sentiments-rss-feed?page={page}"
        return self.make_req(url)
    
    def get_forex_news(self, page: int = 0):
        """Get latest forex news articles"""
        url = f"https://financialmodelingprep.com/api/v4/forex_news?page={page}"
        return self.make_req(url)
    
    def get_crypto_news(self, page: int = 0, symbol: Optional[str] = None):
        """Get latest crypto news articles (symbol format: BTCUSD)"""
        url = f"https://financialmodelingprep.com/api/v4/crypto_news?page={page}"
        if symbol:
            url += f"&symbol={symbol}"
        return self.make_req(url)
    
    def get_press_releases(self, page: int = 0):
        """Get latest press releases from various companies"""
        url = f"https://financialmodelingprep.com/api/v3/press-releases?page={page}"
        return self.make_req(url)
    
    def get_press_releases_by_symbol(self, symbol: str, page: int = 0):
        """Get press releases from a specific company"""
        url = f"https://financialmodelingprep.com/api/v3/press-releases/{symbol}?page={page}"
        return self.make_req(url)
    
    def get_historical_social_sentiment(self, symbol: str, page: int = 0):
        """Get historical social sentiment data for a ticker"""
        url = f"https://financialmodelingprep.com/api/v4/historical/social-sentiment?symbol={symbol}&page={page}"
        return self.make_req(url)
    
    def get_trending_social_sentiment(self, type: str = "bullish", source: str = "stocktwits"):
        """Get trending social sentiment data (type: bullish/bearish, source: stocktwits/twitter)"""
        url = f"https://financialmodelingprep.com/api/v4/social-sentiments/trending?type={type}&source={source}"
        return self.make_req(url)
    
    def get_social_sentiment_changes(self, type: str = "bullish", source: str = "stocktwits"):
        """Get changes in social sentiment over time"""
        url = f"https://financialmodelingprep.com/api/v4/social-sentiments/change?type={type}&source={source}"
        return self.make_req(url)
    
    def get_news_by_symbol(self, symbol: str, limit: int = 50):
        """Get news articles for a specific stock symbol"""
        url = f"https://financialmodelingprep.com/api/v4/news-by-symbol?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_market_news(self, category: str = "general", limit: int = 50):
        """Get market news by category (general, earnings, ipo, mergers, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/market-news?category={category}&limit={limit}"
        return self.make_req(url)
    
    def get_breaking_news(self):
        """Get breaking news and urgent market updates"""
        url = "https://financialmodelingprep.com/api/v4/breaking-news"
        return self.make_req(url)
    
    def get_news_sentiment_analysis(self, symbol: str, days: int = 30):
        """Get sentiment analysis of news for a specific stock"""
        url = f"https://financialmodelingprep.com/api/v4/news-sentiment-analysis?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_sector_news(self, sector: str, limit: int = 50):
        """Get news articles for a specific sector"""
        url = f"https://financialmodelingprep.com/api/v4/sector-news?sector={sector}&limit={limit}"
        return self.make_req(url)
    
    def get_industry_news(self, industry: str, limit: int = 50):
        """Get news articles for a specific industry"""
        url = f"https://financialmodelingprep.com/api/v4/industry-news?industry={industry}&limit={limit}"
        return self.make_req(url)
    
    def get_earnings_news(self, symbol: Optional[str] = None, limit: int = 50):
        """Get earnings-related news (for specific symbol or all)"""
        if symbol:
            url = f"https://financialmodelingprep.com/api/v4/earnings-news?symbol={symbol}&limit={limit}"
        else:
            url = f"https://financialmodelingprep.com/api/v4/earnings-news?limit={limit}"
        return self.make_req(url)
    
    def get_merger_news(self, limit: int = 50):
        """Get merger and acquisition news"""
        url = f"https://financialmodelingprep.com/api/v4/merger-news?limit={limit}"
        return self.make_req(url)
    
    def get_ipo_news(self, limit: int = 50):
        """Get IPO and new listing news"""
        url = f"https://financialmodelingprep.com/api/v4/ipo-news?limit={limit}"
        return self.make_req(url)
    
    def get_analyst_news(self, limit: int = 50):
        """Get analyst reports and research news"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-news?limit={limit}"
        return self.make_req(url)
    
    def get_economic_news(self, limit: int = 50):
        """Get economic and macro news"""
        url = f"https://financialmodelingprep.com/api/v4/economic-news?limit={limit}"
        return self.make_req(url)
    
    def get_regulatory_news(self, limit: int = 50):
        """Get regulatory and policy news"""
        url = f"https://financialmodelingprep.com/api/v4/regulatory-news?limit={limit}"
        return self.make_req(url)
    
    def get_news_impact_score(self, symbol: str, days: int = 7):
        """Get news impact score on stock price"""
        url = f"https://financialmodelingprep.com/api/v4/news-impact-score?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_news_keywords_analysis(self, symbol: str, days: int = 30):
        """Get analysis of keywords in news for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/news-keywords-analysis?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_news_volume_analysis(self, symbol: str, days: int = 30):
        """Get analysis of news volume and frequency"""
        url = f"https://financialmodelingprep.com/api/v4/news-volume-analysis?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def search_news(self, query: str, limit: int = 50, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Search news articles by keyword"""
        url = f"https://financialmodelingprep.com/api/v4/search-news?query={query}&limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_news_sources(self):
        """Get list of available news sources"""
        url = "https://financialmodelingprep.com/api/v4/news-sources"
        return self.make_req(url)
    
    def get_news_by_source(self, source: str, limit: int = 50):
        """Get news from a specific source"""
        url = f"https://financialmodelingprep.com/api/v4/news-by-source?source={source}&limit={limit}"
        return self.make_req(url)
    
    def get_social_media_trends(self, symbol: str, platform: str = "all"):
        """Get social media trends for a stock (platform: twitter, reddit, stocktwits, all)"""
        url = f"https://financialmodelingprep.com/api/v4/social-media-trends?symbol={symbol}&platform={platform}"
        return self.make_req(url)
    
    def get_news_calendar(self, from_date: str, to_date: str):
        """Get news calendar for a date range"""
        url = f"https://financialmodelingprep.com/api/v4/news-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_real_time_news_feed(self, symbols: Optional[str] = None):
        """Get real-time news feed (optional: comma-separated symbols)"""
        if symbols:
            url = f"https://financialmodelingprep.com/api/v4/real-time-news?symbols={symbols}"
        else:
            url = "https://financialmodelingprep.com/api/v4/real-time-news"
        return self.make_req(url)
    
    # ===== SEC FILINGS SECTION =====
    
    def get_sec_rss_feed_8k(self, page: int = 0, from_date: Optional[str] = None, to_date: Optional[str] = None, 
                           has_financial: Optional[bool] = None, limit: int = 100):
        """Get RSS feed of 8-K SEC filings from publicly traded companies"""
        url = f"https://financialmodelingprep.com/api/v4/rss_feed_8k?page={page}&limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        if has_financial is not None:
            url += f"&hasFinancial={str(has_financial).lower()}"
        
        return self.make_req(url)
    
    def get_sec_filings(self, symbol: str, filing_type: Optional[str] = None, page: int = 0, limit: int = 100):
        """Get SEC filings for a specific company (10-K, 10-Q, 8-K, etc.)"""
        url = f"https://financialmodelingprep.com/api/v3/sec_filings/{symbol}?page={page}&limit={limit}"
        
        if filing_type:
            url += f"&type={filing_type}"
        
        return self.make_req(url)
    
    def get_form_13f_filings(self, cik: str, date: Optional[str] = None):
        """Get Form 13F filings (institutional investment manager holdings)"""
        if date:
            url = f"https://financialmodelingprep.com/api/v3/form-thirteen/{cik}?date={date}"
        else:
            url = f"https://financialmodelingprep.com/api/v3/form-thirteen/{cik}"
        return self.make_req(url)
    
    def get_form_13f_dates(self, cik: str):
        """Get available dates for Form 13F filings for a CIK"""
        url = f"https://financialmodelingprep.com/api/v3/form-thirteen-date/{cik}"
        return self.make_req(url)
    
    def get_cik_list(self, page: int = 0):
        """Get list of all CIK numbers and company names"""
        url = f"https://financialmodelingprep.com/api/v3/cik_list?page={page}"
        return self.make_req(url)
    
    def search_cik_by_name(self, name: str, limit: int = 50):
        """Search for CIK numbers by company name"""
        url = f"https://financialmodelingprep.com/api/v3/cik-search/{name}?limit={limit}"
        return self.make_req(url)
    
    def get_company_by_cik(self, cik: str):
        """Get company information by CIK number"""
        url = f"https://financialmodelingprep.com/api/v3/cik/{cik}"
        return self.make_req(url)
    
    def search_by_cusip(self, cusip: str):
        """Search for companies by CUSIP number"""
        url = f"https://financialmodelingprep.com/api/v3/cusip/{cusip}"
        return self.make_req(url)
    
    def get_sec_filing_search(self, query: str, limit: int = 50):
        """Search SEC filings by keyword or company name"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-search?query={query}&limit={limit}"
        return self.make_req(url)
    
    def get_10k_filings(self, symbol: str, year: Optional[int] = None, limit: int = 50):
        """Get 10-K annual report filings for a company"""
        url = f"https://financialmodelingprep.com/api/v4/10-k-filings?symbol={symbol}&limit={limit}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_10q_filings(self, symbol: str, year: Optional[int] = None, quarter: Optional[int] = None, limit: int = 50):
        """Get 10-Q quarterly report filings for a company"""
        url = f"https://financialmodelingprep.com/api/v4/10-q-filings?symbol={symbol}&limit={limit}"
        
        if year:
            url += f"&year={year}"
        if quarter:
            url += f"&quarter={quarter}"
        
        return self.make_req(url)
    
    def get_8k_filings(self, symbol: str, year: Optional[int] = None, limit: int = 50):
        """Get 8-K current report filings for a company"""
        url = f"https://financialmodelingprep.com/api/v4/8-k-filings?symbol={symbol}&limit={limit}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_proxy_statements(self, symbol: str, year: Optional[int] = None, limit: int = 50):
        """Get proxy statements (DEF 14A) for a company"""
        url = f"https://financialmodelingprep.com/api/v4/proxy-statements?symbol={symbol}&limit={limit}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_insider_trading_forms(self, symbol: str, form_type: Optional[str] = None, limit: int = 50):
        """Get insider trading forms (Form 3, 4, 5) for a company"""
        url = f"https://financialmodelingprep.com/api/v4/insider-trading-forms?symbol={symbol}&limit={limit}"
        
        if form_type:  # 3, 4, or 5
            url += f"&formType={form_type}"
        
        return self.make_req(url)
    
    def get_form_4_filings(self, symbol: str, limit: int = 50):
        """Get Form 4 insider trading filings for a company"""
        url = f"https://financialmodelingprep.com/api/v4/form-4-filings?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_form_3_filings(self, symbol: str, limit: int = 50):
        """Get Form 3 initial insider ownership filings"""
        url = f"https://financialmodelingprep.com/api/v4/form-3-filings?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_form_5_filings(self, symbol: str, limit: int = 50):
        """Get Form 5 annual insider trading filings"""
        url = f"https://financialmodelingprep.com/api/v4/form-5-filings?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_s1_filings(self, symbol: Optional[str] = None, limit: int = 50):
        """Get S-1 registration statements (IPO filings)"""
        if symbol:
            url = f"https://financialmodelingprep.com/api/v4/s1-filings?symbol={symbol}&limit={limit}"
        else:
            url = f"https://financialmodelingprep.com/api/v4/s1-filings?limit={limit}"
        return self.make_req(url)
    
    def get_13d_filings(self, symbol: str, limit: int = 50):
        """Get 13D beneficial ownership filings (5%+ ownership)"""
        url = f"https://financialmodelingprep.com/api/v4/13d-filings?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_13g_filings(self, symbol: str, limit: int = 50):
        """Get 13G beneficial ownership filings (passive ownership)"""
        url = f"https://financialmodelingprep.com/api/v4/13g-filings?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_sc_13d_filings(self, symbol: str, limit: int = 50):
        """Get SC 13D filings (tender offer beneficial ownership)"""
        url = f"https://financialmodelingprep.com/api/v4/sc-13d-filings?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_filing_dates(self, symbol: str, filing_type: Optional[str] = None):
        """Get available filing dates for a company"""
        url = f"https://financialmodelingprep.com/api/v4/filing-dates?symbol={symbol}"
        
        if filing_type:
            url += f"&type={filing_type}"
        
        return self.make_req(url)
    
    def get_sec_filing_full_text(self, symbol: str, filing_type: str, date: str):
        """Get full text of a specific SEC filing"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-text?symbol={symbol}&type={filing_type}&date={date}"
        return self.make_req(url)
    
    def get_sec_filing_analysis(self, symbol: str, filing_type: str, date: str):
        """Get AI analysis of SEC filing content"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-analysis?symbol={symbol}&type={filing_type}&date={date}"
        return self.make_req(url)
    
    def get_sec_filing_sentiment(self, symbol: str, filing_type: str, date: str):
        """Get sentiment analysis of SEC filing"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-sentiment?symbol={symbol}&type={filing_type}&date={date}"
        return self.make_req(url)
    
    def get_sec_filing_key_changes(self, symbol: str, filing_type: str, current_date: str, previous_date: str):
        """Compare SEC filings and identify key changes"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-changes?symbol={symbol}&type={filing_type}&current={current_date}&previous={previous_date}"
        return self.make_req(url)
    
    def get_risk_factors_from_filings(self, symbol: str, filing_type: str = "10-K", year: Optional[int] = None):
        """Extract risk factors from SEC filings"""
        url = f"https://financialmodelingprep.com/api/v4/risk-factors?symbol={symbol}&type={filing_type}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_md_a_from_filings(self, symbol: str, filing_type: str = "10-K", year: Optional[int] = None):
        """Extract Management Discussion & Analysis from SEC filings"""
        url = f"https://financialmodelingprep.com/api/v4/md-a-extract?symbol={symbol}&type={filing_type}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_business_description_from_filings(self, symbol: str, filing_type: str = "10-K", year: Optional[int] = None):
        """Extract business description from SEC filings"""
        url = f"https://financialmodelingprep.com/api/v4/business-description?symbol={symbol}&type={filing_type}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_legal_proceedings_from_filings(self, symbol: str, filing_type: str = "10-K", year: Optional[int] = None):
        """Extract legal proceedings from SEC filings"""
        url = f"https://financialmodelingprep.com/api/v4/legal-proceedings?symbol={symbol}&type={filing_type}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_sec_filing_schedule(self, symbol: Optional[str] = None, days_ahead: int = 30):
        """Get upcoming SEC filing schedule"""
        if symbol:
            url = f"https://financialmodelingprep.com/api/v4/sec-filing-schedule?symbol={symbol}&days={days_ahead}"
        else:
            url = f"https://financialmodelingprep.com/api/v4/sec-filing-schedule?days={days_ahead}"
        return self.make_req(url)
    
    def get_sec_filing_calendar(self, from_date: str, to_date: str, filing_type: Optional[str] = None):
        """Get SEC filing calendar for date range"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-calendar?from={from_date}&to={to_date}"
        
        if filing_type:
            url += f"&type={filing_type}"
        
        return self.make_req(url)
    
    def get_sec_filing_statistics(self, symbol: str, year: Optional[int] = None):
        """Get SEC filing statistics and compliance metrics"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-stats?symbol={symbol}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_sec_filing_delays(self, days: int = 30):
        """Get companies with delayed SEC filings"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-delays?days={days}"
        return self.make_req(url)
    
    def get_sec_filing_amendments(self, symbol: str, filing_type: Optional[str] = None, limit: int = 50):
        """Get SEC filing amendments for a company"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-amendments?symbol={symbol}&limit={limit}"
        
        if filing_type:
            url += f"&type={filing_type}"
        
        return self.make_req(url)
    
    def get_xbrl_data(self, symbol: str, filing_type: str = "10-K", year: Optional[int] = None):
        """Get XBRL data from SEC filings"""
        url = f"https://financialmodelingprep.com/api/v4/xbrl-data?symbol={symbol}&type={filing_type}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def search_sec_filings_by_keyword(self, keyword: str, filing_type: Optional[str] = None, 
                                    from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Search SEC filings by keyword or phrase"""
        url = f"https://financialmodelingprep.com/api/v4/search-sec-filings?keyword={keyword}&limit={limit}"
        
        if filing_type:
            url += f"&type={filing_type}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_sec_filing_trends(self, symbol: str, filing_type: str, years: int = 5):
        """Get trends in SEC filing content over time"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-trends?symbol={symbol}&type={filing_type}&years={years}"
        return self.make_req(url)
    
    def get_sec_edgar_search(self, entity_name: Optional[str] = None, cik: Optional[str] = None, 
                           filing_type: Optional[str] = None, date_filed: Optional[str] = None):
        """Search SEC EDGAR database"""
        url = "https://financialmodelingprep.com/api/v4/sec-edgar-search?"
        
        params = []
        if entity_name:
            params.append(f"entity={entity_name}")
        if cik:
            params.append(f"cik={cik}")
        if filing_type:
            params.append(f"type={filing_type}")
        if date_filed:
            params.append(f"date={date_filed}")
        
        url += "&".join(params)
        return self.make_req(url)
    
    def get_institutional_holdings_from_13f(self, cik: str, date: Optional[str] = None, symbol: Optional[str] = None):
        """Get institutional holdings from 13F filings"""
        url = f"https://financialmodelingprep.com/api/v3/form-thirteen/{cik}"
        
        if date:
            url += f"?date={date}"
        if symbol:
            connector = "&" if date else "?"
            url += f"{connector}symbol={symbol}"
        
        return self.make_req(url)
    
    def get_13f_institutional_holdings_summary(self, date: str):
        """Get summary of all institutional holdings for a specific date"""
        url = f"https://financialmodelingprep.com/api/v4/13f-holdings-summary?date={date}"
        return self.make_req(url)
    
    def get_top_institutional_holders(self, symbol: str, date: Optional[str] = None, limit: int = 50):
        """Get top institutional holders for a stock from 13F filings"""
        url = f"https://financialmodelingprep.com/api/v4/top-institutional-holders?symbol={symbol}&limit={limit}"
        
        if date:
            url += f"&date={date}"
        
        return self.make_req(url)
    
    def get_institutional_holdings_changes(self, cik: str, current_date: str, previous_date: str):
        """Get changes in institutional holdings between two periods"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-holdings-changes?cik={cik}&current={current_date}&previous={previous_date}"
        return self.make_req(url)
    
    def get_sec_filing_notifications(self, symbols: str, filing_types: Optional[str] = None):
        """Set up notifications for new SEC filings (comma-separated symbols and types)"""
        url = f"https://financialmodelingprep.com/api/v4/sec-filing-notifications?symbols={symbols}"
        
        if filing_types:
            url += f"&types={filing_types}"
        
        return self.make_req(url)
    
    # ===== DIVIDENDS SECTION =====
    
    def get_dividend_calendar(self, from_date: str, to_date: str):
        """Get dividend calendar showing upcoming dividend payments between dates"""
        url = f"https://financialmodelingprep.com/api/v3/stock_dividend_calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_historical_dividends(self, symbol: str):
        """Get historical dividend payments for a specific stock"""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{symbol}"
        return self.make_req(url)
    
    def get_dividend_yield_history(self, symbol: str, limit: int = 50):
        """Get historical dividend yield data for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-yield-history?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_dividend_growth_rate(self, symbol: str, period: str = "annual", limit: int = 10):
        """Get dividend growth rate analysis (annual or quarterly)"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-growth-rate?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_dividend_payout_ratio(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get dividend payout ratio history"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-payout-ratio?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_dividend_coverage_ratio(self, symbol: str, period: str = "annual", limit: int = 50):
        """Get dividend coverage ratio (earnings/dividends)"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-coverage-ratio?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_dividend_sustainability_score(self, symbol: str):
        """Get dividend sustainability score and analysis"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-sustainability?symbol={symbol}"
        return self.make_req(url)
    
    def get_dividend_aristocrats(self, years: int = 25):
        """Get dividend aristocrats (companies with consecutive dividend increases)"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-aristocrats?years={years}"
        return self.make_req(url)
    
    def get_dividend_kings(self):
        """Get dividend kings (companies with 50+ years of consecutive dividend increases)"""
        url = "https://financialmodelingprep.com/api/v4/dividend-kings"
        return self.make_req(url)
    
    def get_dividend_champions(self, years: int = 10):
        """Get dividend champions (companies with consecutive dividend increases, customizable years)"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-champions?years={years}"
        return self.make_req(url)
    
    def get_ex_dividend_calendar(self, from_date: str, to_date: str):
        """Get ex-dividend dates calendar"""
        url = f"https://financialmodelingprep.com/api/v4/ex-dividend-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_dividend_declaration_date(self, symbol: str, limit: int = 50):
        """Get dividend declaration dates for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-declaration-dates?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_dividend_payment_date(self, symbol: str, limit: int = 50):
        """Get dividend payment dates for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-payment-dates?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_record_date(self, symbol: str, limit: int = 50):
        """Get dividend record dates for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-record-dates?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_dividend_frequency(self, symbol: str):
        """Get dividend payment frequency analysis (monthly, quarterly, annual, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-frequency?symbol={symbol}"
        return self.make_req(url)
    
    def get_special_dividends(self, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get special (one-time) dividends"""
        url = "https://financialmodelingprep.com/api/v4/special-dividends?"
        
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        url += "&".join(params)
        return self.make_req(url)
    
    def get_stock_splits_with_dividends(self, symbol: str, limit: int = 50):
        """Get stock splits that occurred around dividend dates"""
        url = f"https://financialmodelingprep.com/api/v4/splits-with-dividends?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_dividend_adjusted_price(self, symbol: str, from_date: str, to_date: str):
        """Get dividend-adjusted historical stock prices"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-adjusted-price?symbol={symbol}&from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_dividend_reinvestment_returns(self, symbol: str, years: int = 10):
        """Calculate returns with dividend reinvestment"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-reinvestment-returns?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_dividend_capture_dates(self, symbol: str, limit: int = 20):
        """Get optimal dividend capture dates and analysis"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-capture-dates?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_sector_dividend_yields(self, sector: str, date: Optional[str] = None):
        """Get dividend yields for all companies in a sector"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/sector-dividend-yields?sector={sector}&date={date}"
        else:
            url = f"https://financialmodelingprep.com/api/v4/sector-dividend-yields?sector={sector}"
        return self.make_req(url)
    
    def get_industry_dividend_yields(self, industry: str, date: Optional[str] = None):
        """Get dividend yields for all companies in an industry"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/industry-dividend-yields?industry={industry}&date={date}"
        else:
            url = f"https://financialmodelingprep.com/api/v4/industry-dividend-yields?industry={industry}"
        return self.make_req(url)
    
    def get_high_dividend_yield_stocks(self, min_yield: float = 4.0, market_cap_min: int = 1000000000, limit: int = 50):
        """Get stocks with high dividend yields above specified minimum"""
        url = f"https://financialmodelingprep.com/api/v4/high-dividend-stocks?minYield={min_yield}&minMarketCap={market_cap_min}&limit={limit}"
        return self.make_req(url)
    
    def get_dividend_growth_stocks(self, min_growth_years: int = 5, min_growth_rate: float = 5.0, limit: int = 50):
        """Get stocks with consistent dividend growth"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-growth-stocks?minYears={min_growth_years}&minGrowthRate={min_growth_rate}&limit={limit}"
        return self.make_req(url)
    
    def get_dividend_cuts_suspensions(self, from_date: str, to_date: str):
        """Get dividend cuts and suspensions in date range"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-cuts-suspensions?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_dividend_increases(self, from_date: str, to_date: str, min_increase_percent: float = 0.0):
        """Get dividend increases in date range"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-increases?from={from_date}&to={to_date}&minIncrease={min_increase_percent}"
        return self.make_req(url)
    
    def get_dividend_initiations(self, from_date: str, to_date: str):
        """Get companies that initiated dividends"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-initiations?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_dividend_resumptions(self, from_date: str, to_date: str):
        """Get companies that resumed dividend payments after suspension"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-resumptions?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_monthly_dividend_stocks(self, min_yield: Optional[float] = None, limit: int = 50):
        """Get stocks that pay monthly dividends"""
        url = f"https://financialmodelingprep.com/api/v4/monthly-dividend-stocks?limit={limit}"
        
        if min_yield:
            url += f"&minYield={min_yield}"
        
        return self.make_req(url)
    
    def get_quarterly_dividend_calendar(self, year: int, quarter: int):
        """Get quarterly dividend calendar for specific year and quarter"""
        url = f"https://financialmodelingprep.com/api/v4/quarterly-dividend-calendar?year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_dividend_tax_analysis(self, symbol: str, tax_rate: float = 0.2):
        """Get dividend tax analysis and after-tax yields"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-tax-analysis?symbol={symbol}&taxRate={tax_rate}"
        return self.make_req(url)
    
    def get_dividend_reinvestment_plans(self, symbol: Optional[str] = None):
        """Get information about dividend reinvestment plans (DRIPs)"""
        if symbol:
            url = f"https://financialmodelingprep.com/api/v4/dividend-reinvestment-plans?symbol={symbol}"
        else:
            url = "https://financialmodelingprep.com/api/v4/dividend-reinvestment-plans"
        return self.make_req(url)
    
    def get_foreign_dividend_withholding(self, symbol: str, country: Optional[str] = None):
        """Get foreign dividend withholding tax information"""
        url = f"https://financialmodelingprep.com/api/v4/foreign-dividend-withholding?symbol={symbol}"
        
        if country:
            url += f"&country={country}"
        
        return self.make_req(url)
    
    def get_dividend_portfolio_analysis(self, symbols: str, investment_amount: float = 10000.0):
        """Analyze a dividend portfolio (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-portfolio-analysis?symbols={symbols}&amount={investment_amount}"
        return self.make_req(url)
    
    def get_dividend_screener(self, min_yield: Optional[float] = None, max_yield: Optional[float] = None,
                            min_payout_ratio: Optional[float] = None, max_payout_ratio: Optional[float] = None,
                            min_years_growth: Optional[int] = None, sector: Optional[str] = None,
                            market_cap_min: Optional[int] = None, limit: int = 50):
        """Screen stocks based on dividend criteria"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-screener?limit={limit}"
        
        if min_yield:
            url += f"&minYield={min_yield}"
        if max_yield:
            url += f"&maxYield={max_yield}"
        if min_payout_ratio:
            url += f"&minPayoutRatio={min_payout_ratio}"
        if max_payout_ratio:
            url += f"&maxPayoutRatio={max_payout_ratio}"
        if min_years_growth:
            url += f"&minYearsGrowth={min_years_growth}"
        if sector:
            url += f"&sector={sector}"
        if market_cap_min:
            url += f"&minMarketCap={market_cap_min}"
        
        return self.make_req(url)
    
    def get_dividend_forecast(self, symbol: str, periods: int = 4):
        """Get dividend payment forecasts"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-forecast?symbol={symbol}&periods={periods}"
        return self.make_req(url)
    
    def get_dividend_vs_buyback_analysis(self, symbol: str, years: int = 5):
        """Compare dividend payments vs share buybacks"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-vs-buyback?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_div_yield_vs_market(self, symbol: str, benchmark: str = "SPY"):
        """Compare dividend yield vs market benchmark"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-yield-vs-market?symbol={symbol}&benchmark={benchmark}"
        return self.make_req(url)
    
    def get_dividend_safety_score(self, symbol: str):
        """Get comprehensive dividend safety score"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-safety-score?symbol={symbol}"
        return self.make_req(url)
    
    def get_dividend_quality_metrics(self, symbol: str, period: str = "annual", limit: int = 10):
        """Get dividend quality metrics and ratios"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-quality-metrics?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_reit_dividend_analysis(self, symbol: str):
        """Get REIT-specific dividend analysis"""
        url = f"https://financialmodelingprep.com/api/v4/reit-dividend-analysis?symbol={symbol}"
        return self.make_req(url)
    
    def get_utility_dividend_analysis(self, symbol: str):
        """Get utility-specific dividend analysis"""
        url = f"https://financialmodelingprep.com/api/v4/utility-dividend-analysis?symbol={symbol}"
        return self.make_req(url)
    
    def get_dividend_etf_analysis(self, symbol: str):
        """Get dividend ETF holdings and yield analysis"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-etf-analysis?symbol={symbol}"
        return self.make_req(url)
    
    def get_international_dividend_calendar(self, country: str, from_date: str, to_date: str):
        """Get dividend calendar for specific country"""
        url = f"https://financialmodelingprep.com/api/v4/international-dividend-calendar?country={country}&from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_dividend_currency_impact(self, symbol: str, base_currency: str = "USD"):
        """Get currency impact on international dividend payments"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-currency-impact?symbol={symbol}&baseCurrency={base_currency}"
        return self.make_req(url)
    
    # ===== STOCK SPLITS SECTION =====
    
    def get_stock_splits(self, symbol: str):
        """Get historical stock splits for a specific company"""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_split/{symbol}"
        return self.make_req(url)
    
    def get_stock_splits_calendar(self, from_date: str, to_date: str):
        """Get stock splits calendar for a date range"""
        url = f"https://financialmodelingprep.com/api/v3/stock_split_calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_recent_stock_splits(self, days: int = 30, limit: int = 50):
        """Get recent stock splits in the last specified days"""
        url = f"https://financialmodelingprep.com/api/v4/recent-stock-splits?days={days}&limit={limit}"
        return self.make_req(url)
    
    def get_upcoming_stock_splits(self, days_ahead: int = 30, limit: int = 50):
        """Get upcoming stock splits in the next specified days"""
        url = f"https://financialmodelingprep.com/api/v4/upcoming-stock-splits?daysAhead={days_ahead}&limit={limit}"
        return self.make_req(url)
    
    def get_stock_split_history(self, symbol: str, limit: int = 50):
        """Get detailed stock split history with ratios and dates"""
        url = f"https://financialmodelingprep.com/api/v4/stock-split-history?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_split_adjusted_prices(self, symbol: str, from_date: str, to_date: str):
        """Get split-adjusted historical stock prices"""
        url = f"https://financialmodelingprep.com/api/v4/split-adjusted-prices?symbol={symbol}&from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_stock_split_announcements(self, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get stock split announcements"""
        url = f"https://financialmodelingprep.com/api/v4/stock-split-announcements?limit={limit}"
        
        if symbol:
            url += f"&symbol={symbol}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_stock_split_ratios(self, symbol: str):
        """Get stock split ratios and impact analysis"""
        url = f"https://financialmodelingprep.com/api/v4/stock-split-ratios?symbol={symbol}"
        return self.make_req(url)
    
    def get_reverse_splits(self, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get reverse stock splits (stock consolidations)"""
        url = f"https://financialmodelingprep.com/api/v4/reverse-splits?limit={limit}"
        
        if symbol:
            url += f"&symbol={symbol}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_split_impact_analysis(self, symbol: str, days_before: int = 30, days_after: int = 30):
        """Analyze stock price impact before and after splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-impact-analysis?symbol={symbol}&daysBefore={days_before}&daysAfter={days_after}"
        return self.make_req(url)
    
    def get_split_frequency_analysis(self, symbol: str):
        """Get analysis of stock split frequency and patterns"""
        url = f"https://financialmodelingprep.com/api/v4/split-frequency-analysis?symbol={symbol}"
        return self.make_req(url)
    
    def get_sector_split_trends(self, sector: str, years: int = 5):
        """Get stock split trends for entire sector"""
        url = f"https://financialmodelingprep.com/api/v4/sector-split-trends?sector={sector}&years={years}"
        return self.make_req(url)
    
    def get_industry_split_trends(self, industry: str, years: int = 5):
        """Get stock split trends for entire industry"""
        url = f"https://financialmodelingprep.com/api/v4/industry-split-trends?industry={industry}&years={years}"
        return self.make_req(url)
    
    def get_split_calendar_by_month(self, year: int, month: int):
        """Get stock splits calendar for specific month"""
        url = f"https://financialmodelingprep.com/api/v4/split-calendar-monthly?year={year}&month={month}"
        return self.make_req(url)
    
    def get_split_calendar_by_quarter(self, year: int, quarter: int):
        """Get stock splits calendar for specific quarter"""
        url = f"https://financialmodelingprep.com/api/v4/split-calendar-quarterly?year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_pre_split_ownership(self, symbol: str, split_date: str):
        """Get ownership structure before stock split"""
        url = f"https://financialmodelingprep.com/api/v4/pre-split-ownership?symbol={symbol}&splitDate={split_date}"
        return self.make_req(url)
    
    def get_post_split_ownership(self, symbol: str, split_date: str):
        """Get ownership structure after stock split"""
        url = f"https://financialmodelingprep.com/api/v4/post-split-ownership?symbol={symbol}&splitDate={split_date}"
        return self.make_req(url)
    
    def get_split_eligibility_dates(self, symbol: str, limit: int = 20):
        """Get stock split eligibility and record dates"""
        url = f"https://financialmodelingprep.com/api/v4/split-eligibility-dates?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_split_payable_dates(self, symbol: str, limit: int = 20):
        """Get stock split payable/effective dates"""
        url = f"https://financialmodelingprep.com/api/v4/split-payable-dates?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_fractional_shares_treatment(self, symbol: str, split_date: str):
        """Get information about fractional shares treatment in splits"""
        url = f"https://financialmodelingprep.com/api/v4/fractional-shares-treatment?symbol={symbol}&splitDate={split_date}"
        return self.make_req(url)
    
    def get_split_vs_dividend_analysis(self, symbol: str, years: int = 5):
        """Compare stock splits vs dividend payments strategy"""
        url = f"https://financialmodelingprep.com/api/v4/split-vs-dividend-analysis?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_split_motivation_analysis(self, symbol: str):
        """Analyze company motivations and reasons for stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-motivation-analysis?symbol={symbol}"
        return self.make_req(url)
    
    def get_market_cap_split_impact(self, symbol: str, split_date: str):
        """Analyze market cap impact around stock split dates"""
        url = f"https://financialmodelingprep.com/api/v4/market-cap-split-impact?symbol={symbol}&splitDate={split_date}"
        return self.make_req(url)
    
    def get_institutional_split_response(self, symbol: str, split_date: str, days_around: int = 30):
        """Get institutional investor response to stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-split-response?symbol={symbol}&splitDate={split_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_split_liquidity_impact(self, symbol: str, split_date: str, days_around: int = 30):
        """Analyze liquidity impact of stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-liquidity-impact?symbol={symbol}&splitDate={split_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_options_split_adjustment(self, symbol: str, split_date: str):
        """Get options contract adjustments for stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/options-split-adjustment?symbol={symbol}&splitDate={split_date}"
        return self.make_req(url)
    
    def get_split_tax_implications(self, symbol: str, split_date: str, cost_basis: Optional[float] = None):
        """Get tax implications of stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-tax-implications?symbol={symbol}&splitDate={split_date}"
        
        if cost_basis:
            url += f"&costBasis={cost_basis}"
        
        return self.make_req(url)
    
    def get_split_portfolio_impact(self, symbols: str, portfolio_value: float = 100000.0):
        """Analyze stock splits impact on portfolio (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v4/split-portfolio-impact?symbols={symbols}&portfolioValue={portfolio_value}"
        return self.make_req(url)
    
    def get_most_split_active_stocks(self, years: int = 5, limit: int = 50):
        """Get stocks with most frequent splits in specified years"""
        url = f"https://financialmodelingprep.com/api/v4/most-split-active-stocks?years={years}&limit={limit}"
        return self.make_req(url)
    
    def get_large_split_ratios(self, min_ratio: float = 5.0, years: int = 5, limit: int = 50):
        """Get stocks with large split ratios (e.g., 5:1 or higher)"""
        url = f"https://financialmodelingprep.com/api/v4/large-split-ratios?minRatio={min_ratio}&years={years}&limit={limit}"
        return self.make_req(url)
    
    def get_penny_stock_splits(self, max_price: float = 5.0, years: int = 3, limit: int = 50):
        """Get stock splits of penny stocks"""
        url = f"https://financialmodelingprep.com/api/v4/penny-stock-splits?maxPrice={max_price}&years={years}&limit={limit}"
        return self.make_req(url)
    
    def get_tech_stock_splits(self, years: int = 5, limit: int = 50):
        """Get stock splits in technology sector"""
        url = f"https://financialmodelingprep.com/api/v4/tech-stock-splits?years={years}&limit={limit}"
        return self.make_req(url)
    
    def get_faang_splits_history(self):
        """Get historical stock splits for FAANG stocks"""
        url = "https://financialmodelingprep.com/api/v4/faang-splits-history"
        return self.make_req(url)
    
    def get_split_seasonality(self, month: Optional[int] = None):
        """Get stock split seasonality patterns by month"""
        if month:
            url = f"https://financialmodelingprep.com/api/v4/split-seasonality?month={month}"
        else:
            url = "https://financialmodelingprep.com/api/v4/split-seasonality"
        return self.make_req(url)
    
    def get_split_price_targets_impact(self, symbol: str, split_date: str):
        """Get impact of stock splits on analyst price targets"""
        url = f"https://financialmodelingprep.com/api/v4/split-price-targets-impact?symbol={symbol}&splitDate={split_date}"
        return self.make_req(url)
    
    def get_international_splits(self, country: str, years: int = 5):
        """Get stock splits for specific country"""
        url = f"https://financialmodelingprep.com/api/v4/international-splits?country={country}&years={years}"
        return self.make_req(url)
    
    def get_currency_split_impact(self, symbol: str, split_date: str, base_currency: str = "USD"):
        """Get currency impact on international stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/currency-split-impact?symbol={symbol}&splitDate={split_date}&baseCurrency={base_currency}"
        return self.make_req(url)
    
    def get_split_regulatory_filings(self, symbol: str, split_date: str):
        """Get regulatory filings related to stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-regulatory-filings?symbol={symbol}&splitDate={split_date}"
        return self.make_req(url)
    
    def get_board_split_decisions(self, symbol: str, years: int = 5):
        """Get board of directors decisions on stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/board-split-decisions?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_shareholder_split_approval(self, symbol: str, split_date: str):
        """Get shareholder approval information for stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/shareholder-split-approval?symbol={symbol}&splitDate={split_date}"
        return self.make_req(url)
    
    def get_split_market_reaction_analysis(self, symbol: str, split_date: str, days_around: int = 10):
        """Get detailed market reaction analysis around stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-market-reaction?symbol={symbol}&splitDate={split_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_split_volume_analysis(self, symbol: str, split_date: str, days_around: int = 30):
        """Get trading volume analysis around stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-volume-analysis?symbol={symbol}&splitDate={split_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_split_volatility_analysis(self, symbol: str, split_date: str, days_around: int = 30):
        """Get volatility analysis around stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-volatility-analysis?symbol={symbol}&splitDate={split_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_split_insider_trading(self, symbol: str, split_date: str, days_around: int = 60):
        """Get insider trading activity around stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-insider-trading?symbol={symbol}&splitDate={split_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_split_short_interest(self, symbol: str, split_date: str, days_around: int = 30):
        """Get short interest changes around stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-short-interest?symbol={symbol}&splitDate={split_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_split_analyst_coverage(self, symbol: str, split_date: str):
        """Get analyst coverage changes around stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-analyst-coverage?symbol={symbol}&splitDate={split_date}"
        return self.make_req(url)
    
    def get_split_media_coverage(self, symbol: str, split_date: str, days_around: int = 7):
        """Get media coverage and sentiment around stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-media-coverage?symbol={symbol}&splitDate={split_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_split_social_sentiment(self, symbol: str, split_date: str, days_around: int = 7):
        """Get social media sentiment around stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-social-sentiment?symbol={symbol}&splitDate={split_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_split_correlation_analysis(self, symbol: str, years: int = 10):
        """Analyze correlation between stock splits and performance"""
        url = f"https://financialmodelingprep.com/api/v4/split-correlation-analysis?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_split_success_metrics(self, symbol: str):
        """Get success metrics and effectiveness of past stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-success-metrics?symbol={symbol}"
        return self.make_req(url)
    
    def get_split_peer_comparison(self, symbol: str, years: int = 5):
        """Compare stock split activity with industry peers"""
        url = f"https://financialmodelingprep.com/api/v4/split-peer-comparison?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_split_timing_analysis(self, symbol: str):
        """Analyze timing patterns of company's stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-timing-analysis?symbol={symbol}"
        return self.make_req(url)
    
    def get_split_forecasting_model(self, symbol: str):
        """Get AI-based forecasting for potential future stock splits"""
        url = f"https://financialmodelingprep.com/api/v4/split-forecasting-model?symbol={symbol}"
        return self.make_req(url)
    
    def get_stock_buybacks_vs_splits(self, symbol: str, years: int = 5):
        """Compare stock buybacks vs stock splits strategy"""
        url = f"https://financialmodelingprep.com/api/v4/buybacks-vs-splits?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_split_rss_feed(self, page: int = 0):
        """Get RSS feed of latest stock split announcements"""
        url = f"https://financialmodelingprep.com/api/v4/stock-splits-rss-feed?page={page}"
        return self.make_req(url)
    
    def get_split_alerts(self, symbols: str):
        """Get alerts for upcoming stock splits (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v4/stock-splits-alerts?symbols={symbols}"
        return self.make_req(url)
    
    def get_split_export(self, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None, format: str = "csv"):
        """Export stock splits data (csv, excel, json)"""
        url = f"https://financialmodelingprep.com/api/v4/stock-splits-export?format={format}"
        
        if symbol:
            url += f"&symbol={symbol}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)

    # ===== MARKET PERFORMANCE & EARNINGS-RELATED MOVERS SECTION =====
    
    def get_market_gainers(self, limit: int = 50):
        """Get biggest stock gainers for the day"""
        url = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?limit={limit}"
        return self.make_req(url)
    
    def get_market_losers(self, limit: int = 50):
        """Get biggest stock losers for the day"""
        url = f"https://financialmodelingprep.com/api/v3/stock_market/losers?limit={limit}"
        return self.make_req(url)
    
    def get_most_active_stocks(self, limit: int = 50):
        """Get most actively traded stocks by volume"""
        url = f"https://financialmodelingprep.com/api/v3/stock_market/actives?limit={limit}"
        return self.make_req(url)
    
    def get_biggest_earnings_gainers(self, days: int = 1):
        """Get stocks with biggest gains after earnings announcements"""
        url = f"https://financialmodelingprep.com/api/v4/biggest-earnings-gainers?days={days}"
        return self.make_req(url)
    
    def get_biggest_earnings_losers(self, days: int = 1):
        """Get stocks with biggest losses after earnings announcements"""
        url = f"https://financialmodelingprep.com/api/v4/biggest-earnings-losers?days={days}"
        return self.make_req(url)
    
    def get_earnings_movers(self, direction: str = "both", min_move: float = 5.0):
        """Get stocks that moved significantly on earnings (direction: up, down, both)"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-movers?direction={direction}&minMove={min_move}"
        return self.make_req(url)
    
    def get_pre_earnings_movers(self, days_before: int = 3):
        """Get stocks moving in anticipation of earnings"""
        url = f"https://financialmodelingprep.com/api/v4/pre-earnings-movers?daysBefore={days_before}"
        return self.make_req(url)
    
    def get_post_earnings_drift(self, symbol: str, days_after: int = 30):
        """Get post-earnings announcement drift analysis"""
        url = f"https://financialmodelingprep.com/api/v4/post-earnings-drift?symbol={symbol}&daysAfter={days_after}"
        return self.make_req(url)
    
    def get_earnings_reaction_analysis(self, from_date: str, to_date: str, min_surprise: float = 5.0):
        """Get analysis of market reactions to earnings surprises"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-reaction-analysis?from={from_date}&to={to_date}&minSurprise={min_surprise}"
        return self.make_req(url)
    
    def get_sector_performance(self, date: Optional[str] = None, sector: Optional[str] = None):
        """Get sector performance for a specific date or today, optionally filtered by sector"""
        if date:
            url = f"https://financialmodelingprep.com/api/v3/sector-performance?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v3/sector-performance"
        
        # Note: FMP API doesn't support sector filtering in this endpoint
        # If sector is specified, we'll filter the results after receiving them
        result = self.make_req(url)
        
        if sector and result:
            # Filter results to only include the specified sector
            if isinstance(result, list):
                result = [item for item in result if item.get('sector', '').lower() == sector.lower()]
            
        return result
    
    def get_historical_sector_performance(self, limit: int = 50):
        """Get historical sector performance data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-sectors-performance?limit={limit}"
        return self.make_req(url)
    
    def get_industry_pe_ratio(self, date: Optional[str] = None):
        """Get price-to-earnings ratios by industry"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/industry_price_earning_ratio?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v4/industry_price_earning_ratio"
        return self.make_req(url)
    
    def get_market_risk_premium(self, country: str = "US"):
        """Get market risk premium for country"""
        url = f"https://financialmodelingprep.com/api/v4/market_risk_premium?country={country}"
        return self.make_req(url)
    
    def get_commitment_of_traders_report(self, symbol: str):
        """Get Commitment of Traders (COT) report analysis"""
        url = f"https://financialmodelingprep.com/api/v4/commitment_of_traders_report/{symbol}"
        return self.make_req(url)
    
    def get_commitment_of_traders_analysis(self, symbol: str):
        """Get analysis of Commitment of Traders report"""
        url = f"https://financialmodelingprep.com/api/v4/commitment_of_traders_report_analysis/{symbol}"
        return self.make_req(url)
    
    def get_earnings_trading_strategies(self, symbol: str):
        """Get earnings-based trading strategies and recommendations"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-trading-strategies?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_volatility_forecast(self, symbol: str, days_ahead: int = 30):
        """Get earnings volatility forecast"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-volatility-forecast?symbol={symbol}&daysAhead={days_ahead}"
        return self.make_req(url)
    
    def get_earnings_calendar_impact(self, from_date: str, to_date: str):
        """Get expected market impact from upcoming earnings"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-calendar-impact?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_after_hours_movers(self, date: Optional[str] = None):
        """Get after-hours stock movers"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/after-hours-movers?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v4/after-hours-movers"
        return self.make_req(url)
    
    def get_pre_market_movers(self, date: Optional[str] = None):
        """Get pre-market stock movers"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/pre-market-movers?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v4/pre-market-movers"
        return self.make_req(url)
    
    def get_unusual_options_activity(self, symbol: Optional[str] = None, date: Optional[str] = None):
        """Get unusual options activity around earnings"""
        url = "https://financialmodelingprep.com/api/v4/unusual-options-activity?"
        
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if date:
            params.append(f"date={date}")
        
        if params:
            url += "&".join(params)
        
        return self.make_req(url)
    
    def get_earnings_options_flow(self, symbol: str, days_around: int = 5):
        """Get options flow around earnings announcements"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-options-flow?symbol={symbol}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_market_fear_greed_index(self):
        """Get market fear and greed index"""
        url = "https://financialmodelingprep.com/api/v4/market-fear-greed-index"
        return self.make_req(url)
    
    def get_market_sentiment_indicators(self, date: Optional[str] = None):
        """Get market sentiment indicators"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/market-sentiment-indicators?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v4/market-sentiment-indicators"
        return self.make_req(url)
    
    def get_earnings_week_performance(self, year: int, week: int):
        """Get market performance during major earnings weeks"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-week-performance?year={year}&week={week}"
        return self.make_req(url)
    
    def get_earnings_season_calendar(self, year: int, quarter: int):
        """Get earnings season overview and key dates"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-season-calendar?year={year}&quarter={quarter}"
        return self.make_req(url)
    
    def get_sector_earnings_trends(self, sector: str, quarters: int = 4):
        """Get earnings trends for entire sector"""
        url = f"https://financialmodelingprep.com/api/v4/sector-earnings-trends?sector={sector}&quarters={quarters}"
        return self.make_req(url)
    
    def get_market_breadth_indicators(self, date: Optional[str] = None):
        """Get market breadth indicators and advance/decline metrics"""
        if date:
            url = f"https://financialmodelingprep.com/api/v4/market-breadth-indicators?date={date}"
        else:
            url = "https://financialmodelingprep.com/api/v4/market-breadth-indicators"
        return self.make_req(url)
    
    def get_earnings_beat_rate_by_sector(self, quarter: str, year: int):
        """Get earnings beat rates by sector for specific quarter"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-beat-rate-sector?quarter={quarter}&year={year}"
        return self.make_req(url)
    
    def get_institutional_flow_earnings(self, symbol: str, days_around: int = 10):
        """Get institutional money flow around earnings"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-flow-earnings?symbol={symbol}&daysAround={days_around}"
        return self.make_req(url)

    # ===== MERGERS & ACQUISITIONS SECTION =====
    
    def get_mergers_acquisitions_rss_feed(self, page: int = 0):
        """Get RSS feed of M&A news and announcements"""
        url = f"https://financialmodelingprep.com/api/v4/mergers-acquisitions-rss-feed?page={page}"
        return self.make_req(url)
    
    # Core M&A endpoints matching documentation exactly
    def get_ma_rss_feed(self, page: int = 0):
        """Get M&A RSS feed - provides real-time stream of M&A news and announcements"""
        url = f"https://financialmodelingprep.com/api/v4/mergers-acquisitions-rss-feed?page={page}"
        return self.make_req(url)
    
    def search_ma_deals(self, name: str):
        """Search M&A deals by company name"""
        url = f"https://financialmodelingprep.com/api/v4/mergers-acquisitions/search?name={name}"
        return self.make_req(url)
    
    def search_mergers_acquisitions(self, name: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Search for M&A deals based on company name and date range"""
        url = f"https://financialmodelingprep.com/api/v4/mergers-acquisitions/search?limit={limit}"
        
        if name:
            url += f"&name={name}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ma_deals_by_symbol(self, symbol: str, limit: int = 50):
        """Get M&A deals involving a specific company symbol"""
        url = f"https://financialmodelingprep.com/api/v4/ma-deals-by-symbol?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_recent_ma_activity(self, days: int = 30, limit: int = 50):
        """Get recent M&A activity in the last specified days"""
        url = f"https://financialmodelingprep.com/api/v4/recent-ma-activity?days={days}&limit={limit}"
        return self.make_req(url)
    
    def get_largest_ma_deals(self, year: Optional[int] = None, limit: int = 50):
        """Get largest M&A deals by transaction value"""
        url = f"https://financialmodelingprep.com/api/v4/largest-ma-deals?limit={limit}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_ma_deals_by_sector(self, sector: str, year: Optional[int] = None, limit: int = 50):
        """Get M&A deals for a specific sector"""
        url = f"https://financialmodelingprep.com/api/v4/ma-deals-by-sector?sector={sector}&limit={limit}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_ma_deals_by_industry(self, industry: str, year: Optional[int] = None, limit: int = 50):
        """Get M&A deals for a specific industry"""
        url = f"https://financialmodelingprep.com/api/v4/ma-deals-by-industry?industry={industry}&limit={limit}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_ma_calendar(self, from_date: str, to_date: str):
        """Get M&A calendar showing announced and expected deals"""
        url = f"https://financialmodelingprep.com/api/v4/ma-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_pending_ma_deals(self, limit: int = 50):
        """Get pending M&A deals awaiting regulatory approval or completion"""
        url = f"https://financialmodelingprep.com/api/v4/pending-ma-deals?limit={limit}"
        return self.make_req(url)
    
    def get_completed_ma_deals(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get completed M&A deals"""
        url = f"https://financialmodelingprep.com/api/v4/completed-ma-deals?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_failed_ma_deals(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get failed or terminated M&A deals"""
        url = f"https://financialmodelingprep.com/api/v4/failed-ma-deals?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ma_deal_details(self, deal_id: str):
        """Get detailed information about a specific M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-deal-details?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_deal_timeline(self, deal_id: str):
        """Get timeline and milestones for a specific M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-deal-timeline?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_deal_documents(self, deal_id: str):
        """Get SEC filings and documents related to M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-deal-documents?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_deal_valuation(self, deal_id: str):
        """Get valuation metrics and multiples for M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-deal-valuation?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_deal_synergies(self, deal_id: str):
        """Get expected synergies and cost savings from M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-deal-synergies?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_market_impact(self, symbol: str, announcement_date: str, days_around: int = 10):
        """Get market impact analysis around M&A announcement"""
        url = f"https://financialmodelingprep.com/api/v4/ma-market-impact?symbol={symbol}&announcementDate={announcement_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_acquisition_premiums(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get acquisition premiums paid in M&A deals"""
        url = f"https://financialmodelingprep.com/api/v4/acquisition-premiums?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ma_payment_methods(self, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get analysis of M&A payment methods (cash, stock, mixed)"""
        url = "https://financialmodelingprep.com/api/v4/ma-payment-methods?"
        
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_hostile_takeovers(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get hostile takeover attempts and outcomes"""
        url = f"https://financialmodelingprep.com/api/v4/hostile-takeovers?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_leveraged_buyouts(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get leveraged buyout (LBO) transactions"""
        url = f"https://financialmodelingprep.com/api/v4/leveraged-buyouts?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_management_buyouts(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get management buyout (MBO) transactions"""
        url = f"https://financialmodelingprep.com/api/v4/management-buyouts?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_spin_offs(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get corporate spin-off transactions"""
        url = f"https://financialmodelingprep.com/api/v4/spin-offs?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_asset_sales(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get major asset sales and divestitures"""
        url = f"https://financialmodelingprep.com/api/v4/asset-sales?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_joint_ventures(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get joint venture announcements and partnerships"""
        url = f"https://financialmodelingprep.com/api/v4/joint-ventures?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_private_equity_deals(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get private equity investment deals"""
        url = f"https://financialmodelingprep.com/api/v4/private-equity-deals?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_venture_capital_deals(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get venture capital investment deals"""
        url = f"https://financialmodelingprep.com/api/v4/venture-capital-deals?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ma_trends_analysis(self, period: str = "quarterly", years: int = 5):
        """Get M&A trends and market analysis (period: monthly, quarterly, yearly)"""
        url = f"https://financialmodelingprep.com/api/v4/ma-trends-analysis?period={period}&years={years}"
        return self.make_req(url)
    
    def get_ma_multiples_analysis(self, sector: Optional[str] = None, year: Optional[int] = None):
        """Get analysis of M&A valuation multiples"""
        url = "https://financialmodelingprep.com/api/v4/ma-multiples-analysis?"
        
        params = []
        if sector:
            params.append(f"sector={sector}")
        if year:
            params.append(f"year={year}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_cross_border_ma(self, country: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get cross-border M&A transactions"""
        url = f"https://financialmodelingprep.com/api/v4/cross-border-ma?limit={limit}"
        
        if country:
            url += f"&country={country}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ma_regulatory_approvals(self, deal_id: Optional[str] = None, pending_only: bool = True):
        """Get regulatory approval status for M&A deals"""
        url = f"https://financialmodelingprep.com/api/v4/ma-regulatory-approvals?pendingOnly={str(pending_only).lower()}"
        
        if deal_id:
            url += f"&dealId={deal_id}"
        
        return self.make_req(url)
    
    def get_antitrust_investigations(self, status: str = "active", limit: int = 50):
        """Get antitrust investigations and reviews (status: active, completed, all)"""
        url = f"https://financialmodelingprep.com/api/v4/antitrust-investigations?status={status}&limit={limit}"
        return self.make_req(url)
    
    def get_ma_advisor_rankings(self, year: Optional[int] = None, advisor_type: str = "financial"):
        """Get M&A advisor rankings by deal value (advisor_type: financial, legal)"""
        url = f"https://financialmodelingprep.com/api/v4/ma-advisor-rankings?advisorType={advisor_type}"
        
        if year:
            url += f"&year={year}"
        
        return self.make_req(url)
    
    def get_ma_financing_sources(self, deal_id: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get M&A financing sources and structures"""
        url = "https://financialmodelingprep.com/api/v4/ma-financing-sources?"
        
        params = []
        if deal_id:
            params.append(f"dealId={deal_id}")
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_ma_due_diligence_timeline(self, deal_id: str):
        """Get due diligence timeline and milestones for M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-due-diligence?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_integration_plans(self, deal_id: str):
        """Get post-merger integration plans and timelines"""
        url = f"https://financialmodelingprep.com/api/v4/ma-integration-plans?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_success_metrics(self, deal_id: Optional[str] = None, years_post: int = 3):
        """Get M&A success metrics and post-deal performance"""
        url = f"https://financialmodelingprep.com/api/v4/ma-success-metrics?yearsPost={years_post}"
        
        if deal_id:
            url += f"&dealId={deal_id}"
        
        return self.make_req(url)
    
    def get_ma_shareholder_votes(self, deal_id: str):
        """Get shareholder voting results for M&A deals"""
        url = f"https://financialmodelingprep.com/api/v4/ma-shareholder-votes?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_board_approvals(self, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get board of directors approvals for M&A deals"""
        url = "https://financialmodelingprep.com/api/v4/ma-board-approvals?"
        
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_ma_breakup_fees(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get M&A breakup fees and termination costs"""
        url = f"https://financialmodelingprep.com/api/v4/ma-breakup-fees?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ma_collar_structures(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get M&A collar structures and price protection mechanisms"""
        url = f"https://financialmodelingprep.com/api/v4/ma-collar-structures?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ma_earnout_provisions(self, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50):
        """Get M&A earnout provisions and contingent value rights"""
        url = f"https://financialmodelingprep.com/api/v4/ma-earnout-provisions?limit={limit}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        
        return self.make_req(url)
    
    def get_ma_tax_implications(self, deal_id: Optional[str] = None, deal_structure: Optional[str] = None):
        """Get tax implications and structures of M&A deals"""
        url = "https://financialmodelingprep.com/api/v4/ma-tax-implications?"
        
        params = []
        if deal_id:
            params.append(f"dealId={deal_id}")
        if deal_structure:  # stock, asset, merger, etc.
            params.append(f"structure={deal_structure}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_ma_employee_impact(self, deal_id: str):
        """Get employee impact analysis for M&A deals (layoffs, retention, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/ma-employee-impact?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_customer_impact(self, deal_id: str):
        """Get customer impact and market concentration analysis"""
        url = f"https://financialmodelingprep.com/api/v4/ma-customer-impact?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_competitive_analysis(self, deal_id: str):
        """Get competitive landscape analysis for M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-competitive-analysis?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_industry_consolidation(self, industry: str, years: int = 5):
        """Get industry consolidation trends and concentration metrics"""
        url = f"https://financialmodelingprep.com/api/v4/ma-industry-consolidation?industry={industry}&years={years}"
        return self.make_req(url)
    
    def get_ma_deal_rationale(self, deal_id: str):
        """Get strategic rationale and business case for M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-deal-rationale?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_market_reaction_analysis(self, symbol: str, announcement_date: str, days_around: int = 5):
        """Get detailed market reaction analysis for M&A announcement"""
        url = f"https://financialmodelingprep.com/api/v4/ma-market-reaction?symbol={symbol}&announcementDate={announcement_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_ma_arbitrage_opportunities(self, status: str = "pending", min_spread: float = 2.0):
        """Get merger arbitrage opportunities and spreads"""
        url = f"https://financialmodelingprep.com/api/v4/ma-arbitrage-opportunities?status={status}&minSpread={min_spread}"
        return self.make_req(url)
    
    def get_ma_risk_factors(self, deal_id: str):
        """Get risk factors and potential deal breakers for M&A transaction"""
        url = f"https://financialmodelingprep.com/api/v4/ma-risk-factors?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_completion_probability(self, deal_id: str):
        """Get AI-based completion probability analysis for pending M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-completion-probability?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_peer_comparisons(self, deal_id: str):
        """Get peer transaction comparisons and precedent analysis"""
        url = f"https://financialmodelingprep.com/api/v4/ma-peer-comparisons?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_accretion_dilution(self, deal_id: str):
        """Get accretion/dilution analysis for M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-accretion-dilution?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_pro_forma_financials(self, deal_id: str):
        """Get pro forma financial statements for combined entity"""
        url = f"https://financialmodelingprep.com/api/v4/ma-pro-forma-financials?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_credit_rating_impact(self, deal_id: str):
        """Get credit rating impact and debt analysis for M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-credit-rating-impact?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_currency_hedging(self, deal_id: str):
        """Get currency hedging strategies for cross-border M&A"""
        url = f"https://financialmodelingprep.com/api/v4/ma-currency-hedging?dealId={deal_id}"
        return self.make_req(url)
    
    def get_ma_activist_involvement(self, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get activist investor involvement in M&A situations"""
        url = "https://financialmodelingprep.com/api/v4/ma-activist-involvement?"
        
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_ma_poison_pills(self, symbol: Optional[str] = None, status: str = "active"):
        """Get poison pill and takeover defense mechanisms"""
        url = f"https://financialmodelingprep.com/api/v4/ma-poison-pills?status={status}"
        
        if symbol:
            url += f"&symbol={symbol}"
        
        return self.make_req(url)
    
    def get_ma_proxy_contests(self, symbol: Optional[str] = None, year: Optional[int] = None):
        """Get proxy contests related to M&A situations"""
        url = "https://financialmodelingprep.com/api/v4/ma-proxy-contests?"
        
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if year:
            params.append(f"year={year}")
        
        if params:
            url += "&".join(params)
        else:
            url = url.rstrip("?")
        
        return self.make_req(url)
    
    def get_ma_litigation(self, deal_id: Optional[str] = None, status: str = "all"):
        """Get litigation and legal challenges related to M&A deals"""
        url = f"https://financialmodelingprep.com/api/v4/ma-litigation?status={status}"
        
        if deal_id:
            url += f"&dealId={deal_id}"
        
        return self.make_req(url)
    
    def get_ma_insider_trading(self, symbol: str, announcement_date: str, days_before: int = 60):
        """Get insider trading activity before M&A announcements"""
        url = f"https://financialmodelingprep.com/api/v4/ma-insider-trading?symbol={symbol}&announcementDate={announcement_date}&daysBefore={days_before}"
        return self.make_req(url)
    
    def get_ma_options_activity(self, symbol: str, announcement_date: str, days_before: int = 30):
        """Get unusual options activity before M&A announcements"""
        url = f"https://financialmodelingprep.com/api/v4/ma-options-activity?symbol={symbol}&announcementDate={announcement_date}&daysBefore={days_before}"
        return self.make_req(url)
    
    def get_ma_short_interest(self, symbol: str, announcement_date: str, days_around: int = 30):
        """Get short interest changes around M&A announcements"""
        url = f"https://financialmodelingprep.com/api/v4/ma-short-interest?symbol={symbol}&announcementDate={announcement_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_ma_institutional_flow(self, symbol: str, announcement_date: str, days_around: int = 30):
        """Get institutional money flow around M&A announcements"""
        url = f"https://financialmodelingprep.com/api/v4/ma-institutional-flow?symbol={symbol}&announcementDate={announcement_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_ma_sentiment_analysis(self, deal_id: str, days_around: int = 14):
        """Get sentiment analysis from news and social media around M&A"""
        url = f"https://financialmodelingprep.com/api/v4/ma-sentiment-analysis?dealId={deal_id}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_ma_media_coverage(self, deal_id: str, days_around: int = 14):
        """Get media coverage analysis for M&A deal"""
        url = f"https://financialmodelingprep.com/api/v4/ma-media-coverage?dealId={deal_id}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_ma_social_media_buzz(self, symbol: str, announcement_date: str, days_around: int = 7):
        """Get social media buzz and sentiment around M&A announcement"""
        url = f"https://financialmodelingprep.com/api/v4/ma-social-media-buzz?symbol={symbol}&announcementDate={announcement_date}&daysAround={days_around}"
        return self.make_req(url)
    
    def get_ma_analyst_reactions(self, symbol: str, announcement_date: str):
        """Get analyst reactions and rating changes after M&A announcement"""
        url = f"https://financialmodelingprep.com/api/v4/ma-analyst-reactions?symbol={symbol}&announcementDate={announcement_date}"
        return self.make_req(url)
    
    def get_ma_price_target_impact(self, symbol: str, announcement_date: str):
        """Get price target changes after M&A announcement"""
        url = f"https://financialmodelingprep.com/api/v4/ma-price-target-impact?symbol={symbol}&announcementDate={announcement_date}"
        return self.make_req(url)
    
    def get_ma_calendar_alerts(self, symbols: str):
        """Get alerts for M&A calendar updates (comma-separated symbols)"""
        url = f"https://financialmodelingprep.com/api/v4/ma-calendar-alerts?symbols={symbols}"
        return self.make_req(url)
    
    def get_ma_rumor_tracker(self, symbol: Optional[str] = None, credibility_score: float = 0.5):
        """Track M&A rumors and speculation (credibility_score: 0.0-1.0)"""
        url = f"https://financialmodelingprep.com/api/v4/ma-rumor-tracker?credibilityScore={credibility_score}"
        
        if symbol:
            url += f"&symbol={symbol}"
        
        return self.make_req(url)
    
    def get_ma_deal_leaks(self, days: int = 30):
        """Get analysis of deal leaks and information flow"""
        url = f"https://financialmodelingprep.com/api/v4/ma-deal-leaks?days={days}"
        return self.make_req(url)
    
    def get_ma_export(self, from_date: Optional[str] = None, to_date: Optional[str] = None, sector: Optional[str] = None, format: str = "csv"):
        """Export M&A data (csv, excel, json)"""
        url = f"https://financialmodelingprep.com/api/v4/ma-export?format={format}"
        
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        if sector:
            url += f"&sector={sector}"
        
        return self.make_req(url)
    
    def get_ma_api_limits(self):
        """Get API usage limits and remaining calls for M&A endpoints"""
        url = "https://financialmodelingprep.com/api/v4/ma-api-limits"
        return self.make_req(url)
    
    
    # ===== CHARTS & TECHNICAL ANALYSIS SECTION =====
    
    def get_historical_chart_1min(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get 1-minute historical price chart data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{symbol}"
        if from_date and to_date:
            url += f"?from={from_date}&to={to_date}"
        elif from_date:
            url += f"?from={from_date}"
        elif to_date:
            url += f"?to={to_date}"
        return self.make_req(url)
    
    def get_historical_chart_5min(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get 5-minute historical price chart data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/5min/{symbol}"
        if from_date and to_date:
            url += f"?from={from_date}&to={to_date}"
        elif from_date:
            url += f"?from={from_date}"
        elif to_date:
            url += f"?to={to_date}"
        return self.make_req(url)
    
    def get_historical_chart_15min(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get 15-minute historical price chart data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/15min/{symbol}"
        if from_date and to_date:
            url += f"?from={from_date}&to={to_date}"
        elif from_date:
            url += f"?from={from_date}"
        elif to_date:
            url += f"?to={to_date}"
        return self.make_req(url)
    
    def get_historical_chart_30min(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get 30-minute historical price chart data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/30min/{symbol}"
        if from_date and to_date:
            url += f"?from={from_date}&to={to_date}"
        elif from_date:
            url += f"?from={from_date}"
        elif to_date:
            url += f"?to={to_date}"
        return self.make_req(url)
    
    def get_historical_chart_1hour(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get 1-hour historical price chart data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1hour/{symbol}"
        if from_date and to_date:
            url += f"?from={from_date}&to={to_date}"
        elif from_date:
            url += f"?from={from_date}"
        elif to_date:
            url += f"?to={to_date}"
        return self.make_req(url)
    
    def get_historical_chart_4hour(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get 4-hour historical price chart data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/4hour/{symbol}"
        if from_date and to_date:
            url += f"?from={from_date}&to={to_date}"
        elif from_date:
            url += f"?from={from_date}"
        elif to_date:
            url += f"?to={to_date}"
        return self.make_req(url)
    
    def get_historical_chart_daily(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 1000):
        """Get daily historical price chart data"""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?limit={limit}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        return self.make_req(url)
    
    def get_chart_with_interval(self, symbol: str, interval: str = "1day", from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get chart data with custom interval (1min, 5min, 15min, 30min, 1hour, 4hour, 1day)"""
        if interval == "1day":
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        else:
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{symbol}"
        
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "?" + "&".join(params)
        
        return self.make_req(url)
    
    def get_technical_indicator_sma(self, symbol: str, period: int = 20):
        """Get Simple Moving Average (SMA) technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=sma&period={period}"
        return self.make_req(url)
    
    def get_technical_indicator_ema(self, symbol: str, period: int = 20):
        """Get Exponential Moving Average (EMA) technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=ema&period={period}"
        return self.make_req(url)
    
    def get_technical_indicator_wma(self, symbol: str, period: int = 20):
        """Get Weighted Moving Average (WMA) technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=wma&period={period}"
        return self.make_req(url)
    
    def get_technical_indicator_dema(self, symbol: str, period: int = 20):
        """Get Double Exponential Moving Average (DEMA) technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=dema&period={period}"
        return self.make_req(url)
    
    def get_technical_indicator_tema(self, symbol: str, period: int = 20):
        """Get Triple Exponential Moving Average (TEMA) technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=tema&period={period}"
        return self.make_req(url)
    
    def get_technical_indicator_williams(self, symbol: str, period: int = 14):
        """Get Williams %R technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=williams&period={period}"
        return self.make_req(url)
    
    def get_technical_indicator_rsi(self, symbol: str, period: int = 14):
        """Get Relative Strength Index (RSI) technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=rsi&period={period}"
        return self.make_req(url)
    
    def get_technical_indicator_adx(self, symbol: str, period: int = 14):
        """Get Average Directional Index (ADX) technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=adx&period={period}"
        return self.make_req(url)
    
    def get_technical_indicator_standard_deviation(self, symbol: str, period: int = 20):
        """Get Standard Deviation technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=standardDeviation&period={period}"
        return self.make_req(url)
    
    def get_technical_indicator_macd(self, symbol: str):
        """Get MACD (Moving Average Convergence Divergence) technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=macd"
        return self.make_req(url)
    
    def get_technical_indicator_bollinger_bands(self, symbol: str, period: int = 20):
        """Get Bollinger Bands technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=bollinger&period={period}"
        return self.make_req(url)
    
    def get_technical_indicator_stochastic(self, symbol: str, k_period: int = 14, d_period: int = 3):
        """Get Stochastic Oscillator technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=stoch&kPeriod={k_period}&dPeriod={d_period}"
        return self.make_req(url)
    
    def get_technical_indicator_cci(self, symbol: str, period: int = 14):
        """Get Commodity Channel Index (CCI) technical indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/daily/{symbol}?type=cci&period={period}"
        return self.make_req(url)
    
    def get_intraday_chart(self, symbol: str, interval: str = "1min", limit: int = 1000):
        """Get intraday chart data (1min, 5min, 15min, 30min, 1hour)"""
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{symbol}?limit={limit}"
        return self.make_req(url)
    
    def get_real_time_chart(self, symbol: str):
        """Get real-time chart data and current price"""
        url = f"https://financialmodelingprep.com/api/v3/quote-short/{symbol}"
        return self.make_req(url)
    
    def get_volume_chart(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get volume chart data with price"""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        if from_date and to_date:
            url += f"?from={from_date}&to={to_date}"
        elif from_date:
            url += f"?from={from_date}"
        elif to_date:
            url += f"?to={to_date}"
        return self.make_req(url)
    
    def get_candlestick_data(self, symbol: str, interval: str = "1day", from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get candlestick (OHLC) chart data"""
        if interval == "1day":
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        else:
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{symbol}"
        
        params = []
        if from_date:
            params.append(f"from={from_date}")
        if to_date:
            params.append(f"to={to_date}")
        
        if params:
            url += "?" + "&".join(params)
        
        return self.make_req(url)
    
    def get_price_chart_comparison(self, symbols: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Compare price charts of multiple symbols (comma-separated)"""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbols}"
        if from_date and to_date:
            url += f"?from={from_date}&to={to_date}"
        elif from_date:
            url += f"?from={from_date}"
        elif to_date:
            url += f"?to={to_date}"
        return self.make_req(url)
    
    def get_support_resistance_levels(self, symbol: str, period: int = 50):
        """Get support and resistance levels for charting"""
        url = f"https://financialmodelingprep.com/api/v4/support-resistance?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_pivot_points(self, symbol: str, period: str = "daily"):
        """Get pivot points for technical analysis (daily, weekly, monthly)"""
        url = f"https://financialmodelingprep.com/api/v4/pivot-points?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_fibonacci_retracement(self, symbol: str, high_date: str, low_date: str):
        """Get Fibonacci retracement levels"""
        url = f"https://financialmodelingprep.com/api/v4/fibonacci-retracement?symbol={symbol}&highDate={high_date}&lowDate={low_date}"
        return self.make_req(url)
    
    def get_chart_patterns(self, symbol: str, pattern_type: str = "all"):
        """Get chart patterns (head_and_shoulders, double_top, triangle, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/chart-patterns?symbol={symbol}&pattern={pattern_type}"
        return self.make_req(url)
    
    def get_trend_lines(self, symbol: str, period: int = 30):
        """Get trend lines for technical analysis"""
        url = f"https://financialmodelingprep.com/api/v4/trend-lines?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_volume_profile(self, symbol: str, from_date: str, to_date: str):
        """Get volume profile analysis"""
        url = f"https://financialmodelingprep.com/api/v4/volume-profile?symbol={symbol}&from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_price_action_signals(self, symbol: str, signal_type: str = "all"):
        """Get price action trading signals (breakout, reversal, continuation)"""
        url = f"https://financialmodelingprep.com/api/v4/price-action-signals?symbol={symbol}&signalType={signal_type}"
        return self.make_req(url)
    
    def get_market_structure(self, symbol: str, timeframe: str = "daily"):
        """Get market structure analysis (higher highs, lower lows, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/market-structure?symbol={symbol}&timeframe={timeframe}"
        return self.make_req(url)
    
    def get_volatility_chart(self, symbol: str, period: int = 30):
        """Get historical volatility chart data"""
        url = f"https://financialmodelingprep.com/api/v4/historical-volatility?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_momentum_indicators(self, symbol: str, indicator: str = "all"):
        """Get momentum indicators (RSI, MACD, Stochastic, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/momentum-indicators?symbol={symbol}&indicator={indicator}"
        return self.make_req(url)
    
    def get_oscillators(self, symbol: str, oscillator: str = "all"):
        """Get oscillator indicators (Williams %R, CCI, ROC, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/oscillators?symbol={symbol}&oscillator={oscillator}"
        return self.make_req(url)
    
    def get_moving_average_signals(self, symbol: str, ma_type: str = "sma", periods: str = "20,50,200"):
        """Get moving average crossover signals"""
        url = f"https://financialmodelingprep.com/api/v4/ma-signals?symbol={symbol}&maType={ma_type}&periods={periods}"
        return self.make_req(url)
    
    def get_ichimoku_cloud(self, symbol: str):
        """Get Ichimoku Cloud indicator data"""
        url = f"https://financialmodelingprep.com/api/v4/ichimoku-cloud?symbol={symbol}"
        return self.make_req(url)
    
    def get_parabolic_sar(self, symbol: str, acceleration: float = 0.02, maximum: float = 0.2):
        """Get Parabolic SAR indicator"""
        url = f"https://financialmodelingprep.com/api/v4/parabolic-sar?symbol={symbol}&acceleration={acceleration}&maximum={maximum}"
        return self.make_req(url)
    
    def get_candlestick_patterns(self, symbol: str, pattern: str = "all"):
        """Get candlestick patterns (doji, hammer, engulfing, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/candlestick-patterns?symbol={symbol}&pattern={pattern}"
        return self.make_req(url)
    
    def get_chart_alerts(self, symbols: str, alert_type: str = "breakout"):
        """Get chart-based alerts (breakout, support_break, resistance_break)"""
        url = f"https://financialmodelingprep.com/api/v4/chart-alerts?symbols={symbols}&alertType={alert_type}"
        return self.make_req(url)
    
    def get_technical_summary(self, symbol: str):
        """Get comprehensive technical analysis summary"""
        url = f"https://financialmodelingprep.com/api/v4/technical-summary?symbol={symbol}"
        return self.make_req(url)
    
    def get_price_targets_chart(self, symbol: str):
        """Get price targets overlaid on chart data"""
        url = f"https://financialmodelingprep.com/api/v4/price-targets-chart?symbol={symbol}"
        return self.make_req(url)
    
    def get_earnings_impact_chart(self, symbol: str, quarters: int = 8):
        """Get earnings impact on price chart"""
        url = f"https://financialmodelingprep.com/api/v4/earnings-impact-chart?symbol={symbol}&quarters={quarters}"
        return self.make_req(url)
    
    def get_dividend_impact_chart(self, symbol: str, years: int = 3):
        """Get dividend payments impact on price chart"""
        url = f"https://financialmodelingprep.com/api/v4/dividend-impact-chart?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_splits_impact_chart(self, symbol: str, years: int = 5):
        """Get stock splits impact on price chart"""
        url = f"https://financialmodelingprep.com/api/v4/splits-impact-chart?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_institutional_flow_chart(self, symbol: str, quarters: int = 4):
        """Get institutional money flow chart"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-flow-chart?symbol={symbol}&quarters={quarters}"
        return self.make_req(url)
    
    def get_insider_trading_chart(self, symbol: str, months: int = 12):
        """Get insider trading activity chart"""
        url = f"https://financialmodelingprep.com/api/v4/insider-trading-chart?symbol={symbol}&months={months}"
        return self.make_req(url)
    
    def get_short_interest_chart(self, symbol: str, months: int = 12):
        """Get short interest chart over time"""
        url = f"https://financialmodelingprep.com/api/v4/short-interest-chart?symbol={symbol}&months={months}"
        return self.make_req(url)
    
    def get_analyst_ratings_chart(self, symbol: str, months: int = 12):
        """Get analyst ratings changes chart"""
        url = f"https://financialmodelingprep.com/api/v4/analyst-ratings-chart?symbol={symbol}&months={months}"
        return self.make_req(url)
    
    def get_sector_comparison_chart(self, symbols: str, period: str = "1year"):
        """Get sector performance comparison chart"""
        url = f"https://financialmodelingprep.com/api/v4/sector-comparison-chart?symbols={symbols}&period={period}"
        return self.make_req(url)
    
    def get_correlation_chart(self, symbol1: str, symbol2: str, period: int = 252):
        """Get correlation chart between two symbols"""
        url = f"https://financialmodelingprep.com/api/v4/correlation-chart?symbol1={symbol1}&symbol2={symbol2}&period={period}"
        return self.make_req(url)
    
    def get_options_flow_chart(self, symbol: str, days: int = 30):
        """Get options flow chart data"""
        url = f"https://financialmodelingprep.com/api/v4/options-flow-chart?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_dark_pool_chart(self, symbol: str, days: int = 30):
        """Get dark pool trading chart"""
        url = f"https://financialmodelingprep.com/api/v4/dark-pool-chart?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_market_cap_chart(self, symbol: str, years: int = 5):
        """Get market cap evolution chart"""
        url = f"https://financialmodelingprep.com/api/v4/market-cap-chart?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_pe_ratio_chart(self, symbol: str, years: int = 5):
        """Get P/E ratio historical chart"""
        url = f"https://financialmodelingprep.com/api/v4/pe-ratio-chart?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_revenue_chart(self, symbol: str, years: int = 10, period: str = "annual"):
        """Get revenue growth chart"""
        url = f"https://financialmodelingprep.com/api/v4/revenue-chart?symbol={symbol}&years={years}&period={period}"
        return self.make_req(url)
    
    def get_eps_chart(self, symbol: str, years: int = 10, period: str = "annual"):
        """Get earnings per share (EPS) chart"""
        url = f"https://financialmodelingprep.com/api/v4/eps-chart?symbol={symbol}&years={years}&period={period}"
        return self.make_req(url)
    
    def get_free_cash_flow_chart(self, symbol: str, years: int = 10):
        """Get free cash flow chart"""
        url = f"https://financialmodelingprep.com/api/v4/fcf-chart?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_debt_to_equity_chart(self, symbol: str, years: int = 10):
        """Get debt-to-equity ratio chart"""
        url = f"https://financialmodelingprep.com/api/v4/debt-equity-chart?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_roa_roe_chart(self, symbol: str, years: int = 10):
        """Get ROA and ROE chart"""
        url = f"https://financialmodelingprep.com/api/v4/roa-roe-chart?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_chart_export(self, symbol: str, chart_type: str = "price", format: str = "json", 
                        from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Export chart data (json, csv, excel)"""
        url = f"https://financialmodelingprep.com/api/v4/chart-export?symbol={symbol}&chartType={chart_type}&format={format}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        return self.make_req(url)
    
    
    # ===== TECHNICAL INDICATORS SECTION =====
    
    def get_technical_indicators_intraday(self, symbol: str, interval: str = "1min", indicator: str = "sma", period: int = 20):
        """Get intraday technical indicators (1min, 5min, 15min, 30min, 1hour)"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{interval}/{symbol}?type={indicator}&period={period}"
        return self.make_req(url)
    
    def get_all_technical_indicators(self, symbol: str, timeframe: str = "daily"):
        """Get all available technical indicators for a symbol"""
        url = f"https://financialmodelingprep.com/api/v4/technical-indicators-all?symbol={symbol}&timeframe={timeframe}"
        return self.make_req(url)
    
    # Moving Averages
    def get_sma_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Simple Moving Average (SMA) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=sma&period={period}"
        return self.make_req(url)
    
    def get_ema_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Exponential Moving Average (EMA) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=ema&period={period}"
        return self.make_req(url)
    
    def get_wma_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Weighted Moving Average (WMA) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=wma&period={period}"
        return self.make_req(url)
    
    def get_dema_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Double Exponential Moving Average (DEMA) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=dema&period={period}"
        return self.make_req(url)
    
    def get_tema_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Triple Exponential Moving Average (TEMA) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=tema&period={period}"
        return self.make_req(url)
    
    def get_vwma_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Volume Weighted Moving Average (VWMA) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=vwma&period={period}"
        return self.make_req(url)
    
    def get_hull_ma_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Hull Moving Average indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=hma&period={period}"
        return self.make_req(url)
    
    def get_kama_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Kaufman's Adaptive Moving Average (KAMA) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=kama&period={period}"
        return self.make_req(url)
    
    # Momentum Indicators
    def get_rsi_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Relative Strength Index (RSI) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=rsi&period={period}"
        return self.make_req(url)
    
    def get_macd_indicator(self, symbol: str, timeframe: str = "daily", fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Get MACD (Moving Average Convergence Divergence) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=macd&fastPeriod={fast_period}&slowPeriod={slow_period}&signalPeriod={signal_period}"
        return self.make_req(url)
    
    def get_stochastic_indicator(self, symbol: str, k_period: int = 14, d_period: int = 3, timeframe: str = "daily"):
        """Get Stochastic Oscillator (%K, %D) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=stoch&kPeriod={k_period}&dPeriod={d_period}"
        return self.make_req(url)
    
    def get_stoch_rsi_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Stochastic RSI indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=stochrsi&period={period}"
        return self.make_req(url)
    
    def get_williams_r_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Williams %R indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=williams&period={period}"
        return self.make_req(url)
    
    def get_roc_indicator(self, symbol: str, period: int = 10, timeframe: str = "daily"):
        """Get Rate of Change (ROC) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=roc&period={period}"
        return self.make_req(url)
    
    def get_momentum_indicator(self, symbol: str, period: int = 10, timeframe: str = "daily"):
        """Get Momentum indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=momentum&period={period}"
        return self.make_req(url)
    
    def get_ppo_indicator(self, symbol: str, fast_period: int = 12, slow_period: int = 26, timeframe: str = "daily"):
        """Get Percentage Price Oscillator (PPO) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=ppo&fastPeriod={fast_period}&slowPeriod={slow_period}"
        return self.make_req(url)
    
    # Trend Indicators
    def get_adx_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Average Directional Index (ADX) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=adx&period={period}"
        return self.make_req(url)
    
    def get_aroon_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Aroon indicator (Aroon Up, Aroon Down, Aroon Oscillator)"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=aroon&period={period}"
        return self.make_req(url)
    
    def get_psar_indicator(self, symbol: str, acceleration: float = 0.02, maximum: float = 0.2, timeframe: str = "daily"):
        """Get Parabolic SAR indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=psar&acceleration={acceleration}&maximum={maximum}"
        return self.make_req(url)
    
    def get_dmi_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Directional Movement Index (DMI) indicator (+DI, -DI)"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=dmi&period={period}"
        return self.make_req(url)
    
    def get_trix_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get TRIX indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=trix&period={period}"
        return self.make_req(url)
    
    def get_mass_index_indicator(self, symbol: str, period: int = 25, timeframe: str = "daily"):
        """Get Mass Index indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=mass_index&period={period}"
        return self.make_req(url)
    
    # Volatility Indicators
    def get_bollinger_bands_indicator(self, symbol: str, period: int = 20, std_dev: float = 2.0, timeframe: str = "daily"):
        """Get Bollinger Bands indicator (Upper, Middle, Lower)"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=bollinger&period={period}&stdDev={std_dev}"
        return self.make_req(url)
    
    def get_atr_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Average True Range (ATR) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=atr&period={period}"
        return self.make_req(url)
    
    def get_keltner_channels_indicator(self, symbol: str, period: int = 20, multiplier: float = 2.0, timeframe: str = "daily"):
        """Get Keltner Channels indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=keltner&period={period}&multiplier={multiplier}"
        return self.make_req(url)
    
    def get_donchian_channels_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Donchian Channels indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=donchian&period={period}"
        return self.make_req(url)
    
    def get_standard_deviation_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Standard Deviation indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=stddev&period={period}"
        return self.make_req(url)
    
    # Volume Indicators
    def get_obv_indicator(self, symbol: str, timeframe: str = "daily"):
        """Get On-Balance Volume (OBV) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=obv"
        return self.make_req(url)
    
    def get_ad_line_indicator(self, symbol: str, timeframe: str = "daily"):
        """Get Accumulation/Distribution Line indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=ad"
        return self.make_req(url)
    
    def get_chaikin_oscillator_indicator(self, symbol: str, fast_period: int = 3, slow_period: int = 10, timeframe: str = "daily"):
        """Get Chaikin Oscillator indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=chaikin&fastPeriod={fast_period}&slowPeriod={slow_period}"
        return self.make_req(url)
    
    def get_cmf_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Chaikin Money Flow (CMF) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=cmf&period={period}"
        return self.make_req(url)
    
    def get_mfi_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Money Flow Index (MFI) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=mfi&period={period}"
        return self.make_req(url)
    
    def get_vwap_indicator(self, symbol: str, timeframe: str = "1min"):
        """Get Volume Weighted Average Price (VWAP) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=vwap"
        return self.make_req(url)
    
    def get_ease_of_movement_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Ease of Movement indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=eom&period={period}"
        return self.make_req(url)
    
    def get_negative_volume_index_indicator(self, symbol: str, timeframe: str = "daily"):
        """Get Negative Volume Index (NVI) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=nvi"
        return self.make_req(url)
    
    def get_positive_volume_index_indicator(self, symbol: str, timeframe: str = "daily"):
        """Get Positive Volume Index (PVI) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=pvi"
        return self.make_req(url)
    
    # Oscillators
    def get_cci_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Commodity Channel Index (CCI) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=cci&period={period}"
        return self.make_req(url)
    
    def get_ultimate_oscillator_indicator(self, symbol: str, period1: int = 7, period2: int = 14, period3: int = 28, timeframe: str = "daily"):
        """Get Ultimate Oscillator indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=ultosc&period1={period1}&period2={period2}&period3={period3}"
        return self.make_req(url)
    
    def get_awesome_oscillator_indicator(self, symbol: str, timeframe: str = "daily"):
        """Get Awesome Oscillator indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=ao"
        return self.make_req(url)
    
    def get_detrended_price_oscillator_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Detrended Price Oscillator (DPO) indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=dpo&period={period}"
        return self.make_req(url)
    
    def get_fisher_transform_indicator(self, symbol: str, period: int = 10, timeframe: str = "daily"):
        """Get Fisher Transform indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=fisher&period={period}"
        return self.make_req(url)
    
    # Ichimoku Cloud Components
    def get_ichimoku_indicator(self, symbol: str, conversion_period: int = 9, base_period: int = 26, 
                              leading_span_b_period: int = 52, displacement: int = 26, timeframe: str = "daily"):
        """Get complete Ichimoku Cloud indicator (Tenkan, Kijun, Senkou Span A & B, Chikou Span)"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=ichimoku&conversionPeriod={conversion_period}&basePeriod={base_period}&leadingSpanBPeriod={leading_span_b_period}&displacement={displacement}"
        return self.make_req(url)
    
    def get_tenkan_sen_indicator(self, symbol: str, period: int = 9, timeframe: str = "daily"):
        """Get Tenkan-sen (Conversion Line) from Ichimoku"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=tenkan&period={period}"
        return self.make_req(url)
    
    def get_kijun_sen_indicator(self, symbol: str, period: int = 26, timeframe: str = "daily"):
        """Get Kijun-sen (Base Line) from Ichimoku"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=kijun&period={period}"
        return self.make_req(url)
    
    # Custom and Advanced Indicators
    def get_pivot_points_indicator(self, symbol: str, pivot_type: str = "standard", timeframe: str = "daily"):
        """Get Pivot Points (Standard, Fibonacci, Woodie, Camarilla, DeMark)"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=pivot&pivotType={pivot_type}"
        return self.make_req(url)
    
    def get_fibonacci_retracements_indicator(self, symbol: str, high_period: int = 50, low_period: int = 50, timeframe: str = "daily"):
        """Get Fibonacci Retracement levels indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=fibonacci&highPeriod={high_period}&lowPeriod={low_period}"
        return self.make_req(url)
    
    def get_supertrend_indicator(self, symbol: str, period: int = 10, multiplier: float = 3.0, timeframe: str = "daily"):
        """Get SuperTrend indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=supertrend&period={period}&multiplier={multiplier}"
        return self.make_req(url)
    
    def get_zigzag_indicator(self, symbol: str, deviation: float = 5.0, timeframe: str = "daily"):
        """Get ZigZag indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=zigzag&deviation={deviation}"
        return self.make_req(url)
    
    def get_linear_regression_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Linear Regression indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=linreg&period={period}"
        return self.make_req(url)
    
    def get_linear_regression_slope_indicator(self, symbol: str, period: int = 14, timeframe: str = "daily"):
        """Get Linear Regression Slope indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=linregslope&period={period}"
        return self.make_req(url)
    
    def get_standard_error_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Standard Error indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=stderr&period={period}"
        return self.make_req(url)
    
    def get_variance_indicator(self, symbol: str, period: int = 20, timeframe: str = "daily"):
        """Get Variance indicator"""
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{symbol}?type=var&period={period}"
        return self.make_req(url)
    
    # Technical Analysis Summary and Signals
    def get_technical_analysis_summary(self, symbol: str, timeframe: str = "daily"):
        """Get comprehensive technical analysis summary with buy/sell signals"""
        url = f"https://financialmodelingprep.com/api/v4/technical-analysis-summary?symbol={symbol}&timeframe={timeframe}"
        return self.make_req(url)
    
    def get_technical_signals(self, symbol: str, indicator: str = "all", timeframe: str = "daily"):
        """Get technical trading signals (buy, sell, neutral)"""
        url = f"https://financialmodelingprep.com/api/v4/technical-signals?symbol={symbol}&indicator={indicator}&timeframe={timeframe}"
        return self.make_req(url)
    
    def get_indicator_screener(self, indicator: str = "rsi", condition: str = "oversold", 
                              market_cap_min: int = 1000000000, limit: int = 50):
        """Screen stocks based on technical indicator conditions"""
        url = f"https://financialmodelingprep.com/api/v4/technical-screener?indicator={indicator}&condition={condition}&marketCapMin={market_cap_min}&limit={limit}"
        return self.make_req(url)
    
    def get_moving_average_convergence_screener(self, ma_short: int = 20, ma_long: int = 50, 
                                               condition: str = "golden_cross", limit: int = 50):
        """Screen for moving average convergence/divergence patterns"""
        url = f"https://financialmodelingprep.com/api/v4/ma-convergence-screener?maShort={ma_short}&maLong={ma_long}&condition={condition}&limit={limit}"
        return self.make_req(url)
    
    def get_multi_timeframe_analysis(self, symbol: str, timeframes: str = "1hour,4hour,daily,weekly"):
        """Get technical analysis across multiple timeframes"""
        url = f"https://financialmodelingprep.com/api/v4/multi-timeframe-analysis?symbol={symbol}&timeframes={timeframes}"
        return self.make_req(url)
    
    def get_indicator_alerts(self, symbols: str, indicator: str = "rsi", condition: str = "overbought"):
        """Get alerts when indicators reach specific conditions"""
        url = f"https://financialmodelingprep.com/api/v4/indicator-alerts?symbols={symbols}&indicator={indicator}&condition={condition}"
        return self.make_req(url)
    
    def get_custom_indicator(self, symbol: str, formula: str, period: int = 20, timeframe: str = "daily"):
        """Calculate custom technical indicator using formula"""
        url = f"https://financialmodelingprep.com/api/v4/custom-indicator?symbol={symbol}&formula={formula}&period={period}&timeframe={timeframe}"
        return self.make_req(url)
    
    def get_indicator_correlation(self, symbol: str, indicator1: str = "rsi", indicator2: str = "macd", period: int = 100):
        """Get correlation between different technical indicators"""
        url = f"https://financialmodelingprep.com/api/v4/indicator-correlation?symbol={symbol}&indicator1={indicator1}&indicator2={indicator2}&period={period}"
        return self.make_req(url)
    
    def get_indicator_performance(self, symbol: str, indicator: str = "rsi", strategy: str = "mean_reversion", 
                                 backtest_period: int = 252):
        """Get performance analysis of trading strategies based on technical indicators"""
        url = f"https://financialmodelingprep.com/api/v4/indicator-performance?symbol={symbol}&indicator={indicator}&strategy={strategy}&backtestPeriod={backtest_period}"
        return self.make_req(url)
    
    def get_sector_technical_strength(self, sector: str, indicator: str = "rsi"):
        """Get technical strength analysis for entire sectors"""
        url = f"https://financialmodelingprep.com/api/v4/sector-technical-strength?sector={sector}&indicator={indicator}"
        return self.make_req(url)
    
    def get_technical_export(self, symbol: str, indicators: str = "rsi,macd,sma", timeframe: str = "daily", 
                            format: str = "json", from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Export technical indicator data (json, csv, excel)"""
        url = f"https://financialmodelingprep.com/api/v4/technical-export?symbol={symbol}&indicators={indicators}&timeframe={timeframe}&format={format}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        return self.make_req(url)

    # ETF Holdings and Related Endpoints
    def get_etf_holdings(self, symbol: str, date: Optional[str] = None):
        """Get ETF holdings for a specific date"""
        url = f"https://financialmodelingprep.com/api/v4/etf-holdings?symbol={symbol}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_etf_holding_dates(self, symbol: str):
        """Get available holding dates for an ETF"""
        url = f"https://financialmodelingprep.com/api/v4/etf-holdings/portfolio-date?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_holder(self, symbol: str):
        """Get ETF holder information (stocks held by ETF)"""
        url = f"https://financialmodelingprep.com/api/v3/etf-holder/{symbol}"
        return self.make_req(url)
    
    def get_etf_information(self, symbol: str):
        """Get ETF basic information"""
        url = f"https://financialmodelingprep.com/api/v4/etf-info?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_sector_weightings(self, symbol: str):
        """Get ETF sector weightings breakdown"""
        url = f"https://financialmodelingprep.com/api/v3/etf-sector-weightings/{symbol}"
        return self.make_req(url)
    
    def get_etf_country_weightings(self, symbol: str):
        """Get ETF country weightings breakdown"""
        url = f"https://financialmodelingprep.com/api/v3/etf-country-weightings/{symbol}"
        return self.make_req(url)
    
    def get_etf_stock_exposure(self, symbol: str):
        """Get ETF exposure for a specific stock (which ETFs hold this stock)"""
        url = f"https://financialmodelingprep.com/api/v3/etf-stock-exposure/{symbol}"
        return self.make_req(url)
    
    def get_etf_holdings_by_date_range(self, symbol: str, from_date: str, to_date: str):
        """Get ETF holdings changes over a date range"""
        url = f"https://financialmodelingprep.com/api/v4/etf-holdings-date-range?symbol={symbol}&from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_etf_performance(self, symbol: str, period: str = "1y"):
        """Get ETF performance metrics"""
        url = f"https://financialmodelingprep.com/api/v4/etf-performance?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_etf_expense_ratio(self, symbol: str):
        """Get ETF expense ratio and fee information"""
        url = f"https://financialmodelingprep.com/api/v4/etf-expense-ratio?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_dividend_history(self, symbol: str, limit: int = 50):
        """Get ETF dividend payment history"""
        url = f"https://financialmodelingprep.com/api/v4/etf-dividend-history?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_etf_top_holdings(self, symbol: str, limit: int = 10):
        """Get top holdings of an ETF"""
        url = f"https://financialmodelingprep.com/api/v4/etf-top-holdings?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_etf_holdings_changes(self, symbol: str, current_date: str, previous_date: str):
        """Compare ETF holdings between two dates"""
        url = f"https://financialmodelingprep.com/api/v4/etf-holdings-changes?symbol={symbol}&current_date={current_date}&previous_date={previous_date}"
        return self.make_req(url)
    
    def get_etf_sector_exposure(self, symbol: str):
        """Get ETF sector exposure analysis"""
        url = f"https://financialmodelingprep.com/api/v4/etf-sector-exposure?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_geographic_exposure(self, symbol: str):
        """Get ETF geographic/country exposure"""
        url = f"https://financialmodelingprep.com/api/v4/etf-geographic-exposure?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_asset_allocation(self, symbol: str):
        """Get ETF asset allocation breakdown"""
        url = f"https://financialmodelingprep.com/api/v4/etf-asset-allocation?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_risk_metrics(self, symbol: str, period: str = "1y"):
        """Get ETF risk metrics (volatility, beta, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/etf-risk-metrics?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_etf_tracking_error(self, symbol: str, benchmark: Optional[str] = None):
        """Get ETF tracking error vs benchmark"""
        url = f"https://financialmodelingprep.com/api/v4/etf-tracking-error?symbol={symbol}"
        if benchmark:
            url += f"&benchmark={benchmark}"
        return self.make_req(url)
    
    def get_etf_liquidity_metrics(self, symbol: str):
        """Get ETF liquidity metrics"""
        url = f"https://financialmodelingprep.com/api/v4/etf-liquidity?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_premium_discount(self, symbol: str, days: int = 30):
        """Get ETF premium/discount to NAV"""
        url = f"https://financialmodelingprep.com/api/v4/etf-premium-discount?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_etf_creation_redemption(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """Get ETF creation and redemption activity"""
        url = f"https://financialmodelingprep.com/api/v4/etf-creation-redemption?symbol={symbol}"
        if from_date:
            url += f"&from_date={from_date}"
        if to_date:
            url += f"&to_date={to_date}"
        return self.make_req(url)
    
    def get_etf_flows(self, symbol: str, period: str = "monthly", limit: int = 12):
        """Get ETF flow data (inflows/outflows)"""
        url = f"https://financialmodelingprep.com/api/v4/etf-flows?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_etf_holdings_concentration(self, symbol: str):
        """Get ETF holdings concentration analysis"""
        url = f"https://financialmodelingprep.com/api/v4/etf-concentration?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_overlap_analysis(self, symbol1: str, symbol2: str):
        """Analyze holdings overlap between two ETFs"""
        url = f"https://financialmodelingprep.com/api/v4/etf-overlap?symbol1={symbol1}&symbol2={symbol2}"
        return self.make_req(url)
    
    def get_etf_similar_funds(self, symbol: str, limit: int = 10):
        """Find similar ETFs based on holdings"""
        url = f"https://financialmodelingprep.com/api/v4/etf-similar?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_etf_holdings_turnover(self, symbol: str, period: str = "annual"):
        """Get ETF portfolio turnover rate"""
        url = f"https://financialmodelingprep.com/api/v4/etf-turnover?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_etf_screener(self, asset_class: Optional[str] = None, sector: Optional[str] = None, 
                        min_aum: Optional[int] = None, max_expense_ratio: Optional[float] = None,
                        min_yield: Optional[float] = None, limit: int = 50):
        """Screen ETFs by various criteria"""
        url = f"https://financialmodelingprep.com/api/v4/etf-screener?limit={limit}"
        if asset_class:
            url += f"&asset_class={asset_class}"
        if sector:
            url += f"&sector={sector}"
        if min_aum:
            url += f"&min_aum={min_aum}"
        if max_expense_ratio:
            url += f"&max_expense_ratio={max_expense_ratio}"
        if min_yield:
            url += f"&min_yield={min_yield}"
        return self.make_req(url)
    
    def get_etf_holdings_by_symbol(self, held_symbol: str, limit: int = 50):
        """Get all ETFs that hold a specific stock"""
        url = f"https://financialmodelingprep.com/api/v4/etf-holdings-by-symbol?symbol={held_symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_etf_institutional_holders(self, symbol: str, limit: int = 50):
        """Get institutional holders of an ETF"""
        url = f"https://financialmodelingprep.com/api/v4/etf-institutional-holders?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_etf_analyst_recommendations(self, symbol: str):
        """Get analyst recommendations for an ETF"""
        url = f"https://financialmodelingprep.com/api/v4/etf-analyst-recommendations?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_options_chain(self, symbol: str):
        """Get options chain for an ETF"""
        url = f"https://financialmodelingprep.com/api/v4/etf-options?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_short_interest(self, symbol: str, limit: int = 50):
        """Get ETF short interest data"""
        url = f"https://financialmodelingprep.com/api/v4/etf-short-interest?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_etf_tax_efficiency(self, symbol: str):
        """Get ETF tax efficiency metrics"""
        url = f"https://financialmodelingprep.com/api/v4/etf-tax-efficiency?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_carbon_footprint(self, symbol: str):
        """Get ETF carbon footprint and ESG metrics"""
        url = f"https://financialmodelingprep.com/api/v4/etf-carbon-footprint?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_rebalancing_schedule(self, symbol: str):
        """Get ETF rebalancing schedule and history"""
        url = f"https://financialmodelingprep.com/api/v4/etf-rebalancing?symbol={symbol}"
        return self.make_req(url)
    
    def get_etf_distribution_schedule(self, symbol: str):
        """Get ETF distribution/dividend schedule"""
        url = f"https://financialmodelingprep.com/api/v4/etf-distributions?symbol={symbol}"
        return self.make_req(url)

    # Mutual Fund Holdings and Related Endpoints
    def get_mutual_fund_holdings(self, symbol: Optional[str] = None, date: Optional[str] = None, cik: Optional[str] = None):
        """Get mutual fund holdings for a specific date"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-holdings"
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if date:
            params.append(f"date={date}")
        if cik:
            params.append(f"cik={cik}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_mutual_fund_holding_dates(self, symbol: Optional[str] = None, cik: Optional[str] = None):
        """Get available holding dates for a mutual fund"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-holdings/portfolio-date"
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if cik:
            params.append(f"cik={cik}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_mutual_fund_holder(self, symbol: str):
        """Get mutual funds that hold a specific stock"""
        url = f"https://financialmodelingprep.com/api/v3/mutual-fund-holder/{symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_by_name(self, name: str):
        """Get mutual fund information by fund name"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-search?name={name}"
        return self.make_req(url)
    
    def get_mutual_fund_information(self, symbol: str):
        """Get basic mutual fund information"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-info?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_sector_weightings(self, symbol: str):
        """Get mutual fund sector weightings breakdown"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-sector-weightings?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_country_weightings(self, symbol: str):
        """Get mutual fund country weightings breakdown"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-country-weightings?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_top_holdings(self, symbol: str, limit: int = 10):
        """Get top holdings of a mutual fund"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-top-holdings?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_mutual_fund_performance(self, symbol: str, period: str = "1y"):
        """Get mutual fund performance metrics"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-performance?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_mutual_fund_expense_ratio(self, symbol: str):
        """Get mutual fund expense ratio and fee information"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-expense-ratio?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_holdings_changes(self, symbol: str, current_date: str, previous_date: str):
        """Compare mutual fund holdings between two dates"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-holdings-changes?symbol={symbol}&current_date={current_date}&previous_date={previous_date}"
        return self.make_req(url)
    
    def get_mutual_fund_flows(self, symbol: str, period: str = "monthly", limit: int = 12):
        """Get mutual fund flow data (inflows/outflows)"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-flows?symbol={symbol}&period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_mutual_fund_overlap_analysis(self, symbol1: str, symbol2: str):
        """Analyze holdings overlap between two mutual funds"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-overlap?symbol1={symbol1}&symbol2={symbol2}"
        return self.make_req(url)
    
    def get_mutual_fund_similar_funds(self, symbol: str, limit: int = 10):
        """Find similar mutual funds based on holdings"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-similar?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_mutual_fund_screener(self, category: Optional[str] = None, min_aum: Optional[int] = None, 
                                max_expense_ratio: Optional[float] = None, min_return: Optional[float] = None,
                                morningstar_rating: Optional[int] = None, limit: int = 50):
        """Screen mutual funds by various criteria"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-screener?limit={limit}"
        if category:
            url += f"&category={category}"
        if min_aum:
            url += f"&min_aum={min_aum}"
        if max_expense_ratio:
            url += f"&max_expense_ratio={max_expense_ratio}"
        if min_return:
            url += f"&min_return={min_return}"
        if morningstar_rating:
            url += f"&morningstar_rating={morningstar_rating}"
        return self.make_req(url)
    
    def get_mutual_fund_holdings_by_symbol(self, held_symbol: str, limit: int = 50):
        """Get all mutual funds that hold a specific stock"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-holdings-by-symbol?symbol={held_symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_mutual_fund_manager_info(self, symbol: str):
        """Get mutual fund manager information"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-manager?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_risk_metrics(self, symbol: str, period: str = "1y"):
        """Get mutual fund risk metrics (volatility, beta, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-risk-metrics?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_mutual_fund_dividend_history(self, symbol: str, limit: int = 50):
        """Get mutual fund dividend/distribution history"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-dividends?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_mutual_fund_asset_allocation(self, symbol: str):
        """Get mutual fund asset allocation breakdown"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-asset-allocation?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_style_analysis(self, symbol: str):
        """Get mutual fund investment style analysis"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-style?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_load_fees(self, symbol: str):
        """Get mutual fund load fees and sales charges"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-loads?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_turnover_ratio(self, symbol: str):
        """Get mutual fund portfolio turnover ratio"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-turnover?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_benchmark_comparison(self, symbol: str, benchmark: Optional[str] = None):
        """Compare mutual fund performance vs benchmark"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-benchmark?symbol={symbol}"
        if benchmark:
            url += f"&benchmark={benchmark}"
        return self.make_req(url)
    
    def get_mutual_fund_category_analysis(self, category: str):
        """Get analysis of mutual funds in a specific category"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-category-analysis?category={category}"
        return self.make_req(url)
    
    def get_mutual_fund_holdings_concentration(self, symbol: str):
        """Get mutual fund holdings concentration analysis"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-concentration?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_shareholder_information(self, symbol: str):
        """Get mutual fund shareholder information"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-shareholders?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_tax_efficiency(self, symbol: str):
        """Get mutual fund tax efficiency metrics"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-tax-efficiency?symbol={symbol}"
        return self.make_req(url)
    
    def get_mutual_fund_calendar(self, from_date: str, to_date: str):
        """Get mutual fund reporting calendar"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_mutual_fund_list(self, category: Optional[str] = None, limit: int = 100):
        """Get list of available mutual funds"""
        url = f"https://financialmodelingprep.com/api/v4/mutual-fund-list?limit={limit}"
        if category:
            url += f"&category={category}"
        return self.make_req(url)

    # ESG (Environmental, Social, Governance) Endpoints
    def get_esg_search(self, symbol: str):
        """Search for ESG data and ratings for a company"""
        url = f"https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_score(self, symbol: str):
        """Get comprehensive ESG score for a company"""
        url = f"https://financialmodelingprep.com/api/v4/esg-score?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_rating(self, symbol: str):
        """Get ESG rating and grade for a company"""
        url = f"https://financialmodelingprep.com/api/v4/esg-rating?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_benchmark(self, symbol: Optional[str] = None, sector: Optional[str] = None):
        """Get ESG benchmark data for companies or sectors"""
        url = f"https://financialmodelingprep.com/api/v4/esg-benchmark"
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if sector:
            params.append(f"sector={sector}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_environmental_score(self, symbol: str):
        """Get environmental metrics and score"""
        url = f"https://financialmodelingprep.com/api/v4/environmental-score?symbol={symbol}"
        return self.make_req(url)
    
    def get_social_score(self, symbol: str):
        """Get social responsibility metrics and score"""
        url = f"https://financialmodelingprep.com/api/v4/social-score?symbol={symbol}"
        return self.make_req(url)
    
    def get_governance_score(self, symbol: str):
        """Get corporate governance metrics and score"""
        url = f"https://financialmodelingprep.com/api/v4/governance-score?symbol={symbol}"
        return self.make_req(url)
    
    def get_sustainability_score(self, symbol: str):
        """Get overall sustainability score and metrics"""
        url = f"https://financialmodelingprep.com/api/v4/sustainability-score?symbol={symbol}"
        return self.make_req(url)
    
    def get_carbon_footprint(self, symbol: str):
        """Get carbon emissions and environmental impact data"""
        url = f"https://financialmodelingprep.com/api/v4/carbon-footprint?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_risk_rating(self, symbol: str):
        """Get ESG risk assessment and rating"""
        url = f"https://financialmodelingprep.com/api/v4/esg-risk-rating?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_controversies(self, symbol: str):
        """Get ESG controversies and incidents"""
        url = f"https://financialmodelingprep.com/api/v4/esg-controversies?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_sector_performance(self, sector: str, metric: str = "overall"):
        """Get ESG performance analysis by sector"""
        url = f"https://financialmodelingprep.com/api/v4/esg-sector-performance?sector={sector}&metric={metric}"
        return self.make_req(url)
    
    def get_esg_industry_comparison(self, symbol: str):
        """Compare company's ESG performance with industry peers"""
        url = f"https://financialmodelingprep.com/api/v4/esg-industry-comparison?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_trends(self, symbol: str, years: int = 5):
        """Get historical ESG score trends"""
        url = f"https://financialmodelingprep.com/api/v4/esg-trends?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_esg_screener(self, min_esg_score: Optional[float] = None, max_esg_score: Optional[float] = None,
                        environmental_grade: Optional[str] = None, social_grade: Optional[str] = None,
                        governance_grade: Optional[str] = None, sector: Optional[str] = None, limit: int = 50):
        """Screen companies based on ESG criteria"""
        url = f"https://financialmodelingprep.com/api/v4/esg-screener?limit={limit}"
        if min_esg_score:
            url += f"&min_esg_score={min_esg_score}"
        if max_esg_score:
            url += f"&max_esg_score={max_esg_score}"
        if environmental_grade:
            url += f"&environmental_grade={environmental_grade}"
        if social_grade:
            url += f"&social_grade={social_grade}"
        if governance_grade:
            url += f"&governance_grade={governance_grade}"
        if sector:
            url += f"&sector={sector}"
        return self.make_req(url)
    
    def get_green_revenue(self, symbol: str):
        """Get green/sustainable revenue breakdown"""
        url = f"https://financialmodelingprep.com/api/v4/green-revenue?symbol={symbol}"
        return self.make_req(url)
    
    def get_diversity_metrics(self, symbol: str):
        """Get workforce diversity and inclusion metrics"""
        url = f"https://financialmodelingprep.com/api/v4/diversity-metrics?symbol={symbol}"
        return self.make_req(url)
    
    def get_employee_satisfaction(self, symbol: str):
        """Get employee satisfaction and workplace metrics"""
        url = f"https://financialmodelingprep.com/api/v4/employee-satisfaction?symbol={symbol}"
        return self.make_req(url)
    
    def get_board_diversity(self, symbol: str):
        """Get board of directors diversity information"""
        url = f"https://financialmodelingprep.com/api/v4/board-diversity?symbol={symbol}"
        return self.make_req(url)
    
    def get_executive_compensation_esg(self, symbol: str):
        """Get ESG-linked executive compensation data"""
        url = f"https://financialmodelingprep.com/api/v4/executive-compensation-esg?symbol={symbol}"
        return self.make_req(url)
    
    def get_supply_chain_esg(self, symbol: str):
        """Get supply chain ESG practices and scores"""
        url = f"https://financialmodelingprep.com/api/v4/supply-chain-esg?symbol={symbol}"
        return self.make_req(url)
    
    def get_climate_risk_assessment(self, symbol: str):
        """Get climate change risk assessment"""
        url = f"https://financialmodelingprep.com/api/v4/climate-risk?symbol={symbol}"
        return self.make_req(url)
    
    def get_water_usage_metrics(self, symbol: str):
        """Get water usage and conservation metrics"""
        url = f"https://financialmodelingprep.com/api/v4/water-usage?symbol={symbol}"
        return self.make_req(url)
    
    def get_waste_management_metrics(self, symbol: str):
        """Get waste management and recycling metrics"""
        url = f"https://financialmodelingprep.com/api/v4/waste-management?symbol={symbol}"
        return self.make_req(url)
    
    def get_energy_efficiency_metrics(self, symbol: str):
        """Get energy efficiency and renewable energy usage"""
        url = f"https://financialmodelingprep.com/api/v4/energy-efficiency?symbol={symbol}"
        return self.make_req(url)
    
    def get_human_rights_score(self, symbol: str):
        """Get human rights compliance and score"""
        url = f"https://financialmodelingprep.com/api/v4/human-rights?symbol={symbol}"
        return self.make_req(url)
    
    def get_product_safety_metrics(self, symbol: str):
        """Get product safety and quality metrics"""
        url = f"https://financialmodelingprep.com/api/v4/product-safety?symbol={symbol}"
        return self.make_req(url)
    
    def get_cybersecurity_score(self, symbol: str):
        """Get cybersecurity practices and risk score"""
        url = f"https://financialmodelingprep.com/api/v4/cybersecurity-score?symbol={symbol}"
        return self.make_req(url)
    
    def get_tax_transparency(self, symbol: str):
        """Get tax transparency and fair tax practices"""
        url = f"https://financialmodelingprep.com/api/v4/tax-transparency?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_etf_analysis(self, symbol: str):
        """Get ESG analysis for ETFs and funds"""
        url = f"https://financialmodelingprep.com/api/v4/esg-etf-analysis?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_portfolio_analysis(self, symbols: str):
        """Analyze ESG metrics for a portfolio of stocks"""
        url = f"https://financialmodelingprep.com/api/v4/esg-portfolio-analysis?symbols={symbols}"
        return self.make_req(url)
    
    def get_sustainable_development_goals(self, symbol: str):
        """Get UN Sustainable Development Goals alignment"""
        url = f"https://financialmodelingprep.com/api/v4/sdg-alignment?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_materiality_assessment(self, symbol: str):
        """Get ESG materiality assessment for industry"""
        url = f"https://financialmodelingprep.com/api/v4/esg-materiality?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_regulatory_compliance(self, symbol: str):
        """Get ESG regulatory compliance status"""
        url = f"https://financialmodelingprep.com/api/v4/esg-compliance?symbol={symbol}"
        return self.make_req(url)
    
    def get_esg_news_sentiment(self, symbol: str, days: int = 30):
        """Get ESG-related news sentiment analysis"""
        url = f"https://financialmodelingprep.com/api/v4/esg-news-sentiment?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_esg_best_practices(self, sector: str):
        """Get ESG best practices for a sector"""
        url = f"https://financialmodelingprep.com/api/v4/esg-best-practices?sector={sector}"
        return self.make_req(url)
    
    def get_esg_targets_tracking(self, symbol: str):
        """Track company's progress on ESG targets"""
        url = f"https://financialmodelingprep.com/api/v4/esg-targets?symbol={symbol}"
        return self.make_req(url)

    # Senate Trading Endpoints
    def get_senate_trading(self, symbol: Optional[str] = None, limit: int = 100):
        """Get Senate trading disclosures"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading?limit={limit}"
        if symbol:
            url += f"&symbol={symbol}"
        return self.make_req(url)
    
    def get_senate_trading_rss_feed(self, page: int = 0):
        """Get Senate trading RSS feed"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-rss-feed?page={page}"
        return self.make_req(url)
    
    def get_senate_trading_by_senator(self, senator_name: str, limit: int = 50):
        """Get trading activity by specific Senator"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-by-senator?senator={senator_name}&limit={limit}"
        return self.make_req(url)
    
    def get_senate_trading_by_symbol(self, symbol: str, limit: int = 50):
        """Get Senate trading activity for specific stock symbol"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-by-symbol?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_senate_trading_by_date_range(self, from_date: str, to_date: str, limit: int = 100):
        """Get Senate trading within date range"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-date-range?from={from_date}&to={to_date}&limit={limit}"
        return self.make_req(url)
    
    def get_senate_trading_statistics(self, period: str = "monthly"):
        """Get Senate trading statistics and trends"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-stats?period={period}"
        return self.make_req(url)
    
    def get_senator_portfolio(self, senator_name: str):
        """Get current portfolio holdings of a Senator"""
        url = f"https://financialmodelingprep.com/api/v4/senator-portfolio?senator={senator_name}"
        return self.make_req(url)
    
    def get_senate_trading_performance(self, senator_name: Optional[str] = None, timeframe: str = "1y"):
        """Get trading performance analysis for Senators"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-performance?timeframe={timeframe}"
        if senator_name:
            url += f"&senator={senator_name}"
        return self.make_req(url)
    
    def get_most_traded_by_senate(self, period: str = "monthly", limit: int = 20):
        """Get most traded stocks by Senate members"""
        url = f"https://financialmodelingprep.com/api/v4/senate-most-traded?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_senate_trading_sectors(self, senator_name: Optional[str] = None):
        """Get sector breakdown of Senate trading"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-sectors"
        if senator_name:
            url += f"?senator={senator_name}"
        return self.make_req(url)
    
    def get_senate_insider_trading_correlation(self, symbol: str):
        """Correlate Senate trading with stock performance"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-correlation?symbol={symbol}"
        return self.make_req(url)
    
    def get_senate_disclosure_timeline(self, senator_name: str, limit: int = 50):
        """Get disclosure timeline for a Senator"""
        url = f"https://financialmodelingprep.com/api/v4/senate-disclosure-timeline?senator={senator_name}&limit={limit}"
        return self.make_req(url)
    
    def get_senate_trading_alerts(self, symbols: Optional[str] = None, senators: Optional[str] = None):
        """Set up alerts for Senate trading activity"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-alerts"
        params = []
        if symbols:
            params.append(f"symbols={symbols}")
        if senators:
            params.append(f"senators={senators}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_senate_trading_volume_analysis(self, period: str = "quarterly"):
        """Get Senate trading volume analysis"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-volume?period={period}"
        return self.make_req(url)
    
    def get_senate_party_trading_analysis(self, party: Optional[str] = None):
        """Analyze trading patterns by political party"""
        url = f"https://financialmodelingprep.com/api/v4/senate-party-trading"
        if party:
            url += f"?party={party}"
        return self.make_req(url)
    
    def get_senate_committee_trading(self, committee: str):
        """Get trading by Senate committee members"""
        url = f"https://financialmodelingprep.com/api/v4/senate-committee-trading?committee={committee}"
        return self.make_req(url)
    
    def get_senate_trading_frequency(self, senator_name: Optional[str] = None, timeframe: str = "1y"):
        """Get trading frequency analysis for Senators"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-frequency?timeframe={timeframe}"
        if senator_name:
            url += f"&senator={senator_name}"
        return self.make_req(url)
    
    def get_senate_stock_ownership(self, symbol: str):
        """Get Senate members who own specific stock"""
        url = f"https://financialmodelingprep.com/api/v4/senate-stock-ownership?symbol={symbol}"
        return self.make_req(url)
    
    def get_senate_trading_compliance(self, senator_name: Optional[str] = None):
        """Check Senate trading compliance with disclosure rules"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-compliance"
        if senator_name:
            url += f"?senator={senator_name}"
        return self.make_req(url)
    
    def get_senate_trading_impact_analysis(self, symbol: str, days_around: int = 5):
        """Analyze market impact of Senate trading"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-impact?symbol={symbol}&days={days_around}"
        return self.make_req(url)
    
    def get_senate_largest_trades(self, period: str = "monthly", limit: int = 20):
        """Get largest Senate trades by dollar amount"""
        url = f"https://financialmodelingprep.com/api/v4/senate-largest-trades?period={period}&limit={limit}"
        return self.make_req(url)
    
    def get_senate_trading_calendar(self, from_date: str, to_date: str):
        """Get Senate trading calendar/schedule"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_senate_conflict_of_interest(self, senator_name: Optional[str] = None, sector: Optional[str] = None):
        """Analyze potential conflicts of interest"""
        url = f"https://financialmodelingprep.com/api/v4/senate-conflicts"
        params = []
        if senator_name:
            params.append(f"senator={senator_name}")
        if sector:
            params.append(f"sector={sector}")
        if params:
            url += "?" + "&".join(params)
        return self.make_req(url)
    
    def get_senate_trading_trends(self, timeframe: str = "yearly", category: str = "all"):
        """Get long-term Senate trading trends"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-trends?timeframe={timeframe}&category={category}"
        return self.make_req(url)
    
    def get_senate_trading_export(self, from_date: str, to_date: str, format: str = "csv"):
        """Export Senate trading data"""
        url = f"https://financialmodelingprep.com/api/v4/senate-trading-export?from={from_date}&to={to_date}&format={format}"
        return self.make_req(url)

    # Market Performance Endpoints
    def get_market_performance_overview(self, date: Optional[str] = None):
        """Get overall market performance summary"""
        url = f"https://financialmodelingprep.com/api/v4/market-performance"
        if date:
            url += f"?date={date}"
        return self.make_req(url)
    
    def get_market_indices_performance(self, period: str = "1d"):
        """Get performance of major market indices"""
        url = f"https://financialmodelingprep.com/api/v4/market-indices-performance?period={period}"
        return self.make_req(url)
    
    def get_sector_performance_analysis(self, date: Optional[str] = None, period: str = "1d"):
        """Get detailed sector performance analysis"""
        url = f"https://financialmodelingprep.com/api/v4/sector-performance-analysis?period={period}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_market_breadth_indicators(self, date: Optional[str] = None):
        """Get market breadth indicators (advance/decline, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/market-breadth"
        if date:
            url += f"?date={date}"
        return self.make_req(url)
    
    def get_market_volatility_metrics(self, period: str = "1m"):
        """Get market volatility metrics and VIX data"""
        url = f"https://financialmodelingprep.com/api/v4/market-volatility?period={period}"
        return self.make_req(url)
    
    def get_market_sentiment_analysis(self, date: Optional[str] = None):
        """Get comprehensive market sentiment indicators"""
        url = f"https://financialmodelingprep.com/api/v4/market-sentiment"
        if date:
            url += f"?date={date}"
        return self.make_req(url)
    
    def get_market_performance_comparison(self, symbols: str, period: str = "1y"):
        """Compare performance across multiple markets/indices"""
        url = f"https://financialmodelingprep.com/api/v4/market-performance-comparison?symbols={symbols}&period={period}"
        return self.make_req(url)
    
    def get_historical_market_performance(self, symbol: str = "SPY", years: int = 5):
        """Get historical market performance trends"""
        url = f"https://financialmodelingprep.com/api/v4/historical-market-performance?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_market_rotation_analysis(self, period: str = "quarterly"):
        """Analyze sector rotation patterns"""
        url = f"https://financialmodelingprep.com/api/v4/market-rotation?period={period}"
        return self.make_req(url)
    
    def get_market_momentum_indicators(self, timeframe: str = "daily"):
        """Get market momentum and trend indicators"""
        url = f"https://financialmodelingprep.com/api/v4/market-momentum?timeframe={timeframe}"
        return self.make_req(url)
    
    def get_market_risk_metrics(self, period: str = "1y"):
        """Get market risk assessment metrics"""
        url = f"https://financialmodelingprep.com/api/v4/market-risk-metrics?period={period}"
        return self.make_req(url)
    
    def get_market_correlation_matrix(self, symbols: str, period: int = 252):
        """Get correlation matrix for market instruments"""
        url = f"https://financialmodelingprep.com/api/v4/market-correlation?symbols={symbols}&period={period}"
        return self.make_req(url)
    
    def get_market_cycles_analysis(self, cycle_type: str = "business", years: int = 10):
        """Analyze market cycles and patterns"""
        url = f"https://financialmodelingprep.com/api/v4/market-cycles?type={cycle_type}&years={years}"
        return self.make_req(url)
    
    def get_global_market_performance(self, region: str = "all"):
        """Get global markets performance summary"""
        url = f"https://financialmodelingprep.com/api/v4/global-markets?region={region}"
        return self.make_req(url)
    
    def get_market_leadership_analysis(self, period: str = "monthly"):
        """Analyze market leadership by sector/style"""
        url = f"https://financialmodelingprep.com/api/v4/market-leadership?period={period}"
        return self.make_req(url)
    
    def get_market_efficiency_metrics(self, symbol: str = "SPY"):
        """Get market efficiency and liquidity metrics"""
        url = f"https://financialmodelingprep.com/api/v4/market-efficiency?symbol={symbol}"
        return self.make_req(url)
    
    def get_market_stress_indicators(self, date: Optional[str] = None):
        """Get market stress and crisis indicators"""
        url = f"https://financialmodelingprep.com/api/v4/market-stress"
        if date:
            url += f"?date={date}"
        return self.make_req(url)
    
    def get_market_performance_attribution(self, symbol: str, period: str = "1y"):
        """Get performance attribution analysis"""
        url = f"https://financialmodelingprep.com/api/v4/performance-attribution?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_market_regime_analysis(self, lookback_days: int = 252):
        """Identify current market regime (bull/bear/sideways)"""
        url = f"https://financialmodelingprep.com/api/v4/market-regime?lookback={lookback_days}"
        return self.make_req(url)
    
    def get_market_seasonality_patterns(self, symbol: str = "SPY", years: int = 10):
        """Analyze seasonal market patterns"""
        url = f"https://financialmodelingprep.com/api/v4/market-seasonality?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_market_performance_rankings(self, category: str = "sectors", period: str = "1m"):
        """Get performance rankings by category"""
        url = f"https://financialmodelingprep.com/api/v4/performance-rankings?category={category}&period={period}"
        return self.make_req(url)
    
    def get_market_beta_analysis(self, symbol: Optional[str] = None, symbols: Optional[str] = None, benchmark: str = "SPY"):
        """Calculate beta vs market benchmark"""
        # Accept either symbol or symbols parameter for backward compatibility
        if symbol and not symbols:
            symbols = symbol
        elif not symbols:
            raise ValueError("Either 'symbol' or 'symbols' parameter must be provided")
        url = f"https://financialmodelingprep.com/api/v4/market-beta?symbols={symbols}&benchmark={benchmark}"
        return self.make_req(url)
    
    def get_market_drawdown_analysis(self, symbol: str = "SPY", period: str = "max"):
        """Analyze market drawdowns and recovery periods"""
        url = f"https://financialmodelingprep.com/api/v4/market-drawdown?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_market_returns_distribution(self, symbol: str = "SPY", years: int = 5):
        """Get returns distribution analysis"""
        url = f"https://financialmodelingprep.com/api/v4/returns-distribution?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_market_technical_levels(self, symbol: str = "SPY"):
        """Get key technical levels for market indices"""
        url = f"https://financialmodelingprep.com/api/v4/market-technical-levels?symbol={symbol}"
        return self.make_req(url)
    
    def get_market_economic_indicators_impact(self, indicator: str = "all"):
        """Analyze economic indicators impact on market"""
        url = f"https://financialmodelingprep.com/api/v4/economic-market-impact?indicator={indicator}"
        return self.make_req(url)
    
    def get_market_options_flow_impact(self, date: Optional[str] = None):
        """Analyze options flow impact on market direction"""
        url = f"https://financialmodelingprep.com/api/v4/options-market-impact"
        if date:
            url += f"?date={date}"
        return self.make_req(url)
    
    def get_market_institutional_flow(self, period: str = "weekly"):
        """Track institutional money flows"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-market-flow?period={period}"
        return self.make_req(url)
    
    def get_market_retail_sentiment(self, period: str = "daily"):
        """Get retail investor sentiment indicators"""
        url = f"https://financialmodelingprep.com/api/v4/retail-market-sentiment?period={period}"
        return self.make_req(url)
    
    def get_market_performance_alerts(self, thresholds: Optional[str] = None):
        """Set up market performance alerts"""
        url = f"https://financialmodelingprep.com/api/v4/market-performance-alerts"
        if thresholds:
            url += f"?thresholds={thresholds}"
        return self.make_req(url)
    
    def get_market_calendar_impact(self, from_date: str, to_date: str):
        """Analyze market calendar events impact"""
        url = f"https://financialmodelingprep.com/api/v4/market-calendar-impact?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_market_concentration_analysis(self, index: str = "SPY"):
        """Analyze market concentration and top holdings impact"""
        url = f"https://financialmodelingprep.com/api/v4/market-concentration?index={index}"
        return self.make_req(url)
    
    def get_market_performance_forecast(self, symbol: str = "SPY", days_ahead: int = 30):
        """Get market performance forecasts"""
        url = f"https://financialmodelingprep.com/api/v4/market-forecast?symbol={symbol}&days={days_ahead}"
        return self.make_req(url)
    
    def get_market_performance_summary(self, timeframes: str = "1d,1w,1m,3m,6m,1y"):
        """Get multi-timeframe market performance summary"""
        url = f"https://financialmodelingprep.com/api/v4/market-summary?timeframes={timeframes}"
        return self.make_req(url)
    
    def get_market_performance_export(self, from_date: str, to_date: str, 
                                     metrics: str = "all", format: str = "csv"):
        """Export market performance data"""
        url = f"https://financialmodelingprep.com/api/v4/market-performance-export?from={from_date}&to={to_date}&metrics={metrics}&format={format}"
        return self.make_req(url)

    # 13F Institutional Ownership Endpoints
    def get_form_13f_holdings(self, cik: str, date: Optional[str] = None):
        """Get 13F holdings for specific institution by CIK"""
        url = f"https://financialmodelingprep.com/api/v3/form-thirteen/{cik}"
        if date:
            url += f"?date={date}"
        return self.make_req(url)
    
    def get_form_13f_dates(self, cik: str):
        """Get available 13F filing dates for institution"""
        url = f"https://financialmodelingprep.com/api/v3/form-thirteen-date/{cik}"
        return self.make_req(url)
    
    def get_institutional_holders(self, symbol: str, limit: int = 100):
        """Get institutional holders for a stock"""
        url = f"https://financialmodelingprep.com/api/v3/institutional-holder/{symbol}?limit={limit}"
        return self.make_req(url)
    
    def get_institutional_ownership_by_shares(self, symbol: str, date: Optional[str] = None, limit: int = 100):
        """Get institutional ownership by shares held"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-ownership-by-shares-held?symbol={symbol}&limit={limit}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_ownership_percentage(self, symbol: str, date: Optional[str] = None):
        """Get institutional ownership percentage of outstanding shares"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-ownership-percentage?symbol={symbol}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_holdings_summary(self, date: str):
        """Get summary of all institutional holdings for a quarter"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-holdings-summary?date={date}"
        return self.make_req(url)
    
    def get_institutional_portfolio_changes(self, cik: str, current_date: str, previous_date: str):
        """Compare institutional portfolio between two quarters"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-portfolio-changes?cik={cik}&current_date={current_date}&previous_date={previous_date}"
        return self.make_req(url)
    
    def get_top_institutional_holders(self, symbol: str, date: Optional[str] = None, limit: int = 20):
        """Get top institutional holders by position size"""
        url = f"https://financialmodelingprep.com/api/v4/top-institutional-holders?symbol={symbol}&limit={limit}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_activity_feed(self, page: int = 0, limit: int = 100):
        """Get real-time institutional activity feed"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-activity-feed?page={page}&limit={limit}"
        return self.make_req(url)
    
    def get_institutional_flow_analysis(self, symbol: str, quarters: int = 4):
        """Analyze institutional flow trends"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-flow-analysis?symbol={symbol}&quarters={quarters}"
        return self.make_req(url)
    
    def get_institutional_concentration_analysis(self, symbol: str, date: Optional[str] = None):
        """Analyze institutional ownership concentration"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-concentration?symbol={symbol}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_position_changes(self, symbol: str, quarters: int = 4):
        """Track institutional position changes over time"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-position-changes?symbol={symbol}&quarters={quarters}"
        return self.make_req(url)
    
    def get_largest_institutional_positions(self, cik: str, date: Optional[str] = None, limit: int = 20):
        """Get largest positions for an institution"""
        url = f"https://financialmodelingprep.com/api/v4/largest-institutional-positions?cik={cik}&limit={limit}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_new_positions(self, cik: str, current_date: str, previous_date: str):
        """Get new positions added by institution"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-new-positions?cik={cik}&current_date={current_date}&previous_date={previous_date}"
        return self.make_req(url)
    
    def get_institutional_sold_positions(self, cik: str, current_date: str, previous_date: str):
        """Get positions sold by institution"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-sold-positions?cik={cik}&current_date={current_date}&previous_date={previous_date}"
        return self.make_req(url)
    
    def get_institutional_portfolio_performance(self, cik: str, periods: str = "1q,1y"):
        """Calculate institutional portfolio performance"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-portfolio-performance?cik={cik}&periods={periods}"
        return self.make_req(url)
    
    def get_institutional_sector_allocation(self, cik: str, date: Optional[str] = None):
        """Get institutional portfolio sector allocation"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-sector-allocation?cik={cik}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_holding_period(self, cik: str, symbol: str):
        """Analyze how long institution has held a position"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-holding-period?cik={cik}&symbol={symbol}"
        return self.make_req(url)
    
    def get_institutional_turnover_analysis(self, cik: str, quarters: int = 4):
        """Analyze institutional portfolio turnover"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-turnover?cik={cik}&quarters={quarters}"
        return self.make_req(url)
    
    def get_institutional_overlap_analysis(self, cik1: str, cik2: str, date: Optional[str] = None):
        """Analyze portfolio overlap between institutions"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-overlap?cik1={cik1}&cik2={cik2}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_ownership_trends(self, symbol: str, years: int = 3):
        """Get long-term institutional ownership trends"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-ownership-trends?symbol={symbol}&years={years}"
        return self.make_req(url)
    
    def get_institutional_smart_money_tracking(self, strategy: str = "top_performers", limit: int = 50):
        """Track institutional 'smart money' movements"""
        url = f"https://financialmodelingprep.com/api/v4/smart-money-tracking?strategy={strategy}&limit={limit}"
        return self.make_req(url)
    
    def get_institutional_13f_summary(self, quarter: str, year: int):
        """Get quarterly 13F filing summary statistics"""
        url = f"https://financialmodelingprep.com/api/v4/13f-summary?quarter={quarter}&year={year}"
        return self.make_req(url)
    
    def get_institutional_holdings_by_market_cap(self, market_cap_range: str = "large", date: Optional[str] = None, limit: int = 100):
        """Get institutional holdings by market cap category"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-holdings-by-market-cap?range={market_cap_range}&limit={limit}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_consensus_positions(self, symbol: str, min_institutions: int = 10):
        """Get stocks with institutional consensus (high agreement)"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-consensus?symbol={symbol}&min_institutions={min_institutions}"
        return self.make_req(url)
    
    def get_institutional_contrarian_positions(self, date: Optional[str] = None, limit: int = 50):
        """Find institutional contrarian positions"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-contrarian?limit={limit}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_13f_calendar(self, from_date: str, to_date: str):
        """Get 13F filing calendar and deadlines"""
        url = f"https://financialmodelingprep.com/api/v4/13f-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_institutional_risk_analysis(self, cik: str, date: Optional[str] = None):
        """Analyze institutional portfolio risk metrics"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-risk-analysis?cik={cik}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_style_analysis(self, cik: str, date: Optional[str] = None):
        """Analyze institutional investment style"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-style-analysis?cik={cik}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_performance_attribution(self, cik: str, period: str = "1y"):
        """Get institutional performance attribution"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-performance-attribution?cik={cik}&period={period}"
        return self.make_req(url)
    
    def get_institutional_crowded_trades(self, date: Optional[str] = None, limit: int = 50):
        """Identify crowded institutional trades"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-crowded-trades?limit={limit}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_holdings_alerts(self, symbols: Optional[str] = None, institutions: Optional[str] = None, threshold: float = 5.0):
        """Set up institutional holdings change alerts"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-holdings-alerts?threshold={threshold}"
        if symbols:
            url += f"&symbols={symbols}"
        if institutions:
            url += f"&institutions={institutions}"
        return self.make_req(url)
    
    def get_institutional_13f_search(self, institution_name: Optional[str] = None, cik: Optional[str] = None, limit: int = 50):
        """Search for institutions by name or CIK"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-search?limit={limit}"
        if institution_name:
            url += f"&name={institution_name}"
        if cik:
            url += f"&cik={cik}"
        return self.make_req(url)
    
    def get_institutional_ownership_comparison(self, symbols: str, date: Optional[str] = None):
        """Compare institutional ownership across multiple stocks"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-ownership-comparison?symbols={symbols}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_institutional_13f_export(self, cik: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None, format: str = "csv"):
        """Export institutional 13F data"""
        url = f"https://financialmodelingprep.com/api/v4/13f-export?format={format}"
        if cik:
            url += f"&cik={cik}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        return self.make_req(url)

    # Insider Trading Endpoints
    def get_insider_trading(self, symbol: str, limit: int = 100):
        """Get insider trading transactions for a stock"""
        url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_insider_trading_rss_feed(self, page: int = 0):
        """Get insider trading RSS feed"""
        url = f"https://financialmodelingprep.com/api/v4/insider-trading-rss-feed?page={page}"
        return self.make_req(url)
    
    def get_insider_roaster(self, symbol: str):
        """Get list of company insiders"""
        url = f"https://financialmodelingprep.com/api/v4/insider-roaster?symbol={symbol}"
        return self.make_req(url)
    
    def get_insider_roaster_statistic(self, symbol: str):
        """Get insider trading statistics"""
        url = f"https://financialmodelingprep.com/api/v4/insider-roaster-statistic?symbol={symbol}"
        return self.make_req(url)
    
    def get_insider_trading_by_person(self, person_name: str, limit: int = 50):
        """Get insider trading by specific person"""
        url = f"https://financialmodelingprep.com/api/v4/insider-trading-person?name={person_name}&limit={limit}"
        return self.make_req(url)
    
    def get_insider_trading_by_symbol(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 100):
        """Get insider trading for specific symbol with date range"""
        url = f"https://financialmodelingprep.com/api/v4/insider-trading-symbol?symbol={symbol}&limit={limit}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        return self.make_req(url)
    
    def get_insider_trading_sentiment(self, symbol: str, period: str = "3m"):
        """Get insider trading sentiment analysis"""
        url = f"https://financialmodelingprep.com/api/v4/insider-trading-sentiment?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_insider_buying_activity(self, symbol: Optional[str] = None, limit: int = 100):
        """Get recent insider buying activity"""
        url = f"https://financialmodelingprep.com/api/v4/insider-buying-activity?limit={limit}"
        if symbol:
            url += f"&symbol={symbol}"
        return self.make_req(url)
    
    def get_insider_selling_activity(self, symbol: Optional[str] = None, limit: int = 100):
        """Get recent insider selling activity"""
        url = f"https://financialmodelingprep.com/api/v4/insider-selling-activity?limit={limit}"
        if symbol:
            url += f"&symbol={symbol}"
        return self.make_req(url)
    
    def get_insider_ownership_percentage(self, symbol: str):
        """Get percentage of shares owned by insiders"""
        url = f"https://financialmodelingprep.com/api/v4/insider-ownership-percentage?symbol={symbol}"
        return self.make_req(url)
    
    def get_insider_trading_calendar(self, from_date: str, to_date: str):
        """Get insider trading calendar for date range"""
        url = f"https://financialmodelingprep.com/api/v4/insider-trading-calendar?from={from_date}&to={to_date}"
        return self.make_req(url)
    
    def get_insider_net_activity(self, symbol: str, period: str = "3m"):
        """Get net insider activity (buying vs selling)"""
        url = f"https://financialmodelingprep.com/api/v4/insider-net-activity?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_insider_trading_summary(self, symbol: str, period: str = "1y"):
        """Get insider trading summary statistics"""
        url = f"https://financialmodelingprep.com/api/v4/insider-trading-summary?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_insider_trading_volume_analysis(self, symbol: str, days: int = 30):
        """Analyze insider trading volume trends"""
        url = f"https://financialmodelingprep.com/api/v4/insider-volume-analysis?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_insider_trading_timing_analysis(self, symbol: str, limit: int = 50):
        """Analyze timing of insider trades relative to events"""
        url = f"https://financialmodelingprep.com/api/v4/insider-timing-analysis?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_insider_trading_price_impact(self, symbol: str, days_around: int = 5):
        """Analyze price impact of insider trading"""
        url = f"https://financialmodelingprep.com/api/v4/insider-price-impact?symbol={symbol}&days_around={days_around}"
        return self.make_req(url)
    
    def get_insider_trading_clustering(self, symbol: str, days_window: int = 30):
        """Identify clusters of insider trading activity"""
        url = f"https://financialmodelingprep.com/api/v4/insider-clustering?symbol={symbol}&window={days_window}"
        return self.make_req(url)
    
    def get_insider_trading_by_transaction_type(self, symbol: str, transaction_type: str = "P", limit: int = 100):
        """Get insider trading by transaction type (P=Purchase, S=Sale, etc.)"""
        url = f"https://financialmodelingprep.com/api/v4/insider-by-transaction-type?symbol={symbol}&type={transaction_type}&limit={limit}"
        return self.make_req(url)
    
    def get_insider_trading_by_relationship(self, symbol: str, relationship: str = "CEO", limit: int = 50):
        """Get insider trading by insider relationship to company"""
        url = f"https://financialmodelingprep.com/api/v4/insider-by-relationship?symbol={symbol}&relationship={relationship}&limit={limit}"
        return self.make_req(url)
    
    def get_insider_trading_largest_transactions(self, symbol: Optional[str] = None, min_value: int = 1000000, limit: int = 50):
        """Get largest insider trading transactions"""
        url = f"https://financialmodelingprep.com/api/v4/insider-largest-transactions?min_value={min_value}&limit={limit}"
        if symbol:
            url += f"&symbol={symbol}"
        return self.make_req(url)
    
    def get_insider_trading_frequency(self, symbol: str, insider_name: Optional[str] = None):
        """Analyze frequency of insider trading"""
        url = f"https://financialmodelingprep.com/api/v4/insider-frequency?symbol={symbol}"
        if insider_name:
            url += f"&name={insider_name}"
        return self.make_req(url)
    
    def get_insider_ownership_changes(self, symbol: str, quarters: int = 4):
        """Track changes in insider ownership over time"""
        url = f"https://financialmodelingprep.com/api/v4/insider-ownership-changes?symbol={symbol}&quarters={quarters}"
        return self.make_req(url)
    
    def get_insider_trading_before_earnings(self, symbol: str, days_before: int = 30, limit: int = 20):
        """Get insider trading activity before earnings announcements"""
        url = f"https://financialmodelingprep.com/api/v4/insider-before-earnings?symbol={symbol}&days_before={days_before}&limit={limit}"
        return self.make_req(url)
    
    def get_insider_trading_after_earnings(self, symbol: str, days_after: int = 30, limit: int = 20):
        """Get insider trading activity after earnings announcements"""
        url = f"https://financialmodelingprep.com/api/v4/insider-after-earnings?symbol={symbol}&days_after={days_after}&limit={limit}"
        return self.make_req(url)
    
    def get_insider_trading_screener(self, min_transaction_value: Optional[int] = None, max_days_ago: int = 30,
                                   transaction_type: Optional[str] = None, sector: Optional[str] = None, limit: int = 50):
        """Screen for insider trading based on criteria"""
        url = f"https://financialmodelingprep.com/api/v4/insider-screener?max_days={max_days_ago}&limit={limit}"
        if min_transaction_value:
            url += f"&min_value={min_transaction_value}"
        if transaction_type:
            url += f"&type={transaction_type}"
        if sector:
            url += f"&sector={sector}"
        return self.make_req(url)
    
    def get_insider_trading_alerts(self, symbols: Optional[str] = None, min_value: int = 100000, 
                                 transaction_types: str = "P,S"):
        """Set up insider trading alerts"""
        url = f"https://financialmodelingprep.com/api/v4/insider-alerts?min_value={min_value}&types={transaction_types}"
        if symbols:
            url += f"&symbols={symbols}"
        return self.make_req(url)
    
    def get_insider_trading_performance(self, symbol: str, insider_name: Optional[str] = None, period: str = "1y"):
        """Analyze performance of insider trades"""
        url = f"https://financialmodelingprep.com/api/v4/insider-performance?symbol={symbol}&period={period}"
        if insider_name:
            url += f"&name={insider_name}"
        return self.make_req(url)
    
    def get_insider_trading_patterns(self, symbol: str, pattern_type: str = "seasonal"):
        """Identify patterns in insider trading behavior"""
        url = f"https://financialmodelingprep.com/api/v4/insider-patterns?symbol={symbol}&pattern={pattern_type}"
        return self.make_req(url)
    
    def get_insider_trading_correlation(self, symbol: str, metric: str = "price", period: int = 252):
        """Analyze correlation between insider trading and stock metrics"""
        url = f"https://financialmodelingprep.com/api/v4/insider-correlation?symbol={symbol}&metric={metric}&period={period}"
        return self.make_req(url)
    
    def get_insider_trading_by_sector(self, sector: str, transaction_type: Optional[str] = None, limit: int = 100):
        """Get insider trading activity by sector"""
        url = f"https://financialmodelingprep.com/api/v4/insider-by-sector?sector={sector}&limit={limit}"
        if transaction_type:
            url += f"&type={transaction_type}"
        return self.make_req(url)
    
    def get_insider_trading_momentum(self, symbol: str, days: int = 30):
        """Calculate insider trading momentum"""
        url = f"https://financialmodelingprep.com/api/v4/insider-momentum?symbol={symbol}&days={days}"
        return self.make_req(url)
    
    def get_insider_trading_concentration(self, symbol: str, date: Optional[str] = None):
        """Analyze concentration of insider ownership"""
        url = f"https://financialmodelingprep.com/api/v4/insider-concentration?symbol={symbol}"
        if date:
            url += f"&date={date}"
        return self.make_req(url)
    
    def get_insider_vs_institutional_activity(self, symbol: str, period: str = "1y"):
        """Compare insider vs institutional trading activity"""
        url = f"https://financialmodelingprep.com/api/v4/insider-vs-institutional?symbol={symbol}&period={period}"
        return self.make_req(url)
    
    def get_insider_trading_blackout_periods(self, symbol: str):
        """Get information about insider trading blackout periods"""
        url = f"https://financialmodelingprep.com/api/v4/insider-blackout-periods?symbol={symbol}"
        return self.make_req(url)
    
    def get_insider_10b5_1_plans(self, symbol: str, limit: int = 50):
        """Get information about 10b5-1 trading plans"""
        url = f"https://financialmodelingprep.com/api/v4/insider-10b5-1-plans?symbol={symbol}&limit={limit}"
        return self.make_req(url)
    
    def get_insider_trading_compliance(self, symbol: str, insider_name: Optional[str] = None):
        """Analyze insider trading compliance and timing"""
        url = f"https://financialmodelingprep.com/api/v4/insider-compliance?symbol={symbol}"
        if insider_name:
            url += f"&name={insider_name}"
        return self.make_req(url)
    
    def get_insider_trading_anomalies(self, symbol: Optional[str] = None, days: int = 30, limit: int = 50):
        """Detect unusual insider trading activity"""
        url = f"https://financialmodelingprep.com/api/v4/insider-anomalies?days={days}&limit={limit}"
        if symbol:
            url += f"&symbol={symbol}"
        return self.make_req(url)
    
    def get_insider_trading_export(self, symbol: Optional[str] = None, from_date: Optional[str] = None, 
                                 to_date: Optional[str] = None, format: str = "csv"):
        """Export insider trading data"""
        url = f"https://financialmodelingprep.com/api/v4/insider-export?format={format}"
        if symbol:
            url += f"&symbol={symbol}"
        if from_date:
            url += f"&from={from_date}"
        if to_date:
            url += f"&to={to_date}"
        return self.make_req(url)

# ===== FMP API INITIALIZATION =====
# Financial Modeling Prep API configuration and authentication setup
import os
api_key = os.getenv('FMP_API_KEY', '{FMP_API_KEY}')  # Retrieve API key from environment or use placeholder

def initialize_fmp_tools(api_key: str):
    """Dynamically create FMP tool registry with validation for agent integration"""
    fmp_instance = fmp(api_key)

    # Build validated list of FMP API methods for AI agent toolchain
    tools = []
    for name in dir(fmp_instance):
        if (callable(getattr(fmp_instance, name)) and
            not name.startswith('_') and
            name not in ['api_key', 'make_req']):

            # Validate method implementation to ensure functional tools
            method = getattr(fmp_instance, name)
            try:
                # Inspect source code to confirm proper API implementation
                import inspect
                source = inspect.getsource(method)
                # Verify method contains actual API logic
                if ('return self.make_req' in source or
                    'return ' in source or
                    'url =' in source):
                    tools.append(method)
            except (OSError, TypeError):
                # Include methods where source inspection fails (built-ins)
                tools.append(method)

    return tools

# Create FMP tool registry for AI agent integration
fmp_tools = initialize_fmp_tools(api_key)

# Enhanced Web Search Agent with native Google search capabilities
enhanced_web_search_agent = Agent(
    name="Enhanced_Web_Search_Agent",
    model="gemini-2.5-flash-lite",
    planner=thinking_planner,
    instruction="You are an advanced web search specialist with comprehensive internet research capabilities for financial markets and investment analysis. "
    "Your expertise encompasses: real-time financial news aggregation, market sentiment analysis from multiple sources, "
    "regulatory filing research, competitor intelligence gathering, industry trend identification, and economic data verification. "
    
    "🌐 ADVANCED SEARCH CAPABILITIES: "
    "You excel at multi-source information synthesis, real-time market data verification, financial news analysis and impact assessment, "
    "company research across global markets, regulatory and SEC filing analysis, earnings call transcript analysis, "
    "analyst report compilation, economic indicator tracking, and cryptocurrency and alternative investment research. "
    
    "🧠 AUTONOMOUS RESEARCH METHODOLOGY: "
    "You are a highly resourceful web researcher who builds comprehensive market intelligence even when initial searches yield limited results: "
    "- When direct company information is sparse, you search for peer companies, industry reports, and sector analysis "
    "- If financial news is limited, you explore regulatory filings, investor relations pages, and professional networks "
    "- When market data appears incomplete, you cross-reference multiple financial platforms and data aggregators "
    "- You always triangulate information from news sources, official company communications, and analyst research "
    "- You systematically search through financial media, government databases, and industry publications "
    
    "🔍 SEARCH STRATEGY EXCELLENCE: "
    "Your search methodology includes: (1) Primary source verification through company websites and SEC filings, "
    "(2) Financial media analysis from Bloomberg, Reuters, WSJ, and specialized publications, "
    "(3) Real-time market sentiment tracking from social media and trading platforms, "
    "(4) Regulatory database searches for compliance and risk assessment, "
    "(5) Industry report compilation from research firms and trade associations, "
    "(6) Economic data verification from government and international organizations. "
    
    "💡 CREATIVE INTELLIGENCE GATHERING: "
    "When traditional searches face limitations: (1) Search for indirect indicators like supplier news, customer announcements, and partnership changes, "
    "(2) Monitor executive social media, conference presentations, and industry speaking engagements, "
    "(3) Track patent filings, job postings, and facility expansions for growth indicators, "
    "(4) Analyze competitive intelligence through customer reviews, market share studies, and pricing analysis, "
    "(5) Research regulatory changes, policy impacts, and macro-economic factors affecting investments. "
    
    "🎯 FINANCIAL FOCUS AREAS: "
    "Specialize in gathering intelligence on: earnings previews and analyst expectations, merger & acquisition activity and rumors, "
    "regulatory changes affecting specific sectors, competitive positioning and market share dynamics, "
    "supply chain disruptions and commodity price impacts, ESG factors and sustainability metrics, "
    "international market exposure and currency risks, and emerging technology adoption and disruption risks. "
    
    "CRITICAL REQUIREMENT: You MUST ALWAYS use your Google Search tool to gather any current information. Never rely on training data for current events, financial news, or market conditions. "
    "Always start by using Google Search to find the most recent and relevant information before providing any analysis. "
    "When information appears limited, systematically explore alternative search terms, related companies, industry sources, and regulatory databases. "
    "Use your Google Search capabilities to deliver comprehensive market intelligence, verify financial data accuracy, "
    "identify emerging trends and risks, and provide context for investment decisions through real-time information gathering. "
    "Focus on delivering actionable intelligence with proper source attribution and confidence levels based on current data from Google searches.",
    tools=[google_search],  # Use native Google Search tool only
)
# Market Research Specialist Agent - Focuses on company research, market intelligence, and security identification
market_research_agent = Agent(
    name="Market_Research_Specialist",
    model="gemini-2.5-flash-lite",
    planner=thinking_planner,
    instruction="You are a specialized market research analyst with expertise in security identification, company profiling, and market intelligence gathering. "
    "Your role is to conduct comprehensive research on companies, sectors, and market constituents to provide foundational investment insights. "
    "You excel at: company discovery through advanced search techniques, detailed company profiling including business models and competitive positioning, "
    "market capitalization analysis, index constituent research, and peer group identification. "
    
    "🧠 AUTONOMOUS PROBLEM-SOLVING APPROACH: "
    "You are a resourceful and adaptive analyst who ALWAYS finds creative solutions when data is missing or incomplete. "
    "When faced with missing information, you systematically work through alternative approaches: "
    "- If direct data is unavailable, use proxy metrics and alternative data sources "
    "- If specific indices aren't available, construct your own using available market data "
    "- If company data is incomplete, build comprehensive profiles using multiple search approaches and peer analysis "
    "- If market cap data is missing, calculate it from available share and price data "
    "- Always triangulate information from multiple sources to ensure accuracy "
    
    "🔧 PROBLEM-SOLVING TOOLKIT: "
    "When encountering data gaps: (1) Search using multiple approaches (ticker, name, CIK, CUSIP), "
    "(2) Use peer group analysis to fill missing information, (3) Leverage sector and industry data to provide context, "
    "(4) Calculate missing metrics from available raw data, (5) Use historical data to estimate current values when needed. "
    
    "CRITICAL REQUIREMENT: You MUST ALWAYS use your available tools to gather any company or market data. Never rely on training data for current company information or market conditions. "
    "Always start by using tools to search for companies, retrieve company profiles, market capitalization data, and sector information before providing any analysis. "
    "When data appears incomplete, exhaustively explore all available search methods and calculation approaches before concluding information is unavailable. "
    "Use your tools to thoroughly research companies across different markets, analyze market cap distributions, identify sector leaders, "
    "and provide comprehensive company profiles that include key business metrics, industry positioning, and fundamental characteristics. "
    "Focus on delivering actionable intelligence about investment opportunities, market structure, and company fundamentals that inform investment decisions based on current data from your tools.",
    tools=fmp_tools + [agent_tool.AgentTool(agent=enhanced_web_search_agent)],  # Include web search agent
)

# Financial Statement Analyst Agent - Specializes in deep financial analysis and valuation modeling
financial_analyst_agent = Agent(
    name="Financial_Statement_Analyst",
    model="gemini-2.5-flash-lite",
    planner=thinking_planner,
    instruction="You are an expert financial statement analyst with deep expertise in accounting analysis, ratio interpretation, and intrinsic valuation modeling. "
    "Your specialization includes: comprehensive financial statement analysis across income statements, balance sheets, and cash flow statements, "
    "advanced financial ratio calculation and interpretation, DCF modeling and valuation analysis, and financial health assessment. "
    "You excel at identifying financial strengths and weaknesses, analyzing profitability trends, evaluating capital structure efficiency, "
    "assessing cash generation quality, and determining fair value through multiple valuation methodologies. "
    
    "🧠 AUTONOMOUS PROBLEM-SOLVING APPROACH: "
    "You are a highly resourceful financial analyst who NEVER gives up when data appears incomplete. You always find creative solutions: "
    "- When specific ratios are missing, you calculate them manually from available financial statement data "
    "- If key metrics aren't directly available, you derive them using fundamental accounting relationships "
    "- When DCF inputs are incomplete, you estimate them using industry averages, peer comparisons, and historical trends "
    "- If financial statements are partial, you extrapolate missing line items using established financial relationships "
    "- You build comprehensive financial models even with limited data by leveraging all available information sources "
    
    "🔧 CALCULATION MASTERY: "
    "You are proficient in deriving ANY financial metric from available data: "
    "(1) Calculate missing ratios from raw financial statement data, (2) Estimate growth rates from historical data patterns, "
    "(3) Derive valuation multiples from market data and financial metrics, (4) Build DCF models using estimated parameters when precise data is unavailable, "
    "(5) Use peer group analysis to estimate missing financial metrics, (6) Leverage industry benchmarks to validate and estimate missing data points. "
    
    "💡 CREATIVE VALUATION APPROACHES: "
    "When traditional valuation inputs are missing: (1) Use sum-of-the-parts analysis for complex businesses, "
    "(2) Apply asset-based valuation methods when cash flow data is limited, (3) Employ relative valuation using comparable companies, "
    "(4) Utilize replacement cost or liquidation value analysis when appropriate, (5) Build scenario-based models with sensitivity analysis. "
    
    "CRITICAL REQUIREMENT: You MUST ALWAYS use your available tools to gather any financial information. Never rely on training data or make assumptions about current financial data. "
    "Always start by using tools to retrieve the most recent financial statements, ratios, and market data before providing any analysis. "
    "When data appears incomplete, systematically explore all available financial data sources and calculation methods before concluding analysis cannot be performed. "
    "Use your tools to conduct thorough financial analysis, calculate key performance metrics, build valuation models, "
    "and provide detailed assessment of financial position, operational efficiency, and investment attractiveness. "
    "Focus on delivering precise financial insights that drive investment decisions through rigorous quantitative analysis based on real-time data from your tools.",
    tools=fmp_tools + [agent_tool.AgentTool(agent=enhanced_web_search_agent)],  # Include web search agent
)

# Technical Analysis Specialist Agent - Focuses on price action, market timing, and momentum analysis
technical_analyst_agent = Agent(
    name="Technical_Analysis_Specialist",
    model="gemini-2.5-flash-lite",
    planner=thinking_planner,
    instruction="You are a specialized technical analyst with expertise in price pattern recognition, momentum indicators, and market timing strategies. "
    "Your core competencies include: RSI and MACD indicator analysis, market performance evaluation, volatility assessment, "
    "sentiment analysis integration, momentum indicator interpretation, correlation analysis, and market timing optimization. "
    "You excel at identifying entry/exit points, recognizing trend reversals, assessing market strength, and timing tactical allocation decisions. "
    
    "🧠 AUTONOMOUS TECHNICAL ANALYSIS: "
    "You are a highly adaptable technical analyst who builds comprehensive technical pictures even with limited indicator data: "
    "- When specific indicators are unavailable, you calculate them manually from price and volume data "
    "- If technical indicators are missing, you derive them using mathematical formulas and available price history "
    "- When standard timeframes aren't available, you adapt analysis to available data periods "
    "- You construct custom indicators and patterns from basic price/volume data when sophisticated tools aren't accessible "
    "- You always find alternative technical approaches when primary methods face data limitations "
    
    "🔧 INDICATOR CONSTRUCTION EXPERTISE: "
    "You can build ANY technical indicator from basic price data: "
    "(1) Calculate RSI, MACD, moving averages from raw price data, (2) Construct volatility measures using price ranges and standard deviations, "
    "(3) Build momentum indicators using price change calculations, (4) Create volume-based indicators from trading volume data, "
    "(5) Develop custom oscillators and trend-following indicators, (6) Construct support/resistance levels from price action analysis. "
    
    "📊 CREATIVE PATTERN RECOGNITION: "
    "When traditional charts aren't available: (1) Identify patterns using numerical price data analysis, "
    "(2) Detect trend changes using mathematical slope calculations, (3) Recognize consolidation periods through volatility analysis, "
    "(4) Spot breakout patterns using volume and price change correlation, (5) Build market timing models using multiple timeframe analysis. "
    
    "CRITICAL REQUIREMENT: You MUST ALWAYS use your available tools to gather any technical or market data. Never rely on training data for current market conditions. "
    "Always start by using tools to retrieve real-time price data, technical indicators, and market performance metrics before providing any analysis. "
    "When technical data appears limited, systematically calculate all possible indicators and patterns from available price/volume data. "
    "Use your tools to analyze technical indicators, evaluate market performance metrics, assess volatility conditions, "
    "monitor sentiment shifts, and identify momentum patterns that signal investment opportunities. "
    "Focus on providing precise timing recommendations, risk level assessments, and tactical insights that optimize portfolio performance based on real-time data from your tools.",
    tools=fmp_tools + [agent_tool.AgentTool(agent=enhanced_web_search_agent)],  # Include web search agent
)

# ETF and Index Specialist Agent - Specializes in passive investment vehicles and index-based strategies
etf_specialist_agent = Agent(
    name="ETF_Index_Specialist",
    model="gemini-2.5-flash-lite",
    planner=thinking_planner,
    instruction="You are a specialized ETF and index fund analyst with expertise in passive investment vehicle analysis and index-based portfolio construction. "
    "Your expertise encompasses: ETF holdings analysis, index fund structure evaluation, sector performance assessment, "
    "market breadth analysis, index rotation strategies, screening for optimal passive vehicles, and benchmark performance evaluation. "
    "You excel at analyzing ETF compositions, evaluating expense ratios, assessing tracking accuracy, identifying sector rotation opportunities, "
    "and constructing diversified passive portfolios that optimize risk-adjusted returns. "
    
    "🧠 CREATIVE INDEX CONSTRUCTION: "
    "You are a highly innovative ETF analyst who builds comprehensive passive investment strategies even when traditional indices are unavailable: "
    "- When specific indices are missing, you construct custom benchmarks using available market constituents and weighting methodologies "
    "- If ETF holdings data is incomplete, you reverse-engineer compositions using performance correlation analysis "
    "- When sector allocation data is limited, you estimate allocations using individual stock analysis and categorization "
    "- You create synthetic indices and tracking strategies using available individual securities and market data "
    "- You always find alternative passive investment approaches when standard ETF options face data limitations "
    
    "🔧 ETF ANALYSIS MASTERY: "
    "You can construct comprehensive ETF analysis from basic market data: "
    "(1) Calculate tracking error and performance metrics from price comparison analysis, (2) Estimate expense ratios impact using fee-adjusted return calculations, "
    "(3) Build sector allocation models from constituent analysis, (4) Construct correlation matrices for ETF selection optimization, "
    "(5) Develop custom benchmarks using market cap weighted methodologies, (6) Create rebalancing strategies using momentum and mean reversion analysis. "
    
    "📊 SYNTHETIC INDEXING CAPABILITIES: "
    "When traditional passive vehicles aren't available: (1) Build equal-weight and cap-weight indices from available securities, "
    "(2) Create factor-based indices using fundamental screening criteria, (3) Construct sector rotation models using relative performance analysis, "
    "(4) Develop momentum-based passive strategies using trend-following methodologies, (5) Build dividend-focused indices using yield and growth screening. "
    
    "CRITICAL REQUIREMENT: You MUST ALWAYS use your available tools to gather current ETF data, holdings information, and market data. Never rely on outdated training data. "
    "Always start by using tools to retrieve current ETF holdings, expense ratios, performance data, and sector allocations before providing any analysis. "
    "When ETF data appears incomplete, systematically construct alternative passive investment approaches using available market and security data. "
    "Use your tools to research ETF holdings, analyze sector allocations, evaluate market breadth indicators, "
    "screen for investment opportunities, and assess rotation patterns that inform strategic asset allocation. "
    "Focus on delivering comprehensive analysis of passive investment options with specific recommendations for portfolio optimization based on current data from your tools.",
    tools=fmp_tools + [agent_tool.AgentTool(agent=enhanced_web_search_agent)],  # Include web search agent
)

# Earnings and Analyst Coverage Specialist Agent - Focuses on earnings analysis and Wall Street research
earnings_specialist_agent = Agent(
    name="Earnings_Analyst_Specialist",
    model="gemini-2.5-flash-lite",
    planner=thinking_planner,
    instruction="You are a specialized earnings and analyst coverage expert with deep knowledge of earnings analysis, dividend research, and Wall Street consensus interpretation. "
    "Your specialization includes: earnings calendar analysis, earnings surprise evaluation, analyst recommendation synthesis, "
    "price target analysis, dividend yield and growth assessment, stock split impact analysis, and consensus estimate interpretation. "
    "You excel at identifying earnings opportunities, evaluating analyst sentiment shifts, assessing dividend sustainability, "
    "analyzing earnings quality, and interpreting Wall Street research to inform investment decisions. "
    
    "🧠 ADAPTIVE EARNINGS ANALYSIS: "
    "You are a highly resourceful earnings analyst who builds comprehensive earnings assessment even when consensus data is limited: "
    "- When analyst estimates are missing, you construct earnings models using historical trends and peer analysis "
    "- If consensus targets are unavailable, you build price targets using multiple valuation methodologies "
    "- When earnings guidance is incomplete, you estimate management targets using industry trends and company communications "
    "- You create comprehensive earnings analysis using fundamental growth drivers and business model assessment "
    "- You always find alternative earnings evaluation approaches when traditional consensus data faces limitations "
    
    "🔧 EARNINGS MODELING EXPERTISE: "
    "You can build comprehensive earnings analysis from available financial data: "
    "(1) Calculate earnings growth rates and trends from historical financial statements, (2) Estimate future earnings using revenue growth and margin analysis, "
    "(3) Build earnings quality assessments using cash flow and accrual analysis, (4) Construct dividend sustainability models using payout ratio and cash flow analysis, "
    "(5) Develop earnings surprise prediction models using historical variance patterns, (6) Create analyst sentiment indicators using recommendation trend analysis. "
    
    "📈 CREATIVE CONSENSUS BUILDING: "
    "When traditional analyst data isn't available: (1) Build earnings consensus using peer group valuation multiples, "
    "(2) Create price targets using sum-of-the-parts and comparable company analysis, (3) Estimate earnings revisions using management guidance and industry trends, "
    "(4) Develop earnings seasonality models using historical quarterly patterns, (5) Build earnings momentum indicators using surprise history and guidance trends. "
    
    "CRITICAL REQUIREMENT: You MUST ALWAYS use your available tools to gather current earnings data, analyst estimates, and corporate action information. Never rely on outdated training data. "
    "Always start by using tools to retrieve current earnings calendars, analyst recommendations, dividend information, and price targets before providing any analysis. "
    "When earnings data appears incomplete, systematically build alternative earnings models and consensus estimates using available financial and market data. "
    "Use your tools to track earnings calendars, analyze surprise patterns, evaluate analyst recommendations, "
    "assess price target movements, research dividend histories, and monitor corporate actions. "
    "Focus on delivering timely insights about earnings expectations, analyst sentiment, and income-generating opportunities based on real-time data from your tools.",
    tools=fmp_tools + [agent_tool.AgentTool(agent=enhanced_web_search_agent)],  # Include web search agent
)

# Institutional Flow Analyst Agent - Specializes in institutional investor behavior and smart money tracking
institutional_flow_agent = Agent(
    name="Institutional_Flow_Analyst",
    model="gemini-2.5-flash-lite",
    planner=thinking_planner,
    instruction="You are a specialized institutional flow analyst with expertise in tracking smart money movements, institutional investor behavior, and large-scale portfolio positioning. "
    "Your core competencies include: institutional holdings analysis, 13F filing interpretation, market cap-based institutional positioning, "
    "consensus vs. contrarian position identification, crowded trade analysis, institutional ownership comparison, and flow pattern recognition. "
    "You excel at identifying institutional buying/selling patterns, recognizing consensus trades, spotting contrarian opportunities, "
    "and analyzing how institutional flows impact security prices and market dynamics. "
    "CRITICAL REQUIREMENT: You MUST ALWAYS use your available tools to gather current institutional holdings data, 13F filings, and ownership information. Never rely on outdated training data. "
    "Always start by using tools to retrieve the most recent institutional holdings, 13F filings, and ownership data before providing any analysis. "
    "Use your tools to track institutional holdings changes, analyze 13F filings, identify crowded positions, "
    "evaluate consensus trades, and monitor institutional ownership patterns across market capitalizations. "
    "Focus on providing actionable insights about institutional investor behavior that can inform investment positioning and risk management based on current data from your tools.",
    tools=fmp_tools + [agent_tool.AgentTool(agent=enhanced_web_search_agent)],  # Include web search agent
)

# Insider Trading and Sentiment Analyst Agent - Specializes in insider activity and market sentiment analysis
insider_sentiment_agent = Agent(
    name="Insider_Sentiment_Analyst",
    model="gemini-2.5-flash-lite",
    planner=thinking_planner,
    instruction="You are a specialized insider trading and market sentiment analyst with expertise in corporate insider behavior, political trading patterns, and sentiment indicator interpretation. "
    "Your specialization includes: insider trading pattern analysis, insider vs. institutional activity comparison, "
    "anomaly detection in insider behavior, compliance analysis, Senate trading trend evaluation, and sentiment-driven investment insights. "
    "You excel at identifying meaningful insider transactions, recognizing unusual trading patterns, evaluating compliance with trading windows, "
    "and interpreting insider sentiment as leading indicators for security performance. "
    "CRITICAL REQUIREMENT: You MUST ALWAYS use your available tools to gather current insider trading data, Senate trading information, and sentiment indicators. Never rely on outdated training data. "
    "Always start by using tools to retrieve the most recent insider trading activity, Senate trading patterns, and sentiment data before providing any analysis. "
    "Use your tools to monitor insider trading activity, analyze trading anomalies, compare insider vs. institutional flows, "
    "track Senate trading patterns, and evaluate compliance metrics that indicate conviction levels. "
    "Focus on delivering insights about insider confidence, potential catalysts, and sentiment shifts that inform investment timing and positioning based on current data from your tools.",
    tools=fmp_tools + [agent_tool.AgentTool(agent=enhanced_web_search_agent)],  # Include web search agent
)

# Risk Management Specialist Agent - Focuses on portfolio risk assessment and market risk analysis
risk_management_agent = Agent(
    name="Risk_Management_Specialist",
    model="gemini-2.5-flash-lite",
    planner=thinking_planner,
    instruction="You are a specialized risk management analyst with expertise in portfolio risk assessment, market risk evaluation, and systematic risk monitoring. "
    "Your core competencies include: market risk metrics calculation, efficiency analysis, global market risk assessment, "
    "market cycle identification, institutional risk evaluation, performance comparison analysis, and historical risk pattern recognition. "
    "You excel at quantifying portfolio risks, identifying systemic risks, evaluating market efficiency, assessing correlation risks, "
    "and developing risk-adjusted performance metrics that optimize portfolio construction. "
    
    "🧠 ADAPTIVE RISK ASSESSMENT: "
    "You are a highly resourceful risk analyst who builds comprehensive risk models even when standard risk metrics are unavailable: "
    "- When VaR or risk metrics are missing, you calculate them from price history and volatility data "
    "- If correlation data is incomplete, you derive correlations from available price movements and market relationships "
    "- When risk-free rates aren't accessible, you estimate them using available market benchmarks and yield data "
    "- You construct custom risk models using fundamental volatility relationships and market dynamics "
    "- You always find alternative risk measurement approaches when traditional metrics face data constraints "
    
    "🔧 RISK CALCULATION MASTERY: "
    "You can construct ANY risk metric from available market data: "
    "(1) Calculate volatility measures from price movements and standard deviations, (2) Build correlation matrices from price relationship analysis, "
    "(3) Construct Value-at-Risk models using historical simulation and parametric methods, (4) Develop beta calculations from market sensitivity analysis, "
    "(5) Create Sharpe ratios and risk-adjusted returns from performance and volatility data, (6) Build stress test scenarios using historical market events. "
    
    "⚡ SYSTEMATIC RISK DETECTION: "
    "When comprehensive risk data is limited: (1) Identify sector concentration risks through portfolio composition analysis, "
    "(2) Detect correlation clustering using available price relationships, (3) Assess liquidity risks through volume and market cap analysis, "
    "(4) Evaluate tail risks using extreme value analysis of available price data, (5) Monitor regime changes through volatility pattern recognition. "
    
    "CRITICAL REQUIREMENT: You MUST ALWAYS use your available tools to gather current market data, risk metrics, and performance information. Never rely on outdated training data. "
    "Always start by using tools to retrieve real-time market risk indicators, volatility data, and performance metrics before providing any analysis. "
    "When risk data appears incomplete, systematically calculate all possible risk measures from available market and price data. "
    "Use your tools to calculate risk metrics, analyze market efficiency indicators, evaluate global market risks, "
    "identify market cycle patterns, assess institutional risk exposures, and compare risk-adjusted performance across time periods. "
    "Focus on delivering comprehensive risk assessments, systematic risk warnings, and risk-optimized portfolio recommendations based on current data from your tools.",
    tools=fmp_tools + [agent_tool.AgentTool(agent=enhanced_web_search_agent)],  # Include web search agent
)

# Portfolio Manager Agent - Orchestrates overall investment strategy by coordinating specialist insights
investment_banker_agent = Agent(
    name="NomiAI_Financial_Assistant",
    model="gemini-2.5-flash-lite",
    planner=thinking_planner,
    instruction="Sei NomiAI, un assistente finanziario AI esperto e professionale che fornisce analisi di mercato, consigli di investimento e ricerche finanziarie. "
    "Mantieni sempre memoria delle conversazioni precedenti e rispondi in modo conversazionale e naturale in italiano. "

    "🎯 RUOLO E PERSONALITÀ: "
    "Sei un esperto analista finanziario con anni di esperienza nei mercati globali. Comunichi in modo chiaro, professionale ma accessibile. "
    "Mantieni il contesto delle conversazioni precedenti e fai riferimento a discussioni passate quando appropriato. "
    "Sei proattivo nel suggerire analisi approfondite e opportunità di investimento. "

    "🛠️ STRUMENTI SPECIALIZZATI: "
    "Hai accesso a un team di analisti specializzati: Web Search per notizie e dati in tempo reale, Market Research per analisi settoriali, "
    "Financial Analyst per valutazioni e due diligence, Technical Analyst per analisi tecnica, ETF Specialist per investimenti passivi, "
    "Earnings Specialist per analisi degli utili, Institutional Flow per flussi istituzionali, Insider Sentiment per sentiment di mercato, "
    "Risk Management per gestione del rischio. "

    "📊 APPROCCIO ALL'ANALISI: "
    "Per ogni richiesta: (1) Usa sempre prima il Web Search per informazioni aggiornate, (2) Coordina gli specialisti appropriati, "
    "(3) Sintetizza i dati in consigli chiari e azionabili, (4) Fornisci sempre valutazioni del rischio, "
    "(5) Suggerisci approfondimenti correlati quando pertinenti. "

    "💬 STILE COMUNICATIVO: "
    "Rispondi sempre in italiano con un tono professionale ma accessibile. Usa terminologia finanziaria appropriata ma spiegala quando necessario. "
    "Mantieni memoria del contesto conversazionale e fai riferimento a discussioni precedenti. "
    "Se non hai informazioni specifiche, usa il Web Search Agent per trovarle - non dire mai di non avere informazioni senza aver cercato. "

    "IMPORTANTE: Ricorda sempre il contesto delle conversazioni precedenti e costruisci su di esse. Sii conversazionale e naturale, "
    "mantenendo alta qualità professionale nell'analisi finanziaria.",
    tools=[
        agent_tool.AgentTool(agent=enhanced_web_search_agent),  # Added enhanced web search agent
        agent_tool.AgentTool(agent=market_research_agent),
        agent_tool.AgentTool(agent=financial_analyst_agent),
        agent_tool.AgentTool(agent=technical_analyst_agent),
        agent_tool.AgentTool(agent=etf_specialist_agent),
        agent_tool.AgentTool(agent=earnings_specialist_agent),
        agent_tool.AgentTool(agent=institutional_flow_agent),
        agent_tool.AgentTool(agent=insider_sentiment_agent),
        agent_tool.AgentTool(agent=risk_management_agent),
    ],
)
# Set root agent and run system
root_agent = investment_banker_agent
# ===== CHAT SESSION MANAGEMENT =====
# In-memory storage for active chat sessions and conversation history
history = {}
session_service = InMemorySessionService()

class AgentSession():
    """Manages individual chat sessions with conversation state and AI agent interaction"""
    def __init__(self, agent, cid):
        self.agent = agent
        self.list = []  # Conversation history
        self.i = cid    # Chat session identifier
        self.runner = None  # Persistent runner for conversation continuity
    
    async def initialize_session(self):
        """Create new AI agent session with default configuration"""
        self.session = await session_service.create_session(
            app_name="NomiAi",
            user_id="user",
            state={}
        )
        return self.session
    
    async def process_input(self, msg, client_history=None):
        """Process user message through AI agent and return response with conversation logging"""
        try:
            # Initialize session if not already created
            if not hasattr(self, 'session') or self.session is None:
                await self.initialize_session()

            # Initialize runner if not exists to maintain conversation context
            if not hasattr(self, 'runner') or self.runner is None:
                self.runner = Runner(agent=self.agent, app_name="NomiAi", session_service=session_service)

            # Build conversation context from history
            context_msg = msg
            if client_history and len(client_history) > 1:
                # Include recent conversation context (last 6 messages for efficiency)
                recent_history = client_history[-6:] if len(client_history) > 6 else client_history
                context_parts = []
                for hist in recent_history[:-1]:  # Exclude current message
                    role = "Utente" if hist.get('role') == 'user' else "Assistente"
                    context_parts.append(f"{role}: {hist.get('content', '')}")

                if context_parts:
                    context_msg = f"Contesto conversazione precedente:\n{chr(10).join(context_parts)}\n\nMessaggio corrente: {msg}"

            # Convert user message to proper Content format for AI processing
            content = types.Content(role='user', parts=[types.Part(text=context_msg)])

            response_events = []
            print(f"Session ID: {self.session.id}")
            # Stream AI agent response events
            async for event in self.runner.run_async(
                user_id="user",
                session_id=self.session.id,
                new_message=content
            ):
                if hasattr(event, 'content') and event.content:
                    # Parse content parts from agent response
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            # Extract only text content, ignore function calls
                            if hasattr(part, 'text') and part.text is not None and part.text.strip():
                                response_events.append(part.text)
                    else:
                        response_events.append(str(event.content))

            # Compile valid response text
            valid_responses = [resp for resp in response_events if resp is not None]
            response = "\n".join(valid_responses) if valid_responses else "Mi dispiace, non sono riuscito a elaborare una risposta. Potresti riprovare?"
            print(f"Response: {response}")

            # Log conversation exchange for session history
            self.list.append({
                "user_message": msg,
                "agent_response": response,
                "timestamp": str(datetime.now())
            })
            return response
        except Exception as e:
            print(f"Error in process_input: {e}")
            # Return error message instead of infinite retry
            return "Si è verificato un errore. Riprova tra qualche momento."
    
    def dict(self):
        return self.list

# ===== FLASK WEB APPLICATION =====
# Web server setup for NomiAI chat interface
app = Flask(__name__)

async def send_msg(chat_id, msg, client_history=None):
    """Route message to appropriate chat session and return AI agent response"""
    # Create new session if this chat_id hasn't been seen before
    if chat_id not in history:
        agent_session = AgentSession(agent=root_agent, cid=chat_id)
        await agent_session.initialize_session()
        history[chat_id] = agent_session

    # Process message through existing session
    session = history[chat_id]
    response = await session.process_input(msg, client_history)
    print(f"Chat {chat_id}: {msg} -> {response}")
    return response

@app.route('/')
async def home():
    """Serve main chat interface HTML page"""
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
async def chat():
    """Handle incoming chat messages and return AI agent responses"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Messaggio non fornito"}), 400

        message = data['message']
        chat_id = data.get('chat_id', 'default')
        client_history = data.get('history', [])

        # Process message through AI agent with history context
        response = await send_msg(chat_id, message, client_history)

        return jsonify({
            "response": str(response),
            "chat_id": chat_id,
            "status": "success"
        })
    
    except Exception as e:
        print(f"Errore nel chat endpoint: {str(e)}")
        return jsonify({
            "error": f"Errore del server: {str(e)}",
            "status": "error"
        }), 500
asgi_app = WsgiToAsgi(app)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
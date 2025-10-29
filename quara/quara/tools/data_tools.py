"""
Data downloading and analysis tools for QuARA
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta


class DataDownloader:
    """Download data from various sources"""
    
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    async def download_financial_data(self, 
                                     ticker: str,
                                     start_date: str = None,
                                     end_date: str = None,
                                     source: str = "yfinance") -> Dict[str, Any]:
        """
        Download financial data for a ticker
        
        Args:
            ticker: Stock ticker symbol (e.g., 'SPY')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source ('yfinance', 'alphavantage', etc.)
            
        Returns:
            Dictionary with data and metadata
        """
        try:
            if source == "yfinance":
                return await self._download_yfinance(ticker, start_date, end_date)
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
        except Exception as e:
            self.logger.error(f"Failed to download data for {ticker}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _download_yfinance(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Download data using yfinance"""
        try:
            import yfinance as yf
            
            self.logger.info(f"Downloading {ticker} data from yfinance...")
            
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                return {
                    "success": False,
                    "error": "No data returned"
                }
            
            # Ensure the index is named 'Date' for consistency
            if data.index.name != 'Date':
                data.index.name = 'Date'
            
            # Save to cache with Date as a proper column
            cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}.csv"
            data.to_csv(cache_file)
            
            return {
                "success": True,
                "ticker": ticker,
                "rows": len(data),
                "columns": list(data.columns),
                "start_date": str(data.index[0]),
                "end_date": str(data.index[-1]),
                "cache_file": str(cache_file),
                "data": data  # Include actual dataframe
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "yfinance not installed. Run: pip install yfinance"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def download_economic_data(self,
                                    series_id: str,
                                    source: str = "fred") -> Dict[str, Any]:
        """
        Download economic data series
        
        Args:
            series_id: Series identifier (e.g., 'GDP', 'UNRATE')
            source: Data source ('fred', 'worldbank', etc.)
            
        Returns:
            Dictionary with data and metadata
        """
        try:
            if source == "fred":
                return await self._download_fred(series_id)
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
        except Exception as e:
            self.logger.error(f"Failed to download {series_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _download_fred(self, series_id: str) -> Dict[str, Any]:
        """Download data from FRED"""
        try:
            from fredapi import Fred
            import os
            
            api_key = os.getenv("FRED_API_KEY")
            if not api_key:
                return {
                    "success": False,
                    "error": "FRED_API_KEY not set. Get one from https://fred.stlouisfed.org/docs/api/api_key.html"
                }
            
            fred = Fred(api_key=api_key)
            data = fred.get_series(series_id)
            
            # Save to cache
            cache_file = self.cache_dir / f"fred_{series_id}.csv"
            data.to_csv(cache_file)
            
            return {
                "success": True,
                "series_id": series_id,
                "observations": len(data),
                "start_date": str(data.index[0]),
                "end_date": str(data.index[-1]),
                "cache_file": str(cache_file),
                "data": data
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "fredapi not installed. Run: pip install fredapi"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def download_news_sentiment(self,
                                     query: str,
                                     start_date: str,
                                     end_date: str) -> Dict[str, Any]:
        """
        Download news articles and sentiment data
        
        Args:
            query: Search query (e.g., 'US China trade')
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with articles and sentiment scores
        """
        # Placeholder for news API integration
        return {
            "success": False,
            "error": "News API integration not yet implemented"
        }


class DataAnalyzer:
    """Analyze downloaded data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def calculate_returns(self, prices: Any, method: str = "log") -> Dict[str, Any]:
        """
        Calculate returns from price data
        
        Args:
            prices: Price series (pandas Series or DataFrame)
            method: 'simple', 'log', or 'pct_change'
            
        Returns:
            Dictionary with returns data
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(prices, pd.DataFrame):
                prices = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
            
            if method == "log":
                returns = np.log(prices / prices.shift(1))
            elif method == "simple":
                returns = prices.pct_change()
            else:
                returns = prices.pct_change()
            
            return {
                "success": True,
                "returns": returns,
                "mean": float(returns.mean()),
                "std": float(returns.std()),
                "min": float(returns.min()),
                "max": float(returns.max())
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def event_study_analysis(self,
                                  returns: Any,
                                  event_dates: List[str],
                                  window: int = 10) -> Dict[str, Any]:
        """
        Perform event study analysis
        
        Args:
            returns: Return series
            event_dates: List of event dates
            window: Days before/after event to analyze
            
        Returns:
            Dictionary with event study results
        """
        try:
            import pandas as pd
            import numpy as np
            
            results = []
            for event_date in event_dates:
                event_dt = pd.to_datetime(event_date)
                
                # Get window around event
                start_dt = event_dt - pd.Timedelta(days=window)
                end_dt = event_dt + pd.Timedelta(days=window)
                
                window_returns = returns.loc[start_dt:end_dt]
                
                if len(window_returns) > 0:
                    results.append({
                        "event_date": event_date,
                        "pre_event_return": float(window_returns.loc[:event_dt].sum()),
                        "post_event_return": float(window_returns.loc[event_dt:].sum()),
                        "cumulative_return": float(window_returns.sum())
                    })
            
            return {
                "success": True,
                "events": results,
                "average_pre_event": np.mean([r["pre_event_return"] for r in results]),
                "average_post_event": np.mean([r["post_event_return"] for r in results])
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def correlation_analysis(self,
                                  series1: Any,
                                  series2: Any) -> Dict[str, Any]:
        """
        Calculate correlation between two series
        
        Args:
            series1: First data series
            series2: Second data series
            
        Returns:
            Correlation statistics
        """
        try:
            import pandas as pd
            import numpy as np
            from scipy import stats
            
            # Align series
            df = pd.DataFrame({"s1": series1, "s2": series2}).dropna()
            
            correlation = df["s1"].corr(df["s2"])
            pearson_r, pearson_p = stats.pearsonr(df["s1"], df["s2"])
            spearman_r, spearman_p = stats.spearmanr(df["s1"], df["s2"])
            
            return {
                "success": True,
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "observations": len(df)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

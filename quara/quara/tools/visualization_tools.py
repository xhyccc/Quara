"""
Visualization generation tools for QuARA
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path


class VisualizationGenerator:
    """Generate visualizations from data"""
    
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    async def plot_time_series(self,
                               data: Any,
                               title: str = "Time Series",
                               filename: str = "timeseries.png",
                               **kwargs) -> Dict[str, Any]:
        """
        Create time series plot
        
        Args:
            data: Time series data (pandas Series or DataFrame)
            title: Plot title
            filename: Output filename
            **kwargs: Additional matplotlib parameters
            
        Returns:
            Dictionary with plot info
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            plt.figure(figsize=kwargs.get('figsize', (12, 6)))
            
            if hasattr(data, 'plot'):
                data.plot(title=title)
            else:
                plt.plot(data)
                plt.title(title)
            
            plt.xlabel(kwargs.get('xlabel', 'Date'))
            plt.ylabel(kwargs.get('ylabel', 'Value'))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved plot to {output_path}")
            
            return {
                "success": True,
                "path": str(output_path),
                "filename": filename
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create plot: {e}")
            return {"success": False, "error": str(e)}
    
    async def plot_returns_distribution(self,
                                       returns: Any,
                                       title: str = "Returns Distribution",
                                       filename: str = "returns_dist.png") -> Dict[str, Any]:
        """
        Plot returns distribution with histogram and KDE
        
        Args:
            returns: Returns data
            title: Plot title
            filename: Output filename
            
        Returns:
            Dictionary with plot info
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            axes[0].hist(returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
            axes[0].set_title(f"{title} - Histogram")
            axes[0].set_xlabel("Returns")
            axes[0].set_ylabel("Frequency")
            axes[0].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(returns.dropna(), dist="norm", plot=axes[1])
            axes[1].set_title("Q-Q Plot")
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "path": str(output_path),
                "filename": filename
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def plot_event_study(self,
                              event_results: List[Dict],
                              title: str = "Event Study Results",
                              filename: str = "event_study.png") -> Dict[str, Any]:
        """
        Plot event study results
        
        Args:
            event_results: List of event study results
            title: Plot title
            filename: Output filename
            
        Returns:
            Dictionary with plot info
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import numpy as np
            matplotlib.use('Agg')
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Pre vs Post event returns
            pre_returns = [r["pre_event_return"] for r in event_results]
            post_returns = [r["post_event_return"] for r in event_results]
            
            axes[0].bar(['Pre-Event', 'Post-Event'], 
                       [np.mean(pre_returns), np.mean(post_returns)],
                       alpha=0.7, edgecolor='black')
            axes[0].set_title("Average Returns Around Events")
            axes[0].set_ylabel("Return")
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Cumulative returns by event
            axes[1].scatter(range(len(event_results)), 
                          [r["cumulative_return"] for r in event_results],
                          alpha=0.6, s=100)
            axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1].set_title("Cumulative Returns by Event")
            axes[1].set_xlabel("Event Number")
            axes[1].set_ylabel("Cumulative Return")
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "path": str(output_path),
                "filename": filename
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def plot_correlation_matrix(self,
                                     data: Any,
                                     title: str = "Correlation Matrix",
                                     filename: str = "correlation.png") -> Dict[str, Any]:
        """
        Plot correlation heatmap
        
        Args:
            data: DataFrame with multiple columns
            title: Plot title
            filename: Output filename
            
        Returns:
            Dictionary with plot info
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import seaborn as sns
            matplotlib.use('Agg')
            
            plt.figure(figsize=(10, 8))
            
            correlation = data.corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=1)
            
            plt.title(title)
            plt.tight_layout()
            
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "path": str(output_path),
                "filename": filename
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

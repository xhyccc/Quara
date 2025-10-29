"""
File operation tools for QuARA
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re


class FileSaver:
    """Save content to files with organized structure"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
    
    async def save_text(self, 
                       content: str,
                       filename: str,
                       subdirectory: str = None,
                       overwrite: bool = False) -> Dict[str, Any]:
        """
        Save text content to a file
        
        Args:
            content: Text content to save
            filename: Name of the file
            subdirectory: Optional subdirectory
            overwrite: Whether to overwrite existing file
            
        Returns:
            Dictionary with save result
        """
        try:
            # Create directory path
            if subdirectory:
                save_dir = self.base_dir / subdirectory
            else:
                save_dir = self.base_dir
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file path
            file_path = save_dir / filename
            
            # Check if file exists
            if file_path.exists() and not overwrite:
                return {
                    "success": False,
                    "error": f"File already exists: {file_path}. Use overwrite=True to replace."
                }
            
            # Save content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Saved file: {file_path}")
            
            return {
                "success": True,
                "path": str(file_path),
                "size": len(content),
                "lines": content.count('\n') + 1
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def save_json(self,
                       data: Any,
                       filename: str,
                       subdirectory: str = None,
                       pretty: bool = True) -> Dict[str, Any]:
        """
        Save data as JSON
        
        Args:
            data: Data to save (must be JSON serializable)
            filename: Name of the file
            subdirectory: Optional subdirectory
            pretty: Whether to format with indentation
            
        Returns:
            Dictionary with save result
        """
        try:
            # Create directory path
            if subdirectory:
                save_dir = self.base_dir / subdirectory
            else:
                save_dir = self.base_dir
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file path
            file_path = save_dir / filename
            
            # Save JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(data, f, indent=2, default=str)
                else:
                    json.dump(data, f, default=str)
            
            self.logger.info(f"Saved JSON: {file_path}")
            
            return {
                "success": True,
                "path": str(file_path),
                "keys": list(data.keys()) if isinstance(data, dict) else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def save_dataframe(self,
                            df: Any,
                            filename: str,
                            subdirectory: str = None,
                            format: str = "csv") -> Dict[str, Any]:
        """
        Save pandas DataFrame
        
        Args:
            df: pandas DataFrame
            filename: Name of the file
            subdirectory: Optional subdirectory
            format: File format ('csv', 'excel', 'json')
            
        Returns:
            Dictionary with save result
        """
        try:
            import pandas as pd
            
            if not isinstance(df, pd.DataFrame):
                return {
                    "success": False,
                    "error": "Data is not a pandas DataFrame"
                }
            
            # Create directory path
            if subdirectory:
                save_dir = self.base_dir / subdirectory
            else:
                save_dir = self.base_dir
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file path
            file_path = save_dir / filename
            
            # Save based on format
            if format == "csv":
                df.to_csv(file_path, index=True)
            elif format == "excel":
                df.to_excel(file_path, index=True)
            elif format == "json":
                df.to_json(file_path, orient='records', indent=2)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported format: {format}"
                }
            
            self.logger.info(f"Saved DataFrame: {file_path}")
            
            return {
                "success": True,
                "path": str(file_path),
                "rows": len(df),
                "columns": list(df.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save DataFrame: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def append_to_file(self,
                            content: str,
                            filename: str,
                            subdirectory: str = None) -> Dict[str, Any]:
        """
        Append content to an existing file
        
        Args:
            content: Content to append
            filename: Name of the file
            subdirectory: Optional subdirectory
            
        Returns:
            Dictionary with result
        """
        try:
            # Create directory path
            if subdirectory:
                save_dir = self.base_dir / subdirectory
            else:
                save_dir = self.base_dir
            
            file_path = save_dir / filename
            
            # Append content
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Appended to file: {file_path}")
            
            return {
                "success": True,
                "path": str(file_path)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to append to file: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class FileSearcher:
    """Search for content within files"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
    
    async def find_in_file(self,
                          filename: str,
                          pattern: str,
                          regex: bool = False,
                          max_results: int = 100) -> Dict[str, Any]:
        """
        Find pattern in a specific file
        
        Args:
            filename: File to search in
            pattern: Pattern to search for
            regex: Whether pattern is a regex
            max_results: Maximum number of matches to return
            
        Returns:
            Dictionary with search results
        """
        try:
            file_path = self.base_dir / filename
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            matches = []
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if regex:
                import re
                pattern_obj = re.compile(pattern)
                
                for i, line in enumerate(lines, 1):
                    if pattern_obj.search(line):
                        matches.append({
                            "line_number": i,
                            "line": line.strip(),
                            "column": pattern_obj.search(line).start()
                        })
                        
                        if len(matches) >= max_results:
                            break
            else:
                for i, line in enumerate(lines, 1):
                    if pattern in line:
                        matches.append({
                            "line_number": i,
                            "line": line.strip(),
                            "column": line.index(pattern)
                        })
                        
                        if len(matches) >= max_results:
                            break
            
            self.logger.info(f"Found {len(matches)} matches in {filename}")
            
            return {
                "success": True,
                "file": str(file_path),
                "pattern": pattern,
                "matches": matches,
                "count": len(matches)
            }
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def find_in_directory(self,
                               pattern: str,
                               directory: str = ".",
                               file_pattern: str = "*",
                               recursive: bool = True,
                               regex: bool = False) -> Dict[str, Any]:
        """
        Find pattern across multiple files in a directory
        
        Args:
            pattern: Pattern to search for
            directory: Directory to search in
            file_pattern: File pattern to match (e.g., "*.py")
            recursive: Whether to search recursively
            regex: Whether pattern is a regex
            
        Returns:
            Dictionary with search results
        """
        try:
            search_dir = self.base_dir / directory
            
            if not search_dir.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {search_dir}"
                }
            
            results = []
            
            if recursive:
                files = search_dir.rglob(file_pattern)
            else:
                files = search_dir.glob(file_pattern)
            
            for file_path in files:
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.base_dir)
                    file_result = await self.find_in_file(
                        str(rel_path),
                        pattern,
                        regex=regex,
                        max_results=10
                    )
                    
                    if file_result["success"] and file_result["count"] > 0:
                        results.append({
                            "file": str(rel_path),
                            "matches": file_result["matches"]
                        })
            
            total_matches = sum(len(r["matches"]) for r in results)
            
            self.logger.info(f"Found {total_matches} matches across {len(results)} files")
            
            return {
                "success": True,
                "pattern": pattern,
                "directory": str(search_dir),
                "files_matched": len(results),
                "total_matches": total_matches,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Directory search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def grep_files(self,
                        pattern: str,
                        file_paths: List[str],
                        context_lines: int = 0) -> Dict[str, Any]:
        """
        Grep-style search across specific files
        
        Args:
            pattern: Pattern to search for
            file_paths: List of file paths to search
            context_lines: Number of context lines before/after match
            
        Returns:
            Dictionary with results
        """
        results = []
        
        for file_path in file_paths:
            file_result = await self.find_in_file(
                file_path,
                pattern,
                regex=True
            )
            
            if file_result["success"] and file_result["count"] > 0:
                results.append(file_result)
        
        return {
            "success": True,
            "pattern": pattern,
            "files_searched": len(file_paths),
            "files_matched": len(results),
            "results": results
        }

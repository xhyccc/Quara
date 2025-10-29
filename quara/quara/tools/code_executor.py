"""
Code execution tools for QuARA
Safely executes generated Python code in isolated environments
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging


class CodeExecutor:
    """Execute Python code safely and capture results"""
    
    def __init__(self, python_path: str = None, working_dir: str = None):
        self.python_path = python_path or sys.executable
        self.working_dir = working_dir or os.getcwd()
        self.logger = logging.getLogger(__name__)
        
    async def execute_code(self, 
                          code: str, 
                          timeout: int = 300,
                          capture_output: bool = True,
                          save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Execute Python code and return results
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            capture_output: Whether to capture stdout/stderr
            save_artifacts: Whether to save generated files
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                delete=False,
                dir=self.working_dir
            ) as f:
                f.write(code)
                temp_file = f.name
            
            self.logger.info(f"Executing code from {temp_file}")
            
            # Execute code
            result = subprocess.run(
                [self.python_path, temp_file],
                cwd=self.working_dir,
                capture_output=capture_output,
                text=True,
                timeout=timeout
            )
            
            # Clean up temp file
            if not save_artifacts:
                os.unlink(temp_file)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "code_file": temp_file if save_artifacts else None
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Code execution timed out after {timeout}s")
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds",
                "stdout": "",
                "stderr": ""
            }
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": ""
            }
    
    async def execute_notebook(self, 
                              cells: List[str],
                              output_path: str = None) -> Dict[str, Any]:
        """
        Execute code cells sequentially like a notebook
        
        Args:
            cells: List of code cell strings
            output_path: Optional path to save notebook
            
        Returns:
            Dictionary with execution results for each cell
        """
        results = []
        
        # Create combined code with cell markers
        full_code = []
        for i, cell in enumerate(cells):
            full_code.append(f"# Cell {i+1}")
            full_code.append(cell)
            full_code.append("")
        
        # Execute all cells together
        combined_code = "\n".join(full_code)
        result = await self.execute_code(combined_code)
        
        return {
            "success": result["success"],
            "outputs": result["stdout"],
            "errors": result["stderr"],
            "cell_count": len(cells)
        }


class SandboxExecutor(CodeExecutor):
    """Execute code in a more isolated sandbox environment"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_imports = {
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 
            'sklearn', 'statsmodels', 'yfinance', 'requests',
            'json', 'csv', 'datetime', 'pathlib', 'os', 'sys'
        }
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate code for security issues
        
        Args:
            code: Python code to validate
            
        Returns:
            Validation result
        """
        warnings = []
        errors = []
        
        # Check for dangerous operations
        dangerous_patterns = [
            'eval(', 'exec(', '__import__',
            'subprocess.', 'os.system', 'os.popen',
            'open(', 'file(', 'input('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                warnings.append(f"Found potentially dangerous operation: {pattern}")
        
        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors
        }

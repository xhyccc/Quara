"""
Utils module for QuARA system utilities and helpers
"""

import logging
from typing import Dict, Any, List
from datetime import datetime


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """Setup logging configuration for QuARA system"""
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def validate_research_request(request: str) -> Dict[str, Any]:
    """Validate and analyze a research request"""
    
    if not request or not request.strip():
        return {
            "valid": False,
            "error": "Research request cannot be empty"
        }
    
    if len(request.strip()) < 10:
        return {
            "valid": False,
            "error": "Research request too short (minimum 10 characters)"
        }
    
    # Basic analysis
    words = request.lower().split()
    research_indicators = [
        "analyze", "study", "investigate", "examine", "research",
        "effect", "impact", "relationship", "correlation", "cause"
    ]
    
    has_research_indicators = any(word in words for word in research_indicators)
    
    return {
        "valid": True,
        "word_count": len(words),
        "has_research_indicators": has_research_indicators,
        "complexity": "high" if len(words) > 15 else "medium" if len(words) > 8 else "low"
    }


def format_agent_response(agent_role: str, response: Dict[str, Any]) -> str:
    """Format agent response for display"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    formatted = f"[{timestamp}] {agent_role.upper()}: "
    
    if "error" in response:
        formatted += f"ERROR - {response['error']}"
    elif "result" in response:
        result = response["result"]
        if isinstance(result, dict):
            formatted += f"Task completed with {len(result)} components"
        else:
            formatted += f"Result: {str(result)[:100]}..."
    else:
        formatted += "Task in progress"
    
    return formatted


def extract_key_terms(text: str) -> List[str]:
    """Extract key terms from text for indexing and search"""
    
    # Simple keyword extraction
    import re
    
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should"
    }
    
    # Extract words (alphanumeric, 3+ characters)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter stop words and get unique terms
    key_terms = list(set(word for word in words if word not in stop_words))
    
    # Sort by length (longer terms first)
    key_terms.sort(key=len, reverse=True)
    
    return key_terms[:20]  # Return top 20 terms


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity between two texts"""
    
    terms1 = set(extract_key_terms(text1))
    terms2 = set(extract_key_terms(text2))
    
    if not terms1 or not terms2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(terms1.intersection(terms2))
    union = len(terms1.union(terms2))
    
    return intersection / union if union > 0 else 0.0


def generate_project_id(research_request: str) -> str:
    """Generate a project ID from research request"""
    
    import hashlib
    import re
    
    # Extract key words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', research_request.lower())
    key_words = words[:3]  # First 3 significant words
    
    # Create hash
    request_hash = hashlib.md5(research_request.encode()).hexdigest()[:8]
    
    # Combine
    if key_words:
        project_id = f"{'_'.join(key_words)}_{request_hash}"
    else:
        project_id = f"research_{request_hash}"
    
    return project_id


def create_citation(title: str, authors: List[str], year: int, 
                   journal: str = None, doi: str = None) -> str:
    """Create APA-style citation"""
    
    if not authors:
        author_str = "Unknown Author"
    elif len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} & {authors[1]}"
    else:
        author_str = f"{authors[0]} et al."
    
    citation = f"{author_str} ({year}). {title}."
    
    if journal:
        citation += f" {journal}."
    
    if doi:
        citation += f" https://doi.org/{doi}"
    
    return citation


def format_statistical_result(result: Dict[str, Any]) -> str:
    """Format statistical result for readable output"""
    
    if "coefficients" in result and "p_values" in result:
        # Regression-style result
        output = "Statistical Results:\n"
        
        coefficients = result["coefficients"]
        p_values = result["p_values"]
        
        for var in coefficients:
            coef = coefficients[var]
            p_val = p_values.get(var, "N/A")
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            output += f"  {var}: β = {coef:.3f}, p = {p_val:.3f} {significance}\n"
        
        if "r_squared" in result:
            output += f"  R² = {result['r_squared']:.3f}\n"
        
        return output
    
    return str(result)


def estimate_research_time(request: str, complexity: str = None) -> Dict[str, Any]:
    """Estimate time required for research workflow"""
    
    if not complexity:
        validation = validate_research_request(request)
        complexity = validation.get("complexity", "medium")
    
    # Time estimates in minutes (for demonstration)
    time_estimates = {
        "low": {
            "phase_0_design": 5,
            "phase_1_theorist": 10, 
            "phase_2_librarian": 15,
            "phase_3_methodologist": 10,
            "phase_4_analyst": 20,
            "phase_5_scribe": 15
        },
        "medium": {
            "phase_0_design": 10,
            "phase_1_theorist": 20,
            "phase_2_librarian": 30,
            "phase_3_methodologist": 25,
            "phase_4_analyst": 45,
            "phase_5_scribe": 30
        },
        "high": {
            "phase_0_design": 20,
            "phase_1_theorist": 40,
            "phase_2_librarian": 60,
            "phase_3_methodologist": 50,
            "phase_4_analyst": 90,
            "phase_5_scribe": 60
        }
    }
    
    estimates = time_estimates.get(complexity, time_estimates["medium"])
    total_time = sum(estimates.values())
    
    return {
        "complexity": complexity,
        "phase_estimates": estimates,
        "total_minutes": total_time,
        "estimated_hours": total_time / 60,
        "estimated_description": f"Approximately {total_time // 60} hours and {total_time % 60} minutes"
    }


class ProgressTracker:
    """Track progress of research workflow"""
    
    def __init__(self):
        self.phases = [
            "phase_0_design",
            "phase_1_theorist", 
            "phase_2_librarian",
            "phase_3_methodologist",
            "phase_4_analyst",
            "phase_5_scribe"
        ]
        self.current_phase = 0
        self.phase_completion = {phase: False for phase in self.phases}
        self.start_time = datetime.now()
        self.phase_times = {}
    
    def start_phase(self, phase: str):
        """Mark start of a phase"""
        if phase in self.phases:
            self.phase_times[phase] = {"start": datetime.now()}
    
    def complete_phase(self, phase: str):
        """Mark completion of a phase"""
        if phase in self.phases:
            self.phase_completion[phase] = True
            if phase in self.phase_times:
                self.phase_times[phase]["end"] = datetime.now()
                self.phase_times[phase]["duration"] = (
                    self.phase_times[phase]["end"] - self.phase_times[phase]["start"]
                ).total_seconds()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress status"""
        completed_phases = sum(1 for completed in self.phase_completion.values() if completed)
        total_phases = len(self.phases)
        progress_percent = (completed_phases / total_phases) * 100
        
        current_phase = None
        for i, phase in enumerate(self.phases):
            if not self.phase_completion[phase]:
                current_phase = phase
                break
        
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "progress_percent": progress_percent,
            "completed_phases": completed_phases,
            "total_phases": total_phases,
            "current_phase": current_phase,
            "elapsed_seconds": elapsed_time,
            "phase_status": self.phase_completion,
            "phase_times": self.phase_times
        }


__all__ = [
    "setup_logging",
    "validate_research_request",
    "format_agent_response",
    "extract_key_terms",
    "calculate_similarity",
    "generate_project_id",
    "create_citation",
    "format_statistical_result",
    "estimate_research_time",
    "ProgressTracker"
]
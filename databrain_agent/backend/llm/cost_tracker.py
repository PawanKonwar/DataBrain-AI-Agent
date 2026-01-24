"""Track LLM API costs."""
from typing import Dict
from datetime import datetime
import json
from pathlib import Path


class CostTracker:
    """Track API costs for different LLM providers."""
    
    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "openai-gpt-4": {"input": 30.0, "output": 60.0},
        "openai-gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "deepseek-chat": {"input": 0.14, "output": 0.28},
    }
    
    def __init__(self, log_file: str = "./cost_log.json"):
        """Initialize cost tracker."""
        self.log_file = Path(log_file)
        self.costs = self._load_costs()
    
    def _load_costs(self) -> Dict:
        """Load cost history from file."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {"total_cost": 0.0, "by_provider": {}, "history": []}
    
    def _save_costs(self):
        """Save cost history to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.costs, f, indent=2)
    
    def track_usage(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        """Track token usage and calculate cost."""
        model_key = f"{provider}-{model}"
        pricing = self.PRICING.get(model_key, {"input": 0.0, "output": 0.0})
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        # Update totals
        self.costs["total_cost"] += total_cost
        
        if provider not in self.costs["by_provider"]:
            self.costs["by_provider"][provider] = 0.0
        self.costs["by_provider"][provider] += total_cost
        
        # Add to history
        self.costs["history"].append({
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost
        })
        
        self._save_costs()
        return total_cost
    
    def get_total_cost(self) -> float:
        """Get total cost so far."""
        return self.costs["total_cost"]
    
    def get_cost_by_provider(self) -> Dict[str, float]:
        """Get costs broken down by provider."""
        return self.costs["by_provider"]

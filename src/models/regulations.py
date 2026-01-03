"""
Regulation Impact Modifiers for 2026 Season.

This module applies penalties/boosts to team baselines based on 
expected impact of 2026 regulation changes (Active Aero, Engine).
"""

from typing import Dict
from .bayesian import DriverPrior

def apply_2026_regulations(priors: Dict[str, DriverPrior]) -> Dict[str, DriverPrior]:
    """
    Adjust priors based on 2026 Regulation Scenarios.
    
    Hypothesis:
    - New Engine Regs benefit factory teams (Mercedes, Ferrari, RBPT).
    - Customer teams might suffer integration issues early.
    - Audi (Sauber) transition year penalty.
    """
    
    # Configuration: Impact Strength (Rating Points)
    ENGINE_BOOST = 1.5
    NEW_TEAM_PENALTY = -2.0
    CUSTOMER_PENALTY = -0.5
    
    adjusted_priors = {}
    
    for d_num, p in priors.items():
        # Create a copy to modify
        new_mu = p.mu
        new_sigma = p.sigma + 1.0  # Increase uncertainty for everyone in 2026
        
        # Scenario Logic
        if p.team in ['Mercedes', 'Ferrari']:
            new_mu += ENGINE_BOOST  # Factory team advantage
        elif p.team == 'Kick Sauber':
            new_mu += NEW_TEAM_PENALTY # Audi transition risk
        elif 'Customer' in p.team_tier: # Hypothetical tag
            new_mu += CUSTOMER_PENALTY
            
        # Create new Prior object
        adjusted_priors[d_num] = DriverPrior(
            driver_number=p.driver_number,
            driver_code=p.driver_code,
            team=p.team,
            team_tier=p.team_tier,
            mu=new_mu,
            sigma=new_sigma  # 2026 is more uncertain than 2025
        )
        
    return adjusted_priors
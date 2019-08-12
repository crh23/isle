oneriskmodel = False
replicating = False
force_foreground = False
verbose = False
showprogress = False
# Should network be visualized? This should be False by default, to be overridden by commandline arguments
show_network = False
save_network = False
# Should logs be small in ensemble runs (only aggregated level data)?
slim_log = True
buy_bankruptcies = False
enforce_regulations = False
aid_relief = False

simulation_parameters = {
    "no_categories": 4,
    "no_insurancefirms": 20,
    "no_reinsurancefirms": 4,
    "no_riskmodels": 3,
    # values >=1; inaccuracy higher with higher values
    "riskmodel_inaccuracy_parameter": 2,
    # values >=1; factor of additional liquidity beyond value at risk
    "riskmodel_margin_of_safety": 2,
    "margin_increase": 0,
    # "margin_increase" modifies the margin of safety depending on the number of risks models available in the market.
    # When it is 0 all risk models have the same margin of safety.
    "value_at_risk_tail_probability": 0.02,
    # values <1, >0, usually close to 0; tail probability at which the value at risk is taken by the risk models
    "norm_profit_markup": 0.15,
    "rein_norm_profit_markup": 0.15,
    "dividend_share_of_profits": 0.4,
    "mean_contract_runtime": 12,
    "contract_runtime_halfspread": 2,
    "default_contract_payment_period": 3,
    "max_time": 1000,
    "money_supply": 2000000000,
    "event_time_mean_separation": 100 / 3.0,
    "expire_immediately": False,
    "risk_factors_present": False,
    "risk_factor_lower_bound": 0.4,
    "risk_factor_upper_bound": 0.6,
    "initial_acceptance_threshold": 0.5,
    "acceptance_threshold_friction": 0.9,
    "insurance_firm_market_entry_probability": 0.3,  # 0.02,
    "reinsurance_firm_market_entry_probability": 0.05,  # 0.004,
    # Determines the reinsurance type of the simulation. Should be "non-proportional" or "excess-of-loss"
    "simulation_reinsurance_type": "non-proportional",
    "default_non-proportional_reinsurance_deductible": 0.3,
    "default_non-proportional_reinsurance_excess": 1.0,
    "default_non-proportional_reinsurance_premium_share": 0.3,
    "static_non-proportional_reinsurance_levels": False,
    "catbonds_off": False,
    "reinsurance_off": False,
    "capacity_target_decrement_threshold": 1.8,
    "capacity_target_increment_threshold": 1.2,
    "capacity_target_decrement_factor": 24 / 25.0,
    "capacity_target_increment_factor": 25 / 24.0,
    # Retention parameters
    "insurance_retention": 0.85,  # Ratio of insurance contracts retained every iteration.
    "reinsurance_retention": 1,  # Ratio of reinsurance contracts retained every iteration.
    # Premium sensitivity parameters
    "premium_sensitivity": 5,
    # This parameter represents how sensitive is the variation of the insurance premium with respect of the capital
    # of the market. Higher means more sensitive.
    "reinpremium_sensitivity": 6,
    # This parameter represents how sensitive is the variation of the reinsurance premium with respect of the capital
    # of the market. Higher means more sensitive.
    # Balanced portfolio parameters
    "insurers_balance_ratio": 0.1,
    # This ratio represents how low we want to keep the standard deviation of the cash reserved below the mean for
    # insurers. Lower means more balanced.
    "reinsurers_balance_ratio": 20,
    # This ratio represents how low we want to keep the standard deviation of the cash reserved below the mean for
    # reinsurers. Lower means more balanced. (Deactivated for the moment)
    "insurers_recursion_limit": 50,
    # Intensity of the recursion algorithm to balance the portfolio of risks for insurers.
    "reinsurers_recursion_limit": 10,
    # Intensity of the recursion algorithm to balance the portfolio of risks for reinsurers.
    # Market permanency parameters
    "market_permanency_off": False,
    # This parameter activates (deactivates) the following market permanency constraints.
    "cash_permanency_limit": 100,
    # This parameter enforces the limit under which the firms leave the market because they cannot underwrite anything.
    "insurance_permanency_contracts_limit": 4,
    # If insurers stay for too long under this limit of contracts they deccide to leave the market.
    "insurance_permanency_ratio_limit": 0.6,
    # If insurers stay for too long under this limit they deccide to leave the market because they have too much capital.
    "insurance_permanency_time_constraint": 24,
    # The period that the insurers wait before leaving the market if they have few capital or few contract .
    "reinsurance_permanency_contracts_limit": 2,
    # If reinsurers stay for too long under this limit of contracts they deccide to leave the market.
    "reinsurance_permanency_ratio_limit": 0.8,
    # If reinsurers stay for too long under this limit they decide to leave the market because they have too much capital.
    "reinsurance_permanency_time_constraint": 48,
    # This parameter defines the period that the reinsurers wait if they have few capital or few contract before leaving the market.
    # Insurance and Reinsurance deductibles
    "insurance_reinsurance_levels_lower_bound": 0.25,
    "insurance_reinsurance_levels_upper_bound": 0.30,
    "reinsurance_reinsurance_levels_lower_bound": 0.5,
    "reinsurance_reinsurance_levels_upper_bound": 0.95,
    "initial_agent_cash": 80000,
    "initial_reinagent_cash": 2000000,
    "interest_rate": 0.001,
    "reinsurance_limit": 0.1,
    "upper_price_limit": 1.2,
    "lower_price_limit": 0.85,
    "no_risks": 20000,
    "value_per_risk": 1000,
    # Determines the maximum upscaling of premiums based on insurer size - set to 1 to disable scaled premiums.
    # High values will give bigger insurers more money
    # Values between 0 and 1 will make premiums decrease for bigger insurers.
    "max_scale_premiums": 1.2,
    # Determines the minimum fraction of inaccuracy that insurers can achieve - a value of 0 means the biggest insurers
    # can be perfectly accurate, a value of 1 disables changes in inaccuracy based on size
    "scale_inaccuracy": 0.3,
    # The smallest number of tranches that an insurer will issue when asking for reinsurance. Note: even if this is 1,
    # insurers may still end up with layered reinsurance to fill gaps
    "min_tranches": 1,
    "aid_budget": 1000000,
}

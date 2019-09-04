# These should usually be left False by default - they are modified by command line parameters (or similar)
replicating = False
force_foreground = False  # TODO: remove this?
verbose = False
showprogress = False
show_network = False
save_network = False
slim_log = False

# fmt: off
simulation_parameters = {
    "max_time": 2000,
    "no_categories": 4,

    # no_[re]insurancefirms are initial conditions only
    "no_insurancefirms": 20,
    "no_reinsurancefirms": 4,

    # Numer of risk models in the market
    "no_riskmodels": 1,

    # values >=1; inaccuracy higher with higher values
    "riskmodel_inaccuracy_parameter": 2,
    # values >=1; factor of additional liquidity beyond value at risk
    "riskmodel_margin_of_safety": 1.5,

    # "margin_increase" modifies the margin of safety depending on the number of risks models available in the market.
    # When it is 0 all risk models have the same margin of safety.
    "margin_increase": 0,

    # values <1, >0, usually close to 0; tail probability at which the value at risk is taken by the risk models
    "value_at_risk_tail_probability": 0.02,

    # The markup fims aim to make
    "norm_profit_markup": 0.15,
    "rein_norm_profit_markup": 0.15,  # TODO: UNUSED

    # The share of profits that is given back to investors (the simulation)
    "dividend_share_of_profits": 0.4,

    # Insurance contracts have a random runtime,
    "mean_contract_runtime": 12,
    "contract_runtime_halfspread": 2,

    # Reinsurance contracts have a fixed runtime
    "reinsurance_contract_runtime": 12,
    "default_contract_payment_period": 3,

    # The ammount of money in the economy. Mainly used to check that we aren't losing any anywhere
    "money_supply": 2000000000,

    # The mean time between catastrophe events
    "event_time_mean_separation": 100 / 3,

    # Whether contacts expire after the first triggering (only tested on False)
    "expire_immediately": False,

    # Risk factor provide a per-risk heterogeneity - the risk factor is the probability that a risk is affected when
    # a catastrophe occurs in its category. Not properly implemented (insurers can't take into account). If False, all
    # risk factors are 1.
    "risk_factors_present": False,
    "risk_factor_lower_bound": 0.4,
    "risk_factor_upper_bound": 0.6,

    # TODO: Acceptance threshold appears UNUSED
    "initial_acceptance_threshold": 0.5,
    "acceptance_threshold_friction": 0.9,

    # Each timestep a new [re]insurer enters the market with a given probability
    "insurance_firm_market_entry_probability": 0.3,  # 0.02,
    "reinsurance_firm_market_entry_probability": 0.05,  # 0.004,

    # Determines the reinsurance type of the simulation. Should be "non-proportional" or "proportional". Only
    # the former actually works
    "simulation_reinsurance_type": "non-proportional",

    # If True, will use the static deductible (if reinsurance type is non-proportional), otherwise the dynamic
    # deductible settings below are used
    "static_non-proportional_reinsurance_levels": False,
    "default_non-proportional_reinsurance_deductible": 0.3,

    # The upper bound of reinsurance that firms can get. Should usually be 1
    "default_non-proportional_reinsurance_limit": 1.0,

    # The share of the premiums that reinsurers take
    "default_non-proportional_reinsurance_premium_share": 0.3,

    # Insurance and Reinsurance deductible ranges - each firm gets a random value from these ranges
    "insurance_reinsurance_levels_lower_bound": 0.25,
    "insurance_reinsurance_levels_upper_bound": 0.30,
    "reinsurance_reinsurance_levels_lower_bound": 0.5,
    "reinsurance_reinsurance_levels_upper_bound": 0.95,

    # Turn catbonds or reinsurance off
    "catbonds_off": False,
    "reinsurance_off": False,

    # Firms adjust their capacity target based on the ratio between their risk held and the capacity they could gain
    # from reinsurance
    "capacity_target_decrement_threshold": 1.8,  # TODO: What?
    "capacity_target_increment_threshold": 1.2,
    "capacity_target_decrement_factor": 24 / 25,
    "capacity_target_increment_factor": 25 / 24,

    # When a contract expires it has a chance to be immediatly offered again to the insurer - set that probability
    "insurance_retention": 0.85,
    "reinsurance_retention": 1,

    # The market premiums are based on the ammount of cash in the market - how sensitive is it? Higher is more sensitive
    "premium_sensitivity": 5,
    "reinpremium_sensitivity": 6,

    # This ratio represents how low we want to keep the standard deviation of the cash reserved below the mean for
    # [re]insurers. Lower means more balanced.
    "insurers_balance_ratio": 0.1,
    "reinsurers_balance_ratio": 20,


    # Intensity of the recursion algorithm to balance the portfolio of risks
    "insurers_recursion_limit": 50,
    "reinsurers_recursion_limit": 10,

    # Market permanency parameters
    # This parameter activates (deactivates) the following market permanency constraints.
    "market_permanency_off": False,
    # If a firm has cash less that this, they leave the market
    "cash_permanency_limit": 100,
    # If insurers have fewer than this many contracts for the below time period then they leave the market
    "insurance_permanency_contracts_limit": 4,
    # Likewise, but regarding the ratio of actual cash to cash being reserved to cover risk
    "insurance_permanency_ratio_limit": 0.6,
    # The period that the insurers wait before leaving the market in the above situations
    "insurance_permanency_time_constraint": 24,

    # Likewise for reinsurers
    "reinsurance_permanency_contracts_limit": 2,
    "reinsurance_permanency_ratio_limit": 0.8,
    "reinsurance_permanency_time_constraint": 48,

    # The cash a [re]insurer has when it is created
    "initial_agent_cash": 80000,
    "initial_reinagent_cash": 2000000,

    # The per time period bank interest rate (note: time period ~ 1 month)
    "interest_rate": 0.001,

    # TODO: UNUSED
    "reinsurance_limit": 0.1,

    # The limits on the adjustment for market price of insurance
    "upper_price_limit": 1.2,
    "lower_price_limit": 0.85,

    # The total number of (insurance) risks in the market
    "no_risks": 20000,

    # The value of each insurance risk
    "value_per_risk": 1000,

    # The maximum upscaling of premiums based on insurer size - set to 1 to disable scaled premiums.
    # High values will give bigger insurers more money (we can assume they are more "trusted")
    # Values between 0 and 1 will make premiums decrease for bigger insurers.
    "max_scale_premiums": 1,

    # Determines the minimum fraction of inaccuracy that insurers can achieve - a value of 0 means the biggest insurers
    # can be perfectly accurate, a value of 1 disables changes in inaccuracy based on size (bigger insurers have more
    # data/resources to model catastrophes)
    "scale_inaccuracy": 1,

    # The number of tranches that an insurer will get for reinsurance
    "min_tranches": 2,

    # If this is true then reinsurance premiums can be adjusted after cat events
    "adjustable_reinsurance_premiums": True,

    # How many cat events are required to adjust the premium in the above
    "reinsurance_premium_adjustment_frequency": 2,

    # The amount (as a fraction of the existing premium) to increase premimus by in the above
    "reinsurance_premium_adjustment_amount": 0.1,

    # Whether firms can choose to buy out other bankrupt firms
    "buy_bankruptcies": False,

    # Enable or disable the regulator
    "enforce_regulations": False,

    # Enable or disable regulator bailouts and set the budget
    "aid_relief": False,
    "aid_budget": 1000000,
}
# fmt: on

"""insurance_governance.mrm: model risk management framework for insurance pricing models.

A governance wrapper for the Burning Cost ecosystem. This library does not run
statistical tests — it connects the outputs of insurance_governance.validation and
insurance-monitoring to a persistent model inventory, a defensible risk tier
classification engine, and executive committee reports that make PRA model risk
conversations survivable.

Typical workflow::

    from insurance_governance.mrm import (
        ModelCard,
        Assumption,
        Limitation,
        ModelInventory,
        RiskTierScorer,
        GovernanceReport,
    )

    # 1. Build a model card
    card = ModelCard(
        model_id='motor-freq-v2',
        model_name='Motor TPPD Frequency',
        version='2.1.0',
        model_class='pricing',
        intended_use='Frequency pricing for private motor. Not commercial.',
        assumptions=[
            Assumption(
                description='Claim frequency stationarity since 2022',
                risk='MEDIUM',
                mitigation='Quarterly A/E monitoring',
            ),
        ],
    )

    # 2. Score the risk tier
    scorer = RiskTierScorer()
    tier = scorer.score(
        gwp_impacted=125_000_000,
        model_complexity='high',
        deployment_status='champion',
        regulatory_use=False,
        external_data=False,
        customer_facing=True,
    )

    # 3. Register in the inventory
    inventory = ModelInventory('mrm_registry.json')
    inventory.register(card, tier)

    # 4. Generate a governance report
    report = GovernanceReport(card=card, tier=tier)
    report.save_html('motor_freq_mrm_pack.html')
"""

from .model_card import Assumption, Limitation, ModelCard
from .scorer import DimensionScore, RiskTierScorer, TierResult
from .inventory import ModelInventory
from .report import GovernanceReport

__all__ = [
    "Assumption",
    "DimensionScore",
    "GovernanceReport",
    "Limitation",
    "ModelCard",
    "ModelInventory",
    "RiskTierScorer",
    "TierResult",
]

"""Planner / rule-parser selection.

Local (deterministic, offline) is the default. Set FLOORSIGHT_LLM=bedrock to
let Claude on Amazon Bedrock choose the agent's next tool and parse
plain-language rules; the local policy still guards every decision. The
anthropic SDK is imported only in bedrock mode.
"""

from __future__ import annotations

import os

from floorsight.contracts import SitePlan


def llm_mode() -> str:
    return "bedrock" if os.environ.get("FLOORSIGHT_LLM", "").strip().lower() == "bedrock" else "local"


def select_planner():
    if llm_mode() == "bedrock":
        from floorsight.llm.bedrock import BedrockPlanner

        return BedrockPlanner()
    from floorsight.agent.planner import LocalPlanner

    return LocalPlanner()


def select_rule_parser(site: SitePlan):
    if llm_mode() == "bedrock":
        from floorsight.llm.bedrock import BedrockRuleParser

        return BedrockRuleParser(site)
    from floorsight.agent.language import LocalRuleParser

    return LocalRuleParser(site)

"""AgentMarket AI layer: master agent, worker adapters, registry, firewall, judge, audit."""
from __future__ import annotations

from .adapter import BaseWorker
from .master import MasterAgent
from .registry import AgentRegistry
from .workers import BAD_ACTORS, HONEST


def build_marketplace(include_bad_actors: bool = True) -> tuple[AgentRegistry, dict[str, BaseWorker], MasterAgent]:
    """One call to get a demo marketplace with all workers registered."""
    registry = AgentRegistry()
    workers: dict[str, BaseWorker] = {}
    for cls in HONEST + (BAD_ACTORS if include_bad_actors else []):
        w = cls()
        workers[w.card.agent_id] = w
        registry.register(w.card)
    return registry, workers, MasterAgent(registry, workers)

"""Agente Data Editor - Verifica coerenza dati vs conclusioni."""

from crewai import Agent

CRITIC_CONFIG = {
    "role": "Data Editor",
    "goal": (
        "Verificare che ogni affermazione nel report sia supportata dai dati "
        "calcolati. Bloccare il report se trova conclusioni non supportate."
    ),
    "backstory": (
        "Sei un editor di dati con anni di esperienza nel fact-checking "
        "sportivo. Il tuo compito e' garantire che il report finale sia "
        "accurato e che ogni conclusione sia basata sui numeri effettivamente "
        "calcolati. Se trovi discrepanze o affermazioni non supportate, "
        "richiedi correzioni prima della pubblicazione."
    ),
    "verbose": True,
    "allow_delegation": False,
}


def create_critic_agent(llm=None) -> Agent:
    """Crea e restituisce l'agente Critic.

    Parameters
    ----------
    llm : crewai.LLM, optional
        Istanza LLM per l'agente.

    Returns
    -------
    Agent
    """
    kwargs = {**CRITIC_CONFIG}
    if llm:
        kwargs["llm"] = llm
    return Agent(**kwargs)

"""Agente Football Data Analyst - Calcola il TacticalPulse Index."""

from crewai import Agent

ANALYST_CONFIG = {
    "role": "Football Data Analyst",
    "goal": (
        "Calcolare il TacticalPulse Score per ogni squadra/partita "
        "usando le tre componenti: Pressure, Discipline e Network."
    ),
    "backstory": (
        "Sei un analista di dati calcistici con esperienza in Python, pandas e "
        "network analysis. Hai accesso ai moduli core/ che calcolano le tre "
        "componenti dell'indice. Il tuo compito e' eseguire la pipeline di calcolo "
        "e produrre un DataFrame con i punteggi per ogni squadra."
    ),
    "verbose": True,
    "allow_delegation": False,
}


def create_analyst_agent(llm=None) -> Agent:
    """Crea e restituisce l'agente Analyst.

    Parameters
    ----------
    llm : crewai.LLM, optional
        Istanza LLM per l'agente.

    Returns
    -------
    Agent
    """
    kwargs = {**ANALYST_CONFIG}
    if llm:
        kwargs["llm"] = llm
    return Agent(**kwargs)

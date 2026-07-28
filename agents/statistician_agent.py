"""Agente Sports Statistician - Valida significativita statistica."""

from crewai import Agent

STATISTICIAN_CONFIG = {
    "role": "Sports Statistician",
    "goal": (
        "Validare statisticamente i risultati del TacticalPulse Index "
        "usando bootstrap, t-test e correzioni per confronti multipli."
    ),
    "backstory": (
        "Sei uno statistico sportivo specializzato in analisi bayesiana e "
        "frequentista. Usi il modulo stats/significance.py per calcolare "
        "intervalli di confidenza bootstrap, verificare la significativita "
        "delle differenze tra squadre, e applicare correzioni per confronti "
        "multipli. Il tuo compito e' determinare se i pattern osservati sono "
        "statisticamente significativi o solo rumore."
    ),
    "verbose": True,
    "allow_delegation": False,
}


def create_statistician_agent(llm=None) -> Agent:
    """Crea e restituisce l'agente Statistician.

    Parameters
    ----------
    llm : crewai.LLM, optional
        Istanza LLM per l'agente.

    Returns
    -------
    Agent
    """
    kwargs = {**STATISTICIAN_CONFIG}
    if llm:
        kwargs["llm"] = llm
    return Agent(**kwargs)

"""Agente Sports Journalist - Genera report narrativo in markdown."""

from crewai import Agent

WRITER_CONFIG = {
    "role": "Sports Journalist",
    "goal": (
        "Generare un report markdown professionale e leggibile con ranking "
        "squadre per TacticalPulse Score e highlight dei pattern piu interessanti."
    ),
    "backstory": (
        "Sei un giornalista sportivo specializzato in analisi tattica e "
        "statistica. Scriví report chiari e coinvolgenti che spiegano i dati "
        "complessi in modo accessibile. Traduci i numeri del TacticalPulse Index "
        "in narrativa sportiva di qualita', evidenziando le squadre con "
        "punteggi anomali e i pattern tattici piu interessanti."
    ),
    "verbose": True,
    "allow_delegation": False,
}


def create_writer_agent(llm=None) -> Agent:
    """Crea e restituisce l'agente Writer.

    Parameters
    ----------
    llm : crewai.LLM, optional
        Istanza LLM per l'agente.

    Returns
    -------
    Agent
    """
    kwargs = {**WRITER_CONFIG}
    if llm:
        kwargs["llm"] = llm
    return Agent(**kwargs)

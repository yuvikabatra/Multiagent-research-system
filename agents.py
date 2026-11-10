import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from pydantic import BaseModel, ConfigDict, Field
from streamlit.errors import StreamlitSecretNotFoundError

from tools import scrape_webpage, web_search

load_dotenv()


def get_groq_api_key() -> str:
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except StreamlitSecretNotFoundError:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to .env or Streamlit Cloud Secrets."
        )

    return str(api_key)


GROQ_API_KEY = get_groq_api_key()
GROQ_MODEL = "openai/gpt-oss-20b"

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=4096,
    timeout=90,
    max_retries=2,
)


class ResearchReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executive_summary: str = Field(
        description="A concise overview of the most important answer and evidence."
    )
    introduction: str = Field(
        description="A brief overview of the topic and its significance."
    )
    key_findings: list[str] = Field(
        description="Four evidence-based findings from different sources."
    )
    evidence_and_implications: list[str] = Field(
        description="Three points connecting evidence to real-world implications."
    )
    limitations: list[str] = Field(
        description="Two limitations, uncertainties, or gaps in the available research."
    )
    improvements_to_be_done: list[str] = Field(
        description="Three actionable improvements with concrete topic-specific examples."
    )
    conclusion: str = Field(
        description="A summary of the research findings and their implications."
    )
    sources: list[str] = Field(
        description="All URLs of the sources used in the report."
    )


class ResearchReportCritique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(
        description="A score out of 10 based on the quality of the research response."
    )
    score_rationale: str = Field(
        description="Specific reasons for the score, addressing relevance, evidence support, clarity, and completeness."
    )
    feedback: str = Field(description="A concise overall quality assessment.")
    suggestions: list[str] = Field(
        description="Three prioritized, actionable suggestions grounded in visible gaps."
    )
    verdict: str = Field(description="A one-line summary of the evaluation.")


class SummaryReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="A useful one-line answer about the research topic.")
    sources: list[str] = Field(description="The source URLs used for this summary.")


class ResearchAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description="A concise answer grounded only in the retrieved research sources."
    )
    source_ids: list[str] = Field(
        description="IDs of retrieved sources that directly support the answer."
    )


def build_web_search_agent():
    return create_agent(model=llm, tools=[web_search])


def build_web_scraping_agent():
    return create_agent(model=llm, tools=[scrape_webpage])


writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Write a structured, balanced, evidence-based research report. "
        "Use only facts supported by the supplied research. Include the relevant "
        "source URL with each key finding and evidence-and-implication point. "
        "Treat search snippets as preliminary evidence and do not fill gaps with "
        "outside knowledge.",
    ),
    (
        "human",
        """Topic: {Topic}

Use the research below.

Return:
- Executive summary
- Introduction
- Exactly 4 key findings
- Exactly 3 evidence-and-implication points
- Exactly 2 limitations
- Exactly 3 improvements to be done
- Conclusion
- Source URLs

Each improvement must include a concrete topic-specific example. Keep every
section concise.

Research:
{research}""",
    ),
])

writer_model = llm.with_structured_output(
    ResearchReport,
    method="json_schema",
    strict=True,
)
writer_chain = (
    writer_prompt
    | writer_model
    | RunnableLambda(lambda report: report.model_dump())
)


summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a precise one-line research summary."),
    (
        "human",
        """Topic: {Topic}

Using the research below, write one complete sentence that directly explains
the most important answer or finding. Do not return only URLs.

Research:
{research}""",
    ),
])

summary_model = llm.with_structured_output(
    SummaryReport,
    method="json_schema",
    strict=True,
)
summary_chain = (
    summary_prompt
    | summary_model
    | RunnableLambda(lambda report: report.model_dump())
)


qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Answer the question using only the retrieved evidence. If the evidence
does not support an answer, say so clearly and return no source IDs. Do not use
outside knowledge. Return only source IDs shown in the evidence, and include
only sources that directly support your answer.""",
    ),
    (
        "human",
        """Question: {question}

Retrieved evidence:
{context}""",
    ),
])
qa_chain = qa_prompt | llm.with_structured_output(
    ResearchAnswer,
    method="json_schema",
    strict=True,
)


critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Evaluate this research response fairly and constructively. It may be
a full report or a one-line summary; judge it appropriately for its format.
Base the score rationale on visible relevance, evidence support, clarity, and
completeness. Never invent shortcomings.""",
    ),
    (
        "human",
        """Review this research response. Give a score from 0 to 10, explain the
specific reasons for that score, provide concise overall feedback, give exactly
3 prioritized actionable suggestions, and give a one-line verdict. Do not
penalize a summary merely for being short.

Research Response:
{research_report}""",
    ),
])

critic_model = llm.with_structured_output(
    ResearchReportCritique,
    method="json_schema",
    strict=True,
)
critic_chain = (
    critic_prompt
    | critic_model
    | RunnableLambda(lambda report: report.model_dump())
)

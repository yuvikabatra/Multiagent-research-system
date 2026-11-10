from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tools import web_search, scrape_webpage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import os

load_dotenv()

def get_config(key: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(key)
    except StreamlitSecretNotFoundError:
        value = None

    if value is None or value == "":
        value = os.getenv(key, default)

    return value if value is not None else default


OLLAMA_BASE_URL = (
    (get_config("OLLAMA_BASE_URL", "http://localhost:11434") or "")
    .strip()
    .rstrip("/")
    or "http://localhost:11434"
)
OLLAMA_PROXY_TOKEN = get_config("OLLAMA_PROXY_TOKEN")

ollama_kwargs = {}
if OLLAMA_PROXY_TOKEN:
    ollama_kwargs["client_kwargs"] = {
        "headers": {"Authorization": f"Bearer {OLLAMA_PROXY_TOKEN}"}
    }

# model setup
llm = ChatOllama(
    model="qwen2.5:3b",
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_predict=1300,
    sync_client_kwargs={"timeout": 45},
    async_client_kwargs={"timeout": 45},
    **ollama_kwargs,
)


class ResearchReport(BaseModel):
    executive_summary: str = Field(description="A concise overview of the most important answer and evidence.")
    introduction: str = Field(description="A brief overview of the topic and its significance.")
    key_findings: list[str] = Field(description="Four evidence-based findings from different sources.")
    evidence_and_implications: list[str] = Field(description="Three points connecting evidence to real-world implications.")
    limitations: list[str] = Field(description="Two limitations, uncertainties, or gaps in the available research.")
    improvements_to_be_done: list[str] = Field(description="Three actionable improvements for making this report stronger; each item must include a concrete topic-specific example.")
    conclusion: str = Field(description="A summary of the research findings and their implications.")
    sources: list[str] = Field(description="All URLs of the sources used in the report.")


class ResearchReportCritique(BaseModel):
    score: float = Field(description="A score out of 10 based on the quality of the research response.")
    score_rationale: str = Field(description="Specific reasons for the score, addressing relevance, evidence support, clarity, and completeness based on this response.")
    feedback: str = Field(description="A concise overall quality assessment of the research response.")
    suggestions: list[str] = Field(description="Three prioritized, actionable suggestions grounded in gaps visible in this response.")
    verdict: str = Field(description="A one-line summary of the evaluation.")


class SummaryReport(BaseModel):
    summary: str = Field(description="A useful one-line answer about the research topic.")
    sources: list[str] = Field(description="The source URLs used for this summary.")


class ResearchAnswer(BaseModel):
    answer: str = Field(description="A concise answer grounded only in the retrieved research sources.")
    source_ids: list[str] = Field(description="IDs of retrieved sources that directly support the answer.")


#1st agent: Web Search Agent
def build_web_search_agent():
    return create_agent(
  model=llm,
      tools=[web_search])
    
#2nd agent: Web Scraping Agent
def build_web_scraping_agent():
    return create_agent(
  model=llm,
  tools=[scrape_webpage])
    
#writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a structured, balanced, evidence-based research report. Avoid unsupported claims."),
    ("human", """Topic: {Topic}

Use the research below. Return an executive summary, introduction, exactly 4 key findings, 3 evidence-and-implication points, 2 limitations, 3 improvements to be done, a conclusion, and source URLs. Use 1-2 concise sentences for the summary, introduction, and conclusion, and one concise sentence per list item. Each improvement must include a concrete topic-specific example, such as a missing comparison, statistic, case study, or source that would strengthen the report.

Research:
{research}"""),
])

writer_llm = llm.bind(options={"num_predict": 900})
writer_chain = writer_prompt | writer_llm.bind(format=ResearchReport.model_json_schema()) | JsonOutputParser()

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a precise one-line research summary."),
    ("human", """Topic: {Topic}

Using the research below, write one complete sentence that directly explains the most important answer or finding about the topic. Do not return only URLs.

Research:
{research}"""),
])

summary_llm = llm.bind(options={"num_predict": 160})
summary_chain = summary_prompt | summary_llm.bind(format=SummaryReport.model_json_schema()) | JsonOutputParser()

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the question using only the retrieved evidence. If the evidence does not support an answer, say so clearly and return no source IDs. Do not use outside knowledge. Return only source IDs shown in the evidence, and include only sources that directly support your answer."""),
    ("human", """Question: {question}

Retrieved evidence:
{context}"""),
])

qa_chain = qa_prompt | llm.with_structured_output(ResearchAnswer)

#critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "Evaluate this research response fairly and constructively. It may be a full report or a one-line summary; judge it appropriately for its format. Base the score rationale on visible relevance, evidence support, clarity, and completeness. Never invent shortcomings that are not visible in the response."),
    ("human", """Review this research response. Give a score from 0 to 10, explain the specific reasons for that score, provide concise overall feedback, give exactly 3 prioritized actionable suggestions, and give a one-line verdict. Suggestions must address visible gaps and be relevant even when the response is a short summary. Do not penalize a summary merely for being short.

Research Response: {research_report}
"""),
])
critic_llm = llm.bind(options={"num_predict": 350})
critic_chain = critic_prompt | critic_llm.bind(format=ResearchReportCritique.model_json_schema()) | JsonOutputParser()

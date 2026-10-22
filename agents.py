from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from tools import web_search, scrape_webpage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

load_dotenv()

#model setup
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    max_tokens=1300,
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
    score: float = Field(description="A score out of 10 based on the quality of the report.")
    feedback: str = Field(description="Detailed feedback on the report, including areas for improvement.")
    verdict: str = Field(description="A one-line summary of the evaluation.")


class SummaryReport(BaseModel):
    summary: str = Field(description="A useful one-line answer about the research topic.")
    sources: list[str] = Field(description="The source URLs used for this summary.")


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

Use the research below. Return an executive summary, introduction, exactly 4 key findings, 3 evidence-and-implication points, 2 limitations, 3 improvements to be done, a conclusion, and source URLs. Each improvement must include a concrete topic-specific example, such as a missing comparison, statistic, case study, or source that would strengthen the report. Keep every item concise.

Research:
{research}"""),
])

writer_chain = writer_prompt | llm.with_structured_output(ResearchReport)

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a precise one-line research summary."),
    ("human", """Topic: {Topic}

Using the research below, write one complete sentence that directly explains the most important answer or finding about the topic. Do not return only URLs.

Research:
{research}"""),
])

summary_chain = summary_prompt | llm.with_structured_output(SummaryReport)

#critic chain
critic_prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate this research report briefly and constructively."),
        ("human", """Review this report. Give a score from 0 to 10, brief feedback, and a one-line verdict.

Research Report: {research_report}
"""),
])
critic_llm = llm.bind(max_tokens=300)
critic_chain = critic_prompt | critic_llm.with_structured_output(ResearchReportCritique)

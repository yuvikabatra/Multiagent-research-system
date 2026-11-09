import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from agents import (
    OLLAMA_BASE_URL,
    ResearchReport,
    ResearchReportCritique,
    SummaryReport,
    critic_chain,
    qa_chain,
    summary_chain,
    writer_chain,
)
from tools import format_results, scrape_webpage, search_web_results


def raise_model_connection_error(error: Exception, step_name: str) -> None:
    if "connection refused" in str(error).lower():
        raise RuntimeError(
            f"{step_name} could not connect to Ollama at {OLLAMA_BASE_URL}. "
            "For a deployed app, set OLLAMA_BASE_URL to an Ollama server reachable "
            "from the deployment; localhost refers to the app container."
        ) from error


def invoke_with_retry(chain, payload, step_name: str):
    """Retry transient model failures without hiding other errors."""
    delays = (2, 5, 10)
    for attempt in range(len(delays) + 1):
        try:
            return chain.invoke(payload)
        except Exception as error:
            error_text = str(error).lower()
            raise_model_connection_error(error, step_name)
            is_rate_limited = "429" in error_text or "rate_limited" in error_text
            if not is_rate_limited or attempt == len(delays):
                raise
            delay = delays[attempt]
            print(f"{step_name} rate-limited; retrying in {delay} seconds...")
            time.sleep(delay)


def stream_with_retry(chain, payload, step_name: str, output_model, on_update=None, on_retry=None):
    """Stream partial JSON objects, then validate the complete structured result."""
    delays = (2, 5, 10)
    for attempt in range(len(delays) + 1):
        latest_output = {}
        try:
            for partial_output in chain.stream(payload):
                latest_output = partial_output
                if on_update:
                    on_update(partial_output)
            return output_model.model_validate(latest_output)
        except Exception as error:
            error_text = str(error).lower()
            raise_model_connection_error(error, step_name)
            is_rate_limited = "429" in error_text or "rate_limited" in error_text
            if not is_rate_limited or attempt == len(delays):
                raise
            delay = delays[attempt]
            message = f"{step_name} rate-limited; retrying in {delay} seconds..."
            print(message)
            if on_retry:
                on_retry(message)
            time.sleep(delay)


def answer_research_question(question: str, source_documents: list[dict]) -> dict:
    """Retrieve relevant research chunks and answer with validated source IDs."""
    if not question.strip():
        raise ValueError("Please enter a question.")

    documents = [
        Document(page_content=source["content"], metadata={"source": source["url"]})
        for source in source_documents
        if source.get("url") and source.get("content", "").strip()
    ]
    if not documents:
        raise ValueError("There is no source text available to answer questions.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    retriever = BM25Retriever.from_documents(chunks, k=min(4, len(chunks)))
    retrieved_chunks = retriever.invoke(question)

    source_ids_by_url = {}
    context_parts = []
    for chunk in retrieved_chunks:
        url = chunk.metadata.get("source")
        if not url:
            continue
        if url not in source_ids_by_url:
            source_ids_by_url[url] = f"S{len(source_ids_by_url) + 1}"
        source_id = source_ids_by_url[url]
        context_parts.append(f"[{source_id}] {url}\n{chunk.page_content}")

    answer = invoke_with_retry(
        qa_chain,
        {"question": question, "context": "\n\n".join(context_parts)},
        "Research Q&A",
    )
    valid_source_ids = set(source_ids_by_url.values())
    citations = [
        {"id": source_id, "url": url}
        for url, source_id in source_ids_by_url.items()
        if source_id in answer.source_ids and source_id in valid_source_ids
    ]
    return {"answer": answer.answer, "sources": citations}


def run_research_pipeline(
    topic: str,
    report_mode: str = "detailed",
    on_progress: Callable[[str, str], None] | None = None,
    on_writer_update: Callable[[dict], None] | None = None,
    on_reviewer_update: Callable[[dict], None] | None = None,
) -> dict:
    """Search, scrape, write, and critique a report for the supplied topic."""
    if not topic.strip():
        raise ValueError("Please enter a research topic.")
    if report_mode not in {"detailed", "summary"}:
        raise ValueError("Report mode must be 'detailed' or 'summary'.")

    def progress(stage: str, message: str) -> None:
        print(message)
        if on_progress:
            on_progress(stage, message)

    progress("search", f"Sending research query to web search: {topic}")
    search_data = search_web_results(topic)
    if not isinstance(search_data, dict):
        raise TypeError("Web search must return a results dictionary.")
    results = search_data.get("results", [])
    if not results:
        raise RuntimeError("Web search returned no results.")
    progress("search", f"Search returned {len(results)} results.")

    scraped_contents = []
    sources = []
    source_documents = []
    max_sources = min(5, len(results))
    source_urls = [result.get("url") for result in results[:max_sources] if result.get("url")]
    fetched_by_index = {}
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(source_urls)))) as executor:
        pending_scrapes = {}
        for index, url in enumerate(source_urls, start=1):
            progress("reader", f"Fetching source {index} of {len(source_urls)}.")
            pending_scrapes[executor.submit(scrape_webpage.invoke, url)] = (index, url)

        for pending_scrape in as_completed(pending_scrapes):
            index, url = pending_scrapes[pending_scrape]
            try:
                content = pending_scrape.result()
            except Exception as error:
                progress("reader", f"Source {index} fetch failed.")
                continue
            if not content.startswith("Error fetching"):
                fetched_by_index[index] = (url, content)
                progress("reader", f"Source {index} of {len(source_urls)} fetched.")
            else:
                progress("reader", f"Source {index} fetch failed.")

    for url, content in fetched_by_index.values():
        scraped_contents.append(f"Source: {url}\n{content}")
        sources.append(url)
        source_documents.append({"url": url, "content": content})

    research = ("\n\n".join(scraped_contents) or format_results(search_data))[:6500]
    if not source_documents:
        source_documents = [
            {"url": result["url"], "content": content[:1000]}
            for result in results[:max_sources]
            if (url := result.get("url"))
            and (content := result.get("content") or result.get("snippet") or "").strip()
        ]
    progress("writer", "Writer agent is shaping the research into a report")
    report_chain = writer_chain if report_mode == "detailed" else summary_chain
    report_model = ResearchReport if report_mode == "detailed" else SummaryReport
    report = stream_with_retry(
        report_chain,
        {"Topic": topic, "research": research},
        "Report writer",
        report_model,
        on_writer_update,
        lambda message: progress("writer", message),
    )
    report.sources = sources or report.sources

    critique = None
    if os.getenv("ENABLE_CRITIC", "true").lower() == "true":
        progress("reviewer", "Reviewer agent is scoring the report")
        critique = stream_with_retry(
            critic_chain,
            {"research_report": report.model_dump_json()},
            "Report critic",
            ResearchReportCritique,
            on_reviewer_update,
            lambda message: progress("reviewer", message),
        )
    return {
        "topic": topic,
        "report_mode": report_mode,
        "report": report,
        "critique": critique,
        "source_documents": source_documents,
    }


def print_report(result: dict) -> None:
    report = result["report"]
    critique = result["critique"]
    print("\n" + "=" * 70)
    print(f"RESEARCH REPORT: {result['topic']}")
    print("=" * 70)
    if result["report_mode"] == "summary":
        print(f"\nSUMMARY\n{report.summary}")
    else:
        print(f"\nINTRODUCTION\n{report.introduction}")
        print("\nKEY FINDINGS")
        for index, finding in enumerate(report.key_findings, start=1):
            print(f"{index}. {finding}")
        print("\nIMPROVEMENTS TO BE DONE")
        for improvement in getattr(report, "improvements_to_be_done", []):
            print(f"- {improvement}")
        print(f"\nCONCLUSION\n{report.conclusion}")
    print("\nSOURCES")
    for source in report.sources:
        print(f"- {source}")
    if critique:
        print(f"\nCRITIC SCORE: {critique.score}/10")
        print(f"FEEDBACK: {critique.feedback}")
        print(f"VERDICT: {critique.verdict}")


if __name__ == "__main__":
    topic = input("Enter a research topic: ").strip()
    print("\nChoose report type:")
    print("1. Detailed report")
    print("2. One-line summary")
    choice = input("Enter 1/2 or detailed/summary: ").strip().lower()
    report_mode = "summary" if choice in {"2", "summary"} else "detailed"
    try:
        print_report(run_research_pipeline(topic, report_mode))
    except Exception as error:
        error_text = str(error)
        if "connection" in error_text or "ollama" in error_text:
            print("\nResearch pipeline failed: Ollama is not running.")
            print("Start Ollama, then run this command again: ollama serve")
        elif "429" in error_text or "rate_limited" in error_text:
            print("\nResearch pipeline failed: the local model is busy.")
            print("Wait briefly and run again.")
        else:
            print(f"\nResearch pipeline failed: {error}")
import time
import os
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from typing import TypeVar

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents import (
    ResearchReport,
    ResearchReportCritique,
    SummaryReport,
    critic_chain,
    qa_chain,
    summary_chain,
    writer_chain,
)
from tools import format_results, scrape_webpage, search_web_results

REQUEST_DEADLINE_SECONDS = 100
SEARCH_TIMEOUT_SECONDS = 10
SCRAPE_TIMEOUT_SECONDS = 12
WRITER_RESERVE_SECONDS = 3
MAX_SOURCES_TO_SCRAPE = 3
REVIEW_TIMEOUT_SECONDS = 20

T = TypeVar("T")


def run_bounded(
    operation: Callable[[Callable[[tuple], None]], T],
    timeout_seconds: float,
    on_event: Callable[[tuple], None] | None = None,
) -> T:
    """Run blocking work without letting it hold the request past its deadline."""
    events: queue.Queue[tuple[str, object]] = queue.Queue()

    def run() -> None:
        try:
            emit = lambda event: events.put(("event", event))
            events.put(("result", operation(emit)))
        except Exception as error:
            events.put(("error", error))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    deadline = time.monotonic() + max(0, timeout_seconds)

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The operation exceeded its time budget.")
        try:
            event_type, payload = events.get(timeout=min(0.1, remaining))
        except queue.Empty:
            continue

        if event_type == "event":
            if on_event:
                on_event(payload)
        elif event_type == "error":
            if isinstance(payload, Exception):
                raise payload
            raise RuntimeError("The bounded operation failed unexpectedly.")
        else:
            return payload


def build_fallback_report(topic: str, report_mode: str, source_documents: list[dict]):
    """Create an explicitly limited report using the retrieved evidence."""
    source_urls = [source["url"] for source in source_documents if source.get("url")]
    uses_search_snippets = any(
        source.get("source_type") == "search_snippet"
        for source in source_documents
    )
    evidence_basis = (
        "available search snippets"
        if uses_search_snippets
        else "retrieved webpage text"
        if source_documents
        else "available search material"
    )
    evidence_limitation = (
        "Search snippets may omit context, methods, dates, or qualifications from "
        "the linked pages."
        if uses_search_snippets
        else "Extracted webpage text may omit context, methods, dates, or "
        "qualifications from the linked pages."
        if source_documents
        else "No source text was available to substantiate findings."
    )
    evidence = []
    seen_sentences = set()
    for source in source_documents:
        url = source.get("url", "")
        for sentence in re.split(r"(?<=[.!?])\s+", source.get("content", "")):
            sentence = " ".join(sentence.split())
            if not sentence or sentence.casefold() in seen_sentences:
                continue
            seen_sentences.add(sentence.casefold())
            evidence.append(f"{sentence[:260]} (Source: {url})")
            if len(evidence) == 4:
                break
        if len(evidence) == 4:
            break

    if report_mode == "summary":
        if evidence:
            summary = (
                f"Quick findings about {topic}, based on {evidence_basis}: "
                f"{evidence[0]}"
            )
        else:
            summary = (
                f"No search evidence was available to produce a sourced summary "
                f"about {topic}."
            )
        return SummaryReport(summary=summary, sources=source_urls)

    findings = evidence + [
        "Available search material did not establish another independent finding."
    ] * (4 - len(evidence))
    summary = (
        f"Quick report about {topic}, based only on {evidence_basis}. "
        f"{evidence[0]}"
        if evidence
        else f"No search evidence was available to substantiate findings about {topic}."
    )
    return ResearchReport(
        executive_summary=summary,
        introduction=(
            f"This time-limited brief summarizes {evidence_basis} about {topic}. "
            f"{evidence_limitation} Verify claims against the linked sources before "
            "relying on them."
        ),
        key_findings=findings,
        evidence_and_implications=[
            f"{finding} Its broader implications need confirmation from the full source."
            for finding in findings[:3]
        ],
        limitations=[
            "This brief uses the time-limited fallback rather than AI synthesis.",
            evidence_limitation,
        ],
        improvements_to_be_done=[
            f"Compare {topic} across at least two independent primary sources and include dates or measurements.",
            f"Add a recent dataset or case study relevant to {topic}, including its methodology and sample size.",
            f"Verify each claim against the linked source; this quick brief uses "
            f"{evidence_basis}.",
        ],
        conclusion=(
            "Treat this quick brief as preliminary; verify the claims against the linked "
            "full sources before relying on them."
        ),
        sources=source_urls,
    )


def raise_model_connection_error(error: Exception, step_name: str) -> None:
    error_text = str(error).lower()
    if any(
        marker in error_text
        for marker in (
            "connection refused",
            "name or service not known",
            "getaddrinfo failed",
            "nodename nor servname",
            "temporary failure in name resolution",
        )
    ):
        raise RuntimeError(
            f"{step_name} cannot reach the Groq API. Check the deployment's "
            "internet/DNS connectivity and confirm GROQ_API_KEY is configured "
            "in Streamlit Secrets or the hosting environment."
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
    include_critic: bool | None = None,
) -> dict:
    """Search, scrape, and write a report; optionally critique it."""
    if not topic.strip():
        raise ValueError("Please enter a research topic.")
    if report_mode not in {"detailed", "summary"}:
        raise ValueError("Report mode must be 'detailed' or 'summary'.")

    deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS

    def remaining_budget() -> float:
        return max(0, deadline - time.monotonic() - WRITER_RESERVE_SECONDS)

    def progress(stage: str, message: str) -> None:
        print(message)
        if on_progress:
            on_progress(stage, message)

    def dispatch_event(event: tuple) -> None:
        event_type, *payload = event
        if event_type == "progress":
            progress(payload[0], payload[1])
        elif event_type == "writer_update" and on_writer_update:
            on_writer_update(payload[0])
        elif event_type == "reviewer_update" and on_reviewer_update:
            on_reviewer_update(payload[0])

    should_run_critic = (
        include_critic
        if include_critic is not None
        else os.getenv("ENABLE_CRITIC", "false").lower() == "true"
    )

    progress("search", f"Sending research query to web search: {topic}")
    fallback_reason = None
    try:
        search_data = run_bounded(
            lambda _emit: search_web_results(topic),
            min(SEARCH_TIMEOUT_SECONDS, remaining_budget()),
        )
    except TimeoutError:
        search_data = {"results": []}
        fallback_reason = "Web search exceeded its time budget."
    except Exception as error:
        search_data = {"results": []}
        fallback_reason = f"Web search failed: {error}"

    if not isinstance(search_data, dict):
        raise TypeError("Web search must return a results dictionary.")
    results = search_data.get("results", [])
    if not results:
        fallback_reason = fallback_reason or "Web search returned no results."
        report = build_fallback_report(topic, report_mode, [])
        progress("writer", "Returning a quick fallback report.")
        return {
            "topic": topic,
            "report_mode": report_mode,
            "report": report,
            "critique": None,
            "source_documents": [],
            "review_note": (
                "Quality review was skipped because no report could be generated "
                "without search results."
                if should_run_critic
                else "Quality review was not requested."
            ),
            "review_requested": should_run_critic,
            "generation_note": (
                f"{fallback_reason} The returned report contains no sourced claims."
            ),
        }
    progress("search", f"Search returned {len(results)} results.")

    scraped_contents = []
    sources = []
    source_documents = []
    max_sources = min(MAX_SOURCES_TO_SCRAPE, len(results))
    source_urls = [result.get("url") for result in results[:max_sources] if result.get("url")]
    for index in range(1, len(source_urls) + 1):
        progress("reader", f"Fetching source {index} of {len(source_urls)}.")

    def scrape_sources() -> dict:
        fetched = {}
        with ThreadPoolExecutor(max_workers=min(5, max(1, len(source_urls)))) as executor:
            pending_scrapes = {
                executor.submit(scrape_webpage.invoke, url): (index, url)
                for index, url in enumerate(source_urls, start=1)
            }
            for pending_scrape in as_completed(pending_scrapes):
                index, url = pending_scrapes[pending_scrape]
                try:
                    content = pending_scrape.result()
                except Exception:
                    continue
                if not content.startswith("Error fetching"):
                    fetched[index] = (url, content)
        return fetched

    scrape_timed_out = False
    try:
        fetched_by_index = run_bounded(
            lambda _emit: scrape_sources(),
            min(SCRAPE_TIMEOUT_SECONDS, remaining_budget()),
        )
    except TimeoutError:
        fetched_by_index = {}
        scrape_timed_out = True
        progress("reader", "Source reading exceeded its time budget; using search snippets.")
    except Exception as error:
        fetched_by_index = {}
        progress("reader", f"Source reading failed; using search snippets: {error}")

    for url, content in fetched_by_index.values():
        scraped_contents.append(f"Source: {url}\n{content}")
        sources.append(url)
        source_documents.append(
            {"url": url, "content": content, "source_type": "webpage"}
        )

    if not source_documents:
        source_documents = [
            {
                "url": result["url"],
                "content": content[:1000],
                "source_type": "search_snippet",
            }
            for result in results[:max_sources]
            if (url := result.get("url"))
            and (content := result.get("content") or result.get("snippet") or "").strip()
        ]
    research = ("\n\n".join(scraped_contents) or format_results(search_data))[:6500]
    if scrape_timed_out:
        fallback_reason = "Source reading exceeded its time budget."

    progress("writer", "Writer agent is shaping the research into a report")
    report_chain = writer_chain if report_mode == "detailed" else summary_chain
    report_model = ResearchReport if report_mode == "detailed" else SummaryReport
    writer_timed_out = False
    writer_failure_reason = None
    if remaining_budget() <= 0:
        writer_timed_out = True
    else:
        try:
            report = run_bounded(
                lambda emit: stream_with_retry(
                    report_chain,
                    {"Topic": topic, "research": research},
                    "Report writer",
                    report_model,
                    lambda partial: emit(("writer_update", partial)),
                    lambda message: emit(("progress", "writer", message)),
                ),
                remaining_budget(),
                dispatch_event,
            )
            report.sources = sources or report.sources
        except TimeoutError:
            writer_timed_out = True
        except Exception as error:
            writer_timed_out = True
            writer_failure_reason = f"AI report writing failed: {error}"

    if writer_timed_out:
        report = build_fallback_report(topic, report_mode, source_documents)
        report.sources = sources or report.sources
        if writer_failure_reason:
            fallback_reason = writer_failure_reason
        elif fallback_reason:
            fallback_reason += " AI report writing also exceeded its time budget."
        else:
            fallback_reason = "AI report writing exceeded the time budget."

    critique = None
    review_note = None
    if should_run_critic:
        if writer_timed_out:
            review_note = (
                "Quality review was skipped because AI report writing failed. "
                "Check the Groq API key and connectivity, then try again."
            )
            progress("reviewer", review_note)
        elif remaining_budget() > 0:
            progress("reviewer", "Reviewer agent is scoring the report")
            try:
                critique = run_bounded(
                    lambda emit: stream_with_retry(
                        critic_chain,
                        {"research_report": report.model_dump_json()},
                        "Report critic",
                        ResearchReportCritique,
                        lambda partial: emit(("reviewer_update", partial)),
                        lambda message: emit(("progress", "reviewer", message)),
                    ),
                    min(REVIEW_TIMEOUT_SECONDS, remaining_budget()),
                    dispatch_event,
                )
            except TimeoutError:
                review_note = "Quality review exceeded its time budget; the report is ready."
                progress("reviewer", review_note)
            except Exception as error:
                review_note = f"Quality review failed: {error}"
                progress("reviewer", review_note)
        else:
            review_note = "Quality review skipped to return the report within its time budget."
            progress("reviewer", review_note)
    else:
        review_note = "Quality review was not requested."
    if source_documents and any(
        source.get("source_type") == "search_snippet"
        for source in source_documents
    ):
        evidence_note = (
            "Claims use only available search snippets; verify the full sources "
            "before relying on them."
        )
    elif source_documents:
        evidence_note = (
            "Claims use retrieved webpage text, which may be incomplete; verify "
            "them against the linked sources before relying on them."
        )
    else:
        evidence_note = "No source text was available; the report contains no sourced claims."

    return {
        "topic": topic,
        "report_mode": report_mode,
        "report": report,
        "critique": critique,
        "review_note": review_note,
        "review_requested": should_run_critic,
        "source_documents": source_documents,
        "generation_note": (
            f"Quick fallback report: {fallback_reason} {evidence_note}"
            if writer_timed_out
            else (
                f"{fallback_reason} {evidence_note}"
                if fallback_reason
                else None
            )
        ),
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
        if "connection" in error_text or "groq" in error_text:
            print("\nResearch pipeline failed: the Groq API is unreachable.")
            print("Check GROQ_API_KEY and the network connection, then try again.")
        elif "429" in error_text or "rate_limited" in error_text:
            print("\nResearch pipeline failed: the local model is busy.")
            print("Wait briefly and run again.")
        else:
            print(f"\nResearch pipeline failed: {error}")
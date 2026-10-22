import time
import os
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable

from agents import critic_chain, summary_chain, writer_chain
from tools import format_results, scrape_webpage, web_search


def invoke_with_retry(chain, payload, step_name: str):
    """Retry transient model failures without hiding other errors."""
    delays = (2, 5, 10)
    for attempt in range(len(delays) + 1):
        try:
            return chain.invoke(payload)
        except Exception as error:
            error_text = str(error).lower()
            is_rate_limited = "429" in error_text or "rate_limited" in error_text
            if not is_rate_limited or attempt == len(delays):
                raise
            delay = delays[attempt]
            print(f"{step_name} rate-limited; retrying in {delay} seconds...")
            time.sleep(delay)


def run_research_pipeline(
    topic: str,
    report_mode: str = "detailed",
    on_progress: Callable[[str, str], None] | None = None,
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

    progress("search", f"Search agent is finding sources for: {topic}")
    search_data = web_search.invoke(topic)
    results = search_data.get("results", [])
    if not results:
        raise RuntimeError("Web search returned no results.")

    scraped_contents = []
    sources = []
    max_sources = min(5, len(results))
    source_urls = [result.get("url") for result in results[:max_sources] if result.get("url")]
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(source_urls)))) as executor:
        pending_scrapes = []
        for index, url in enumerate(source_urls, start=1):
            progress("reader", f"Reader agent is reading source {index} of {len(source_urls)}")
            pending_scrapes.append((url, executor.submit(scrape_webpage.invoke, url)))

        for url, pending_scrape in pending_scrapes:
            content = pending_scrape.result()
            if not content.startswith("Error fetching"):
                scraped_contents.append(f"Source: {url}\n{content}")
                sources.append(url)

    research = ("\n\n".join(scraped_contents) or format_results(search_data))[:6500]
    progress("writer", "Writer agent is shaping the research into a report")
    report_chain = writer_chain if report_mode == "detailed" else summary_chain
    report = invoke_with_retry(
        report_chain,
        {"Topic": topic, "research": research},
        "Report writer",
    )
    report.sources = sources or report.sources

    critique = None
    if report_mode == "detailed" and os.getenv("ENABLE_CRITIC", "true").lower() == "true":
        time.sleep(2)
        progress("reviewer", "Reviewer agent is scoring the report")
        critique = invoke_with_retry(
            critic_chain,
            {"research_report": report.model_dump_json()},
            "Report critic",
        )
    return {
        "topic": topic,
        "report_mode": report_mode,
        "report": report,
        "critique": critique,
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
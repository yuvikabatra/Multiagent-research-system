
import os
import streamlit as st
import requests

from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

# Load environment variables from .env (local development)
load_dotenv()

MAX_SEARCH_RESULTS = 5


def get_tavily_api_key():
    """
    Get Tavily API key from Streamlit Cloud Secrets
    or local .env file.
    """

    try:
        api_key = st.secrets.get("TAVILY_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not found. "
            "Add it to Streamlit Secrets or your .env file."
        )

    return api_key


# Initialize Tavily Client
TAVILY_API_KEY = get_tavily_api_key()

tavily = TavilyClient(
    api_key=TAVILY_API_KEY
)


# ==========================================
# 2. FORMAT TAVILY SEARCH RESULTS
# ==========================================

def format_results(result: dict) -> str:

    output = []

    for item in result.get("results", []):

        content = (
            item.get("content")
            or item.get("snippet")
            or ""
        )

        output.append(
            f"Title: {item.get('title', 'Untitled')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {content[:300]}\n"
        )

    if not output:
        return "No search results found."

    return "\n".join(output)


# ==========================================
# 3. WEB SEARCH TOOL
# ==========================================

def search_web_results(query: str) -> dict:
    """Return raw Tavily search results for use by the research pipeline."""
    try:
        return tavily.search(
            query=query,
            max_results=MAX_SEARCH_RESULTS,
            timeout=8,
        )
    except requests.exceptions.ConnectionError:
        return tavily.search(
            query=query,
            max_results=MAX_SEARCH_RESULTS,
            timeout=8,
        )


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information
    related to the given query using Tavily API.
    """

    try:

        results = search_web_results(query)

        formatted_results = format_results(results)

        return formatted_results

    except Exception as e:

        return f"Error during Tavily search: {str(e)}"


# ==========================================
# 4. WEBPAGE SCRAPING TOOL
# ==========================================

@tool
def scrape_webpage(url: str) -> str:
    """
    Scrape the content of a webpage given its URL.
    Extracts readable text using BeautifulSoup.
    """

    try:

        response = requests.get(
            url,
            timeout=6,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary HTML elements
        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside"
            ]
        ):
            tag.decompose()

        # Extract webpage text
        page_content = soup.get_text(
            separator=" ",
            strip=True
        )

        return page_content[:1000]

    except requests.RequestException as e:

        return f"Error fetching webpage: {str(e)}"


# ==========================================
# 5. LOCAL TESTING
# ==========================================

if __name__ == "__main__":

    query = "Latest advancements in AI research"

    print("\nSearching the web...\n")

    result = web_search.invoke(
        {"query": query}
    )

    print(result)

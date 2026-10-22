from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()

tavily=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information related to the given query using Tavily API.
    """
    results=tavily.search(query=query, max_results=5)
    return results


def format_results(result: dict) -> str:
    out = []
    for item in result.get("results", []):
        snippet = item.get("snippet") or item.get("content", "")
        out.append(
            f"Title: {item.get('title', 'Untitled')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Snippet: {snippet[:300]}\n"
        )
    return "\n".join(out)


if __name__ == "__main__":
    result = web_search.invoke("What are the latest advancements in AI research?")
    print(format_results(result))
    
@tool
def scrape_webpage(url: str) -> str:
    """
    Scrape the content of a webpage given its URL.
    """
    try:
        response = requests.get(url, timeout=10,headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from the webpage
        for tag in soup(['script', 'style','nav','footer']):  # Remove script, style, nav, header, footer, and aside elements
            tag.decompose()  # Remove script and style elements
        return soup.get_text(separator=' ', strip=True)[:1000]  # Keep multi-source input bounded.
    except requests.RequestException as e:
        return f"Error fetching the webpage: {e}"

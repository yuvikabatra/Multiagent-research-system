# Multiagent-research-system

## Groq configuration

The app uses the `openai/gpt-oss-20b` model through Groq. Set `GROQ_API_KEY`
in your local `.env` file:

```shell
GROQ_API_KEY=your-groq-api-key
TAVILY_API_KEY=your-tavily-api-key
streamlit run app.py
```

For Streamlit Community Cloud, add these values under the app's **Settings >
Secrets**:

```toml
GROQ_API_KEY = "your-groq-api-key"
TAVILY_API_KEY = "your-tavily-api-key"
```

The app reads both keys from environment variables on other hosting platforms
as well. Keep API keys secret; do not commit them to the repository.

### Faster report generation

The detailed report has a 4,096-token output budget so the model has room to
complete the structured report. The quality review is enabled by default and
can be unchecked to skip its second, sequential model call for the quickest result.
To enable review for command-line runs by default, set `ENABLE_CRITIC=true`.
Choose **One-line summary** when you need the fastest, shortest output.

Each research request has a 100-second application-side budget. If search,
source reading, or AI writing exceeds its budget, the app returns a clearly
labeled fallback from the retrieved webpage text or search snippets instead
of waiting for the model indefinitely. This bounds the app's wait under normal
runtime conditions; it cannot guarantee a successful, fully AI-written report when
external services or the hosting platform are unavailable. Fallback reports
identify whether they use search snippets or retrieved webpage text and warn
that claims should be checked against the linked sources.

To keep more time available for report writing, web search is capped at 8
seconds and source scraping at 12 seconds. The app scrapes up to three sources
in parallel and uses the search snippets when source pages are slow.

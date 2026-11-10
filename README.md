# Multiagent-research-system

## Ollama configuration

The app uses the `qwen2.5:3b` model through Ollama. Locally, it connects to
`http://localhost:11434` by default. Start Ollama and pull the model before
running the app:

```shell
ollama pull qwen2.5:3b
streamlit run app.py
```

When deploying, the app cannot connect to Ollama on your personal computer:
`localhost` refers to the deployment container. Run Ollama on a server that the
deployment can reach, make sure that server has the `qwen2.5:3b` model, and set
the `OLLAMA_BASE_URL` environment variable in your hosting platform to that
server's URL (for example, `http://ollama-host:11434`). Do not use
`localhost` unless Ollama is running in the same container as the app. Keep the
Ollama server private or protected rather than exposing it publicly without
access controls.

On Streamlit Community Cloud, add these values under the app's **Settings >
Secrets** (use the actual Ollama URL and keep the Tavily key secret):

```toml
OLLAMA_BASE_URL = "http://ollama-host:11434"
TAVILY_API_KEY = "your-tavily-api-key"
```

The app also reads `OLLAMA_BASE_URL` and `TAVILY_API_KEY` from environment
variables on other hosting platforms.

### Faster report generation

The detailed report is generated with a capped output budget to reduce writer
latency. The quality review is optional because it requires a second, sequential
model call; leave **Run quality review** unchecked for the quickest result.
To enable review for command-line runs by default, set `ENABLE_CRITIC=true`.
Choose **One-line summary** when you need the fastest, shortest output.

Each research request has a 100-second application-side budget. If search,
source reading, or AI writing exceeds its budget, the app returns a clearly
labeled fallback from the search snippets it has received instead of waiting
for the model indefinitely. This bounds the app's wait under normal runtime
conditions; it cannot guarantee a successful, fully AI-written report when
external services or the hosting platform are unavailable.

### Optional Ollama proxy

If Ollama runs on a separate machine, `ollama_proxy.py` can forward authenticated
requests to its local Ollama service. Install the requirements, set
`OLLAMA_PROXY_TOKEN` to a random secret of at least 32 characters on the proxy
host, then run:

```shell
uvicorn ollama_proxy:app --host 0.0.0.0 --port 8000
```

Configure the app with `OLLAMA_BASE_URL` pointing to the proxy and the same
`OLLAMA_PROXY_TOKEN`. The proxy forwards to Ollama at
`http://127.0.0.1:11434`. Only expose it through a private network or a
TLS-enabled, access-controlled gateway; do not expose the unauthenticated
Ollama service directly to the internet.
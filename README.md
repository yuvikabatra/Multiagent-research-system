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
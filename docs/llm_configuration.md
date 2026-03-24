# LLM Configuration

The voice agent benchmark uses **LiteLLM** for universal LLM support. All model routing is configured through the `EVA_MODEL_LIST` environment variable, which defines LiteLLM Router deployments.

## Configuration

All LLM configuration is done through `EVA_MODEL_LIST` in your `.env` file. Each entry maps a **model name** (the alias your code uses) to a **provider-specific model identifier** in `litellm_params.model`.

```bash
EVA_MODEL_LIST='[
  {
    "model_name": "gpt-5.2",
    "litellm_params": {
      "model": "azure/gpt-5.2",
      "api_key": "os.environ/AZURE_API_KEY",
      "api_base": "https://your-resource.openai.azure.com"
    }
  }
]'
```

**Key concepts:**
- `model_name` — Clean alias used in your code and config (e.g., `"gpt-5.2"`)
- `litellm_params.model` — Provider-prefixed identifier for LiteLLM routing (e.g., `"azure/gpt-5.2"`)
- `EVA_MODEL__LLM` env var — Selects which `model_name` to use for the benchmark assistant

## Provider Examples

### Azure OpenAI

```bash
EVA_MODEL_LIST='[
  {
    "model_name": "gpt-5.2",
    "litellm_params": {
      "model": "azure/gpt-5.2",
      "api_key": "os.environ/AZURE_API_KEY",
      "api_base": "https://your-resource.openai.azure.com",
      "api_version": "2024-08-01-preview"
    }
  }
]'
```

### OpenAI (Direct API)

```bash
EVA_MODEL_LIST='[
  {
    "model_name": "gpt-4o",
    "litellm_params": {
      "model": "openai/gpt-4o",
      "api_key": "os.environ/OPENAI_API_KEY"
    }
  }
]'
```

### Anthropic (Claude)

```bash
# Direct API
EVA_MODEL_LIST='[
  {
    "model_name": "claude-sonnet",
    "litellm_params": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "api_key": "os.environ/ANTHROPIC_API_KEY"
    }
  }
]'

# Via AWS Bedrock
EVA_MODEL_LIST='[
  {
    "model_name": "claude-opus",
    "litellm_params": {
      "model": "bedrock/us.anthropic.claude-opus-4-6-v1",
      "aws_access_key_id": "os.environ/AWS_ACCESS_KEY_ID",
      "aws_secret_access_key": "os.environ/AWS_SECRET_ACCESS_KEY"
    }
  }
]'
```

### Google Gemini

```bash
# Via Vertex AI
EVA_MODEL_LIST='[
  {
    "model_name": "gemini-3-pro",
    "litellm_params": {
      "model": "vertex_ai/gemini-3-pro-preview",
      "vertex_project": "your-gcp-project",
      "vertex_location": "global",
      "vertex_credentials": "os.environ/GOOGLE_APPLICATION_CREDENTIALS"
    }
  }
]'

# Direct API
EVA_MODEL_LIST='[
  {
    "model_name": "gemini-3-pro",
    "litellm_params": {
      "model": "gemini/gemini-3-pro-preview",
      "api_key": "os.environ/GOOGLE_API_KEY"
    }
  }
]'
```

### Other Providers

For any LiteLLM-supported provider, use the provider prefix in `litellm_params.model`:

```bash
EVA_MODEL_LIST='[
  {
    "model_name": "command-r-plus",
    "litellm_params": {
      "model": "cohere/command-r-plus",
      "api_key": "os.environ/COHERE_API_KEY"
    }
  }
]'
```

## Load Balancing

Add multiple entries with the same `model_name` to load-balance across deployments:

```bash
EVA_MODEL_LIST='[
  {
    "model_name": "gpt-5.2",
    "litellm_params": {
      "model": "azure/gpt-5.2",
      "api_key": "os.environ/AZURE_KEY_A",
      "api_base": "https://endpoint-a.openai.azure.com",
      "max_parallel_requests": 5
    }
  },
  {
    "model_name": "gpt-5.2",
    "litellm_params": {
      "model": "azure/gpt-5.2",
      "api_key": "os.environ/AZURE_KEY_B",
      "api_base": "https://endpoint-b.openai.azure.com",
      "max_parallel_requests": 5
    }
  }
]'
```

You can also load-balance across providers:

```bash
EVA_MODEL_LIST='[
  {
    "model_name": "gpt-5.2",
    "litellm_params": {
      "model": "azure/gpt-5.2",
      "api_key": "os.environ/AZURE_API_KEY",
      "api_base": "https://your-resource.openai.azure.com"
    }
  },
  {
    "model_name": "gpt-5.2",
    "litellm_params": {
      "model": "openai/gpt-5.2",
      "api_key": "os.environ/OPENAI_API_KEY"
    }
  }
]'
```

## Concurrency Control

Set `max_parallel_requests` per deployment to limit concurrent API calls:

```bash
EVA_MODEL_LIST='[
  {
    "model_name": "gpt-5.2",
    "litellm_params": {
      "model": "azure/gpt-5.2",
      "api_key": "os.environ/AZURE_API_KEY",
      "api_base": "https://your-resource.openai.azure.com",
      "max_parallel_requests": 5
    }
  }
]'
```

You can also set `rpm` (requests per minute) and `tpm` (tokens per minute) limits.

## Advanced: Model Parameters

Configure model parameters (temperature, max_tokens, reasoning_effort, top_p, frequency_penalty, etc.) in `litellm_params` within `EVA_MODEL_LIST` (see `.env.example`).

## Troubleshooting

### "Authentication Error"
- Check that the correct API key is set in `litellm_params` or as an environment variable
- Verify the API key is valid

### "Model Not Found"
- Verify the `model_name` in your code matches an entry in `EVA_MODEL_LIST`
- Check that `litellm_params.model` uses the correct provider prefix
- See LiteLLM docs: https://docs.litellm.ai/docs/providers

### "Rate Limit Exceeded"
- Adjust `max_parallel_requests` per deployment in `EVA_MODEL_LIST`

## Retry Logic and Rate Limits

The LLM client automatically handles rate limit errors with exponential backoff:

- **Max retries**: 5 (configurable)
- **Backoff strategy**: Exponential with jitter
- **Detects**: HTTP 429 errors, rate limit exceptions
- **Logs**: Warning messages for each retry attempt

This ensures reliable operation even when hitting provider rate limits.

## Reference

Full LiteLLM documentation: https://docs.litellm.ai/

Supported providers: https://docs.litellm.ai/docs/providers

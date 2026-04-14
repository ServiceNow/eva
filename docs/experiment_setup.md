# Experimental Setup

## Hosted Model Configurations

| Model | GPU Type | # GPUs | Precision | Deployment | Inference Params |
|---|---|---|---|---|---|
| gpt-oss-20b | NVIDIA A100 | 2 | BF16 | vLLM | default vLLM |
| gpt-oss-120b | NVIDIA H100 | 2 | BF16 | vLLM | max-logprobs=10 |
| qwen3.5-27b | NVIDIA H100 | 2 | BF16 | vLLM | max-model-len=32768, max-num-seqs=8 |
| parakeet-ctc-1.1b | NVIDIA H100 | 1 | | | |
| voxtral-mini-3b | NVIDIA H100 | 1 | BF16 | vLLM | max-logprobs=10, max-model-len=32768 |
| whisper-large-v3 | NVIDIA A100 | 1 | FP16 | vLLM | max-logprobs=10, do_sample=false + defaults |
| magpie | NVIDIA H100 | 1 | | | |
| kokoro | Tesla V100-SXM2-32GB | 1 | | | |
| chatterbox turbo | NVIDIA A100 | 1 | FP32 | [chatterbox-tts-api](https://github.com/travisvn/chatterbox-tts-api/pull/74) | chunk_size=200, strategy=sentence, quality=balanced |
| ultravox-v0_7-glm-4_6 | NVIDIA H100-80GB-HBM3 | 16 | BF16 | vLLM | max-logprobs=10, max-model-len=16000 |

## LLM Configuration

### Reasoning effort/level for each LLM evaluated:
- **gpt-oss-20b** - default (medium)
- **gpt-oss-120b** - default (medium)
- **qwen3.5-27b** - no thinking 
- **sonnet 4.6** - low thinking 
- **gpt-5-mini** - minimal thinking


### LLM setups in `.env` for evaluated models and judge models:

```json
EVA_MODEL_LIST='[
  {
    "model_name": "gpt-5-mini",
    "litellm_params": {
      "model": "openai/gpt-5-mini",
      "api_key": "",
      "max_parallel_requests": 50,
      "reasoning_effort": "minimal"
    },
    "model_info": {"base_model": "gpt-5-mini"}
  },
  {
    "model_name": "gpt-oss-20b",
    "litellm_params": {
      "model": "openai/<vllm served model name>",
      "api_key": "",
      "api_base": ""
    }
  },
  {
    "model_name": "gpt-oss-120b",
    "litellm_params": {
      "model": "openai/<vllm served model name>",
      "api_key": "",
      "api_base": ""
    }
  },
  {
    "model_name": "qwen35-27B",
    "litellm_params": {
      "model": "openai/<vllm served model name>",
      "api_key": "",
      "api_base": "",
      "temperature": 1.0,
      "top_p": 0.95,
      "top_k": 20,
      "min_p": 0.0,
      "presence_penalty": 1.5,
      "repetition_penalty": 1.0,
      "extra_body": {
        "chat_template_kwargs": {
          "enable_thinking": false
        }
      }
    }
  },
  {
    "model_name": "sonnet-4-6",
    "litellm_params": {
      "model": "bedrock/us.anthropic.claude-sonnet-4-6",
      "aws_access_key_id": "",
      "aws_secret_access_key": "",
      "reasoning_effort": "low"
    }
  },
  {
    "model_name": "gpt-5.2",
    "litellm_params": {
      "model": "openai/gpt-5.2",
      "api_key": ""
    },
    "model_info": {"base_model": "gpt-5.2"}
  },
  {
    "model_name": "gemini-3.1-pro-preview",
    "litellm_params": {
      "model": "vertex_ai/gemini-3.1-pro-preview",
      "vertex_project": "",
      "vertex_location": "",
      "vertex_credentials": "os.environ/GOOGLE_APPLICATION_CREDENTIALS",
      "reasoning_effort": "low"
    }
  },
  {
    "model_name": "us.anthropic.claude-opus-4-6-v1",
    "litellm_params": {
      "model": "bedrock/us.anthropic.claude-opus-4-6-v1",
      "aws_access_key_id": "",
      "aws_secret_access_key": ""
    }
  }
]'
```

## ElevenLabs User Simulator

We created 2 ElevenLabs Agents for the user simulator, one with a female and one with a male voice. When you create a new agent, create a "Blank Agent".
Then, use the following configuration:

| Parameter                            | Value                                                                             |
|--------------------------------------|-----------------------------------------------------------------------------------|
| Voice (female)                       | Natalee Champlin                                                                  |
| Voice (male)                         | Eric - Smooth, Trustworthy                                                        |
| TTS model family                     | V3 Conversational                                                                 |
| Expressive mode                      | Enabled (no tags selected)                                                        |
| Language                             | English                                                                           |
| LLM                                  | GPT-5.1                                                                           |
| System prompt                        | {{prompt}}                                                                        |
| Default personality                  | Disabled                                                                          |
| First message                        | None (remove the default first message, as the agent speaks first)                |
| Interruptible                        | Disabled                                                                          |
| Advanced > Input audio               | μ-law telephony, 8000 Hz                                                          |
| Advanced > Take turn after silence   | 15ms                                                                              |
| Advanced > Max conversation duration | 600s                                                                              |
| Tools > System tools                 | Enable "End conversation" (Name is `end_call`, and Description is provided below) |

The simulator is prompted with a specific user goal and is instructed to stay on task, communicate all required named entities clearly, and terminate when the goal is accomplished or the task is clearly unlikely to succeed.

The ElevenLabs agent also has the end_call tool enabled which it allows it to end the call. The description of the end_call tool, which is provided to the agent, is shown below.

```
Use this to end the phone call and hang up.

Call this function when any ONE of the following is true:
1. The agent has confirmed your request is resolved and you have said goodbye
2. The agent says they are transferring you to a live agent
3. The agent has been unable to make progress for at least 5 consecutive turns
4. The agent says goodbye or indicates the conversation is over
5. The agent indicates that the remainder of your request cannot be fulfilled.
6. If the assistant says something along the lines of "I'm sorry I encountered an error processing your request."

Before calling this tool, always say a brief goodbye first.
```

You can then get your `agent-id` from the Widget tab of your agent.

## Evaluate all scenarios and get overall scores

When running against AIS:

```
GOOGLE_GENAI_USE_VERTEXAI=0 python main.py --debug
```

When running against Vertex:

```
GOOGLE_GENAI_USE_VERTEXAI=1 python main.py --debug
```

Always use `--debug` to test as it executes a single case run only.

# The official vLLM image already contains the engine, CUDA runtime and the OpenAI-compatible
# server. This adds the entrypoint, which turns the task's environment into engine flags, and the
# weight fetcher it runs first.
#
# Pinned rather than :latest. The engine's API surface and flag names change between releases, and
# an unpinned base means a rebuild can silently change behaviour.
FROM vllm/vllm-openai:v0.29.0

COPY serve fetch_weights.py /usr/local/bin/
RUN chmod +x /usr/local/bin/serve /usr/local/bin/fetch_weights.py

# Override the base image's entrypoint, which expects vLLM's own arguments.
ENTRYPOINT ["/usr/local/bin/serve"]

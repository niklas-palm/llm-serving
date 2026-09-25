# The official vLLM image already contains the engine, CUDA runtime and the OpenAI-compatible
# server. This adds only the entrypoint, which turns the task's environment into engine flags.
#
# Pinned rather than :latest. The engine's API surface and flag names change between releases, and
# an unpinned base means a rebuild can silently change behaviour.
FROM vllm/vllm-openai:v0.29.0

COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

# Override the base image's entrypoint, which expects vLLM's own arguments.
ENTRYPOINT ["/usr/local/bin/serve"]

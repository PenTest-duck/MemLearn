# Brainstorm

This file is for rough brainstorming and ramblings...

## Implementation of the underlying FS

### FUSE 

FUSE seems to be the tool for the job here - ideally, all FS operations go through the proxy FUSE layer, which might perform operations like embedding and summarisation, allowing the main agent to run arbitrary Bash commands.

But how to handle when the agent wants/needs to directly imbue files with metadata (e.g. purpose)?

### AgentFS

Turso's AgentFS ticks a lot of the criteria for a base FS - just without the semantic, LLM-tailored features.

https://github.com/tursodatabase/agentfs

### Raw FS + Tools

The alternative is the simple vision of giving the agent a raw FS. It can of course use traditional Bash tools to full extent. You can use tools like `semtools` by LlamaIndex to perform semantic search and parsing. While our philosophy is 'simple is best', this sounds too simple to be true, doesn't it?

### References
  - https://jakobemmerling.de/posts/fuse-is-all-you-need/
  - https://github.com/run-llama/semtools

## Interesting Reads

- How Clawdbot memory works: https://x.com/manthanguptaa/status/2015780646770323543?s=20
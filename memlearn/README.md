# MemLearn

MemLearn is an open-source agentic memory & continual learning system. We adopt
a filesystem-like approach to storing, managing and retrieving memory and
improving agent performance.

## Philosophy

We believe a key challenge for LLM agents is not necessarily cognitive
capability of the foundational model, but lies in memory - the ability to
effectively hold and utilise knowledge and self-improve through continual
learning.

However, we don't believe the solution lies in a form of neural/parametric
memory or reinforcement learning (RL), as they are model-specific and offer low
interpretability. We instead believe that memory architecture should be
free-form - the philosophy here is that ‘the agent knows itself best’. We can
provide various underlying infrastructures so that the agent can achieve this.
Another key is 'simple is best' - we really don't want to over-engineer our
architecture, or be overly prescriptive in our solution. In saying that, our
solution should be extensible enough such that as foundational models get
smarter and smarter, our flexible architecture can still be used with maximal
potential. Hence we aim to be open-source and model-agnostic to easily be
compatible with any foundational LLM or agent framework.

Furthermore, right now, memory and learning seems to be implemented on a
case-by-case basis, i.e. the customer support agent company implements it from
scratch in their own way, so doees the therapist agent company, or the coding
agent company. Also, existing solutions seem to tackle the memory problem on a
base level, or the learning problem on a base level, but not both. We want a
supercharged memory architecture that also provides a toolkit for agentic
learning. The vision for MemLearn is a unified infrastructure layer for agentic
memory and learning that companies can easily plug-and-play into their use case.

## Concepts

MemLearn is based on two fundamental concepts for agents: memory and learning.

### Memory

Simply put, memory is 'what the agent knows and how it knows it'. This can
include insights, observations, user preferences, conversation history,
learnings from past mistakes, external knowledge, real-time information, and
many more.

At the lowest layer, LLMs already come embued with foundational knowledge within
its parameters. So the key challenge is how agents acquire and utilise new
knowledge/information as it interacts with its environment and performs tasks.
There are many types of memory and sources of knowledge.

The LLM's immediate context window can be considered a form of short-term
memory. Initially it may contain a system prompt and tool definitions. As a
conversation goes on or as the agent performs tasks, it fills up with raw
prompts/responses and tool calls. In one sense, it's the ONLY piece of memory
that the LLM has access to as it generates tokens. Hence the entire art of
agentic memory may be distilled down into figuring out what goes in the context
window, and when, and how, to result in optimal performance while balancing
cost/speed/token efficiency. This is also called 'context engineering'.

But the context window gets cleared per agent session (also known as being
'stateless'). This has called for a need for long-term memory (archival memory).
Simply put, this is any form of knowledge that does not immediately live inside
the context window. The challenge for us is how to store/use/manage this memory.
As we saw, there are many use cases and types of memory - keeping in tune with
our 'free-form' philosophy, we want to let the agent itself have a high-degree
of control and freedom with how it architects this memory, and MemLearn provides
the infrastructure layer to fully support these agents.

There are numerous types of memory, such as:

- Raw documents
- Raw conversation histories
- Raw task execution trajectories / logs
- External connections, like Drive, Notion, Gmail, S3 buckets, Postgres DB etc.
- Procedural memory, such as:
  - Markdown files containing anything from simple reminders to cookbooks based
    on tool sequences
  - Executable code snippets
  - Tool schemas
  - OpenAPI / MCP clients

Memory may be scoped to different levels, such as:

- Platform-wide:
  - E.g. hard-coded system prompts
  - Perhaps anonymised, even higher-level abstracted learnings
- Customer tenant / organisation:
  - E.g. abstracted learnings
  - Organisation-specific preferences/memory
- User
  - Useful for persistent conversational memory
  - (Perhaps this concept can even be merged and abstracted with tenant?)
- Agent
  - Giving the agent a stateful entity that lasts across sessions
  - Useful for multi-agent systems, multi-interaction agents (e.g. chatbot)
- Run / session
  - Scratchpad, working memory
  - Though for long time-horizon tasks, this might be really important
- Custom:
  - E.g. memory that is relevant to a specific code repository

### Learning

The concept of agentic learning is closely tied with agentic memory. For us,
learning means improving an agent's performance for a specific task. For
example, this might be a customer support agent that improves its ability to
accurately respond to tickets, or a therapist agent that learns about the user
to provide hyper-personalised advice that improves user satisfaction, or a
coding agent that learns from its mistakes or gets feedback from human
developers to improve its ability to write working code aligned with best
practices.

A typical foundational approach to achieving improvement is reinforcement
learning. However, per our philosophy, we are seeking to achieve learning
without modifying LLM parameters, also known as in-context learning. This
involves engineering the agent's context window to improve its performance. This
notion is very similar to memory management - figuring out what goes into the
context and how it gets there - where learning has a stronger focus on turning
our memory into actionable improvement outcomes, and the whole lifecycle of how
agents can autonomously and continuously advance its learnings.

## The Approach

### MemFS

The technical approach that MemLearn takes is a filesystem-like architecture
that equips LLM agents with memory and self-learning capabilities. It is both an
open-source and managed cloud solution that seeks to make it effortless for the
human developer to empower their agentic capabilities with memory and learning,
as well as providing observability, customisability etc. for the developers to
apply MemLearn effectively for their specific use cases.

At a foundational level, MemLearn works like a file-system that is dynamically
constructed and hot-loaded for each agent run (aka session). Let's call this FS
component MemFS. It should be like a lightweight sandbox where agents can both
read and write, with files and folders. The LLM itself will have tool/s
available in its context window to interact with this FS, and potentially
additional text in its system prompt to provide guidance on how the FS works.
Recall that MemLearn is model-agnostic, and so our overall API/SDK should be
easily applicable to any model or agent framework.

We like the FS approach because it is both interpretable to humans and agents,
and the LLMs are trained on similar folder usage. MemFS should hence support
standard FS features, like:

- Hierarchical structures using folders (directories) and files
- Storing ctime/atime/mtime timestamps for observability
- Permissions (apply access control) and ownership (provenance for agents)
- File and folder sizes
- Absolute and relative path resolution
- Finding or grepping through files to locate relevant content
- Executing code snippets as procedural memory
- Git-like version control for observability & easy restoration

However, LLMs have unique demands that traditional FSs that are targeted for
humans and computers don't. Therefore, our FS should sit on a fabric of
additional features that are crucial for LLMs. These include:

- Storing the embeddings of every file and/or folder to enable semantic search
  over its contents (perhaps folders can provide a HNSW-like clustering)
- Storing metadata for each file (e.g. summary, tags) for rapid search and
  briefly checking a long file's contents without clogging up the context window
- LLM-friendly out-of-the-box tools for managing the FS, reading files (e.g.
  with line numbers), editing files (e.g. applying diffs), searching (e.g. a
  simple tool that offers an abstracted hybrid search and/or reranking over the
  FS based on keyword/semantic/metadata) etc.

Our underlying ephemeral FS itself should have fast init, fast mounting, strict
isolation etc.

Remember that memory can have multiple scopes. At initialisation time, all
memory within the relevant scope of the agent run should be dynamically mounted
onto the FS.

We also consider facilitating multi-agent collaboration - this should also be
like a mounted FS (similar to NFS) where multiple agents can concurrently read
and write to it, either during runs at different times, or at the same time. It
acts like a shared workbench that facilitates cross-agent knowledge sharing,
hand offs, and even IPC-like communication.

### Short-term Memory

While MemFS and its capability for long-term persistent memory handles the bulk
of memory use cases, we believe there is still value in providing developers
with an easy way to handle short-term memory as well. Short-term memory may also
be viewed as optimising the context window token space. This is especially
relevant for basic tasks or conversational agents which may benefit from a
super-fast and token-efficient method to utilise short-term memory.

Some particular points of interest include:

- Compaction/summarisation for long conversation history
- Clearing unneeded tool calls and results
- Potentially dynamically mounting and unmounting tools (though this may
  adversely influence prompt caching)

### Memory Engine

The memory engine is how our self-evolving memory system manages itself.
However, we have to be really careful here as our core philosophy is 'simple is
best' - we don't want to over-engineer this, and perhaps we don't even need a
memory engine and let the agent self-regulate its memory while it's operating
tasks.

The main focus of the memory engine is how the MemFS continues to evolve,
whether through consolidation, re-organisation, extrapolation,
archival/deletion/pruning etc. It also includes how the memory system may
respond when the human developer manually reaches in to add/edit metadata fields
which may need to trigger processing of all existing memory.

### Learning Engine

The learning engine facilitates our non-RL agentic learning process - namely,
how to learn and how to apply the learnings. Perhaps there is a simple human
guidance prompt that tells the agent what to learn, what not to learn etc. (but
we need to keep in mind that human prompts are generally going to be vague, so
the engine needs to provide high performance even on low guidance).

Let's first look at 'how to learn'. The key here is: given raw trajectories and
its corresponding result, how do we produce abstract, generalised insights
and/or situation-specific learnings that can improve performance later? One of
the most straightforward ways of learning is allowing the developer to configure
an agent to call some reflect tool at the end of a task run, which positions the
agent to look back over its trajectory and produce a condensed insight that can
be stored in our MemFS. After many task runs, we may or may not need to have a
background agent run to consolidate and update the memory of learnings.

Let's also look at 'how to apply the learnings'. A way is instruction tuning,
which can modify or append to the system prompt (e.g. using a prompt optimiser
like Langchain or even just a static text blob the developer can concatenate to
their prompt) - because this takes up precious space in our context window, we
potentially only want core learnings here, e.g. only the most critical issues,
most relevant situations, or most repetitive mistakes. Another way is to store
these learnings in MemFS and conditionally load them in per standard MemFS
operations (perhaps with learning-specific operations).

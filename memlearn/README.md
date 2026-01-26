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

Continual learning is also called online learning, test-time training,
in-context learning etc.

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

## Technical Architecture

### MemFS Folder Structure

```
/: This is the root directory of the FS.
  -> AGENTS.md: This comprehensive markdown file contains all information/instructions on how to use this FS.
  -> /raw: This special folder keeps a store of raw conversation histories and artifacts. It is read-only for the agent. It can only be written to by the system / human developer using the SDK.
    -> /conversation-history: This folder stores all past conversation history.
      -> /CURRENT: This is the current running conversation history. When the session starts, this is the initial conversation history (e.g. system prompt, tool definitions). When the session ends, this is renamed per our timestamp format.
      -> /YYYY-MM-DDThh:mm:ss: Files are named as the timestamp of when the conversation STARTED.
    -> /artifacts: This folder stores any files that the human developer uploads manually through the SDK.
      -> e.g. documents, images, videos, audio, folders, etc.
    -> /skills: This folder stores skills that are loaded in by the human developer
  -> /memory: This is the main folder of MemFS for long-term persistent memory - the agent has full control/freedom over how it architects The agent has rwx access. This memory persists even after the agent session finishes.
    -> e.g. insights/observations that should persist across time, generalisable learnings, understanding of user preferences / agent environment
    -> this folder is a free-form space that the agent has full control to manipulate and evolve that encompasses all of semantic, episodic, procedural memory
  -> /tmp: This folder is for short-term memory that should not be committed into long-term memory. Be aware that it is going to be cleared at the end of the agent run / session. Agents have full rwx access.
    -> e.g. temporary plans, to-do lists, scratchpad etc.
  -> /mnt: This folder contains additionally mounted folders, typically in a shared scope. The agent has read and write access, but the agent must be aware that this folder is potentially shared across multiple agents and consider the implications. The direct children of /mnt are always mounted folders (and no files can be a direct child).
    -> e.g. /mnt/agents/<agent-id>: the core memory of the current agent is going to be in /memory, so this folder might access the memory of a different agent for multi-agent system scenarios
    -> e.g. /mnt/users/<user-id>: this might be memory that is pertinent to a specific user, such as for multiple conversational agents interacting with the same user, or a single agent interacting with the user multiple times
    -> e.g. /mnt/organizations/<org-id>: this might be shared memory that applies to a whole customer organisation
```

The agent cannot create/modify/delete in the root directory. All root-level
directories and mandatory directories are initialised to empty by default.

The agent may execute certain executable files, e.g. code snippets.

#### Mountable Folders

The /mnt folder is special as it is the keystone to enabling shared memory
across multiple agents. Typically the human developer will specify which folders
are mounted upon FS spin-up at the start of an agent session - generally, it
won't be the agent's responsibility to create new mountable folders.

We will have to think about this more, but if we decide that agents should have
the ability to create new mountable folders, and mount/unmount on its own, then
the metadata description of the /mnt folder should contain information about
mountable unmounted folders, and the agent should have appropriate tooling to
create/mount/unmount folders.

### MemFS Operations

MemFS should support operations from basic file system manipulation to
LLM-friendly tools tailored for memory and learning purposes.

- File operations:
  - Read file/s: we need to make LLM-friendly considerations, such as displaying
    the line numbers, truncating & paginating its contents to not overwhelm the
    context window, specifying line numbers to show etc. We should also think
    about what it means for an agent to read a PDF file, or an image, or a
    video. In each multimodal case, a corresponding text representation may be
    stored in the metadata and displayed, like the parsed PDF text, or a VLM's
    description of the image/video
  - Create file/s: with our without initial text content. The corresponding
    metadata should be generated under the hood.
  - Edit file/s: we need to be LLM-optimised here. For now, let's use exact
    string replacement. We can experiment later with diff patch-style editing.
    The corresponding metadata should be modified under the hood (e.g.
    identifying modified chunks and re-embedding them).
- Folder operations:
  - Create folder/s: equivalent to `mkdir -p`. The corresponding metadata should
    be generated under the hood.
  - List: an LLM-friendly variant of `ls` that shows useful metadata like
    description, number of files in each folder. In traditional file systems,
    LLMs can often waste lots of turns on just navigating the FS - so perhaps we
    may consider showing truncated lists of nested folders by default. We need
    to balance not taking up too many tokens with being informative.
- Other basic operations:
  - Move/rename path/s: equivalent to `mv` or `mv -r`
  - Delete path/s: equivalent to `rm` or `rm -r`
  - Run arbitrary CLI commands (read-only for now): e.g. grep, sed, awk, find
  - Executing files: TODO
- MemFS operations:
  - Peek path/s: show important metadata in an LLM-friendly format, e.g.
    description, tags, line length / size / token count
  - Search (from base path): allow keyword (like grep), semantic, lexical
    (BM25+), metadata-based, or hybrid search that returns chunks with
    information like score, path etc. A lot of the underlying inner-workings of
    this tool should be abstracted away from the agent, e.g. applying multiple
    strategies, query reconstruction, reranking etc.
  - Undo: reverts the last operation, using the power of our version-controlled
    FS. Optionally specify how many operations to undo (default 1).
- Underlying automatic operations:
  - Logging each MemFS operation
  - Maintaining version control (probably exclude /raw)
  - Generating and keeping metadata up to date
  - Parsing (if necessary) and embedding any files uploaded into the /raw
  - Continuously storing the raw conversation history of the agent inside
    /raw/conversation-history

### MemFS Metadata Structure

```json
{
  "type": "object",
  "description": "The metadata for a specific node (i.e. file or folder) in MemFS.",
  "properties": {
    "timestamps": {
      "type": "object",
      "properties": {
        "created_at": {
          "type": "integer",
          "description": "Creation time (unix)"
        },
        "accessed_at": {
          "type": "string",
          "description": "Last access/read time (unix)"
        },
        "modified_at": {
          "type": "string",
          "description": "Last modification time (unix)"
        }
      },
      "required": ["created_at", "accessed_at", "modified_at"]
    },
    "permissions": {
      "type": "object",
      "properties": {
        "readable": {
          "type": "boolean",
          "description": "Whether this node is readable by the agent"
        },
        "writable": {
          "type": "boolean",
          "description": "Whether this node is writable by the agent"
        },
        "executable": {
          "type": "boolean",
          "description": "Whether this node is executable by the agent"
        }
      },
      "required": ["readable", "writable", "executable"]
    },
    "owner": {
      "type": "string",
      "description": "The ID of the agent that owns the node, or 'system' if it was manually created. By default, this is the entity that created the node."
    },
    "type": {
      "type": "string",
      "description": "The type of this node, e.g. folder, document, image, video"
    },
    "description": {
      "type": "string",
      "description": "A human-readable and LLM-friendly description of this file/folder. It is the responsibility of the agent to provide a detailed description and summary of its contents upon creation."
    },
    "tags": {
      "type": "array",
      "description": "A list of 0 or more tags the agent can provide to enable it to more easily filter for folders/files.",
      "items": {
        "type": "string"
      }
    },
    "embedding": {
      "type": "array",
      "description": "The vector embedding of the description of this file or folder. This allows semantic search.",
      "items": {
        "type": "float"
      }
    },
    "chunks": {
      "type": "array",
      "description": "A list of 0 or more chunks in the file content. Does not apply to folders.",
      "items": {
        "type": "object",
        "properties": {
          "embedding": {
            "type": "array",
            "items": {
              "type": "float"
            }
          },
          "start": {
            "type": "integer"
          },
          "end": {
            "type": "integer"
          },
          "hash": {
            "type": "string",
            "description": "A hash of the chunk contents"
          }
        },
        "required": ["embedding", "start", "end", "hash"]
      }
    },
    "extra": {
      "type": "object",
      "description": "Any extra metadata fields for this file/folder."
    }
  },
  "required": [
    "timestamps",
    "permissions",
    "owner",
    "type",
    "description",
    "embedding"
  ]
}
```

This metadata is stored in a separate relational DB and must be synchronised
with the FS. This can get tricky as the agent can execute scripts that modify
the FS. We need to think about how we maintain 1-to-1 consistency.

This metadata is supposed to support a variety of search/retrieval strategies,
including metadata keyword search, file content semantic search, hybrid search,
etc.

### MemFS Log Structure

```json
{
  "type": "object",
  "properties": {
    "timestamp": {
      "type": "string"
    },
    "message": {
      "type": "string"
    },
    "agent": {
      "type": "string",
      "description": "The ID of the agent, otherwise 'system'."
    }
  },
  "required": ["timestamp", "message", "agent"]
}
```

Every MemFS operation is logged for observability and traceability.

MemFS should have Git-like version control system that allows humans to navigate
the diffs for observability. It also allows agents to revert changes easily,
such as an undo tool to revert changes before the last operation was performed.
This can even cover scenarios where a script was executed that modified the FS.

### Short-Term Memory

Our MemLearn SDK should provide a compaction tool for task-running agents to use
when their context window gets close to full. This compaction tool should make
the agent reflect on its entire context window, summarise/distill it down to the
core necessary amount, while the raw conversation history is still preserved in
our MemFS. The key here is to balance pruning of unneeded context with keeping
crucial specifics that improve agent performance.

### Learning Engine

TODO on how we can best support agents with their continual learning and
self-improvement. Ideas include reflection/introspection, building internal
mental models etc.

### MemLearn Config Structure

MemFS config parameters include, but not limited to:

- Infrastructure stack (for our open-source use case):
  - Databases (e.g. Sqlite, Postgres)
  - Vector stores (e.g. Qdrant, Chroma)
  - LLMs (e.g. OpenAI, Anthropic, Gemini)
  - Embedding models (e.g. OpenAI, Voyage AI)
    - Chunking strategies
    - Embedding dimensions
  - Rerankers (e.g. Cohere)
  - Sandboxes (e.g. local, E2B)
- Target agent framework / tool structure (e.g. OpenAI, Anthropic, Langchain)
- FS settings:
  - Read file tool truncation, toggling line number display
  - Edit file tool method of editing, e.g. str_replace, udiff

### Developer APIs

At times, for certain tasks, the human developer may want to directly surface
memory or metadata, or manually modify things. For this, we should expose a wide
variety of useful APIs for the developer to achieve this.

Examples of this can include:

- Adding files to the read-only section of the FS
- Reverting the FS to a certain version
- Fetching data stored in the FS

For querying purposes, a developer might want to attach an 'agent head' to a
MemFS instance, where the entire FS is mounted in read-only mode such that this
dedicated memory agent can answer questions based on the persistent memory that
the main task agent has created.

### Further Considerations

All error messages / exceptions must be LLM-friendly i.e. explains in plain
words on what the specific issue is, and potentially how to resolve it if
relevant.

MemFS should be highly performant, securely isolated, and most of all, very
friendly for the LLM agents to use.

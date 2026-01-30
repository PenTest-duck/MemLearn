# Brainstorm

This file is for rough brainstorming and ramblings...

## Implementation of the underlying FS

### FUSE 

FUSE seems to be the tool for the job here - ideally, all FS operations go through the proxy FUSE layer, which might perform operations like embedding and summarisation, allowing the main agent to run arbitrary Bash commands.

But how to handle when the agent wants/needs to directly imbue files with metadata (e.g. purpose)?

### Watchfiles

Watch file changes and update metadata.
Supposedly even better performance!?

See: https://www.nijho.lt/post/file-based-rag-memory/

### AgentFS

Turso's AgentFS ticks a lot of the criteria for a base FS - just without the semantic, LLM-tailored features.

https://github.com/tursodatabase/agentfs

### Raw FS + Tools

The alternative is the simple vision of giving the agent a raw FS. It can of course use traditional Bash tools to full extent. You can use tools like `semtools` by LlamaIndex to perform semantic search and parsing. While our philosophy is 'simple is best', this sounds too simple to be true, doesn't it?

### References
  - https://jakobemmerling.de/posts/fuse-is-all-you-need/
  - https://github.com/run-llama/semtools


## MemAgent: Dedicated Agent Approach

On top of the underlying MemFS system, we could have a dedicated MemFS agent (let's call it MemAgent) that is in charge of everything to do with memory. The main task agent would focus on solving the tasks, and would have a single tool call to interact with the MemAgent for retrieving relevant memory with natural language.

The MemAgent would 'eavesdrop' on the rolling conversation history of the task agent. This can either be through a continuous harness, or beat discrete times like the end of a conversation or upon compaction - i.e. whenever some or all parts of the conversation history leaves the immediate context window of the task agent. There could even be two agents - MemAgent-R (Retrieval) for recollection of memory, MemAgent-S (Synthesis) for storage and maintenance of memory. This allows each agent to specialise and focus on what they each do best: the task agent solving the task, and MemAgent providing great memory recollection and synthesis.

Well actually, it might be that MemAgent-R is also subdivided into two agents. There is the one that is triggered by the task agent, but as it searches through MemFS and potentially notices memory that needs re-organising or outdated/misleading memory that needs pruning or other memory upkeep, it may call up the MemAgent-S to asynchronously perform this in the background (or if it's critical to the retrieval process, it may synchronously call it). This then further separates the duties of MemAgent-R so that the retrieval portion can focus on retrieving and providing the best response to the task agent, and then the async MemAgent-S can focus on the maintenance (potentially in the background) without affecting latency.

The downside is that the task agent needs to communicate well with MemAgent and not knowing what memory is stored or how the underlying memory is structured may be a fog of war that clouds judgement for the task agent. MemAgent needs to do the tough balancing act of giving sufficiently rich and specific context while not clogging up token space. It also needs to understand the intent of the task agent well.

Eventually, there can be a fine-tuned LLM model to be used as the base of the MemAgent, but also allow the option for the developer to choose any model (probably the same one as their task agent) and provide an instruction-tuned system prompt for it.

## MemFS Utilisation Problem

```
So I've been running our MemLearn system through some manual evaluation, and I've been noticing a trend. 

It's that the agent often doesn't make full usage of the memory scatteredly stored in the MemFS. For example, I gave the agent in the first session a large conversation between two people in a single context window and told it to use memfs to remember it. It created a summary.md in /memory, and then during spindown, the /raw/conversation-history was also stored. 

Now in the second session where I quizzed the agent on its memory, it only found the /memory summary.md and used only it to try and answer questions (which it was not very successful), until i told it that it should also check the conversation-history.

I'm worried that this problem is generalizable to the agent not being able to locate certain memory or not knowing certain memory exists. Also on even new agent session, while our system prompt tells it the structure of the memory, the agent has pretty much no knowledge of what is actually currently stored so far or how the memory has been used (or not used) so far.

How can we resolve this problem?
```

A consideration is how much we want to delegate to the agent, and how much we want to delegate to MemFS. Because we want MemLearn to be as portable as possible, I'm leaning towards delegating to MemFS, in particular, its `spinup` and `spindown` phases.

We can have dynamic system prompts, like how Letta has memory blocks, but that may not be ergonomic for the human developer.

### Solution: Dynamic Memory Notes (IMPLEMENTED)

We implemented a "dynamic memory note" approach that separates:
- **Form**: The structural architecture of memory (`/memory/`, `/raw/`, etc.) - documented in the static system prompt
- **Dynamics**: What's actually stored and how it's organized - captured in a dynamic note

The memory note is:
1. Stored as a field on the Agent entity in the database
2. Initially a short sentence saying memory is empty
3. Auto-updated on `spindown()` by an LLM that reviews the existing note + filesystem listing
4. Retrieved via `memfs.get_memory_note()` for injection into system prompts
5. Configurable max tokens (default 256) to respect context window budget

This is lightweight (no extra agent, no complex protocol) and developer-friendly (just call one method and inject into system prompt). Future enhancement: diff-based updates using git-like version control.

See `get_memfs_system_prompt_with_note()` in `prompts.py` for usage.

## MemFS Graph

Can we create graph edges (links and backlinks) between files? E.g. `/memory/SUMMARY.md` -summary_of-> `/raw/conversation-history/xxxx.md`

Useful for searching, e.g. `ls`, in showing related files. Like how our brain thinks of associated memory.

The question is whether to do this on a file-by-file basis, or chunk-by-chunk basis, or both. And the key challenge is keeping this sync'd and updated over time.

## Agents as Stateful Entities

Let's add a system_prompt field to the Agent entity, which is an optional string that if provided, can serve as the system prompt for the agent. 

We are going to consolidate the notion that an agent is a stateful entity. 

That means we are going

Update the types / functions, and demo code as necessary.

## Temporal Memory

Quite a lot of questions in LoCoMo benchmark is asking the agent to recall and reason temporally, e.g. "when was ...?", "when did ...?". And need to calculate dates like "last Friday", "next week" etc.

Although I am taking LoCoMo and traditional memory benchmarks with a grain of salt because it's quite non-realistic and very straightforward conversation-based.

## Folders as RAG

If MemFS really works well, you could just store files inside it and it becomes RAG.

## Agent Learning

Learning = memory consolidation.

Bloom's Taxonomy of human learning:
1. **Remember**: Exhibit memory of previously learned material by recalling facts, concepts, and answers
2. **Understand**: Demonstrate an understanding of the facts by explaining ideas or concepts
3. **Apply**: Use existing knowledge to solve new problems or apply acquired knowledge in new situations
4. **Analyze**: Examine and break information into parts to explore relationships
5. **Evaluate**: Defend opinions and decisions; justify a course of action by making judgements about information
6. **Create**: Generate new ideas and products or compile information in a new way

- Level 1: recall facts
- Level 2: master all it has seen
- Level 3: generalise to new scenarios
- Level 4: innovate

## Better Benchmarks

It seems like the benchmarks for LoCoMo, LongEval, LongBench v2 are all either single lumps of text or conversations between users. I expected to see benchmarks that either I inject as conversation history into the context window (i.e. agent-tool-user interactions) or a sequence of user prompts and tool schemas to test the learning capability of the agent over time.

MemoryAgentBench: http://arxiv.org/pdf/2507.05257

## Showerthoughts

- Maybe tool outputs need to provide suggestions of which tools to potentially call next 
- Internal models / mental models for learning engine
- Should probably leave like a NOTE.md (or even system prompt) for the next agent to immediately read and understand the architecture of the memory
- Separate `description` and `summary` metadata fields

## Scratchpad

You are going to be given a conversation between Caroline and Melanie that occurred over multiple sessions at different datetimes. Your job is to read, learn, remember this conversation using your MemFS. Because afterwards, I'm going to end this session and start a fresh new session with the same persistent MemFS and ask questions to test your knowledge and understanding of the conversation between Caroline and Melanie. Do you understand, and are you ready to receive the (quite large) conversation? (it will be in json format)

In a previous session, I showed you a conversation that occurred over multiple sessions (at different times) between Caroline and Melanie. Now I am going to test your memory and understanding of that conversation by asking you a series of questions. You should respond with your answer, then a brief justification / evidence of how you got that answer. Are you ready?

## Interesting Reads

- How Clawdbot memory works: https://x.com/manthanguptaa/status/2015780646770323543?s=20
- Watchfile-based RAG & Memory: https://www.nijho.lt/post/file-based-rag-memory/
- VexFS - AI-native Semantic Filesystem: https://github.com/lspecian/vexfs
- General Intelligence Company - Memory is the Last Problem to Solve to Reach AGI: https://www.generalintelligencecompany.com/writing/memory-is-the-last-problem-to-solve-to-reach-agi
- General Intelligence Company's Cofounder: https://www.generalintelligencecompany.com/writing/introducing-cofounder-our-state-of-the-art-memory-system-in-an-agent
- Letta's Sleep Time Compute - https://arxiv.org/pdf/2504.13171
import asyncio
from agentfs_sdk import AgentFS, AgentFSOptions


async def main():
    # Open an agent filesystem
    agent = await AgentFS.open(AgentFSOptions(id="my-agent"))

    # Use key-value store
    await agent.kv.set("config", {"debug": True})
    config = await agent.kv.get("config")
    print(config)

    # Use filesystem
    await agent.fs.write_file("/notes.txt", "Hello, AgentFS!")
    content = await agent.fs.read_file("/notes.txt")
    print(content)

    # Track tool calls
    call_id = await agent.tools.start("search", {"query": "Python"})
    await agent.tools.success(call_id, {"results": ["result1"]})

    await agent.close()


asyncio.run(main())

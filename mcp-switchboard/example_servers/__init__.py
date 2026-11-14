"""Three small, keyless MCP servers to plug into the switchboard.

    sandbox_files  read/write files, but only inside ./sandbox   (stdio)
    calc_clock     safe arithmetic, unit-free maths, time zones   (stdio)
    web_reader     fetch a page as text, keyless web + Wikipedia search
                   (stdio by default, or Streamable HTTP with --http)

Each one is an ordinary MCP server: it knows nothing about the switchboard
and would work unchanged in Claude Desktop, Claude Code or any other client.
"""

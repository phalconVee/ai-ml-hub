# Notes on context engineering (sample document)

Context engineering is the practice of deciding what information goes into a
language model's context window for each request. Prompt engineering is about
how you phrase an instruction; context engineering is about which facts,
documents, tool results and memories sit next to it.

A larger context window does not remove the problem. Long contexts cost more,
run slower, and models tend to use information at the start and end of a long
context more reliably than information buried in the middle, an effect often
called "lost in the middle". Irrelevant passages can also distract the model
and pull its answer off-topic.

Common techniques: retrieve candidates from several sources, re-rank them for
relevance, remove near-duplicates, cap how much any single source contributes,
and pack the best items into a fixed token budget. Conversation memory should
be retrieved selectively rather than replayed in full.

Grounded answers should cite the passage behind every claim, and a system
should verify that each citation points at a passage that was actually in the
context window.

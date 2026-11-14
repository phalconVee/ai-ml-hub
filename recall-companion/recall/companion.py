"""One chat turn, start to finish: recall → answer → remember.

    user message
        │
        ├─ 1 save it (short-term memory: this session's transcript)
        ├─ 2 plan the recall      (recall_planner.py, small Claude call)
        ├─ 3 retrieve facts       (retriever.py, local, no LLM)
        ├─ 4 answer               (Claude, with memories in the system prompt)
        ├─ 5 log which facts were used for this answer (the inspector shows it)
        └─ 6 remember             (extractor.py, small Claude call → store)

Steps 4 and 6 are separate public functions so the UI can show the answer
first and run the memory step right after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .extractor import extract_and_store
from .llm import ask, text_of
from .recall_planner import RecallPlan, plain_plan, plan_recall
from .retriever import Recalled, retrieve
from .store import Fact, MemoryStore, WriteResult

SYSTEM = """You are a warm, concise personal assistant with long-term memory of this user.

Below are MEMORIES retrieved from earlier conversations (possibly from other sessions).
Use them naturally, the way a friend who remembers would: don't recite them, don't say
"according to my memory", and only bring up the ones that help with the current message.
Respect hard constraints (allergies, diets, dislikes) without being asked.
Memories marked NO LONGER TRUE describe the past; never treat them as current.
If a memory conflicts with what the user says now, trust the user and go with the new information.
If you don't know something about the user, say so rather than guessing.

USER'S NAME: {name}
TODAY: {today}

MEMORIES:
{memories}"""


def render_memory(f: Fact) -> str:
    """One memory line for the prompt, with its time context."""
    when = f"since {f.valid_from[:10]}"
    if f.occurred_at:
        when += f", happens/happened {f.occurred_at}"
    if not f.active:
        when = f"NO LONGER TRUE since {(f.valid_to or '')[:10]}, was true from {f.valid_from[:10]}"
    return f"- [M{f.id}] {f.statement} ({when})"


def build_system(name: str, today: str, recalled: list[Recalled]) -> str:
    memories = "\n".join(render_memory(r.fact) for r in recalled) or "(nothing relevant remembered yet)"
    return SYSTEM.format(name=name, today=today, memories=memories)


def as_api_messages(history: list[dict], new_text: str) -> list[dict]:
    """Transcript → Messages API format. The API wants alternating roles that
    start with "user", so we drop leading assistant turns and merge runs of the
    same role (which can happen if a previous answer failed half-way)."""
    out: list[dict] = []
    for m in [*history, {"role": "user", "content": new_text}]:
        if not out and m["role"] != "user":
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] += "\n\n" + m["content"]
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


@dataclass
class Turn:
    """Everything one turn produced; the UI renders all of it."""
    answer: str
    plan: RecallPlan
    recalled: list[Recalled]
    user_message_id: int
    assistant_message_id: int
    writes: list[WriteResult] = field(default_factory=list)


def answer(store: MemoryStore, client: Any, profile_id: int, session_id: int, text: str,
           *, smart_recall: bool = True) -> Turn:
    """Steps 1-5: save the message, recall, answer, log the recall."""
    profile_name = dict(store.profiles())[profile_id]
    history = store.messages(session_id, last=SETTINGS.recent_messages)   # before saving the new one
    user_id = store.add_message(session_id, "user", text)                  # 1

    plan = plan_recall(client, text, history) if smart_recall else plain_plan(text)   # 2
    recalled = retrieve(store, profile_id, plan.query, entity_hints=plan.entities,     # 3
                        include_history=plan.include_history)

    system = build_system(profile_name, store.clock().date().isoformat(), recalled)
    response = ask(client, system=system, effort=SETTINGS.effort_answer,               # 4
                   messages=as_api_messages(history, text))
    if getattr(response, "stop_reason", "") == "refusal":
        reply = "I can't help with that one, but I'm happy to talk about something else."
    else:
        reply = text_of(response).strip() or "(no answer)"

    assistant_id = store.add_message(session_id, "assistant", reply)
    store.record_recall(assistant_id, [(r.fact.id, r.score) for r in recalled])        # 5
    return Turn(reply, plan, recalled, user_id, assistant_id)


def remember(store: MemoryStore, client: Any, profile_id: int, session_id: int, turn: Turn,
             user_text: str) -> Turn:
    """Step 6: extract durable facts from the exchange and store them."""
    turn.writes = extract_and_store(store, client, profile_id, user_text=user_text,
                                    reply=turn.answer, session_id=session_id,
                                    message_id=turn.user_message_id)
    return turn


def chat_turn(store: MemoryStore, client: Any, profile_id: int, session_id: int, text: str,
              *, smart_recall: bool = True) -> Turn:
    """The whole loop in one call (used by the tests and by scripts)."""
    turn = answer(store, client, profile_id, session_id, text, smart_recall=smart_recall)
    return remember(store, client, profile_id, session_id, turn, text)

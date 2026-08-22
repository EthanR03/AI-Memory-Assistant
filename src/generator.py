"""Milestone 4 - Route: answer from notes, from the NFL store, or both.

This was a fixed pipeline: retrieve four chunks, hand them to the model,
tell it to answer from them alone. That is right for notes and wrong for
the predictor, whose data is 7,500 structured records - the shape RAG is
worst at, as the Fact Book proved when 883 pages of tables went in as
800-character chunks and came back unusable.

So the model now chooses. Prose questions go to the vector store;
football questions go to SQL. One assistant, two backends - which is the
point: "what did I decide about Elo, and what does the data say now" is
one question, and answering it needs both.

    python -m src.ask "which club has the best record since 2016?"

The loop is the standard tool-calling shape: ask the model, run whatever
tools it asks for, feed the results back, repeat until it answers in
prose or hits MAX_STEPS. Tool errors are returned to the model rather
than raised, because a bad query is usually one the model can fix once
it sees why it failed.
"""
import json
from datetime import date

from . import tools
from .embedder import _get_client  # reuse the same OpenAI client
from .retriever import retrieve

# Routing plus correct SQL against a real schema is a lot more than
# answering from four supplied chunks, and this is where the design fails
# if it fails. Worth more than the mini model the fixed pipeline used.
CHAT_MODEL = "gpt-4o"

# Enough for a notes lookup, a query, and a retry after a SQL error.
MAX_STEPS = 6

NOTES_TOOL = {
    "name": "search_notes",
    "description": (
        "Semantic search over the user's own written notes. Use for what "
        "THEY recorded, decided or thought - reasoning, plans, opinions. "
        "Not a source of NFL facts; use query_nfl_db for those."),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "What to look for, in natural language."},
            "n_results": {"type": "integer",
                          "description": "How many chunks (default 4)."},
        },
        "required": ["query"],
    },
}


def _openai_tool(spec: dict) -> dict:
    """Map a provider-neutral spec onto OpenAI's function-tool shape."""
    return {"type": "function", "function": {
        "name": spec["name"],
        "description": spec["description"],
        "parameters": spec["input_schema"],
    }}


TOOLS = [_openai_tool(NOTES_TOOL), _openai_tool(tools.TOOL_SPEC)]

SYSTEM_PROMPT = """\
You are a personal assistant with two sources, and picking the right one
is most of your job.

- search_notes: the user's own notes. Their thinking, decisions, plans.
- query_nfl_db: a SQLite store of NFL data, 1999-2026. Every game, score,
  closing line, weather reading and model forecast, plus bios for every
  player since 1974.

Rules:

1. NEVER state a number from memory that a query could compute. "Which
   club has the best record since 2016" is a SELECT, not a recollection.
   If you catch yourself recalling a statistic, query it instead.
2. Use both tools when a question spans both, and say which came from
   which. Cite the note's source file, or say the figure came from the
   store.
3. If a query errors, read the message and try a corrected query. The
   schema notes on query_nfl_db describe the traps.
4. If the store lacks part of what was asked, give what you DO have and
   name what is missing. Do not offer to look something up - look it up,
   then answer. "Who is X" is answerable in two queries: the bio from
   `players` and the win-loss record derived from `games`. Do both.
   What is still absent is per-game statistics - no passing, rushing or
   receiving numbers, no rosters, no injuries - so say that plainly and
   never stop at the caveat.
5. For notes questions where retrieval finds nothing relevant, say you
   have no memory of it. Do not fall back on general knowledge and
   present it as the user's note.
6. Model forecasts in the `predictions` table are the model's OPINION.
   This model does not beat the market - 64.6% against the closing
   spread's 66.4% over 2010-2025, with no filter clearing break-even by
   more than noise. When you report a pick, say so. Never present one as
   profitable betting advice.
7. Never call a player currently active on the strength of
   players.status; it reads 'ACT' for men who retired decades ago. Judge
   by last_season against the current season.

Be concise. Answer the question that was asked.
"""


# --- Tool implementations -------------------------------------------------

def _run_search_notes(args: dict, sources: list) -> str:
    hits = retrieve(args["query"], n_results=args.get("n_results", 4))
    if not hits:
        return "(no matching notes)"
    sources.extend(hits)
    return "\n\n".join(
        f"[source: {h['source']}, chunk {h['chunk_index']}]\n{h['text']}"
        for h in hits)


def _run_query_nfl_db(args: dict, sources: list) -> str:
    sql = args.get("sql", "")
    try:
        result = tools.query(sql)
    except tools.QueryError as exc:
        # Handed back as text, not raised: the model can usually fix a
        # query once it can read why the last one failed.
        return f"ERROR: {exc}"
    sources.append({"source": "nfl.db", "sql": sql,
                    "chunk_index": result["row_count"]})
    return tools.format_result(result)


DISPATCH = {"search_notes": _run_search_notes,
            "query_nfl_db": _run_query_nfl_db}


# --- The loop -------------------------------------------------------------

def answer(question: str, max_steps: int = MAX_STEPS) -> dict:
    """Answer a question, using whichever tools the model decides it needs.

    Returns {"answer": str, "sources": [...], "steps": int} - `sources`
    holds note chunks and executed queries in the order they were used,
    so a caller can show its work.
    """
    client = _get_client()
    # The model cannot reason about "tomorrow" or "this season" without
    # knowing when now is.
    messages = [
        {"role": "system",
         "content": f"{SYSTEM_PROMPT}\nToday's date is {date.today()}."},
        {"role": "user", "content": question},
    ]
    sources: list[dict] = []

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOLS,
            # Deterministic on purpose. At the default temperature the
            # model writes a different query each time and the SAME
            # question returned 137 wins on one run and 135 on the next.
            # For SQL there is no upside to sampling: one of those was
            # simply wrong, and a wrong number is worse here than a dull
            # one.
            temperature=0,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return {"answer": message.content or "",
                    "sources": sources, "steps": step}

        for call in message.tool_calls:
            handler = DISPATCH.get(call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            content = (handler(args, sources) if handler
                       else f"ERROR: no tool named {call.function.name}")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": content})

    # Out of steps with no prose answer - report it rather than looping,
    # so a question the model cannot resolve fails visibly.
    return {
        "answer": f"(gave up after {max_steps} tool calls without an answer)",
        "sources": sources, "steps": max_steps,
    }

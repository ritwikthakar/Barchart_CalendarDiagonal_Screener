"""
chatbot.py - drop-in chat assistant for a Streamlit screener app.

Usage in app.py:
    from chatbot import render_chatbot
    ...
    # after your screener produces a results DataFrame `df`:
    with st.expander("Ask the screener assistant", expanded=False):
        render_chatbot(df, extra_context="Calendar/diagonal spreads from Barchart CSV")
"""

import os

import pandas as pd
import streamlit as st
from anthropic import Anthropic

MODEL = "claude-sonnet-5-5"
MAX_ROWS = 60          # rows of the results table sent to the model
MAX_HISTORY = 20       # most recent chat messages sent each turn

SYSTEM_PROMPT = """You are an options-trading screener assistant embedded in a Streamlit app.
The user has uploaded Barchart calendar/diagonal spread data and the app has produced a
results table (provided below as CSV). Answer questions about that table: explain columns,
compare candidates, flag risks (earnings dates, low liquidity, wide bid/ask, short DTE),
and suggest how to filter or rank. Only cite numbers that appear in the table. If the data
needed isn't in the table, say so. You are not a financial advisor; keep answers concise
and note key risks briefly rather than lecturing."""


def _get_api_key() -> str | None:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("ANTHROPIC_API_KEY") or st.session_state.get("user_api_key")


def _table_context(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "(No results table is currently loaded.)"
    shown = df.head(MAX_ROWS)
    note = f"Showing first {len(shown)} of {len(df)} rows.\n" if len(df) > len(shown) else ""
    return note + shown.to_csv(index=False)


def render_chatbot(df: pd.DataFrame | None = None, extra_context: str = "", key: str = "chat"):
    msgs_key = f"{key}_messages"
    st.session_state.setdefault(msgs_key, [])

    # API key: secrets/env first, otherwise ask in the sidebar
    if not (os.getenv("ANTHROPIC_API_KEY") or _secret_exists()):
        st.sidebar.text_input("Anthropic API key", type="password", key="user_api_key")
    api_key = _get_api_key()

    if st.button("Clear chat", key=f"{key}_clear"):
        st.session_state[msgs_key] = []

    for m in st.session_state[msgs_key]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask about these results...", key=f"{key}_input")
    if not prompt:
        return
    if not api_key:
        st.warning("Add an Anthropic API key (sidebar or Streamlit secrets) to chat.")
        return

    st.session_state[msgs_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system = (
        f"{SYSTEM_PROMPT}\n\n{extra_context}\n\n"
        f"<results_table>\n{_table_context(df)}\n</results_table>"
    )
    history = st.session_state[msgs_key][-MAX_HISTORY:]

    with st.chat_message("assistant"):
        try:
            client = Anthropic(api_key=api_key)
            with client.messages.stream(
                model=MODEL,
                max_tokens=1500,
                system=system,
                messages=history,
            ) as stream:
                reply = st.write_stream(stream.text_stream)
        except Exception as e:
            reply = f"Sorry, the request failed: {e}"
            st.error(reply)

    st.session_state[msgs_key].append({"role": "assistant", "content": reply})


def _secret_exists() -> bool:
    try:
        return "ANTHROPIC_API_KEY" in st.secrets
    except Exception:
        return False

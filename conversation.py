"""
Conversation memory: lets follow-up questions reference earlier answers in
the same session ("what about its limitations?" right after asking about
the main findings).

Deliberately narrow in scope:

  - Threaded into GENERATION only, as proper multi-turn chat messages
    (Ollama accepts a list of role/content turns natively - this is the
    standard mechanism for multi-turn chat, not a homemade string format).
    This lets the model resolve pronouns/follow-up phrasing when composing
    its answer.

  - Retrieval stays single-turn: hybrid_retrieve/get_chunks_by_pages search
    using only the CURRENT question's text, never history. Two reasons:
    (1) query_understanding.parse_query uses plain regex to detect
    'page N'/'filter:'/'broad:'/'summarize:' - if an earlier answer's text
    got concatenated into the next query before parsing, a stray token in
    that answer could misfire intent routing on a totally unrelated
    question; (2) it keeps retrieval predictable and tied to what you just
    typed, rather than silently drifting based on history you're not
    necessarily tracking mentally.

  - Only the question and final answer text are stored per turn - not the
    retrieved context that produced the answer. Re-sending full retrieved
    context on every subsequent turn would make each turn's request grow
    without bound and duplicate information the answer already
    synthesized; the answer text is what a follow-up actually needs to
    refer back to.

  - Summarization turns are never added (see main.py) - summaries can be
    long, and every future turn would carry that weight for the life of
    the history window.
"""
from collections import deque


class ConversationMemory:
    """
    Bounded history of (question, answer) turns. Toggling `enabled` off
    doesn't discard what's already stored - it just stops threading it
    into new generation calls, so turning memory back on resumes with
    whatever context had already accumulated.
    """

    def __init__(self, max_turns):
        self.max_turns = max_turns
        self.enabled = True
        self._turns = deque(maxlen=max_turns)

    def add(self, question, answer):
        self._turns.append((question, answer))

    def clear(self):
        self._turns.clear()

    def __len__(self):
        return len(self._turns)

    def as_messages(self):
        """Chat-formatted history for the current turn's generation call.
        Empty whenever memory is toggled off, regardless of what's stored,
        or when nothing has been recorded yet."""
        if not self.enabled:
            return []
        messages = []
        for question, answer in self._turns:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return messages

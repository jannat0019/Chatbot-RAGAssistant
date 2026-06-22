"""
Conversation Memory

LangChain concepts applied (from Coursera lab):
  - Memory management
  - Chat message types (HumanMessage, AIMessage)
  
Maintains conversation history for all three modes.
"""

from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage


class ConversationMemory:
    """
    Simple in-memory store for the conversation.
    Converts to LangChain message format when needed.
    """

    def __init__(self, max_turns: int = 10):
        self._messages: List[Dict[str, Any]] = []
        self.max_turns = max_turns

    def add_message(self, role: str, content: str, **kwargs):
        """Add a message. role = 'user' | 'assistant'"""
        entry = {"role": role, "content": content}
        entry.update(kwargs)   # e.g. sources=["page 3", "page 7"]
        self._messages.append(entry)

        # Keep only last max_turns pairs (2 messages per turn)
        if len(self._messages) > self.max_turns * 2:
            self._messages = self._messages[-(self.max_turns * 2):]

    def get_messages(self) -> List[Dict[str, Any]]:
        return self._messages

    def get_langchain_history(self) -> List:
        """
        Return history as LangChain message objects.
        Used by ConversationalRetrievalChain and AgentExecutor.
        """
        history = []
        for msg in self._messages:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        return history

    def clear(self):
        self._messages = []

    def __len__(self):
        return len(self._messages)

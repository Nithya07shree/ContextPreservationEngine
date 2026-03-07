import time
from retrieval.retriever import retrieve_context
from generation.generator import generate_response
from config.logger import get_logger

logger = get_logger(__name__)
MAX_HISTORY_TURNS = 10


class DocumentationBot:
    def __init__(self, user_role: str = "onboarding") -> None:
        self.user_role = user_role
        self.conversation_history: list[dict] = []
        self._total_queries = 0 


    def ask(self, query: str, retries: int = 2, delay: float = 1.0) -> tuple[str, list]:
        if not query or not query.strip():
            logger.warning("ask() called with empty query")
            return "Please enter a question.",[]

        logger.info("User query: '%s...'", query[:100])

        last_exc = None
        wait = delay

        for attempt in range(1, retries + 2): 
            try:
                context_chunks = retrieve_context(query, user_role = self.user_role)

                if not context_chunks:
                    logger.warning("No context chunks retrieved for query: '%s...'", query[:60])

                response = generate_response(
                    user_query           = query,
                    context_chunks       = context_chunks,
                    conversation_history = self.conversation_history,
                )

                self._append_to_history(query, response)
                self._total_queries += 1

                logger.info(
                    "Query answered | attempt=%d  history_turns=%d  total_queries=%d",
                    attempt, len(self.conversation_history) // 2, self._total_queries,
                )
                return response, context_chunks

            except Exception as e:
                last_exc = e
                if attempt <= retries:
                    logger.warning(
                        "Pipeline failed (attempt %d/%d), retrying in %.1fs — %s",
                        attempt, retries + 1, wait, e,
                    )
                    time.sleep(wait)
                    wait *= 2  
                else:
                    logger.error("All %d pipeline attempts failed — %s", retries + 1, e)

        return (
            "Sorry, I ran into an issue processing your question. "
            "The local model or database may be temporarily unavailable — "
            "please try again in a moment.",[]
        )

    def reset(self) -> None:
        previous_turns = len(self.conversation_history) // 2
        self.conversation_history.clear()
        logger.info("Conversation reset | cleared %d turn(s)", previous_turns)

    @property
    def history_turns(self) -> int:
        return len(self.conversation_history) // 2

    def _append_to_history(self, query: str, response: str) -> None:
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        self._trim_history()

    def _trim_history(self) -> None:
        max_messages = MAX_HISTORY_TURNS * 2

        if len(self.conversation_history) > max_messages:
            removed = len(self.conversation_history) - max_messages
            self.conversation_history = self.conversation_history[-max_messages:]
            logger.debug("Trimmed %d old message(s) from history", removed)
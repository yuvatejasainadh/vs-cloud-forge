import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import threading
from telemetry.logging import get_logger

logger = get_logger("lacr")


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    total_chunks: int = 0
    lid_evaluations: int = 0
    current_language: Optional[str] = None
    current_group: Optional[str] = None
    current_vm: Optional[str] = None
    confidence_history: List[float] = field(default_factory=list)
    language_switches: int = 0
    is_locked: bool = False


class LACRManager:
    """
    Language-Aware Continuous Routing (LACR) Manager.
    Coordinates temporal LID routing across streaming audio chunks, reducing
    computational overhead while preserving agile mid-call code-switching detection.
    """

    def __init__(
        self,
        checkpoint_interval: int = 5,
        session_ttl_seconds: float = 3600.0,
        switch_confidence_threshold: float = 0.80
    ):
        self.checkpoint_interval = checkpoint_interval
        self.session_ttl_seconds = session_ttl_seconds
        self.switch_confidence_threshold = switch_confidence_threshold
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def should_evaluate_lid(self, session_id: str, chunk_index: int) -> bool:
        """
        Determines whether the given chunk requires full LID evaluation.
        First chunk (index 0 or 1) and every `checkpoint_interval` chunk triggers LID.
        """
        if not session_id:
            return True

        with self._lock:
            self._cleanup_expired_sessions()
            session = self._sessions.get(session_id)
            if not session:
                return True

            # If no language assigned yet, must evaluate
            if session.current_language is None:
                return True

            # Chunk index based evaluation
            if chunk_index <= 1:
                return True

            # Periodic checkpointing
            if chunk_index % self.checkpoint_interval == 1 or (self.checkpoint_interval == 1):
                return True

            return False

    def update_session(
        self,
        session_id: str,
        chunk_index: int,
        predicted_language: str,
        predicted_group: str,
        predicted_vm: str,
        confidence: float
    ) -> Tuple[str, str, str, bool]:
        """
        Updates session state with latest LID inference.

        Returns:
            Tuple[effective_language, effective_group, effective_vm, has_switched]
        """
        if not session_id:
            return predicted_language, predicted_group, predicted_vm, False

        with self._lock:
            self._cleanup_expired_sessions()
            session = self._sessions.get(session_id)
            if not session:
                session = SessionState(session_id=session_id)
                self._sessions[session_id] = session

            session.total_chunks = max(session.total_chunks + 1, chunk_index)
            session.lid_evaluations += 1
            session.last_updated_at = time.time()
            session.confidence_history.append(confidence)

            has_switched = False

            # Initial assignment
            if session.current_language is None:
                session.current_language = predicted_language
                session.current_group = predicted_group
                session.current_vm = predicted_vm
                return session.current_language, session.current_group, session.current_vm, False

            # Check for language switch
            if predicted_language != session.current_language:
                if confidence >= self.switch_confidence_threshold:
                    logger.warning(
                        f"LACR: Session {session_id} code-switch detected: "
                        f"{session.current_language} ({session.current_vm}) -> {predicted_language} ({predicted_vm}) "
                        f"with confidence {confidence:.3f}"
                    )
                    session.language_switches += 1
                    session.current_language = predicted_language
                    session.current_group = predicted_group
                    session.current_vm = predicted_vm
                    has_switched = True
                else:
                    logger.debug(
                        f"LACR: Session {session_id} low-confidence divergence ({predicted_language} @ {confidence:.3f}), "
                        f"retaining current assignment {session.current_language}"
                    )

            return session.current_language, session.current_group, session.current_vm, has_switched

    def get_cached_routing(self, session_id: str) -> Optional[Tuple[str, str, str]]:
        """Returns cached (language, group, vm) for a session if active."""
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.current_language:
                return session.current_language, session.current_group, session.current_vm
            return None

    def _cleanup_expired_sessions(self) -> None:
        """Removes sessions exceeding TTL."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_updated_at > self.session_ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]


# Global LACR singleton
_lacr_manager: Optional[LACRManager] = None


def get_lacr_manager(checkpoint_interval: int = 5) -> LACRManager:
    global _lacr_manager
    if _lacr_manager is None:
        _lacr_manager = LACRManager(checkpoint_interval=checkpoint_interval)
    return _lacr_manager

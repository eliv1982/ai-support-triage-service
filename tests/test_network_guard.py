import socket

import pytest

from app.services import triage_service


def test_external_connections_are_blocked():
    # 203.0.113.0/24 is reserved for documentation (TEST-NET-3); the guard refuses
    # before any packet could leave the machine.
    with pytest.raises(RuntimeError, match="Network access blocked"):
        socket.create_connection(("203.0.113.1", 443), timeout=1)


def test_real_openai_client_cannot_be_built_by_accident():
    with pytest.raises(RuntimeError, match="inject a fake llm_client"):
        triage_service.TriageService()

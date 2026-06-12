from sqlalchemy import select

from app.models import Ticket
from app.services.triage_service import TriageService, fallback_response


def build_mock_llm_client(content: str | None = None, error: Exception | None = None):
    class MockCompletions:
        @staticmethod
        def create(**kwargs):
            if error:
                raise error
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {"content": content},
                                )()
                            },
                        )()
                    ]
                },
            )()

    class MockClient:
        chat = type("MockChat", (), {"completions": MockCompletions()})()

    return MockClient()


def test_successful_triage_with_mocked_llm(client, db_session):
    service = TriageService(
        llm_client=build_mock_llm_client(
            content='{"category":"support","draft_reply":"We received your request and will help shortly.","confidence":"high","escalate":false}'
        )
    )
    test_client = client(service)

    response = test_client.post(
        "/triage",
        json={
            "text": "My account is locked.",
            "channel": "chat",
            "client_id": "client-1",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "category": "support",
        "draft_reply": "We received your request and will help shortly.",
        "confidence": "high",
        "escalate": False,
    }

    ticket = db_session.scalar(select(Ticket))
    assert ticket is not None
    assert ticket.category == "support"
    assert ticket.error is None


def test_validation_error_for_empty_text(client):
    service = TriageService(llm_client=build_mock_llm_client(content="{}"))
    test_client = client(service)

    response = test_client.post(
        "/triage",
        json={"text": "", "channel": "chat", "client_id": "client-1"},
    )

    assert response.status_code == 422


def test_validation_error_for_invalid_channel(client):
    service = TriageService(llm_client=build_mock_llm_client(content="{}"))
    test_client = client(service)

    response = test_client.post(
        "/triage",
        json={
            "text": "Need help",
            "channel": "phone",
            "client_id": "client-1",
        },
    )

    assert response.status_code == 422


def test_rate_limit_returns_429(client):
    service = TriageService(
        llm_client=build_mock_llm_client(
            content='{"category":"support","draft_reply":"We will review your request.","confidence":"medium","escalate":false}'
        )
    )
    test_client = client(service)

    request_body = {
        "text": "Need help",
        "channel": "chat",
        "client_id": "client-1",
    }

    assert test_client.post("/triage", json=request_body).status_code == 200
    assert test_client.post("/triage", json=request_body).status_code == 200
    response = test_client.post("/triage", json=request_body)

    assert response.status_code == 429
    assert response.json()["detail"] == "Rate limit exceeded"


def test_llm_failure_returns_fallback_and_saves_ticket(client, db_session):
    service = TriageService(
        llm_client=build_mock_llm_client(error=RuntimeError("LLM unavailable"))
    )
    test_client = client(service)

    response = test_client.post(
        "/triage",
        json={
            "text": "I want a refund and no one replied.",
            "channel": "email",
            "client_id": "client-2",
        },
    )

    assert response.status_code == 200
    assert response.json() == fallback_response().model_dump()

    ticket = db_session.scalar(select(Ticket))
    assert ticket is not None
    assert ticket.category == "other"
    assert ticket.error == "LLM unavailable"

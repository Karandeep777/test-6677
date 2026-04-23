
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import types

import agent
from agent import MeetingNotesSummarizerAgent, MeetingSummaryRequest, MeetingSummaryResponse, InputHandler, ComplianceGuard, LLMService, FALLBACK_RESPONSE

# For endpoint tests
from fastapi.testclient import TestClient

@pytest.fixture
def agent_instance():
    return MeetingNotesSummarizerAgent()

@pytest.fixture
def valid_transcript():
    return "Alice: Let's review the project plan.\nBob: I'll send the API docs by Friday."

@pytest.fixture
def valid_request(valid_transcript):
    return MeetingSummaryRequest(
        transcript_text=valid_transcript,
        summary_length="full",
        follow_up_query=None,
        user_email=None,
        user_consent=False
    )

@pytest.mark.asyncio
async def test_process_valid_transcript_text_input(agent_instance, valid_request):
    """Validates process_input correctly normalizes direct transcript text input."""
    result = await agent_instance.process_input(valid_request)
    assert isinstance(result, str)
    assert result.strip() != ""
    # Should collapse whitespace and remove control chars
    assert "\r" not in result
    assert result.startswith("Alice:")
    assert "Bob:" in result

@pytest.mark.asyncio
async def test_generate_summary_with_valid_transcript(agent_instance, valid_transcript):
    """Ensures generate_summary produces a structured summary dict for valid transcript."""
    # Patch LLMService.generate_summary to avoid real LLM call
    with patch.object(agent.LLMService, "get_llm_client") as mock_client:
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Meeting Overview: ...\nKey Points: ..."))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_llm

        result = await agent_instance.generate_summary(
            transcript_text=valid_transcript,
            summary_length="full",
            follow_up_query=None
        )
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert isinstance(result.get("summary"), str)
        assert len(result.get("summary")) > 0

def test_format_summary_as_email(agent_instance):
    """Checks format_summary returns properly formatted email-ready summary."""
    summary_dict = {"summary": "Meeting Overview: Discussed project plan.\nAction Items: Bob to send docs."}
    result = agent_instance.format_summary(summary_dict, output_format="email")
    assert isinstance(result, str)
    assert result.startswith("Dear Team,")
    assert result.strip().endswith("Best regards,\nMeeting Notes Summarizer Agent")

@pytest.mark.asyncio
async def test_send_summary_email_with_consent(agent_instance):
    """Verifies send_summary_email sends email when user_consent is True."""
    with patch.object(agent.EmailSender, "send_email", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = "Summary sent to alice@example.com."
        result = await agent_instance.send_summary_email(
            user_email="alice@example.com",
            summary_content="Summary body",
            user_consent=True
        )
        assert "Summary sent to" in result
        assert mock_send.await_count == 1

@pytest.mark.asyncio
async def test_reject_email_send_without_consent(agent_instance):
    """Ensures send_summary_email raises error if user_consent is False."""
    with pytest.raises(ValueError) as excinfo:
        await agent_instance.send_summary_email(
            user_email="bob@example.com",
            summary_content="Summary body",
            user_consent=False
        )
    assert "ERR_EMAIL_CONSENT_REQUIRED" in str(excinfo.value)

@pytest.mark.asyncio
async def test_handle_unsupported_file_type_in_extract_text():
    """Checks extract_text raises ValueError for unsupported file types."""
    audit_logger = agent.AuditLogger()
    handler = InputHandler(audit_logger)
    # Simulate UploadFile with .pdf extension
    fake_file = MagicMock()
    fake_file.filename = "meeting.pdf"
    # .read() should not be called, but if it is, return dummy bytes
    fake_file.read = AsyncMock(return_value=b"dummy")
    with pytest.raises(ValueError) as excinfo:
        await handler.extract_text(fake_file)
    assert "Unsupported file type" in str(excinfo.value)

def test_compliance_guard_enforces_transcript_size_limit():
    """Validates ComplianceGuard.validate_processing raises error for large transcripts."""
    audit_logger = agent.AuditLogger()
    guard = ComplianceGuard(audit_logger)
    big_text = "A" * 50001
    with pytest.raises(ValueError) as excinfo:
        guard.validate_processing(big_text)
    assert "too large for in-memory processing" in str(excinfo.value)

def test_meeting_summary_request_validates_invalid_email():
    """Checks MeetingSummaryRequest raises validation error for invalid email."""
    with pytest.raises(agent.ValidationError) as excinfo:
        agent.MeetingSummaryRequest(
            transcript_text="Some transcript",
            summary_length="full",
            user_email="not-an-email"
        )
    assert "Invalid email address" in str(excinfo.value)

@pytest.mark.asyncio
async def test_llmservice_generate_summary_retries_and_fallback():
    """Ensures LLMService.generate_summary retries and returns fallback summary on persistent failure."""
    audit_logger = agent.AuditLogger()
    llm_service = LLMService(audit_logger)
    # Patch get_llm_client to always raise Exception
    with patch.object(llm_service, "get_llm_client", side_effect=Exception("LLM down")):
        result = await llm_service.generate_summary(
            transcript_text="Some transcript",
            summary_length="full",
            follow_up_query=None
        )
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert result.get("summary") == FALLBACK_RESPONSE
        assert "error" in result

def test_handle_unsupported_file_type_in_extract_text_sync():
    """(Edge) extract_text should raise ValueError for .pdf even if called sync (defensive)."""
    audit_logger = agent.AuditLogger()
    handler = InputHandler(audit_logger)
    fake_file = MagicMock()
    fake_file.filename = "meeting.pdf"
    fake_file.read = AsyncMock(return_value=b"dummy")
    # Defensive: ensure error is always raised for .pdf
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(handler.extract_text(fake_file))
    assert "Unsupported file type" in str(excinfo.value)

def test_compliance_guard_enforces_transcript_size_limit_sync():
    """(Edge) ComplianceGuard.validate_processing raises error for >50k chars (sync defensive)."""
    audit_logger = agent.AuditLogger()
    guard = ComplianceGuard(audit_logger)
    big_text = "B" * 60000
    with pytest.raises(ValueError) as excinfo:
        guard.validate_processing(big_text)
    assert "too large for in-memory processing" in str(excinfo.value)

@pytest.mark.asyncio
async def test_process_input_empty_transcript_raises(agent_instance):
    """process_input should raise ValueError if transcript_text is empty or whitespace."""
    req = MeetingSummaryRequest(transcript_text="   ")
    with pytest.raises(ValueError) as excinfo:
        await agent_instance.process_input(req)
    assert "ERR_NO_TRANSCRIPT" in str(excinfo.value)

@pytest.mark.asyncio
async def test_generate_summary_missing_transcript_raises(agent_instance):
    """generate_summary should raise error if transcript_text is missing."""
    with pytest.raises(Exception):
        await agent_instance.generate_summary(
            transcript_text=None,
            summary_length="full",
            follow_up_query=None
        )

def test_format_summary_returns_fallback_on_exception(agent_instance):
    """format_summary returns fallback response if formatting fails."""
    # Patch agent_instance.summary_formatter.format_summary to raise
    with patch.object(agent_instance.summary_formatter, "format_summary", side_effect=Exception("fail")):
        result = agent_instance.format_summary({"summary": "irrelevant"}, output_format="email")
        assert result == agent.FALLBACK_RESPONSE

@pytest.mark.asyncio
async def test_generate_summary_llm_api_error_triggers_fallback(agent_instance, valid_transcript):
    """generate_summary returns fallback summary if LLM API fails."""
    with patch.object(agent.LLMService, "get_llm_client", side_effect=Exception("LLM error")):
        result = await agent_instance.generate_summary(
            transcript_text=valid_transcript,
            summary_length="full",
            follow_up_query=None
        )
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert result.get("summary") == agent.FALLBACK_RESPONSE

def test_meeting_summary_request_validates_summary_length():
    """MeetingSummaryRequest raises validation error for invalid summary_length."""
    with pytest.raises(agent.ValidationError) as excinfo:
        agent.MeetingSummaryRequest(
            transcript_text="Some transcript",
            summary_length="invalid"
        )
    assert "summary_length must be one of" in str(excinfo.value)

def test_meeting_summary_request_validates_empty_transcript():
    """MeetingSummaryRequest raises validation error for empty transcript_text."""
    with pytest.raises(agent.ValidationError) as excinfo:
        agent.MeetingSummaryRequest(
            transcript_text="   ",
            summary_length="full"
        )
    assert "transcript_text cannot be empty" in str(excinfo.value)

@pytest.mark.asyncio
async def test_meeting_summary_request_validates_long_transcript():
    """MeetingSummaryRequest raises validation error for transcript_text > 50,000 chars."""
    long_text = "A" * 50001
    with pytest.raises(agent.ValidationError) as excinfo:
        agent.MeetingSummaryRequest(
            transcript_text=long_text,
            summary_length="full"
        )
    assert "transcript_text exceeds 50,000 character limit" in str(excinfo.value)

@pytest.mark.asyncio
async def test_summarize_endpoint_returns_error_on_malformed_json():
    """Ensures /summarize endpoint returns error for malformed JSON input."""
    from agent import app
    client = TestClient(app)
    # Malformed JSON (missing closing brace)
    response = client.post("/summarize", data='{"transcript_text": "abc"')
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Malformed JSON request" in data["error"]
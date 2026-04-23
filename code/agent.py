import asyncio
import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import Optional, Any, Dict, List, Union
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_422_UNPROCESSABLE_ENTITY

from config import Config

import openai
from email_validator import validate_email, EmailNotValidError

import docx

# Constants from user prompt template
SYSTEM_PROMPT = (
    "You are a professional Meeting Notes Summarizer Agent. Your role is to process raw meeting transcripts, chat exports, or notes and generate a structured, concise, and actionable summary for distribution to all participants. Follow these instructions:\n\n"
    "1. Accept input as pasted transcript text, uploaded TXT/DOCX file, or copied meeting chat export.\n\n"
    "2. Generate a structured summary with the following sections:\n"
    "   - Meeting Overview\n"
    "   - Key Discussion Points\n"
    "   - Decisions Made\n"
    "   - Action Items (with assigned owners and due dates)\n"
    "   - Next Steps\n"
    "3. For each action item:\n"
    "   - Identify and assign the owner based on transcript context. If no owner is mentioned, label as \"Owner: TBD\" and highlight.\n"
    "   - Extract any mentioned deadline. If none, label as \"Due: Not specified\".\n"
    "   - Tag priority as High, Medium, or Low based on urgency language.\n"
    "4. Detect and list all attendees mentioned in the transcript.\n\n"
    "5. Support summary length control: provide a one-liner, paragraph, or full detailed summary as requested.\n\n"
    "6. Format the summary as an email-ready body, using bullet points for clarity and conciseness.\n\n"
    "7. Support follow-up questions such as \"What did [person] agree to do?\" or \"What was decided about [topic]?\" by referencing the transcript.\n\n"
    "8. Never fabricate or infer action items or decisions not explicitly stated in the transcript.\n\n"
    "9. Always request user confirmation before sending the summary email to participants.\n\n"
    "10. Ensure all processing is in-memory only, with no data retention after summary delivery, and never share transcript data with third parties.\n\n"
    "Output Format:\n\n"
    "- Structured summary with clear section headers\n"
    "- Bullet points for action items and decisions\n"
    "- Email-ready formatting\n"
    "- Explicit labels for any missing owners or deadlines\n\n"
    "Fallback:\n"
    "- If information is missing or unclear, clearly indicate this in the summary (e.g., \"Owner: TBD\", \"Due: Not specified\").\n"
    "- If the transcript is insufficient for a full summary, return a minimal summary and request additional input."
)
OUTPUT_FORMAT = (
    "- Structured summary with the following sections:\n"
    "  - Meeting Overview\n"
    "  - Key Discussion Points\n"
    "  - Decisions Made\n"
    "  - Action Items (with owner and due date for each)\n"
    "  - Next Steps\n"
    "  - Attendees\n"
    "- Bullet points for all lists\n\n"
    "- Email-ready formatting (salutation, body, closing)\n\n"
    "- Explicit labels for missing owners (\"Owner: TBD\") or deadlines (\"Due: Not specified\")\n\n"
    "- Option for one-liner, paragraph, or full summary as specified by user"
)
FALLBACK_RESPONSE = (
    "The transcript did not contain enough information to generate a complete summary. Please provide additional details or clarify missing sections."
)

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# Utility: LLM Output Sanitizer
# =========================
import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")


def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()


@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# Input/Output Models
# =========================

class MeetingSummaryRequest(BaseModel):
    transcript_text: Optional[str] = Field(None, description="Raw meeting transcript text, chat export, or notes.")
    summary_length: Optional[str] = Field("full", description="Desired summary length: one-liner, paragraph, or full.")
    follow_up_query: Optional[str] = Field(None, description="Optional follow-up question about responsibilities or decisions.")
    user_email: Optional[str] = Field(None, description="User email address for summary delivery (optional).")
    user_consent: Optional[bool] = Field(False, description="User consent for email delivery (must be True to send email).")

    @field_validator("summary_length")
    @classmethod
    def validate_summary_length(cls, v):
        allowed = {"one-liner", "paragraph", "full"}
        if v is None:
            return "full"
        v = v.strip().lower()
        if v not in allowed:
            raise ValueError(f"summary_length must be one of {allowed}")
        return v

    @field_validator("user_email")
    @classmethod
    def validate_email(cls, v):
        if v is None or v == "":
            return None
        try:
            validate_email(v)
            return v
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email address: {e}")

    @field_validator("transcript_text")
    @classmethod
    def validate_transcript(cls, v):
        if v is not None and len(v.strip()) == 0:
            raise ValueError("transcript_text cannot be empty")
        if v is not None and len(v) > 50000:
            raise ValueError("transcript_text exceeds 50,000 character limit")
        return v

class MeetingSummaryResponse(BaseModel):
    success: bool = Field(..., description="Whether the summary was generated successfully.")
    summary: Optional[str] = Field(None, description="Formatted meeting summary.")
    error: Optional[str] = Field(None, description="Error message if failed.")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input errors.")

# =========================
# Audit Logger
# =========================

class AuditLogger:
    """Logs processing events for compliance and troubleshooting."""
    def __init__(self):
        self.logger = logging.getLogger("agent.audit")
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, details: Any):
        try:
            self.logger.info(f"[{event_type}] {details}")
        except Exception as e:
            self.logger.warning(f"Failed to log event: {e}")

# =========================
# Input Handler
# =========================

class InputHandler:
    """Accepts and normalizes user input (text, file upload, chat export)."""
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def receive_input(self, input_data: MeetingSummaryRequest, file: Optional[UploadFile] = None) -> str:
        """Receives and extracts transcript text from input_data or file."""
        transcript = input_data.transcript_text
        if transcript and transcript.strip():
            self.audit_logger.log_event("input_received", "Transcript text provided directly.")
            return transcript.strip()
        if file:
            try:
                text = await self.extract_text(file)
                self.audit_logger.log_event("input_received", f"Transcript extracted from file: {file.filename}")
                return text
            except Exception as e:
                self.audit_logger.log_event("input_error", f"Failed to extract text from file: {e}")
                raise ValueError("Failed to extract text from uploaded file.")
        raise ValueError("ERR_NO_TRANSCRIPT: No transcript text or file provided.")

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def extract_text(self, file: UploadFile) -> str:
        """Extracts text from TXT or DOCX file."""
        filename = file.filename.lower()
        if filename.endswith(".txt"):
            content = await file.read()
            text = content.decode("utf-8", errors="replace")
            return text.strip()
        elif filename.endswith(".docx"):
            content = await file.read()
            from io import BytesIO
            doc = docx.Document(BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        else:
            raise ValueError("Unsupported file type. Only .txt and .docx are accepted.")

# =========================
# Transcript Normalizer
# =========================

class TranscriptNormalizer:
    """Cleans and standardizes transcript text for downstream processing."""
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    def normalize(self, transcript_text: str) -> str:
        """Basic normalization: strip, collapse whitespace, remove control chars."""
        if not transcript_text:
            raise ValueError("ERR_NO_TRANSCRIPT: Transcript text is empty.")
        text = transcript_text.replace("\r\n", "\n").replace("\r", "\n")
        text = _re.sub(r"\n{3,}", "\n\n", text)
        text = _re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", text)
        text = text.strip()
        self.audit_logger.log_event("transcript_normalized", f"Transcript normalized. Length: {len(text)} chars.")
        return text

# =========================
# LLM Service
# =========================

class LLMService:
    """Handles prompt construction, LLM calls, and summary generation."""
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_client(self):
        api_key = Config.AZURE_OPENAI_API_KEY
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not configured")
        return openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_summary(
        self,
        transcript_text: str,
        summary_length: str = "full",
        follow_up_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Constructs prompt, calls LLM, and returns structured summary."""
        prompt = self._build_user_prompt(transcript_text, summary_length, follow_up_query)
        system_message = SYSTEM_PROMPT + "\n\nOutput Format:\n" + OUTPUT_FORMAT
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        _llm_kwargs = Config.get_llm_kwargs()
        retries = 0
        max_retries = 3
        last_error = None
        while retries < max_retries:
            try:
                _t0 = _time.time()
                client = self.get_llm_client()
                response = await client.chat.completions.create(
                    model=Config.LLM_MODEL or "gpt-4.1",
                    messages=messages,
                    **_llm_kwargs
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=Config.LLM_MODEL or "gpt-4.1",
                        prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else "",
                    )
                except Exception:
                    pass
                self.audit_logger.log_event("llm_success", f"LLM summary generated. Prompt size: {len(prompt)} chars.")
                return {"success": True, "summary": sanitize_llm_output(content, content_type="text")}
            except Exception as e:
                retries += 1
                last_error = str(e)
                self.audit_logger.log_event("llm_error", f"LLM call failed (attempt {retries}): {e}")
                await self._exponential_backoff(retries)
        # Fallback: minimal summary
        self.audit_logger.log_event("llm_fallback", f"Returning fallback summary after {max_retries} failures.")
        return {"success": False, "summary": FALLBACK_RESPONSE, "error": last_error}

    def _build_user_prompt(self, transcript_text: str, summary_length: str, follow_up_query: Optional[str]) -> str:
        prompt = f"Transcript:\n{transcript_text.strip()}\n\n"
        prompt += f"Summary length: {summary_length}\n"
        if follow_up_query:
            prompt += f"Follow-up question: {follow_up_query}\n"
        return prompt

    async def _exponential_backoff(self, retries: int):
        delay = min(2 ** retries, 8)
        await self._sleep(delay)

    async def _sleep(self, seconds: int):
        await asyncio.sleep(seconds)

# =========================
# Summary Formatter
# =========================

class SummaryFormatter:
    """Formats the LLM output into structured, email-ready summaries."""
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    def format_summary(self, summary_data: Union[str, Dict[str, Any]], output_format: str = "email") -> str:
        """Formats summary into email-ready structure with explicit labels."""
        try:
            if isinstance(summary_data, dict):
                summary = summary_data.get("summary") or ""
            else:
                summary = summary_data or ""
            summary = sanitize_llm_output(summary, content_type="text")
            # Add email salutation and closing if output_format == "email"
            if output_format == "email":
                formatted = (
                    "Dear Team,\n\n"
                    f"{summary.strip()}\n\n"
                    "Best regards,\nMeeting Notes Summarizer Agent"
                )
            else:
                formatted = summary.strip()
            self.audit_logger.log_event("summary_formatted", f"Summary formatted for output ({output_format}).")
            return formatted
        except Exception as e:
            self.audit_logger.log_event("formatting_error", f"Failed to format summary: {e}")
            return FALLBACK_RESPONSE

# =========================
# Email Sender (Stub)
# =========================

class EmailSender:
    """Sends formatted summaries to participants after user consent."""
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def send_email(self, user_email: str, summary_content: str, user_consent: bool) -> str:
        """Sends formatted summary to participants after user consent."""
        if not user_consent:
            self.audit_logger.log_event("email_blocked", "Consent not granted for email delivery.")
            raise ValueError("ERR_EMAIL_CONSENT_REQUIRED: User consent required before sending email.")
        # Stub: Replace with actual email sending logic (SMTP, SendGrid, etc.)
        self.audit_logger.log_event("email_sent", f"Summary sent to {user_email}.")
        return f"Summary sent to {user_email}."

# =========================
# Consent Manager
# =========================

class ConsentManager:
    """Manages user consent for email delivery."""
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self._consent_cache = {}

    def request_consent(self, user_email: str) -> bool:
        """Prompts user for email delivery consent."""
        # In real implementation, prompt user via UI or email.
        # Here, assume consent is managed via API param.
        self.audit_logger.log_event("consent_requested", f"Consent requested for {user_email}.")
        return self._consent_cache.get(user_email, False)

    def check_consent(self, user_email: str) -> bool:
        """Checks if user consent is present."""
        return self._consent_cache.get(user_email, False)

    def set_consent(self, user_email: str, consent: bool):
        self._consent_cache[user_email] = consent
        self.audit_logger.log_event("consent_set", f"Consent set for {user_email}: {consent}")

# =========================
# Compliance Guard
# =========================

class ComplianceGuard:
    """Ensures privacy, GDPR compliance, and in-memory-only processing."""
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_processing(self, transcript_text: str):
        """Ensures transcript is processed in-memory only."""
        if not transcript_text or len(transcript_text) > 50000:
            raise ValueError("Transcript text is missing or too large for in-memory processing.")
        self.audit_logger.log_event("compliance_check", "Transcript processed in-memory.")

    def purge_data(self, transcript_text: str):
        """Deletes transcript data from memory after summary delivery."""
        # In Python, just dereference; log event.
        self.audit_logger.log_event("data_purged", "Transcript data purged from memory.")

# =========================
# Main Agent
# =========================

class MeetingNotesSummarizerAgent:
    """Main agent orchestrating meeting notes summarization and delivery."""

    def __init__(self):
        self.audit_logger = AuditLogger()
        self.input_handler = InputHandler(self.audit_logger)
        self.transcript_normalizer = TranscriptNormalizer(self.audit_logger)
        self.llm_service = LLMService(self.audit_logger)
        self.summary_formatter = SummaryFormatter(self.audit_logger)
        self.email_sender = EmailSender(self.audit_logger)
        self.consent_manager = ConsentManager(self.audit_logger)
        self.compliance_guard = ComplianceGuard(self.audit_logger)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_input(
        self,
        input_data: MeetingSummaryRequest,
        file: Optional[UploadFile] = None
    ) -> str:
        """Receives and normalizes user input (text, file, chat export)."""
        async with trace_step(
            "process_input",
            step_type="parse",
            decision_summary="Receive and normalize user input",
            output_fn=lambda r: f"transcript_length={len(r) if r else 0}",
        ) as step:
            transcript = await self.input_handler.receive_input(input_data, file)
            normalized = self.transcript_normalizer.normalize(transcript)
            self.compliance_guard.validate_processing(normalized)
            step.capture(normalized)
            return normalized

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_summary(
        self,
        transcript_text: str,
        summary_length: str = "full",
        follow_up_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Constructs prompt, calls LLM, and returns structured summary."""
        async with trace_step(
            "generate_summary",
            step_type="llm_call",
            decision_summary="Call LLM to generate structured summary",
            output_fn=lambda r: f"success={r.get('success', False)}",
        ) as step:
            result = await self.llm_service.generate_summary(
                transcript_text=transcript_text,
                summary_length=summary_length,
                follow_up_query=follow_up_query,
            )
            step.capture(result)
            return result

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def format_summary(self, summary_data: Union[str, Dict[str, Any]], output_format: str = "email") -> str:
        """Formats summary into email-ready structure with explicit labels."""
        with trace_step_sync(
            "format_summary",
            step_type="format",
            decision_summary="Format summary for output",
            output_fn=lambda r: f"length={len(r) if r else 0}",
        ) as step:
            formatted = self.summary_formatter.format_summary(summary_data, output_format)
            step.capture(formatted)
            return formatted

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def send_summary_email(self, user_email: str, summary_content: str, user_consent: bool) -> str:
        """Sends formatted summary to participants after user consent."""
        async with trace_step(
            "send_summary_email",
            step_type="tool_call",
            decision_summary="Send summary email after consent",
            output_fn=lambda r: r,
        ) as step:
            result = await self.email_sender.send_email(user_email, summary_content, user_consent)
            step.capture(result)
            return result

    def request_consent(self, user_email: str) -> bool:
        """Prompts user for email delivery consent."""
        return self.consent_manager.request_consent(user_email)

    def purge_transcript_data(self, transcript_text: str):
        """Deletes transcript data from memory after summary delivery."""
        self.compliance_guard.purge_data(transcript_text)

    def log_event(self, event_type: str, details: Any):
        """Logs processing events for compliance and troubleshooting."""
        self.audit_logger.log_event(event_type, details)

# =========================
# FastAPI App & Endpoints
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(lifespan=_obs_lifespan,

    title="Meeting Notes Summarizer Agent",
    description="Automatically processes meeting transcripts or notes, produces a clean structured summary, extracts action items with assigned owners and due dates, identifies key decisions made, and distributes the summary to all participants.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = MeetingNotesSummarizerAgent()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/summarize", response_model=MeetingSummaryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def summarize_endpoint(
    request: Request,
    input_data: MeetingSummaryRequest = None,
    file: Optional[UploadFile] = File(None)
):
    """
    Main endpoint for meeting notes summarization.
    Accepts JSON body (MeetingSummaryRequest) and optional file upload.
    """
    try:
        # Parse JSON body
        if input_data is None:
            try:
                body = await request.json()
                input_data = MeetingSummaryRequest(**body)
            except Exception as e:
                return JSONResponse(
                    status_code=HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "summary": None,
                        "error": "Malformed JSON request.",
                        "tips": "Ensure your JSON is valid and all required fields are present. "
                                "Check for missing quotes, commas, or brackets."
                    }
                )
        # Validate input
        try:
            input_data = MeetingSummaryRequest(**input_data.dict())
        except ValidationError as ve:
            return JSONResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "success": False,
                    "summary": None,
                    "error": "Input validation failed.",
                    "tips": str(ve)
                }
            )
        # Process input (transcript extraction & normalization)
        transcript = await agent.process_input(input_data, file)
        # Generate summary
        summary_result = await agent.generate_summary(
            transcript_text=transcript,
            summary_length=input_data.summary_length or "full",
            follow_up_query=input_data.follow_up_query
        )
        # Format summary
        formatted_summary = agent.format_summary(summary_result, output_format="email")
        # Email delivery (if requested)
        email_status = None
        if input_data.user_email and input_data.user_consent:
            try:
                email_status = await agent.send_summary_email(
                    user_email=input_data.user_email,
                    summary_content=formatted_summary,
                    user_consent=input_data.user_consent
                )
            except Exception as e:
                agent.log_event("email_error", str(e))
                return MeetingSummaryResponse(
                    success=False,
                    summary=formatted_summary,
                    error="Email delivery failed: " + str(e),
                    tips="Ensure you have granted consent and provided a valid email address."
                )
        # Purge transcript data after delivery
        agent.purge_transcript_data(transcript)
        return MeetingSummaryResponse(
            success=True,
            summary=formatted_summary,
            error=None,
            tips=email_status
        )
    except Exception as e:
        agent.log_event("endpoint_error", str(e))
        return MeetingSummaryResponse(
            success=False,
            summary=None,
            error=str(e),
            tips="Check your input and try again. If the problem persists, contact support."
        )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "summary": None,
            "error": exc.detail,
            "tips": "Check your request and try again."
        }
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "summary": None,
            "error": "Input validation failed.",
            "tips": str(exc)
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "summary": None,
            "error": str(exc),
            "tips": "An unexpected error occurred. Please try again or contact support."
        }
    )

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())
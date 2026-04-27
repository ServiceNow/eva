"""Shared base class for speech fidelity metrics (agent and user)."""

import asyncio
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Any

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types as genai_types
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from eva.metrics.base import AudioJudgeMetric, MetricContext
from eva.metrics.utils import aggregate_per_turn_scores, normalize_rating, resolve_turn_id
from eva.models.results import MetricScore
from eva.utils.json_utils import extract_and_load_json


class SpeechFidelityBaseMetric(AudioJudgeMetric):
    """Base class for speech fidelity metrics.

    Subclasses must set:
        - name, description, category (metric metadata)
        - role: "assistant" or "user"
        - rating_scale: tuple of (min, max) valid ratings
    """

    role: str  # "assistant" or "user" — set by subclass
    rating_scale: tuple[int, int]  # (min, max) valid ratings — set by subclass
    max_empty_retries: int = 6

    # Silence trimming parameters — collapse long silences to reduce audio token cost.
    silence_thresh_dbfs: int = -45
    min_silence_len_ms: int = 3000
    speech_padding_ms: int = 100
    inter_segment_pause_ms: int = 3000
    trailing_silence_ms: int = 500
    silence_seek_step_ms: int = 50
    save_trimmed_file: bool = False

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.aggregation = self.config.get("aggregation", "mean")

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute speech fidelity score using audio + LLM judge."""
        try:
            audio_segment = self.load_role_audio(context, self.role)
            if audio_segment is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error=f"No {self.role} audio file available",
                )

            audio_segment = self._trim_silence(audio_segment, context)

            intended_turns = self._get_intended_turns(context)
            num_turns = len(intended_turns)
            audio_b64 = self.encode_audio_segment(audio_segment)
            intended_turns_formatted = self._format_intended_turns(intended_turns)

            prompt = self.get_judge_prompt(
                prompt_key="user_prompt",
                intended_turns_formatted=intended_turns_formatted,
            )

            messages = self.create_audio_message(audio_b64, prompt)

            per_turn_ratings: dict[int, int | None] = {}
            per_turn_explanations: dict[int, str] = {}
            per_turn_transcripts: dict[int, str] = {}
            per_turn_normalized: dict[int, float] = {}
            tts_turn_ids = sorted(intended_turns.keys())
            min_rating, max_rating = self.rating_scale
            valid_ratings_range = list(range(min_rating, max_rating + 1))

            response_text, turns = await self._call_and_parse(messages, context, audio_segment, prompt)

            if response_text is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No response from judge",
                )

            self.logger.debug(f"Raw judge response: {response_text[:200]}")

            if len(turns) != num_turns:
                self.logger.warning(
                    f"[{context.record_id}] Expected {num_turns} ratings for {self.role} tts fidelity, got {len(turns)}"
                )

            for response_item in turns:
                turn_id = resolve_turn_id(response_item, tts_turn_ids, self.name)
                if turn_id is None:
                    self.logger.warning(
                        f"[{context.record_id}] Could not resolve turn ID for {response_item} turn_ids {tts_turn_ids}"
                    )
                    continue
                rating = response_item.get("rating")
                transcript = response_item.get("transcript")
                explanation = response_item.get("explanation", "")

                if rating not in valid_ratings_range:
                    self.logger.warning(f"[{context.record_id}] Invalid rating {rating} for turn {turn_id}")
                    per_turn_ratings[turn_id] = None
                    per_turn_explanations[turn_id] = f"Invalid rating: {rating}"
                    continue

                per_turn_ratings[turn_id] = rating
                per_turn_explanations[turn_id] = explanation
                per_turn_transcripts[turn_id] = transcript
                per_turn_normalized[turn_id] = normalize_rating(rating, min_rating, max_rating)

            aggregated_score = aggregate_per_turn_scores(list(per_turn_normalized.values()), self.aggregation)

            valid_ratings = [r for r in per_turn_ratings.values() if r is not None]
            avg_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else 0.0

            details: dict[str, Any] = {
                "aggregation": self.aggregation,
                "num_turns": num_turns,
                "num_evaluated": len(valid_ratings),
                "per_turn_ratings": per_turn_ratings,
                "per_turn_explanations": per_turn_explanations,
                "judge_prompt": prompt,
                "judge_raw_response": response_text,
            }
            if min_rating != 0 or max_rating != 1:
                details["per_turn_normalized"] = per_turn_normalized

            return MetricScore(
                name=self.name,
                score=round(avg_rating, 3),
                normalized_score=round(aggregated_score, 3) if aggregated_score is not None else 0,
                details=details,
                error="Aggregation failed" if aggregated_score is None else None,
            )

        except Exception as e:
            return self._handle_error(e, context)

    async def _call_and_parse(
        self,
        messages: list[dict],
        context: MetricContext,
        audio_segment: AudioSegment,
        prompt: str,
    ) -> tuple[str | None, list]:
        """Call the judge and parse the response, retrying on empty turns (transient Gemini issue).

        On "no audio" errors (Gemini drops large inline payloads), falls back to
        uploading the audio via Google's File Upload API and retrying via the
        google.genai SDK directly (bypassing litellm, which mis-transforms file_id
        messages for Gemini).
        """
        response_text = None
        used_file_upload = False
        uploaded_file = None  # google.genai File object, set on fallback upload
        try:
            for attempt in range(1 + self.max_empty_retries):
                if used_file_upload and uploaded_file is not None:
                    # Call Gemini directly via google.genai SDK — litellm's file_id
                    # message transformation is broken for Gemini (INVALID_ARGUMENT).
                    response_text = await self._generate_with_file(uploaded_file, prompt, context)
                else:
                    response_text, usage = await self.llm_client.generate_text(messages)
                    self._log_token_usage(context, self.llm_client.model, self.llm_client.params, prompt, usage)
                if response_text is None:
                    return None, []

                parsed = extract_and_load_json(response_text)
                if not isinstance(parsed, dict):
                    return response_text, []

                # Detect "no audio" — Gemini may report it at the top level OR
                # in every per-turn explanation when it drops inline audio payloads.
                explanation = parsed.get("explanation", "")
                turns = parsed.get("turns", [])
                all_turns_no_audio = turns and all(
                    "no audio" in t.get("explanation", "").lower() for t in turns if isinstance(t, dict)
                )
                if "no audio" in explanation.lower() or all_turns_no_audio:
                    no_audio_source = "top-level" if "no audio" in explanation.lower() else "per-turn"
                    if not used_file_upload:
                        self.logger.warning(
                            f"[{context.record_id}] Gemini reports no audio ({no_audio_source}) — "
                            f"falling back to Gemini File Upload API."
                        )
                        try:
                            uploaded_file = await self._upload_audio_file(audio_segment, context)
                            used_file_upload = True
                        except Exception as upload_err:
                            self.logger.error(
                                f"[{context.record_id}] File upload failed: {upload_err}. Continuing with inline retries."
                            )
                    else:
                        self.logger.warning(
                            f"[{context.record_id}] Gemini still reports no audio ({no_audio_source}) after file upload "
                            f"(attempt {attempt + 1}/{1 + self.max_empty_retries}), retrying..."
                        )
                    await asyncio.sleep(2**attempt)
                    continue

                if turns or attempt == self.max_empty_retries:
                    return response_text, turns

                self.logger.warning(
                    f"[{context.record_id}] Gemini returned empty turns (attempt {attempt + 1}/{1 + self.max_empty_retries}), retrying..."
                )
                await asyncio.sleep(2**attempt)

            return response_text, []
        finally:
            if uploaded_file is not None:
                await self._delete_uploaded_file(uploaded_file, context)

    def _trim_silence(self, audio_segment: AudioSegment, context: MetricContext) -> AudioSegment:
        """Collapse long silences in the role audio to reduce Gemini audio token cost.

        Single-channel role audio contains long stretches of silence while the other
        speaker is talking. Gemini bills audio per second of duration (~32 tokens/s),
        so trimming these silences directly cuts cost.

        Preserves enough inter-segment pause for the judge to distinguish turn boundaries
        and a tail of trailing silence so the prompt's end-of-audio cutoff leniency
        remains correctly calibrated.
        """
        segments = detect_nonsilent(
            audio_segment,
            min_silence_len=self.min_silence_len_ms,
            silence_thresh=self.silence_thresh_dbfs,
            seek_step=self.silence_seek_step_ms,
        )
        if not segments:
            self.logger.warning(f"[{context.record_id}] No speech detected in {self.role} audio — sending untrimmed.")
            return audio_segment

        pause = AudioSegment.silent(duration=self.inter_segment_pause_ms, frame_rate=audio_segment.frame_rate)
        trailing = AudioSegment.silent(duration=self.trailing_silence_ms, frame_rate=audio_segment.frame_rate)

        trimmed = AudioSegment.empty()
        for i, (start, end) in enumerate(segments):
            s = max(0, start - self.speech_padding_ms)
            e = min(len(audio_segment), end + self.speech_padding_ms)
            trimmed += audio_segment[s:e]
            if i < len(segments) - 1:
                trimmed += pause
        trimmed += trailing

        original_ms = len(audio_segment)
        trimmed_ms = len(trimmed)
        self.logger.info(
            f"[{context.record_id}] Trimmed {self.role} audio: "
            f"{original_ms / 1000:.1f}s → {trimmed_ms / 1000:.1f}s "
            f"({100 * trimmed_ms / original_ms:.0f}% of original, {len(segments)} speech segments)"
        )

        if context.output_dir and self.save_trimmed_file:
            try:
                out_path = Path(context.output_dir) / f"audio_{self.role}_trimmed.wav"
                trimmed.export(str(out_path), format="wav")
                self.logger.info(f"[{context.record_id}] Saved trimmed audio for inspection: {out_path}")
            except Exception as e:
                self.logger.warning(f"[{context.record_id}] Failed to save trimmed audio: {e}")

        return trimmed

    async def _upload_audio_file(self, audio_segment: AudioSegment, context: MetricContext):
        """Upload audio to Google's File API and poll until ACTIVE."""
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        buffer = BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)
        buffer.name = "audio.wav"

        uploaded_file = await client.aio.files.upload(
            file=buffer,
            config=genai_types.UploadFileConfig(mime_type="audio/wav"),
        )

        # Poll until ACTIVE (max ~30s)
        for _i in range(15):
            file_info = await client.aio.files.get(name=uploaded_file.name)
            state_str = file_info.state.name if hasattr(file_info.state, "name") else str(file_info.state)
            if state_str == "ACTIVE":
                break
            self.logger.debug(f"[{context.record_id}] File {uploaded_file.name} state={state_str}, waiting...")
            await asyncio.sleep(2)
        else:
            raise RuntimeError(f"File {uploaded_file.name} not ACTIVE after 30s (state={file_info.state})")

        self.logger.info(
            f"[{context.record_id}] Audio uploaded successfully "
            f"(name={uploaded_file.name}, uri={uploaded_file.uri}), retrying..."
        )
        return uploaded_file

    async def _delete_uploaded_file(self, uploaded_file, context: MetricContext) -> None:
        """Delete an uploaded file from Google's File API (best-effort)."""
        try:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            await client.aio.files.delete(name=uploaded_file.name)
            self.logger.info(f"[{context.record_id}] Deleted uploaded file {uploaded_file.name}")
        except Exception as e:
            self.logger.warning(f"[{context.record_id}] Failed to delete uploaded file {uploaded_file.name}: {e}")

    _RETRYABLE_GOOGLE_ERRORS = (
        google_exceptions.ServiceUnavailable,
        google_exceptions.TooManyRequests,
        google_exceptions.InternalServerError,
        google_exceptions.DeadlineExceeded,
        google_exceptions.ResourceExhausted,
        google_exceptions.BadGateway,
        google_exceptions.GatewayTimeout,
        ConnectionError,
        asyncio.TimeoutError,
    )

    async def _generate_with_file(
        self,
        uploaded_file,
        prompt: str,
        context: MetricContext,
        max_retries: int = 5,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 60.0,
        retry_multiplier: float = 2.0,
    ) -> str | None:
        """Call Gemini generateContent with an uploaded file reference via google.genai SDK.

        Includes retry logic with exponential backoff for transient Google API errors,
        matching the retry behaviour of LLMClient.generate_text.
        """
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model = self.llm_client.model

        for attempt in range(max_retries + 1):
            try:
                self.logger.debug(
                    f"[{context.record_id}] Calling generateContent via google.genai "
                    f"(model={model}, file_uri={uploaded_file.uri}, attempt {attempt + 1}/{max_retries + 1})"
                )

                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[uploaded_file, prompt],
                    config=genai_types.GenerateContentConfig(
                        temperature=self.llm_client.params.get("temperature", 1.0),
                        max_output_tokens=self.llm_client.params.get("max_tokens", 30000),
                    ),
                )
                return response.text

            except self._RETRYABLE_GOOGLE_ERRORS as e:
                if attempt == max_retries:
                    self.logger.error(
                        f"[{context.record_id}] google.genai generateContent failed after "
                        f"{max_retries + 1} attempts: {e}"
                    )
                    raise
                delay = min(retry_min_wait * (retry_multiplier**attempt), retry_max_wait)
                jitter = delay * 0.2 * (2 * random.random() - 1)
                delay = max(0, delay + jitter)
                self.logger.warning(
                    f"[{context.record_id}] google.genai generateContent transient error "
                    f"(attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        return None

    def _get_intended_turns(self, context: MetricContext) -> dict[int, str]:
        """Return intended turns for this metric's role."""
        return context.intended_assistant_turns if self.role == "assistant" else context.intended_user_turns

    @staticmethod
    def _format_intended_turns(intended_turns: dict[int, str]) -> str:
        """Format intended turns dictionary as numbered list."""
        return "\n".join(f"Turn {turn_id}: {text}" for turn_id, text in intended_turns.items())

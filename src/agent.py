from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
    room_io,
    TurnHandlingOptions,
    ModelSettings
)

from livekit.plugins import silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import dtln
from livekit import rtc
from typing import AsyncIterable
import numpy as np
import re
import io
import httpx
import soundfile as sf
import librosa

load_dotenv("./environment/.env.local")

# =========================================================
# CONFIG
# =========================================================
TTS_BASE_URL  = "http://192.168.134.138:8080"
TTS_VOICE     = "cahya"          # maps to default in your server
TTS_FORMAT    = "wav"
# MIN_CHUNK_CHARS = 60          # ignore chunks shorter than this
MIN_CHUNK_WORDS = 50


# =========================================================
# HELPERS
# =========================================================
async def _http_tts_chunk(text: str) -> bytes | None:
    """
    Send one text chunk to the VoxCPM2 server.
    Returns raw WAV bytes, or None if the request fails.
    """
    payload = {
        "model": "voxcpm2",
        "input": text,
        "voice": TTS_VOICE,
        "response_format": TTS_FORMAT,
    }
    try:
        async with httpx.AsyncClient(timeout=1200.0) as client:
            resp = await client.post(
                f"{TTS_BASE_URL}/v1/audio/speech",
                json=payload,
            )
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        print(f"[TTS] HTTP error for chunk '{text[:40]}…': {exc}")
        return None


def _wav_bytes_to_frames(
    wav_bytes: bytes,
    target_sample_rate: int = 48000,
) -> list[rtc.AudioFrame]:
    """
    Decode WAV bytes → list of rtc.AudioFrame (int16, mono).
    Resamples if the WAV sample rate doesn't match target_sample_rate.
    """
    buf = io.BytesIO(wav_bytes)
    audio_f32, sr = sf.read(buf, dtype="float32", always_2d=False)

    # mix to mono if needed
    if audio_f32.ndim == 2:
        audio_f32 = audio_f32.mean(axis=1)

    # resample if needed
    if sr != target_sample_rate:
        try:
            audio_f32 = librosa.resample(
                audio_f32, orig_sr=sr, target_sr=target_sample_rate
            )
            sr = target_sample_rate
        except ImportError:
            print("[TTS] librosa not available — skipping resample")

    # convert float32 → int16 PCM
    audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)

    # LiveKit wants frames of fixed size; 20 ms @ sr samples is a good default
    samples_per_frame = sr // 50  # 20 ms
    frames: list[rtc.AudioFrame] = []

    for start in range(0, len(audio_i16), samples_per_frame):
        chunk = audio_i16[start : start + samples_per_frame]
        # zero-pad the last frame if shorter
        if len(chunk) < samples_per_frame:
            chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)))
        frames.append(
            rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sr,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
        )

    return frames


# =========================================================
# ASSISTANT
# =========================================================
class Assistant(Agent):
    def __init__(self, delimiters: str = r'[.!?\n]') -> None:
        super().__init__(
            instructions="""
            Anda adalah asisten virtual Biznet yang akan membantu pelanggan anda
            """,
        )
        self.delimiter_pattern = delimiters

    # =========================================================
    # CORE FIX: chunked TTS node
    # =========================================================
    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        """
        Splits the LLM text stream into sentence-level chunks,
        fires one HTTP TTS request per chunk, and yields AudioFrames
        as each chunk's audio arrives — so playback starts with
        chunk 1 while chunk 2 is still being generated.
        """

        pattern = self.delimiter_pattern

        # async def _sentence_chunks() -> AsyncIterable[str]:
        #     """Buffer incoming tokens and yield complete sentences."""
        #     buffer = ""
        #     async for token in text:
        #         buffer += token
        #         while True:
        #             match = re.search(pattern, buffer)
        #             if not match:
        #                 break
        #             chunk = buffer[: match.end()].strip()
        #             buffer = buffer[match.end():]
        #             if len(chunk) >= MIN_CHUNK_CHARS:
        #                 yield chunk
        #             else:
        #                 # too short — prepend back to buffer, wait for more
        #                 buffer = chunk + " " + buffer
        #                 break

        #     # flush remainder
        #     remainder = buffer.strip()
        #     if remainder and len(remainder) >= 2:
        #         yield remainder



        # async def _sentence_chunks() -> AsyncIterable[str]:
        #     buffer = ""
        #     accumulated = ""

        #     async for token in text:
        #         buffer += token
        #         while True:
        #             match = re.search(pattern, buffer)
        #             if not match:
        #                 break
        #             chunk = buffer[: match.end()].strip()
        #             buffer = buffer[match.end():]

        #             if not chunk:
        #                 break

        #             accumulated = (accumulated + " " + chunk).strip() if accumulated else chunk

        #             # only yield when accumulated is long enough
        #             if len(accumulated) >= MIN_CHUNK_CHARS:
        #                 yield accumulated
        #                 accumulated = ""

        #     # flush whatever remains
        #     if accumulated:
        #         remainder = (accumulated + " " + buffer.strip()).strip()
        #     else:
        #         remainder = buffer.strip()

        #     if remainder and len(remainder) >= 2:
        #         yield remainder



        async def _sentence_chunks() -> AsyncIterable[str]:
            buffer = ""
            accumulated = ""

            # only split on . ! ? followed by space/end, or newline
            # avoids splitting on 385.000 or decimals
            sentence_end = re.compile(r'[.!?](?=\s|$)|\n')

            async for token in text:
                buffer += token
                while True:
                    match = sentence_end.search(buffer)
                    if not match:
                        break

                    chunk = buffer[: match.end()].strip()
                    buffer = buffer[match.end():]

                    if not chunk:
                        break

                    # always accumulate full sentences first
                    accumulated = (accumulated + " " + chunk).strip() if accumulated else chunk

                    # only yield when we have enough WORDS
                    # this guarantees we never cut mid-sentence
                    word_count = len(accumulated.split())
                    if word_count >= MIN_CHUNK_WORDS:
                        yield accumulated
                        accumulated = ""

                        print(f"word counted : {word_count}, accumlated: {accumulated}")

            # flush remainder
            remainder = (accumulated + " " + buffer.strip()).strip()
            if remainder and len(remainder) >= 2:
                yield remainder

        # -------------------------------------------------------
        # For each sentence chunk: TTS → decode → yield frames
        # -------------------------------------------------------
        async for sentence in _sentence_chunks():
            print(f"[TTS] synthesising chunk: '{sentence}'")

            wav_bytes = await _http_tts_chunk(sentence)
            if not wav_bytes:
                print(f"[TTS] skipping chunk (no audio returned)")
                continue

            frames = _wav_bytes_to_frames(wav_bytes)
            print(f"[TTS] yielding {len(frames)} frames for chunk")

            for frame in frames:
                yield frame

    # =====================================================
    # USER TURN
    # =====================================================
    async def on_user_turn_completed(
        self,
        turn_ctx,
        new_message,
    ):
        # content = new_message.content
        content = """Baik, saya akan sebutkan nilai-nilai Biznet singkat ya.\n
            Hasrat Terpadu: bekerja dengan sepenuh hasrat dan suka cita untuk menciptakan produk inovatif dan layanan terpadu.\n
            Layanan Sepenuh Hati: melayani pelanggan dengan sepenuh hati untuk memenuhi kebutuhan layanan yang cepat, terpercaya, dan terjangkau.\n
            Semangat untuk Maju: bekerja cerdas dan belajar dari tantangan agar maju dan berkembang bersama.\n
            Mau saya jelaskan salah satu nilai lebih detail atau ada pertanyaan lain?
            Baik, saya akan sebutkan nilai-nilai Biznet singkat ya.\n
            Hasrat Terpadu: bekerja dengan sepenuh hasrat dan suka cita untuk menciptakan produk inovatif dan layanan terpadu.\n
            Layanan Sepenuh Hati: melayani pelanggan dengan sepenuh hati untuk memenuhi kebutuhan layanan yang cepat, terpercaya, dan terjangkau.\n
            Semangat untuk Maju: bekerja cerdas dan belajar dari tantangan agar maju dan berkembang bersama.\n
            Mau saya jelaskan salah satu nilai lebih detail atau ada pertanyaan lain?
        """

        if isinstance(content, str):
            user_text = content
        elif isinstance(content, list):
            user_text = " ".join(
                part.text if hasattr(part, "text") else str(part)
                for part in content
                if (
                    not hasattr(part, "type")
                    or getattr(part, "type", "") == "text"
                )
            )
        else:
            user_text = str(content)

        user_text = user_text.strip()
        print(f">>> user said: '{user_text}'")

        if not user_text or len(user_text) < 2:
            return

        user_text = re.sub(r"\s+", " ", user_text)

        reply = f"Anda mengucapkan: {user_text}"
        if reply[-1] not in ".!?":
            reply += "."

        await self.session.say(
            reply,
            allow_interruptions=False,
        )


# =========================================================
# SERVER
# =========================================================
server = AgentServer()


# =========================================================
# RTC SESSION
# =========================================================
@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):

    session = AgentSession(

        # =================================================
        # STT
        # =================================================
        stt=openai.STT(
            model="qwen3-asr",
            base_url="http://192.168.134.138:6666/v1",
            api_key="empty",
            language="id",
        ),

        # =================================================
        # TTS
        # =================================================
        # NOTE: still required by AgentSession constructor,
        # but our custom tts_node above bypasses it for LLM replies.
        # It IS still used by session.say() — so keep it pointed
        # at your server; short say() strings are fine for it.
        tts=openai.TTS(
            model="tts-1",
            base_url="http://192.168.134.138:8080/v1",
            api_key="empty",
            response_format="wav",
        ),

        # =================================================
        # VAD
        # =================================================
        vad=silero.VAD.load(
            activation_threshold=0.3,
            min_speech_duration=0.25,
            min_silence_duration=0.45,
            padding_duration=0.15,
        ),

        # =================================================
        # TURN HANDLING
        # =================================================
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            min_endpointing_delay=0.30,
            preemptive_synthesis=False,
        ),
    )

    # =====================================================
    # START SESSION
    # =====================================================
    await session.start(
        room=ctx.room,
        agent=Assistant(
            delimiters=r'[.!?\n]'
        ),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=dtln.noise_suppression(
                    strength=0.50,
                )
            ),
            audio_output=room_io.AudioOutputOptions(
                sample_rate=48000
            ),
        ),
    )

    # =====================================================
    # GREETING
    # =====================================================
    await session.say(
        "Saya virtual asisten Biznet, apakah ada yang bisa saya bantu?",
        allow_interruptions=False,
    )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    agents.cli.run_app(server)
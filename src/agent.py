from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent, room_io, TurnHandlingOptions
from livekit.plugins import silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit import rtc
from livekit.plugins import dtln
from openai import AsyncOpenAI
import numpy as np
import httpx

load_dotenv("./environment/.env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""Anda adalah asisten virtual yang akan membantu pelanggan anda.
            TENTANG BIZNET:
            - Biznet adalah penyedia layanan internet serat optik terbesar di Indonesia
            - Produk: Biznet Home (perumahan), Biznet Metronet (bisnis), Biznet Networks
            - Kecepatan internet: 50Mbps, 100Mbps, 300Mbps, 500Mbps, 1Gbps
            - Jangkauan: Jawa, Bali, Sumatra
            - Dukungan: 021-5714 2888, email support@biznet.id
            - Situs web: biznet.id

            PRODUK:
            - Biznet Home 50Mbps: 250.000/bulan (Terkecil)
            - Biznet Home 100Mbps: 385.000/bulan  
            - Biznet Home 300Mbps: 550.000/bulan
            - Biznet Home 500Mbps: 750.000/bulan
            - Biznet Home 1Gbps: 1.100.000/bulan (Termurah)

            Jawaban harus SINGKAT, JELAS, maksimal 2 kalimat.
            """,
        )

    # async def stt_node(self, audio, model_settings):
    #     async def filtered_audio():
    #         async for frame in audio:
    #             if isinstance(frame, rtc.AudioFrame):
    #                 samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
    #                 rms = np.sqrt(np.mean(samples ** 2))
                    
    #                 # Log RMS with visual bar
    #                 bar_length = int(rms / 200)  # scale to visible range
    #                 bar = "|||" * min(bar_length, 40)
    #                 print(f"RMS: {rms:6.1f} |{bar:<40}| {'PASS' if rms > 500 else 'DROP'}")
                    
    #                 if rms > 800:
    #                     yield frame
    #             else:
    #                 yield frame

    #     async for result in super().stt_node(filtered_audio(), model_settings):
    #         yield result

    async def on_user_turn_completed(self, turn_ctx, new_message):
    # ChatMessage.content can be a string or a list of content parts
        content = new_message.content
        
        if isinstance(content, str):
            user_text = content
        elif isinstance(content, list):
            # extract text from content parts
            user_text = " ".join(
                part.text if hasattr(part, 'text') else str(part)
                for part in content
                if not hasattr(part, 'type') or getattr(part, 'type', '') == 'text'
            )
        else:
            user_text = str(content)
        
        user_text = user_text.strip()
        
        print(f">>> user said: '{user_text}'")
        
        if not user_text or len(user_text) < 2:
            return
        
        await self.session.say(f"Anda mengucapkan: {user_text}")


server = AgentServer()


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):

    # custom_http_client = httpx.AsyncClient(
    #     verify=False, 
    #     proxy=None,       # Ignore any proxy servers
    #     trust_env=False     # Ignore HTTP_PROXY from your .env.local
    # )

    # tts_client = AsyncOpenAI(
    #     base_url='http://192.168.134.138:8000/v1', # Ensure this is http://
    #     api_key='empty',
    #     http_client=custom_http_client
    # )

    session = AgentSession(
        stt=openai.STT(
            model='large-fast',
            # base_url='http://192.168.134.190:5001/v1',
            base_url='http://192.168.134.190:7777/v1',
            api_key='empty',
            language= 'id'
        ),
        tts=openai.TTS(
            model='tts-1',
            # base_url='http://192.168.134.138:8000/v1',
            base_url='http://127.0.0.1:8000/v1',
            api_key='empty',
            response_format='wav',
            # client=tts_client
            # voice='alloy'
        ),

    vad=silero.VAD.load(
        activation_threshold=0.75,
        min_speech_duration=0.3,
        min_silence_duration=0.2,   # ← cut to 200ms, was longer
        padding_duration=0.05,      # ← minimal padding
    ),

    turn_handling=TurnHandlingOptions(
        turn_detection=MultilingualModel(),
        min_endpointing_delay=0.0,  # ← no extra delay
    ),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # noise_cancellation=dtln.noise_suppression(
                #     debug_logging=True,
                #     strength= 0.0
                # )
            ),
        ),
    )

    # Use say() instead of generate_reply() — no LLM needed
    await session.say("Halo, saya virtual asisten Biznut. Apakah ada yang bisa saya bantu?")


if __name__ == "__main__":
    agents.cli.run_app(server)
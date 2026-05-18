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
        # content = new_message.content
        content = """
        Baik, saya jelaskan semua poin utama paket Biznet Home di Denpasar, Bali secara singkat ya:

        HOME 0D: kecepatan seratus Mbps, kuota utama seribu lima ratus GB, bonus tiga ratus tujuh puluh lima GB, ideal untuk satu sampai tiga perangkat, setelah FUP turun ke lima Mbps, tipe koneksi dynamic private, harga dua ratus lima puluh ribu rupiah per bulan (sebelum pajak).
        HOME 1D: kecepatan tiga ratus Mbps, kuota utama empat ribu GB, bonus seribu GB, ideal untuk satu sampai sepuluh perangkat, setelah FUP turun ke lima belas Mbps, tipe dynamic private, harga tiga ratus tujuh puluh lima ribu rupiah per bulan (sebelum pajak).
        HOME 2D: kecepatan empat ratus Mbps, kuota utama delapan ribu GB, bonus dua ribu GB, ideal untuk sebelas sampai dua puluh perangkat, setelah FUP turun ke dua puluh lima Mbps, tipe dynamic private, harga lima ratus tujuh puluh lima ribu rupiah per bulan (sebelum pajak).
        GAMERS 3D: kecepatan lima ratus Mbps, kuota utama sepuluh ribu GB, bonus dua ribu lima ratus GB, ideal untuk dua puluh satu sampai empat puluh perangkat, setelah FUP turun ke tiga puluh Mbps, tipe dynamic public, harga tujuh ratus ribu rupiah per bulan (sebelum pajak).
        Tambahan layanan dan info singkat:

        Biznet IPTV tersedia dengan set top box, biaya perangkat dan langganan terpisah.
        Mesh WiFi TP‑Link Deco tersedia sebagai add-on untuk menghilangkan dead zone.
        Pelanggan bisa sewa atau beli modem; ada biaya instalasi awal.
        Semua harga belum termasuk pajak dan bisa berubah sesuai promo.
        Mau saya cek ketersediaan pemasangan di alamat Raja Pasar, Denpasar sekarang atau bantu daftar pemasangan?
        """
        
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
            base_url='http://192.168.134.138:6666/v1',
            api_key='empty',
            language= 'id'
        ),
        tts=openai.TTS(
            model='tts-1',
            # base_url='http://192.168.134.138:8000/v1',
            base_url='http://192.168.134.138:8080/v1',
            api_key='empty',
            response_format='wav',
            # voice='pace_kobo_clean.wav'
            # client=tts_client
            # voice='alloy'
        ),

    vad=silero.VAD.load(
        activation_threshold=0.3,
        min_speech_duration=0.15,
        min_silence_duration=0.45,   # ← cut to 200ms, was longer
        padding_duration=0.05,      # ← minimal padding
        sample_rate= 16000
    ),

    # turn_handling=TurnHandlingOptions(
    #     turn_detection=MultilingualModel(),
    #     min_endpointing_delay=0.0,  # ← no extra delay
    # ),

    turn_handling=TurnHandlingOptions(
        turn_detection=MultilingualModel(),
        min_endpointing_delay=0.30,
        preemptive_synthesis=False,
    ),
    
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=dtln.noise_suppression(
                    strength=0.50
                ),
                sample_rate= 16000
            ),
            audio_output=room_io.AudioOutputOptions(
                sample_rate= 48000
            ),
        ),
    )

    # Use say() instead of generate_reply() — no LLM needed
    await session.say("Halo, saya virtual asisten Biznut. Apakah ada yang bisa saya bantu?", allow_interruptions= False)


if __name__ == "__main__":
    agents.cli.run_app(server)
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins.nvidia.experimental import realtime as nvidia_realtime
from livekit.plugins import dtln
import aiohttp, ssl


load_dotenv("./environment/.env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice assistant for Biznet, an Indonesian internet service provider. Please explain clearly.

            ABOUT BIZNET:
            - Biznet is Indonesia's largest fiber optic internet provider
            - Products: Biznet Home (residential), Biznet Metronet (business), Biznet Networks
            - Internet speeds: 50Mbps, 100Mbps, 300Mbps, 500Mbps, 1Gbps
            - Coverage: Java, Bali, Sumatra
            - Support: 021-5714 2888, email support@biznet.id
            - Website: biznet.id

            PRODUCTS:
            - Biznet Home 50Mbps: 250.000/month (Slowest)
            - Biznet Home 100Mbps: 385.000/month  
            - Biznet Home 300Mbps: 550.000/month
            - Biznet Home 500Mbps: 750.000/month
            - Biznet Home 1Gbps: 1.100.000/month (Cheapest)

            Keep responses SHORT, CONCISE, max 2 sentences.""",

            # instructions= """
            # Kamu adalah asisten AI yang cerdas dan ramah.

            # ATURAN BAHASA (WAJIB DIIKUTI):
            # - SELALU gunakan Bahasa Indonesia dalam setiap respons.
            # - TIDAK BOLEH menggunakan bahasa Inggris atau bahasa lain, dalam kondisi apapun.
            # - Jika pengguna berbicara dalam bahasa lain, tetap jawab dalam Bahasa Indonesia.
            # - Gunakan Bahasa Indonesia yang natural, sopan, dan mudah dipahami.
            # - Hindari kata-kata asing yang tidak perlu; gunakan padanan Bahasa Indonesia.

            # GAYA BERBICARA:
            # - Bicara dengan santai tapi tetap sopan.
            # - Jawaban singkat dan jelas, sesuai konteks percakapan suara.
            # - Gunakan kata sapaan seperti "Halo", "Baik", "Tentu saja", "Dengan senang hati".
            # """
        )

server = AgentServer()


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    http_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_ctx)
    )
    session = AgentSession(
        llm=nvidia_realtime.RealtimeModel(
            # base_url="wss://192.168.134.138:8998",
            base_url="wss://127.0.0.1:8998",
            voice="NATF2",
            silence_threshold_ms= 500,
            http_session= http_session
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=dtln.noise_suppression(),
            )
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
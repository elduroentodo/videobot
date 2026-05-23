import os
import uuid
import subprocess
import requests
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_ASR_URL = f"{GROQ_BASE}/audio/transcriptions"
GROQ_CHAT_URL = f"{GROQ_BASE}/chat/completions"

ASR_MODEL = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3-turbo")
LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

MAX_MB = 24.5

if not TELEGRAM_TOKEN:
    raise RuntimeError("Falta TELEGRAM_TOKEN en variables de entorno.")

if not GROQ_API_KEY:
    raise RuntimeError("Falta GROQ_API_KEY en variables de entorno.")


def ffprobe_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        return 0.0

    try:
        return float(result.stdout.decode().strip())
    except Exception:
        return 0.0


def extract_audio_mp3(video_path: str, audio_path: str, bitrate_k: int = 64):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        f"{bitrate_k}k",
        "-f",
        "mp3",
        audio_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Error con FFmpeg: {error}")


def split_audio_mp3(audio_path: str, chunk_seconds: int = 180) -> list:
    output_pattern = f"/tmp/chunk_{uuid.uuid4()}_%03d.mp3"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-c",
        "copy",
        output_pattern,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Error dividiendo audio: {error}")

    prefix = os.path.basename(output_pattern.replace("%03d.mp3", ""))

    chunks = sorted(
        [
            os.path.join("/tmp", file)
            for file in os.listdir("/tmp")
            if file.startswith(prefix) and file.endswith(".mp3")
        ]
    )

    return chunks


def groq_transcribe(audio_path: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }

    with open(audio_path, "rb") as file:
        files = {
            "file": (os.path.basename(audio_path), file, "audio/mpeg"),
        }

        data = {
            "model": ASR_MODEL,
            "language": "es",
            "response_format": "text",
        }

        response = requests.post(
            GROQ_ASR_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=240,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Error en transcripción Groq {response.status_code}: {response.text}"
        )

    return response.text.strip()


def groq_generate_copy(transcript: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""
Actúa como experto en marketing digital, storytelling y copywriting para redes sociales.

Con base en esta transcripción de video, entrega:

1. Resumen breve del contenido.
2. Idea principal.
3. Copy para Instagram con emojis y CTA.
4. Copy para TikTok con hook fuerte y CTA.
5. Copy para LinkedIn con tono profesional.
6. Cinco hooks alternativos.
7. Diez hashtags relevantes.
8. Caption ultra corto de una línea.
9. Recomendación de título para el video.

Reglas:
- Responde en español.
- No inventes datos que no estén en la transcripción.
- Si el contenido es educativo, conviértelo en contenido de valor.
- Si el contenido vende algo, enfócate en beneficios, objeciones y llamada a la acción.
- Mantén el resultado claro, usable y listo para publicar.

Transcripción:
{transcript}
""".strip()

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Eres un copywriter senior experto en redes sociales.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.7,
    }

    response = requests.post(
        GROQ_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Error generando copy Groq {response.status_code}: {response.text}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def chunk_text(text: str, limit: int = 3500):
    for i in range(0, len(text), limit):
        yield text[i : i + limit]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hola. Envíame un video de 2 a 10 minutos y te devolveré:\n\n"
        "✅ Transcripción\n"
        "✅ Copy para Instagram\n"
        "✅ Copy para TikTok\n"
        "✅ Copy para LinkedIn\n"
        "✅ Hooks\n"
        "✅ Hashtags\n"
        "✅ CTA\n\n"
        "Tip: mientras más claro sea el audio, mejor será el resultado."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Instrucciones:\n\n"
        "1. Envía un video directamente al bot.\n"
        "2. Espera mientras extraigo el audio.\n"
        "3. Luego transcribo el contenido.\n"
        "4. Finalmente genero copys para redes sociales.\n\n"
        "Recomendación: videos entre 2 y 10 minutos."
    )


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📥 Video recibido. Descargando...")

    video = update.message.video or update.message.document

    if not video:
        await update.message.reply_text("Por favor envíame un archivo de video válido.")
        return

    telegram_file = await video.get_file()

    uid = str(uuid.uuid4())
    video_path = f"/tmp/{uid}.mp4"
    audio_path = f"/tmp/{uid}.mp3"

    chunk_paths = []

    try:
        await telegram_file.download_to_drive(video_path)

        await update.message.reply_text("🎧 Extrayendo y comprimiendo audio...")

        extract_audio_mp3(video_path, audio_path, bitrate_k=64)

        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        duration = ffprobe_duration_seconds(audio_path)

        await update.message.reply_text(
            f"📊 Audio preparado: {duration:.1f} segundos, {size_mb:.2f} MB."
        )

        if size_mb > MAX_MB:
            await update.message.reply_text(
                "⚠️ El audio quedó grande. Lo dividiré en partes para procesarlo."
            )
            chunk_paths = split_audio_mp3(audio_path, chunk_seconds=180)
        else:
            chunk_paths = [audio_path]

        await update.message.reply_text("📝 Transcribiendo con IA...")

        transcripts = []

        for index, chunk in enumerate(chunk_paths, start=1):
            await update.message.reply_text(f"Procesando parte {index}/{len(chunk_paths)}...")
            part_text = groq_transcribe(chunk)
            transcripts.append(f"[Parte {index}]\n{part_text}")

        full_transcript = "\n\n".join(transcripts).strip()

        if not full_transcript:
            await update.message.reply_text("No pude detectar voz clara en el video.")
            return

        await update.message.reply_text("✍️ Generando copys...")

        copy_result = groq_generate_copy(full_transcript)

        final_response = f"""
🧾 TRANSCRIPCIÓN

{full_transcript}

==============================

🚀 COPYS GENERADOS

{copy_result}
""".strip()

        for part in chunk_text(final_response):
            await update.message.reply_text(part)

    except Exception as error:
        await update.message.reply_text(f"❌ Error: {error}")

    finally:
        files_to_delete = [video_path, audio_path]

        for chunk in chunk_paths:
            if chunk != audio_path:
                files_to_delete.append(chunk)

        for file_path in files_to_delete:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Envíame un video y generaré la transcripción y los copys."
    )


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.Document.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot ejecutándose...")
    app.run_polling()


if __name__ == "__main__":
    main()

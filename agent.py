"""
LiveKit STT Agent — faster-whisper (CPU, бесплатно)
Транскрибирует каждого участника отдельно и шлёт текст через DataChannel.

Автоопределение языка:
- По умолчанию язык не задан — Whisper определяет сам на каждом чанке.
- Для каждого участника ведётся "языковая память" (скользящее окно вероятностей).
- Если последние LANG_LOCK_CHUNKS чанков уверенно один язык (>= LANG_LOCK_THRESHOLD)
  — он фиксируется для экономии времени детекции.
- condition_on_previous_text=False — модель не застревает в языке при code-switching.
- FORCE_LANGUAGE=ru/en/... — принудительно задать язык через env (отключает авто).
"""

import asyncio
import json
import logging
import os
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-agent")

# ── Настройки модели ───────────────────────────────────────────────────────────
# tiny  — быстрее всего, хуже качество
# base  — баланс скорости и качества (рекомендуется)
# small — лучше, но медленнее на CPU
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Оставь пустым для автоопределения (рекомендуется).
# Установи "ru", "en" и т.д. чтобы принудительно зафиксировать язык.
FORCE_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "") or None

# Накапливаем аудио кусками по N секунд, потом транскрибируем
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "3"))
SAMPLE_RATE = 16000  # Whisper всегда хочет 16kHz

# ── Настройки языковой памяти ──────────────────────────────────────────────────
# Сколько последних чанков анализировать для определения "доминирующего" языка
LANG_WINDOW = int(os.getenv("LANG_WINDOW", "4"))
# Если средняя вероятность языка >= порога — фиксируем его (не тратим время на детекцию)
LANG_LOCK_THRESHOLD = float(os.getenv("LANG_LOCK_THRESHOLD", "0.85"))
# Сколько подряд идущих чанков должны подтвердить язык перед фиксацией
LANG_LOCK_CHUNKS = int(os.getenv("LANG_LOCK_CHUNKS", "3"))
# Если вероятность зафиксированного языка упала ниже — снова переходим в авто
LANG_UNLOCK_THRESHOLD = float(os.getenv("LANG_UNLOCK_THRESHOLD", "0.60"))

# Глобальный конспект комнаты: { participant_identity: [{"time":..., "text":...}] }
session_transcript: dict[str, list] = defaultdict(list)


def load_model() -> WhisperModel:
    logger.info(f"Загружаем Whisper модель '{WHISPER_MODEL}' (CPU)...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    logger.info("Модель загружена ✓")
    return model


# Загружаем один раз при старте воркера
whisper = load_model()


# ── Языковая память участника ──────────────────────────────────────────────────

class LanguageTracker:
    """
    Скользящее окно вероятностей языков для одного участника.
    Автоматически фиксирует язык когда уверенность стабильно высокая,
    и разблокирует при смене языка (code-switching).
    """

    def __init__(self):
        self.window: deque[tuple[str, float]] = deque(maxlen=LANG_WINDOW)
        self.locked_lang: str | None = None
        self.lock_streak: int = 0

    def update(self, detected_lang: str, probability: float) -> str | None:
        """
        Обновляет статистику и возвращает язык для следующего чанка.
        None = оставить авто-детекцию.
        """
        self.window.append((detected_lang, probability))

        if self.locked_lang:
            # Проверяем — не сменился ли язык
            if probability < LANG_UNLOCK_THRESHOLD or detected_lang != self.locked_lang:
                logger.info(
                    f"Смена языка: {self.locked_lang} → {detected_lang} "
                    f"(уверенность {probability:.2f}), переходим в авто"
                )
                self.locked_lang = None
                self.lock_streak = 0
            return self.locked_lang  # None если только что разблокировали

        # Накапливаем стрик для фиксации
        if detected_lang == (self.window[-2][0] if len(self.window) >= 2 else detected_lang):
            self.lock_streak += 1
        else:
            self.lock_streak = 1

        avg_prob = sum(p for _, p in self.window) / len(self.window)
        if self.lock_streak >= LANG_LOCK_CHUNKS and avg_prob >= LANG_LOCK_THRESHOLD:
            self.locked_lang = detected_lang
            logger.info(
                f"Язык зафиксирован: {detected_lang} "
                f"(средняя уверенность {avg_prob:.2f})"
            )

        return None  # пока не зафиксирован — авто


# ── Транскрипция ───────────────────────────────────────────────────────────────

def transcribe_chunk(
    audio_data: np.ndarray,
    tracker: LanguageTracker,
) -> tuple[str, str | None]:
    """
    Транскрибирует numpy float32 16kHz.
    Возвращает (text, detected_language).
    """
    if len(audio_data) < SAMPLE_RATE * 0.3:
        return "", None

    # Если язык принудительно задан — используем его всегда
    lang_hint = FORCE_LANGUAGE or tracker.locked_lang

    segments, info = whisper.transcribe(
        audio_data,
        language=lang_hint,           # None = авто-детекция Whisper
        beam_size=1,                  # быстрее на CPU
        vad_filter=True,              # фильтрует тишину
        vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False,  # не застревать в языке при code-switching
    )

    text = " ".join(s.text.strip() for s in segments).strip()
    detected = info.language          # язык определённый Whisper
    prob = info.language_probability  # вероятность (0.0 – 1.0)

    # Обновляем трекер только если языка не было задан принудительно
    if not FORCE_LANGUAGE and detected:
        tracker.update(detected, prob)
        if text:
            logger.debug(f"Язык: {detected} ({prob:.2f}), locked={tracker.locked_lang}")

    return text, detected


# ── Обработка аудио участника ──────────────────────────────────────────────────

async def transcribe_participant_audio(
    audio_stream: rtc.AudioStream,
    participant: rtc.RemoteParticipant,
    room: rtc.Room,
):
    identity = participant.identity
    role = participant.metadata or "participant"
    logger.info(f"Начинаем транскрипцию для {identity} (роль: {role})")

    tracker = LanguageTracker()
    buffer: list[np.ndarray] = []
    buffer_samples = 0
    target_samples = int(SAMPLE_RATE * CHUNK_SECONDS)

    async for event in audio_stream:
        frame = event.frame

        # LiveKit отдаёт int16 PCM → конвертируем в float32 для Whisper
        pcm_int16 = np.frombuffer(frame.data, dtype=np.int16)
        pcm_float32 = pcm_int16.astype(np.float32) / 32768.0

        # Ресемплируем если нужно (LiveKit обычно 48kHz)
        if frame.sample_rate != SAMPLE_RATE:
            ratio = SAMPLE_RATE / frame.sample_rate
            new_len = int(len(pcm_float32) * ratio)
            pcm_float32 = np.interp(
                np.linspace(0, len(pcm_float32), new_len),
                np.arange(len(pcm_float32)),
                pcm_float32,
            )

        buffer.append(pcm_float32)
        buffer_samples += len(pcm_float32)

        if buffer_samples >= target_samples:
            chunk = np.concatenate(buffer)
            buffer = []
            buffer_samples = 0

            text, lang = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_chunk, chunk, tracker
            )

            if not text:
                continue

            logger.info(f"[{identity}] ({lang}): {text}")

            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "participant": identity,
                "role": role,
                "text": text,
                "lang": lang,  # язык чанка — фронтенд может показать флаг
            }
            session_transcript[identity].append(entry)

            payload = json.dumps(
                {"type": "transcript_chunk", **entry},
                ensure_ascii=False,
            ).encode()

            await room.local_participant.publish_data(payload, reliable=True)


# ── Финальный конспект ─────────────────────────────────────────────────────────

def build_final_transcript() -> str:
    all_entries = [e for entries in session_transcript.values() for e in entries]
    all_entries.sort(key=lambda e: e["time"])

    lines = ["📝 КОНСПЕКТ", "=" * 40]
    for e in all_entries:
        role_label = {"tutor": "Репетитор", "student": "Ученик"}.get(e["role"], e["role"])
        lang_tag = f"[{e.get('lang', '?')}] " if e.get("lang") else ""
        lines.append(f"[{e['time']}] {e['participant']} ({role_label}): {lang_tag}{e['text']}")

    return "\n".join(lines)


# ── LiveKit Agent entrypoint ───────────────────────────────────────────────────

async def entrypoint(ctx: JobContext):
    logger.info(f"Агент подключается к комнате: {ctx.room.name}")
    await ctx.connect()

    room = ctx.room
    active_streams: dict[str, asyncio.Task] = {}

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        logger.info(f"Новый аудиотрек от {participant.identity}")
        audio_stream = rtc.AudioStream(track)
        task = asyncio.ensure_future(
            transcribe_participant_audio(audio_stream, participant, room)
        )
        active_streams[participant.identity] = task

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        identity = participant.identity
        if identity in active_streams:
            active_streams[identity].cancel()
            del active_streams[identity]
            logger.info(f"Остановили транскрипцию для {identity}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        identity = participant.identity
        entries = session_transcript.get(identity, [])
        if entries:
            asyncio.ensure_future(
                room.local_participant.publish_data(
                    json.dumps({
                        "type": "participant_summary",
                        "participant": identity,
                        "entries": entries,
                    }, ensure_ascii=False).encode(),
                    reliable=True,
                )
            )

    @room.on("disconnected")
    def on_room_disconnected():
        final = build_final_transcript()
        logger.info("Итоговый конспект:\n" + final)
        asyncio.ensure_future(
            room.local_participant.publish_data(
                json.dumps({
                    "type": "session_transcript",
                    "transcript": final,
                    "entries": [e for entries in session_transcript.values() for e in entries],
                }, ensure_ascii=False).encode(),
                reliable=True,
            )
        )

    await asyncio.sleep(float("inf"))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

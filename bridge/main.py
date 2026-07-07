from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from kafka import KafkaConsumer



def main():
    print("Hello from bridge!")


if __name__ == "__main__":
    main()




"""
FastAPI SSE bridge between Kafka and the React dashboard.

One background thread consumes 4 Kafka topics into per-client asyncio queues.
Each browser client that connects to /stream gets its own queue and receives
messages as they arrive, formatted as Server-Sent Events.
"""


# ------------------------------------------------------------ config
KAFKA_BOOTSTRAP = "localhost:9092"
TOPICS = ["predictions", "coordinates_nd", "shift_events",
          "model_update_complete"]
QUEUE_MAXSIZE = 500      # backpressure cap per client
KEEPALIVE_SECS = 15      # SSE comment sent when idle to keep proxies happy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("bridge")

# ------------------------------------------------------------ shared state
# Registry of connected clients: id → asyncio.Queue
_clients: dict[str, asyncio.Queue] = {}
_clients_lock = threading.Lock()
_main_loop: asyncio.AbstractEventLoop | None = None


def _kafka_loop():
    """Background thread: consume Kafka, fan out to all connected clients."""
    consumer = KafkaConsumer(
        *TOPICS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=f"bridge_{uuid.uuid4().hex[:8]}",  # ephemeral group
        value_deserializer=lambda v: v.decode("utf-8"),
        auto_offset_reset="latest",  # live tail; don't replay history to browsers
        enable_auto_commit=False,
    )
    log.info("kafka consumer joined; subscribed to %s", TOPICS)

    for msg in consumer:
        payload = json.dumps({"topic": msg.topic, "data": json.loads(msg.value)})
        with _clients_lock:
            targets = list(_clients.items())
        for cid, q in targets:
            # Non-blocking put; drop for slow clients rather than backing up Kafka.
            try:
                if _main_loop is not None:
                    _main_loop.call_soon_threadsafe(_try_put, q, payload)
            except Exception:
                log.exception("failed to enqueue for client %s", cid)


def _try_put(q: asyncio.Queue, payload: str) -> None:
    try:
        q.put_nowait(payload)
    except asyncio.QueueFull:
        # Slow client — drop the message. Their tab will just show slightly stale data.
        pass


# ------------------------------------------------------------ FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    thread = threading.Thread(target=_kafka_loop, daemon=True)
    thread.start()
    log.info("bridge started")
    yield
    log.info("bridge shutting down")


app = FastAPI(title="MetroPT SSE Bridge", lifespan=lifespan)

# Allow the React dev server (port 3000) to connect during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    with _clients_lock:
        n = len(_clients)
    return {"status": "ok", "connected_clients": n}


@app.get("/stream")
async def stream():
    """Server-Sent Events endpoint. Each client gets its own queue."""
    cid = uuid.uuid4().hex[:8]
    q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    with _clients_lock:
        _clients[cid] = q
    log.info("client %s connected (total=%d)", cid, len(_clients))

    async def event_generator():
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=KEEPALIVE_SECS)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # SSE comment (starts with ':') — keeps proxies from timing out.
                    yield ": keepalive\n\n"
        finally:
            with _clients_lock:
                _clients.pop(cid, None)
            log.info("client %s disconnected (total=%d)", cid, len(_clients))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering, harmless otherwise
        },
    )
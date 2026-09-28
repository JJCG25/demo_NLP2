// --- CONFIGURACIÓN ---
// The model no longer runs in the browser. It runs on the T4 box under vLLM,
// which exposes an OpenAI-compatible endpoint. Point this at the SSH tunnel
// (ssh -L 8000:localhost:<port> user@eisi) so the browser only ever talks to
// localhost: no CORS, no exposed port.
const API_BASE = "http://localhost:8000/v1";
const MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ";

// Latency knobs. MAX_SIDE is the strongest one by far: every pixel becomes
// vision tokens the T4s have to chew through, so downscaling the frame before
// it leaves the browser cuts prefill time more than anything else here.
const MAX_SIDE = 768;
const JPEG_QUALITY = 0.8;

// Room to finish a sentence. Decoding is token-by-token, so this is a direct
// latency budget: at ~10 tok/s every 10 extra tokens costs about a second. The
// prompt asks for two sentences, so this is headroom, not a target.
const MAX_TOKENS = 128;

// Nobody at a stand types on a stranger's laptop, so these carry the demo. The
// last one shows off OCR, which is where a 32B model most obviously beats a
// small one.
const PRESETS = [
    ["👤", "¿Qué llevo puesto?"],
    ["🔢", "¿Cuántas personas hay?"],
    ["🔍", "¿Qué objeto tengo en la mano?"],
    ["📖", "Lee el texto que aparece en la imagen."],
    ["🖼️", "Describe la escena."],
];

// Same model, different system prompt. Cheap to add and it makes people try the
// demo several times instead of once. Temperature varies on purpose: a factual
// description wants 0, a poem wants room to invent.
const MODES = [
    {
        id: "descriptivo", icon: "🔍", label: "Descriptivo",
        question: "¿Qué ves?",
        temperature: 0,
        system: "Eres un asistente experto en análisis visual. " +
                "Responde SIEMPRE en español, en dos frases como máximo, " +
                "sin relleno conversacional y sin enumerar detalles irrelevantes."
    },
    {
        id: "poeta", icon: "🎭", label: "Poeta",
        question: "Escribe un poema sobre lo que ves.",
        temperature: 0.9,
        system: "Eres un poeta que mira el mundo a través de una cámara. " +
                "Responde SIEMPRE en español con un poema de tres o cuatro versos " +
                "sobre lo que ves. Sin preámbulos ni explicaciones."
    },
    {
        id: "adivina", icon: "🎲", label: "Adivina",
        question: "¿Qué objeto sostengo? Adivina.",
        temperature: 0.7,
        system: "Eres un asistente que juega a adivinar. Responde SIEMPRE en español, " +
                "en una o dos frases: di qué objeto crees que sostiene la persona y por qué. " +
                "Arriésgate aunque no estés seguro, y nunca pidas más información."
    },
    {
        id: "traduce", icon: "🌐", label: "Traduce",
        question: "Lee el texto de la imagen y tradúcelo al inglés.",
        temperature: 0,
        system: "Eres un traductor que lee texto en imágenes. Transcribe el texto visible " +
                "y añade su traducción al inglés, con este formato exacto:\n" +
                "Texto: ...\nInglés: ...\n" +
                "Si no hay texto legible, responde solo: No veo texto."
    },
];

// Attractor mode: an idle screen draws nobody, so after this long without a
// touch the demo asks itself a question and keeps rotating. Set to 0 to disable.
const IDLE_MS = 30000;

// --- Auto-resize textareas ---
function autoResizeTextarea(el) {
    if (!el) return;
    el.style.height = "auto";
    el.style.height = (el.scrollHeight) + "px";
}

// VARIABLES GLOBALES
let video, canvas, stream;
let isThinking = false;
let serverReady = false;
let currentMode = MODES[0];
let idleTimer = null;
let attractorIndex = 0;

// DOM references
let instructionText, responseText, startButton, loadingOverlay, presetsBox, speakToggle;
let modesBox, lastFrame, autoBadge, timingBox;

function setResponse(text) {
    if (!responseText) return;
    responseText.textContent = text;
    // Subtitles sit at the bottom of the frame, so a long answer scrolls to its
    // newest line instead of growing upwards over the video.
    const box = responseText.parentElement;
    if (box) box.scrollTop = box.scrollHeight;
}

// Latency numbers are for whoever runs the stand, not for the subtitles the
// audience reads.
function setTiming(text) {
    if (timingBox) timingBox.textContent = text;
}

function setLoadingVisible(visible) {
    if (loadingOverlay) {
        loadingOverlay.style.display = visible ? "block" : "none";
    }
}

// --- VOZ ---
// Reading the answer out loud carries a noisy stand better than text does. It
// speaks sentence by sentence as they stream in, so the audio starts while the
// model is still writing instead of after it finishes.
let ttsEnabled = false;
let spokenUpTo = 0;
let spanishVoice = null;

function pickVoice() {
    const voices = speechSynthesis.getVoices();
    spanishVoice = voices.find(v => v.lang.toLowerCase().startsWith("es")) || null;
}

function resetSpeech() {
    spokenUpTo = 0;
    if ("speechSynthesis" in window) speechSynthesis.cancel();
}

function flushSpeech(fullText, finished) {
    if (!ttsEnabled || !("speechSynthesis" in window)) return;

    let pending = fullText.slice(spokenUpTo);
    if (!finished) {
        // Only speak up to the last completed sentence, so the voice never
        // stops mid-clause waiting for the next token.
        const match = pending.match(/^[\s\S]*[.!?…](?=\s|$)/);
        if (!match) return;
        pending = match[0];
    }
    spokenUpTo += pending.length;

    const text = pending.trim();
    if (!text) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "es-ES";
    if (spanishVoice) utterance.voice = spanishVoice;
    speechSynthesis.speak(utterance);
}

function setTtsEnabled(on) {
    ttsEnabled = on;
    speakToggle.classList.toggle("active", on);
    speakToggle.textContent = on ? "🔊" : "🔇";
    if (!on) resetSpeech();
    try {
        localStorage.setItem("tts", on ? "1" : "0");
    } catch (e) {
        // Private windows and blocked site data throw here; the toggle still
        // works for this session.
    }
}

async function initCamera() {
    video = document.getElementById("videoFeed");
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        await video.play();
    } catch (err) {
        setResponse("No se pudo acceder a la cámara.");
    }
}

// Replaces the old WebGPU model load: nothing downloads here any more, we just
// confirm the vLLM endpoint is up and serving the model we expect.
async function initModel() {
    setLoadingVisible(true);
    setResponse("Conectando con el servidor...");

    try {
        const res = await fetch(`${API_BASE}/models`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const served = (await res.json()).data.map(m => m.id);
        if (!served.includes(MODEL_ID)) {
            setResponse(
                `⚠️ El servidor responde, pero sirve: ${served.join(", ")}\n` +
                `Ajusta MODEL_ID en script.js.`
            );
        } else {
            setResponse("✅ ¡Listo! Elige una pregunta o pulsa Analizar.");
        }

        serverReady = true;
        startButton.textContent = "Analizar";
    } catch (err) {
        setResponse(
            `❌ No se puede conectar con ${API_BASE} (${err.message}).\n` +
            `Comprueba que el túnel SSH sigue abierto.`
        );
    } finally {
        setLoadingVisible(false);
    }
}

// Downscales to MAX_SIDE on the way out and returns a JPEG data URL, which is
// the format the OpenAI image_url field takes.
function captureImage() {
    if (!stream || !video || !video.videoWidth) return null;

    const scale = Math.min(1, MAX_SIDE / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);

    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", JPEG_QUALITY);
}

function setBusy(busy) {
    isThinking = busy;
    startButton.disabled = busy;
    presetsBox.querySelectorAll("button").forEach(b => { b.disabled = busy; });
    modesBox.querySelectorAll("button").forEach(b => { b.disabled = busy; });

    if (busy) {
        // Injects a self-animating SVG spinner along with the text
        startButton.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;">
                <path d="M21 12a9 9 0 1 1-6.219-8.56">
                    <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
                </path>
            </svg>
            Analizando...
        `;
    } else {
        startButton.textContent = "Analizar";
    }
}

// --- TRIGGERED INFERENCE ---
async function handleAnalyze() {
    if (!serverReady || isThinking) return;

    setBusy(true);
    setResponse("");
    setTiming("analizando...");
    resetSpeech();

    const instruction = instructionText.value || currentMode.question;
    const dataUrl = captureImage();

    if (!dataUrl) {
        setBusy(false);
        return;
    }

    // Show the exact frame that was sent. People assume the model watches live
    // video, and seeing the still it actually looked at explains the whole thing
    // without a word of explanation.
    lastFrame.src = dataUrl;
    lastFrame.classList.remove("hidden");

    // A 32B model follows the instruction on its own, so the few-shot example
    // and the "Descripción:" primer the 0.5B needed are gone.
    const messages = [
        { role: "system", content: currentMode.system },
        {
            role: "user",
            content: [
                { type: "image_url", image_url: { url: dataUrl } },
                { type: "text", text: instruction }
            ]
        }
    ];

    const started = performance.now();
    let ttft = null;
    let answer = "";
    let tokens = 0;

    try {
        const res = await fetch(`${API_BASE}/chat/completions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: MODEL_ID,
                messages,
                max_tokens: MAX_TOKENS,
                temperature: currentMode.temperature,
                // Streaming is what makes a multi-second answer feel fast: the
                // first words land in about a second and the rest reads as
                // typing rather than waiting.
                stream: true,
                stream_options: { include_usage: true },
            })
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        // Server-sent events arrive as "data: {...}" lines, and a chunk can
        // split mid-line, so the tail stays in the buffer until completed.
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const payload = line.slice(6).trim();
                if (!payload || payload === "[DONE]") continue;

                const chunk = JSON.parse(payload);
                if (chunk.usage) tokens = chunk.usage.completion_tokens;

                const piece = chunk.choices?.[0]?.delta?.content;
                if (piece) {
                    if (ttft === null) ttft = (performance.now() - started) / 1000;
                    answer += piece;
                    setResponse(answer);
                    flushSpeech(answer, false);
                }
            }
        }

        flushSpeech(answer, true);

        const elapsed = (performance.now() - started) / 1000;
        console.log(
            `[timing] ${elapsed.toFixed(1)}s total, first token ${ttft?.toFixed(1)}s, ` +
            `${tokens} tokens @ ${canvas.width}x${canvas.height}`
        );
        setResponse(answer.trim());
        setTiming(`${elapsed.toFixed(1)}s · primera palabra ${ttft?.toFixed(1)}s · ${tokens} tokens`);

    } catch (e) {
        setResponse(`Error: ${e.message}`);
    } finally {
        setBusy(false);
        scheduleIdle();
    }
}

// --- MODO ATRACTOR ---
function scheduleIdle() {
    if (idleTimer) clearTimeout(idleTimer);
    if (!IDLE_MS) return;
    idleTimer = setTimeout(runAttractor, IDLE_MS);
}

function runAttractor() {
    if (!serverReady || isThinking) {
        scheduleIdle();
        return;
    }
    // Rotates through the presets so the screen never repeats itself twice in a
    // row, which is what makes a passer-by stop and read.
    const [, question] = PRESETS[attractorIndex % PRESETS.length];
    attractorIndex++;

    instructionText.value = question;
    autoResizeTextarea(instructionText);
    autoBadge.classList.remove("hidden");
    handleAnalyze();
}

function noteInteraction() {
    autoBadge.classList.add("hidden");
    scheduleIdle();
}

// --- UI ---
function renderModes() {
    for (const mode of MODES) {
        const chip = document.createElement("button");
        chip.className = "chip mode";
        chip.textContent = `${mode.icon} ${mode.label}`;
        chip.classList.toggle("active", mode === currentMode);
        chip.addEventListener("click", () => {
            currentMode = mode;
            modesBox.querySelectorAll("button").forEach(b => b.classList.remove("active"));
            chip.classList.add("active");
            // The mode brings its own question, so one tap both switches the
            // personality and shows it off.
            instructionText.value = mode.question;
            autoResizeTextarea(instructionText);
            handleAnalyze();
        });
        modesBox.appendChild(chip);
    }
}

function renderPresets() {
    for (const [icon, question] of PRESETS) {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.textContent = `${icon} ${question}`;
        chip.addEventListener("click", () => {
            instructionText.value = question;
            autoResizeTextarea(instructionText);
            handleAnalyze();
        });
        presetsBox.appendChild(chip);
    }
}

window.addEventListener("DOMContentLoaded", async () => {
    instructionText = document.getElementById("instructionText");
    responseText = document.getElementById("responseText");
    startButton = document.getElementById("startButton");
    loadingOverlay = document.getElementById("loadingOverlay");
    canvas = document.getElementById("canvas");
    presetsBox = document.getElementById("presets");
    speakToggle = document.getElementById("speakToggle");
    modesBox = document.getElementById("modes");
    lastFrame = document.getElementById("lastFrame");
    autoBadge = document.getElementById("autoBadge");
    timingBox = document.getElementById("timing");

    startButton.addEventListener("click", handleAnalyze);
    speakToggle.addEventListener("click", () => setTtsEnabled(!ttsEnabled));

    // Any sign of a human postpones the attractor.
    document.addEventListener("pointerdown", noteInteraction);
    document.addEventListener("keydown", noteInteraction);

    // Enter sends the instruction; Shift+Enter still inserts a newline, since
    // this is a textarea. handleAnalyze already ignores presses while a request
    // is in flight.
    instructionText.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleAnalyze();
        }
    });

    renderModes();
    renderPresets();

    if ("speechSynthesis" in window) {
        pickVoice();
        // The voice list is often empty on first call and fills in later.
        speechSynthesis.addEventListener("voiceschanged", pickVoice);
        let remembered = "0";
        try {
            remembered = localStorage.getItem("tts") ?? "0";
        } catch (e) { /* see setTtsEnabled */ }
        setTtsEnabled(remembered === "1");
    } else {
        speakToggle.style.display = "none";
    }

    await initCamera();
    await initModel();

    // Only start the attractor once there is a server to ask.
    if (serverReady) scheduleIdle();
});

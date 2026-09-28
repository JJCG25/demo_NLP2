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

// Speech recognition locale. "es-CO", "es-MX" or "es-AR" recognise local accents
// noticeably better than the Castilian default.
const SPEECH_LANG = "es-ES";

// Wake words for hands-free mode. Lowercase, and short ones win: the recogniser
// hears "oye" reliably, whole phrases much less so.
const WAKE_WORDS = ["oye", "asistente", "hola"];

// "local"   -> Whisper on the workstation CPUs. Nothing leaves the machine, and
//              it needs the second tunnel (-L 8100:<node>:<stt port>).
// "browser" -> Chrome's SpeechRecognition, which streams the audio to Google.
const STT_BACKEND = "local";
const STT_URL = "http://localhost:8100/transcribe";

// Speech out. "local" plays Piper from the workstation: a neural voice, and the
// answer text stays on the machine. "browser" uses the laptop's own voices, which
// on Windows sound robotic unless Chrome picks a cloud one. Local falls back to
// the browser automatically if /speak is unavailable.
const TTS_BACKEND = "local";
const TTS_URL = "http://localhost:8100/speak";

// Voice activity detection for the local backend. SPEECH_RMS is loudness on a
// 0..1 scale: raise it in a noisy room, lower it if quiet speech is missed.
const SPEECH_RMS = 0.02;
const SILENCE_MS = 900;          // pause that ends an utterance
const MAX_UTTERANCE_MS = 15000;  // hard stop, so one noise cannot record forever

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
let modesBox, lastFrame, autoBadge, timingBox, micButton, handsFreeButton;

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
    const spanish = speechSynthesis.getVoices()
        .filter(v => v.lang.toLowerCase().startsWith("es"));

    // localService tells them apart: the system voices (Microsoft, Apple)
    // synthesise on this machine, while Chrome's "Google ..." voices render on
    // Google's servers, which would send every answer off the machine. Prefer
    // local, and say so in the console when only a remote one is available.
    spanishVoice = spanish.find(v => v.localService) || spanish[0] || null;

    if (spanishVoice) {
        console.log(`[tts] ${spanishVoice.name} — ` +
                    (spanishVoice.localService ? "local" : "REMOTA: el texto sale a internet"));
    }
}

function resetSpeech() {
    spokenUpTo = 0;
    if ("speechSynthesis" in window) speechSynthesis.cancel();
    stopLocalSpeech();
    speaking = false;
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

    if (TTS_BACKEND === "local") enqueueLocalSpeech(text);
    else speakWithBrowser(text);
}

// Mic must stay shut from the moment there is speech pending, not from the moment
// audio starts: otherwise the watchdog reopens it during the gap and hands-free
// mode transcribes the demo's own answer.
function beginSpeaking() {
    speaking = true;
    if (STT_BACKEND === "browser") {
        if (recognition && listening) recognition.stop();
    } else if (listening) {
        stopLocalListening();
    }
}

function endSpeaking() {
    speaking = false;
    maybeResumeListening();
}

function speakWithBrowser(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "es-ES";
    if (spanishVoice) utterance.voice = spanishVoice;
    utterance.addEventListener("start", beginSpeaking);
    utterance.addEventListener("end", endSpeaking);
    speechSynthesis.speak(utterance);
}

// --- Voz local (Piper en la workstation) ---
// Sentences are synthesised one at a time so speech starts while the model is
// still writing. The chain keeps them in order: the requests would otherwise
// finish out of sequence and the answer would be read scrambled.
let ttsChain = Promise.resolve();
let ttsQueue = [];
let ttsAudio = null;
let ttsPlaying = false;

function enqueueLocalSpeech(text) {
    beginSpeaking();
    ttsChain = ttsChain.then(() => synthesise(text)).catch(() => {});
}

async function synthesise(text) {
    try {
        const res = await fetch(TTS_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        ttsQueue.push(URL.createObjectURL(await res.blob()));
        playQueued();
    } catch (e) {
        console.warn(`[tts] Piper no disponible (${e.message}), uso la voz del navegador`);
        speakWithBrowser(text);
    }
}

function playQueued() {
    if (ttsPlaying || !ttsQueue.length) return;

    const url = ttsQueue.shift();
    ttsPlaying = true;
    if (!ttsAudio) ttsAudio = new Audio();
    ttsAudio.src = url;

    const done = () => {
        URL.revokeObjectURL(url);
        ttsPlaying = false;
        if (ttsQueue.length) playQueued();
        else endSpeaking();
    };
    ttsAudio.onended = done;
    ttsAudio.onerror = done;
    ttsAudio.play().catch(() => done());
}

function stopLocalSpeech() {
    ttsQueue.forEach(URL.revokeObjectURL);
    ttsQueue = [];
    ttsPlaying = false;
    if (ttsAudio) {
        ttsAudio.pause();
        ttsAudio.src = "";
    }
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

// --- ESCUCHA (voz a voz) ---
// Two backends. "local" records audio and posts it to Whisper on the
// workstation, so nothing leaves the machine; "browser" uses Chrome's
// SpeechRecognition, which streams the microphone to Google's servers.
// Push to talk rather than always-on listening by default: a stand is noisy, and
// an open mic picks up the crowd and the demo's own answers.
let recognition = null;
let listening = false;      // the mic is open right now
let wantListening = false;  // it should be open, even if it just closed
let handsFree = false;      // listen continuously and wait for a wake word
let speaking = false;       // the demo is talking, so the mic must stay shut

// Local backend state
let audioStream = null, analyser = null, vadTimer = null;
let recorder = null, chunks = [], recording = false, forcedRecording = false;
let dropUtterance = false;  // stop the recorder without sending what it captured
let silenceSince = 0, recordingSince = 0;

// In hands-free mode everything is ignored until a wake word shows up, so the
// crowd's conversation does not keep triggering the demo. What follows the wake
// word is the question; a bare "oye" falls back to the mode's own question.
function extractAfterWakeWord(transcript) {
    const lower = transcript.toLowerCase();
    for (const word of WAKE_WORDS) {
        const at = lower.indexOf(word);
        if (at === -1) continue;
        const rest = transcript.slice(at + word.length).replace(/^[\s,.:;!?¿¡]+/, "").trim();
        return rest || currentMode.question;
    }
    return "";
}

function paintMicState() {
    micButton.classList.toggle("listening", listening);
    micButton.textContent = handsFree ? "👂" : (listening ? "⏹" : "🎤");
    micButton.classList.toggle("hands-free", handsFree);
}

// Whatever the backend heard ends up here.
function onTranscript(transcript) {
    if (isThinking) {
        // Same collision as above, reached from the browser backend.
        maybeResumeListening();
        return;
    }
    const question = handsFree ? extractAfterWakeWord(transcript) : transcript.trim();
    if (!question) {
        // Heard something, but not for us. Keep waiting.
        maybeResumeListening();
        return;
    }
    instructionText.value = question;
    autoResizeTextarea(instructionText);
    handleAnalyze();
}

// --- Backend local: Whisper en la workstation ---
async function ensureMic() {
    if (audioStream) return true;
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
    } catch (e) {
        setResponse("Permite el acceso al micrófono para hablarle.");
        return false;
    }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    ctx.createMediaStreamSource(audioStream).connect(analyser);
    return true;
}

// Loudness of the current frame, 0..1. Cheap enough to run every 50ms and it is
// all the voice detection this needs.
function micLevel() {
    const buf = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (const v of buf) sum += v * v;
    return Math.sqrt(sum / buf.length);
}

function startRecording() {
    chunks = [];
    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus" : "audio/webm";
    recorder = new MediaRecorder(audioStream, { mimeType: mime });
    recorder.addEventListener("dataavailable", (e) => chunks.push(e.data));
    recorder.addEventListener("stop", sendUtterance);
    recorder.start();
    recording = true;
    recordingSince = Date.now();
}

function stopRecording(discard) {
    dropUtterance = Boolean(discard);
    if (recorder && recording) recorder.stop();
    recording = false;
    forcedRecording = false;
}

async function sendUtterance() {
    const blob = new Blob(chunks, { type: "audio/webm" });
    chunks = [];

    // Audio captured while an answer was already being produced is not a new
    // question: transcribing it used to overwrite the streaming answer with
    // "Transcribiendo..." and then be dropped by handleAnalyze, which looked
    // exactly like a freeze.
    const drop = dropUtterance;
    dropUtterance = false;
    if (drop || isThinking || speaking) {
        maybeResumeListening();
        return;
    }

    // A fragment this short is a cough or a door, not a question.
    if (blob.size < 4000) {
        maybeResumeListening();
        return;
    }

    setResponse("Transcribiendo...");
    const form = new FormData();
    form.append("audio", blob, "utterance.webm");

    try {
        const res = await fetch(STT_URL, { method: "POST", body: form });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = await res.json();
        setTiming(`voz: ${body.seconds}s`);
        if (body.text) {
            onTranscript(body.text);
        } else {
            maybeResumeListening();
        }
    } catch (e) {
        setResponse(`No se pudo transcribir (${e.message}). ¿Está abierto el túnel del puerto 8100?`);
        paintMicState();
        // A failed transcription must not end the conversation: without this the
        // mic stayed shut and hands-free mode froze.
        maybeResumeListening();
    }
}

function vadTick() {
    // Never record while the demo talks or thinks, or it transcribes its own
    // answer and asks itself about it.
    if (speaking || isThinking) {
        if (recording) stopRecording(true);
        return;
    }

    const loud = micLevel() > SPEECH_RMS;

    if (!recording) {
        if (loud || forcedRecording) startRecording();
        return;
    }

    if (loud) {
        silenceSince = 0;
    } else if (!silenceSince) {
        silenceSince = Date.now();
    } else if (Date.now() - silenceSince > SILENCE_MS) {
        // A pause this long means they finished the sentence.
        listening = false;
        clearInterval(vadTimer);
        vadTimer = null;
        paintMicState();
        stopRecording();
        return;
    }

    if (Date.now() - recordingSince > MAX_UTTERANCE_MS) {
        stopRecording();
    }
}

function startLocalListening(forceRecord) {
    silenceSince = 0;
    forcedRecording = forceRecord;
    listening = true;
    if (!vadTimer) vadTimer = setInterval(vadTick, 50);
    paintMicState();
}

function stopLocalListening() {
    if (vadTimer) clearInterval(vadTimer);
    vadTimer = null;
    if (recording) {
        // Drop what was being recorded: the user asked it to stop.
        stopRecording(true);
    }
    listening = false;
    paintMicState();
}

// --- Backend del navegador: SpeechRecognition (el audio va a Google) ---
function initBrowserSTT() {
    const Impl = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Impl) return false;

    recognition = new Impl();
    recognition.lang = SPEECH_LANG;
    recognition.interimResults = true;

    recognition.addEventListener("result", (e) => {
        const result = e.results[e.results.length - 1];
        const transcript = result[0].transcript;

        // Show the words as they are recognised, so people see it heard them.
        if (!handsFree) {
            instructionText.value = transcript;
            autoResizeTextarea(instructionText);
        }
        if (!result.isFinal) return;
        if (!handsFree) {
            listening = false;
            paintMicState();
        }
        onTranscript(transcript);
    });

    recognition.addEventListener("end", () => {
        listening = false;
        // Chrome ends recognition on its own after a stretch of silence, so
        // hands-free mode has to keep restarting it.
        if (!maybeResumeListening()) paintMicState();
    });

    recognition.addEventListener("error", (e) => {
        if (e.error === "not-allowed" || e.error === "service-not-allowed") {
            handsFree = false;
            wantListening = false;
            setResponse("Permite el acceso al micrófono para hablarle.");
        } else if (e.error === "no-speech" && !handsFree) {
            setResponse("No te he oído. Pulsa el micrófono y habla de nuevo.");
        }
        paintMicState();
    });
    return true;
}

function startBrowserListening() {
    try {
        recognition.start();
        listening = true;
    } catch (e) {
        // Already started; harmless.
    }
    paintMicState();
}

// --- Control común ---
function initSpeech() {
    const ok = STT_BACKEND === "browser"
        ? initBrowserSTT()
        : (window.MediaRecorder && navigator.mediaDevices);

    if (!ok) {
        micButton.style.display = "none";
        handsFreeButton.style.display = "none";
    }
}

// A stand runs unattended for hours, so rather than trusting every path to
// reopen the mic, this checks a couple of times a second that hands-free mode is
// actually listening and restarts it when it is not. It covers the failure modes
// nobody thought of, not just the known ones.
let watchdogTimer = null;

function startWatchdog() {
    if (watchdogTimer) return;
    watchdogTimer = setInterval(() => {
        if (!handsFree || !wantListening) return;
        if (listening || isThinking || speaking) return;
        if (STT_BACKEND === "browser") startBrowserListening();
        else startLocalListening(false);
    }, 2000);
}

function stopWatchdog() {
    if (watchdogTimer) clearInterval(watchdogTimer);
    watchdogTimer = null;
}

// Returns true when it took responsibility for reopening the mic.
function maybeResumeListening() {
    if (!wantListening || !handsFree || speaking || isThinking) return false;
    setTimeout(() => {
        if (!wantListening || !handsFree || speaking || isThinking) return;
        if (STT_BACKEND === "browser") startBrowserListening();
        else startLocalListening(false);
    }, 300);
    return true;
}

async function toggleListening() {
    if (isThinking) return;

    if (listening) {
        wantListening = false;
        if (STT_BACKEND === "browser") recognition.stop();
        else stopLocalListening();
        return;
    }

    // Never listen while the demo is talking, or the mic transcribes its own
    // answer straight back.
    resetSpeech();
    // Someone talking to it expects to be answered out loud.
    if (!ttsEnabled) setTtsEnabled(true);

    wantListening = true;
    setResponse("Escuchando...");

    if (STT_BACKEND === "browser") {
        startBrowserListening();
    } else {
        if (!await ensureMic()) return;
        startLocalListening(true);
    }
}

async function toggleHandsFree() {
    handsFree = !handsFree;
    if (recognition) recognition.continuous = handsFree;

    if (handsFree) {
        if (!ttsEnabled) setTtsEnabled(true);
        wantListening = true;
        setResponse(`Manos libres: di "${WAKE_WORDS[0]}" y tu pregunta.`);
        startWatchdog();
        if (STT_BACKEND === "browser") {
            if (!listening) startBrowserListening();
        } else {
            if (!await ensureMic()) { handsFree = false; return; }
            if (!listening) startLocalListening(false);
        }
    } else {
        wantListening = false;
        stopWatchdog();
        if (STT_BACKEND === "browser") {
            if (listening) recognition.stop();
        } else {
            stopLocalListening();
        }
    }
    paintMicState();
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
        serverReady = false;
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
    micButton.disabled = busy;
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
    if (isThinking) return;

    // Retry the connection instead of refusing in silence. The page used to give
    // up for the whole session if it loaded before vLLM had finished starting,
    // and this doubles as recovery when the Slurm job is restarted mid-event.
    if (!serverReady) {
        setResponse("Reconectando con el modelo...");
        await initModel();
        if (!serverReady) return;   // initModel already said what is wrong
    }

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
        // With nothing left to say, hands-free mode goes back to listening. If
        // the answer is still being spoken, the utterance's end handler does it.
        if (!speaking) maybeResumeListening();
    }
}

// --- MODO ATRACTOR ---
function scheduleIdle() {
    if (idleTimer) clearTimeout(idleTimer);
    if (!IDLE_MS) return;
    idleTimer = setTimeout(runAttractor, IDLE_MS);
}

function runAttractor() {
    if (!serverReady || isThinking || listening || speaking) {
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
    micButton = document.getElementById("micButton");
    handsFreeButton = document.getElementById("handsFreeButton");

    startButton.addEventListener("click", handleAnalyze);
    speakToggle.addEventListener("click", () => setTtsEnabled(!ttsEnabled));
    micButton.addEventListener("click", toggleListening);
    handsFreeButton.addEventListener("click", toggleHandsFree);

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
    initSpeech();

    if ("speechSynthesis" in window) {
        pickVoice();
        // The voice list is often empty on first call and fills in later.
        speechSynthesis.addEventListener("voiceschanged", pickVoice);
        let remembered = "0";
        try {
            remembered = localStorage.getItem("tts") ?? "0";
        } catch (e) { /* see setTtsEnabled */ }
        setTtsEnabled(remembered === "1");
    } else if (TTS_BACKEND !== "local") {
        // Only hide the toggle when there is no way to speak at all: with Piper
        // the browser's own synthesis is just the fallback.
        speakToggle.style.display = "none";
    }

    await initCamera();
    await initModel();

    // Only start the attractor once there is a server to ask.
    if (serverReady) scheduleIdle();
});

// --- CONFIGURACIÓN ---
// The model no longer runs in the browser. It runs on the T4 box under vLLM,
// which exposes an OpenAI-compatible endpoint. Point this at the SSH tunnel
// (ssh -L 8000:localhost:8000 user@t4-box) so the browser only ever talks to
// localhost: no CORS, no exposed port.
const API_BASE = "http://localhost:8000/v1";
const MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ";

// Latency knobs. MAX_SIDE is the strongest one by far: every pixel becomes
// vision tokens the T4s have to chew through, so downscaling the frame before
// it leaves the browser cuts prefill time more than anything else here.
const MAX_SIDE = 768;
const JPEG_QUALITY = 0.8;
const MAX_TOKENS = 64;

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

// DOM references
let instructionText, responseText, startButton, loadingOverlay;

function setResponse(text) {
    if (responseText) {
        responseText.value = text;
        autoResizeTextarea(responseText);
    }
}

function setLoadingVisible(visible) {
    if (loadingOverlay) {
        loadingOverlay.style.display = visible ? "block" : "none";
    }
}

async function initCamera() {
    video = document.getElementById("videoFeed");
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        await video.play();
    } catch (err) {
        setResponse("Could not connect to camera.");
    }
}

// Replaces the old WebGPU model load: nothing downloads here any more, we just
// confirm the vLLM endpoint is up and serving the model we expect.
async function initModel() {
    setLoadingVisible(true);
    setResponse("Connecting to the vLLM server...");

    try {
        const res = await fetch(`${API_BASE}/models`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const served = (await res.json()).data.map(m => m.id);
        if (!served.includes(MODEL_ID)) {
            setResponse(
                `⚠️ Server is up but serving: ${served.join(", ")}\n` +
                `Set MODEL_ID in script.js to one of those.`
            );
        } else {
            setResponse("✅ Ready! Press 'Analyze' to start.");
        }

        serverReady = true;
        startButton.textContent = "Analyze";
    } catch (err) {
        setResponse(
            `❌ Cannot reach ${API_BASE} (${err.message}).\n` +
            `Start vLLM on the T4 box, then open the tunnel:\n` +
            `ssh -L 8000:localhost:8000 user@t4-box`
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

// --- TRIGGERED INFERENCE ---
async function handleAnalyze() {
    if (!serverReady || isThinking) return;

    isThinking = true;
    startButton.disabled = true;

    // Injects a self-animating SVG spinner along with the text
    startButton.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 5px;">
            <path d="M21 12a9 9 0 1 1-6.219-8.56">
                <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
            </path>
        </svg>
        Analyzing...
    `;

    // This clears the answer section immediately when the button is pressed
    setResponse("");

    const instruction = instructionText.value || "¿Qué ves?";
    const dataUrl = captureImage();

    if (!dataUrl) {
        isThinking = false;
        startButton.disabled = false;
        startButton.textContent = "Analyze"; // Reset to standard text
        return;
    }

    // A 32B model follows the instruction on its own, so the few-shot example
    // and the "Descripción:" primer the 0.5B needed are gone.
    const messages = [
        {
            role: "system",
            content: "Eres un asistente experto en análisis visual. " +
                     "Responde SIEMPRE en español, de forma concisa y directa, sin relleno conversacional."
        },
        {
            role: "user",
            content: [
                { type: "image_url", image_url: { url: dataUrl } },
                { type: "text", text: instruction }
            ]
        }
    ];

    const started = performance.now();

    try {
        const res = await fetch(`${API_BASE}/chat/completions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: MODEL_ID,
                messages,
                max_tokens: MAX_TOKENS,
                temperature: 0,
            })
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);

        const body = await res.json();
        const elapsed = ((performance.now() - started) / 1000).toFixed(1);
        const tokens = body.usage?.completion_tokens ?? "?";
        console.log(`[timing] ${elapsed}s for ${tokens} tokens @ ${canvas.width}x${canvas.height}`);

        setResponse(`${body.choices[0].message.content.trim()}\n\n(${elapsed}s)`);

    } catch (e) {
        setResponse(`Error: ${e.message}`);
    } finally {
        isThinking = false;
        startButton.disabled = false;
        // Reset to just "Analyze" when finished, which removes the SVG
        startButton.textContent = "Analyze";
    }
}

// --- UI y Draggable ---
function makeDraggable(el) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    el.addEventListener('mousedown', dragStart);
    function dragStart(e) {
        if (['TEXTAREA', 'BUTTON'].includes(e.target.tagName)) return;
        pos3 = e.clientX; pos4 = e.clientY;
        document.onmouseup = () => { document.onmouseup = null; document.onmousemove = null; };
        document.onmousemove = (e) => {
            pos1 = pos3 - e.clientX; pos2 = pos4 - e.clientY;
            pos3 = e.clientX; pos4 = e.clientY;
            el.style.top = (el.offsetTop - pos2) + "px";
            el.style.left = (el.offsetLeft - pos1) + "px";
        };
    }
}

window.addEventListener("DOMContentLoaded", async () => {
    instructionText = document.getElementById("instructionText");
    responseText = document.getElementById("responseText");
    startButton = document.getElementById("startButton");
    loadingOverlay = document.getElementById("loadingOverlay");
    canvas = document.getElementById("canvas");

    startButton.addEventListener("click", handleAnalyze);

    const ioAreas = document.querySelector('.io-areas');
    if (ioAreas) makeDraggable(ioAreas);

    await initCamera();
    await initModel();
});

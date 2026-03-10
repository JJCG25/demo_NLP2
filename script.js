// --- Importaciones ---
import {
    AutoProcessor,
    AutoModelForImageTextToText,
    RawImage,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers/dist/transformers.min.js";

// CONFIGURACIÓN DE ENTORNO
env.allowLocalModels = false;
env.useBrowserCache = true;

// --- Auto-resize textareas ---
function autoResizeTextarea(el) {
    if (!el) return;
    el.style.height = "auto";
    el.style.height = (el.scrollHeight) + "px";
}

// VARIABLES GLOBALES
let video, canvas, processor, model, stream;
let isProcessing = false;

// DOM references
let instructionText, responseText, startButton, loadingOverlay;

// Helper: set response text
function setResponse(text) {
    console.log("[Status]", text);
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
        setResponse("Model loading...");
    } catch (err) {
        console.error("Camera error:", err);
        setResponse("Could not connect to camera. Please check your permissions.");
    }
}

async function initModel() {
    const modelId = "onnx-community/FastVLM-0.5B-ONNX";
    setLoadingVisible(true);
    setResponse("Model loading...");

    let useWebGPU = false;
    if (navigator.gpu) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
                const info = adapter.info || {};
                const desc = (info.description || "").toLowerCase();
                if (!desc.includes("software") && !desc.includes("swiftshader")) {
                    useWebGPU = true;
                }
            }
        } catch (e) {
            console.warn("WebGPU adapter request failed:", e);
        }
    }


    function onProgress(p) {
        if (p.status === "downloading" && p.total > 0) {
            const pct = Math.round((p.loaded / p.total) * 100);
            const name = (p.file || "").split("/").pop();
            setResponse(`Model loading... ${pct}%`);
        } else if (p.status === "loading") {
            setResponse("Model loading...");
        } else if (p.status === "done") {
        }
    }

    try {
        console.log("Loading processor...");
        processor = await AutoProcessor.from_pretrained(modelId, { progress_callback: onProgress });

        console.log("Loading model...");
        model = await AutoModelForImageTextToText.from_pretrained(modelId, {
            device: useWebGPU ? "webgpu" : "wasm",
            dtype: useWebGPU ? {
                embed_tokens: "fp16",
                vision_encoder: "q4",
                decoder_model_merged: "q4",
            } : "q8",
            progress_callback: onProgress,
        });

        setResponse("✅ Ready! Press Start whenever you want.");
    } catch (err) {
        console.error("Model Loading Error:", err);
        setResponse(`❌ Error: ${err.message}`);
        setLoadingVisible(false);
    } finally {
        setLoadingVisible(false);
    }
}

function captureImage() {
    if (!stream || !video || !video.videoWidth) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = context.getImageData(0, 0, canvas.width, canvas.height);
    return new RawImage(frame.data, frame.width, frame.height, 4);
}

async function runInference() {
    if (!model || !processor || !isProcessing) return;

    const instruction = instructionText.value || "What do you see in this image?";
    const rawImg = captureImage();
    if (!rawImg) return;

    try {
        // Prepare prompt using FastVLM format
        const messages = [{ role: "user", content: `<image>${instruction}` }];
        const prompt = processor.apply_chat_template(messages, {
            add_generation_prompt: true
        });

        // Process image and text
        const inputs = await processor(rawImg, prompt);

        // Generate text
        const generateOptions = {
            ...inputs,
            max_new_tokens: 128,
            do_sample: false,
        };


        const outputs = await model.generate(generateOptions);

        // Decode output, slicing off the prompt tokens
        const decoded = processor.batch_decode(
            outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
            { skip_special_tokens: true }
        );

        setResponse(decoded[0].trim());

    } catch (e) {
        console.error("Inference Error:", e);
        setResponse(`Something went wrong: ${e.message}`);
        handleStop();
    }
}


async function loop() {
    while (isProcessing) {
        await runInference();
        if (isProcessing) await new Promise(r => setTimeout(r, 1000));
    }
}

function handleStart() {
    if (!model) {
        alert("Model not loaded yet.");
        return;
    }
    isProcessing = true;
    startButton.textContent = "Stop";
    startButton.classList.replace("start", "stop");
    setResponse("Thinking...");
    loop();
}

function handleStop() {
    isProcessing = false;
    startButton.textContent = "Start";
    startButton.classList.replace("stop", "start");
    setResponse("Paused.");
}

function makeDraggable(el) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;

    el.addEventListener('mousedown', dragStart);
    el.addEventListener('touchstart', dragStart, { passive: false });

    function dragStart(e) {
        // Only drag if clicking the panel itself or the label, not textareas or buttons
        if (['TEXTAREA', 'BUTTON'].includes(e.target.tagName)) return;

        const clientX = e.type === 'touchstart' ? e.touches[0].clientX : e.clientX;
        const clientY = e.type === 'touchstart' ? e.touches[0].clientY : e.clientY;

        pos3 = clientX;
        pos4 = clientY;

        document.addEventListener('mouseup', dragEnd);
        document.addEventListener('mousemove', dragMove);
        document.addEventListener('touchend', dragEnd);
        document.addEventListener('touchmove', dragMove, { passive: false });

        el.style.cursor = 'grabbing';
    }

    function dragMove(e) {
        const clientX = e.type === 'touchmove' ? e.touches[0].clientX : e.clientX;
        const clientY = e.type === 'touchmove' ? e.touches[0].clientY : e.clientY;

        pos1 = pos3 - clientX;
        pos2 = pos4 - clientY;
        pos3 = clientX;
        pos4 = clientY;

        // Reset transform if it's currently used for centering
        if (el.style.transform !== 'none') {
            const rect = el.getBoundingClientRect();
            el.style.transform = 'none';
            el.style.top = rect.top + 'px';
            el.style.left = rect.left + 'px';
            el.style.bottom = 'auto'; // Disable bottom constraint
            el.style.margin = '0';
        }

        el.style.top = (el.offsetTop - pos2) + "px";
        el.style.left = (el.offsetLeft - pos1) + "px";
    }

    function dragEnd() {
        document.removeEventListener('mouseup', dragEnd);
        document.removeEventListener('mousemove', dragMove);
        document.removeEventListener('touchend', dragEnd);
        document.removeEventListener('touchmove', dragMove);
        el.style.cursor = 'grab';
    }
}

window.addEventListener("DOMContentLoaded", async () => {
    instructionText = document.getElementById("instructionText");
    responseText = document.getElementById("responseText");
    startButton = document.getElementById("startButton");
    loadingOverlay = document.getElementById("loadingOverlay");
    canvas = document.getElementById("canvas");

    instructionText.value = "What do you see?";

    startButton.addEventListener("click", () => isProcessing ? handleStop() : handleStart());

    // Initialize in order
    const ioAreas = document.querySelector('.io-areas');
    if (ioAreas) makeDraggable(ioAreas);

    await initCamera();
    await initModel();
});

window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach(t => t.stop());
});
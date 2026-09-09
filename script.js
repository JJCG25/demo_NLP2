import {
    AutoProcessor,
    AutoModelForImageTextToText,
    RawImage,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js";

// --- CONFIGURACIÓN DE ENTORNO (OPTIMIZADO PARA CPU) ---
// Load straight from the HuggingFace Hub. There is no local ./models/ copy,
// so leaving this on just produced a failed 404 round-trip per file.
env.allowLocalModels = false;
env.useBrowserCache = true;

// Habilitar multi-hilo 
env.backends.onnx.wasm.numThreads = navigator.hardwareConcurrency || 4;

// --- Auto-resize textareas ---
function autoResizeTextarea(el) {
    if (!el) return;
    el.style.height = "auto";
    el.style.height = (el.scrollHeight) + "px";
}

// VARIABLES GLOBALES
let video, canvas, processor, model, stream;
let isThinking = false;

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

async function initModel() {
    const modelId = "onnx-community/FastVLM-0.5B-ONNX";
    setLoadingVisible(true);
    setResponse("Loading model (WebGPU)...");

    try {
        processor = await AutoProcessor.from_pretrained(modelId);

        model = await AutoModelForImageTextToText.from_pretrained(modelId, {
            device: "webgpu",
            dtype: {
                embed_tokens: "fp16",
                vision_encoder: "q4",
                decoder_model_merged: "q4",
            }
        });

        // Changed text to just "Analyze"
        setResponse("✅ Ready! Press 'Analyze' to start.");
        startButton.textContent = "Analyze";
    } catch (err) {
        setResponse(`❌ Error loading: ${err.message}`);
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

// --- TRIGGERED INFERENCE ---
async function handleAnalyze() {
    if (!model || isThinking) return;

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

    const instruction = instructionText.value || "What do you see?";
    const rawImg = captureImage();
    
    if (!rawImg) {
        isThinking = false;
        startButton.disabled = false;
        startButton.textContent = "Analyze"; // Reset to standard text
        return;
    }

    try {
        const messages = [
            { 
                role: "system", 
                content: "You are an expert visual analysis assistant. " + 
                        "CRITICAL RULE: Respond ONLY in English. Be extremely concise, direct, and avoid any conversational filler." 
            },
            { 
                role: "user", 
                content: "<image>Instruction: Describe this image." 
            },
            { 
                role: "assistant", 
                content: "An office environment with several people working on computers." 
            },
            { 
                role: "user", 
                content: `Instruction: ${instruction}. Respond only in English and keep it brief.` 
            },
            { 
                role: "assistant", 
                content: "Description:" 
            }
        ];
        
        const prompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
        const inputs = await processor(rawImg, prompt);

        const outputs = await model.generate({
            ...inputs,
            max_new_tokens: 64, 
            do_sample: false, 
            temperature: 0.0,
            repetition_penalty: 1.2,
        });

        const decoded = processor.batch_decode(
            outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
            { skip_special_tokens: true }
        );

        setResponse(decoded[0].trim());

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
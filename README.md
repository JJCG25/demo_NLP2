# Asistente Visual — Qwen2.5-VL 32B sobre 4× Tesla T4

Demo de cámara que responde en español preguntas sobre lo que ve, con
**Qwen2.5-VL-32B-Instruct-AWQ** servido por vLLM en la workstation de 4 T4.

El navegador se queda con la cámara y la interfaz; los modelos corren en la
workstation y se alcanzan por un túnel SSH, así que la página solo habla con
`localhost`. Nada de audio o imagen sale de las dos máquinas.

```
portátil                    workstation (job de Slurm)
--------                    --------------------------
cámara + interfaz   :8000 → vLLM, 4× T4, TP=4  ·  Qwen2.5-VL-32B-AWQ
                    :8100 → Whisper small (CPU) para el micrófono
         \________ ssh -L, dos reenvíos ________/
```

## Qué hace

- **Una foto por pregunta**: captura un fotograma, lo analiza y muestra la
  respuesta como subtítulos sobre el vídeo. La miniatura del panel enseña
  exactamente qué imagen vio el modelo.
- **Preguntas rápidas**: chips de un toque, sin teclado. Incluyen una de OCR.
- **Cuatro modos** que cambian la personalidad: descriptivo, poeta, adivina y
  traduce.
- **Micrófono manual**: se pulsa 🎤, se habla una vez, y Whisper transcribe en la
  workstation.
- **Modo atractor**: si nadie toca nada en 30 s, se pregunta solo y va rotando.
- **Métricas**: bajo la miniatura aparecen los segundos de cada respuesta.

---

## 1. Preparación (una sola vez, en el nodo de login)

Clona **bajo `/disk`**, no en `$HOME`: el home de esta máquina está casi lleno y
Slurm escribe los logs en la carpeta desde donde envías el job.

```bash
cd /disk/$USER
git clone https://github.com/JJCG25/demo_NLP2 && cd demo_NLP2
bash cluster/setup_env.sh
```

Crea el venv, instala vLLM, faster-whisper y Piper, y descarga los pesos: ~20 GB
del modelo de visión, ~650 MB de Whisper y ~200 MB de voces. Tarda media hora
larga la primera vez. Se descarga aquí porque el job corre con `HF_HUB_OFFLINE=1`.

Si tu espacio no está en `/disk/$USER`, cambia `DISK_ROOT` arriba en
[cluster/setup_env.sh](cluster/setup_env.sh) y [cluster/vllm.sbatch](cluster/vllm.sbatch).

## 2. Levantar los servidores (en el nodo de login)

```bash
cd /disk/$USER/demo_NLP2
sbatch cluster/vllm.sbatch
```

Apunta el **jobid** que te devuelve, porque los puertos salen de él:

| Servicio | Puerto remoto | Ejemplo con job 218 |
|---|---|---|
| vLLM (visión) | `8000 + jobid % 1000` | 8218 |
| Whisper (voz) | `9000 + jobid % 1000` | 9218 |

Sigue el arranque:

```bash
tail -f vllm-<jobid>.out
```

Esperas tres líneas, en este orden: **`whisper ready`** (rápido),
**`piper ready`** y **`Application startup complete`** (unos 4-5 minutos,
mientras reparte 20 GB entre las cuatro tarjetas). El log también imprime el
comando del túnel ya con el nodo y los puertos puestos.

Comprobación:

```bash
curl -s localhost:<puerto-vllm>/health && echo " vLLM OK"   # responde vacío
curl -s localhost:<puerto-stt>/health                        # devuelve JSON
```

El job dura **12 horas** (`--time` en el sbatch). Al morir, se relanza igual y
los pesos ya están en caché, así que solo cuesta la carga.

## 3. Abrir el túnel (en el portátil)

```bash
ssh -L 8000:GPU-Workstation:<puerto-vllm> -L 8100:GPU-Workstation:<puerto-stt> <usuario>@eisi
```

**Deja esa terminal abierta**: si la cierras, el túnel muere y la app pierde los
servidores. Tu lado siempre es 8000 y 8100, así que `script.js` nunca cambia; lo
único que se ajusta al relanzar el job es la parte derecha de cada `-L`.

## 4. Servir la página (en el portátil)

```bash
cd <ruta-del-repo>
python serve.py 8080
```

Abre `http://localhost:8080`. **8080, no 8000**, que ese lo ocupa el túnel.
Permite el acceso a la cámara y al micrófono. Si la caja de respuesta dice
**¡Listo!**, el circuito completo está en pie.

Tras cambiar `script.js`, `style.css` o `index.html`, basta **Ctrl+F5**.

---

## Si algo falla

| Síntoma | Causa y arreglo |
|---|---|
| `channel N: open failed: Connection refused` en el túnel | Al otro lado no escucha nadie: el job aún carga, o los puertos no coinciden con el jobid. |
| La app dice que no puede conectar | Túnel cerrado o job muerto. Revisa con `squeue -u $USER`. Al volver, reintenta sola en cuanto pulses algo. |
| `failed to fetch` al transcribir | El servicio de voz no está: mira `grep -i whisper vllm-<jobid>.err`. Si pide un tamaño que no se descargó, el modo offline lo mata al arrancar. |
| El nodo aparece `down` en `sinfo` | `slurmd` caído. Hace falta un admin: `sudo systemctl restart slurmd` y `scontrol update nodename=GPU-Workstation state=resume`. |
| `nvidia-smi` da `Failed to initialize NVML` fuera de un job | Normal: el cgroup de Slurm solo concede las GPUs dentro de un job. |
| Los cambios de la app no se ven | Caché del navegador. Ctrl+Shift+R, o DevTools → Network → *Disable cache*. |
| Se queda colgado | A los 60 s se recupera solo y avisa en pantalla. En la consola (F12) queda el `[watchdog]`. |

Los logs del job están en `vllm-<jobid>.out` (arranque, `[stt]`, `[tts]`) y
`vllm-<jobid>.err` (fallos), en la carpeta del repo en la workstation.

## Ajustes

Los tres que más se notan, todos en [script.js](script.js):

| Variable | Qué hace |
|---|---|
| `MAX_SIDE` | Resolución del fotograma (768). Menos píxeles, menos tokens de visión, respuesta más rápida. Mantén `max_pixels` del sbatch en sintonía. |
| `MAX_TOKENS` | Largo máximo de la respuesta (128). La generación es token a token, así que esto escala la espera directamente. |
| `SPEECH_RMS` / `SILENCE_MS` | Umbral de voz y pausa que cierra la frase. Súbelo si el ruido de la sala dispara el micro; bájalo si no te oye. |

Y dos que están apagados tras probarlos en el stand, con su interruptor en el
mismo archivo:

- **`ENABLE_HANDS_FREE`** — escucha continua con palabra clave. Se disparaba con
  el ruido y repetía la misma pregunta.
- **`ENABLE_TTS`** — leer la respuesta en voz alta con Piper. Funciona bien, pero
  compite con el micrófono.

En el servidor, [cluster/vllm.sbatch](cluster/vllm.sbatch) acepta al enviar:

```bash
STT_MODEL=base sbatch cluster/vllm.sbatch                    # transcribe más rápido, peor
TTS_VOICE_NAME=es_AR-daniela-high sbatch cluster/vllm.sbatch # otra voz
```

### Medir la latencia

En la workstation, contra `localhost`, para que no entren la red ni el navegador.
No necesita GPU ni reserva de Slurm:

```bash
source /disk/$USER/venv/bin/activate
python cluster/bench.py --port <puerto-vllm> --sweep 512,768,1024
```

Da dos números por tamaño de imagen, con causas distintas: **tiempo hasta la
primera palabra** (prefill y encoder de visión, lo que mueve `MAX_SIDE`) y
**tok/s de generación** (lo que mueve el tamaño del modelo y `MAX_TOKENS`).

Medido en esta máquina: Whisper ~2,8 s, visión ~4 s, Piper ~0,7 s por frase.

### Si hace falta más velocidad

`Qwen/Qwen2.5-VL-7B-Instruct` en fp16 cabe en una sola T4 y es varias veces más
rápido. Cambia `MODEL_ID` en [script.js](script.js), y `MODEL`, `--gres=gpu:1` y
`--tensor-parallel-size 1` en el sbatch.

Para experimentar sin encolar un job cada vez:

```bash
salloc --partition=main --gres=gpu:4 --cpus-per-task=16 --mem=64G --time=1:00:00
```

---

## Notas técnicas

- **Modelo**: [`Qwen/Qwen2.5-VL-32B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ), int4, ~20 GB, TP=4.
- **Precisión `float16` obligatoria**: T4 es Turing (sm75) y no tiene bfloat16;
  tampoco FlashAttention-2, que pide Ampere.
- **Nada de Qwen3-VL**: [no tiene backend de vLLM para Turing](https://github.com/vllm-project/vllm/issues/29743).
- **Voz a texto**: Whisper `small` int8 en CPU vía faster-whisper, en el mismo
  job. El audio no sale de la máquina. `STT_BACKEND = "browser"` usa el
  reconocedor de Chrome, que **sí** envía el audio a Google.
- **Texto a voz** (apagado): Piper `es_MX-ald-medium`, cargado una vez y servido
  como WAV desde `/speak`. Piper no tiene voz colombiana; es_MX es lo más cercano.
- **Almacenamiento**: venv, caché de HF, pip, XDG y CUDA, todo bajo `/disk/$USER`
  y nunca en `$HOME`.
- **`serve.py`**: las cabeceras COOP/COEP que manda son un resto de cuando el
  modelo corría en el navegador; `python -m http.server 8080` sirve igual.

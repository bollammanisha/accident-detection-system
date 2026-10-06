import streamlit as st
import av
import numpy as np
from PIL import Image
import requests
import base64
import io
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

# Text model requested by you
CHAT_MODEL = "llama3.2"

# Vision model is needed because llama3.2 itself is text-only.
# Change this to any vision model installed in your Ollama.
VISION_MODEL = "llama3.2-vision:latest"

NUM_FRAMES = 8

SYSTEM_PROMPT = """
You are an AI accident detection assistant for dashcam videos.

Use the supplied accident-detection result as context.
Explain the result clearly and simply.

If an accident is detected:
- Explain why it may be an accident.
- Mention the confidence.
- Give basic safety advice.

If no accident is detected:
- Clearly say that no accident was detected.
- Do not claim that the result is 100% certain.

Never invent details that are not present in the detection result.
"""

# ============================================================
# OLLAMA CHECK
# ============================================================

def check_ollama():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        return response.ok
    except Exception:
        return False


def get_installed_models():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        if response.ok:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_frames(video_file, num_frames=NUM_FRAMES):
    video_file.seek(0)

    container = av.open(video_file)
    frames = []

    try:
        total_frames = container.streams.video[0].frames

        if total_frames and total_frames > 0:
            wanted = np.linspace(
                0,
                total_frames - 1,
                min(num_frames, total_frames),
                dtype=int
            )
            wanted = set(wanted.tolist())

            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    image = frame.to_image().convert("RGB")
                    frames.append(image)

                if len(frames) >= num_frames:
                    break
        else:
            for frame in container.decode(video=0):
                if len(frames) >= num_frames:
                    break
                frames.append(frame.to_image().convert("RGB"))

    finally:
        container.close()

    return frames


# ============================================================
# IMAGE -> BASE64
# ============================================================

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# OLLAMA VISION ANALYSIS
# ============================================================

def analyze_frame_with_ollama(image):
    image_b64 = image_to_base64(image)

    prompt = """
Analyze this dashcam image for possible road accidents.

Return ONLY valid JSON in this format:

{
  "accident": true or false,
  "confidence": number between 0 and 1,
  "reason": "short explanation"
}

Look for visible:
- vehicle collisions
- crashed vehicles
- severe impact
- vehicles overturned
- obvious accident scenes

If the image does not provide enough evidence, use:
"accident": false
and a lower confidence.

Do not invent an accident.
"""

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    data = response.json()
    text = data["message"]["content"].strip()

    # Remove markdown code fences if the model returns them
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {
            "accident": False,
            "confidence": 0.0,
            "reason": "The vision model returned an unreadable result."
        }

    return {
        "accident": bool(result.get("accident", False)),
        "confidence": float(result.get("confidence", 0.0)),
        "reason": str(result.get("reason", "No reason provided."))
    }


# ============================================================
# VIDEO PREDICTION
# ============================================================

def predict_video(frames):
    if not frames:
        return {
            "accident": False,
            "confidence": 0.0,
            "label": "No frames found",
            "reason": "Could not extract frames from the video.",
            "frame_results": []
        }

    frame_results = []

    progress = st.progress(0)

    for i, frame in enumerate(frames):
        try:
            result = analyze_frame_with_ollama(frame)
            frame_results.append(result)
        except Exception as e:
            frame_results.append({
                "accident": False,
                "confidence": 0.0,
                "reason": f"Vision model error: {e}"
            })

        progress.progress((i + 1) / len(frames))

    progress.empty()

    accident_results = [
        r for r in frame_results
        if r["accident"] and r["confidence"] >= 0.50
    ]

    if accident_results:
        best = max(
            accident_results,
            key=lambda x: x["confidence"]
        )

        return {
            "accident": True,
            "confidence": best["confidence"],
            "label": "Accident Detected",
            "reason": best["reason"],
            "frame_results": frame_results
        }

    # If no accident frame was found
    best_non_accident = max(
        frame_results,
        key=lambda x: x["confidence"]
    )

    return {
        "accident": False,
        "confidence": 1 - best_non_accident["confidence"],
        "label": "No Accident Detected",
        "reason": "No sampled frame provided sufficient evidence of an accident.",
        "frame_results": frame_results
    }


# ============================================================
# LLAMA 3.2 CHAT
# ============================================================

def ask_llama(user_messages, detection_context):
    detection_json = json.dumps(
        detection_context,
        indent=2
    )

    context_message = {
        "role": "system",
        "content": (
            SYSTEM_PROMPT
            + "\n\nCURRENT DETECTION RESULT:\n"
            + detection_json
        )
    }

    messages = [
        context_message,
        *user_messages
    ]

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="AI Accident Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AI Dashcam Accident Detection")
st.caption("Local Ollama • No API key • No PyTorch")

# ------------------------------------------------------------
# Ollama status
# ------------------------------------------------------------

if not check_ollama():
    st.error(
        "Ollama is not running. Open Ollama and make sure it is running "
        "before starting this application."
    )
    st.stop()

installed_models = get_installed_models()

with st.sidebar:
    st.header("⚙️ Ollama Settings")

    st.write("Chat model:")
    st.code(CHAT_MODEL)

    st.write("Vision model:")
    st.code(VISION_MODEL)

    st.write("Installed models:")
    if installed_models:
        for model in installed_models:
            st.write("•", model)
    else:
        st.warning("No Ollama models found.")

    st.info(
        "llama3.2 is used for the assistant chat. "
        "A vision model is required to inspect dashcam images."
    )

# ------------------------------------------------------------
# Model availability warning
# ------------------------------------------------------------

# Ollama usually returns names such as "llama3.2:latest".
# Treat "llama3.2" and "llama3.2:latest" as the same model.
chat_model_installed = any(
    model == CHAT_MODEL or model == f"{CHAT_MODEL}:latest"
    for model in installed_models
)

if not chat_model_installed:
    st.warning(
        f"'{CHAT_MODEL}' is not installed. Run: "
        f"ollama pull {CHAT_MODEL}"
    )

vision_model_installed = any(
    model == VISION_MODEL
    or model == f"{VISION_MODEL}:latest"
    or model.startswith(f"{VISION_MODEL}:")
    for model in installed_models
)

if not vision_model_installed:
    st.warning(
        f"'{VISION_MODEL}' is not installed. "
        f"Run: ollama pull {VISION_MODEL}"
    )

# ------------------------------------------------------------
# Video upload
# ------------------------------------------------------------

uploaded_video = st.file_uploader(
    "Upload a dashcam video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:

    st.subheader("🎥 Uploaded Video")

    st.video(uploaded_video)

    if st.button("🔍 Detect Accident", type="primary"):

        with st.spinner("Extracting video frames..."):
            frames = extract_frames(
                uploaded_video,
                NUM_FRAMES
            )

        if not frames:
            st.error("No frames could be extracted.")
            st.stop()

        st.success(
            f"Successfully extracted {len(frames)} frames."
        )

        # ----------------------------------------------------
        # Display sampled frames
        # ----------------------------------------------------

        st.subheader("🖼️ Sampled Frames")

        cols = st.columns(4)

        for i, frame in enumerate(frames):
            with cols[i % 4]:
                st.image(
                    frame,
                    caption=f"Frame {i + 1}",
                    use_container_width=True
                )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        st.subheader("🤖 Accident Detection")

        try:
            result = predict_video(frames)

            st.session_state["detection_result"] = result

            col1, col2 = st.columns(2)

            with col1:
                if result["accident"]:
                    st.error("🚨 ACCIDENT DETECTED")
                else:
                    st.success("✅ NO ACCIDENT DETECTED")

            with col2:
                st.metric(
                    "Confidence",
                    f"{result['confidence'] * 100:.1f}%"
                )

            st.write("**Reason:**")
            st.write(result["reason"])

            # ------------------------------------------------
            # Frame-by-frame results
            # ------------------------------------------------

            st.subheader("📊 Frame Results")

            for i, frame_result in enumerate(
                result["frame_results"]
            ):
                status = (
                    "🚨 Accident"
                    if frame_result["accident"]
                    else "✅ No accident"
                )

                st.write(
                    f"**Frame {i + 1}:** {status} | "
                    f"Confidence: "
                    f"{frame_result['confidence'] * 100:.1f}%"
                )

                st.caption(
                    frame_result["reason"]
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )

        except Exception as e:
            st.error(f"Detection error: {e}")


# ============================================================
# CHAT ASSISTANT
# ============================================================

st.divider()

st.subheader("💬 Llama 3.2 Accident Assistant")

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

for message in st.session_state["chat_messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input(
    "Ask about the accident detection result..."
)

if user_input:

    st.session_state["chat_messages"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    detection_context = st.session_state.get(
        "detection_result",
        {
            "accident": False,
            "confidence": 0,
            "label": "No video analyzed yet",
            "reason": "Upload and analyze a video first."
        }
    )

    try:
        with st.chat_message("assistant"):

            answer = ask_llama(
                st.session_state["chat_messages"],
                detection_context
            )

            st.write(answer)

        st.session_state["chat_messages"].append({
            "role": "assistant",
            "content": answer
        })

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to Ollama. "
            "Please make sure Ollama is running."
        )

    except Exception as e:
        st.error(f"Chat error: {e}")

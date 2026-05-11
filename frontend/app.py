# ============================================================
# 🧠 MentiMotive - Emotion Detection Frontend
# ============================================================

import streamlit as st
import requests
import time

# ============================================================
# Configuration
# ============================================================
FASTAPI_BASE_URL = "http://localhost:8000"
TEXT_ENDPOINT = f"{FASTAPI_BASE_URL}/classify/text"
VOICE_ENDPOINT = f"{FASTAPI_BASE_URL}/analyze/voice"
PARALLEL_ENDPOINT = f"{FASTAPI_BASE_URL}/analyze/parallel-sentiment"

# Chat endpoints
CHAT_TEXT_ENDPOINT = f"{FASTAPI_BASE_URL}/chat/text"
CHAT_VOICE_ENDPOINT = f"{FASTAPI_BASE_URL}/chat/voice"
CHAT_HISTORY_ENDPOINT = f"{FASTAPI_BASE_URL}/chat/history"
CHAT_INGEST_ENDPOINT = f"{FASTAPI_BASE_URL}/chat/ingest"
TTS_ENGINE_ENDPOINT = f"{FASTAPI_BASE_URL}/chat/tts/engine"

# ============================================================
# Page Setup
# ============================================================
st.set_page_config(
    page_title="MentiMotive - Mental Health Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Session State Initialization
# ============================================================
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# Custom Styling
# ============================================================
st.markdown("""
    <style>
    .main { background-color: #1a1a2e; }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338CA;
    }
    .result-box {
        background-color: #16213e;
        color: #e4e4e4;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #4F46E5;
        margin-top: 20px;
    }
    /* Dark theme for chat area */
    [data-testid="stChatMessageContent"] {
        background-color: #16213e;
        color: #e4e4e4;
        padding: 15px;
        border-radius: 10px;
    }
    /* User message */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: #0f3460;
    }
    /* Assistant message */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background-color: #16213e;
    }
    /* Override text colors */
    .stMarkdown, p, span, div {
        color: #e4e4e4 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    /* Input fields */
    .stTextInput input {
        background-color: #16213e;
        color: #e4e4e4;
        border: 1px solid #4F46E5;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================
st.title("🧠 MentiMotive")
st.markdown("### AI-Powered Mental Health Chatbot")
st.markdown("---")

# ============================================================
# Tab Navigation
# ============================================================
tab_chat, tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Text Emotion", "🎤 Voice Emotion", "⚡ Parallel Sentiment"])

# ============================================================
# TAB 0: Chat Interface
# ============================================================
with tab_chat:
    st.header("💬 Mental Health Chat Assistant")
    st.markdown("Chat with our empathetic AI assistant using voice or text")
    
    # Sidebar for session management
    with st.sidebar:
        st.markdown("### 🔧 Session Management")
        if st.session_state.chat_session_id:
            st.success(f"Session: {st.session_state.chat_session_id[:8]}...")
        else:
            st.info("No active session")
        
        if st.button("🗑️ Clear History", key="clear_history"):
            if st.session_state.chat_session_id:
                try:
                    response = requests.delete(f"{CHAT_HISTORY_ENDPOINT}/{st.session_state.chat_session_id}")
                    if response.status_code == 200:
                        st.session_state.chat_history = []
                        st.success("History cleared!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("No active session")
        
        if st.button("🔄 New Session", key="new_session"):
            st.session_state.chat_session_id = None
            st.session_state.chat_history = []
            st.success("New session started!")
            st.rerun()

        # TTS Engine Selection
        st.markdown("---")
        st.markdown("### 🎤 TTS Engine")
        st.markdown("Select text-to-speech engine for voice responses")

        # Initialize TTS engine state
        if "tts_engine" not in st.session_state:
            st.session_state.tts_engine = "Edge TTS (Cloud)"

        tts_engine = st.radio(
            "Engine:",
            ["Edge TTS (Cloud)", "TinyTTS (Local)"],
            index=0 if st.session_state.tts_engine == "Edge TTS (Cloud)" else 1,
            key="tts_engine_selector",
            help="Edge TTS: Fast, cloud-based, multiple voices\nTinyTTS: Local, offline, single voice"
        )

        # Update backend when engine changes
        if tts_engine != st.session_state.tts_engine:
            engine_value = "edge" if tts_engine == "Edge TTS (Cloud)" else "tiny"
            try:
                response = requests.post(TTS_ENGINE_ENDPOINT, params={"engine": engine_value})
                if response.status_code == 200:
                    st.session_state.tts_engine = tts_engine
                    st.success(f"✅ Switched to {tts_engine}")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to switch engine: {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

        if tts_engine == "Edge TTS (Cloud)":
            st.info("🌐 Using Edge TTS (Primary)")
        else:
            st.info("💻 Using TinyTTS (Backup)")

        # Document ingestion section
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base")
        st.markdown("Upload mental health documents to enhance AI responses")
        
        uploaded_docs = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        
        if uploaded_docs and st.button("📤 Upload Documents", key="upload_docs"):
            with st.spinner("Processing documents..."):
                try:
                    files = [("files", (file.name, file, file.type)) for file in uploaded_docs]
                    response = requests.post(CHAT_INGEST_ENDPOINT, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Processed {result['files_processed']} files, created {result['chunks_created']} chunks")
                    else:
                        st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            st.success("New session started!")
            st.rerun()
    
    # Chat display area
    st.markdown("### 💭 Conversation")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "user":
                    source = msg.get("source", "text")
                    with st.chat_message("user"):
                        st.markdown(content)
                        if source == "voice":
                            st.caption("🎤 Voice message")
                        # Show emotions if available
                        if "emotions" in msg and msg["emotions"]:
                            with st.expander("🎭 Emotions"):
                                emotions = msg["emotions"]
                                if "audio" in emotions:
                                    st.write(f"🎵 Voice: {emotions['audio']['emotion']} ({emotions['audio']['confidence']*100:.1f}%)")
                                if "text" in emotions:
                                    st.write(f"📝 Text: {emotions['text']['emotion']} ({emotions['text']['confidence']*100:.1f}%)")
                
                elif role == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(content)

                        # Show audio player if audio is available
                        if "audio" in msg and msg["audio"]:
                            import base64
                            audio_base64 = msg["audio"]
                            audio_format = msg.get("audio_format", "mp3")
                            tts_engine = msg.get("tts_engine", "edge")

                            # Decode base64 audio
                            audio_bytes = base64.b64decode(audio_base64)

                            # Display audio player with autoplay
                            st.audio(audio_bytes, format=f"audio/{audio_format}", autoplay=True)
                            st.caption(f"🔊 Audio response (Engine: {tts_engine})")

                        # Show RAG sources if available
                        if "rag_sources" in msg and msg["rag_sources"]:
                            with st.expander("📚 Sources"):
                                for source in msg["rag_sources"]:
                                    st.write(f"- {source}")
        else:
            st.info("👋 Start a conversation! Type a message or record audio below.")
    
    # Input area
    st.markdown("---")
    st.markdown("### 📤 Send Message")
    
    # Choose input method
    input_mode = st.radio(
        "Input mode:",
        ["💬 Text", "🎤 Voice"],
        horizontal=True,
        key="chat_input_mode"
    )
    
    if input_mode == "💬 Text":
        # Text input
        col1, col2 = st.columns([4, 1])
        with col1:
            user_message = st.text_input(
                "Your message",
                placeholder="Type your message here...",
                key="chat_text_input",
                label_visibility="collapsed"
            )
        with col2:
            send_button = st.button("📤 Send", key="send_text_chat", use_container_width=True)
        
        if send_button and user_message.strip():
            with st.spinner("💭 Thinking..."):
                try:
                    payload = {"text": user_message}
                    if st.session_state.chat_session_id:
                        payload["session_id"] = st.session_state.chat_session_id
                    
                    response = requests.post(CHAT_TEXT_ENDPOINT, json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Update session ID
                        st.session_state.chat_session_id = result["session_id"]
                        
                        # Add user message to history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_message,
                            "source": "text",
                            "emotions": result.get("emotions", {})
                        })
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["response"],
                            "rag_sources": result.get("rag_sources", [])
                        })
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Error {response.status_code}: {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure the FastAPI server is running on port 8000")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    else:  # Voice input
        st.info("🎙️ Click to record your message")
        
        recorded_audio = st.audio_input("Record message", key="chat_voice_input")
        
        if recorded_audio is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.audio(recorded_audio)
            with col2:
                if st.button("📤 Send Voice", key="send_voice_chat", use_container_width=True):
                    with st.spinner("🎤 Processing voice..."):
                        try:
                            # Prepare form data
                            files = {"audio": ("recording.webm", recorded_audio, "audio/webm")}
                            data = {}
                            if st.session_state.chat_session_id:
                                data["session_id"] = st.session_state.chat_session_id
                            
                            response = requests.post(CHAT_VOICE_ENDPOINT, files=files, data=data)
                            
                            if response.status_code == 200:
                                result = response.json()

                                # Update session ID
                                st.session_state.chat_session_id = result["session_id"]

                                # Add user message to history (use transcript)
                                st.session_state.chat_history.append({
                                    "role": "user",
                                    "content": result.get("transcript", "[Voice message]"),
                                    "source": "voice",
                                    "emotions": result.get("emotions", {})
                                })

                                # Add assistant response to history with audio
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": result["response"],
                                    "rag_sources": result.get("rag_sources", []),
                                    "audio": result.get("audio"),  # Base64 audio
                                    "audio_format": result.get("audio_format", "mp3"),
                                    "tts_engine": result.get("tts_engine", "edge")
                                })

                                st.rerun()
                            else:
                                st.error(f"❌ Error {response.status_code}: {response.text}")
                        
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Cannot connect to backend. Make sure the FastAPI server is running on port 8000")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

# ============================================================
# TAB 1: Text Emotion Analysis
# ============================================================
with tab1:
    st.header("📝 Text Emotion Analysis")
    st.markdown("Enter text to detect the emotion expressed")
    
    # Text input
    user_text = st.text_area(
        "Your Text",
        placeholder="Type or paste your text here...",
        height=150,
        key="text_input"
    )
    
    # Analyze button
    if st.button("🔍 Analyze Text", key="analyze_text"):
        if not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze")
        else:
            with st.spinner("Analyzing emotion..."):
                try:
                    response = requests.post(
                        TEXT_ENDPOINT,
                        json={"text": user_text}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display result
                        st.markdown("### 🎯 Analysis Result")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                label="Detected Emotion",
                                value=result["emotion"].upper()
                            )
                        
                        with col2:
                            confidence_pct = result["confidence_score"] * 100
                            st.metric(
                                label="Confidence",
                                value=f"{confidence_pct:.1f}%"
                            )
                        
                        # Emotion emoji mapping
                        emotion_emojis = {
                            "joy": "😊",
                            "sadness": "😢",
                            "anger": "😠",
                            "fear": "😨",
                            "surprise": "😲",
                            "love": "❤️",
                            "neutral": "😐"
                        }
                        
                        emoji = emotion_emojis.get(result["emotion"].lower(), "🎭")
                        st.markdown(f"### {emoji} {result['emotion'].title()}")
                        
                    else:
                        st.error(f"❌ Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure the FastAPI server is running on port 8000")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============================================================
# TAB 2: Voice Emotion Analysis
# ============================================================
with tab2:
    st.header("🎤 Voice Emotion Analysis")
    st.markdown("Upload an audio file or record audio to analyze voice and speech emotions")
    
    # Choose input method
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload File", "🎙️ Record Audio"],
        horizontal=True,
        key="voice_input_method"
    )
    
    audio_file = None
    
    if input_method == "📁 Upload File":
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="audio_upload"
        )
        
        if uploaded_file is not None:
            audio_file = uploaded_file
            st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")
    
    else:  # Record Audio
        st.info("🎙️ Click the button below to start/stop recording (WebM format, resampled to 16kHz during processing)")
        
        # Use st.audio_input for recording
        recorded_audio = st.audio_input("Record audio", key="voice_recorder")
        
        if recorded_audio is not None:
            audio_file = recorded_audio
            st.audio(recorded_audio)
            st.success("✅ Audio recorded successfully!")
    
    if audio_file is not None:
        # Analyze button
        if st.button("🔍 Analyze Voice", key="analyze_voice"):
            with st.spinner("Analyzing voice emotion (this may take a moment)..."):
                try:
                    # Prepare file for upload
                    # Recorded audio is WebM format, uploaded files keep their original format
                    mime_type = audio_file.type if hasattr(audio_file, 'type') else "audio/webm"
                    filename = audio_file.name if hasattr(audio_file, 'name') else "recording.webm"
                    files = {"file": (filename, audio_file, mime_type)}
                    
                    response = requests.post(VOICE_ENDPOINT, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display transcript
                        st.markdown("### 📝 Transcript")
                        st.info(result["transcript"] or "No speech detected")
                        
                        # Display results
                        st.markdown("### 🎯 Emotion Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📝 Text Emotion")
                            st.metric(
                                label="From Speech Content",
                                value=result["text_emotion"]["label"].upper()
                            )
                            st.progress(result["text_emotion"]["confidence"])
                            st.caption(f"Confidence: {result['text_emotion']['confidence']*100:.1f}%")
                        
                        with col2:
                            st.markdown("#### 🎵 Voice Emotion")
                            st.metric(
                                label="From Voice Tone",
                                value=result["voice_emotion"]["label"].upper()
                            )
                            st.progress(result["voice_emotion"]["confidence"])
                            st.caption(f"Confidence: {result['voice_emotion']['confidence']*100:.1f}%")
                        
                        # Show all voice emotions
                        if "voice_emotion_all" in result and len(result["voice_emotion_all"]) > 1:
                            st.markdown("#### 🎼 All Detected Voice Emotions")
                            for emotion in result["voice_emotion_all"]:
                                st.write(f"- **{emotion['label'].title()}**: {emotion['confidence']*100:.1f}%")
                        
                    else:
                        st.error(f"❌ Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure the FastAPI server is running on port 8000")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============================================================
# TAB 3: Parallel Sentiment Analysis
# ============================================================
with tab3:
    st.header("⚡ Parallel Sentiment Analysis")
    st.markdown("Upload or record audio to analyze with **parallel processing** for maximum speed")
    st.info("🚀 This route processes **Path 1** (STT → Text Emotion) and **Path 2** (Audio Emotion) in parallel!")
    
    # Choose input method
    input_method_parallel = st.radio(
        "Choose input method:",
        ["📁 Upload File", "🎙️ Record Audio"],
        horizontal=True,
        key="parallel_input_method"
    )
    
    audio_file_parallel = None
    
    if input_method_parallel == "📁 Upload File":
        # File uploader
        uploaded_file_parallel = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="audio_upload_parallel"
        )
        
        if uploaded_file_parallel is not None:
            audio_file_parallel = uploaded_file_parallel
            st.audio(uploaded_file_parallel, format=f"audio/{uploaded_file_parallel.type.split('/')[-1]}")
    
    else:  # Record Audio
        st.info("🎙️ Click the button below to start/stop recording (WebM format, resampled to 16kHz during processing)")
        
        # Use st.audio_input for recording
        recorded_audio_parallel = st.audio_input("Record audio", key="parallel_recorder")
        
        if recorded_audio_parallel is not None:
            audio_file_parallel = recorded_audio_parallel
            st.audio(recorded_audio_parallel)
            st.success("✅ Audio recorded successfully!")
    
    if audio_file_parallel is not None:
        # Analyze button
        if st.button("⚡ Analyze (Parallel)", key="analyze_parallel"):
            with st.spinner("Processing in parallel mode..."):
                try:
                    # Prepare file for upload
                    # Recorded audio is WebM format, uploaded files keep their original format
                    mime_type = audio_file_parallel.type if hasattr(audio_file_parallel, 'type') else "audio/webm"
                    filename = audio_file_parallel.name if hasattr(audio_file_parallel, 'name') else "recording.webm"
                    files = {"file": (filename, audio_file_parallel, mime_type)}
                    
                    response = requests.post(PARALLEL_ENDPOINT, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display transcript
                        st.markdown("### 📝 Transcript")
                        st.info(result["transcript"] or "No speech detected")
                        
                        # Display parallel processing results
                        st.markdown("### 🎯 Parallel Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📝 Path 1: Text Emotion")
                            st.caption("(Audio → STT → Text Sentiment)")
                            st.metric(
                                label="Emotion",
                                value=result["path1_text_emotion"]["label"].upper()
                            )
                            st.progress(result["path1_text_emotion"]["confidence"])
                            st.caption(f"Confidence: {result['path1_text_emotion']['confidence']*100:.1f}%")
                        
                        with col2:
                            st.markdown("#### 🎵 Path 2: Audio Emotion")
                            st.caption("(Audio → Audio Sentiment)")
                            st.metric(
                                label="Emotion",
                                value=result["path2_audio_emotion"]["label"].upper()
                            )
                            st.progress(result["path2_audio_emotion"]["confidence"])
                            st.caption(f"Confidence: {result['path2_audio_emotion']['confidence']*100:.1f}%")
                        
                        # Show all audio emotions
                        if "path2_audio_emotion_all" in result and len(result["path2_audio_emotion_all"]) > 1:
                            st.markdown("#### 🎼 All Detected Audio Emotions (Path 2)")
                            for emotion in result["path2_audio_emotion_all"]:
                                st.write(f"- **{emotion['label'].title()}**: {emotion['confidence']*100:.1f}%")
                        
                        # Show performance stats
                        if "stats" in result:
                            st.markdown("### ⏱️ Performance Statistics")
                            stats = result["stats"]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Time", f"{stats['total_processing_ms']:.0f} ms")
                            with col2:
                                st.metric("STT Time", f"{stats['breakdown_ms']['stt_whisper']:.0f} ms")
                            with col3:
                                st.metric("Parallel Status", stats.get('parallel_check', 'N/A'))
                            
                            with st.expander("📊 Detailed Breakdown"):
                                st.json(stats["breakdown_ms"])
                        
                    else:
                        st.error(f"❌ Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure the FastAPI server is running on port 8000")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; font-size: 14px;'>
        <p>🧠 MentiMotive - AI Mental Health Chatbot</p>
        <p>Powered by FastAPI + LangChain + Gemini + ChromaDB</p>
        <p>Emotion Detection: DistilBERT + Wav2Vec2</p>
    </div>
    """,
    unsafe_allow_html=True
)
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"
    if "transcript" in st.session_state:
        del st.session_state.transcript

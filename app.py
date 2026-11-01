import io
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from scipy.signal import butter, sosfilt

# ------------------------------------------------------------
# SERENITY | Procedural meditation soundscape prototype
# ------------------------------------------------------------
SR = 22050
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def filter_noise(noise: np.ndarray, cutoff: float, filter_type: str) -> np.ndarray:
    sos = butter(4, cutoff, btype=filter_type, fs=SR, output="sos")
    return sosfilt(sos, noise)


def generate_soundscape(theme: str, duration: int = 60, seed: int = 42) -> np.ndarray:
    """Generate a natural-style ambience using the project's original method."""
    rng = np.random.default_rng(seed)
    n = int(SR * duration)
    noise = rng.normal(0, 1, n)
    t = np.arange(n) / SR

    if theme == "Rain":
        rain = filter_noise(noise, 2500, "high")
        rain = rain / (np.std(rain) + 1e-8)
        ambience = 0.22 * rain
    elif theme == "Ocean":
        waves = filter_noise(noise, 400, "low")
        waves = waves / (np.std(waves) + 1e-8)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.08 * t)
        ambience = 0.35 * waves * modulation
    elif theme == "Forest":
        wind = filter_noise(noise, 600, "low")
        wind = wind / (np.std(wind) + 1e-8)
        ambience = 0.18 * wind
        for start in [5, 15, 28, 42, 52]:
            idx = (t >= start) & (t <= start + 1.0)
            local_t = t[idx] - start
            chirp = np.sin(2 * np.pi * (1500 * local_t + 300 * local_t**2))
            envelope = np.sin(np.pi * local_t) ** 2
            ambience[idx] += 0.035 * chirp * envelope
    else:
        raise ValueError("Unknown soundscape theme")

    fade_length = min(int(SR * 2), max(1, n // 2))
    fade = np.linspace(0, 1, fade_length)
    ambience[:fade_length] *= fade
    ambience[-fade_length:] *= fade[::-1]
    ambience = ambience / (np.max(np.abs(ambience)) + 1e-8) * 0.75
    return ambience.astype(np.float32)


def analyze_audio_quality(audio: np.ndarray, sr: int = SR) -> dict:
    """Calculate the same core quality indicators used in the notebook."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Audio is empty.")
    if not np.all(np.isfinite(audio)):
        return {
            "RMS Energy": np.nan, "Peak Amplitude": np.nan,
            "Clipping (%)": np.nan, "RMS Variation": np.nan,
            "Spectral Centroid (Hz)": np.nan,
            "Loop Boundary Difference": np.nan,
            "Quality Status": "Invalid Audio",
        }

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    clipping = (np.sum(np.abs(audio) >= 0.99) / len(audio)) * 100
    rms_frames = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    rms_variation = np.std(rms_frames)
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    window_size = min(int(0.1 * sr), len(audio) // 2)
    loop_difference = np.sqrt(np.mean((audio[:window_size] - audio[-window_size:]) ** 2))

    if clipping > 0:
        status = "Check Clipping"
    elif rms < 0.001:
        status = "Very Quiet"
    else:
        status = "Passed Basic Checks"

    return {
        "RMS Energy": round(float(rms), 5),
        "Peak Amplitude": round(float(peak), 5),
        "Clipping (%)": round(float(clipping), 5),
        "RMS Variation": round(float(rms_variation), 5),
        "Spectral Centroid (Hz)": round(float(centroid), 2),
        "Loop Boundary Difference": round(float(loop_difference), 5),
        "Quality Status": status,
    }


def build_test_results(audio_by_theme: dict) -> pd.DataFrame:
    """Reproduce the normal, quiet, clipped and sudden-spike test cases."""
    rain = audio_by_theme["Rain"]
    forest = audio_by_theme["Forest"]
    quiet = rain * 0.001
    clipped = rain.copy()
    clipped[:min(5000, len(clipped))] = np.clip(clipped[:min(5000, len(clipped))] * 5, -1.0, 1.0)
    spike = forest.copy()
    spike[len(spike) // 2] = 1.0

    cases = {
        "Normal Rain": rain,
        "Normal Ocean": audio_by_theme["Ocean"],
        "Very Quiet Audio": quiet,
        "Clipped Audio": clipped,
        "Sudden Spike Audio": spike,
    }
    rows = []
    for name, audio in cases.items():
        metrics = analyze_audio_quality(audio)
        differences = np.abs(np.diff(audio))
        threshold = max(0.15, 8 * np.median(differences)) if differences.size else 0.15
        transient_count = int(np.sum(differences > threshold))
        if not np.all(np.isfinite(audio)):
            status = "FAIL - Invalid values"
        elif metrics["Clipping (%)"] > 0:
            status = "WARNING - Clipping detected"
        elif metrics["RMS Energy"] < 0.001:
            status = "WARNING - Very low volume"
        elif transient_count > 0:
            status = "WARNING - Sudden transient"
        else:
            status = "PASS"
        rows.append({
            "Test Case": name,
            "RMS Energy": metrics["RMS Energy"],
            "Peak Amplitude": metrics["Peak Amplitude"],
            "Clipping (%)": metrics["Clipping (%)"],
            "Transient Count": transient_count,
            "Test Status": status,
        })
    return pd.DataFrame(rows)


def wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return buffer.getvalue()


def make_waveform(audio: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    time = np.arange(len(audio)) / SR
    # Downsample for responsive plotting only; audio itself is unchanged.
    step = max(1, len(audio) // 12000)
    ax.plot(time[::step], audio[::step], linewidth=0.55)
    ax.set(title=title, xlabel="Time (seconds)", ylabel="Amplitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def make_spectrogram(audio: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    image = librosa.display.specshow(spec, sr=SR, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


# ---------------------------- UI ----------------------------
st.set_page_config(
    page_title="Serenity | Meditation Soundscapes",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    :root {
        --serenity-ink: #173b35;
        --serenity-muted: #63756e;
        --serenity-green: #32665a;
        --serenity-line: #e4ebe6;
        --serenity-surface: rgba(255, 255, 255, 0.88);
    }
    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(205, 226, 214, 0.38), transparent 34rem),
            #f5f7f4;
        color: #233a34;
    }
    [data-testid="stHeader"] { background: rgba(245, 247, 244, 0.82); }
    [data-testid="stMain"] > div { padding-top: 2rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #edf3ee 0%, #e8f0ea 100%);
        border-right: 1px solid #dfe9e1;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: var(--serenity-ink);
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: 0.25rem 0.55rem;
        transition: background 160ms ease, color 160ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(50, 102, 90, 0.08);
    }
    .hero {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        padding: clamp(1.7rem, 4vw, 2.7rem);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 26px;
        background: linear-gradient(118deg, #153a33 0%, #28584d 58%, #397365 100%);
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 18px 44px rgba(26, 63, 53, 0.16);
    }
    .hero::after {
        content: "";
        position: absolute;
        z-index: -1;
        width: 21rem;
        height: 21rem;
        right: -5rem;
        top: -11rem;
        border: 1px solid rgba(232, 247, 237, 0.17);
        border-radius: 50%;
        box-shadow: 0 0 0 2.4rem rgba(232, 247, 237, 0.035),
                    0 0 0 5rem rgba(232, 247, 237, 0.025);
    }
    .hero-kicker {
        margin-bottom: 0.55rem;
        color: #b7d9c8;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: white;
        margin: 0;
        font-size: clamp(2.4rem, 5vw, 3.35rem);
        font-weight: 700;
        letter-spacing: -0.055em;
        line-height: 1.08;
    }
    .hero p {
        max-width: 39rem;
        color: #e0eee6;
        margin: 0.85rem 0 1.25rem;
        font-size: 1.02rem;
        line-height: 1.65;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 0.75rem;
        border: 1px solid rgba(231, 247, 237, 0.22);
        border-radius: 999px;
        background: rgba(231, 247, 237, 0.09);
        color: #edf7f0;
        font-size: 0.78rem;
        font-weight: 600;
    }
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: var(--serenity-ink);
        letter-spacing: -0.035em;
    }
    [data-testid="stMarkdownContainer"] h2 { margin-top: 0.35rem; }
    [data-testid="stMarkdownContainer"] p { line-height: 1.65; }
    .small-note { color: var(--serenity-muted); font-size: .88rem; }
    div[data-testid="stMetric"] {
        min-height: 6.3rem;
        background: var(--serenity-surface);
        padding: 15px 16px;
        border-radius: 16px;
        border: 1px solid var(--serenity-line);
        box-shadow: 0 5px 16px rgba(34, 62, 49, 0.045);
    }
    div[data-testid="stMetric"] {
        transition: transform 160ms ease, box-shadow 160ms ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 9px 22px rgba(34, 62, 49, 0.08);
    }
    div[data-testid="stMetricLabel"] { color: var(--serenity-muted); }
    div[data-testid="stMetricValue"] { color: var(--serenity-ink); }
    [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid rgba(50, 102, 90, 0.12);
    }
    [data-testid="stAudio"],
    [data-testid="stFileUploader"],
    [data-testid="stDataFrame"] {
        border-radius: 14px;
    }
    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--serenity-line);
        background: white;
    }
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        min-height: 2.8rem;
        border-radius: 11px;
        font-weight: 600;
        transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    }
    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button[kind="primary"] {
        border: 1px solid #28584d;
        background: linear-gradient(135deg, #28584d, #397365);
        box-shadow: 0 5px 14px rgba(40, 88, 77, 0.17);
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(40, 88, 77, 0.12);
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        border-radius: 10px;
    }
    @media (max-width: 640px) {
        [data-testid="stMain"] > div { padding-top: 1rem; }
        .hero { border-radius: 20px; margin-bottom: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-kicker">A quieter space, made by you</div>
  <h1>🌿 Serenity</h1>
  <p>Create a gentle ambient soundscape, listen mindfully, and explore its sound quality.</p>
  <span class="hero-tag">♫ &nbsp; Procedural audio studio</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Project navigation")
    page = st.radio(
        "Choose a section",
        ["Soundscape Studio", "Quality & Visual Analysis", "Testing", "Tool Comparison", "Listener Feedback", "About & Ethics"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Project prototype • Python • Streamlit")
    st.caption("Audio sample rate: 22,050 Hz")

if "audio_bank" not in st.session_state:
    st.session_state.audio_bank = {}
if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = "Rain"
if "duration" not in st.session_state:
    st.session_state.duration = 60

if page == "Soundscape Studio":
    st.markdown("## Create your ambience")
    st.write("Choose a natural sound, set its duration and generate a WAV sample.")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        theme = st.selectbox("Soundscape", ["Rain", "Ocean", "Forest"], key="theme_choice")
    with c2:
        duration = st.select_slider("Duration", options=[10, 30, 60], value=60, format_func=lambda x: f"{x} seconds")
    with c3:
        seed = st.number_input("Generation seed", min_value=0, max_value=999999, value=42, step=1)

    descriptions = {
        "Rain": "Soft filtered rainfall texture.",
        "Ocean": "Low-frequency ambience with slow wave-like modulation.",
        "Forest": "Low-frequency wind texture with short synthetic bird-like chirps.",
    }
    st.info(f"**{theme}:** {descriptions[theme]}")

    if st.button("✨ Generate soundscape", type="primary", use_container_width=True):
        with st.spinner("Generating audio and calculating quality indicators..."):
            audio = generate_soundscape(theme, int(duration), int(seed))
            metrics = analyze_audio_quality(audio)
            st.session_state.audio_bank[theme] = {
                "audio": audio, "duration": int(duration), "seed": int(seed), "metrics": metrics
            }
            st.session_state.selected_theme = theme
            filename = OUTPUT_DIR / f"{theme.lower()}_soundscape.wav"
            sf.write(filename, audio, SR)
        st.success(f"{theme} soundscape generated and saved.")

    if theme in st.session_state.audio_bank:
        item = st.session_state.audio_bank[theme]
        st.markdown("### Listen")
        st.audio(wav_bytes(item["audio"]), format="audio/wav")
        st.download_button(
            "Download WAV",
            data=wav_bytes(item["audio"]),
            file_name=f"{theme.lower()}_soundscape.wav",
            mime="audio/wav",
            use_container_width=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Duration", f"{item['duration']} s")
        m2.metric("Sample rate", f"{SR:,} Hz")
        m3.metric("Quality status", item["metrics"]["Quality Status"])
        st.caption("The generated sound is procedural audio, not output from a trained generative-audio model.")
    else:
        st.warning("Generate this soundscape to enable playback and download.")

    if st.session_state.audio_bank:
        st.markdown("### Generated in this session")
        rows = []
        for name, item in st.session_state.audio_bank.items():
            rows.append({"Soundscape": name, "Duration (s)": item["duration"], "Seed": item["seed"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

elif page == "Quality & Visual Analysis":
    st.markdown("## Audio-quality analysis")
    if not st.session_state.audio_bank:
        st.info("Generate at least one soundscape in Soundscape Studio first.")
    else:
        theme = st.selectbox("Choose generated audio", list(st.session_state.audio_bank.keys()))
        audio = st.session_state.audio_bank[theme]["audio"]
        metrics = analyze_audio_quality(audio)
        st.caption("These indicators describe signal properties; they do not by themselves prove listener comfort or seamless looping.")
        metric_cols = st.columns(3)
        metric_items = list(metrics.items())
        for idx, (label, value) in enumerate(metric_items):
            metric_cols[idx % 3].metric(label, str(value))
        st.markdown("### Waveform")
        fig = make_waveform(audio, f"{theme} soundscape — waveform")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("### Spectrogram")
        fig = make_spectrogram(audio, f"{theme} soundscape — spectrogram")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        csv = pd.DataFrame([{"Soundscape": theme, **metrics}]).to_csv(index=False).encode("utf-8")
        st.download_button("Download quality metrics (CSV)", csv, f"{theme.lower()}_quality_metrics.csv", "text/csv")

elif page == "Testing":
    st.markdown("## Test plan and measured results")
    st.write("The application reproduces the project's normal, edge and failure-oriented test cases.")
    if not all(k in st.session_state.audio_bank for k in ["Rain", "Ocean", "Forest"]):
        st.warning("Generate Rain, Ocean and Forest in Soundscape Studio first. The test suite uses all three.")
    else:
        audio_by_theme = {k: v["audio"] for k, v in st.session_state.audio_bank.items()}
        results = build_test_results(audio_by_theme)
        st.dataframe(results, hide_index=True, use_container_width=True)
        st.caption("PASS/WARNING labels follow the prototype's simple threshold checks. They are not a clinical or perceptual quality certification.")
        st.download_button(
            "Download test results (CSV)",
            results.to_csv(index=False).encode("utf-8"),
            "soundscape_test_results.csv",
            "text/csv",
        )
        st.markdown("### Test-case coverage")
        st.markdown("""
        - **Normal cases:** Rain and Ocean generated audio.
        - **Edge case:** Very quiet audio created by scaling Rain to 0.001.
        - **Failure-oriented case:** A section of Rain is amplified and clipped.
        - **Transient case:** One sample in Forest audio is set to a sudden peak.
        """)

elif page == "Tool Comparison":
    st.markdown("## Audio-generation tool comparison")
    st.write("Compare a Python-generated sample with a Stable Audio WAV export using the same analysis functions.")
    if "Rain" not in st.session_state.audio_bank:
        st.warning("Generate Rain in Soundscape Studio first.")
    else:
        uploaded = st.file_uploader("Upload your Stable Audio rain sample (WAV)", type=["wav"])
        if uploaded is not None:
            stable, source_sr = sf.read(uploaded)
            if stable.ndim > 1:
                stable = np.mean(stable, axis=1)
            stable = librosa.resample(stable.astype(np.float32), orig_sr=source_sr, target_sr=SR)
            duration_s = 10
            py_audio = st.session_state.audio_bank["Rain"]["audio"][:SR * duration_s]
            stable = stable[:SR * duration_s]
            if len(stable) == 0:
                st.error("The uploaded WAV file contains no audio samples.")
            else:
                results = []
                for tool_name, sample in [("Python Generator", py_audio), ("Stable Audio", stable)]:
                    met = analyze_audio_quality(sample)
                    results.append({
                        "Tool": tool_name,
                        "Duration (seconds)": round(len(sample) / SR, 2),
                        **{k: met[k] for k in [
                            "RMS Energy", "Peak Amplitude", "Clipping (%)",
                            "RMS Variation", "Spectral Centroid (Hz)", "Loop Boundary Difference"
                        ]}
                    })
                comparison = pd.DataFrame(results)
                st.dataframe(comparison, hide_index=True, use_container_width=True)
                st.caption("Interpret these measurements cautiously. They describe the selected samples and do not establish that one tool is universally better.")
                metric = st.selectbox("Visualize metric", comparison.columns[2:])
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.bar(comparison["Tool"], comparison[metric])
                ax.set_ylabel(metric)
                ax.set_title(f"Comparison — {metric}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.download_button(
                    "Download comparison results (CSV)",
                    comparison.to_csv(index=False).encode("utf-8"),
                    "audio_tool_comparison.csv",
                    "text/csv",
                )
        else:
            st.info("Upload the Stable Audio WAV sample you generated with the same rain prompt. This comparison is optional until that sample is available.")

elif page == "Listener Feedback":
    st.markdown("## Human listening evaluation")
    st.write("Collect real listener ratings for each soundscape. Do not enter example or assumed scores.")
    with st.form("listener_feedback"):
        listener = st.text_input("Anonymous listener ID", placeholder="e.g. Listener 01")
        sound_name = st.selectbox("Soundscape", ["Rain", "Ocean", "Forest"])
        st.caption("Rate each item from 1 (very low) to 5 (very high).")
        naturalness = st.slider("Naturalness", 1, 5, 3)
        relaxation = st.slider("Relaxation", 1, 5, 3)
        consistency = st.slider("Sound consistency", 1, 5, 3)
        suitability = st.slider("Meditation suitability", 1, 5, 3)
        overall = st.slider("Overall quality", 1, 5, 3)
        consent = st.checkbox("I have explained the purpose and received the listener's voluntary permission.")
        submitted = st.form_submit_button("Save feedback")
    if submitted:
        if not listener.strip():
            st.error("Enter an anonymous listener ID.")
        elif not consent:
            st.error("Confirm voluntary permission before saving feedback.")
        else:
            record = {
                "Listener": listener.strip(), "Soundscape": sound_name,
                "Naturalness": naturalness, "Relaxation": relaxation,
                "Sound Consistency": consistency,
                "Meditation Suitability": suitability,
                "Overall Quality": overall,
            }
            path = OUTPUT_DIR / "human_evaluation.csv"
            new_df = pd.DataFrame([record])
            if path.exists():
                old_df = pd.read_csv(path)
                new_df = pd.concat([old_df, new_df], ignore_index=True)
            new_df.to_csv(path, index=False)
            st.success("Feedback saved. Only actual voluntary ratings should be included in your report.")
    feedback_path = OUTPUT_DIR / "human_evaluation.csv"
    if feedback_path.exists():
        feedback = pd.read_csv(feedback_path)
        st.markdown("### Collected responses")
        st.dataframe(feedback, hide_index=True, use_container_width=True)
        st.markdown("### Average ratings")
        score_cols = ["Naturalness", "Relaxation", "Sound Consistency", "Meditation Suitability", "Overall Quality"]
        averages = feedback.groupby("Soundscape")[score_cols].mean().round(2)
        st.dataframe(averages, use_container_width=True)
        st.download_button("Download feedback CSV", feedback.to_csv(index=False).encode("utf-8"), "human_evaluation.csv", "text/csv")
        st.download_button("Download average ratings CSV", averages.to_csv().encode("utf-8"), "average_ratings.csv", "text/csv")
    else:
        st.info("No listener feedback has been saved in this application session yet.")

elif page == "About & Ethics":
    st.markdown("## About the project")
    st.write(
        "Serenity is an experimental prototype that generates natural-style ambient audio "
        "using Python signal-processing methods and evaluates selected signal-quality indicators."
    )
    st.markdown("### Project workflow")
    st.markdown("""
    1. Select Rain, Ocean or Forest.
    2. Generate a soundscape using procedural audio methods.
    3. Listen to and download the WAV output.
    4. Examine numerical indicators, waveform and spectrogram.
    5. Run normal, quiet, clipped and transient test cases.
    6. Optionally compare the Rain output with a Stable Audio WAV sample.
    7. Collect voluntary, anonymous listener ratings.
    """)
    st.markdown("### Ethical considerations")
    st.markdown("""
    - **Copyright and licensing:** The Python-generated samples are created by this prototype. Check the terms and licence of any external audio-generation service before reuse or redistribution.
    - **Consent:** Obtain voluntary permission before collecting listener ratings.
    - **Privacy:** Use anonymous listener IDs and avoid collecting unnecessary personal information.
    - **Fairness:** Use the same rating questions for each soundscape; individual preferences may differ.
    - **Listener safety:** Encourage comfortable playback volume and allow listeners to stop if uncomfortable.
    - **Disclosure and oversight:** Explain that this prototype uses procedural Python audio generation and automated signal analysis, not a trained generative-audio model. Human listening remains important.
    """)
    st.markdown("### Limitations")
    st.markdown("""
    - The soundscapes are simplified synthetic ambience, not field recordings.
    - RMS variation is only a rough measure of energy consistency.
    - Spectral centroid is a brightness indicator, not a direct measure of harshness.
    - The loop-boundary difference is a rough comparison of the start and end windows; it does not prove a seamless loop.
    - Automated metrics cannot replace human listening evaluation.
    - The prototype is for general relaxation use and is not a medical or therapeutic product.
    """)
    st.info("Project title: AI Soundscape Generator for Meditation | Implementation: Python + Streamlit")

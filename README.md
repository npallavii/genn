# Serenity — AI Soundscape Generator for Meditation

A small Streamlit application for demonstrating the project's procedural Python soundscape generator and audio-quality evaluation.

## What it demonstrates

- Generate Rain, Ocean and Forest ambience.
- Listen to and download generated WAV files.
- Calculate RMS energy, peak amplitude, clipping percentage, RMS variation, spectral centroid and a rough loop-boundary difference.
- View waveform and spectrogram plots.
- Run normal, very quiet, clipped and sudden-spike test cases.
- Optionally compare a Python Rain sample with a Stable Audio WAV sample.
- Collect voluntary listener feedback and calculate average ratings.
- Review the project's ethics and limitations.

## Important project disclosure

The current implementation is a **procedural Python-based soundscape generator with automated signal analysis**. It is not a trained machine-learning or generative-audio model. The project title is retained as supplied.

The numeric audio indicators are descriptive signal measurements. They do not establish perceived relaxation, listener comfort, or seamless looping. Human listening feedback is included separately.

## Run in VS Code (Windows)

1. Install Python 3.10 or 3.11 from https://www.python.org/downloads/
2. Extract this ZIP folder.
3. Open the extracted `serenity_soundscape_vscode` folder in VS Code.
4. Open **Terminal → New Terminal**.
5. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   ```

6. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt terminal and run:

   ```cmd
   .venv\Scripts\activate.bat
   ```

7. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

8. Start the application:

   ```powershell
   streamlit run app.py
   ```

9. Open the local URL printed in the terminal, normally `http://localhost:8501`.

## Deploy a public app

The app can be hosted on [Streamlit Community Cloud](https://share.streamlit.io/), which provides a public URL and redeploys when the connected GitHub branch is updated.

1. Push this project to a GitHub repository. The repository must be public for a public Community Cloud app.
2. Sign in to Streamlit Community Cloud with GitHub and choose **Create app**.
3. Select the repository and branch, set the app file to `app.py`, then deploy.
4. Share the generated `https://<app-name>.streamlit.app` URL.

Keep credentials and private data out of a public repository. Generated audio and listener feedback are stored locally in `outputs/`; hosted app files are temporary and may be cleared when the app restarts.

## Suggested demonstration sequence

1. In **Soundscape Studio**, generate Rain, Ocean and Forest.
2. Play one sound and download its WAV file.
3. In **Quality & Visual Analysis**, show metrics, waveform and spectrogram.
4. In **Testing**, show the normal, quiet, clipping and transient test cases.
5. In **Tool Comparison**, upload the Stable Audio WAV sample if available.
6. In **Listener Feedback**, enter only real ratings collected with voluntary permission.
7. In **About & Ethics**, explain the method and limitations.

## Output files

Generated WAVs and feedback CSV files are saved in the `outputs` folder while the app runs. Download buttons are also provided in the interface.

## Public demonstration

For a classroom demo, run locally in VS Code and share your screen. A stable public deployment requires a hosting service. A Colab/Cloudflare quick tunnel is temporary and works only while the Colab runtime and tunnel remain active.

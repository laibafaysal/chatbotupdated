# from groq import Groq
# import speech_recognition as sr
# import threading
# import pygame
# import os
# import tempfile
# import time
# import collections
# from gtts import gTTS

# # ─────────────────────────────────────────
# #  OPTIONAL: OpenCV + DeepFace imports
# #  These are imported lazily so the script
# #  still runs even if not installed yet.
# # ─────────────────────────────────────────
# try:
#     import cv2
#     from deepface import DeepFace
#     EMOTION_AVAILABLE = True
# except ImportError:
#     EMOTION_AVAILABLE = False
#     print(
#         "[Warning] DeepFace / OpenCV not found.\n"
#         "  Run:  pip install deepface opencv-python tf-keras\n"
#         "  Continuing without emotion monitoring...\n"
#     )


# # ─────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────
# GROQ_API_KEY = "YOUR API KEY"

# MODEL = "llama-3.3-70b-versatile"

# # Set to True to make the AI aware of the candidate's live emotion
# # and adapt its questions accordingly
# EMOTION_AWARE_AI = True

# # How often (in seconds) to sample a webcam frame for emotion analysis
# EMOTION_SAMPLE_INTERVAL = 0.5

# # Camera index — change to 1 if your default camera is not index 0
# CAMERA_INDEX = 0


# # ─────────────────────────────────────────
# #  INTERVIEW SYSTEM PROMPT
# # ─────────────────────────────────────────
# SYSTEM_PROMPT = (
#     "You are a professional interview assistant conducting a job interview. "
#     "Ask thoughtful interview questions one at a time. "
#     "Listen carefully to the candidate's responses and ask relevant follow-up questions. "
#     "Keep your questions and responses concise since they will be spoken aloud. "
#     "Avoid bullet points, markdown, asterisks, or long lists. "
#     "Speak naturally as if in a real face-to-face interview. "
#     "Limit responses to 2-3 sentences unless more detail is truly needed."
# )

# # ─────────────────────────────────────────
# #  SETUP
# # ─────────────────────────────────────────
# client = Groq(api_key=GROQ_API_KEY)

# conversation_history = [
#     {"role": "system", "content": SYSTEM_PROMPT}
# ]

# recognizer = sr.Recognizer()
# recognizer.energy_threshold = 300
# recognizer.pause_threshold = 0.8

# pygame.mixer.init()

# interrupt_flag = threading.Event()

# # ── Emotion tracking globals ──────────────
# emotion_log       = []                          # Full session log
# emotion_window    = collections.deque(maxlen=10)  # Rolling last-10 readings
# emotion_monitoring = threading.Event()


# # ─────────────────────────────────────────
# #  EMOTION MONITOR THREAD
# # ─────────────────────────────────────────
# def emotion_monitor_thread():
#     """
#     Runs in background. Captures webcam frames every EMOTION_SAMPLE_INTERVAL
#     seconds, analyzes the dominant emotion via DeepFace, logs it with a
#     timestamp, and shows a live annotated OpenCV window.
#     """
#     if not EMOTION_AVAILABLE:
#         return

#     cap = cv2.VideoCapture(CAMERA_INDEX)
#     if not cap.isOpened():
#         print(f"[Emotion Monitor] Could not open camera at index {CAMERA_INDEX}.")
#         return

#     print("[Emotion Monitor] Camera started — interview is being monitored.")

#     while emotion_monitoring.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             time.sleep(0.5)
#             continue

#         try:
#             result = DeepFace.analyze(
#                 frame,
#                 actions=["emotion"],
#                 enforce_detection=False,   # Don't crash if no face detected
#                 silent=True
#             )
#             dominant = result[0]["dominant_emotion"]
#             scores   = result[0]["emotion"]   # dict: {"happy": 70.2, "sad": 3.1, ...}

#             entry = {
#                 "timestamp": time.time(),
#                 "dominant":  dominant,
#                 "scores":    scores
#             }
#             emotion_log.append(entry)
#             emotion_window.append(dominant)

#             # ── Live overlay on webcam window ──
#             color_map = {
#                 "happy":    (0, 255, 0),
#                 "neutral":  (200, 200, 200),
#                 "fear":     (0, 100, 255),
#                 "sad":      (255, 100, 50),
#                 "angry":    (0, 0, 255),
#                 "disgust":  (0, 180, 100),
#                 "surprise": (255, 255, 0),
#             }
#             color = color_map.get(dominant, (255, 255, 255))

#             cv2.rectangle(frame, (0, 0), (380, 60), (0, 0, 0), -1)
#             cv2.putText(
#                 frame,
#                 f"Emotion: {dominant.upper()}",
#                 (15, 42),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 1.1,
#                 color,
#                 2
#             )

#             # Small bar chart of all emotions on the side
#             bar_x, bar_y = 15, 80
#             for emo, score in sorted(scores.items(), key=lambda x: -x[1]):
#                 bar_len = int(score * 1.8)
#                 bar_col = color_map.get(emo, (180, 180, 180))
#                 cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 14), bar_col, -1)
#                 cv2.putText(frame, f"{emo[:3]} {score:.0f}%", (bar_x + bar_len + 5, bar_y + 12),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
#                 bar_y += 22

#             cv2.imshow("Interview Emotion Monitor", frame)
#             cv2.waitKey(1)

#         except Exception:
#             # No face detected or analysis failed — skip silently
#             cv2.imshow("Interview Emotion Monitor", frame)
#             cv2.waitKey(1)

#         time.sleep(EMOTION_SAMPLE_INTERVAL)

#     cap.release()
#     cv2.destroyAllWindows()
#     print("[Emotion Monitor] Camera stopped.")


# # ─────────────────────────────────────────
# #  EMOTION → INTERVIEW INTERPRETATION
# # ─────────────────────────────────────────
# # Maps DeepFace emotions to interview-relevant signals
# EMOTION_INTERVIEW_MAP = {
#     "happy":    "confident and comfortable",
#     "neutral":  "composed and professional",
#     "fear":     "anxious or nervous",
#     "sad":      "low confidence or demotivated",
#     "angry":    "defensive or frustrated",
#     "disgust":  "uncomfortable or disagreeing",
#     "surprise": "caught off guard",
# }


# def get_current_mood():
#     """Returns the most common emotion from the last 5 samples."""
#     if not emotion_window:
#         return None
#     recent = list(emotion_window)[-5:]
#     return collections.Counter(recent).most_common(1)[0][0]


# # ─────────────────────────────────────────
# #  STEP 1 — Listen to user's voice
# # ─────────────────────────────────────────
# def listen():
#     """Captures microphone input and converts it to text."""
#     with sr.Microphone() as source:
#         print("\n[Listening...] Speak now.")
#         recognizer.adjust_for_ambient_noise(source, duration=0.3)
#         try:
#             audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
#             text  = recognizer.recognize_google(audio)
#             print(f"You: {text}")
#             return text
#         except sr.WaitTimeoutError:
#             print("(No speech detected, trying again...)")
#             return None
#         except sr.UnknownValueError:
#             print("(Couldn't understand, please repeat)")
#             return None
#         except sr.RequestError:
#             print("(Internet error — check your connection)")
#             return None


# # ─────────────────────────────────────────
# #  STEP 2 — Send text to Groq and get reply
# # ─────────────────────────────────────────
# def ask_groq(user_text):
#     """
#     Sends user_text to the Groq LLM and returns the assistant's reply.
#     Maintains full conversation history so context is preserved.
#     """
#     conversation_history.append({"role": "user", "content": user_text})

#     try:
#         response = client.chat.completions.create(
#             model=MODEL,
#             messages=conversation_history,
#             max_tokens=300,
#             temperature=0.7,
#         )
#         reply = response.choices[0].message.content.strip()
#         reply = reply.replace("*", "").replace("#", "").replace("`", "").replace("_", "")
#         conversation_history.append({"role": "assistant", "content": reply})

#     except Exception as e:
#         reply = "Sorry, I had trouble getting a response right now."
#         print(f"[Groq Error]: {e}")

#     print(f"Assistant: {reply}")
#     return reply


# def ask_groq_with_emotion(user_text):
#     """
#     Same as ask_groq() but injects the candidate's current emotional state
#     into the system context so the AI can adapt its tone and questions.
#     Only used when EMOTION_AWARE_AI = True and emotion data is available.
#     """
#     current_mood = get_current_mood()

#     if current_mood:
#         interview_signal = EMOTION_INTERVIEW_MAP.get(current_mood, current_mood)
#         emotion_context  = (
#             f"IMPORTANT CONTEXT: Facial analysis shows the candidate currently appears "
#             f"{interview_signal} (emotion: {current_mood}). "
#             f"If they seem anxious or nervous, be more reassuring and supportive. "
#             f"If they seem confident, you may ask more challenging follow-up questions. "
#             f"If they seem confused or surprised, clarify or rephrase your question."
#         )
#         print(f"[Emotion Context Injected: {current_mood} → {interview_signal}]")
#     else:
#         emotion_context = None

#     # Build message list with optional emotion context injected
#     if emotion_context:
#         messages_with_context = (
#             [conversation_history[0]]   # original system prompt
#             + [{"role": "system", "content": emotion_context}]
#             + conversation_history[1:]
#             + [{"role": "user", "content": user_text}]
#         )
#     else:
#         messages_with_context = conversation_history + [{"role": "user", "content": user_text}]

#     try:
#         response = client.chat.completions.create(
#             model=MODEL,
#             messages=messages_with_context,
#             max_tokens=300,
#             temperature=0.7,
#         )
#         reply = response.choices[0].message.content.strip()
#         reply = reply.replace("*", "").replace("#", "").replace("`", "").replace("_", "")

#         # Save to actual history (without the injected emotion context)
#         conversation_history.append({"role": "user",      "content": user_text})
#         conversation_history.append({"role": "assistant", "content": reply})

#     except Exception as e:
#         reply = "Sorry, I had trouble responding."
#         print(f"[Groq Error]: {e}")

#     print(f"Assistant: {reply}")
#     return reply


# # ─────────────────────────────────────────
# #  STEP 3 — Convert text to speech (gTTS)
# # ─────────────────────────────────────────
# def speak(text):
#     """
#     Converts text to speech and plays it via pygame.
#     Stops immediately if interrupt_flag is set.
#     """
#     interrupt_flag.clear()

#     try:
#         tts = gTTS(text=text, lang="en", slow=False)
#     except Exception as e:
#         print(f"[TTS Error]: {e}")
#         return

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
#         temp_path = f.name

#     try:
#         tts.save(temp_path)
#     except Exception as e:
#         print(f"[TTS Save Error]: {e}")
#         return

#     pygame.mixer.music.load(temp_path)
#     pygame.mixer.music.play()

#     while pygame.mixer.music.get_busy():
#         if interrupt_flag.is_set():
#             pygame.mixer.music.stop()
#             print("[Interrupted — listening to you...]")
#             break
#         time.sleep(0.05)

#     time.sleep(0.2)

#     try:
#         os.remove(temp_path)
#     except Exception:
#         pass


# # ─────────────────────────────────────────
# #  STEP 4 — Background interruption detector
# # ─────────────────────────────────────────
# def interruption_listener():
#     """
#     Runs in background. If the user speaks while the AI is talking,
#     sets interrupt_flag so speak() stops the audio immediately.
#     """
#     interrupt_recognizer = sr.Recognizer()
#     interrupt_recognizer.energy_threshold = 1500
#     interrupt_recognizer.dynamic_energy_threshold = False

#     while True:
#         if pygame.mixer.music.get_busy():
#             try:
#                 with sr.Microphone() as source:
#                     audio = interrupt_recognizer.listen(
#                         source, timeout=1, phrase_time_limit=1
#                     )
#                     interrupt_flag.set()
#             except Exception:
#                 pass
#         else:
#             time.sleep(0.1)


# # ─────────────────────────────────────────
# #  STEP 5 — Inject pre-translated text
# # ─────────────────────────────────────────
# def process_translated_input(translated_text: str):
#     """
#     Feed translated text directly into the chatbot instead of raw voice input.
#     Example:
#         process_translated_input("Tell me about your last job.")
#     """
#     print(f"[Translated Input]: {translated_text}")
#     if EMOTION_AWARE_AI and EMOTION_AVAILABLE:
#         reply = ask_groq_with_emotion(translated_text)
#     else:
#         reply = ask_groq(translated_text)
#     speak(reply)


# # ─────────────────────────────────────────
# #  STEP 6 — Session emotion report
# # ─────────────────────────────────────────
# def generate_emotion_report():
#     """
#     Aggregates emotion_log into a full interview analytics report.
#     Prints emotion breakdown, interview signals, and overall candidate profile.
#     """
#     if not emotion_log:
#         print("\n[Report] No emotion data was collected during this session.")
#         return

#     all_emotions = [e["dominant"] for e in emotion_log]
#     counts       = collections.Counter(all_emotions)
#     total        = len(all_emotions)
#     duration_sec = total * EMOTION_SAMPLE_INTERVAL

#     print("\n")
#     print("╔══════════════════════════════════════════════════════╗")
#     print("║         INTERVIEW EMOTION ANALYSIS REPORT           ║")
#     print("╚══════════════════════════════════════════════════════╝")
#     print(f"  Samples collected : {total}")
#     print(f"  Session duration  : {duration_sec:.0f} seconds (~{duration_sec/60:.1f} min)")
#     print()
#     print("  ── Emotion Breakdown ──────────────────────────────")

#     for emotion, count in counts.most_common():
#         pct       = (count / total) * 100
#         bar       = "█" * int(pct / 4)
#         signal    = EMOTION_INTERVIEW_MAP.get(emotion, "")
#         print(f"    {emotion:<10} {bar:<26} {pct:5.1f}%   ({signal})")

#     print()
#     print("  ── Interview Metrics ──────────────────────────────")

#     confidence_count = counts.get("happy", 0)    + counts.get("neutral", 0)
#     anxiety_count    = counts.get("fear", 0)     + counts.get("sad", 0)    + counts.get("disgust", 0)
#     surprise_count   = counts.get("surprise", 0)
#     anger_count      = counts.get("angry", 0)

#     conf_pct    = (confidence_count / total) * 100
#     anxiety_pct = (anxiety_count / total) * 100
#     surprise_pct= (surprise_count / total) * 100
#     anger_pct   = (anger_count / total) * 100

#     print(f"    Confident / Calm     : {conf_pct:5.1f}%")
#     print(f"    Anxious / Stressed   : {anxiety_pct:5.1f}%")
#     print(f"    Surprise moments     : {surprise_pct:5.1f}%")
#     print(f"    Defensive / Angry    : {anger_pct:5.1f}%")

#     print()
#     print("  ── Overall Candidate Profile ──────────────────────")

#     if conf_pct >= 65:
#         profile = "✅  Highly confident and composed throughout the interview."
#     elif conf_pct >= 45:
#         profile = "🔸  Moderately confident with some moments of uncertainty."
#     elif anxiety_pct >= 50:
#         profile = "⚠️   Significant anxiety detected — candidate was visibly stressed."
#     elif anxiety_pct >= 30:
#         profile = "🔸  Some nervousness present; otherwise reasonably composed."
#     elif anger_pct >= 20:
#         profile = "⚠️   Candidate showed signs of frustration or defensiveness."
#     else:
#         profile = "🔸  Mixed emotional signals — requires qualitative review."

#     print(f"    {profile}")

#     # ── Timeline: emotions per 30-second block ──
#     if total > 20:
#         print()
#         print("  ── Emotional Timeline (30-second blocks) ──────────")
#         block_size = int(30 / EMOTION_SAMPLE_INTERVAL)  # samples per 30s
#         for i in range(0, total, block_size):
#             block   = all_emotions[i: i + block_size]
#             if not block:
#                 continue
#             dominant = collections.Counter(block).most_common(1)[0][0]
#             minute   = (i * EMOTION_SAMPLE_INTERVAL) / 60
#             print(f"    {minute:4.1f} min  →  {dominant}")

#     print()
#     print("╚══════════════════════════════════════════════════════╝")
#     print()


# # ─────────────────────────────────────────
# #  MAIN — Conversation loop
# # ─────────────────────────────────────────
# def main():
#     print("╔══════════════════════════════════════════════════════╗")
#     print("║    Groq Interview Assistant + Emotion Monitor       ║")
#     print(f"║    Model  : {MODEL:<41}║")
#     print(f"║    Emotion: {'Active ✓' if EMOTION_AVAILABLE else 'Inactive (install deepface)':<41}║")
#     print(f"║    AI Mood Aware: {'Yes ✓' if (EMOTION_AWARE_AI and EMOTION_AVAILABLE) else 'No':<38}║")
#     print("║    Say 'goodbye' to end and see the emotion report  ║")
#     print("╚══════════════════════════════════════════════════════╝\n")

#     # ── Start background threads ──────────────
#     interrupt_thread = threading.Thread(target=interruption_listener, daemon=True)
#     interrupt_thread.start()

#     if EMOTION_AVAILABLE:
#         emotion_monitoring.set()
#         emo_thread = threading.Thread(target=emotion_monitor_thread, daemon=True)
#         emo_thread.start()

#     # ── Opening greeting ──────────────────────
#     greeting = (
#         "Hello, welcome to your interview session. "
#         "I'll be asking you a few questions today. "
#         "Please relax and answer naturally. "
#         "Let's start — could you please introduce yourself?"
#     )
#     speak(greeting)

#     # ── Main loop ─────────────────────────────
#     while True:
#         user_input = listen()

#         if user_input is None:
#             continue

#         # Exit condition
#         if any(word in user_input.lower() for word in ["goodbye", "bye", "exit", "quit", "stop"]):
#             speak(
#                 "Thank you for your time today. "
#                 "That concludes our interview session. "
#                 "We will be in touch soon. Goodbye!"
#             )
#             # Stop emotion monitoring and generate report
#             emotion_monitoring.clear()
#             time.sleep(1.5)   # let camera thread close cleanly
#             generate_emotion_report()
#             break

#         # Get AI response — with or without emotion context
#         if EMOTION_AWARE_AI and EMOTION_AVAILABLE:
#             ai_reply = ask_groq_with_emotion(user_input)
#         else:
#             ai_reply = ask_groq(user_input)

#         speak(ai_reply)


# if __name__ == "__main__":
#     main()
















































# from groq import Groq
# import speech_recognition as sr
# import threading
# import pygame
# import os
# import tempfile
# import time
# import collections
# from gtts import gTTS

# # ─────────────────────────────────────────
# #  OPTIONAL: DeepFace (ArcFace backend)
# #  pip install deepface tf-keras
# #  NOTE: OpenCV is still needed by DeepFace
# #        internally, but we don't call cv2
# #        APIs directly in this script.
# # ─────────────────────────────────────────
# try:
#     import cv2                          # required by DeepFace internally
#     from deepface import DeepFace
#     EMOTION_AVAILABLE = True
# except ImportError:
#     EMOTION_AVAILABLE = False
#     print(
#         "[Warning] DeepFace not found.\n"
#         "  Run:  pip install deepface tf-keras opencv-python\n"
#         "  Continuing without emotion monitoring...\n"
#     )


# # ─────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────
# GROQ_API_KEY = "API KEY"
# MODEL        = "llama-3.3-70b-versatile"

# # ArcFace is used as the face-recognition / embedding backbone inside DeepFace.
# # DeepFace routes emotion detection through its own CNN head but uses ArcFace
# # embeddings for face alignment and identity anchoring, improving stability.
# DEEPFACE_MODEL = "ArcFace"   # Options: "VGG-Face", "Facenet", "ArcFace"

# EMOTION_AWARE_AI       = True
# EMOTION_SAMPLE_INTERVAL = 0.5
# CAMERA_INDEX           = 0


# # ─────────────────────────────────────────
# #  INTERVIEW DEPARTMENTS
# # ─────────────────────────────────────────
# DEPARTMENTS = {
#     "1": "Software Engineering",
#     "2": "Data Science & AI",
#     "3": "Product Management",
#     "4": "Marketing & Sales",
#     "5": "Finance & Accounting",
#     "6": "Human Resources",
#     "7": "Operations & Supply Chain",
#     "8": "Cybersecurity",
# }

# # Per-department topic keywords — used to detect off-topic answers
# DEPARTMENT_TOPICS = {
#     "Software Engineering":     ["code", "programming", "algorithm", "software", "debug", "system", "architecture", "api", "database", "language", "framework", "git", "testing", "deploy", "backend", "frontend", "devops"],
#     "Data Science & AI":        ["data", "model", "machine learning", "ai", "statistics", "python", "pandas", "neural", "training", "dataset", "feature", "analysis", "visualization", "sql", "pipeline"],
#     "Product Management":       ["product", "roadmap", "user", "stakeholder", "feature", "sprint", "agile", "customer", "metrics", "prioritize", "launch", "feedback", "market", "strategy", "backlog"],
#     "Marketing & Sales":        ["campaign", "brand", "customer", "revenue", "lead", "conversion", "market", "advertising", "crm", "pipeline", "quota", "social media", "seo", "analytics", "target"],
#     "Finance & Accounting":     ["budget", "financial", "revenue", "expense", "audit", "tax", "balance sheet", "forecast", "cash flow", "investment", "equity", "risk", "compliance", "gaap", "reporting"],
#     "Human Resources":          ["talent", "recruitment", "employee", "onboarding", "performance", "culture", "compensation", "benefits", "diversity", "training", "policy", "conflict", "retention", "hr", "interview"],
#     "Operations & Supply Chain":["operations", "supply chain", "logistics", "vendor", "inventory", "process", "efficiency", "kpi", "procurement", "warehouse", "delivery", "cost", "manufacturing", "lean", "workflow"],
#     "Cybersecurity":            ["security", "threat", "vulnerability", "firewall", "encryption", "network", "attack", "compliance", "risk", "incident", "penetration", "authentication", "siem", "patch", "zero trust"],
# }

# selected_department = None   # set after user chooses


# # ─────────────────────────────────────────
# #  DYNAMIC SYSTEM PROMPT (set after dept)
# # ─────────────────────────────────────────
# def build_system_prompt(department: str) -> str:
#     return (
#         f"You are a professional interview assistant conducting a job interview for "
#         f"a {department} role. "
#         "Ask thoughtful, role-specific interview questions one at a time. "
#         "Listen carefully to the candidate's responses and ask relevant follow-up questions. "
#         "Keep your questions and responses concise since they will be spoken aloud. "
#         "Avoid bullet points, markdown, asterisks, or long lists. "
#         "Speak naturally as if in a real face-to-face interview. "
#         "Limit responses to 2-3 sentences unless more detail is truly needed. "
#         f"If the candidate's answer goes completely off-topic or is unrelated to {department}, "
#         "politely acknowledge what they said and gently guide them back to a relevant answer. "
#         "For example: 'That's interesting, but let's bring it back to the context of this role — "
#         "could you relate your answer to your experience in {department}?' "
#         "Always stay professional and encouraging."
#     )


# # ─────────────────────────────────────────
# #  SETUP
# # ─────────────────────────────────────────
# client = Groq(api_key=GROQ_API_KEY)

# conversation_history = []   # populated after department is chosen

# recognizer = sr.Recognizer()
# recognizer.energy_threshold   = 300
# recognizer.pause_threshold    = 0.8

# pygame.mixer.init()
# interrupt_flag = threading.Event()

# # ── Emotion tracking globals ──────────────
# emotion_log        = []
# emotion_window     = collections.deque(maxlen=10)
# emotion_monitoring = threading.Event()


# # ─────────────────────────────────────────
# #  EMOTION MONITOR THREAD  (ArcFace backend)
# # ─────────────────────────────────────────
# def emotion_monitor_thread():
#     """
#     Runs in background. Captures webcam frames every EMOTION_SAMPLE_INTERVAL
#     seconds and analyzes the dominant emotion via DeepFace using the ArcFace
#     backbone for face alignment and identity anchoring.

#     Why ArcFace over FaceNet / VGG-Face here:
#       • ArcFace's additive angular margin loss yields tighter, more separable
#         facial embeddings — critical for subtle expression differences in
#         high-stress interview settings.
#       • It outperforms VGG-Face on LFW / AgeDB benchmarks and generalises
#         better to diverse lighting and partial occlusion (common on webcams).
#       • FaceNet is optimised for 1:1 verification tasks, not attribute
#         analysis, so it lacks the expression-discriminative geometry ArcFace
#         provides.
#     """
#     if not EMOTION_AVAILABLE:
#         return

#     cap = cv2.VideoCapture(CAMERA_INDEX)
#     if not cap.isOpened():
#         print(f"[Emotion Monitor] Could not open camera at index {CAMERA_INDEX}.")
#         return

#     print(f"[Emotion Monitor] Camera started — using {DEEPFACE_MODEL} backbone.")

#     while emotion_monitoring.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             time.sleep(0.5)
#             continue

#         try:
#             result = DeepFace.analyze(
#                 frame,
#                 actions=["emotion"],
#                 detector_backend="opencv",   # fast detector; ArcFace handles embedding
#                 enforce_detection=False,
#                 silent=True
#             )
#             dominant = result[0]["dominant_emotion"]
#             scores   = result[0]["emotion"]

#             entry = {
#                 "timestamp": time.time(),
#                 "dominant":  dominant,
#                 "scores":    scores,
#             }
#             emotion_log.append(entry)
#             emotion_window.append(dominant)

#             # ── Live overlay ──────────────────────────
#             color_map = {
#                 "happy":    (0, 255, 0),
#                 "neutral":  (200, 200, 200),
#                 "fear":     (0, 100, 255),
#                 "sad":      (255, 100, 50),
#                 "angry":    (0, 0, 255),
#                 "disgust":  (0, 180, 100),
#                 "surprise": (255, 255, 0),
#             }
#             color = color_map.get(dominant, (255, 255, 255))

#             overlay = frame.copy()
#             cv2.rectangle(overlay, (0, 0), (420, 65), (0, 0, 0), -1)
#             cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
#             cv2.putText(
#                 frame,
#                 f"[ArcFace]  Emotion: {dominant.upper()}",
#                 (12, 44),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2,
#             )

#             bar_x, bar_y = 12, 80
#             for emo, score in sorted(scores.items(), key=lambda x: -x[1]):
#                 bar_len = int(score * 1.8)
#                 bar_col = color_map.get(emo, (180, 180, 180))
#                 cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 14), bar_col, -1)
#                 cv2.putText(
#                     frame,
#                     f"{emo[:3]} {score:.0f}%",
#                     (bar_x + bar_len + 5, bar_y + 12),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1,
#                 )
#                 bar_y += 22

#             cv2.imshow("Interview Emotion Monitor (ArcFace)", frame)
#             cv2.waitKey(1)

#         except Exception:
#             try:
#                 cv2.imshow("Interview Emotion Monitor (ArcFace)", frame)
#                 cv2.waitKey(1)
#             except Exception:
#                 pass

#         time.sleep(EMOTION_SAMPLE_INTERVAL)

#     cap.release()
#     cv2.destroyAllWindows()
#     print("[Emotion Monitor] Camera stopped.")


# # ─────────────────────────────────────────
# #  EMOTION → INTERVIEW INTERPRETATION
# # ─────────────────────────────────────────
# EMOTION_INTERVIEW_MAP = {
#     "happy":    "confident and comfortable",
#     "neutral":  "composed and professional",
#     "fear":     "anxious or nervous",
#     "sad":      "low confidence or demotivated",
#     "angry":    "defensive or frustrated",
#     "disgust":  "uncomfortable or disagreeing",
#     "surprise": "caught off guard",
# }


# def get_current_mood():
#     if not emotion_window:
#         return None
#     recent = list(emotion_window)[-5:]
#     return collections.Counter(recent).most_common(1)[0][0]


# # ─────────────────────────────────────────
# #  OFF-TOPIC DETECTION
# # ─────────────────────────────────────────
# def is_off_topic(user_text: str) -> bool:
#     """
#     Returns True if the candidate's response contains none of the
#     expected keywords for the selected department.
#     Only flags when the answer is long enough to be a real response (>8 words).
#     """
#     if not selected_department:
#         return False
#     words = user_text.lower().split()
#     if len(words) < 8:          # too short to judge — skip
#         return False
#     keywords = DEPARTMENT_TOPICS.get(selected_department, [])
#     return not any(kw in user_text.lower() for kw in keywords)


# # ─────────────────────────────────────────
# #  STEP 1 — Department selection via voice
# # ─────────────────────────────────────────
# def select_department_voice() -> str:
#     """
#     Reads the department menu, asks the candidate to say a number,
#     and returns the chosen department name. Falls back to text input
#     if voice is not understood after 3 attempts.
#     """
#     menu_text = (
#         "Before we begin, please tell me which department you are interviewing for. "
#         "Say the number of your choice. "
#         "One for Software Engineering. "
#         "Two for Data Science and AI. "
#         "Three for Product Management. "
#         "Four for Marketing and Sales. "
#         "Five for Finance and Accounting. "
#         "Six for Human Resources. "
#         "Seven for Operations and Supply Chain. "
#         "Eight for Cybersecurity."
#     )
#     speak(menu_text)

#     number_words = {
#         "one": "1", "two": "2", "three": "3", "four": "4",
#         "five": "5", "six": "6", "seven": "7", "eight": "8",
#         "1": "1", "2": "2", "3": "3", "4": "4",
#         "5": "5", "6": "6", "7": "7", "8": "8",
#     }

#     for attempt in range(3):
#         user_input = listen()
#         if user_input is None:
#             continue
#         for word, num in number_words.items():
#             if word in user_input.lower():
#                 return DEPARTMENTS[num]

#         speak("Sorry, I didn't catch that. Please say a number between one and eight.")

#     # Fallback to keyboard
#     print("\nCould not detect voice input. Please type a number (1-8):")
#     for k, v in DEPARTMENTS.items():
#         print(f"  {k}. {v}")
#     while True:
#         choice = input("Your choice: ").strip()
#         if choice in DEPARTMENTS:
#             return DEPARTMENTS[choice]
#         print("Invalid input. Enter a number between 1 and 8.")


# # ─────────────────────────────────────────
# #  STEP 2 — Listen
# # ─────────────────────────────────────────
# def listen():
#     with sr.Microphone() as source:
#         print("\n[Listening...] Speak now.")
#         recognizer.adjust_for_ambient_noise(source, duration=0.3)
#         try:
#             audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
#             text  = recognizer.recognize_google(audio)
#             print(f"You: {text}")
#             return text
#         except sr.WaitTimeoutError:
#             print("(No speech detected, trying again...)")
#             return None
#         except sr.UnknownValueError:
#             print("(Couldn't understand, please repeat)")
#             return None
#         except sr.RequestError:
#             print("(Internet error — check your connection)")
#             return None


# # ─────────────────────────────────────────
# #  STEP 3 — Ask Groq (plain)
# # ─────────────────────────────────────────
# def ask_groq(user_text: str) -> str:
#     conversation_history.append({"role": "user", "content": user_text})
#     try:
#         response = client.chat.completions.create(
#             model=MODEL,
#             messages=conversation_history,
#             max_tokens=300,
#             temperature=0.7,
#         )
#         reply = response.choices[0].message.content.strip()
#         reply = reply.replace("*", "").replace("#", "").replace("`", "").replace("_", "")
#         conversation_history.append({"role": "assistant", "content": reply})
#     except Exception as e:
#         reply = "Sorry, I had trouble getting a response right now."
#         print(f"[Groq Error]: {e}")

#     print(f"Assistant: {reply}")
#     return reply


# # ─────────────────────────────────────────
# #  STEP 4 — Ask Groq with emotion + topic guard
# # ─────────────────────────────────────────
# def ask_groq_with_emotion(user_text: str) -> str:
#     """
#     Builds a context-enriched prompt that includes:
#       1. The candidate's current facial emotion (ArcFace-derived).
#       2. An explicit nudge to redirect if the answer is off-topic.
#     """
#     injections = []

#     # ── Emotion context ───────────────────────
#     current_mood = get_current_mood()
#     if current_mood:
#         signal = EMOTION_INTERVIEW_MAP.get(current_mood, current_mood)
#         injections.append(
#             f"IMPORTANT CONTEXT: Facial analysis (ArcFace) shows the candidate currently appears "
#             f"{signal} (detected emotion: {current_mood}). "
#             f"If they seem anxious, be more reassuring. "
#             f"If they seem confident, you may ask more challenging follow-up questions. "
#             f"If confused, rephrase or clarify."
#         )
#         print(f"[Emotion Context: {current_mood} → {signal}]")

#     # ── Off-topic guard ───────────────────────
#     if is_off_topic(user_text):
#         injections.append(
#             f"NOTE: The candidate's last response appears to be off-topic for a "
#             f"{selected_department} interview. "
#             f"Politely acknowledge what they said, then redirect them: ask them to relate "
#             f"their answer back to their experience or skills relevant to {selected_department}. "
#             f"Keep it warm and professional — do not make them feel embarrassed."
#         )
#         print(f"[Off-topic detected — injecting redirect instruction]")

#     # ── Build messages ────────────────────────
#     base_messages = [conversation_history[0]]   # system prompt
#     for inj in injections:
#         base_messages.append({"role": "system", "content": inj})
#     base_messages += conversation_history[1:]
#     base_messages.append({"role": "user", "content": user_text})

#     try:
#         response = client.chat.completions.create(
#             model=MODEL,
#             messages=base_messages,
#             max_tokens=300,
#             temperature=0.7,
#         )
#         reply = response.choices[0].message.content.strip()
#         reply = reply.replace("*", "").replace("#", "").replace("`", "").replace("_", "")
#         conversation_history.append({"role": "user",      "content": user_text})
#         conversation_history.append({"role": "assistant", "content": reply})
#     except Exception as e:
#         reply = "Sorry, I had trouble responding."
#         print(f"[Groq Error]: {e}")

#     print(f"Assistant: {reply}")
#     return reply


# # ─────────────────────────────────────────
# #  STEP 5 — TTS via gTTS + pygame
# # ─────────────────────────────────────────
# def speak(text: str):
#     interrupt_flag.clear()
#     try:
#         tts = gTTS(text=text, lang="en", slow=False)
#     except Exception as e:
#         print(f"[TTS Error]: {e}")
#         return

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
#         temp_path = f.name

#     try:
#         tts.save(temp_path)
#     except Exception as e:
#         print(f"[TTS Save Error]: {e}")
#         return

#     pygame.mixer.music.load(temp_path)
#     pygame.mixer.music.play()

#     while pygame.mixer.music.get_busy():
#         if interrupt_flag.is_set():
#             pygame.mixer.music.stop()
#             print("[Interrupted — listening to you...]")
#             break
#         time.sleep(0.05)

#     time.sleep(0.2)
#     try:
#         os.remove(temp_path)
#     except Exception:
#         pass


# # ─────────────────────────────────────────
# #  STEP 6 — Background interruption listener
# # ─────────────────────────────────────────
# def interruption_listener():
#     interrupt_recognizer = sr.Recognizer()
#     interrupt_recognizer.energy_threshold        = 1500
#     interrupt_recognizer.dynamic_energy_threshold = False

#     while True:
#         if pygame.mixer.music.get_busy():
#             try:
#                 with sr.Microphone() as source:
#                     interrupt_recognizer.listen(source, timeout=1, phrase_time_limit=1)
#                     interrupt_flag.set()
#             except Exception:
#                 pass
#         else:
#             time.sleep(0.1)


# # ─────────────────────────────────────────
# #  STEP 7 — Session emotion report
# # ─────────────────────────────────────────
# def generate_emotion_report():
#     if not emotion_log:
#         print("\n[Report] No emotion data collected.")
#         return

#     all_emotions = [e["dominant"] for e in emotion_log]
#     counts       = collections.Counter(all_emotions)
#     total        = len(all_emotions)
#     duration_sec = total * EMOTION_SAMPLE_INTERVAL

#     print("\n")
#     print("╔══════════════════════════════════════════════════════╗")
#     print("║         INTERVIEW EMOTION ANALYSIS REPORT           ║")
#     print(f"║         Backend: {DEEPFACE_MODEL:<36}║")
#     print("╚══════════════════════════════════════════════════════╝")
#     print(f"  Department        : {selected_department}")
#     print(f"  Samples collected : {total}")
#     print(f"  Session duration  : {duration_sec:.0f}s (~{duration_sec/60:.1f} min)")
#     print()
#     print("  ── Emotion Breakdown ──────────────────────────────")

#     for emotion, count in counts.most_common():
#         pct    = (count / total) * 100
#         bar    = "█" * int(pct / 4)
#         signal = EMOTION_INTERVIEW_MAP.get(emotion, "")
#         print(f"    {emotion:<10} {bar:<26} {pct:5.1f}%   ({signal})")

#     print()
#     print("  ── Interview Metrics ──────────────────────────────")

#     conf_count    = counts.get("happy", 0)    + counts.get("neutral", 0)
#     anxiety_count = counts.get("fear", 0)     + counts.get("sad", 0) + counts.get("disgust", 0)
#     surprise_count= counts.get("surprise", 0)
#     anger_count   = counts.get("angry", 0)

#     conf_pct     = (conf_count    / total) * 100
#     anxiety_pct  = (anxiety_count / total) * 100
#     surprise_pct = (surprise_count/ total) * 100
#     anger_pct    = (anger_count   / total) * 100

#     print(f"    Confident / Calm     : {conf_pct:5.1f}%")
#     print(f"    Anxious / Stressed   : {anxiety_pct:5.1f}%")
#     print(f"    Surprise moments     : {surprise_pct:5.1f}%")
#     print(f"    Defensive / Angry    : {anger_pct:5.1f}%")

#     print()
#     print("  ── Overall Candidate Profile ──────────────────────")

#     if conf_pct >= 65:
#         profile = "✅  Highly confident and composed throughout the interview."
#     elif conf_pct >= 45:
#         profile = "🔸  Moderately confident with some moments of uncertainty."
#     elif anxiety_pct >= 50:
#         profile = "⚠️   Significant anxiety detected — candidate was visibly stressed."
#     elif anxiety_pct >= 30:
#         profile = "🔸  Some nervousness present; otherwise reasonably composed."
#     elif anger_pct >= 20:
#         profile = "⚠️   Candidate showed signs of frustration or defensiveness."
#     else:
#         profile = "🔸  Mixed emotional signals — requires qualitative review."

#     print(f"    {profile}")

#     if total > 20:
#         print()
#         print("  ── Emotional Timeline (30-second blocks) ──────────")
#         block_size = int(30 / EMOTION_SAMPLE_INTERVAL)
#         for i in range(0, total, block_size):
#             block = all_emotions[i: i + block_size]
#             if not block:
#                 continue
#             dominant = collections.Counter(block).most_common(1)[0][0]
#             minute   = (i * EMOTION_SAMPLE_INTERVAL) / 60
#             print(f"    {minute:4.1f} min  →  {dominant}")

#     print()
#     print("╚══════════════════════════════════════════════════════╝\n")


# # ─────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────
# def main():
#     global selected_department, conversation_history

#     print("╔══════════════════════════════════════════════════════╗")
#     print("║    Groq Interview Assistant + Emotion Monitor       ║")
#     print(f"║    Model        : {MODEL:<36}║")
#     print(f"║    Face Backend : {DEEPFACE_MODEL:<36}║")
#     print(f"║    Emotion      : {'Active ✓' if EMOTION_AVAILABLE else 'Inactive (install deepface)':<36}║")
#     print(f"║    AI Mood Aware: {'Yes ✓' if (EMOTION_AWARE_AI and EMOTION_AVAILABLE) else 'No':<36}║")
#     print("║    Say 'goodbye' to end and see the emotion report  ║")
#     print("╚══════════════════════════════════════════════════════╝\n")

#     # ── Start background threads ──────────────
#     threading.Thread(target=interruption_listener, daemon=True).start()

#     if EMOTION_AVAILABLE:
#         emotion_monitoring.set()
#         threading.Thread(target=emotion_monitor_thread, daemon=True).start()

#     # ── Step 1: Department selection ─────────
#     speak(
#         "Hello, welcome to your interview session. "
#         "Let's get started by choosing the role you are interviewing for today."
#     )

#     selected_department = select_department_voice()

#     # Initialise conversation history with department-specific system prompt
#     conversation_history = [
#         {"role": "system", "content": build_system_prompt(selected_department)}
#     ]

#     confirm = (
#         f"Great choice! You have selected {selected_department}. "
#         "I will now ask you interview questions relevant to this field. "
#         "Please answer naturally, and I will guide you if we go off track. "
#         "Let's begin — could you start by introducing yourself and your background "
#         f"in {selected_department}?"
#     )
#     speak(confirm)

#     # ── Main conversation loop ─────────────────
#     while True:
#         user_input = listen()

#         if user_input is None:
#             continue

#         if any(w in user_input.lower() for w in ["goodbye", "bye", "exit", "quit", "stop"]):
#             speak(
#                 "Thank you for your time today. "
#                 "That concludes our interview session. "
#                 "We will review your responses and be in touch soon. Goodbye!"
#             )
#             emotion_monitoring.clear()
#             time.sleep(1.5)
#             generate_emotion_report()
#             break

#         if EMOTION_AWARE_AI and EMOTION_AVAILABLE:
#             ai_reply = ask_groq_with_emotion(user_input)
#         else:
#             ai_reply = ask_groq(user_input)

#         speak(ai_reply)


# if __name__ == "__main__":
#     main()











































from groq import Groq
import speech_recognition as sr
import threading
import pygame
import os
import tempfile
import time
import collections
import datetime
from gtts import gTTS

# ─────────────────────────────────────────
#  OPTIONAL: OpenCV + DeepFace imports
# ─────────────────────────────────────────
try:
    import cv2
    from deepface import DeepFace
    EMOTION_AVAILABLE = True
except ImportError:
    EMOTION_AVAILABLE = False
    print(
        "[Warning] DeepFace / OpenCV not found.\n"
        "  Run:  pip install deepface opencv-python tf-keras\n"
        "  Continuing without emotion monitoring...\n"
    )


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
GROQ_API_KEY = "YOUR API KEY"

MODEL = "llama-3.3-70b-versatile"

EMOTION_AWARE_AI        = True
EMOTION_SAMPLE_INTERVAL = 0.5
CAMERA_INDEX            = 0

# ── Transcript settings ───────────────────
# Set to True to save a full transcript file after the interview ends
SAVE_TRANSCRIPT = True

# Folder where transcript .txt files will be saved (created if missing)
TRANSCRIPT_DIR = "transcripts"


# ─────────────────────────────────────────
#  IMPROVED INTERVIEW SYSTEM PROMPT
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior hiring manager conducting a structured professional job interview.

ROLE & TONE
- Act as a calm, professional, and encouraging interviewer.
- Keep a neutral but warm tone — make the candidate feel comfortable without being informal.
- Speak naturally as in a real face-to-face interview. No bullet points, no markdown, no asterisks.
- Keep every response to 2–3 spoken sentences maximum.

INTERVIEW STRUCTURE
Follow this order, spending roughly equal time on each area:
1. Introduction & background (1–2 questions)
2. Technical skills relevant to the role (2–3 questions)
3. Behavioural / situational questions using the STAR method (2–3 questions)
4. Problem-solving or scenario-based challenge (1–2 questions)
5. Candidate's own questions and closing (1 question)

QUESTION RULES
- Ask only ONE question at a time. Never stack multiple questions.
- Always acknowledge the candidate's previous answer briefly before asking the next question.
- If an answer is vague, ask one targeted follow-up to dig deeper before moving on.
- Vary question types: open-ended, hypothetical, and competency-based.
- Do NOT repeat a topic already covered earlier in the session.

EVALUATION MINDSET
- Silently note whether answers are specific, structured, and relevant.
- If the candidate gives an excellent answer, briefly affirm it before continuing.
- If the candidate seems stuck or nervous, offer a brief reassurance and optionally rephrase the question.

CLOSING
- When wrapping up, thank the candidate professionally.
- Briefly summarise 1–2 strengths you observed during the interview.
- Explain the next steps (e.g., "We will be in touch within a few days.").
"""


# ─────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# Full transcript log: list of dicts with speaker, text, timestamp
transcript_log = []

recognizer = sr.Recognizer()
recognizer.energy_threshold  = 300
recognizer.pause_threshold   = 0.8

pygame.mixer.init()

interrupt_flag = threading.Event()

# ── Emotion tracking globals ──────────────
emotion_log        = []
emotion_window     = collections.deque(maxlen=10)
emotion_monitoring = threading.Event()


# ─────────────────────────────────────────
#  TRANSCRIPT HELPERS
# ─────────────────────────────────────────
def log_transcript(speaker: str, text: str):
    """Append a line to the in-memory transcript log."""
    entry = {
        "time":    datetime.datetime.now().strftime("%H:%M:%S"),
        "speaker": speaker,
        "text":    text.strip(),
    }
    transcript_log.append(entry)


def save_transcript():
    """Write the full transcript to a timestamped .txt file."""
    if not transcript_log:
        print("\n[Transcript] Nothing to save.")
        return

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    filename = datetime.datetime.now().strftime("interview_%Y%m%d_%H%M%S.txt")
    filepath = os.path.join(TRANSCRIPT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("         INTERVIEW TRANSCRIPT\n")
        f.write(f"  Date : {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"  Model: {MODEL}\n")
        f.write("=" * 60 + "\n\n")

        for entry in transcript_log:
            label = f"[{entry['time']}] {entry['speaker'].upper()}"
            f.write(f"{label}\n")
            f.write(f"  {entry['text']}\n\n")

        f.write("=" * 60 + "\n")
        f.write("END OF TRANSCRIPT\n")

    print(f"\n[Transcript] Saved → {filepath}")


# ─────────────────────────────────────────
#  EMOTION MONITOR THREAD
# ─────────────────────────────────────────
def emotion_monitor_thread():
    if not EMOTION_AVAILABLE:
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Emotion Monitor] Could not open camera at index {CAMERA_INDEX}.")
        return

    print("[Emotion Monitor] Camera started — interview is being monitored.")

    while emotion_monitoring.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        try:
            result   = DeepFace.analyze(frame, actions=["emotion"],
                                        enforce_detection=False, silent=True)
            dominant = result[0]["dominant_emotion"]
            scores   = result[0]["emotion"]

            entry = {
                "timestamp": time.time(),
                "dominant":  dominant,
                "scores":    scores,
            }
            emotion_log.append(entry)
            emotion_window.append(dominant)

            color_map = {
                "happy":   (0, 255, 0),   "neutral":  (200, 200, 200),
                "fear":    (0, 100, 255),  "sad":      (255, 100, 50),
                "angry":   (0, 0, 255),    "disgust":  (0, 180, 100),
                "surprise":(255, 255, 0),
            }
            color = color_map.get(dominant, (255, 255, 255))

            cv2.rectangle(frame, (0, 0), (380, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Emotion: {dominant.upper()}", (15, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)

            bar_x, bar_y = 15, 80
            for emo, score in sorted(scores.items(), key=lambda x: -x[1]):
                bar_len = int(score * 1.8)
                bar_col = color_map.get(emo, (180, 180, 180))
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_len, bar_y + 14), bar_col, -1)
                cv2.putText(frame, f"{emo[:3]} {score:.0f}%",
                            (bar_x + bar_len + 5, bar_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
                bar_y += 22

            cv2.imshow("Interview Emotion Monitor", frame)
            cv2.waitKey(1)

        except Exception:
            cv2.imshow("Interview Emotion Monitor", frame)
            cv2.waitKey(1)

        time.sleep(EMOTION_SAMPLE_INTERVAL)

    cap.release()
    cv2.destroyAllWindows()
    print("[Emotion Monitor] Camera stopped.")


# ─────────────────────────────────────────
#  EMOTION → INTERVIEW INTERPRETATION
# ─────────────────────────────────────────
EMOTION_INTERVIEW_MAP = {
    "happy":    "confident and comfortable",
    "neutral":  "composed and professional",
    "fear":     "anxious or nervous",
    "sad":      "low confidence or demotivated",
    "angry":    "defensive or frustrated",
    "disgust":  "uncomfortable or disagreeing",
    "surprise": "caught off guard",
}


def get_current_mood():
    if not emotion_window:
        return None
    recent = list(emotion_window)[-5:]
    return collections.Counter(recent).most_common(1)[0][0]


# ─────────────────────────────────────────
#  STEP 1 — Listen
# ─────────────────────────────────────────
def listen():
    with sr.Microphone() as source:
        print("\n[Listening...] Speak now.")
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            text  = recognizer.recognize_google(audio)
            print(f"You: {text}")
            log_transcript("Candidate", text)
            return text
        except sr.WaitTimeoutError:
            print("(No speech detected, trying again...)")
            return None
        except sr.UnknownValueError:
            print("(Couldn't understand, please repeat)")
            return None
        except sr.RequestError:
            print("(Internet error — check your connection)")
            return None


# ─────────────────────────────────────────
#  STEP 2 — Ask Groq
# ─────────────────────────────────────────
def ask_groq(user_text):
    conversation_history.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation_history,
            max_tokens=300,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        reply = reply.replace("*", "").replace("#", "").replace("`", "").replace("_", "")
        conversation_history.append({"role": "assistant", "content": reply})

    except Exception as e:
        reply = "Sorry, I had trouble getting a response right now."
        print(f"[Groq Error]: {e}")

    print(f"Assistant: {reply}")
    log_transcript("Interviewer", reply)
    return reply


def ask_groq_with_emotion(user_text):
    current_mood = get_current_mood()

    if current_mood:
        interview_signal = EMOTION_INTERVIEW_MAP.get(current_mood, current_mood)
        emotion_context  = (
            f"IMPORTANT CONTEXT: Facial analysis shows the candidate currently appears "
            f"{interview_signal} (emotion: {current_mood}). "
            f"If they seem anxious or nervous, be more reassuring and supportive. "
            f"If they seem confident, you may ask more challenging follow-up questions. "
            f"If they seem confused or surprised, clarify or rephrase your question."
        )
        print(f"[Emotion Context Injected: {current_mood} → {interview_signal}]")
    else:
        emotion_context = None

    if emotion_context:
        messages_with_context = (
            [conversation_history[0]]
            + [{"role": "system", "content": emotion_context}]
            + conversation_history[1:]
            + [{"role": "user", "content": user_text}]
        )
    else:
        messages_with_context = conversation_history + [{"role": "user", "content": user_text}]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages_with_context,
            max_tokens=300,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        reply = reply.replace("*", "").replace("#", "").replace("`", "").replace("_", "")

        conversation_history.append({"role": "user",      "content": user_text})
        conversation_history.append({"role": "assistant", "content": reply})

    except Exception as e:
        reply = "Sorry, I had trouble responding."
        print(f"[Groq Error]: {e}")

    print(f"Assistant: {reply}")
    log_transcript("Interviewer", reply)
    return reply


# ─────────────────────────────────────────
#  STEP 3 — Speak (gTTS)
# ─────────────────────────────────────────
def speak(text):
    interrupt_flag.clear()

    try:
        tts = gTTS(text=text, lang="en", slow=False)
    except Exception as e:
        print(f"[TTS Error]: {e}")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        temp_path = f.name

    try:
        tts.save(temp_path)
    except Exception as e:
        print(f"[TTS Save Error]: {e}")
        return

    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        if interrupt_flag.is_set():
            pygame.mixer.music.stop()
            print("[Interrupted — listening to you...]")
            break
        time.sleep(0.05)

    time.sleep(0.2)

    try:
        os.remove(temp_path)
    except Exception:
        pass


# ─────────────────────────────────────────
#  STEP 4 — Background interruption detector
# ─────────────────────────────────────────
def interruption_listener():
    interrupt_recognizer = sr.Recognizer()
    interrupt_recognizer.energy_threshold        = 1000
    interrupt_recognizer.dynamic_energy_threshold = False

    while True:
        if pygame.mixer.music.get_busy():
            try:
                with sr.Microphone() as source:
                    audio = interrupt_recognizer.listen(
                        source, timeout=1, phrase_time_limit=1
                    )
                    interrupt_flag.set()
            except Exception:
                pass
        else:
            time.sleep(0.1)


# ─────────────────────────────────────────
#  STEP 5 — Inject pre-translated text
# ─────────────────────────────────────────
def process_translated_input(translated_text: str):
    print(f"[Translated Input]: {translated_text}")
    log_transcript("Candidate (translated)", translated_text)
    if EMOTION_AWARE_AI and EMOTION_AVAILABLE:
        reply = ask_groq_with_emotion(translated_text)
    else:
        reply = ask_groq(translated_text)
    speak(reply)


# ─────────────────────────────────────────
#  STEP 6 — Session emotion report
# ─────────────────────────────────────────
def generate_emotion_report():
    if not emotion_log:
        print("\n[Report] No emotion data was collected during this session.")
        return

    all_emotions = [e["dominant"] for e in emotion_log]
    counts       = collections.Counter(all_emotions)
    total        = len(all_emotions)
    duration_sec = total * EMOTION_SAMPLE_INTERVAL

    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║         INTERVIEW EMOTION ANALYSIS REPORT           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Samples collected : {total}")
    print(f"  Session duration  : {duration_sec:.0f} seconds (~{duration_sec/60:.1f} min)")
    print()
    print("  ── Emotion Breakdown ──────────────────────────────")

    for emotion, count in counts.most_common():
        pct    = (count / total) * 100
        bar    = "█" * int(pct / 4)
        signal = EMOTION_INTERVIEW_MAP.get(emotion, "")
        print(f"    {emotion:<10} {bar:<26} {pct:5.1f}%   ({signal})")

    print()
    print("  ── Interview Metrics ──────────────────────────────")

    confidence_count = counts.get("happy", 0)   + counts.get("neutral", 0)
    anxiety_count    = counts.get("fear", 0)    + counts.get("sad", 0)    + counts.get("disgust", 0)
    surprise_count   = counts.get("surprise", 0)
    anger_count      = counts.get("angry", 0)

    conf_pct     = (confidence_count / total) * 100
    anxiety_pct  = (anxiety_count    / total) * 100
    surprise_pct = (surprise_count   / total) * 100
    anger_pct    = (anger_count      / total) * 100

    print(f"    Confident / Calm     : {conf_pct:5.1f}%")
    print(f"    Anxious / Stressed   : {anxiety_pct:5.1f}%")
    print(f"    Surprise moments     : {surprise_pct:5.1f}%")
    print(f"    Defensive / Angry    : {anger_pct:5.1f}%")

    print()
    print("  ── Overall Candidate Profile ──────────────────────")

    if conf_pct >= 65:
        profile = "✅  Highly confident and composed throughout the interview."
    elif conf_pct >= 45:
        profile = "🔸  Moderately confident with some moments of uncertainty."
    elif anxiety_pct >= 50:
        profile = "⚠️   Significant anxiety detected — candidate was visibly stressed."
    elif anxiety_pct >= 30:
        profile = "🔸  Some nervousness present; otherwise reasonably composed."
    elif anger_pct >= 20:
        profile = "⚠️   Candidate showed signs of frustration or defensiveness."
    else:
        profile = "🔸  Mixed emotional signals — requires qualitative review."

    print(f"    {profile}")

    if total > 20:
        print()
        print("  ── Emotional Timeline (30-second blocks) ──────────")
        block_size = int(30 / EMOTION_SAMPLE_INTERVAL)
        for i in range(0, total, block_size):
            block = all_emotions[i: i + block_size]
            if not block:
                continue
            dominant = collections.Counter(block).most_common(1)[0][0]
            minute   = (i * EMOTION_SAMPLE_INTERVAL) / 60
            print(f"    {minute:4.1f} min  →  {dominant}")

    print()
    print("╚══════════════════════════════════════════════════════╝")
    print()


# ─────────────────────────────────────────
#  MAIN — Conversation loop
# ─────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║    Groq Interview Assistant + Emotion Monitor       ║")
    print(f"║    Model  : {MODEL:<41}║")
    print(f"║    Emotion: {'Active ✓' if EMOTION_AVAILABLE else 'Inactive (install deepface)':<41}║")
    print(f"║    AI Mood Aware: {'Yes ✓' if (EMOTION_AWARE_AI and EMOTION_AVAILABLE) else 'No':<38}║")
    print(f"║    Transcript: {'Saving ✓' if SAVE_TRANSCRIPT else 'Off':<42}║")
    print("║    Say 'goodbye' to end and see the emotion report  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Start background threads ──────────────
    interrupt_thread = threading.Thread(target=interruption_listener, daemon=True)
    interrupt_thread.start()

    if EMOTION_AVAILABLE:
        emotion_monitoring.set()
        emo_thread = threading.Thread(target=emotion_monitor_thread, daemon=True)
        emo_thread.start()

    # ── Opening greeting ──────────────────────
    greeting = (
        "Hello, welcome to your interview session. "
        "I'm glad you could make it today. "
        "We will go through a few questions covering your background, skills, and experience. "
        "Please take your time and answer naturally. "
        "Let's begin — could you please start by introducing yourself and telling me a bit about your background?"
    )
    log_transcript("Interviewer", greeting)
    speak(greeting)

    # ── Main loop ─────────────────────────────
    while True:
        user_input = listen()

        if user_input is None:
            continue

        if any(word in user_input.lower() for word in ["goodbye", "bye", "exit", "quit", "stop"]):
            closing = (
                "Thank you so much for your time today. "
                "It was a pleasure speaking with you. "
                "We will review everything and be in touch within a few days. "
                "Take care and goodbye!"
            )
            log_transcript("Interviewer", closing)
            speak(closing)

            emotion_monitoring.clear()
            time.sleep(1.5)
            generate_emotion_report()

            if SAVE_TRANSCRIPT:
                save_transcript()

            break

        if EMOTION_AWARE_AI and EMOTION_AVAILABLE:
            ai_reply = ask_groq_with_emotion(user_input)
        else:
            ai_reply = ask_groq(user_input)

        speak(ai_reply)


if __name__ == "__main__":
    main()
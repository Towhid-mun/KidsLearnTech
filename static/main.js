const gradeInput = document.getElementById("grade-input");
const topicInput = document.getElementById("topic-input");
const button = document.getElementById("generate-btn");
const statusEl = document.getElementById("status");
const resultSection = document.getElementById("result");
const planOverviewEl = document.getElementById("plan-overview");
const planStepsEl = document.getElementById("plan-steps");
const audioScriptEl = document.getElementById("audio-script");
const videoNotesEl = document.getElementById("video-notes");
const lessonVideo = document.getElementById("lesson-video");

const renderList = (items = [], container) => {
  container.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    container.appendChild(li);
  });
};

button.addEventListener("click", async () => {
  const grade = gradeInput.value.trim();
  const topic = topicInput.value.trim();
  if (!grade || !topic) {
    statusEl.textContent = "Please enter both grade and topic.";
    resultSection.classList.add("hidden");
    return;
  }

  button.disabled = true;
  statusEl.textContent = "Building plan, generating audio + video ...";
  resultSection.classList.add("hidden");

  try {
    const response = await fetch("/api/generate-lesson", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grade, topic }),
    });

    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    planOverviewEl.textContent = data.plan_overview;
    renderList(data.learning_steps, planStepsEl);
    audioScriptEl.textContent = data.audio_context;
    renderList(data.video_context, videoNotesEl);

    lessonVideo.src = data.video_url;
    lessonVideo.load();

    resultSection.classList.remove("hidden");
    statusEl.textContent = "Lesson ready!";
  } catch (err) {
    console.error(err);
    statusEl.textContent = err.message || "Failed to generate lesson.";
  } finally {
    button.disabled = false;
  }
});

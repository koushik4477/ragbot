<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Persona Builder Test</title>
  <style>
    body { font-family: Arial; background: #f4f6fb; color: #0f172a; margin:0; }
    .wrap { max-width: 600px; margin: 40px auto; padding: 20px; }
    .card { background: #fff; padding: 16px; border-radius: 10px; box-shadow: 0 6px 18px rgba(12,20,40,0.06); margin-top: 16px; }
    .btn { background: #0f172a; color: #fff; padding: 10px 16px; border-radius: 8px; border: 0; cursor: pointer; margin-right: 8px; }
    .chatbox { min-height: 150px; border-radius: 8px; border: 1px solid #dbeafe; padding: 12px; overflow:auto; background:#fbfdff; margin-top:12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <!-- MAIN PAGE -->
    <section id="mainPage" class="card">
      <h2>Main Page</h2>
      <button id="createBtn" class="btn" type="button">Build your own persona</button>
      <div>Click the button to start interview.</div>
    </section>

    <!-- INTERVIEW SECTION -->
    <section id="interviewSection" class="card" style="display:none;">
      <h2>Interview</h2>
      <div id="interviewChat" class="chatbox"></div>
      <input type="text" id="answerInput" placeholder="Type your answer..." />
      <button id="sendAnswer" class="btn" type="button">Send</button>
    </section>
  </div>

<script>
document.addEventListener('DOMContentLoaded', () => {
  const mainPage = document.getElementById('mainPage');
  const createBtn = document.getElementById('createBtn');
  const interviewSection = document.getElementById('interviewSection');
  const interviewChat = document.getElementById('interviewChat');

  const questions = [
    "Describe your daily routine.",
    "What are your hobbies?",
    "How formal should your persona speak?"
  ];
  let currentQ = 0;

  function appendLine(text) {
    const div = document.createElement('div');
    div.textContent = text;
    interviewChat.appendChild(div);
    interviewChat.scrollTop = interviewChat.scrollHeight;
  }

  createBtn.addEventListener('click', e => {
    e.preventDefault();
    mainPage.style.display = 'none';
    interviewSection.style.display = 'block';
    appendLine("System: Starting interview...");
    appendLine("Q: " + questions[currentQ]);
  });

  document.getElementById('sendAnswer').addEventListener('click', e => {
    e.preventDefault();
    const answer = document.getElementById('answerInput').value.trim();
    if(!answer) return;
    appendLine("A: " + answer);
    document.getElementById('answerInput').value = '';
    currentQ++;
    if(currentQ < questions.length){
      appendLine("Q: " + questions[currentQ]);
    } else {
      appendLine("System: Interview complete!");
    }
  });
});
</script>
</body>
</html>

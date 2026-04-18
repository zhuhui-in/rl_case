(async function () {
  const els = {
    hpFill: document.getElementById("hpFill"),
    hpVal: document.getElementById("hpVal"),
    defVal: document.getElementById("defVal"),
    moneyVal: document.getElementById("moneyVal"),
    killedVal: document.getElementById("killedVal"),
    monsters: document.getElementById("monsters"),
    stepIdx: document.getElementById("stepIdx"),
    stepTotal: document.getElementById("stepTotal"),
    actionLabel: document.getElementById("actionLabel"),
    rewardLabel: document.getElementById("rewardLabel"),
    totalReward: document.getElementById("totalReward"),
    actionLog: document.getElementById("actionLog"),
    resetBtn: document.getElementById("resetBtn"),
    prevBtn: document.getElementById("prevBtn"),
    playBtn: document.getElementById("playBtn"),
    nextBtn: document.getElementById("nextBtn"),
    speedSel: document.getElementById("speedSel"),
  };

  let traj;
  try {
    const resp = await fetch("trajectory.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    traj = await resp.json();
  } catch (e) {
    document.body.innerHTML = `<main><h1>Failed to load trajectory.json</h1><p>${e}</p>
      <p>Run <code>python evaluate.py</code> first, then serve via <code>python -m http.server</code>.</p></main>`;
    return;
  }

  const N = traj.config.N;
  const totalSteps = traj.steps.length;
  els.stepTotal.textContent = totalSteps;

  for (let i = 0; i < N; i++) {
    const cell = document.createElement("div");
    cell.className = "monster";
    cell.textContent = "M";
    cell.dataset.idx = i;
    els.monsters.appendChild(cell);
  }

  traj.steps.forEach((s, i) => {
    const li = document.createElement("li");
    const cls = s.action_name === "FIGHT" ? "fight" : s.action_name === "BUY_HEAL" ? "heal" : "def";
    li.className = cls;
    li.textContent = `[${i + 1}] ${s.action_name}  ->  HP=${s.after.hp} d=${s.after.d} $=${s.after.money} killed=${s.after.killed}  (r=${s.reward.toFixed(2)})`;
    li.dataset.idx = i;
    els.actionLog.appendChild(li);
  });

  let cursor = 0;
  let playing = false;
  let timer = null;

  function stateAt(idx) {
    if (idx === 0) {
      return { ...traj.initial, action: "INIT", reward: 0, totalReward: 0 };
    }
    const s = traj.steps[idx - 1];
    let total = 0;
    for (let i = 0; i < idx; i++) total += traj.steps[i].reward;
    return { ...s.after, action: s.action_name, reward: s.reward, totalReward: total };
  }

  function render() {
    const st = stateAt(cursor);
    const hpPct = Math.max(0, Math.min(100, (st.hp / traj.config.H0) * 100));
    els.hpFill.style.width = hpPct + "%";
    els.hpVal.textContent = `${st.hp} / ${traj.config.H0}`;
    els.defVal.textContent = st.d;
    els.moneyVal.textContent = st.money;
    els.killedVal.textContent = `${st.killed} / ${N}`;
    els.stepIdx.textContent = cursor;
    els.actionLabel.textContent = st.action;
    els.rewardLabel.textContent = cursor === 0 ? "-" : (st.reward >= 0 ? "+" : "") + st.reward.toFixed(2);
    els.totalReward.textContent = st.totalReward.toFixed(2);

    const monsters = els.monsters.children;
    for (let i = 0; i < N; i++) {
      monsters[i].classList.remove("dead", "current");
      if (i < st.killed) monsters[i].classList.add("dead");
    }
    if (st.killed < N && cursor > 0 && st.action === "FIGHT") {
      monsters[st.killed - 1].classList.add("current");
    } else if (st.killed < N) {
      monsters[st.killed].classList.add("current");
    }

    Array.from(els.actionLog.children).forEach((li, i) => {
      li.classList.toggle("active", i === cursor - 1);
    });
    if (cursor > 0) {
      const active = els.actionLog.children[cursor - 1];
      if (active) active.scrollIntoView({ block: "nearest" });
    }
  }

  function stop() {
    playing = false;
    if (timer) { clearInterval(timer); timer = null; }
    els.playBtn.textContent = "Play";
  }

  function play() {
    if (cursor >= totalSteps) cursor = 0;
    playing = true;
    els.playBtn.textContent = "Pause";
    const interval = parseInt(els.speedSel.value, 10);
    timer = setInterval(() => {
      if (cursor >= totalSteps) { stop(); return; }
      cursor += 1;
      render();
    }, interval);
  }

  els.playBtn.addEventListener("click", () => (playing ? stop() : play()));
  els.nextBtn.addEventListener("click", () => { stop(); if (cursor < totalSteps) { cursor += 1; render(); } });
  els.prevBtn.addEventListener("click", () => { stop(); if (cursor > 0) { cursor -= 1; render(); } });
  els.resetBtn.addEventListener("click", () => { stop(); cursor = 0; render(); });
  els.speedSel.addEventListener("change", () => { if (playing) { stop(); play(); } });

  render();
})();

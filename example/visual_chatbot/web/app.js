const state = {
  selectedPath: null,
  inspectorMode: "fusion",
};

let currentLang = 'en';
let lastTreeSnapshot = null;
let lastComputationData = null;

const translations = {
  en: {
    app_title: "PAMT Memory Tree",
    app_subtitle: "Live Preference Learning",
    status_ready: "Ready",
    status_error: "Error",
    status_loading: "Loading...",
    status_lost: "Connection Lost",
    chat_title: "Dialogue Stream",
    chat_desc: "Interact with the model to grow the tree.",
    chat_empty: "Start a conversation to begin",
    chat_placeholder: "Type your message...",
    chat_use_tree: "Use Memory Tree",
    chat_use_context: "Use Context",
    chat_compare_hint: "Toggle to compare",
    mode_label: "Mode",
    context_label: "Context",
    mode_tree: "Memory Tree",
    mode_baseline: "Baseline",
    context_on: "On",
    context_off: "Off",
    inspector_view: "View",
    view_fusion: "Fusion",
    view_sw: "SW",
    view_ema: "EMA",
    inspector_title: "Node Inspector",
    inspector_empty: "Select a node to view details...",
    inspector_cold: "Cold Start - No data yet",
    inspector_active: "ACTIVE",
    inspector_empty_badge: "EMPTY",
    inspector_path: "Path",
    inspector_children: "Children",
    debug_title: "Computation",
    debug_idle: "System idle.",
    debug_stage: "Current Stage",
    debug_latency: "Latency Breakdown",
    debug_path: "Traversal Path",
    debug_updates: "Knowledge Updates",
    debug_decision: "Decision",
    debug_memory: "Memory Context",
    memory_path: "Selected Path",
    memory_strategy: "Strategy",
    detail_title: "Dialogue Computation",
    detail_idle: "No computation data yet.",
    detail_metrics: "SW / EMA / Fusion",
    detail_update: "Response Update",
    detail_before: "Before",
    detail_after: "After",
    detail_change: "Change",
    detail_triggered: "Triggered",
    detail_stable: "Stable",
    detail_no_update: "No update data yet.",
    detail_no_metrics: "No SW/EMA/Fusion data yet.",
    canvas_controls: "SCROLL TO ZOOM / DRAG TO PAN",
    tree_branches: "branches",
    tree_waiting: "Waiting for data...",
    lbl_length: "Length",
    lbl_density: "Density",
    lbl_formality: "Formality",
    lbl_tone: "Tone",
    lbl_emotion: "Emotion",
    val_short: "short",
    val_medium: "medium",
    val_long: "long",
    val_sparse: "sparse",
    val_dense: "dense",
    val_casual: "casual",
    val_neutral: "neutral",
    val_formal: "formal",
    val_friendly: "friendly",
    val_joy: "joy",
    val_sadness: "sadness",
    val_anger: "anger",
    val_fear: "fear",
    stage_queued: "Queued",
    stage_start: "Start",
    stage_retrieving: "Retrieving",
    stage_prompting: "Prompting",
    stage_generating: "Generating",
    stage_extracting: "Extracting",
    stage_routing: "Routing",
    stage_updating: "Updating",
    stage_ready: "Ready",
    err_job_failed: "Job fetch failed",
    err_chat_failed: "Chat request failed",
    err_generic: "Sorry, something went wrong. Please try again."
  },
  zh: {
    app_title: "PAMT 记忆树",
    app_subtitle: "实时偏好学习系统",
    status_ready: "就绪",
    status_error: "错误",
    status_loading: "加载中...",
    status_lost: "连接中断",
    chat_title: "对话流",
    chat_desc: "与模型交互以扩展记忆树。",
    chat_empty: "开始对话以启动",
    chat_placeholder: "输入你的消息...",
    chat_use_tree: "使用记忆树",
    chat_use_context: "使用上下文",
    chat_compare_hint: "切换以对比",
    mode_label: "模式",
    context_label: "上下文",
    mode_tree: "记忆树",
    mode_baseline: "基线",
    context_on: "开启",
    context_off: "关闭",
    inspector_view: "视图",
    view_fusion: "融合",
    view_sw: "SW",
    view_ema: "EMA",
    inspector_title: "节点详情",
    inspector_empty: "选择一个节点查看详情...",
    inspector_cold: "冷启动 - 暂无数据",
    inspector_active: "活跃",
    inspector_empty_badge: "空",
    inspector_path: "路径",
    inspector_children: "子节点数",
    debug_title: "计算过程",
    debug_idle: "系统空闲。",
    debug_stage: "当前阶段",
    debug_latency: "延迟拆分",
    debug_path: "检索路径",
    debug_updates: "更新信息",
    debug_decision: "决策",
    debug_memory: "记忆上下文",
    memory_path: "选择路径",
    memory_strategy: "策略",
    detail_title: "对话计算",
    detail_idle: "暂无计算数据。",
    detail_metrics: "SW / EMA / 融合",
    detail_update: "新回复更新",
    detail_before: "更新前",
    detail_after: "更新后",
    detail_change: "变化",
    detail_triggered: "触发",
    detail_stable: "稳定",
    detail_no_update: "暂无更新数据。",
    detail_no_metrics: "暂无 SW/EMA/融合 数据。",
    canvas_controls: "滚轮缩放 / 拖拽平移",
    tree_branches: "分支",
    tree_waiting: "等待数据...",
    lbl_length: "长度",
    lbl_density: "密度",
    lbl_formality: "正式度",
    lbl_tone: "语气",
    lbl_emotion: "情绪",
    val_short: "短",
    val_medium: "中",
    val_long: "长",
    val_sparse: "稀疏",
    val_dense: "密集",
    val_casual: "随和",
    val_neutral: "中性",
    val_formal: "正式",
    val_friendly: "友好",
    val_joy: "喜悦",
    val_sadness: "悲伤",
    val_anger: "愤怒",
    val_fear: "恐惧",
    stage_queued: "排队中",
    stage_start: "开始",
    stage_retrieving: "检索中",
    stage_prompting: "构建提示",
    stage_generating: "生成中",
    stage_extracting: "提取特征",
    stage_routing: "路由中",
    stage_updating: "更新中",
    stage_ready: "就绪",
    err_job_failed: "任务获取失败",
    err_chat_failed: "对话请求失败",
    err_generic: "抱歉，出错了，请重试。"
  }
};

// Stage key mapping from backend to translation keys
const stageKeys = {
  "queued": "stage_queued",
  "start": "stage_start",
  "retrieving": "stage_retrieving",
  "prompting": "stage_prompting",
  "generating": "stage_generating",
  "extracting": "stage_extracting",
  "routing": "stage_routing",
  "updating": "stage_updating",
  "ready": "stage_ready"
};

// --- Translation Helper ---
function t(key) {
  return translations[currentLang][key] || translations['en'][key] || key;
}

function updateStaticContent() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    el.textContent = t(key);
  });
  // Update placeholder
  if (chatText) {
    chatText.placeholder = t('chat_placeholder');
  }
  refreshDynamicPanels();
}

// --- DOM References ---
const treeSvg = document.getElementById("tree-svg");
const treeCanvas = document.querySelector(".tree-canvas");
const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatText = document.getElementById("chat-text");
const chatButton = chatForm ? chatForm.querySelector("button[type='submit']") : null;
const useTreeToggle = document.getElementById("use-tree-toggle");
const useContextToggle = document.getElementById("use-context-toggle");
const nodeDetail = document.getElementById("node-detail");
const debugLog = document.getElementById("debug-log");
const computationDetail = document.getElementById("computation-detail");
const debugToggle = document.getElementById("debug-toggle");
const detailToggle = document.getElementById("detail-toggle");
const statusPill = document.getElementById("status-pill");
const langToggle = document.getElementById("lang-toggle");
const chatPlaceholder = document.getElementById("chat-placeholder");

const accordionItems = [
  { name: "debug", toggle: debugToggle, panel: debugLog },
  { name: "detail", toggle: detailToggle, panel: computationDetail }
];

function setAccordionOpen(item, open) {
  if (!item || !item.toggle || !item.panel) return;
  item.panel.classList.toggle("hidden", !open);
  item.toggle.setAttribute("aria-expanded", open ? "true" : "false");
  const icon = item.toggle.querySelector("[data-accordion-icon]");
  if (icon) icon.classList.toggle("rotate-180", open);
}

function toggleAccordion(name) {
  const target = accordionItems.find((item) => item.name === name);
  if (!target || !target.toggle || !target.panel) return;
  const isOpen = !target.panel.classList.contains("hidden");

  if (isOpen) {
    setAccordionOpen(target, false);
    return;
  }

  accordionItems.forEach((item) => {
    if (!item.toggle || !item.panel) return;
    setAccordionOpen(item, item.name === name);
  });
}

function initAccordions() {
  const activeItems = accordionItems.filter((item) => item.toggle && item.panel);
  if (activeItems.length === 0) return;

  const firstOpen = activeItems.find((item) => item.toggle.getAttribute("aria-expanded") === "true") || activeItems[0];
  activeItems.forEach((item) => setAccordionOpen(item, item === firstOpen));
  activeItems.forEach((item) => {
    item.toggle.addEventListener("click", () => toggleAccordion(item.name));
  });
}

initAccordions();

function refreshDynamicPanels() {
  if (lastComputationData) {
    renderComputation(lastComputationData, { force: true });
  }
  if (lastTreeSnapshot) {
    const selectedNode = state.selectedPath ? findNodeByPath(lastTreeSnapshot, state.selectedPath) : null;
    renderNodeDetail(selectedNode);
  }
}

// --- Tree Layout Constants ---
const nodeWidth = 180;
const nodeHeight = 110;
const nodeOffsetX = 0;
const nodeOffsetY = 0;
const horizontalSpacing = 60;
const verticalSpacing = 40;

// --- Zoom State ---
const zoomState = {
  scale: 1,
  min: 0.5,
  max: 2,
  step: 0.1,
  baseWidth: null,
  baseHeight: null
};

// --- Language Toggle ---
if (langToggle) {
  langToggle.addEventListener("click", () => {
    currentLang = currentLang === 'en' ? 'zh' : 'en';
    langToggle.textContent = currentLang === 'en' ? 'EN / 中' : '中 / EN';
    updateStaticContent();
  });
}

// --- Status Display ---
function setStatus(text, type = 'success') {
  if (!statusPill) return;
  const dot = statusPill.querySelector('span:first-child');
  const label = statusPill.querySelector('span:last-child');

  label.textContent = text;

  // Reset classes
  dot.className = 'w-2 h-2 rounded-full';
  statusPill.className = 'px-2.5 py-1 rounded-full border text-xs font-medium flex items-center gap-1.5 transition-colors';

  if (type === 'success') {
    dot.classList.add('bg-emerald-500');
    statusPill.classList.add('bg-slate-100', 'border-slate-200', 'text-slate-600');
  } else if (type === 'error') {
    dot.classList.add('bg-red-500');
    statusPill.classList.add('bg-red-50', 'border-red-200', 'text-red-700');
  } else if (type === 'working') {
    dot.classList.add('bg-amber-500', 'animate-pulse');
    statusPill.classList.add('bg-amber-50', 'border-amber-200', 'text-amber-700');
  }
}

// --- Chat Rendering ---
function renderMessages(messages) {
  if (!chatWindow) return;

  chatWindow.innerHTML = '';

  if (!messages || messages.length === 0) {
    // Show placeholder
    const placeholder = document.createElement('div');
    placeholder.id = 'chat-placeholder';
    placeholder.className = 'flex flex-col items-center justify-center h-full text-center text-slate-400 p-8 opacity-60';
    placeholder.innerHTML = `
      <svg class="w-10 h-10 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>
      <p class="text-sm" data-i18n="chat_empty">${t('chat_empty')}</p>
    `;
    chatWindow.appendChild(placeholder);
    return;
  }

  messages.forEach(msg => {
    appendMessageElement(msg.role, msg.content);
  });

  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendMessage(role, content) {
  // Remove placeholder if exists
  const placeholder = document.getElementById('chat-placeholder');
  if (placeholder) placeholder.remove();

  appendMessageElement(role, content);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendMessageElement(role, content) {
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble-anim';

  const isUser = role === 'user';
  const alignment = isUser ? 'justify-end' : 'justify-start';
  const bgColor = isUser ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-200 text-slate-800';
  const rounded = isUser ? 'rounded-2xl rounded-tr-md' : 'rounded-2xl rounded-tl-md';

  bubble.innerHTML = `
    <div class="flex ${alignment}">
      <div class="max-w-[85%] px-4 py-2.5 ${bgColor} ${rounded} shadow-sm text-sm leading-relaxed">
        ${escapeHtml(content)}
      </div>
    </div>
  `;

  chatWindow.appendChild(bubble);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatNumber(value, digits = 3) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function normalizeCategoricalPayload(catData) {
  if (!catData) return null;
  if (Array.isArray(catData) && catData.length >= 2) {
    return { index: catData[0], probs: catData[1] || [] };
  }
  if (typeof catData === "object") {
    const index = Number(catData.index);
    const probs = Array.isArray(catData.probs) ? catData.probs : [];
    return { index: Number.isFinite(index) ? index : 0, probs };
  }
  return null;
}

function getCategoricalSummary(catData, labels) {
  const normalized = normalizeCategoricalPayload(catData);
  if (!normalized) return { label: t("val_neutral"), prob: 0 };
  const maxProb = normalized.probs.length ? Math.max(...normalized.probs) : 0;
  const label = labels[normalized.index] || t("val_neutral");
  return { label, prob: maxProb };
}

function formatFusionInline(fusion) {
  if (!fusion) return "-";
  return `len ${formatNumber(fusion.length)} / den ${formatNumber(fusion.density)} / form ${formatNumber(fusion.formality)}`;
}

function formatFusionMetrics(fusion) {
  const toneLabels = [t("val_neutral"), t("val_friendly"), t("val_formal"), t("val_casual")];
  const emotionLabels = [t("val_neutral"), t("val_joy"), t("val_sadness"), t("val_anger"), t("val_fear")];
  const tone = getCategoricalSummary(fusion.tone, toneLabels);
  const emotion = getCategoricalSummary(fusion.emotion, emotionLabels);
  const toneProb = tone.prob ? ` (${Math.round(tone.prob * 100)}%)` : "";
  const emotionProb = emotion.prob ? ` (${Math.round(emotion.prob * 100)}%)` : "";

  return `
    <div class="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] text-slate-600">
      <div class="flex items-center justify-between">
        <span>${t("lbl_length")}</span>
        <span class="font-mono text-slate-700">${formatNumber(fusion.length)}</span>
      </div>
      <div class="flex items-center justify-between">
        <span>${t("lbl_density")}</span>
        <span class="font-mono text-slate-700">${formatNumber(fusion.density)}</span>
      </div>
      <div class="flex items-center justify-between">
        <span>${t("lbl_formality")}</span>
        <span class="font-mono text-slate-700">${formatNumber(fusion.formality)}</span>
      </div>
      <div class="flex items-center justify-between">
        <span>${t("lbl_tone")}</span>
        <span class="text-slate-600">${tone.label}${toneProb}</span>
      </div>
      <div class="flex items-center justify-between">
        <span>${t("lbl_emotion")}</span>
        <span class="text-slate-600">${emotion.label}${emotionProb}</span>
      </div>
    </div>
  `;
}

function renderMetricSection(title, fusion) {
  const content = fusion
    ? formatFusionMetrics(fusion)
    : `<div class="text-[11px] text-slate-400 italic">-</div>`;

  return `
    <div class="rounded-lg border border-slate-100 bg-slate-50/70 p-2">
      <div class="flex items-center justify-between mb-1">
        <span class="text-[10px] font-semibold text-slate-500 uppercase tracking-wide">${title}</span>
      </div>
      ${content}
    </div>
  `;
}

function renderUpdateRow(title, change, weight) {
  if (!change) return "";
  const changeSignal = change.change_signal;
  const isTriggered = changeSignal && changeSignal.overall_triggered;
  const badgeClass = changeSignal
    ? (isTriggered ? "bg-amber-50 text-amber-700" : "bg-emerald-50 text-emerald-600")
    : "bg-slate-100 text-slate-500";
  const badgeText = changeSignal ? (isTriggered ? t("detail_triggered") : t("detail_stable")) : t("detail_change");
  const weightLabel = Number.isFinite(Number(weight)) ? `w=${formatNumber(weight, 2)}` : "w=-";

  return `
    <div class="rounded-lg border border-slate-100 bg-white p-2">
      <div class="flex items-center justify-between mb-1">
        <span class="text-[10px] font-semibold text-slate-700">${title}</span>
        <div class="flex items-center gap-1.5">
          <span class="text-[10px] text-slate-400">${weightLabel}</span>
          <span class="text-[10px] ${badgeClass} px-1.5 py-0.5 rounded-full">${badgeText}</span>
        </div>
      </div>
      <div class="text-[10px] text-slate-500">${t("detail_before")}: <span class="font-mono text-slate-600">${formatFusionInline(change.before)}</span></div>
      <div class="text-[10px] text-slate-500">${t("detail_after")}: <span class="font-mono text-slate-700">${formatFusionInline(change.after)}</span></div>
    </div>
  `;
}

function findNodeByPath(node, path) {
  if (!node || !path || path.length === 0) return null;
  if (Array.isArray(node.path) && node.path.length === path.length) {
    const matches = node.path.every((value, idx) => value === path[idx]);
    if (matches) return node;
  }
  if (!Array.isArray(node.children)) return null;
  for (const child of node.children) {
    const found = findNodeByPath(child, path);
    if (found) return found;
  }
  return null;
}

function getRootName(tree) {
  if (!tree) return null;
  if (Array.isArray(tree.path) && tree.path.length > 0) return tree.path[0];
  return tree.name || null;
}

// --- Preference Label Formatting ---
function formatPreferenceLabels(fusion) {
  if (!fusion) return { lengthLabel: '-', formalityLabel: '-' };

  const lengthVal = fusion.length || 0.5;
  const formalityVal = fusion.formality || 0.5;

  let lengthLabel = t('val_medium');
  if (lengthVal < 0.33) lengthLabel = t('val_short');
  else if (lengthVal > 0.66) lengthLabel = t('val_long');

  let formalityLabel = t('val_neutral');
  if (formalityVal < 0.33) formalityLabel = t('val_casual');
  else if (formalityVal > 0.66) formalityLabel = t('val_formal');

  return { lengthLabel, formalityLabel };
}

// --- Node Detail Rendering ---
function renderNodeDetail(node) {
  if (!nodeDetail) return;

  if (!node) {
    nodeDetail.innerHTML = `
      <div class="flex flex-col items-center justify-center py-6 text-center">
        <div class="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mb-3">
          <svg class="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
          </svg>
        </div>
        <p class="text-sm text-slate-400">${t('inspector_empty')}</p>
      </div>
    `;
    return;
  }

  const hasFusion = node.fusion && Object.keys(node.fusion).length > 0;
  const statusBadge = hasFusion
    ? `<span class="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-50 text-emerald-600 rounded-full text-[10px] font-semibold border border-emerald-100">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
        ${t('inspector_active')}
       </span>`
    : `<span class="inline-flex items-center gap-1 px-2 py-0.5 bg-slate-50 text-slate-500 rounded-full text-[10px] font-semibold border border-slate-200">
        <span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
        ${t('inspector_empty_badge')}
       </span>`;

  // Build preference bars - all 5 metrics
  let fusionHtml = `
    <div class="flex flex-col items-center justify-center py-4 text-center">
      <div class="w-10 h-10 rounded-full bg-amber-50 flex items-center justify-center mb-2">
        <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707"></path>
        </svg>
      </div>
      <p class="text-xs text-slate-400">${t('inspector_cold')}</p>
    </div>
  `;

  if (hasFusion) {
    const fusion = node.fusion;

    // Helper function to get categorical label from tone/emotion
    const getCategoricalLabel = (catData, labels) => {
      if (!catData) return { label: t('val_neutral'), prob: 0.5 };
      // catData is [index, probArray] format
      if (Array.isArray(catData) && catData.length >= 2) {
        const idx = catData[0];
        const probs = catData[1];
        const maxProb = probs ? Math.max(...probs) : 0.5;
        return { label: labels[idx] || t('val_neutral'), prob: maxProb };
      }
      return { label: t('val_neutral'), prob: 0.5 };
    };

    // Tone labels
    const toneLabels = [t('val_neutral'), t('val_friendly'), t('val_formal'), t('val_casual')];
    const toneData = getCategoricalLabel(fusion.tone, toneLabels);

    // Emotion labels
    const emotionLabels = [t('val_neutral'), t('val_joy'), t('val_sadness'), t('val_anger'), t('val_fear')];
    const emotionData = getCategoricalLabel(fusion.emotion, emotionLabels);

    // Continuous metrics
    const lengthVal = (fusion.length !== undefined ? fusion.length : 0.5) * 100;
    const densityVal = (fusion.density !== undefined ? fusion.density : 0.5) * 100;
    const formalityVal = (fusion.formality !== undefined ? fusion.formality : 0.5) * 100;

    // Labels for continuous metrics
    const lengthLabel = lengthVal < 33 ? t('val_short') : lengthVal > 66 ? t('val_long') : t('val_medium');
    const densityLabel = densityVal < 33 ? t('val_sparse') : densityVal > 66 ? t('val_dense') : t('val_medium');
    const formalityLabel = formalityVal < 33 ? t('val_casual') : formalityVal > 66 ? t('val_formal') : t('val_neutral');

    fusionHtml = `
      <div class="space-y-2.5">
        <!-- Tone (Categorical) -->
        <div class="preference-item">
          <div class="flex items-center justify-between mb-1">
            <div class="flex items-center gap-2">
              <div class="w-5 h-5 rounded bg-amber-50 flex items-center justify-center">
                <svg class="w-3 h-3 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                </svg>
              </div>
              <span class="text-[11px] font-medium text-slate-600">${t('lbl_tone')}</span>
            </div>
            <span class="text-[10px] font-semibold text-amber-600 bg-amber-50 px-1.5 py-0.5 rounded">${toneData.label}</span>
          </div>
          <div class="h-1 bg-slate-100 rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-amber-300 to-amber-500 rounded-full transition-all duration-500" style="width: ${toneData.prob * 100}%"></div>
          </div>
        </div>
        
        <!-- Emotion (Categorical) -->
        <div class="preference-item">
          <div class="flex items-center justify-between mb-1">
            <div class="flex items-center gap-2">
              <div class="w-5 h-5 rounded bg-pink-50 flex items-center justify-center">
                <svg class="w-3 h-3 text-pink-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
                </svg>
              </div>
              <span class="text-[11px] font-medium text-slate-600">${t('lbl_emotion')}</span>
            </div>
            <span class="text-[10px] font-semibold text-pink-600 bg-pink-50 px-1.5 py-0.5 rounded">${emotionData.label}</span>
          </div>
          <div class="h-1 bg-slate-100 rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-pink-300 to-pink-500 rounded-full transition-all duration-500" style="width: ${emotionData.prob * 100}%"></div>
          </div>
        </div>
        
        <!-- Length (Continuous) -->
        <div class="preference-item">
          <div class="flex items-center justify-between mb-1">
            <div class="flex items-center gap-2">
              <div class="w-5 h-5 rounded bg-blue-50 flex items-center justify-center">
                <svg class="w-3 h-3 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h7"></path>
                </svg>
              </div>
              <span class="text-[11px] font-medium text-slate-600">${t('lbl_length')}</span>
            </div>
            <span class="text-[10px] font-semibold text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">${lengthLabel}</span>
          </div>
          <div class="h-1 bg-slate-100 rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-blue-300 to-blue-500 rounded-full transition-all duration-500" style="width: ${lengthVal}%"></div>
          </div>
        </div>
        
        <!-- Density (Continuous) -->
        <div class="preference-item">
          <div class="flex items-center justify-between mb-1">
            <div class="flex items-center gap-2">
              <div class="w-5 h-5 rounded bg-teal-50 flex items-center justify-center">
                <svg class="w-3 h-3 text-teal-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"></path>
                </svg>
              </div>
              <span class="text-[11px] font-medium text-slate-600">${t('lbl_density')}</span>
            </div>
            <span class="text-[10px] font-semibold text-teal-600 bg-teal-50 px-1.5 py-0.5 rounded">${densityLabel}</span>
          </div>
          <div class="h-1 bg-slate-100 rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-teal-300 to-teal-500 rounded-full transition-all duration-500" style="width: ${densityVal}%"></div>
          </div>
        </div>
        
        <!-- Formality (Continuous) -->
        <div class="preference-item">
          <div class="flex items-center justify-between mb-1">
            <div class="flex items-center gap-2">
              <div class="w-5 h-5 rounded bg-violet-50 flex items-center justify-center">
                <svg class="w-3 h-3 text-violet-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                </svg>
              </div>
              <span class="text-[11px] font-medium text-slate-600">${t('lbl_formality')}</span>
            </div>
            <span class="text-[10px] font-semibold text-violet-600 bg-violet-50 px-1.5 py-0.5 rounded">${formalityLabel}</span>
          </div>
          <div class="h-1 bg-slate-100 rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-violet-300 to-violet-500 rounded-full transition-all duration-500" style="width: ${formalityVal}%"></div>
          </div>
        </div>
      </div>
    `;
  }

  nodeDetail.innerHTML = `
    <!-- Header -->
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white text-xs font-bold shadow-sm">
          ${node.name.charAt(0).toUpperCase()}
        </div>
        <div>
          <h4 class="font-semibold text-slate-800 text-sm leading-tight">${node.name}</h4>
          <p class="text-[10px] text-slate-400">${(node.children || []).length} ${t('tree_branches')}</p>
        </div>
      </div>
      ${statusBadge}
    </div>
    
    <!-- Path -->
    <div class="mb-4 p-2 bg-slate-50 rounded-lg border border-slate-100">
      <div class="flex items-center gap-1.5 text-[10px] text-slate-500">
        <svg class="w-3 h-3 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"></path>
        </svg>
        <span class="font-mono truncate">${node.path.join(' / ')}</span>
      </div>
    </div>
    
    <!-- Preferences -->
    <div class="pt-3 border-t border-slate-100">
      <h5 class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-3">Preferences (5 metrics)</h5>
      ${fusionHtml}
    </div>
  `;
}

// --- Computation/Debug Log Rendering ---
let lastComputationState = null;

function renderComputationDetail(data) {
  if (!computationDetail) return;

  const progress = data?.progress || {};
  const debug = data?.debug || {};
  const memory = progress.memory || debug.memory;
  const updateInfo = progress.update || debug.update;
  const retrieval = progress.retrieval || debug.retrieval;

  if (!memory && !updateInfo && !retrieval) {
    computationDetail.innerHTML = `
      <div class="p-4 text-slate-400 italic">${t("detail_idle")}</div>
    `;
    return;
  }

  const tree = data?.tree || lastTreeSnapshot;
  const rootName = getRootName(tree);
  let focusPath = null;
  if (updateInfo && Array.isArray(updateInfo.path) && rootName) {
    focusPath = [rootName, ...updateInfo.path];
  }
  const candidatePath = memory?.path || retrieval?.selected_path;
  if (!focusPath && Array.isArray(candidatePath)) {
    focusPath = candidatePath.slice();
  }
  if (focusPath && rootName && focusPath[0] !== rootName && focusPath.length === 2) {
    focusPath = [rootName, ...focusPath];
  }

  const rootNode = tree || null;
  const leafNode = focusPath && tree ? findNodeByPath(tree, focusPath) : null;
  const categoryNode = focusPath && tree && focusPath.length >= 2 ? findNodeByPath(tree, focusPath.slice(0, 2)) : null;
  const metricsNode = leafNode || categoryNode || rootNode;

  const pathText = focusPath ? focusPath.join(" / ") : "-";
  const mode = progress.mode || debug.mode;
  const useContext = progress.use_context ?? debug.use_context;
  const strategy = memory?.strategy || retrieval?.strategy;
  const badges = [];

  if (mode) {
    const modeLabel = mode === "baseline" ? t("mode_baseline") : t("mode_tree");
    badges.push(`<span class="px-2 py-0.5 rounded bg-slate-100 text-slate-600">${t("mode_label")}: ${modeLabel}</span>`);
  }
  if (useContext !== undefined && useContext !== null) {
    const contextLabel = useContext ? t("context_on") : t("context_off");
    badges.push(`<span class="px-2 py-0.5 rounded bg-slate-100 text-slate-600">${t("context_label")}: ${contextLabel}</span>`);
  }
  if (strategy) {
    badges.push(`<span class="px-2 py-0.5 rounded bg-slate-100 text-slate-600">${t("memory_strategy")}: ${escapeHtml(String(strategy))}</span>`);
  }

  const hasMetrics = metricsNode && (metricsNode.sw || metricsNode.ema || metricsNode.fusion);
  const metricsHtml = hasMetrics
    ? [
      renderMetricSection("SW", metricsNode?.sw),
      renderMetricSection("EMA", metricsNode?.ema),
      renderMetricSection("Fusion", metricsNode?.fusion)
    ].join("")
    : `<div class="text-[11px] text-slate-400 italic">${t("detail_no_metrics")}</div>`;

  let updateHtml = `<div class="text-[11px] text-slate-400 italic">${t("detail_no_update")}</div>`;
  if (updateInfo && updateInfo.changes) {
    updateHtml = `
      <div class="space-y-2">
        ${renderUpdateRow("Leaf", updateInfo.changes.leaf, updateInfo.weights?.leaf)}
        ${renderUpdateRow("Category", updateInfo.changes.category, updateInfo.weights?.category)}
        ${renderUpdateRow("Root", updateInfo.changes.root, updateInfo.weights?.root)}
      </div>
    `;
  }

  computationDetail.innerHTML = `
    <div class="p-4 space-y-3">
      <div>
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold uppercase tracking-wider text-slate-400">${t("memory_path")}</span>
          ${metricsNode ? `<span class="text-[10px] text-slate-400">Node: ${escapeHtml(metricsNode.name)}</span>` : ""}
        </div>
        <div class="mt-1 text-[11px] text-slate-600 font-mono break-words">${escapeHtml(pathText)}</div>
        ${badges.length ? `<div class="mt-2 flex flex-wrap gap-1.5 text-[10px]">${badges.join("")}</div>` : ""}
      </div>
      <div class="pt-3 border-t border-slate-100">
        <h5 class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-2">${t("detail_metrics")}</h5>
        <div class="space-y-2">
          ${metricsHtml}
        </div>
      </div>
      <div class="pt-3 border-t border-slate-100">
        <h5 class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-2">${t("detail_update")}</h5>
        ${updateHtml}
      </div>
    </div>
  `;
}

function renderComputation(data, options = {}) {
  const force = options && options.force === true;
  lastComputationData = data;
  renderComputationDetail(data);
  if (!debugLog) return;

  // Create a state key to check if we need to re-render
  const stateKey = JSON.stringify({
    status: data?.status,
    stage: data?.progress?.stage,
    timingKeys: data?.progress?.timing_ms ? Object.keys(data.progress.timing_ms).join(',') : ''
  });

  // Skip re-render if state hasn't changed (prevents flashing)
  if (!force && lastComputationState === stateKey) return;
  lastComputationState = stateKey;

  if (!data || data.status === 'idle' || (!data.progress && data.status !== 'error' && data.status !== 'done')) {
    debugLog.innerHTML = `
      <div class="flex flex-col items-center justify-center py-8 text-center">
        <div class="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mb-3">
          <svg class="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"></path>
          </svg>
        </div>
        <p class="text-sm text-slate-400">${t('debug_idle')}</p>
      </div>
    `;
    return;
  }

  if (data.status === 'error') {
    debugLog.innerHTML = `
      <div class="p-4">
        <div class="flex items-center gap-3 p-3 bg-red-50 border border-red-100 rounded-lg">
          <div class="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0">
            <svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <div class="flex-1 min-w-0">
            <p class="text-sm font-medium text-red-700">${t('status_error')}</p>
            <p class="text-xs text-red-500 truncate">${data.error || t('err_generic')}</p>
          </div>
        </div>
      </div>
    `;
    return;
  }

  const progress = data.progress || {};
  // If status is 'done', treat stage as 'ready'
  const isDone = data.status === 'done';
  const stage = isDone ? 'ready' : (progress.stage || 'unknown');
  const timing = progress.timing_ms || {};

  // Stage icons mapping
  const stageIcons = {
    'queued': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>',
    'start': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>',
    'retrieving': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>',
    'prompting': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path>',
    'generating': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>',
    'extracting': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>',
    'routing': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"></path>',
    'updating': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>',
    'ready': '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>'
  };

  const stageColors = {
    'queued': { bg: '#fffbeb', border: '#fef3c7', icon: '#f59e0b', text: '#b45309', light: '#fde68a' },
    'start': { bg: '#eff6ff', border: '#dbeafe', icon: '#3b82f6', text: '#1d4ed8', light: '#bfdbfe' },
    'retrieving': { bg: '#ecfeff', border: '#cffafe', icon: '#06b6d4', text: '#0e7490', light: '#a5f3fc' },
    'prompting': { bg: '#f5f3ff', border: '#ede9fe', icon: '#8b5cf6', text: '#6d28d9', light: '#ddd6fe' },
    'generating': { bg: '#eef2ff', border: '#e0e7ff', icon: '#6366f1', text: '#4338ca', light: '#c7d2fe' },
    'extracting': { bg: '#faf5ff', border: '#f3e8ff', icon: '#a855f7', text: '#7e22ce', light: '#e9d5ff' },
    'routing': { bg: '#fdf2f8', border: '#fce7f3', icon: '#ec4899', text: '#be185d', light: '#fbcfe8' },
    'updating': { bg: '#fff7ed', border: '#ffedd5', icon: '#f97316', text: '#c2410c', light: '#fed7aa' },
    'ready': { bg: '#ecfdf5', border: '#d1fae5', icon: '#10b981', text: '#047857', light: '#a7f3d0' }
  };

  const colors = stageColors[stage] || { bg: '#f8fafc', border: '#e2e8f0', icon: '#64748b', text: '#475569', light: '#cbd5e1' };
  const icon = stageIcons[stage] || stageIcons['queued'];
  const stageKey = stageKeys[stage] || stage;
  const stageDisplay = t(stageKey);

  // Build timing items
  let timingHtml = '';
  if (Object.keys(timing).length > 0) {
    const timingItems = Object.entries(timing).map(([k, v]) => {
      const maxTime = Math.max(...Object.values(timing), 1);
      const widthPercent = (v / maxTime) * 100;
      const displayTime = Math.round(v); // Round to integer
      return `
        <div class="timing-item">
          <div class="flex items-center justify-between mb-1">
            <span class="text-[10px] text-slate-500 capitalize">${k.replace(/_/g, ' ')}</span>
            <span class="text-[10px] font-mono font-semibold text-slate-700">${displayTime}ms</span>
          </div>
          <div class="h-1 bg-slate-100 rounded-full overflow-hidden">
            <div class="h-full rounded-full transition-all duration-300" style="width: ${widthPercent}%; background: linear-gradient(to right, ${colors.light}, ${colors.icon});"></div>
          </div>
        </div>
      `;
    }).join('');

    timingHtml = `
      <div class="mt-4 pt-4 border-t border-slate-100">
        <h5 class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          ${t('debug_latency')}
        </h5>
        <div class="space-y-2.5">
          ${timingItems}
        </div>
      </div>
    `;
  }

  // Build path breadcrumb
  let pathHtml = '';
  if (progress.path && progress.path.length > 0) {
    const pathItems = progress.path.map((p, i) => {
      const isLast = i === progress.path.length - 1;
      return `
        <span class="inline-flex items-center">
          <span class="px-1.5 py-0.5 rounded text-[10px]" style="background: ${isLast ? colors.bg : '#f1f5f9'}; color: ${isLast ? colors.text : '#475569'}; ${isLast ? 'font-weight: 600;' : ''}">${p}</span>
          ${!isLast ? '<svg class="w-3 h-3 text-slate-300 mx-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>' : ''}
        </span>
      `;
    }).join('');

    pathHtml = `
      <div class="mt-4 pt-4 border-t border-slate-100">
        <h5 class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6"></path>
          </svg>
          ${t('debug_path')}
        </h5>
        <div class="flex flex-wrap items-center gap-0.5">
          ${pathItems}
        </div>
      </div>
    `;
  }

  debugLog.innerHTML = `
    <div class="p-4">
      <!-- Stage Indicator -->
      <div class="flex items-center gap-3 p-3 rounded-xl" style="background: ${colors.bg}; border: 1px solid ${colors.border};">
        <div class="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0" style="background: ${colors.light};">
          <svg class="w-5 h-5 ${stage !== 'ready' ? 'animate-pulse' : ''}" style="color: ${colors.icon};" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            ${icon}
          </svg>
        </div>
        <div class="flex-1 min-w-0">
          <p class="text-xs font-semibold" style="color: ${colors.text};">${stageDisplay}</p>
          <p class="text-[10px]" style="color: ${colors.icon};">${stage !== 'ready' ? 'Processing...' : 'Complete'}</p>
        </div>
        ${stage !== 'ready' ? `
        <div class="flex-shrink-0">
          <div class="w-5 h-5 rounded-full animate-spin" style="border: 2px solid ${colors.light}; border-top-color: ${colors.icon};"></div>
        </div>
        ` : `
        <div class="flex-shrink-0">
          <svg class="w-5 h-5" style="color: ${colors.icon};" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
          </svg>
        </div>
        `}
      </div>
      
      ${timingHtml}
      ${pathHtml}
    </div>
  `;
}

// --- Tree Rendering ---
function flattenTree(node, depth = 0, path = [], parent = null) {
  const currentPath = [...path, node.name];
  const result = [{
    ...node,
    depth,
    path: currentPath,
    _parent: parent,
    _childNodes: [] // Will store references to child node objects
  }];

  const currentNode = result[0];

  if (node.children && node.children.length > 0) {
    node.children.forEach(child => {
      const childResults = flattenTree(child, depth + 1, currentPath, currentNode);
      currentNode._childNodes.push(childResults[0]); // Store reference to child
      result.push(...childResults);
    });
  }

  return result;
}

function layoutTree(root) {
  // Simple tree layout algorithm
  const nodes = flattenTree(root);
  const depthGroups = {};

  nodes.forEach(node => {
    if (!depthGroups[node.depth]) depthGroups[node.depth] = [];
    depthGroups[node.depth].push(node);
  });

  // Assign positions
  Object.keys(depthGroups).forEach(depth => {
    const group = depthGroups[depth];
    const totalWidth = group.length * nodeWidth + (group.length - 1) * horizontalSpacing;
    let startX = -totalWidth / 2 + nodeWidth / 2;

    group.forEach((node, i) => {
      node._x = startX + i * (nodeWidth + horizontalSpacing);
      node._y = depth * (nodeHeight + verticalSpacing);
    });
  });

  return nodes;
}

function renderTree(root) {
  if (!treeSvg || !root) return;

  lastTreeSnapshot = root;

  // Clear existing
  treeSvg.innerHTML = '';

  const nodes = layoutTree(root);

  // Calculate bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  nodes.forEach(node => {
    minX = Math.min(minX, node._x);
    maxX = Math.max(maxX, node._x + nodeWidth);
    minY = Math.min(minY, node._y);
    maxY = Math.max(maxY, node._y + nodeHeight);
  });

  const padding = 60;
  const width = maxX - minX + padding * 2;
  const height = maxY - minY + padding * 2;

  // Offset all nodes so they start from padding
  const offsetX = -minX + padding;
  const offsetY = -minY + padding;

  zoomState.baseWidth = width;
  zoomState.baseHeight = height;
  treeSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  // 1. Render Links (Back layer)
  nodes.forEach(node => {
    if (!node._childNodes || node._childNodes.length === 0) return;

    const parentCenterX = node._x + offsetX + nodeWidth / 2;
    const parentBottomY = node._y + offsetY + nodeHeight;

    node._childNodes.forEach(child => {
      if (!child) return;

      const childCenterX = child._x + offsetX + nodeWidth / 2;
      const childTopY = child._y + offsetY;

      // Curved path
      const startX = parentCenterX;
      const startY = parentBottomY;
      const endX = childCenterX;
      const endY = childTopY;
      const cp1Y = startY + (endY - startY) * 0.5;
      const cp2Y = startY + (endY - startY) * 0.5;

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M ${startX} ${startY} C ${startX} ${cp1Y}, ${endX} ${cp2Y}, ${endX} ${endY}`);
      path.setAttribute("class", "tree-link");
      path.setAttribute("stroke", "#cbd5e1");
      path.setAttribute("stroke-width", "2");
      path.setAttribute("fill", "none");
      treeSvg.appendChild(path);
    });
  });

  // 2. Render Nodes (Front layer)
  nodes.forEach((node) => {
    const x = node._x + offsetX;
    const y = node._y + offsetY;

    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "tree-node");
    group.setAttribute("transform", `translate(${x}, ${y})`);

    const isSelected = state.selectedPath && state.selectedPath.join("/") === node.path.join("/");
    if (isSelected) group.classList.add("active");

    // Card Background (Rect)
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("width", `${nodeWidth}`);
    rect.setAttribute("height", `${nodeHeight}`);
    rect.setAttribute("rx", "12");
    rect.setAttribute("class", "node-card-bg");
    rect.setAttribute("fill", "#ffffff");
    rect.setAttribute("stroke", "#e2e8f0");
    rect.setAttribute("stroke-width", "1");
    group.appendChild(rect);

    // Status Indicator Strip (Top)
    const strip = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    strip.setAttribute("x", "12");
    strip.setAttribute("y", "12");
    strip.setAttribute("width", "24");
    strip.setAttribute("height", "4");
    strip.setAttribute("rx", "2");
    strip.setAttribute("fill", node.has_data ? "#10b981" : "#cbd5e1");
    group.appendChild(strip);

    // Title
    const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
    title.setAttribute("x", "16");
    title.setAttribute("y", "45");
    title.setAttribute("class", "node-title");
    title.setAttribute("font-size", "14");
    title.setAttribute("fill", "#1e293b");
    title.setAttribute("font-weight", "600");
    title.textContent = node.name;
    group.appendChild(title);

    // Stats / Info
    const subtitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
    subtitle.setAttribute("x", "16");
    subtitle.setAttribute("y", "70");
    subtitle.setAttribute("class", "node-subtitle");
    subtitle.setAttribute("font-size", "11");
    subtitle.setAttribute("fill", "#64748b");

    if (node.fusion) {
      const labels = formatPreferenceLabels(node.fusion);
      subtitle.textContent = `${labels.lengthLabel} · ${labels.formalityLabel}`;
    } else {
      subtitle.textContent = t('tree_waiting');
    }
    group.appendChild(subtitle);

    // ID / Children Count
    const meta = document.createElementNS("http://www.w3.org/2000/svg", "text");
    meta.setAttribute("x", "16");
    meta.setAttribute("y", "92");
    meta.setAttribute("class", "node-subtitle");
    meta.setAttribute("font-size", "10");
    meta.setAttribute("fill", "#94a3b8");
    meta.textContent = `${(node.children || []).length} ${t('tree_branches')}`;
    group.appendChild(meta);

    // Click Event
    group.addEventListener("click", (e) => {
      e.stopPropagation();
      state.selectedPath = node.path;
      renderNodeDetail(node);
      renderTree(root);
    });

    treeSvg.appendChild(group);
  });

  // Ensure selection state persists
  const selected = nodes.find((node) => state.selectedPath && node.path.join("/") === state.selectedPath.join("/"));
  if (selected) {
    renderNodeDetail(selected);
  } else if (!state.selectedPath) {
    renderNodeDetail(null);
  }

  applyZoom();
}


// --- Logic & Network ---

async function loadState() {
  setStatus(t('status_loading'), 'working');
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    renderMessages(data.messages || []);
    renderTree(data.tree);
    renderComputation(data);
    setStatus(t('status_ready'), 'success');
  } catch(e) {
    setStatus("Error loading state", 'error');
  }
}

let activeJobId = null;
let pollTimer = null;

function setInputEnabled(enabled) {
  if (chatText) chatText.disabled = !enabled;
  if (chatButton) chatButton.disabled = !enabled;
  if (useTreeToggle) useTreeToggle.disabled = !enabled;
  if (useContextToggle) useContextToggle.disabled = !enabled;
  if (!enabled) {
    if (chatForm) chatForm.classList.add('opacity-50');
    if (chatButton) chatButton.innerHTML = `<svg class="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>`;
  } else {
    if (chatForm) chatForm.classList.remove('opacity-50');
    if (chatButton) chatButton.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M12 5l7 7-7 7"></path></svg>`;
    if (chatText) chatText.focus();
  }
}

async function pollJob() {
  if (!activeJobId) return;

  try {
    const res = await fetch(`/api/job?job_id=${encodeURIComponent(activeJobId)}`);
    if (!res.ok) throw new Error("Job fetch failed");

    const data = await res.json();
    renderComputation(data);

    if (data.status === "done") {
      renderMessages(data.messages || []);
      renderTree(data.tree);
      renderComputation(data);
      setStatus(t('status_ready'), 'success');
      clearInterval(pollTimer);
      pollTimer = null;
      activeJobId = null;
      setInputEnabled(true);
    } else if (data.status === "error") {
      setStatus(t('status_error'), 'error');
      clearInterval(pollTimer);
      pollTimer = null;
      activeJobId = null;
      setInputEnabled(true);
    } else {
      const rawStage = data.progress && data.progress.stage ? data.progress.stage : "Working";
      const stageKey = stageKeys[rawStage] || rawStage;
      const stage = t(stageKey);
      setStatus(stage, 'working');
    }
  } catch (e) {
    setStatus(t('status_lost'), 'error');
    renderComputation({ status: "error", error: t('status_lost') });
    clearInterval(pollTimer);
    setInputEnabled(true);
  }
}

if (chatForm) {
  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = chatText.value.trim();
    if (!message) return;
    const useTree = useTreeToggle ? useTreeToggle.checked : true;
    const useContext = useContextToggle ? useContextToggle.checked : true;

    chatText.value = "";
    appendMessage("user", message);
    setStatus(t('stage_queued'), 'working');
    setInputEnabled(false);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, use_tree: useTree, use_context: useContext }),
      });

      if (!res.ok) throw new Error("Chat request failed");

      const data = await res.json();
      activeJobId = data.job_id;
      renderComputation({ status: "running", progress: { stage: "queued", timing_ms: {} } });

      if (pollTimer) clearInterval(pollTimer);
      await pollJob();
      pollTimer = setInterval(pollJob, 500);
    } catch (e) {
      setStatus(t('status_error'), 'error');
      appendMessage("assistant", t('err_generic'));
      setInputEnabled(true);
    }
  });
}

// --- Pan & Zoom ---

let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let panScrollLeft = 0;
let panScrollTop = 0;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function applyZoom() {
  if (!zoomState.baseWidth || !zoomState.baseHeight) return;
  const width = zoomState.baseWidth * zoomState.scale;
  const height = zoomState.baseHeight * zoomState.scale;
  treeSvg.setAttribute("width", `${width}`);
  treeSvg.setAttribute("height", `${height}`);
}

if (treeCanvas) {
  treeCanvas.addEventListener("mousedown", (event) => {
    if (event.button !== 0) return;
    if (event.target.closest(".tree-node")) return;

    event.preventDefault();

    isPanning = true;
    treeCanvas.style.cursor = "grabbing";
    panStartX = event.pageX;
    panStartY = event.pageY;
    panScrollLeft = treeCanvas.scrollLeft;
    panScrollTop = treeCanvas.scrollTop;
  });

  treeCanvas.addEventListener("wheel", (event) => {
    if (!zoomState.baseWidth) return;

    // Always zoom on wheel (no need for Ctrl key)
    event.preventDefault();
    const rect = treeCanvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    const originX = mouseX + treeCanvas.scrollLeft;
    const originY = mouseY + treeCanvas.scrollTop;

    const direction = event.deltaY < 0 ? 1 : -1;
    const nextScale = clamp(zoomState.scale + direction * zoomState.step, zoomState.min, zoomState.max);

    if (nextScale === zoomState.scale) return;

    const prevScale = zoomState.scale;
    zoomState.scale = nextScale;
    applyZoom();

    const ratio = nextScale / prevScale;
    treeCanvas.scrollLeft = originX * ratio - mouseX;
    treeCanvas.scrollTop = originY * ratio - mouseY;
  }, { passive: false });
}

window.addEventListener("mousemove", (event) => {
  if (!isPanning) return;
  event.preventDefault();
  const dx = event.pageX - panStartX;
  const dy = event.pageY - panStartY;
  if (treeCanvas) {
    treeCanvas.scrollLeft = panScrollLeft - dx;
    treeCanvas.scrollTop = panScrollTop - dy;
  }
});

window.addEventListener("mouseup", () => {
  if (isPanning) {
    isPanning = false;
    if (treeCanvas) treeCanvas.style.cursor = "grab";
  }
});


// Initial Load
updateStaticContent();
loadState();

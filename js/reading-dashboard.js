// Month data is fetched on demand. Only IDs are retained; read state always
// comes from the same store used by the paper cards.
(() => {
  const days = new Map();
  const pending = new Map();
  let selectedMonth = '';
  let generation = 0;
  let panel, monthSelect, status, refresh;

  async function getDay(date) {
    if (days.has(date) && !days.get(date).error) return;
    if (pending.has(date)) return pending.get(date);
    const task = (async () => {
      try {
        const language = selectLanguageForDate(date);
        const response = await fetch(DATA_CONFIG.getDataUrl(`data/${date}_AI_enhanced_${language}.jsonl`), {
          signal: AbortSignal.timeout(30000)
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const text = await response.text();
        // Validate first: a broken response must not look like a completed day.
        text.split('\n').filter(line => line.trim()).forEach(line => JSON.parse(line));
        const papers = text.trim() ? Object.values(parseJsonlData(text, date)).flat() : [];
        const ids = [...new Set(papers.map(getSavedPaperId).filter(Boolean))];
        days.set(date, { ids });
      } catch (error) {
        console.error(`读取 ${date} 阅读统计失败:`, error);
        days.set(date, { error: true });
      }
    })();
    pending.set(date, task);
    try { await task; } finally { pending.delete(date); }
  }

  function monthDates() {
    return availableDates.filter(date => date.startsWith(selectedMonth));
  }

  function render() {
    if (!panel || panel.hidden || !selectedMonth) return;
    const dates = monthDates();
    const available = new Set(dates);
    let total = 0, read = 0, completed = 0, loaded = 0, failed = 0;
    dates.forEach(date => {
      const day = days.get(date);
      if (!day) return;
      if (day.error) { failed++; return; }
      loaded++;
      const count = day.ids.filter(id => readPapers.has(id)).length;
      total += day.ids.length;
      read += count;
      if (day.ids.length > 0 && count === day.ids.length) completed++;
    });
    const loading = dates.length - loaded - failed;
    const partial = loaded !== dates.length;
    status.textContent = loading ? `正在加载：${loaded + failed} / ${dates.length} 天…` :
      failed ? `${failed} 天加载失败，汇总仅包含已加载日期。点击「刷新」重试。` :
      dates.length ? `已统计 ${dates.length} 个推送日期。` : '这个月暂无推送数据。';
    const metrics = document.getElementById('readingMetrics');
    metrics.replaceChildren();
    for (const [label, value] of [['论文总数', total], ['已看过', read], ['待读', total - read], ['完成天数', completed]]) {
      const item = document.createElement('div');
      const number = document.createElement('strong');
      number.textContent = loaded || !dates.length ? String(value) : '—';
      const caption = document.createElement('span');
      caption.textContent = label + (partial ? '（已加载）' : '');
      item.append(number, caption);
      metrics.append(item);
    }
    const percent = total ? Math.round(read / total * 100) : 0;
    document.getElementById('readingMonthPercent').textContent = total ? `${percent}%` : '—';
    document.getElementById('readingMonthProgressLabel').textContent = partial ? '已加载日期的阅读完成率' : '当月阅读完成率';
    const overall = document.getElementById('readingMonthProgress');
    overall.setAttribute('aria-valuenow', percent);
    overall.firstElementChild.style.width = `${percent}%`;
    const calendar = document.getElementById('readingCalendar');
    const fragment = document.createDocumentFragment();
    const [year, month] = selectedMonth.split('-').map(Number);
    const offset = (new Date(year, month - 1, 1).getDay() + 6) % 7;
    const length = new Date(year, month, 0).getDate();
    for (let i = 0; i < offset; i++) {
      const spacer = document.createElement('div');
      spacer.setAttribute('aria-hidden', 'true');
      fragment.append(spacer);
    }
    for (let n = 1; n <= length; n++) {
      const date = `${selectedMonth}-${String(n).padStart(2, '0')}`;
      const day = days.get(date);
      const hasData = available.has(date);
      const count = day?.ids?.filter(id => readPapers.has(id)).length || 0;
      const size = day?.ids?.length || 0;
      const state = !hasData ? 'absent' : !day ? 'loading' : day.error ? 'error' :
        !size ? 'empty' : count === size ? 'complete' : count ? 'started' : 'unread';
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `reading-day reading-day-${state}`;
      button.dataset.date = date;
      button.disabled = !hasData || !day || day.error || !size;
      const caption = !hasData ? '无推送' : !day ? '加载中' : day.error ? '加载失败' : !size ? '0 篇' : `${count} / ${size}`;
      const progress = size ? Math.round(count / size * 100) : 0;
      const detail = size ? (count === size ? '已完成' : count ? `${progress}%` : '未开始') : '';
      button.setAttribute('aria-label', `${date}，${size ? `已看过 ${count} 篇，共 ${size} 篇，${detail}，查看当天论文` : caption}`);
      button.title = button.getAttribute('aria-label');
      // All interpolated values are numeric or generated locally.
      button.innerHTML = `<span class="reading-day-number">${n}</span><strong>${caption}</strong><span class="reading-day-track"><span style="width:${progress}%"></span></span><small>${detail}</small>`;
      if (date === currentDate) button.setAttribute('aria-current', 'date');
      button.addEventListener('click', async () => {
        currentCategory = 'all';
        const url = new URL(window.location);
        url.searchParams.delete('category');
        window.history.replaceState({}, '', url);
        setSavedPapersView(false);
        await loadPapersByDate(date);
        render();
        document.getElementById('paperContainer').scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      fragment.append(button);
    }
    // Preserve the focused calendar date during progress/storage updates.
    const focusedDate = calendar.contains(document.activeElement) ? document.activeElement.dataset.date : null;
    calendar.replaceChildren(fragment);
    if (focusedDate) calendar.querySelector(`[data-date="${focusedDate}"]`)?.focus({ preventScroll: true });
  }

  async function loadMonth(force = false) {
    const run = ++generation;
    refresh.disabled = true;
    try {
      if (!availableDates.length) {
        status.textContent = '正在读取推送日期…';
        await fetchAvailableDates();
        if (run !== generation) return;
      }
      const months = [...new Set(availableDates.map(date => date.slice(0, 7)))].sort().reverse();
      if (!months.length) {
        status.textContent = '暂无可用日期或日期列表加载失败，请点击「刷新」重试。';
        return;
      }
      if (!months.includes(selectedMonth)) selectedMonth = months[0];
      monthSelect.replaceChildren(...months.map(month => new Option(`${month.slice(0, 4)} 年 ${Number(month.slice(5))} 月`, month)));
      monthSelect.value = selectedMonth;
      const dates = monthDates();
      if (force) dates.forEach(date => days.delete(date));
      render();
      let cursor = 0;
      // Bound network concurrency and stop scheduling obsolete month requests.
      await Promise.all(Array.from({ length: Math.min(4, dates.length) }, async () => {
        while (run === generation && cursor < dates.length) {
          await getDay(dates[cursor++]);
          if (run === generation) render();
        }
      }));
    } finally {
      if (run === generation) refresh.disabled = false;
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    panel = document.getElementById('readingDashboard');
    monthSelect = document.getElementById('readingMonth');
    status = document.getElementById('readingDashboardStatus');
    refresh = document.getElementById('refreshReadingDashboard');
    document.getElementById('readingDashboardToggle').addEventListener('click', event => {
      panel.hidden = !panel.hidden;
      event.currentTarget.setAttribute('aria-expanded', String(!panel.hidden));
      if (!panel.hidden) loadMonth();
      else generation++;
    });
    monthSelect.addEventListener('change', () => {
      selectedMonth = monthSelect.value;
      loadMonth();
    });
    refresh.addEventListener('click', () => loadMonth(true));
    window.addEventListener('arxiv:reading-progress', render);
  });
})();

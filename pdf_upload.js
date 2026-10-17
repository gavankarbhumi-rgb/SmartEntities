
(() => {
  const API = "http://127.0.0.1:8000";
  const TYPES = ["PER", "ORG", "LOC", "MISC"];
  const LABELS = {
    PER: "People",
    ORG: "Organizations",
    LOC: "Locations",
    MISC: "Miscellaneous"
  };

  let pdfText = "";
  let pdfEntities = [];
  let comparisonRows = [];

  const style = document.createElement("style");
  style.textContent = `
    .pdf-panel{margin:20px 0;padding:24px;border:1px solid #37345f;
      border-radius:18px;background:linear-gradient(135deg,#17172c,#111827);
      color:#f5f3ff}
    .pdf-panel h2,.pdf-panel h3{color:#fff;margin-top:0}
    .pdf-actions{display:flex;gap:10px;flex-wrap:wrap;margin:15px 0}
    .pdf-btn{border:0;border-radius:10px;padding:11px 16px;cursor:pointer;
      background:linear-gradient(90deg,#7655ed,#367be8);color:white;font-weight:600}
    .pdf-btn:disabled{opacity:.5;cursor:wait}
    .pdf-input{display:block;width:100%;padding:14px;border:1px dashed #7771b5;
      border-radius:12px;background:#17172c;color:white}
    .pdf-muted{color:#b8b7d3;font-size:13px}
    .pdf-status{margin:12px 0;color:#c4b5fd;white-space:pre-wrap}
    .pdf-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin:18px 0}
    .pdf-stat{padding:15px;border:1px solid #383553;border-radius:12px;background:#1d1c35}
    .pdf-stat strong{display:block;font-size:24px;margin-top:6px;color:#c4b5fd}
    .pdf-table-wrap{overflow:auto;margin-top:15px}
    .pdf-table{border-collapse:collapse;width:100%;font-size:14px}
    .pdf-table th,.pdf-table td{text-align:left;padding:10px;border-bottom:1px solid #393650}
    .pdf-table th{color:#c4b5fd}
    .pdf-tag{display:inline-block;border-radius:6px;padding:3px 7px;background:#37305c}
    .pdf-chart-row{display:grid;grid-template-columns:125px 1fr 35px;gap:10px;align-items:center;margin:13px 0}
    .pdf-track{height:12px;background:#302d49;border-radius:9px;overflow:hidden}
    .pdf-fill{height:100%;border-radius:9px;background:linear-gradient(90deg,#8b5cf6,#38bdf8)}
  `;
  document.head.appendChild(style);

  function makePanel(title, description) {
    const panel = document.createElement("section");
    panel.className = "pdf-panel";
    panel.innerHTML = `<h2>${title}</h2><p class="pdf-muted">${description}</p>`;
    return panel;
  }

  // Add PDF upload to the existing NER Analyzer page.
  const analyzer = document.getElementById("page-analyzer");
  if (!analyzer) {
    console.error("PDF feature: could not find #page-analyzer.");
    return;
  }

  const panel = makePanel(
    "PDF Intelligence Analyzer",
    "Upload a text-based PDF. Its extracted text will be analyzed by your existing BERT model."
  );

  panel.insertAdjacentHTML("beforeend", `
    <label for="pdfFile"><strong>Choose a PDF file</strong></label>
    <input class="pdf-input" id="pdfFile" type="file" accept=".pdf,application/pdf">
    <div class="pdf-actions">
      <button class="pdf-btn" id="pdfAnalyze" disabled>Analyze full PDF</button>
      <button class="pdf-btn" id="pdfCompare" disabled>Compare BERT vs CRF</button>
      <button class="pdf-btn" id="pdfCsv" disabled>Download CSV</button>
    </div>
    <div id="pdfStatus" class="pdf-status" role="status">Choose a PDF to begin.</div>
    <div id="pdfStats" class="pdf-stats"></div>
    <div id="pdfResults" class="pdf-table-wrap"></div>
  `);
  analyzer.prepend(panel);

  // Add a separate PDF graph to the existing Analytics page.
  const analytics = document.getElementById("page-analytics");
  let chart;
  if (analytics) {
    chart = makePanel(
      "PDF Entity Distribution",
      "These charts update after a PDF has been analyzed."
    );
    chart.id = "pdfAnalyticsChart";
    analytics.appendChild(chart);
  }

  const $ = id => document.getElementById(id);
  const fileInput = $("pdfFile");
  const analyzeButton = $("pdfAnalyze");
  const compareButton = $("pdfCompare");
  const csvButton = $("pdfCsv");

  function status(message) {
    $("pdfStatus").textContent = message;
  }

  function chunksFromText(text, maxLength = 1200) {
    // Keep chunks small enough for token-classification models.
    const sentences = text.match(/[^.!?。！？\n]+[.!?。！？]?/g) || [text];
    const chunks = [];
    let current = "";

    for (const sentence of sentences) {
      const s = sentence.trim();
      if (!s) continue;

      if (s.length > maxLength) {
        if (current) chunks.push(current);
        current = "";
        for (let i = 0; i < s.length; i += maxLength) {
          chunks.push(s.slice(i, i + maxLength));
        }
        continue;
      }

      if ((current + " " + s).trim().length > maxLength) {
        if (current) chunks.push(current);
        current = s;
      } else {
        current = (current + " " + s).trim();
      }
    }

    if (current) chunks.push(current);
    return chunks;
  }

  async function post(path, text) {
    const response = await fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      throw new Error(`${path} failed (HTTP ${response.status}).`);
    }
    return response.json();
  }

  function renderResults(entities) {
    $("pdfStats").innerHTML = `
      <div class="pdf-stat">Extracted characters<strong>${pdfText.length.toLocaleString()}</strong></div>
      <div class="pdf-stat">Recognized entities<strong>${entities.length.toLocaleString()}</strong></div>
      <div class="pdf-stat">People<strong>${entities.filter(e => e.type === "PER").length}</strong></div>
      <div class="pdf-stat">Organizations<strong>${entities.filter(e => e.type === "ORG").length}</strong></div>
    `;

    if (!entities.length) {
      $("pdfResults").innerHTML =
        "<p>No entities were recognized. Try a text-based English PDF with named people, organizations or places.</p>";
      csvButton.disabled = true;
      return;
    }

    const rows = entities.map((e, i) => `
      <tr>
        <td>${i + 1}</td>
        <td>${escapeHtml(e.text ?? "")}</td>
        <td><span class="pdf-tag">${escapeHtml(e.type ?? "Unknown")}</span></td>
        <td>${Number.isFinite(Number(e.confidence))
          ? (Number(e.confidence) <= 1
              ? (Number(e.confidence) * 100).toFixed(2)
              : Number(e.confidence).toFixed(2)) + "%"
          : "N/A"}</td>
      </tr>
    `).join("");

    $("pdfResults").innerHTML = `
      <table class="pdf-table">
        <thead><tr><th>#</th><th>Entity</th><th>Type</th><th>Confidence</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
    csvButton.disabled = false;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, char => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;"
    })[char]);
  }

  function renderChart(entities) {
    if (!chart) return;
    const counts = Object.fromEntries(
      TYPES.map(type => [type, entities.filter(e => e.type === type).length])
    );
    const max = Math.max(1, ...Object.values(counts));

    chart.innerHTML = `
      <h2>PDF Entity Distribution</h2>
      <p class="pdf-muted">Counts from the most recently analyzed PDF.</p>
      ${TYPES.map(type => `
        <div class="pdf-chart-row">
          <span>${LABELS[type]}</span>
          <div class="pdf-track"><div class="pdf-fill"
            style="width:${counts[type] / max * 100}%"></div></div>
          <strong>${counts[type]}</strong>
        </div>
      `).join("")}
    `;
  }

  fileInput.addEventListener("change", async () => {
    pdfText = "";
    pdfEntities = [];
    comparisonRows = [];
    analyzeButton.disabled = true;
    compareButton.disabled = true;
    csvButton.disabled = true;
    $("pdfStats").replaceChildren();
    $("pdfResults").replaceChildren();

    const file = fileInput.files?.[0];
    if (!file) {
      status("Choose a PDF to begin.");
      return;
    }
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      status("Please choose a PDF file.");
      fileInput.value = "";
      return;
    }

    try {
      status("Reading PDF pages…");
      const buffer = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data: new Uint8Array(buffer) }).promise;
      const pageTexts = [];

      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
        status(`Extracting text: page ${pageNumber} of ${pdf.numPages}…`);
        const page = await pdf.getPage(pageNumber);
        const content = await page.getTextContent();
        const pageText = content.items.map(item => item.str).join(" ");
        if (pageText.trim()) pageTexts.push(pageText);
      }

      pdfText = pageTexts.join("\n\n").trim();
      if (!pdfText) {
        status("No selectable text was found. This may be a scanned PDF; OCR is needed for that file.");
        return;
      }

      analyzeButton.disabled = false;
      compareButton.disabled = false;
      status(`PDF ready: ${pdf.numPages} pages, ${pdfText.length.toLocaleString()} characters extracted.`);
    } catch (error) {
      console.error(error);
      status("Could not read this PDF. Try another PDF or check the browser console.");
    }
  });

  analyzeButton.addEventListener("click", async () => {
    const chunks = chunksFromText(pdfText);
    pdfEntities = [];
    analyzeButton.disabled = true;
    compareButton.disabled = true;
    csvButton.disabled = true;

    try {
      for (let i = 0; i < chunks.length; i++) {
        status(`Analyzing text chunk ${i + 1} of ${chunks.length} with BERT…`);
        const data = await post("/analyze", chunks[i]);
        pdfEntities.push(...(data.entities || []));
      }

      renderResults(pdfEntities);
      renderChart(pdfEntities);
      compareButton.disabled = false;
      status(`Analysis complete. ${pdfEntities.length} entities recognized across ${chunks.length} chunks.`);
    } catch (error) {
      console.error(error);
      status(`Analysis failed: ${error.message}. Check that your FastAPI backend is running.`);
    } finally {
      analyzeButton.disabled = false;
    }
  });

  compareButton.addEventListener("click", async () => {
    if (!pdfText) return;
    const chunks = chunksFromText(pdfText);
    const bert = [];
    const crf = [];
    comparisonRows = [];
    compareButton.disabled = true;

    try {
      for (let i = 0; i < chunks.length; i++) {
        status(`Comparing BERT and CRF: chunk ${i + 1} of ${chunks.length}…`);
        const data = await post("/compare", chunks[i]);
        const b = data.bert?.entities || [];
        const c = data.crf?.entities || [];
        bert.push(...b);
        crf.push(...c);
      }

      comparisonRows = [
        ...bert.map(e => ({ model: "BERT", ...e })),
        ...crf.map(e => ({ model: "CRF", ...e }))
      ];

      const bertCount = bert.length;
      const crfCount = crf.length;
      status(`Comparison complete. BERT: ${bertCount} entities; CRF: ${crfCount} entities. Counts are not accuracy scores.`);
      alert(`PDF comparison complete\n\nBERT entities: ${bertCount}\nCRF entities: ${crfCount}\n\nUse CSV download to export the BERT and CRF predictions.`);
    } catch (error) {
      console.error(error);
      status(`Comparison failed: ${error.message}.`);
    } finally {
      compareButton.disabled = false;
    }
  });

  csvButton.addEventListener("click", () => {
    const rows = comparisonRows.length
      ? comparisonRows.map(e => [e.model, e.text, e.type, e.confidence ?? ""])
      : pdfEntities.map(e => ["BERT", e.text, e.type, e.confidence ?? ""]);

    if (!rows.length) return;

    const quote = value => `"${String(value ?? "").replace(/"/g, '""')}"`;
    const csv = [
      ["Model", "Entity", "Type", "Confidence"],
      ...rows
    ].map(row => row.map(quote).join(",")).join("\r\n");

    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "ner_pdf_results.csv";
    link.click();
    URL.revokeObjectURL(url);
  });
})();

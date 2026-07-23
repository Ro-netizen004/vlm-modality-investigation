import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const input = process.argv[2];
const output = process.argv[3];
if (!input || !output) throw new Error("usage: node build_chartqa_curation_workbook.mjs INPUT.tsv OUTPUT.xlsx");

function parseDelimited(text, delimiter = "\t") {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === delimiter) { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  return rows;
}

const parsed = parseDelimited(await fs.readFile(input, "utf8"));
const headers = parsed[0];
const records = parsed.slice(1).filter(r => r.length > 1).map(values =>
  Object.fromEntries(headers.map((header, i) => [header, values[i] ?? ""]))
);
const workbook = Workbook.create();
const instructions = workbook.worksheets.add("Instructions");
const curation = workbook.worksheets.add("Curation");
const sources = workbook.worksheets.add("Source Tables");

instructions.showGridLines = false;
instructions.getRange("A1:H1").merge();
instructions.getRange("A1").values = [["ChartQA evidence-bearing counterfactual curation"]];
instructions.getRange("A1:H1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 16 }, rowHeight: 30 };
instructions.getRange("A3:B10").values = [
  ["Goal", "Select exactly 300 valid items from the reserve pool. The chart supports A; the report must contain coherent counterfactual facts that entail B."],
  ["Status", "Use include, exclude, or reserve. Every exclusion needs a reason."],
  ["Preferred B", "Prefer another chart value, nearby-category value, rank swap, or arithmetic alternative. Unit-preserving perturbations are last resort."],
  ["Report", "State the evidence needed to answer the same question. Do not write 'the answer is B'."],
  ["Required", "For included rows fill B, strategy, unit, report, entailed=yes, valid=yes, and reviewer."],
  ["Reject", "Gold/table disagreement, normalized-equivalent A/B, unit or date mismatch, ambiguous visual mapping, incoherent derived answers."],
  ["Attribution", "Generated answers use exact normalized final-answer matching only; no fuzzy or reasoning-trace rescore."],
  ["Source", "Use the Source Tables sheet to inspect the official table and suggested chart values."],
];
instructions.getRange("A3:A10").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
instructions.getRange("A3:B10").format.wrapText = true;
instructions.getRange("A3:A10").format.columnWidth = 18;
instructions.getRange("B3:B10").format.columnWidth = 95;
instructions.getRange("A3:B10").format.rowHeight = 42;

const cols = [
  ["conflict_id", "Pool ID"], ["status", "Status"], ["question", "Question"],
  ["image_answer", "A (chart)"], ["text_answer", "B (report)"], ["answer_type", "Type"],
  ["counterfactual_strategy", "Strategy"], ["unit_class", "Unit"], ["text_report", "Evidence-bearing report"],
  ["entailed", "Entailed?"], ["counterfactual_valid", "Valid?"], ["reviewer", "Reviewer"],
  ["exclusion_reason", "Exclusion reason"], ["notes", "Notes"], ["source_table", "Source table"],
];
const curationRows = [cols.map(c => c[1]), ...records.map(r => cols.map(c => r[c[0]]))];
curation.getRangeByIndexes(0, 0, curationRows.length, cols.length).values = curationRows;
curation.showGridLines = false;
curation.freezePanes.freezeRows(1);
curation.freezePanes.freezeColumns(2);
curation.getRange(`A1:O1`).format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" }, rowHeight: 28 };
curation.getRange(`A2:O${curationRows.length}`).format = { verticalAlignment: "top" };
curation.getRange(`C2:C${curationRows.length}`).format.wrapText = true;
curation.getRange(`I2:I${curationRows.length}`).format.wrapText = true;
curation.getRange(`M2:N${curationRows.length}`).format.wrapText = true;
const widths = [10, 11, 48, 12, 12, 10, 24, 18, 72, 11, 11, 16, 45, 38, 35];
widths.forEach((width, i) => curation.getRangeByIndexes(0, i, curationRows.length, 1).format.columnWidth = width);
curation.getRange(`B2:B${curationRows.length}`).dataValidation = { rule: { type: "list", values: ["reserve", "include", "exclude"] } };
curation.getRange(`G2:G${curationRows.length}`).dataValidation = { rule: { type: "list", values: ["chart_value", "nearby_category_value", "rank_swap", "arithmetic_alternative", "unit_preserving_perturbation", "boolean_flip"] } };
curation.getRange(`J2:K${curationRows.length}`).dataValidation = { rule: { type: "list", values: ["", "yes", "no"] } };
curation.getRange(`B2:B${curationRows.length}`).conditionalFormats.add("containsText", { text: "include", format: { fill: "#E2F0D9", font: { color: "#215E21" } } });
curation.getRange(`B2:B${curationRows.length}`).conditionalFormats.add("containsText", { text: "exclude", format: { fill: "#FCE4D6", font: { color: "#9C0006" } } });
curation.getRange(`B2:B${curationRows.length}`).conditionalFormats.add("containsText", { text: "reserve", format: { fill: "#FFF2CC", font: { color: "#7F6000" } } });
curation.tables.add(`A1:O${curationRows.length}`, true, "ChartQACuration").style = "TableStyleMedium2";

const sourceRows = [["Pool ID", "Question", "Official table data", "Candidate chart values"],
  ...records.map(r => [r.conflict_id, r.question, r.table_data, r.chart_value_candidates])];
sources.getRangeByIndexes(0, 0, sourceRows.length, 4).values = sourceRows;
sources.showGridLines = false;
sources.freezePanes.freezeRows(1);
sources.freezePanes.freezeColumns(1);
sources.getRange(`A1:D1`).format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" }, rowHeight: 28 };
sources.getRange(`B2:D${sourceRows.length}`).format.wrapText = true;
[10, 48, 75, 48].forEach((width, i) => sources.getRangeByIndexes(0, i, sourceRows.length, 1).format.columnWidth = width);
sources.getRange(`A2:D${sourceRows.length}`).format = { verticalAlignment: "top" };
sources.tables.add(`A1:D${sourceRows.length}`, true, "ChartQASources").style = "TableStyleMedium2";

instructions.getRange("D3:E8").values = [
  ["Live status", "Count"], ["Included", null], ["Excluded", null], ["Reserve", null], ["Reviewed", null], ["Remaining to 300", null],
];
instructions.getRange("E4:E8").formulas = [
  [`=COUNTIF('Curation'!$B$2:$B$${curationRows.length},"include")`],
  [`=COUNTIF('Curation'!$B$2:$B$${curationRows.length},"exclude")`],
  [`=COUNTIF('Curation'!$B$2:$B$${curationRows.length},"reserve")`],
  ["=E4+E5"], ["=MAX(0,300-E4)"],
];
instructions.getRange("D3:E3").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" } };
instructions.getRange("D4:D8").format = { fill: "#D9EAF7", font: { bold: true } };
instructions.getRange("D3:E8").format.borders = { preset: "outside", style: "thin", color: "#9EADBA" };
instructions.getRange("D3:D8").format.columnWidth = 22;
instructions.getRange("E3:E8").format.columnWidth = 14;

await fs.mkdir(new URL(".", `file:///${output.replaceAll("\\", "/")}`).pathname, { recursive: true }).catch(() => {});
const blob = await SpreadsheetFile.exportXlsx(workbook);
await blob.save(output);
for (const sheetName of ["Instructions", "Curation", "Source Tables"]) {
  const preview = await workbook.render({ sheetName, range: sheetName === "Instructions" ? "A1:H12" : (sheetName === "Curation" ? "A1:O12" : "A1:D8"), scale: 1, format: "png" });
  await fs.writeFile(output.replace(/\.xlsx$/i, `_${sheetName.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log((await workbook.inspect({ kind: "table", range: "Instructions!D3:E8", include: "values,formulas", tableMaxRows: 8, tableMaxCols: 3 })).ndjson);

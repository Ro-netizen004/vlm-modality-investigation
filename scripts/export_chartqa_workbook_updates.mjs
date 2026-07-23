import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  throw new Error("usage: node export_chartqa_workbook_updates.mjs INPUT.xlsx OUTPUT.json");
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const sheet = workbook.worksheets.getItem("Curation");
const used = sheet.getUsedRange(true);
const values = used.values;
const displayToField = {
  "Pool ID": "conflict_id",
  "Status": "status",
  "B (report)": "text_answer",
  "Strategy": "counterfactual_strategy",
  "Unit": "unit_class",
  "Evidence-bearing report": "text_report",
  "Entailed?": "entailed",
  "Valid?": "counterfactual_valid",
  "Reviewer": "reviewer",
  "Exclusion reason": "exclusion_reason",
  "Notes": "notes",
};
const headers = values[0].map(value => String(value ?? "").trim());
const updates = [];
for (const row of values.slice(1)) {
  const update = {};
  for (let column = 0; column < headers.length; column++) {
    const field = displayToField[headers[column]];
    if (field) update[field] = String(row[column] ?? "").trim();
  }
  if (update.conflict_id !== "") {
    update.conflict_id = Number(update.conflict_id);
    updates.push(update);
  }
}
await fs.writeFile(output, JSON.stringify(updates, null, 2), "utf8");
console.log(JSON.stringify({ rows: updates.length, status: updates.reduce((counts, row) => {
  const status = row.status || "blank";
  counts[status] = (counts[status] || 0) + 1;
  return counts;
}, {}) }));

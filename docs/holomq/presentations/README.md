# holomq presentations

Slide decks for holomq design and architecture topics.

## Decks

| File | Topic |
|------|--------|
| [`holomq-high-level-design.md`](holomq-high-level-design.md) | Public API design (bottom-up) |
| [`buffer-sync-deep-dive.md`](buffer-sync-deep-dive.md) | GPU buffer synchronization deep dive (companion to HLD) |

Contributor design notes: [`../design-notes/README.md`](../design-notes/README.md).

## Format: [Marp](https://marp.app/)

Slides are plain Markdown with [Marp](https://marp.app/). Decks use a **hybrid** header:

- YAML front matter with only `marp: true` — required by the VS Code / Cursor Marp extension to enable slide mode.
- An HTML comment block for everything else (`paginate`, `header`, `footer`, `style`, …) — hidden from standard Markdown renderers.

```markdown
---
marp: true
---

<!--
paginate: true
header: 'my deck'
footer: 'NVIDIA holomq'
style: |
  section { font-size: 26px; }
-->

# First slide
```

On GitHub, the YAML front matter is not shown in the rendered page. In the VS Code plain Markdown preview you may still see `marp: true` between horizontal rules; use the Marp preview for slides.

### Option A — VS Code / Cursor extension

1. Install the **Marp for VS Code** extension (`marp-team.marp-vscode`).
2. Open a `.md` deck in this folder.
3. Use the preview pane, or export to PDF / HTML / PPTX from the command palette (`Marp: Export slide deck`).

### Option B — Marp CLI

```bash
# Install (Node.js)
npm install -g @marp-team/marp-cli

# HTML (single file, good for sharing in browser)
marp holomq/docs/presentations/buffer-sync-deep-dive.md -o buffer-sync-deep-dive.html

# PDF (requires Chrome/Chromium)
marp holomq/docs/presentations/buffer-sync-deep-dive.md --pdf -o buffer-sync-deep-dive.pdf

# PowerPoint
marp holomq/docs/presentations/buffer-sync-deep-dive.md --pptx -o buffer-sync-deep-dive.pptx
```

Run from the repository root, or adjust paths accordingly.

### Option C — reveal.js (manual)

Marp can export HTML; reveal.js is another common open-source option if you prefer that toolchain. The Markdown content in these decks can be adapted with minimal changes (split on `---` slide breaks).

## Adding a new deck

1. Create `your-topic.md` with an opening Marp HTML comment (copy from an existing deck).
2. Separate slides with `---` on its own line.
3. List it in the table above.

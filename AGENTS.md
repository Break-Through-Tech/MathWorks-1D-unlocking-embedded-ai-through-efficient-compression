# AGENTS.md

## Project Layout

| Path | Contents |
|------|----------|
| `data/` | Dataset files (`train.mat`, `val.mat`, `test.mat`) |
| `notebooks/` | Jupyter notebooks, one per task |
| `tasks/` | `TASKS.md` (the task index) plus one write-up per task |

## Naming Convention

Every task can have a **notebook** and a **task write-up**. They share the same name and differ only by extension, so it's always obvious which files belong together.

```
notebooks/M1_02_initial_data_analysis.ipynb
tasks/M1_02_initial_data_analysis.md
```

### Rules

1. **Format:** `M<milestone>_NN_short_description`
   - `M<milestone>` is an uppercase `M` followed by the milestone number from `tasks/TASKS.md` (`M1` = September, `M2` = the next milestone, and so on).
   - `NN` is the two-digit task number within that milestone (`01`, `02`, … `10`).
   - `short_description` is lowercase `snake_case`: letters, digits, and underscores only. No spaces, hyphens, or capitals.
   - Keep it short (2–4 words) and descriptive of the work, e.g. `M1_03_equal_row_counts`, `M1_04_visualize_signals`.
2. **Extensions:** notebooks are `.ipynb` in `notebooks/`, write-ups are `.md` in `tasks/`.
3. **Matching names:** a task's notebook and write-up use the exact same `M<milestone>_NN_short_description`.
4. **Numbering:** task numbers restart at `01` for each milestone. The milestone prefix keeps names unique (`M1_01_...` and `M2_01_...` never clash).
5. **Location:** all task Markdown files live in `tasks/`, never at the repo root.
6. **Exception:** `tasks/TASKS.md` is the task index and keeps its name. When you add a write-up, link it from that task's entry in `TASKS.md`.

### Examples

| ✅ Good | ❌ Bad | Why it's bad |
|--------|--------|--------------|
| `M1_02_initial_data_analysis.md` | `Task2-Initial-Data-Analysis.md` | Capitals, hyphens, missing milestone and task number |
| `M1_05_standardize_data.ipynb` | `m1_05_standardize_data.ipynb` | Milestone prefix must use an uppercase `M` |
| `M2_01_train_model.ipynb` | `01_train_model.ipynb` | Missing milestone prefix |
| `M1_04_visualize_signals.ipynb` | `M1_4 Visualize Signals.ipynb` | Task number must be two digits; no spaces or capitals |

### Enforcement

Agents: every new or renamed notebook or task file must follow these rules. If you find an existing file that breaks them, point it out and suggest the correct name. Don't rename it without asking.

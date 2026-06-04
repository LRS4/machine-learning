# Titanic Data Analysis — Pandas & Tidyverse Examples

Side-by-side reference for the two example scripts in this repo:
- [pandas_examples.py](pandas_examples.py) — Python / Pandas
- [tidyverse_examples.R](tidyverse_examples.R) — R / Tidyverse

---

## Section overview

| Section | Operations covered |
|---|---|
| **Load** | Read CSV / built-in dataset into a dataframe |
| **Inspect** | Shape, dtypes, head, descriptive stats, missing values |
| **Selection** | Column select, boolean filter, label/position indexing, query |
| **Cleaning** | Fill NA, drop NA, drop columns, rename, cast types, deduplicate |
| **Feature engineering** | Arithmetic columns, binning, apply / mutate |
| **Aggregation** | groupby / group\_by, named aggs, broadcast transform, value counts |
| **Pivot / reshape** | pivot table, melt / pivot\_longer, crosstab |
| **Sort & rank** | Top-N, rank |
| **Join** | Left join on a lookup table |
| **String ops** | Upper-case, extract, starts-with, length |
| **Window functions** | Rolling mean, cumulative sum |
| **Visualisation** | Bar, histogram, box plot, grouped bar |
| **Export** | CSV and JSON output |

---

## Pandas vs Tidyverse — operation mapping

| Section | pandas | tidyverse equivalent |
|---|---|---|
| Load | `pd.read_csv` / `sns.load_dataset` | `as_tibble(titanic_train)` |
| Inspect | `dtypes`, `describe`, `isnull` | `glimpse`, `summary`, `summarise(across(..., is.na))` |
| Selection | `[]`, `loc` / `iloc`, `query()` | `select`, `filter`, `slice` |
| Cleaning | `fillna`, `dropna`, `drop`, `astype` | `replace_na`, `drop_na`, `select(-col)`, `as.logical` |
| Feature engineering | arithmetic, `pd.cut`, `apply()` | `mutate`, `cut`, `sqrt` |
| Aggregation | `groupby().agg()`, `transform` | `group_by + summarise`, `mutate` (broadcast) |
| Pivot / reshape | `pivot_table`, `melt`, `crosstab` | `pivot_wider`, `pivot_longer`, `count + pivot_wider` |
| Sort & rank | `nlargest`, `rank` | `slice_max`, `min_rank` |
| Join | `merge(how="left")` | `left_join` |
| Strings | `.str.*` | `str_to_upper`, `str_extract`, `str_length` |
| Window | `rolling`, `cumsum` | `slide_dbl` (slider pkg), `cumsum` |
| Visualisation | `matplotlib` | `ggplot2` + `patchwork` |
| Export | `to_csv`, `to_json` | `write_csv`, `jsonlite::write_json` |

---

## Requirements

**Python**
```
pip install pandas seaborn matplotlib
```

**R**
```r
install.packages(c("tidyverse", "titanic", "patchwork", "slider", "jsonlite"))
```

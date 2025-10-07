# PCprophet

PCprophet is a tool for predicting and analyzing protein complexes from co-fractionation mass spectrometry (MS) data, with both hypothesis-driven (from databases) and data-driven discovery approaches. It includes built-in differential analysis, GO-based scoring, and support for visualization in Cytoscape.

---

## 🔧 Quick Start

After installing dependencies, run PCprophet on the example dataset:

```bash
python3 main.py -sid test/test_ids.txt -is_ppi False -db coreComplexes.txt
```

- Default input folder: `./test/`
- Default output folder: `./Output/`
- Runtime: ~15 minutes per SEC sample.

---

## 📁 Input Format

Input data must be a wide-format matrix with:

- `gene_name`: maps to `subunits_gene_name` in your database.
- `protein_id`: optional; UniProt, Ensembl, or NCBI identifier. Not essential
  (can be the same as *gene_name*)

Remaining columns = ordered fraction intensities. Column names can be any
format (e.g., `F01`, `1`, etc.), but **must be in correct order**. Formats like
MS1, MS2 XIC, SPCs, TMT, and SILAC are supported. The best results are obtained
using DIA-MS or DDA-MS data.

See `test/test_fract.txt` for examples.

*NOTE*: while called *gene_name* no lookup is done to confirm gene_names are
used. Any identifier works as long as it matches into *subunits_gene_name*

*NOTE*: Several search engines report protein groups concatenated by a
semicolumn. Replace it with another separator (like . or _ ) because ; defines
complex subunits as well so number of subunits per complex gets inflated due to
the mixup between protein group components and protein complex subunits

---

## Parameter Overview

All parameters can be configured via:

1. `ProphetConfig.conf`
2. Command line arguments

Command-line arguments always override the config file.

```bash
python3 main.py --help
```

### Key Parameters

#### Global

| Flag         | Description                                      | Default                     |
|--------------|--------------------------------------------------|-----------------------------|
| `-db`        | Path to CORUM or PPI database                    | `meta/corum_allComplexes.txt` |
| `-sid`       | Sample ID file                                   | `./sample_ids.txt`          |
| `-output`    | Output folder path                               | `./Output`                  |
| `-cal`       | Calibration file (fraction to MW)                | None                        |
| `-mw_uniprot`| Uniprot-based mass file                          | None                        |
| `-v`         | Verbose output                                   | 1                           |
| `-skip`      | Skip feature generation                          | False                       |
| `-dif`       | Run differential analysis                        | True                        |

#### Pre-processing

| Flag       | Description                                        | Default |
|------------|----------------------------------------------------|---------|
| `-is_ppi`  | Use PPI network instead of complex DB              | False   |
| `-a`       | Use all or subset of fractions                     | all     |
| `-ma`      | Hypothesis generation mode (`all` or `reference`)  | all     |

#### Post-processing

| Flag       | Description                                      | Default |
|------------|--------------------------------------------------|---------|
| `-co`      | Collapse strategy (`GO`, `CAL`, `PROB`, `SUPER`, `NONE`) | GO      |
| `-fdr`     | False discovery rate threshold                   | 0.2     |

---

## Sample ID File

The `sample_ids.txt` must include:

| Sample           | cond   | group | short_id  | repl | fr |
|------------------|--------|-------|-----------|------|----|
| `./Input/c1r1.txt` | Ctrl   | 1     | ipsc_2i | 1    | 65 |

- `Sample`: Full path to input file  
- `cond`: Experimental condition (e.g., Ctrl, Treat1)  
- `group`: 1 for control, 2+ for treatments  
- `short_id`: Short identifier -> can be anything but needs to match group i.e.
  all group ==1 should have the same identifier
- `repl`: Replicate number  
- `fr`: Number of fractions

> **Tip:** For missing fractions, pad with columns of 0s instead of removing them.

---

## Database Formats

### Complex DB (CORUM)

Must include:

- `complex_id`
- `complex_name`
- `subunits_gene_name` (semicolon-separated)

### PPI Network

| protein1 | protein2 |
|-------|-------|
| A     | B     |

If a PPI is used (`-is_ppi True`), PCprophet clusters it using MCL and assigns IDs as `ppi__<cluster_id>`.

---

## Collapse Modes (`-co`)

| Mode   | Description |
|--------|-------------|
| `GO`   | Keep complex with best GO score per dendrogram branch |
| `PROB` | Keep complex with highest prediction probability |
| `SUPER`| Keep superset (most members) |
| `CAL`  | Match to MW using calibration file and uniprot data |
| `eCAL`  | Fit fraction number to number of subunits. A good proxy if there are complexes in the whole MW range and no calibration is available |

| `NONE` | Keep all |

> **Recommendation:** Use `CAL` only if you have a full calibration curve covering the MW range. Otherwise, use `GO`.

---

## Running PCprophet

With defaults:

```bash
python3 main.py
```

With custom FDR and skipping feature generation:

```bash
python3 main.py -skip True -fdr 0.5
```

With a PPI network:

```bash
python3 main.py -db myppi.txt -is_ppi True
```

---

## 📊 Output Files

All outputs are saved in:

- `./tmp/`: intermediate results can be safely deleted
- `./Output/`: final results

### Main Outputs

| File | Description |
|------|-------------|
| `complex_report.txt` | Final complexes |
| `ppi_report.txt`     | Network view of complexes (Cytoscape-compatible) |

### Differential Output

| File | Description |
|------|-------------|
| `differential_complex_report.txt` | Complex-level Bayesian comparisons |
| `differential_protein_report.txt` | Protein-level differential probabilities |

---

## Cytoscape Integration

- Import `ppi_report.txt` directly.
- Use `differential_protein_report.txt` to overlay differential nodes.

---

## 🛠 Common Errors

| Error | Cause |
|-------|-------|
| `MissingColumnError` | Missing `gene_name` or `protein_id` |
| `DuplicateRowError` | Duplicate gene names (e.g., isoforms). Add `_1` suffix to resolve |

---

## ❓ FAQ

**Why are multiple complex IDs joined with `#` in `complex_report.txt`?**  
Parsimony: complexes with the same identified proteins are grouped.

**How to speed up re-running with different FDR/collapse settings?**  
Use `-skip True` to bypass preprocessing:

```bash
python3 main.py -skip True -fdr 0.3 -co SUPER
```

**`-co CAL` gives error?**  
Check your calibration file format:

| Fraction | MW (kDa) |
|----------|----------|
| 15       | 1398     |
| 24       | 699      |

Also make sure `-mw_uniprot` is passed and formatted like:

- Columns: `Gene names`, `Mass`
- File type: tab-separated `.txt`

> **Note:** The mass column should contain comma-separated values if multiple entries exist.

**What is `-eCAL`?**  
This is an experimental version of CAL mode that uses extrapolation without
needing UniProt MW. It utilizes the number of subunits as proxy for MW using
reported complexes and select the complex having the closest number of subunits
to the predicted number.


---

## Contact

For support, contact or issues, visit the [PCprophet GitHub repository](https://github.com/fossatiA/PCprophet).
import re
import pandas as pd
import PCprophet.exceptions as PCpexc


class InputTester:
    """
    Validate inputs before anything else.

    Parameters
    ----------
    path : str | None
        Path to input file. Used if `infile` is None.
    filetype : {"ids","db","in"}
        Which schema to validate against.
    infile : pd.DataFrame | None
        If provided, use this instead of reading from `path`.
    """

    def __init__(self, path, filetype, infile=None):
        self.infile = infile
        self.filetype = filetype
        self.path = path

    # ---------- IO ----------
    def read_infile(self):
        if self.infile is not None:
            return
        if self.path is None:
            raise ValueError("Either `infile` or a valid `path` must be provided.")
        # sensible default separator from extension
        sep = "\t" if str(self.path).lower().endswith((".tsv", ".txt")) else ","
        # treat common textual NA markers as NA
        self.infile = pd.read_csv(
            self.path,
            sep=sep,
            na_values=["NA", "NaN", ""],
            keep_default_na=True,
        )

    # ---------- Column checks ----------
    def test_missing_col(self, required_cols):
        missing = [c for c in required_cols if c not in self.infile.columns]
        if missing:
            raise PCpexc.MissingColumnError(missing=missing, path=self.path)

    def test_empty(self, cols):
        for c in cols:
            if c not in self.infile.columns:
                raise PCpexc.MissingColumnError(missing=[c], path=self.path)
            s = self.infile[c]
            empty_mask = s.isna() | (s.astype(str).str.strip() == "")
            if empty_mask.any():
                idx = empty_mask[empty_mask].index[:10]
                raise PCpexc.EmptyColumnError(column=c, path=self.path, indices=idx)

    def test_uniqueid(self, cols):
        if not cols:
            return
        dup_mask = self.infile.duplicated(subset=cols, keep=False)
        if dup_mask.any():
            # optional: print a compact view of offending keys
            dups = self.infile.loc[dup_mask, cols].drop_duplicates()
            if not dups.empty:
                print(f"Duplicated key combinations in {self.path}:")
                print(dups)
            raise PCpexc.DuplicateIdentifierError(keys=cols, path=self.path)

    def test_na_anywhere(self):
        if self.infile.isna().values.any():
            raise PCpexc.NaInMatrixError(path=self.path)

    # ---------- Schema-specific checks ----------
    def test_cond_values(self):
        """
        Require 'Ctrl' and sequential TreatN (Treat1..TreatK) with no gaps and no extras.
        """
        if "cond" not in self.infile.columns:
            raise PCpexc.MissingColumnError(missing=["cond"], path=self.path)

        cond_values = set(map(str, self.infile["cond"].dropna().unique()))
        if not cond_values:
            raise PCpexc.ConditionError("cond column is empty", path=self.path)

        if "Ctrl" not in cond_values:
            raise PCpexc.ConditionError("missing 'Ctrl'", path=self.path)

        treat_pattern = re.compile(r"^Treat(\d+)$")
        treat_nums = sorted(
            int(m.group(1))
            for v in cond_values
            if (m := treat_pattern.match(v)) is not None
        )

        if not treat_nums:
            raise PCpexc.ConditionError("missing TreatN (e.g., Treat1)", path=self.path)

        # Only allow Ctrl + TreatN labels
        allowed = {"Ctrl"} | {f"Treat{n}" for n in treat_nums}
        extras = sorted(cond_values - allowed)
        if extras:
            raise PCpexc.ConditionError(f"unexpected labels: {extras}", path=self.path)

        # Must be sequential from 1..K with no gaps
        expected = list(range(1, max(treat_nums) + 1))
        if treat_nums != expected:
            found_str = ", ".join(f"Treat{n}" for n in treat_nums) or "none"
            raise PCpexc.ConditionError(
                f"TreatN must be sequential from Treat1..Treat{max(expected)} (found: {found_str})",
                path=self.path,
            )

    # ---------- Orchestrator ----------
    def test_all(self, required_cols, unique_cols=None, also_empty_cols=None, check_na=False):
        self.test_missing_col(required_cols)
        if also_empty_cols:
            self.test_empty(also_empty_cols)
        if unique_cols:
            self.test_uniqueid(unique_cols)
        if check_na:
            self.test_na_anywhere()

    def test_file(self):
        self.read_infile()

        if self.filetype == "ids":
            required = ["Sample", "cond", "group", "short_id", "repl", "fr"]
            unique = ["short_id", "repl", "fr"]
            self.test_all(required_cols=required, unique_cols=unique, also_empty_cols=required)
            self.test_cond_values()

        elif self.filetype == "db":
            required = ["complex_id", "complex_name", "subunits_gene_name"]
            try:
                self.test_all(
                    required_cols=required,
                    unique_cols=["complex_id", "complex_name"],
                    also_empty_cols=required,
                )
            except PCpexc.MissingColumnError:
                # fallback: simple PPI edges
                required_ppi = ["protein1", "protein2"]
                self.test_all(required_cols=required_ppi, also_empty_cols=required_ppi)

        elif self.filetype == "in":
            required = ["gene_name", "protein_id"]
            self.test_all(
                required_cols=required,
                unique_cols=["gene_name"],
                also_empty_cols=required,
                check_na=True,
            )

        else:
            raise ValueError(f"Unknown filetype: {self.filetype}")

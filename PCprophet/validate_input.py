import re
import PCprophet.exceptions as PCpexc
import pandas as pd


class InputTester(object):
    """
    validate all inputs before anything
    infile is a pandas dataframe
    """

    def __init__(self, path, filetype, infile=None):
        super(InputTester, self).__init__()
        self.infile = infile
        self.filetype = filetype
        self.path = path

    def read_infile(self):
        self.infile = pd.read_csv(self.path, sep="\t", index_col=False)

    def test_missing_col(self, col):
        if not all([x in self.infile.columns for x in col]):
            raise PCpexc.MissingColumnError(self.path)

    def test_empty(self, col):
        for x in col:
            if self.infile[x].isnull().values.any():
                raise PCpexc.EmptyColumnError(self.path)

    def test_uniqueid(self, totest):
        if self.infile.duplicated(totest).any():
            print("The following rows in {} are duplicated".format(self.path))
            print(self.infile[self.infile.duplicated(totest)])
            raise PCpexc.DuplicateIdentifierError(self.path)

    def test_all(self, *args):
        self.test_missing_col(args[0])
        self.test_uniqueid(args[1])

    def test_na(self):
        if self.infile.isnull().values.any():
            raise PCpexc.NaInMatrixError(self.path)

    def test_cond_values(self):
        """
        Check that 'cond' column contains 'Ctrl' and sequential TreatN starting from 1.
        """
        cond_values = set(self.infile["cond"].unique())

        if "Ctrl" not in cond_values:
            raise PCpexc.ConditionError(f"{self.path}: missing 'Ctrl' in cond column")
        # Extract TreatN values
        treat_pattern = re.compile(r"^Treat(\d+)$")
        treat_nums = sorted(int(m.group(1)) for val in cond_values if (m := treat_pattern.match(val)))

        if not treat_nums:
            raise PCpexc.ConditionError(f"{self.path}: missing 'TreatN' (e.g. Treat1) in cond column")

        # Check sequential order starting at 1
        expected = list(range(1, max(treat_nums) + 1))
        if treat_nums != expected:
            raise PCpexc.ConditionError(
                f"{self.path}: TreatN conditions must be sequential from Treat1 to Treat{max(expected)} "
                f"(found: {', '.join(f'Treat{n}' for n in treat_nums)})"
            )

    def test_file(self):
        self.read_infile()
        if self.filetype == "ids":
            col = ["Sample", "cond", "group", "short_id", "repl", "fr"]
            unique = ["repl", "short_id"]
            self.test_all(col, unique)
            self.test_empty(col)
            self.test_cond_values()  # <-- new check
        elif self.filetype == "db":
            try:
                col = ["complex_id", "complex_name", "subunits_gene_name"]
                unique = ["complex_id", "complex_name"]
                self.test_all(col, unique)
                [self.test_empty(x for x in col)]
            except PCpexc.MissingColumnError as e:
                self.test_missing_col(["protein1", "protein2"])
        elif self.filetype == "in":
            col = ["gene_name", "protein_id"]
            unique = ["gene_name"]
            self.test_all(col, unique)
            self.test_na()

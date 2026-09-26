import pathlib
import re
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
UPSTREAM_BASE = "a327b494f874a319605e6fd7e3439678daa4d07d"


class CheckpointContractTests(unittest.TestCase):
    def native_diff(self) -> str:
        return subprocess.run(
            ["git", "diff", UPSTREAM_BASE, "--", "c", "runtime"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout

    def test_state_transfer_has_no_app_or_file_policy(self) -> None:
        additions = "\n".join(
            line[1:] for line in self.native_diff().splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        for forbidden in ("<cstdio>", "<unistd.h>", "<fcntl.h>", "FILE*",
                          "fopen(", "fsync(", "rename(", "unlink(", "PETAI_",
                          "identity", "SHA256"):
            self.assertNotIn(forbidden, additions)
        self.assertIn("litert_lm_session_transfer_state", additions)

    def test_native_changes_preserve_upstream_comments(self) -> None:
        for line in self.native_diff().splitlines():
            if line.startswith(("---", "+++")) or not line.startswith(("+", "-")):
                continue
            code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line[1:])
            self.assertNotIn("//", code, line)
            self.assertNotIn("/*", code, line)


if __name__ == "__main__":
    unittest.main()

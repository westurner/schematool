"""
schematool.tests.test_gen_ctags
"""
import os
import pytest
import shutil

from schematool.schematool import generate_ctags


def is_ctags_available():
    """Check if ctags is available in the path."""
    return shutil.which("ctags") is not None


@pytest.mark.skipif(not is_ctags_available(), reason="ctags not installed")
def test_generate_ctags_integration(tmp_path):
    """
    Test that generate_ctags actually runs ctags and creates a tags file.
    This is an integration test that requires ctags to be installed.
    """
    # Create a dummy python file
    dummy_py = tmp_path / "dummy_schema.py"
    dummy_py.write_text("class Test:\n    pass\n")

    # Change CWD to tmp_path so the 'tags' file is created there
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        generate_ctags(dummy_py)

        # Check if tags file was created
        tags_file = tmp_path / "tags"
        assert tags_file.exists()
        assert tags_file.stat().st_size > 0
        content = tags_file.read_text()
        assert "Test" in content
        assert "dummy_schema.py" in content
    finally:
        os.chdir(cwd)

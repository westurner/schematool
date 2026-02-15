import argparse
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from schematool.schematool import main, generate_python_constants


@pytest.fixture
def mock_args():
    args = argparse.Namespace()
    args.schema_inventory = Path("inventory.json")
    args.schema_dir = Path("schema")
    args.docs_vis_path = Path("docs.md")
    args.no_docs = True
    args.max_depth = 1
    args.target_formats = ["ttl"]
    args.use_pyoxigraph = False
    args.cache_path = Path("cache.sqlite")
    args.log_file = Path("schematool.log")
    args.gen_python = None
    return args


def test_generate_python_constants(tmp_path):
    # Setup
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir()

    # Create a simplified test TTL file
    ttl_content = """
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix my: <http://example.org/ontology/> .

    my:MyClass a owl:Class .
    my:myProperty a owl:ObjectProperty .
    my:otherProp a rdf:Property .
    """
    (schema_dir / "my.ttl").write_text(ttl_content)

    inventory = {
        "my": {"url": "http://example.org/ontology/", "sources": [{"path": "my.ttl"}]}
    }

    output_path = tmp_path / "schema.py"

    # Execute
    generate_python_constants(inventory, schema_dir, output_path)

    # Verify
    assert output_path.exists()
    content = output_path.read_text()

    assert "class MY:" in content
    assert 'MyClass = ox.NamedNode(_BASE + "MyClass")' in content
    assert 'myProperty = ox.NamedNode(_BASE + "myProperty")' in content
    assert 'otherProp = ox.NamedNode(_BASE + "otherProp")' in content
    assert "import pyoxigraph as ox" in content


def test_generate_python_constants_no_source(tmp_path, caplog):
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir()

    inventory = {
        "missing": {
            "url": "http://example.org/missing/",
            "sources": [{"path": "missing.ttl"}],  # File does not exist
        }
    }
    output_path = tmp_path / "schema.py"

    generate_python_constants(inventory, schema_dir, output_path)

    assert (
        output_path.exists()
    )  # Should still create the file, likely empty or just imports
    content = output_path.read_text()
    assert "class MISSING:" not in content
    assert "No Turtle source found for missing" in caplog.text


def test_generate_python_constants_invalid_ttl(tmp_path, caplog):
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir()

    (schema_dir / "bad.ttl").write_text("Invalid Turtle Content")

    inventory = {
        "bad": {"url": "http://example.org/bad/", "sources": [{"path": "bad.ttl"}]}
    }
    output_path = tmp_path / "schema.py"

    generate_python_constants(inventory, schema_dir, output_path)

    assert "Failed to process bad" in caplog.text


def test_generate_python_constants_nested_inventory(tmp_path):
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir()

    (schema_dir / "root.ttl").write_text(
        "@prefix : <http://root/> . :RootClass a <http://www.w3.org/2000/01/rdf-schema#Class> ."
    )
    (schema_dir / "dep.ttl").write_text(
        "@prefix : <http://dep/> . :DepClass a <http://www.w3.org/2000/01/rdf-schema#Class> ."
    )

    inventory = {
        "root": {
            "url": "http://root/",
            "sources": [{"path": "root.ttl"}],
            "dependencies": {
                "dep": {"url": "http://dep/", "sources": [{"path": "dep.ttl"}]}
            },
        }
    }

    output_path = tmp_path / "schema.py"
    generate_python_constants(inventory, schema_dir, output_path)

    content = output_path.read_text()
    assert "class ROOT:" in content
    assert "class DEP:" in content
    assert "RootClass =" in content
    assert "DepClass =" in content


def test_main_cli_gen_python(tmp_path, monkeypatch, mock_args):
    # Setup mocks
    mock_args.gen_python = tmp_path / "schema.py"
    mock_args.schema_inventory = tmp_path / "inventory.json"

    # Create rudimentary inventory
    inventory = {"test": {"url": "http://test", "sources": []}}
    with open(mock_args.schema_inventory, "w") as f:
        json.dump(inventory, f)

    mock_load_inventory = MagicMock(return_value=inventory)
    mock_generate = MagicMock()

    with (
        patch("schematool.schematool.load_inventory", mock_load_inventory),
        patch("schematool.schematool.generate_python_constants", mock_generate),
        patch("argparse.ArgumentParser.parse_args", return_value=mock_args),
    ):
        main()

        mock_generate.assert_called_once()

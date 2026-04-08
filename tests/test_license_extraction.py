"""
schematool.tests.test_license_extraction
"""
import pytest
from pathlib import Path
from schematool.schematool import generate_python_constants

def test_license_extraction(tmp_path):
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    ttl_path = schema_dir / "license_test.ttl"

    ttl_content = """
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix dcterms: <http://purl.org/dc/terms/> .
    @prefix cc: <http://creativecommons.org/ns#> .
    @prefix my: <http://example.org/lic/> .

    <http://example.org/lic/> a owl:Ontology ;
        rdfs:comment "Ontology with license." ;
        dcterms:license <http://creativecommons.org/licenses/by/4.0/> .

    my:Term a owl:Class .
    """
    ttl_path.write_text(ttl_content)

    inventory = {
        "my": {"url": "http://example.org/lic/", "sources": [{"path": "license_test.ttl"}]}
    }

    output_path = tmp_path / "output_lic.py"
    generate_python_constants(
        inventory, schema_dir, output_path, include_docstrings=True
    )

    content = output_path.read_text()

    assert "class MY:" in content
    assert "Ontology with license." in content
    assert "License: http://creativecommons.org/licenses/by/4.0/" in content

def test_multiple_licenses(tmp_path):
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    ttl_path = schema_dir / "license_multi.ttl"

    # Testing fallback or multiple licenses? Usually taking first found is fine or listing all.
    # Let's list all unique licenses found.
    ttl_content = """
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix dcterms: <http://purl.org/dc/terms/> .
    @prefix cc: <http://creativecommons.org/ns#> .
    @prefix my: <http://example.org/lic2/> .

    <http://example.org/lic2/> a owl:Ontology ;
        rdfs:comment "Ontology with multiple licenses." ;
        dcterms:license <http://license1.org> ;
        cc:license <http://license2.org> .
    """
    ttl_path.write_text(ttl_content)

    inventory = {
        "my": {"url": "http://example.org/lic2/", "sources": [{"path": "license_multi.ttl"}]}
    }

    output_path = tmp_path / "output_lic2.py"
    generate_python_constants(
        inventory, schema_dir, output_path, include_docstrings=True
    )

    content = output_path.read_text()

    # Check that at least one is present, or both if we implement aggregation
    assert "License: http://license1.org" in content or "License: http://license2.org" in content

"""
schematool.tests.test_docstrings_formatting
"""
import pytest
from pathlib import Path
from schematool.schematool import generate_python_constants


def test_docstrings_formatting(tmp_path):
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    ttl_path = schema_dir / "test.ttl"

    # Create test turtle file
    ttl_content = """
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix my: <http://example.org/> .

    my:Term1 a rdfs:Class ;
           rdfs:comment "A description." ;
           rdfs:label "Label 1", "Label 2" .

    my:Term2 a rdfs:Class ;
           rdfs:label "Just Label" .

    my:Term3 a rdfs:Class ;
           rdfs:comment "Just Comment" .

    my:Term4 a rdfs:Class ;
           rdfs:comment "Description." ;
           rdfs:label "Single Label" .
    """
    ttl_path.write_text(ttl_content)

    inventory = {
        "my": {"url": "http://example.org/", "sources": [{"path": "test.ttl"}]}
    }

    output_path = tmp_path / "output.py"
    generate_python_constants(
        inventory, schema_dir, output_path, include_docstrings=True
    )

    content = output_path.read_text()

    # Check Term1: Description + Multiple Labels (Code logic: "A description\nLabels: Label 1, Label 2")
    # Labels are sorted.
    assert "Term1: ox.NamedNode" in content
    # Updated expectation for newline join and block format
    assert '"""\n    A description.\n    - Labels: Label 1, Label 2\n    """' in content

    # Check Term2: Single Label (Code logic: "Label: Just Label")
    assert "Term2: ox.NamedNode" in content
    # Single line docstring remains single line if no newline
    assert '"""- Label: Just Label"""' in content

    # Check Term3: Just Comment (Code logic: "Just Comment")
    assert "Term3: ox.NamedNode" in content
    assert '"""Just Comment"""' in content

    # Check Term4: Description + Single Label (Code logic: "Description.\nLabel: Single Label")
    assert "Term4: ox.NamedNode" in content
    assert '"""\n    Description.\n    - Label: Single Label\n    """' in content


def test_ontology_docstring(tmp_path):
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    ttl_path = schema_dir / "ont_test.ttl"

    ttl_content = """
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix dcterms: <http://purl.org/dc/terms/> .
    @prefix my: <http://example.org/ont/> .

    <http://example.org/ont/> a owl:Ontology ;
        rdfs:comment "Comment on ontology." ;
        dcterms:description "Description of ontology." .

    my:Term a owl:Class .
    """
    ttl_path.write_text(ttl_content)

    inventory = {
        "my": {"url": "http://example.org/ont/", "sources": [{"path": "ont_test.ttl"}]}
    }

    output_path = tmp_path / "output_ont.py"
    generate_python_constants(
        inventory, schema_dir, output_path, include_docstrings=True
    )

    content = output_path.read_text()

    # Check if description mentions are in the docstring
    # The logic picks the longest description if multiple are found.
    # "Description of ontology." is longer than "Comment on ontology."
    assert "class MY:" in content
    assert "Description of ontology" in content
    assert "Comment on ontology" not in content  # Since we picked max


def test_extended_docstrings(tmp_path):
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    ttl_path = schema_dir / "extended.ttl"

    ttl_content = """
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix my: <http://example.org/ext/> .

    # Hierarchy Test
    my:GrandParent a owl:Class .
    my:Parent a owl:Class ;
        rdfs:subClassOf my:GrandParent .
    my:Child a owl:Class ;
        rdfs:subClassOf my:Parent ;
        rdfs:comment "A child class." .

    # Domain/Range Test
    my:hasValue a owl:ObjectProperty ;
        rdfs:domain my:Child ;
        rdfs:range rdfs:Literal ;
        rdfs:label "has value" .
    """
    ttl_path.write_text(ttl_content)

    inventory = {
        "my": {"url": "http://example.org/ext/", "sources": [{"path": "extended.ttl"}]}
    }

    output_path = tmp_path / "output_ext.py"
    generate_python_constants(
        inventory, schema_dir, output_path, include_docstrings=True
    )

    content = output_path.read_text()

    # Check Breadcrumbs
    # Hierarchy: GrandParent > Parent > Child
    assert "Child: ox.NamedNode" in content
    # We expect "| Hierarchy: GrandParent > Parent > Child"
    # Or "| Hierarchy: http://example.org/ext/GrandParent > http://example.org/ext/Parent > Child" if simplification fails
    # But get_local_name_or_uri handles base_uri "http://example.org/ext/" so it should be local names.
    assert "- Hierarchy: GrandParent > Parent > Child" in content
    assert "A child class." in content

    # Check Domain/Range
    assert "hasValue: ox.NamedNode" in content
    assert "- Domain: Child" in content
    # rdfs:Literal is standard, might not be simplified if not in base uri.
    # rdfs:Literal -> http://www.w3.org/2000/01/rdf-schema#Literal -> Literal (due to # split)
    assert "- Range: Literal" in content
    assert "- Label: has value" in content


def test_language_labels(tmp_path):
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    ttl_path = schema_dir / "lang.ttl"

    ttl_content = """
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix my: <http://example.org/lang/> .

    my:TermEn a owl:Class ;
        rdfs:label "English Label"@en .

    my:TermFr a owl:Class ;
        rdfs:label "French Label"@fr .

    my:TermMixed a owl:Class ;
        rdfs:label "Hello"@en, "Bonjour"@fr .
    """
    ttl_path.write_text(ttl_content)

    inventory = {
        "my": {"url": "http://example.org/lang/", "sources": [{"path": "lang.ttl"}]}
    }

    output_path = tmp_path / "output_lang.py"
    generate_python_constants(
        inventory, schema_dir, output_path, include_docstrings=True
    )

    content = output_path.read_text()

    assert "TermEn: ox.NamedNode" in content
    assert (
        "- Label: @en: English Label" in content
        or "- Labels: @en: English Label" in content
    )

    assert "TermFr: ox.NamedNode" in content
    assert "- Label: @fr: French Label" in content

    assert "TermMixed: ox.NamedNode" in content
    # Labels should be sorted? If so, @en comes before @fr alphabetically? Or sorted by value?
    # The code does: sorted(set(labels), key=labels.index) -> preservation of insertion order from triples?
    # rdflib triples might return in any order.
    # But later I do: unique_labels = sorted(set(labels), key=labels.index) which preserves order of first appearance.
    # Wait, rdflib doesn't guarantee order.
    # Let's just check both are present.
    assert "@en: Hello" in content
    assert "@fr: Bonjour" in content

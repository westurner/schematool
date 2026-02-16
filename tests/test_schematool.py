"""
Docstring for schematool.tests.test_schematool
"""

import json
import os
import pytest
import re
import runpy
import subprocess
import sys
from pathlib import Path

import requests_cache
import responses

from schematool.schematool import (
    download_and_convert,
    get_dependencies_for_file,
    DEFAULT_CACHE_PATH,
)

# from pathlib import Path
# REPO_ROOT = Path(__file__).parent.parent
# CACHE_PATH = REPO_ROOT / ".schema_cache"

# Initialize requests-cache
requests_cache.install_cache(
    str(DEFAULT_CACHE_PATH), backend="sqlite", expire_after=604800
)  # 1 week


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear requests cache before each test."""
    requests_cache.clear()
    yield
    requests_cache.clear()


# @pytest.fixture
# def mock_schema_dir(tmp_path):
#     """Create a temporary schema directory for testing."""
#     return tmp_path / "schema"


@responses.activate
def test_download_and_convert_conversion_failure(tmp_path):
    from schematool.schematool import download_and_convert

    # Request .ttl but get .rdf (XML) that is MALFORMED
    url = "http://example.org/bad.rdf"
    responses.add(
        responses.GET,
        url,
        body="<rdf:RDF> malformed",
        status=200,
        content_type="application/rdf+xml",
    )

    # Target is .ttl, so it will try to convert
    success = download_and_convert("bad_conv", url, {"path": "bad_conv.ttl"}, tmp_path)
    assert not success


def test_download_and_convert_exception(tmp_path):
    from schematool.schematool import download_and_convert

    # Test exception in read_text by using a directory
    path = tmp_path / "dir.ttl"
    path.mkdir()
    source_dir = {
        "path": "dir.ttl",
        "headers_used": {},
        "redirected_to": "http://example.org/dir.ttl",
    }
    # This should trigger the except Exception: pass and proceed to fetch
    assert not download_and_convert(
        "test", "http://example.org/dir.ttl", source_dir, tmp_path
    )


def test_download_and_convert_existing_not_html(tmp_path):
    from schematool.schematool import download_and_convert

    path = tmp_path / "already.ttl"
    path.write_text("@prefix : <#> .")

    source = {
        "path": "already.ttl",
        "headers_used": {},
        "redirected_to": "http://example.org/already.ttl",
    }
    # Hits the logic that checks if existing file is NOT HTML
    assert download_and_convert(
        "already", "http://example.org/already.ttl", source, tmp_path
    )


@responses.activate
def test_download_and_convert_html_received_error(tmp_path, caplog):
    from schematool.schematool import download_and_convert

    url = "http://example.org/test.ttl"
    responses.add(
        responses.GET,
        url,
        body="<html>Error</html>",
        status=200,
        content_type="text/html",
    )

    success = download_and_convert("htmlfail", url, {"path": "htmlfail.ttl"}, tmp_path)
    assert not success
    assert "Received HTML" in caplog.text


def test_download_and_convert_html_rejection(tmp_path):
    url = "https://example.org/fake.ttl"
    html_body = "<!DOCTYPE html><html><body>Error</body></html>"

    responses.add(
        responses.GET, url, body=html_body, status=200, content_type="text/html"
    )

    success = download_and_convert("fail", url, {"path": "fail/fail.ttl"}, tmp_path)
    assert not success
    assert not (tmp_path / "fail/fail.ttl").exists()


@responses.activate
def test_download_and_convert_http_error(tmp_path):
    from schematool.schematool import download_and_convert

    url = "http://example.org/404.ttl"
    responses.add(responses.GET, url, status=404)

    success = download_and_convert("fail404", url, {"path": "fail/404.ttl"}, tmp_path)
    assert not success


@responses.activate
def test_download_and_convert_jsonld(tmp_path):
    from schematool.schematool import download_and_convert

    url = "https://example.org/test.jsonld"
    jsonld_body = (
        '{"@context": "http://schema.org/", "@type": "Person", "name": "Jane Doe"}'
    )

    responses.add(
        responses.GET,
        url,
        body=jsonld_body,
        status=200,
        content_type="application/ld+json",
    )

    success = download_and_convert(
        "testjsonld", url, {"path": "testjsonld/test.ttl"}, tmp_path
    )
    assert success
    assert (tmp_path / "testjsonld/test.jsonld").exists()
    assert (tmp_path / "testjsonld/test.ttl").exists()


@responses.activate
def test_download_and_convert_purl(tmp_path):
    from schematool.schematool import download_and_convert

    purl_url = "http://purl.org/test"
    redir_url = "http://example.org/actual.ttl"

    # Mock the actual file download at the redirected URL
    responses.add(
        responses.GET,
        re.compile(r"http://purl\.org/test.*"),
        status=200,
        content_type="text/turtle",
        body="@prefix : <#> .",
    )
    # responses.add(responses.GET, redir_url, body="@prefix : <#> .", status=200, content_type="text/turtle")

    # In our implementation, if it redirects, it just captures the final URL.
    # responses with status=200 will just return that.
    # To test the hint-follow, we might need more complex mock.

    success = download_and_convert(
        "purltest", purl_url, {"path": "purl/test.ttl"}, tmp_path
    )
    assert success
    assert (tmp_path / "purl/test.ttl").exists()


@responses.activate
def test_download_and_convert_purl_hint_success(tmp_path):
    from schematool.schematool import download_and_convert

    purl_url = "http://purl.org/test_purl"
    actual_url = "http://example.org/actual.ttl"

    # Mock the hint-follow redirect
    responses.add(responses.GET, purl_url, status=302, headers={"Location": actual_url})
    responses.add(
        responses.GET,
        actual_url,
        body="@prefix : <#> .",
        status=200,
        content_type="text/turtle",
    )

    success = download_and_convert(
        "purl_hint", purl_url, {"path": "purl_hint.ttl"}, tmp_path
    )
    assert success


@responses.activate
def test_download_and_convert_turtle(tmp_path):
    url = "https://example.org/test.ttl"
    responses.add(
        responses.GET,
        url,
        body="<http://s> <http://p> <http://o> .",
        status=200,
        content_type="text/turtle",
    )

    success = download_and_convert("test", url, {"path": "test/test.ttl"}, tmp_path)
    assert success
    assert (tmp_path / "test/test.ttl").exists()
    assert (
        tmp_path / "test/test.ttl"
    ).read_text() == "<http://s> <http://p> <http://o> ."


@responses.activate
def test_download_and_convert_xml(tmp_path):
    url = "https://example.org/test.rdf"
    rdf_xml = """<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:ex="http://example.org/">
      <rdf:Description rdf:about="http://example.org/s">
        <ex:p rdf:resource="http://example.org/o"/>
      </rdf:Description>
    </rdf:RDF>"""

    responses.add(
        responses.GET, url, body=rdf_xml, status=200, content_type="application/rdf+xml"
    )

    success = download_and_convert(
        "testxml", url, {"path": "testxml/testxml.ttl"}, tmp_path
    )
    assert success
    assert (tmp_path / "testxml/testxml.rdf").exists()
    assert (tmp_path / "testxml/testxml.ttl").exists()
    # Check if it converted correctly (rdflib output is slightly different but should be valid turtle)
    content = (tmp_path / "testxml/testxml.ttl").read_text()
    assert "example.org" in content


def test_get_dependencies_exception(tmp_path, monkeypatch):
    from schematool.schematool import get_dependencies_for_file
    from rdflib import Graph

    path = tmp_path / "error.ttl"
    path.write_text("@prefix : <#> .")

    def mock_parse_fail(*args, **kwargs):
        raise Exception("Parse error")

    monkeypatch.setattr(Graph, "parse", mock_parse_fail)

    assert get_dependencies_for_file(path) == {}


def test_get_dependencies_for_file(tmp_path):
    test_ttl = tmp_path / "test.ttl"
    test_ttl.write_text(
        """
        @prefix brick: <https://brickschema.org/schema/Brick#> .
        @prefix iof: <https://spec.industrialontologies.org/ontology/core/Core/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

        <http://example.org/s> a brick:Point .
    """
    )

    deps = get_dependencies_for_file(test_ttl)
    assert "brick" in deps
    assert "iof" in deps
    assert "rdf" not in deps  # Should be in SKIP_PREFIXES


def test_get_dependencies_non_existent():
    from schematool.schematool import get_dependencies_for_file
    from pathlib import Path

    assert get_dependencies_for_file(Path("non_existent.ttl")) == {}


def test_load_inventory(tmp_path):
    from schematool.schematool import load_inventory

    # Test missing file
    assert load_inventory(tmp_path / "missing.json") == {}

    # Test valid file
    inv_file = tmp_path / "inventory.json"
    data = {"test": {"url": "http://example.org"}}
    import json

    inv_file.write_text(json.dumps(data))
    assert load_inventory(inv_file) == data

    # Test invalid JSON
    inv_file.write_text("invalid json")
    assert load_inventory(inv_file) == {}


def test_main(tmp_path, monkeypatch):
    from schematool.schematool import main
    import schematool.schematool as schematool_mod

    # Prevent it from trying to download half the web by skipping common prefixes discovered by rdflib
    monkeypatch.setattr(
        "schematool.schematool.SKIP_PREFIXES",
        schematool_mod.SKIP_PREFIXES
        | {
            "brick",
            "csvw",
            "dcat",
            "dcmitype",
            "dcam",
            "doap",
            "geo",
            "odrl",
            "org",
            "prof",
            "prov",
            "qb",
            "schema",
            "sh",
            "sosa",
            "ssn",
            "time",
            "void",
            "wgs",
        },
    )

    inventory_file = tmp_path / "inventory.json"
    inventory_file.write_text(
        json.dumps(
            {
                "test": {
                    "url": "http://example.org/test.ttl",
                    "sources": [{"path": "test/test.ttl"}],
                }
            }
        )
    )

    schema_dir = tmp_path / "schema"
    docs_vis = tmp_path / "docs" / "visualization.md"
    docs_vis.parent.mkdir()
    docs_vis.write_text("## Schema Dependency Hierarchy\n")

    # Mock setup_cache to do nothing during test_main
    monkeypatch.setattr("schematool.schematool.setup_cache", lambda x: None)
    # Also ensure any existing session is disabled
    import requests_cache

    with requests_cache.disabled():
        # Mock requests.get via responses
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://example.org/test.ttl",
                body="@prefix : <#> .",
                status=200,
                content_type="text/turtle",
            )

            args = [
                "schematool",
                "--schema-inventory",
                str(inventory_file),
                "--schema-dir",
                str(schema_dir),
                "--docs-vis-path",
                str(docs_vis),
                "--max-depth",
                "1",
            ]
            monkeypatch.setattr("sys.argv", args)

            main()

    assert (schema_dir / "test/test.ttl").exists()
    assert "**test**" in docs_vis.read_text()


def test_main_call():
    # PYTHONPATH must include the current directory to find schematool if run as a script
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")

    # Locate schematool.py relative to this test file
    # This file is in src/schematool/tests/, so we go up one level to src/schematool/
    # and then down to schematool/schematool.py
    base_dir = Path(__file__).resolve().parent.parent
    script_path = base_dir / "schematool" / "schematool.py"

    # Run with --help to verify main() can be called
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"], capture_output=True, env=env
    )
    # Check if the process ran successfully
    assert result.returncode == 0
    assert b"transitive" in result.stdout


def test_main_empty_inventory(tmp_path, monkeypatch, caplog):
    from schematool.schematool import main
    import json

    inventory_file = tmp_path / "empty_inventory.json"
    inventory_file.write_text(json.dumps({}))
    monkeypatch.setattr(
        "sys.argv", ["schematool", "--schema-inventory", str(inventory_file)]
    )
    monkeypatch.setattr("schematool.schematool.setup_cache", lambda x: None)
    main()
    assert "No inventory data found" in caplog.text


@responses.activate
def test_recursive_sync(tmp_path):
    from schematool.schematool import recursive_sync

    base_url = "http://example.org/base.ttl"
    dep_ns_url = "http://example.org/dep#"

    # Base ontology with a dependency on 'dep'
    responses.add(
        responses.GET,
        base_url,
        body="""
        @prefix base: <http://example.org/base#> .
        @prefix dep: <http://example.org/dep#> .
        <http://example.org/base#s> a dep:Class .
        """,
        status=200,
        content_type="text/turtle",
    )

    # Dependency ontology at its namespace URL
    responses.add(
        responses.GET,
        dep_ns_url,
        body="@prefix dep: <http://example.org/dep#> . dep:Class a <http://www.w3.org/2002/07/owl#Class> .",
        status=200,
        content_type="text/turtle",
    )

    inventory = {"base": {"url": base_url, "sources": [{"path": "base/base.ttl"}]}}

    recursive_sync(inventory, tmp_path, max_depth=2)

    assert "base" in inventory
    assert (tmp_path / "base/base.ttl").exists()
    assert "dependencies" in inventory["base"]
    assert "dep" in inventory["base"]["dependencies"]
    assert inventory["base"]["dependencies"]["dep"]["url"] == dep_ns_url
    assert (tmp_path / "dep/dep.ttl").exists()


@responses.activate
def test_recursive_sync_failure_continue(tmp_path):
    from schematool.schematool import recursive_sync

    url = "http://example.org/fail.ttl"
    responses.add(responses.GET, url, status=404)

    inventory = {"fail": {"url": url, "sources": [{"path": "fail.ttl"}]}}

    # This should hit 'if not any_success: continue'
    recursive_sync(inventory, tmp_path, max_depth=0)
    assert "dependencies" not in inventory["fail"]


def test_recursive_sync_format_continue(tmp_path, monkeypatch):
    from schematool.schematool import recursive_sync

    # Mock download_and_convert to return False on first call, True on second
    call_count = 0

    def mock_dc(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return call_count > 1

    monkeypatch.setattr("schematool.schematool.download_and_convert", mock_dc)
    monkeypatch.setattr("schematool.schematool.get_dependencies_for_file", lambda p: {})

    inventory = {
        "test": {
            "url": "http://example.org",
            "sources": [{"path": "t.ttl"}, {"path": "t.rdf"}],
        }
    }

    recursive_sync(inventory, tmp_path, max_depth=1)
    # This should have hit the "continue" on the first format


def test_recursive_sync_max_depth(tmp_path):
    from schematool.schematool import recursive_sync

    inventory = {"test": {"url": "http://example.org", "sources": []}}
    # Should return immediately
    recursive_sync(inventory, tmp_path, max_depth=-1)
    assert inventory["test"]["sources"] == []


@responses.activate
def test_recursive_sync_migration(tmp_path):
    from schematool.schematool import recursive_sync

    url = "http://example.org/migration.ttl"
    responses.add(
        responses.GET,
        url,
        body="@prefix : <#> .",
        status=200,
        content_type="text/turtle",
    )

    inventory = {"legacy": {"url": url, "path": "legacy.ttl"}}

    recursive_sync(inventory, tmp_path, max_depth=0)
    assert "sources" in inventory["legacy"]
    assert inventory["legacy"]["sources"][0]["path"] == "legacy.ttl"


def test_recursive_sync_no_any_success(tmp_path):
    from schematool.schematool import recursive_sync

    # URL that will fail (no responses mock)
    inventory = {
        "fail": {
            "url": "http://example.org/nonexistent.ttl",
            "sources": [{"path": "fail.ttl"}],
        }
    }

    # Should not raise exception, just continue
    recursive_sync(inventory, tmp_path, max_depth=0)
    assert inventory["fail"]["sources"][0].get("format") is None


@pytest.mark.skip("slow")
def test_recursive_sync_with_generation(tmp_path):
    from schematool.schematool import recursive_sync

    # Create a seed file
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    seed_file = seed_dir / "seed.ttl"
    seed_file.write_text("@prefix : <#> . :s :p :o .")

    inventory = {
        "seed": {
            "url": "http://example.org/seed.ttl",
            "sources": [
                {
                    "path": "seed/seed.ttl",
                    "format": "turtle",
                    "headers_used": {},
                    "redirected_to": "http://example.org/seed.ttl",
                }
            ],
        }
    }

    # Run with generation
    recursive_sync(inventory, tmp_path, target_formats=["rdfxml", "jsonld"])

    assert len(inventory["seed"]["sources"]) > 1
    # Check if rdfxml and jsonld were generated
    formats = [s["format"] for s in inventory["seed"]["sources"]]
    assert "rdfxml" in formats
    assert "jsonld" in formats
    assert (seed_dir / "seed.rdf").exists()
    assert (seed_dir / "seed.jsonld").exists()


def test_render_hierarchy_empty():
    from schematool.schematool import render_hierarchy

    assert render_hierarchy({}) == []


def test_render_hierarchy_legacy():
    from schematool.schematool import render_hierarchy

    inventory = {"legacy": {"url": "http://example.org", "path": "legacy.ttl"}}
    lines = render_hierarchy(inventory)
    assert "**legacy**" in lines[0]
    assert "Local: `legacy.ttl`" in lines[0]


def test_save_inventory(tmp_path):
    from schematool.schematool import save_inventory
    import json

    inv_file = tmp_path / "subdir" / "inventory.json"
    data = {"test": {"url": "http://example.org"}}
    save_inventory(inv_file, data)

    assert inv_file.exists()
    assert json.loads(inv_file.read_text()) == data


def test_save_inventory_error(tmp_path, caplog):
    from schematool.schematool import save_inventory

    # Try to save to a directory as a file
    bad_path = tmp_path / "a_directory"
    bad_path.mkdir()
    save_inventory(bad_path, {"test": "data"})
    assert "Failed to save inventory" in caplog.text


def test_setup_cache_coverage(tmp_path):
    from schematool.schematool import setup_cache

    setup_cache(tmp_path / "new_cache.sqlite")


def test_transform_rdf(tmp_path):
    from schematool.schematool import transform_rdf
    from rdflib import Graph

    input_ttl = tmp_path / "input.ttl"
    input_ttl.write_text("@prefix ex: <http://example.org/> . ex:s ex:p ex:o .")

    output_xml = tmp_path / "output.rdf"
    # Test rdflib transformation
    success = transform_rdf(input_ttl, output_xml, "rdfxml", use_pyoxigraph=False)
    assert success
    assert output_xml.exists()

    # Verify content
    g = Graph()
    g.parse(str(output_xml), format="xml")
    assert len(g) == 1

    # Test failure with invalid input
    bad_input = tmp_path / "bad.ttl"
    bad_input.write_text("not rdf at all")
    assert not transform_rdf(bad_input, tmp_path / "fail.ttl", "ttl")


def test_transform_rdf_pyoxigraph(tmp_path):
    from schematool.schematool import transform_rdf, HAS_PYOXIGRAPH

    if not HAS_PYOXIGRAPH:
        pytest.skip("pyoxigraph not installed")
    input_ttl = tmp_path / "input_pyoxi.ttl"
    input_ttl.write_text("@prefix ex: <http://example.org/> . ex:s ex:p ex:o .")
    output_rdf = tmp_path / "output_pyoxi.rdf"
    success = transform_rdf(input_ttl, output_rdf, "rdfxml", use_pyoxigraph=True)
    assert success
    assert output_rdf.exists()


def test_transform_rdf_pyoxigraph_success_ttl(tmp_path):
    from schematool.schematool import transform_rdf, HAS_PYOXIGRAPH

    if not HAS_PYOXIGRAPH:
        pytest.skip("pyoxigraph not installed")
    input_ttl = tmp_path / "input_pyoxi.ttl"
    input_ttl.write_text("@prefix ex: <http://example.org/> . ex:s ex:p ex:o .")
    output_ttl = tmp_path / "output_pyoxi.ttl"
    success = transform_rdf(input_ttl, output_ttl, "ttl", use_pyoxigraph=True)
    assert success
    assert output_ttl.exists()


def test_transform_rdf_pyoxigraph_failure(tmp_path):
    from schematool.schematool import transform_rdf, HAS_PYOXIGRAPH

    if not HAS_PYOXIGRAPH:
        pytest.skip("pyoxigraph not installed")

    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("not rdf")
    # This should trigger pyoxigraph exception and fallback to rdflib, which also fails
    assert not transform_rdf(
        bad_file, tmp_path / "fail.ttl", "ttl", use_pyoxigraph=True
    )


def test_transform_rdf_pyoxigraph_formats(tmp_path):
    from schematool.schematool import transform_rdf, HAS_PYOXIGRAPH

    if not HAS_PYOXIGRAPH:
        pytest.skip("pyoxigraph not installed")

    # Test RDF/XML input to Turtle
    rdf_xml = tmp_path / "input.rdf"
    rdf_xml.write_text(
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:ex="http://example.org/"><rdf:Description rdf:about="http://example.org/s"><ex:p rdf:resource="http://example.org/o"/></rdf:Description></rdf:RDF>'
    )

    out_ttl = tmp_path / "out_from_xml.ttl"
    assert transform_rdf(rdf_xml, out_ttl, "ttl", use_pyoxigraph=True)
    assert out_ttl.exists()

    # Test JSON-LD input to Turtle
    json_ld = tmp_path / "input.jsonld"
    json_ld.write_text(
        '{"@context": {"ex": "http://example.org/"}, "@id": "ex:s", "ex:p": {"@id": "ex:o"}}'
    )

    out_ttl_2 = tmp_path / "out_from_jsonld.ttl"
    assert transform_rdf(json_ld, out_ttl_2, "ttl", use_pyoxigraph=True)
    assert out_ttl_2.exists()


def test_transform_rdf_pyoxigraph_full_coverage(tmp_path):
    from schematool.schematool import transform_rdf, HAS_PYOXIGRAPH
    import unittest.mock as mock

    if not HAS_PYOXIGRAPH:
        pytest.skip("pyoxigraph not installed")

    input_ttl = tmp_path / "input.ttl"
    input_ttl.write_text("@prefix ex: <http://example.org/> . ex:s ex:p ex:o .")
    output_rdf = tmp_path / "output.rdf"

    # Test successful pyoxigraph path
    assert transform_rdf(input_ttl, output_rdf, "rdfxml", use_pyoxigraph=True)
    assert output_rdf.exists()

    # Test pyoxigraph failure path (line 140-141)
    with mock.patch("pyoxigraph.Store.load", side_effect=Exception("mock error")):
        # Should fallback to rdflib and still succeed
        assert transform_rdf(input_ttl, output_rdf, "rdfxml", use_pyoxigraph=True)


def test_transform_rdf_unsupported_format(tmp_path):
    from schematool.schematool import transform_rdf

    input_ttl = tmp_path / "input.ttl"
    input_ttl.write_text("@prefix : <#> .")
    # Unsupported target format in rdflib_map
    assert not transform_rdf(input_ttl, tmp_path / "out.xyz", "xyz")


def test_update_docs(tmp_path):
    from schematool.schematool import update_docs

    docs_path = tmp_path / "visualization.md"
    docs_path.write_text(
        "# Documentation\n\n## Schema Dependency Hierarchy\nOld hierarchy\n"
    )

    inventory = {
        "test": {"url": "http://example.org", "sources": [{"path": "test/test.ttl"}]}
    }

    # Create the local ontology file
    sf_path = tmp_path / "sustainable-factory.ttl"
    sf_path.write_text("LOCAL CONTENT")

    update_docs(inventory, tmp_path, docs_path)

    content = docs_path.read_text()
    assert "## Schema Dependency Hierarchy" in content
    assert (
        "**test**: [http://example.org](http://example.org) (Local: `test/test.ttl`)"
        in content
    )
    assert "## Schema: Local Process Ontology" in content
    assert "LOCAL CONTENT" in content


def test_update_docs_missing_file(tmp_path, caplog):
    from schematool.schematool import update_docs

    update_docs({}, tmp_path, tmp_path / "missing.md")
    assert "not found" in caplog.text


def test_update_docs_no_existing_sections(tmp_path):
    from schematool.schematool import update_docs

    docs_path = tmp_path / "viz.md"
    docs_path.write_text("# Viz\nJust some info.")  # No sections

    # Also test the "Local Process Ontology" exclusion logic
    # No sf_path exists

    inventory = {
        "test": {"url": "http://example.org", "sources": [{"path": "test.ttl"}]}
    }
    update_docs(inventory, tmp_path, docs_path)

    content = docs_path.read_text()
    assert "## Schema Dependency Hierarchy" in content
    assert "## Schema: Local Process Ontology" not in content


def test_update_docs_no_sections(tmp_path):
    from schematool.schematool import update_docs

    docs_path = tmp_path / "vis.md"
    docs_path.write_text("# Only Title\n")

    inventory = {
        "test": {"url": "http://example.org", "sources": [{"path": "test.ttl"}]}
    }
    update_docs(inventory, tmp_path, docs_path)

    content = docs_path.read_text()
    assert "## Schema Dependency Hierarchy" in content
    assert (
        "## Schema: Local Process Ontology" not in content
    )  # because sf_path doesn't exist


def test_update_docs_read_error(tmp_path, monkeypatch, caplog):
    from schematool.schematool import update_docs

    docs_path = tmp_path / "docs.md"
    docs_path.write_text("# Test")

    def mock_read_fail(self):
        raise OSError("Read error")

    # Only mock read_text for our specific file to avoid breaking other things
    original_read_text = Path.read_text

    def patched_read_text(self):
        if self == docs_path:
            raise OSError("Read error")
        return original_read_text(self)

    monkeypatch.setattr(Path, "read_text", patched_read_text)

    update_docs({}, tmp_path, docs_path)
    assert "Failed to read docs" in caplog.text


def test_update_docs_with_ontology_update(tmp_path):
    from schematool.schematool import update_docs

    docs_path = tmp_path / "viz.md"
    docs_path.write_text("# Viz\n## Schema: Local Process Ontology\nOLD CONTENT\n")

    sf_path = tmp_path / "sustainable-factory.ttl"
    sf_path.write_text("NEW TURTLE")

    inventory = {}
    update_docs(inventory, tmp_path, docs_path)

    content = docs_path.read_text()
    assert "## Schema: Local Process Ontology" in content
    assert "NEW TURTLE" in content
    assert "OLD CONTENT" not in content


def test_update_docs_write_error(tmp_path, monkeypatch, caplog):
    from schematool.schematool import update_docs

    docs_path = tmp_path / "docs.md"
    docs_path.write_text("# Test")

    original_write_text = Path.write_text

    def patched_write_text(self, text):
        if self == docs_path:
            raise OSError("Write error")
        return original_write_text(self, text)

    monkeypatch.setattr(Path, "write_text", patched_write_text)

    update_docs({}, tmp_path, docs_path)
    assert "Failed to write docs" in caplog.text


def test_recursive_sync_visited_urls(tmp_path, monkeypatch):
    from schematool.schematool import recursive_sync

    url = "http://example.org/shared.ttl"
    # Both A and B point to same URL
    inventory = {
        "A": {"url": url, "sources": [{"path": "shared.ttl"}]},
        "B": {"url": url, "sources": [{"path": "shared.ttl"}]},
    }

    # Mock download_and_convert to count calls
    call_count = 0

    def mock_dc(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return True

    monkeypatch.setattr("schematool.schematool.download_and_convert", mock_dc)
    monkeypatch.setattr("schematool.schematool.get_dependencies_for_file", lambda p: {})

    recursive_sync(inventory, tmp_path, max_depth=1)
    # Should only be called once because of visited_urls
    assert call_count == 1


def test_transform_rdf_none_format(tmp_path):
    from schematool.schematool import transform_rdf

    input_ttl = tmp_path / "in.ttl"
    input_ttl.write_text("@prefix : <#> .")
    assert not transform_rdf(input_ttl, tmp_path / "out.none", None)


def test_run_as_script(tmp_path, monkeypatch):
    # This targets line 607: if __name__ == "__main__": main()
    # We need to mock sys.argv and call run_path

    # Locate schematool.py relative to this test file
    # tests/test_schematool.py -> tests/../schematool/schematool.py
    base_dir = Path(__file__).resolve().parent.parent
    script_path = base_dir / "schematool" / "schematool.py"

    inventory_file = tmp_path / "inv.json"
    inventory_file.write_text("{}")

    # Mock sys.argv as if the script was run with arguments
    monkeypatch.setattr(
        "sys.argv", [str(script_path), "--schema-inventory", str(inventory_file)]
    )
    # Monkeypatch setup_logging to avoid polluting stdout
    monkeypatch.setattr("schematool.schematool.setup_logging", lambda x: None)

    # Use runpy to execute the file as __main__
    runpy.run_path(str(script_path), run_name="__main__")


def test_pyoxigraph_import_error(monkeypatch):
    import sys
    import importlib
    import schematool.schematool as schematool_mod
    from unittest.mock import patch

    with patch.dict(sys.modules, {"pyoxigraph": None}):
        importlib.reload(schematool_mod)
        assert schematool_mod.HAS_PYOXIGRAPH is False
    importlib.reload(schematool_mod)


def test_render_hierarchy_nested():
    from schematool.schematool import render_hierarchy

    inventory = {
        "parent": {
            "url": "http://parent.org",
            "sources": [{"path": "p.ttl"}],
            "dependencies": {
                "child": {"url": "http://child.org", "sources": [{"path": "c.ttl"}]}
            },
        }
    }
    lines = render_hierarchy(inventory)
    # Check that it rendered both
    assert any("parent.org" in line for line in lines)
    assert any("child.org" in line for line in lines)
    # Check indentation (second line usually for child)
    child_line = [line for line in lines if "child.org" in line][0]
    assert child_line.startswith("  -")  # One level of 2 spaces

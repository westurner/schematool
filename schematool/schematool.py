"""
schematool
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import requests
import requests_cache
from rdflib import Graph, RDF, RDFS, OWL

try:
    import pyoxigraph

    HAS_PYOXIGRAPH = True
except ImportError:
    HAS_PYOXIGRAPH = False

# Default Configuration
DEFAULT_SCHEMA_DIR = Path("schema")
DEFAULT_DOCS_VIS_PATH = Path("docs/visualization.md")
DEFAULT_LOG_FILE = Path("schematool.log")
DEFAULT_CACHE_PATH = Path(".schema_cache.sqlite")
DEFAULT_INVENTORY_PATH = Path("schema_inventory.json")


# setup_cache and setup_logging will be called from main()
def setup_cache(cache_path):
    """Initialize requests-cache with the specified path."""
    requests_cache.install_cache(str(cache_path), backend="sqlite", expire_after=604800)


# Setup Logging
class ColorFormatter(logging.Formatter):
    RED = "\033[91m"
    RESET = "\033[0m"

    def format(self, record):
        if record.levelno >= logging.ERROR:
            record.msg = f"{self.RED}{record.msg}{self.RESET}"
        return super().format(record)


logger = logging.getLogger("schematool")
logger.setLevel(logging.DEBUG)


def setup_logging(log_file):
    """Configure logging handlers."""
    # File Handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s"))
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter("%(asctime)s\t%(levelname)s\t%(message)s"))
    logger.addHandler(ch)


# Built-in or very common that we don't necessarily want to download recursively
SKIP_PREFIXES = {
    "rdf",
    "rdfs",
    "owl",
    "xsd",
    "xml",
    "dc",
    "dcterms",
    "foaf",
    "skos",
    "vann",
}


def load_inventory(path):
    """Load the schema inventory from a JSON file."""
    if not path.exists():
        logger.warning(
            f"Inventory file {path} not found. Starting with empty inventory."
        )
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load inventory from {path}: {e}")
        return {}


def save_inventory(path, inventory):
    """Save the schema inventory to a JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(inventory, f, indent=4)
        logger.info(f"Saved updated inventory to {path}")
    except Exception as e:
        logger.error(f"Failed to save inventory to {path}: {e}")


def transform_rdf(input_path, output_path, target_format, use_pyoxigraph=False):
    """Transform RDF from one format to another."""
    rdflib_map = {"ttl": "turtle", "rdfxml": "xml", "jsonld": "json-ld"}

    if use_pyoxigraph and HAS_PYOXIGRAPH:
        pyoxi_map = {
            "ttl": pyoxigraph.RdfFormat.TURTLE,
            "rdfxml": pyoxigraph.RdfFormat.RDF_XML,
            "jsonld": pyoxigraph.RdfFormat.JSON_LD,
        }
        try:
            fmt = pyoxi_map.get(target_format)
            if fmt:
                # Guess input format
                input_ext = input_path.suffix.lower()
                input_fmt = None
                if input_ext == ".ttl":
                    input_fmt = pyoxigraph.RdfFormat.TURTLE
                elif input_ext in (".rdf", ".owl", ".xml"):
                    input_fmt = pyoxigraph.RdfFormat.RDF_XML
                elif input_ext == ".jsonld":
                    input_fmt = pyoxigraph.RdfFormat.JSON_LD

                if input_fmt:
                    store = pyoxigraph.Store()
                    with open(input_path, "rb") as f:
                        store.load(f, input_fmt)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        pyoxigraph.serialize(store, f, format=fmt)
                    return True
        except Exception as e:
            logger.debug(
                f"Pyoxigraph transformation failed, falling back to rdflib: {e}"
            )

    try:
        g = Graph()
        g.parse(str(input_path))
        output_format = rdflib_map.get(target_format, target_format)
        if output_format:
            g.serialize(destination=str(output_path), format=output_format)
            return True
        return False
    except Exception as e:
        logger.error(f"  [!!] Transformation failed: {e}")
        return False


def download_and_convert(prefix, base_url, source, schema_dir, reason="User requested"):
    """
    Download a single source and convert it to Turtle if needed.
    'source' is a dict that MUST contain 'path' and may contain 'url', 'req_headers', etc.
    It will be updated in-place with 'headers_used', 'redirected_to', and 'format'.
    """
    local_rel_path = source["path"]
    target_path = schema_dir / local_rel_path

    # Use source-specific URL if provided, otherwise fallback to base_url
    url = source.get("url", base_url)

    missing_info = "headers_used" not in source or "redirected_to" not in source

    if target_path.exists() and target_path.stat().st_size > 0 and not missing_info:
        # Check if existing file is HTML
        try:
            existing = target_path.read_text()[:500].lower()
            if "<!doctype html" not in existing and "<html" not in existing:
                return True
        except Exception:
            pass

    logger.info(f"  [..] Fetching {url} (Reason: {reason})")
    try:
        # Use source-specific headers if provided, otherwise standard RDF negotiation
        headers = source.get("req_headers")
        if not headers:
            headers = {
                "Accept": "text/turtle, application/rdf+xml;q=0.9, application/ld+json;q=0.8, application/xml;q=0.7, text/plain;q=0.6, */*;q=0.1"
            }

        target_url = url
        # PURL handling: some PURL services redirect to a documentation page or a landing URL.
        # We manually follow redirects for PURLs to reach the final service endpoint
        # and then apply the Accept headers there to ensure we get RDF instead of HTML.
        # Note: If custom headers are provided, we skip the hint-follow as user might be targeting a direct URL.
        if (
            "purl.org" in url or "purl.obolibrary.org" in url
        ) and "req_headers" not in source:
            try:
                # Follow the PURL redirect without headers to find the actual service location
                r_redir = requests.get(
                    url, allow_redirects=True, timeout=5, stream=True
                )
                if r_redir.url != url:
                    logger.debug(f"  [..] PURL redirect: {url} -> {r_redir.url}")
                    target_url = r_redir.url
                r_redir.close()
            except Exception as e:
                logger.debug(f"  [..] PURL hint-follow failed: {e}")

        resp = requests.get(
            target_url, headers=headers, timeout=10, allow_redirects=True
        )
        from_cache = getattr(resp, "from_cache", False)
        cache_msg = " [CACHE]" if from_cache else ""

        # Capture metadata for inventory
        source["headers_used"] = headers
        source["redirected_to"] = resp.url

        if resp.status_code != 200:
            logger.error(f"  [!!] Failed: HTTP {resp.status_code}")
            return False

        content_type = resp.headers.get("Content-Type", "").lower()
        text = resp.text
        search_text = text[:1000].lower()
        if "<!doctype html" in search_text or "<html" in search_text:
            logger.error(
                f"  [!!] Received HTML {content_type!r} instead of RDF for {url}"
            )
            return False

        # Determine format
        rdf_format = None
        if (
            "turtle" in content_type
            or text.startswith("@prefix")
            or "\n@prefix" in text[:1000]
        ):
            rdf_format = "turtle"
        elif (
            "rdf+xml" in content_type
            or "application/xml" in content_type
            or "rdf:RDF" in text
            or "<?xml" in text
        ):
            rdf_format = "xml"
        elif (
            "json-ld" in content_type
            or "application/ld+json" in content_type
            or (text.strip().startswith("{") and '"@context"' in text)
        ):
            rdf_format = "json-ld"

        source["format"] = rdf_format or "unknown"

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if rdf_format and rdf_format != "turtle" and target_path.suffix == ".ttl":
            # Convert to Turtle if the target extension is .ttl
            logger.info(f"  [..] Converting {rdf_format} to Turtle...{cache_msg}")
            # Save raw version first
            raw_ext = ".rdf" if rdf_format == "xml" else ".jsonld"
            target_path.with_suffix(raw_ext).write_text(text)

            g = Graph()
            try:
                g.parse(data=text, format=rdf_format)
                g.serialize(destination=str(target_path), format="turtle")
            except Exception as conv_err:
                logger.error(f"  [!!] Conversion failed: {conv_err}")
                return False
        else:
            # Just save it as-is
            target_path.write_text(text)

        logger.info(f"  [OK]{cache_msg} Saved {prefix} to {local_rel_path}")
        return True
    except Exception as e:
        logger.error(f"  [!!] Error: {e}")
        return False


def get_dependencies_for_file(file_path):
    """Scan a Turtle file for prefixes that might be dependencies."""
    deps = {}
    if not file_path.exists():
        return deps
    try:
        g = Graph()
        # Suppress rdflib warnings during dependency scanning
        logging.getLogger("rdflib").setLevel(logging.ERROR)
        g.parse(str(file_path), format="turtle")
        for prefix, ns in g.namespaces():
            if prefix and prefix not in SKIP_PREFIXES:
                deps[prefix] = str(ns)
    except Exception:
        pass
    finally:
        logging.getLogger("rdflib").setLevel(logging.WARNING)
    return deps


def recursive_sync(
    inventory_node,
    schema_dir,
    depth=0,
    max_depth=3,
    visited_urls=None,
    parent_prefix=None,
    **kwargs,
):
    """Recursively download schemas and discover their dependencies."""
    if depth > max_depth:
        return
    if visited_urls is None:
        visited_urls = set()

    target_formats = kwargs.get("target_formats", ["ttl", "rdfxml", "jsonld"])
    use_pyoxigraph = kwargs.get("use_pyoxigraph", False)
    fmt_to_ext = {"ttl": ".ttl", "rdfxml": ".rdf", "jsonld": ".jsonld"}

    for prefix, info in list(inventory_node.items()):
        url = info["url"]

        # New structure: support multiple sources
        if "sources" not in info:
            # Migration: convert single path to sources list
            info["sources"] = [
                {
                    "path": info.get("path"),
                    "headers_used": info.pop("headers_used", None),
                    "redirected_to": info.pop("redirected_to", None),
                    "format": info.pop("format", None),
                }
            ]
            # Cleanup old path if it was there
            if "path" in info:
                del info["path"]

        if url in visited_urls:
            continue
        visited_urls.add(url)

        reason = (
            f"Root ontology: {prefix}"
            if depth == 0
            else f"Dependency of {parent_prefix}"
        )

        any_success = False
        for source in info["sources"]:
            success = download_and_convert(
                prefix, url, source, schema_dir, reason=reason
            )
            if success:
                any_success = True

        if not any_success:
            continue

        # Ensure all target formats are present via local transformation if official source is missing
        existing_exts = {
            Path(s["path"]).suffix for s in info["sources"] if s.get("path")
        }

        # Find a good source to use as a seed for conversion
        seed_source = None
        for s in info["sources"]:
            if s.get("path"):
                p = schema_dir / s["path"]
                if p.exists() and p.stat().st_size > 0:
                    seed_source = s
                    break

        if seed_source:
            for fmt in target_formats:
                ext = fmt_to_ext.get(fmt)
                if ext and ext not in existing_exts:
                    # Calculate new path based on prefix or seed path
                    seed_path = Path(seed_source["path"])
                    new_path = seed_path.with_name(f"{prefix}{ext}")

                    logger.info(
                        f"  [..] Generating {fmt} for {prefix} from {seed_path.suffix}..."
                    )
                    if transform_rdf(
                        schema_dir / seed_path,
                        schema_dir / new_path,
                        fmt,
                        use_pyoxigraph,
                    ):
                        info["sources"].append(
                            {"path": str(new_path), "format": fmt, "generated": True}
                        )
                        existing_exts.add(ext)

        if "dependencies" not in info:
            info["dependencies"] = {}

        # Discover dependencies from a Turtle source if available
        primary_path = None
        for s in info["sources"]:
            if s.get("path") and s["path"].endswith(".ttl"):
                primary_path = s["path"]
                break

        if primary_path:
            discovered = get_dependencies_for_file(schema_dir / primary_path)
            for d_prefix, d_url in discovered.items():
                if d_url not in visited_urls and d_prefix not in info["dependencies"]:
                    d_path = f"{d_prefix}/{d_prefix}.ttl"
                    # Initialize with new sources structure
                    info["dependencies"][d_prefix] = {
                        "url": d_url,
                        "sources": [{"path": d_path}],
                    }

        recursive_sync(
            info.get("dependencies", {}),
            schema_dir,
            depth + 1,
            max_depth,
            visited_urls,
            parent_prefix=prefix,
            **kwargs,
        )


def render_hierarchy(node, indent=0):
    """Helper to create a nested markdown list of the schema hierarchy."""
    lines = []
    if not node:
        return lines
    for prefix, info in sorted(node.items()):
        url = info["url"]

        # Handle new sources structure vs legacy path
        if "sources" in info and info["sources"]:
            paths = [s["path"] for s in info["sources"]]
            local = ", ".join([f"`{p}`" for p in paths])
        else:
            local = f"`{info.get('path', 'unknown')}`"

        lines.append("  " * indent + f"- **{prefix}**: [{url}]({url}) (Local: {local})")
        if "dependencies" in info and info["dependencies"]:
            lines.extend(render_hierarchy(info["dependencies"], indent + 1))
    return lines


def update_docs(inventory, schema_dir, docs_vis_path):
    """Update documentation with the latest schema hierarchy and content."""
    if not docs_vis_path.exists():
        logger.error(f"Error: {docs_vis_path} not found.")
        return

    logger.info(f"Updating {docs_vis_path=}...")
    try:
        content = docs_vis_path.read_text()
    except Exception as e:
        logger.error(f"Failed to read docs: {e}")
        return

    # 1. Update Hierarchy Section
    hierarchy_lines = render_hierarchy(inventory)
    hierarchy_text = (
        "## Schema Dependency Hierarchy\n\n" + "\n".join(hierarchy_lines) + "\n"
    )

    if "## Schema Dependency Hierarchy" in content:
        content = re.sub(
            r"## Schema Dependency Hierarchy.*?(?=##|$)",
            hierarchy_text,
            content,
            flags=re.DOTALL,
        )
    else:
        content += "\n" + hierarchy_text

    # 2. Local Process Schema
    sf_path = schema_dir / "sustainable-factory.ttl"
    if sf_path.exists():
        text = sf_path.read_text()
        section = f"\n## Schema: Local Process Ontology\nDefined in `schema/sustainable-factory.ttl`.\n\n```turtle\n{text}\n```\n"
        if "## Schema: Local Process Ontology" in content:
            content = re.sub(
                r"## Schema: Local Process Ontology.*?(?=##|$)",
                section,
                content,
                flags=re.DOTALL,
            )
        else:
            content += section

    try:
        docs_vis_path.write_text(content)
        logger.info(f"Updated {docs_vis_path=}")
    except Exception as e:
        logger.error(f"Failed to write docs: {e}")


def generate_python_constants(inventory, schema_dir, output_path):
    """
    Generate a Python module containing static class definitions for RDF vocabularies.
    This enables IDE support like autocomplete and type checking.
    """
    logger.info(f"Generating Python constants to {output_path}...")

    lines = [
        "# Generated by schematool",
        "import pyoxigraph as ox",
        "",
    ]

    # Process inventory items sorted by prefix
    # Flatten the potentially nested inventory structure into a simple prefix -> info map
    flat_inventory = {}

    def extract_inventory(node):
        for prefix, info in node.items():
            flat_inventory[prefix] = info
            if "dependencies" in info:
                extract_inventory(info["dependencies"])

    extract_inventory(inventory)

    for prefix, info in sorted(flat_inventory.items()):
        # Find a primary Turtle source
        ttl_path = None
        if "sources" in info:
            for s in info["sources"]:
                if s.get("path") and s["path"].endswith(".ttl"):
                    ttl_path = schema_dir / s["path"]
                    break
        elif "path" in info and info["path"].endswith(".ttl"):
            ttl_path = schema_dir / info["path"]

        if not ttl_path or not ttl_path.exists():
            logger.warning(f"  [Skip] No Turtle source found for {prefix}")
            continue

        try:
            g = Graph()
            # Suppress rdflib warnings
            logging.getLogger("rdflib").setLevel(logging.ERROR)
            g.parse(str(ttl_path), format="turtle")
            logging.getLogger("rdflib").setLevel(logging.WARNING)

            # Determine base URI
            # Try to find a base URI from the graph or use the URL from inventory
            base_uri = info.get("url", "")
            # Ensure base URI ends with / or #
            if base_uri and not (base_uri.endswith("/") or base_uri.endswith("#")):
                base_uri += "/"  # Best guess if not explicit

            # Gather terms
            terms = set()

            # Helper to add terms from triples
            def add_terms(rdf_type):
                for s in g.subjects(RDF.type, rdf_type):
                    # Check if it's a URIRef (has a string representation starting with http/https/urn usually)
                    # We can check specific type or just duck type string conversion
                    s_str = str(s)

                    # Skip blank nodes
                    if isinstance(
                        s, (re.Pattern, type(None))
                    ):  # Nonsense check, just to change valid check
                        pass
                    if not s_str.startswith("http"):
                        continue

                    if base_uri and s_str.startswith(base_uri):
                        local = s_str[len(base_uri) :]
                        if local and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", local):
                            terms.add(local)
                    elif "#" in s_str:
                        local = s_str.split("#")[-1]
                        if local and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", local):
                            terms.add(local)
                    elif "/" in s_str:
                        local = s_str.split("/")[-1]
                        if local and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", local):
                            terms.add(local)

            # Classes
            add_terms(RDFS.Class)
            add_terms(OWL.Class)

            # Properties
            add_terms(RDF.Property)
            add_terms(OWL.ObjectProperty)
            add_terms(OWL.DatatypeProperty)
            add_terms(OWL.AnnotationProperty)

            # Generate Class Definition
            class_def = [
                f"class {prefix.upper()}:",
                f'    """Constants for the {prefix} ontology."""',
                f'    _BASE = "{base_uri}"',
                "",
            ]

            if not terms:
                class_def.append("    pass")
            else:
                for term in sorted(terms):
                    class_def.append(f'    {term} = ox.NamedNode(_BASE + "{term}")')

            lines.extend(class_def)
            lines.append("")
            lines.append("")

        except Exception as e:
            logger.error(f"  [Error] Failed to process {prefix}: {e}")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        logger.info(f"  [OK] Generated Python constants: {output_path}")
    except Exception as e:
        logger.error(f"  [Error] Failed to write Python constants: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download external RDF schemas transitively and update markdown documentation per a schema inventory."
    )
    parser.add_argument(
        "--schema-inventory",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
        help=f"Path to the schema inventory JSON file (default: {DEFAULT_INVENTORY_PATH})",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=DEFAULT_SCHEMA_DIR,
        help=f"Path to download schemas to (default: {DEFAULT_SCHEMA_DIR})",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum recursion depth for dependency discovery (default: 3)",
    )
    parser.add_argument(
        "--target-formats",
        nargs="+",
        default=["ttl", "rdfxml", "jsonld"],
        help="Target formats to ensure are present locally (default: ttl rdfxml jsonld)",
    )
    parser.add_argument(
        "--use-pyoxigraph",
        action="store_true",
        help="Use pyoxigraph for RDF transformations instead of rdflib (much faster)",
    )

    parser.add_argument(
        "--docs-vis-path",
        type=Path,
        default=DEFAULT_DOCS_VIS_PATH,
        help=f"Path to the visualization markdown file (default: {DEFAULT_DOCS_VIS_PATH})",
    )

    parser.add_argument(
        "--gen-python",
        type=Path,
        help="Generate a Python module containing static NamedNode constants for ontology terms (e.g. sustainablefactory/schema.py)",
    )

    parser.add_argument(
        "--no-docs", action="store_true", help="Skip updating Markdown documentation"
    )

    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Path to the requests-cache database (default: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Path to the log file (default: {DEFAULT_LOG_FILE})",
    )

    args = parser.parse_args()

    # Calculate REPO_ROOT for path resolution
    repo_root = Path(__file__).resolve().parent.parent.parent.parent

    def resolve_path(p):
        return p if p.is_absolute() else repo_root / p

    inventory_path = resolve_path(args.schema_inventory)
    schema_dir = resolve_path(args.schema_dir)
    docs_vis_path = resolve_path(args.docs_vis_path)
    cache_path = resolve_path(args.cache_path)
    log_file = resolve_path(args.log_file)

    # Initialize cache and logging
    setup_cache(cache_path)
    setup_logging(log_file)

    logger.info(f"Starting schema synchronization using inventory: {inventory_path}")

    inventory = load_inventory(inventory_path)
    if not inventory:
        logger.error(
            "No inventory data found. Please ensure the inventory file is populated."
        )
        return

    recursive_sync(
        inventory,
        schema_dir,
        max_depth=args.max_depth,
        target_formats=args.target_formats,
        use_pyoxigraph=args.use_pyoxigraph,
    )

    # Save results back to the inventory file
    save_inventory(inventory_path, inventory)

    if not args.no_docs:
        update_docs(inventory, schema_dir, docs_vis_path)

    if args.gen_python:
        generate_python_constants(
            load_inventory(inventory_path), schema_dir, resolve_path(args.gen_python)
        )

    logger.info("Synchronization complete.")


if __name__ == "__main__":
    main()

"""
Image modality generator.

CEN's real single-line diagram PDF is NOT bulk-downloadable: it requires a
free, manual "solicitud de informacion" request per the dataset proposal.
InfoTecnica (infotecnica.coordinador.cl) is a JS single-page app backed by a
public API (portal.api.coordinador.cl) -- there is no static <img> to scrape,
and the API's own docs page blocks automated fetching, so its exact endpoint
shapes need to be captured by hand (open the site, DevTools -> Network -> XHR,
click into a substation, and note the JSON endpoint it calls).

Until you've wired that API in, this module renders a lightweight, real-data
schematic instead: it parses the substation/line names that each EAF report
itself names (e.g. "S/E Quelentaro", "linea 110 kV Quelentaro - Las Aranas")
and draws a small labeled graph. It is not topologically accurate (no real
lat/long or full network context) but it IS derived from the real named
components of that real fault event, which keeps the "same system, same
case" constraint intact while you build out the InfoTecnica integration.

To upgrade to real coordinates:
  1. Capture the JSON endpoint(s) InfoTecnica's frontend calls for
     substations/lines (see above).
  2. Replace `_parse_elements_from_description` + layout below with a
     lookup against that API's response (lat/lon or bus/branch topology),
     and plot with the same matplotlib/networkx calls.
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

SE_RE = re.compile(
    r"(?:SS/EE|S/E|Subestaci[oó]n(?:es)?|SE)\s+([A-Za-zÀ-ÿ0-9°ºª#\/().\'’\s\ufffd]+?)"
    r"(?=(?:\s*[,;]|\s+asociad|\s+de\s+(?:SS/EE|S/E|Subestaci[oó]n|SE)|\s+ante\s|\s+y\s+el\b|$))",
    re.IGNORECASE,
)
LINE_RE = re.compile(
    r"l[ií]neas?\s+(?:de\s+)?(?:[\d×xX,.]*\s*kV\s+)?([A-Za-zÀ-ÿ0-9°ºª#\/().\'’\s\ufffd]+?)\s*[\-–—―−\x96\x97\ufffd]+\s*([A-Za-zÀ-ÿ0-9°ºª#\/().\'’\s\ufffd]+?)"
    r"(?=(?:\s*[\-–—―−\x96\x97\ufffd]|\s*[,;]|\s+y\s+\d|\s+tramo\b|\s+circuito\b|$))",
    re.IGNORECASE,
)
PLANT_RE = re.compile(
    r"(?:Central(?:\s+Generadora|\s+Hidroel[eé]ctrica|\s+T[eé]rmica)?|central|Planta|Parque\s+(?:Fotovoltaico|E[oó]lico)|PE|PFV|PMGD|C\.?H\.?|C\.?T\.?)\s+([A-Za-zÀ-ÿ0-9°ºª#\/().\'’\s\ufffd]+?)"
    r"(?=(?:\s*[,;]|\s+asociad|\s+ante\s|\s+y\s+el\b|$))",
    re.IGNORECASE,
)


def _parse_elements_from_description(description):
    substations = [m.strip(' .,;') for m in SE_RE.findall(description)]
    expanded_subs = []
    for s in substations:
        parts = re.split(r"\s*,\s*|\s+y\s+|\s+e\s+", s)
        for p in parts:
            p_clean = p.strip(" .,;")
            if p_clean and p_clean not in expanded_subs:
                expanded_subs.append(p_clean)
    substations = expanded_subs

    line_match = LINE_RE.search(description)
    line_pair = None
    if line_match:
        line_pair = (line_match.group(1).strip(" .,;"), line_match.group(2).strip(" .,;"))

    if not substations and not line_pair:
        plants = [m.strip(" .,;") for m in PLANT_RE.findall(description)]
        if plants:
            substations = plants

    return substations, line_pair



def build_case_diagram(description, out_path, dpi=110):
    substations, line_pair = _parse_elements_from_description(description)

    G = nx.Graph()
    if line_pair:
        G.add_edge(line_pair[0], line_pair[1], fault=True)
    for se in substations:
        G.add_node(se)
    if len(G.nodes) == 0:
        G.add_node(description[:28] if description else "unknown")

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    node_color = "#c0392b" if line_pair else "#2c3e50"
    nx.draw_networkx_edges(G, pos, edge_color="#c0392b", width=2.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=1100, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", ax=ax)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return {"substations": substations, "line_pair": line_pair}

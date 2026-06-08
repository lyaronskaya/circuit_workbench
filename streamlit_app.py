import streamlit as st
import pandas as pd
import numpy as np
import html
from typing import Any, List, Dict, Tuple, Optional
import json
import requests
from dataclasses import dataclass
from datetime import datetime
import logging
import io
import os
from pathlib import Path
import plotly.graph_objects as go
from config.model_groups import MODEL_SPECIFIC_GROUPS
from config.model_defaults import MODEL_DEFAULTS
from scipy import interpolate

# Set up logging with a StringIO buffer to capture logs
log_buffer = io.StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=log_buffer
)
logger = logging.getLogger(__name__)

# Constants
COLOR_PALETTE = [
    "#4ECDC4",  # Turquoise
    "#FFD166",  # Warm yellow
    "#7EDC11",  # Bright lime
    "#FF1493",  # Deep pink
    "#1A9CE0",  # Azure Blue
    "#FF8C42",  # Bright orange
    "#06D6A0",  # Bright turquoise
    "#FFBB33",  # Amber
    "#4B0082",  # Indigo
    "#A7E541",  # Lime green
    "#FF5C5C",  # Coral Red
    "#66D7EE",  # Sky Blue
    "#FFE066",  # Yellow Gold
    "#233FD2",  # Royal Blue
    "#74E39A",  # Mint Green
    "#FF3377",  # Hot Pink
    "#5BC0EB",  # Light blue
    "#FFA07A",  # Light salmon orange
    "#118AB2",  # Blue
    "#C1FF72",  # Lime Green
    "#D90368",  # Magenta
    "#00AA5B",  # Emerald Green
    "#FF6B35",  # Deep orange
    "#F8E16C",  # Light yellow
    "#9F0162",  # Deep Magenta
    "#1EAE98",  # Teal green
    "#FF3366",  # Coral pink
    "#731DD8",  # Electric Purple
]

MODEL_INFO = {
    "gpt2-small": {"layers": 12, "heads": 12},
    "pythia-2.8b": {"layers": 32, "heads": 32},
}

def get_random_color() -> str:
    """Get a random color from the palette."""
    return COLOR_PALETTE[np.random.randint(0, len(COLOR_PALETTE))]

@dataclass
class AttentionPattern:
    sourceLayer: int
    sourceToken: int
    destToken: int
    weight: float
    head: int
    headType: Optional[str] = None

@dataclass
class HeadPair:
    layer: int
    head: int
    color: Optional[str] = None  # Add color field

@dataclass
class HeadGroup:
    id: int
    name: str
    heads: List[HeadPair]
    description: Optional[str] = None
    color: Optional[str] = None  # Add color field
    group_type: str = "custom"

def get_head_color(layer: int, head: int, head_groups: List[HeadGroup]) -> str:
    """Get the color for a head based on its group membership."""
    # Check which group this head belongs to
    for group in head_groups:
        if any(h.layer == layer and h.head == head for h in group.heads):
            # Use group's custom color if set, otherwise use default from palette
            return group.color or COLOR_PALETTE[group.id % len(COLOR_PALETTE)]
    
    # Default blue for heads not in any group
    return '#3B82F6'

def get_active_head_groups(all_groups: List[HeadGroup], active_group_ids: List[int]) -> List[HeadGroup]:
    """Return only groups that are currently active in the graph."""
    active_ids = set(active_group_ids)
    return [group for group in all_groups if group.id in active_ids]


def iter_attention_patterns(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize backend and sample attention payloads into a flat edge list."""
    if not data:
        return []

    if "attentionHeads" in data:
        patterns: List[Dict[str, Any]] = []
        for head_payload in data.get("attentionHeads", []):
            layer = head_payload["layer"]
            head = head_payload["head"]
            for source_token, dest_token, weight in head_payload.get("edges", []):
                patterns.append(
                    {
                        "sourceLayer": layer,
                        "sourceToken": source_token,
                        "destToken": dest_token,
                        "weight": weight,
                        "head": head,
                    }
                )
        return patterns

    return data.get("attentionPatterns", [])


def get_attention_pattern_count(data: Optional[Dict[str, Any]]) -> int:
    """Return the number of visible serialized edges in either payload format."""
    if not data:
        return 0
    if "numEdges" in data:
        return int(data["numEdges"])
    return len(data.get("attentionPatterns", []))

class APIService:
    def __init__(self, base_url: str = None):
        # If no base_url is provided, try to get it from environment variable or use local IP
        if base_url is None:
            import os
            # Try to get URL from environment variable first
            env_url = os.getenv('BACKEND_URL')
            if env_url:
                self.base_url = env_url
            else:
                import subprocess
                try:
                    # Get IP address using ifconfig (macOS compatible)
                    result = subprocess.run(['ifconfig'], capture_output=True, text=True)
                    if result.returncode == 0:
                        # Parse the output to find the first non-localhost IP
                        for line in result.stdout.split('\n'):
                            if 'inet ' in line and '127.0.0.1' not in line:
                                local_ip = line.strip().split(' ')[1]
                                self.base_url = f"http://{local_ip}:8000"
                                break
                        else:
                            # Fallback to localhost if no IP found
                            self.base_url = "http://localhost:8000"
                    else:
                        # Fallback to localhost if ifconfig fails
                        self.base_url = "http://localhost:8000"
                except Exception:
                    # Fallback to localhost if any error occurs
                    self.base_url = "http://localhost:8000"
        else:
            self.base_url = base_url

    def check_backend_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            logger.warning(f"Could not connect to backend at {self.base_url}. Make sure the backend server is running.")
            return False
        except requests.exceptions.Timeout:
            logger.warning("Backend request timed out. The server might be overloaded or not responding.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking backend health: {str(e)}")
            return False

    def process_text(
        self,
        text: str,
        model: str,
        threshold: float,
        top_k: int,
        selected_heads: List[Dict[str, int]],
    ) -> Dict:
        response = requests.post(
            f"{self.base_url}/process",
            json={
                "text": text,
                "model_name": model,
                "threshold": threshold,
                "top_k": top_k,
                "selected_heads": selected_heads,
            },
            timeout=30
        )
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
        return response.json()

    def evaluate_text(
        self,
        text: str,
        model: str,
        target_token: Optional[str],
        ablated_heads: List[Dict[str, int]]
    ) -> Dict:
        response = requests.post(
            f"{self.base_url}/evaluate",
            json={
                "text": text,
                "model_name": model,
                "target_token": target_token,
                "ablated_heads": ablated_heads,
            },
            timeout=120
        )
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
        return response.json()

    def list_datasets(self) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.base_url}/datasets", timeout=30)
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
        return response.json()["datasets"]

    def get_dataset(self, dataset_name: str) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/datasets/{dataset_name}", timeout=30)
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
        return response.json()

    def validate_dataset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/datasets/validate", json={"payload": payload}, timeout=30)
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
        return response.json()

    def save_dataset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/datasets/save", json={"payload": payload}, timeout=30)
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
        return response.json()

    def evaluate_dataset(
        self,
        dataset_name: str,
        model: Optional[str],
        ablated_heads: List[Dict[str, int]]
    ) -> Dict:
        response = requests.post(
            f"{self.base_url}/evaluate-dataset",
            json={
                "dataset_name": dataset_name,
                "model_name": model,
                "ablated_heads": ablated_heads,
            },
            timeout=180
        )
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
        return response.json()

    def build_max_logit_diff_graph(
        self,
        text: str,
        corrupted_text: str,
        model: str,
        target_token: Optional[str],
        top_k: int,
        top_heads_per_layer: int,
        selected_heads: List[Dict[str, int]],
    ) -> Dict:
        response = requests.post(
            f"{self.base_url}/max-logit-diff-graph",
            json={
                "text": text,
                "corrupted_text": corrupted_text,
                "model_name": model,
                "target_token": target_token,
                "top_k": top_k,
                "top_heads_per_layer": top_heads_per_layer,
                "selected_heads": selected_heads,
            },
            timeout=180,
        )
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
        return response.json()

def get_default_text_for_model(model: str) -> str:
    """Get the default text for a given model from configuration."""
    return MODEL_DEFAULTS.get(model, {}).get("default_text", "")

def get_task_presets_for_model(model: str) -> List[Dict[str, str]]:
    """Return task presets for a model."""
    return MODEL_DEFAULTS.get(model, {}).get("task_presets", [])

def get_model_dimensions(model: str, attention_data: Optional[Dict] = None) -> Tuple[int, int]:
    """Return source-layer count and heads per layer for a model."""
    if attention_data:
        num_layers = max(int(attention_data.get("numLayers", 1)) - 1, 1)
        num_heads = int(attention_data.get("numHeads", 0))
        if num_heads > 0:
            return num_layers, num_heads

    model_info = MODEL_INFO.get(model)
    if model_info:
        return model_info["layers"], model_info["heads"]
    return 0, 0

def parse_uploaded_dataset(uploaded_file) -> Dict[str, Any]:
    """Parse uploaded JSON or CSV dataset into the canonical payload."""
    file_name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()

    if file_name.endswith(".json"):
        payload = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON dataset must be an object.")
        return payload

    if file_name.endswith(".csv"):
        frame = pd.read_csv(io.BytesIO(raw_bytes))
        required_columns = {"id", "text", "target_token"}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(f"CSV dataset is missing required columns: {', '.join(sorted(missing_columns))}")

        examples = []
        for row in frame.fillna("").to_dict(orient="records"):
            example = {
                "id": str(row["id"]),
                "text": str(row["text"]),
                "target_token": str(row["target_token"]),
            }
            if str(row.get("corrupted_text", "")).strip():
                example["corrupted_text"] = str(row["corrupted_text"])
            metadata = {
                key: value for key, value in row.items()
                if key not in {"id", "text", "target_token", "corrupted_text"} and str(value).strip()
            }
            if metadata:
                example["metadata"] = metadata
            examples.append(example)

        return {
            "name": Path(uploaded_file.name).stem,
            "description": "Uploaded from CSV",
            "metric": "target_probability",
            "examples": examples,
        }

    raise ValueError("Unsupported file type. Please upload a JSON or CSV dataset.")

def create_attention_graph(
    data: Dict,
    threshold: float,
    selected_heads: List[HeadPair],
    head_groups: List[HeadGroup]
) -> None:
    """Create graph visualization using D3.js."""
    logger.info(
        "Creating graph with data: numLayers=%s, numTokens=%s, numPatterns=%s",
        data["numLayers"],
        data["numTokens"],
        get_attention_pattern_count(data),
    )
    logger.info(f"Selected heads: {[(h.layer, h.head) for h in selected_heads]}")
    logger.info(f"Head groups: {[(g.name, len(g.heads)) for g in head_groups]}")
    
    # Filter attention patterns based on threshold and selected heads
    filtered_patterns = get_filtered_attention_patterns(data, threshold, selected_heads, head_groups)

    # Prepare data for D3 visualization
    viz_data = {
        'numLayers': data['numLayers'],
        'numTokens': data['numTokens'],
        'numHeads': data['numHeads'],
        'tokens': data.get('tokens', [f'T{i}' for i in range(data['numTokens'])]),
        'attentionPatterns': filtered_patterns,
        'headGroups': [
            {
                'id': group.id,
                'name': group.name,
                'description': group.description,
                'color': group.color,
                'heads': [{'layer': h.layer, 'head': h.head} for h in group.heads]
            }
            for group in head_groups
        ],
        'selectedHeads': [{'layer': h.layer, 'head': h.head} for h in selected_heads]
    }

    # Create HTML with embedded D3.js visualization
    html = f"""
    <div id="visualization-container" style="width: 100%; height: 800px;"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .attention-line {{
            stroke-opacity: 0.6;
            stroke-width: 4;
            cursor: pointer;
        }}
        .attention-line:hover {{
            stroke-opacity: 0.9;
            stroke-width: 6;
        }}
        .grid-point {{
            fill: #e5e7eb;
            cursor: pointer;
        }}
        .grid-point:hover {{
            fill: #d1d5db;
            r: 8;
        }}
        .hover-target {{
            fill: transparent;
            cursor: pointer;
        }}
        #graph-tooltip {{
            display: none;
            position: absolute;
            background: white;
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
        }}
    </style>
    <script>
        // Color palette for individual heads
        const colorPalette = [
            "#38B2AC", "#9F7AEA", "#F6AD55", "#68D391", "#F687B3",
            "#4FD1C5", "#B794F4", "#7F9CF5", "#C6F6D5", "#FBD38D",
            "#76E4F7", "#E9D8FD", "#90CDF4", "#FEB2B2", "#81E6D9",
            "#D6BCFA", "#FBB6CE", "#B2F5EA", "#667EEA", "#ED64A6"
        ];

        // Graph dimensions
        const graphDimensions = {{
            width: 1000,
            height: 700,
            padding: {{ top: 40, right: 180, bottom: 60, left: 60 }}
        }};

        // Data from Streamlit
        const data = {json.dumps(viz_data)};
        let headGroups = data.headGroups || [];
        let selectedHeads = data.selectedHeads || [];

        // Function to get visible heads
        function getVisibleHeads() {{
            return [...selectedHeads, ...headGroups.flatMap(g => g.heads)];
        }}

        // Function to get head group
        function getHeadGroup(layer, head) {{
            for (const group of headGroups) {{
                if (group.heads.some(h => h.layer === layer && h.head === head)) {{
                    return group.id;
                }}
            }}
            return -1;
        }}

        // Function to get group color
        function getGroupColor(groupId) {{
            const group = headGroups.find(g => g.id === groupId);
            if (group?.color) {{
                return group.color;
            }}
            return colorPalette[groupId % colorPalette.length];
        }}

        // Function to get head color
        function getHeadColor(layer, head) {{
            // Check if this is a wildcard head (has grey color)
            const headObj = selectedHeads.find(h => h.layer === layer && h.head === head);
            if (headObj && headObj.color && headObj.color.startsWith('#')) {{
                // Check if it's a grey color (all RGB components are equal)
                const r = parseInt(headObj.color.slice(1, 3), 16);
                const g = parseInt(headObj.color.slice(3, 5), 16);
                const b = parseInt(headObj.color.slice(5, 7), 16);
                if (r === g && g === b) {{
                    return headObj.color;
                }}
            }}
            // For non-wildcard heads, use the original color palette
            return individualHeadColorScale(head.toString());
        }}

        // Main drawing function
        function drawGraph() {{
            const svg = d3.select("#visualization-container")
                .append("svg")
                .attr("width", graphDimensions.width)
                .attr("height", graphDimensions.height);

            const width = graphDimensions.width;
            const height = graphDimensions.height;
            const padding = graphDimensions.padding;
            const legendWidth = padding.right;
            const graphWidth = width - padding.left - padding.right;
            const graphHeight = height - padding.top - padding.bottom;
            const tokenWidth = graphWidth / data.numTokens;
            const layerHeight = graphHeight / (data.numLayers - 1);

            // Create nodes
            const nodes = [];
            for (let l = 0; l < data.numLayers; l++) {{
                for (let t = 0; t < data.numTokens; t++) {{
                    nodes.push({{
                        id: `${{l}}-${{t}}`,
                        layer: l,
                        token: t,
                        x: padding.left + t * tokenWidth + tokenWidth / 2,
                        y: height - (padding.bottom + l * layerHeight)
                    }});
                }}
            }}

            // Create color scales
            const individualHeadColorScale = d3.scaleOrdinal(colorPalette)
                .domain(Array.from({{ length: data.numHeads }}, (_, i) => i.toString()));

            // Filter edges
            const visibleHeadPairs = getVisibleHeads();
            const links = data.attentionPatterns
                .filter(edge => {{
                    const isVisible = visibleHeadPairs.some(h =>
                        h.layer === edge.sourceLayer && h.head === edge.head
                    );
                    return edge.weight >= {threshold} && isVisible;
                }})
                .map(edge => ({{
                    source: `${{edge.sourceLayer}}-${{edge.sourceToken}}`,
                    target: `${{edge.sourceLayer + 1}}-${{edge.destToken}}`,
                    weight: edge.weight,
                    head: edge.head,
                    groupId: getHeadGroup(edge.sourceLayer, edge.head) ?? -1
                }}));

            // Draw layers and tokens labels
            const g = svg.append("g");

            // Layer labels
            for (let l = 0; l < data.numLayers; l++) {{
                g.append("text")
                    .attr("x", padding.left / 2 + 25)
                    .attr("y", height - (padding.bottom + l * layerHeight))
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "middle")
                    .text(l.toString());
            }}

            // Y-axis label
            g.append("text")
                .attr("x", padding.left / 2)
                .attr("y", height / 2)
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle")
                .attr("font-size", "14px")
                .attr("font-weight", "medium")
                .text("Layer");

            // Token labels
            for (let t = 0; t < data.numTokens; t++) {{
                g.append("text")
                    .attr("x", padding.left + t * tokenWidth + tokenWidth / 2)
                    .attr("y", height - padding.bottom / 2)
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "middle")
                    .text(data.tokens?.[t] || `T${{t}}`);
            }}

            // X-axis label
            g.append("text")
                .attr("x", width / 2)
                .attr("y", height - padding.bottom / 4)
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle")
                .attr("font-size", "14px")
                .attr("font-weight", "medium")
                .text("Token");

            // Draw edges with curved paths
            const linkElements = g.selectAll("path")
                .data(links)
                .enter()
                .append("path")
                .attr("d", d => {{
                    const source = nodes.find(n => n.id === d.source);
                    const target = nodes.find(n => n.id === d.target);
                    const dx = target.x - source.x;
                    const controlPoint1x = source.x + dx * 0.5;
                    const controlPoint1y = source.y;
                    const controlPoint2x = target.x - dx * 0.5;
                    const controlPoint2y = target.y;
                    return `M ${{source.x}} ${{source.y}} C ${{controlPoint1x}} ${{controlPoint1y}}, ${{controlPoint2x}} ${{controlPoint2y}}, ${{target.x}} ${{target.y}}`;
                }})
                .attr("fill", "none")
                .attr("stroke", d => {{
                    if (d.groupId >= 0) {{
                        return getGroupColor(d.groupId);
                    }}
                    return getHeadColor(d.sourceLayer, d.head);
                }})
                .attr("stroke-width", 4)
                .attr("opacity", 0.6)
                .attr("class", "attention-line")
                .on("mouseover", function(event, d) {{
                    d3.select(this)
                        .attr("opacity", 1)
                        .attr("stroke-width", 6);
                    const tooltip = d3.select("#graph-tooltip");
                    const group = headGroups.find(g => g.id === d.groupId);
                    const headObj = selectedHeads.find(h => h.layer === d.sourceLayer && h.head === d.head);
                    const isWildcard = headObj && headObj.color && headObj.color.startsWith('#') && 
                                     parseInt(headObj.color.slice(1, 3), 16) === parseInt(headObj.color.slice(3, 5), 16) &&
                                     parseInt(headObj.color.slice(3, 5), 16) === parseInt(headObj.color.slice(5, 7), 16);
                    tooltip.style("display", "block")
                        .html(`Head: Layer ${{d.source.split("-")[0]}}, Head ${{d.head}}${{isWildcard ? ' (Wildcard)' : ''}}<br>
                               Weight: ${{d.weight.toFixed(4)}}${{group ?
                               `<br>Group: ${{group.name}}${{group.description ?
                               `<br><span style="font-style: italic; font-size: 11px;">${{group.description}}</span>` : ''}}` :
                               '<br>Individual Head'}}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mouseout", function() {{
                    d3.select(this)
                        .attr("opacity", 0.6)
                        .attr("stroke-width", 4);
                    d3.select("#graph-tooltip").style("display", "none");
                }});

            // Draw nodes
            const nodeElements = g.selectAll("circle")
                .data(nodes)
                .enter()
                .append("circle")
                .attr("class", "grid-point")
                .attr("cx", d => d.x)
                .attr("cy", d => d.y)
                .attr("r", 6)
                .attr("fill", "#e5e7eb")
                .on("mouseover", function(event, d) {{
                    d3.select(this)
                        .attr("r", 8)
                        .attr("fill", "#d1d5db");
                    const tooltip = d3.select("#graph-tooltip");
                    tooltip.style("display", "block")
                        .html(`Layer ${{d.layer}}, Token ${{d.token}}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mouseout", function() {{
                    d3.select(this)
                        .attr("r", 6)
                        .attr("fill", "#e5e7eb");
                    d3.select("#graph-tooltip").style("display", "none");
                }});

            // Add invisible hover targets for easier interaction
            g.selectAll("circle.hover-target")
                .data(nodes)
                .enter()
                .append("circle")
                .attr("class", "hover-target")
                .attr("cx", d => d.x)
                .attr("cy", d => d.y)
                .attr("r", 12)
                .attr("fill", "transparent")
                .on("mouseover", function(event, d) {{
                    const tooltip = d3.select("#graph-tooltip");
                    tooltip.style("display", "block")
                        .html(`Layer ${{d.layer}}, Token ${{d.token}}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                    d3.select(this.parentNode)
                        .select(`circle:not(.hover-target)[data-node-id="${{d.id}}"]`)
                        .attr("r", 8)
                        .attr("fill", "#d1d5db");
                }})
                .on("mouseout", function(event, d) {{
                    d3.select("#graph-tooltip").style("display", "none");
                    d3.select(this.parentNode)
                        .select(`circle:not(.hover-target)[data-node-id="${{d.id}}"]`)
                        .attr("r", 6)
                        .attr("fill", "#e5e7eb");
                }});

            // Draw legend
            const legend = g.append("g")
                .attr("transform", `translate(${{width - legendWidth + 20}}, ${{padding.top}})`);

            // Add legend title
            legend.append("text")
                .attr("x", 0)
                .attr("y", 0)
                .attr("font-size", "14px")
                .attr("font-weight", "bold")
                .text("Legend");

            // Add group colors to legend
            headGroups.forEach((group, i) => {{
                const y = 30 + i * 25;
                legend.append("rect")
                    .attr("x", 0)
                    .attr("y", y)
                    .attr("width", 15)
                    .attr("height", 15)
                    .attr("fill", getGroupColor(group.id));
                const groupText = legend.append("text")
                    .attr("x", 25)
                    .attr("y", y + 12)
                    .attr("font-size", "12px")
                    .text(group.name);
                if (group.description) {{
                    groupText
                        .on("mouseenter", function(event) {{
                            const tooltip = d3.select("#graph-tooltip");
                            tooltip.style("display", "block")
                                .html(`<strong>${{group.name}}</strong><br>${{group.description}}`)
                                .style("left", (event.pageX + 10) + "px")
                                .style("top", (event.pageY - 10) + "px");
                        }})
                        .on("mouseleave", function() {{
                            d3.select("#graph-tooltip").style("display", "none");
                        }});
                }}
            }});

            // Add separator
            const separatorY = 30 + headGroups.length * 25 + 10;
            legend.append("line")
                .attr("x1", 0)
                .attr("x2", legendWidth - padding.left)
                .attr("y1", separatorY)
                .attr("y2", separatorY)
                .attr("stroke", "#e5e7eb")
                .attr("stroke-width", 2);

            // Add individual heads section
            legend.append("text")
                .attr("x", 0)
                .attr("y", separatorY + 25)
                .attr("font-size", "12px")
                .attr("font-weight", "bold")
                .text("Individual Heads");

            // Add individual head colors to legend
            const visibleIndividualHeads = selectedHeads.filter(h =>
                !headGroups.some(g => g.heads.some(gh => gh.layer === h.layer && gh.head === h.head))
            );

            visibleIndividualHeads.forEach((head, i) => {{
                const y = separatorY + 40 + i * 25;
                legend.append("rect")
                    .attr("x", 0)
                    .attr("y", y)
                    .attr("width", 15)
                    .attr("height", 15)
                    .attr("fill", getHeadColor(head.layer, head.head));
                legend.append("text")
                    .attr("x", 25)
                    .attr("y", y + 12)
                    .attr("font-size", "12px")
                    .text(`Layer ${{head.layer}}, Head ${{head.head}}${{head.color && head.color.startsWith('#') && 
                          parseInt(head.color.slice(1, 3), 16) === parseInt(head.color.slice(3, 5), 16) &&
                          parseInt(head.color.slice(3, 5), 16) === parseInt(head.color.slice(5, 7), 16) ? ' (Wildcard)' : ''}}`);
            }});
        }}

        // Add tooltip div
        const tooltip = document.createElement("div");
        tooltip.id = "graph-tooltip";
        document.body.appendChild(tooltip);

        // Draw the graph
        drawGraph();
    </script>
    """
    
    # Display the visualization
    st.components.v1.html(html, height=800)

def create_max_logit_diff_graph(data: Dict) -> None:
    """Create a graph for the best clean-corrupted logit-diff head in each layer at the final position."""
    html = f"""
    <div id="max-logit-diff-container" style="width: 100%; height: 800px;"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .causal-line {{
            fill: none;
            stroke-linecap: round;
            cursor: pointer;
        }}
        .causal-node {{
            cursor: pointer;
        }}
        #causal-graph-tooltip {{
            display: none;
            position: absolute;
            background: rgba(255, 255, 255, 0.98);
            padding: 8px 10px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
        }}
    </style>
    <script>
        const data = {json.dumps(data)};
        const dims = {{
            width: 1000,
            height: 700,
            padding: {{ top: 40, right: 40, bottom: 60, left: 60 }}
        }};

        const svg = d3.select("#max-logit-diff-container")
            .append("svg")
            .attr("width", dims.width)
            .attr("height", dims.height);

        const width = dims.width;
        const height = dims.height;
        const padding = dims.padding;
        const graphWidth = width - padding.left - padding.right;
        const graphHeight = height - padding.top - padding.bottom;
        const tokenWidth = graphWidth / data.numTokens;
        const layerHeight = graphHeight / (data.numLayers - 1);
        const selectedNodes = data.importantNodes || [];
        const selectedNodeLookup = new Map();
        selectedNodes.forEach(node => {{
            const key = `${{node.layer}}-${{node.token}}`;
            if (!selectedNodeLookup.has(key)) {{
                selectedNodeLookup.set(key, []);
            }}
            selectedNodeLookup.get(key).push(node);
        }});
        const edgeScores = data.attentionPatterns.map(edge => Math.abs(edge.combined_score));
        const maxEdgeScore = d3.max(edgeScores) || 1;
        const colorScale = d3.scaleLinear()
            .domain([-1, 0, 1])
            .range(["#b91c1c", "#a8a29e", "#166534"]);

        const nodes = [];
        for (let layer = 0; layer < data.numLayers; layer++) {{
            for (let token = 0; token < data.numTokens; token++) {{
                nodes.push({{
                    id: `${{layer}}-${{token}}`,
                    layer,
                    token,
                    x: padding.left + token * tokenWidth + tokenWidth / 2,
                    y: height - (padding.bottom + layer * layerHeight)
                }});
            }}
        }}

        const g = svg.append("g");
        const tooltip = document.createElement("div");
        tooltip.id = "causal-graph-tooltip";
        document.body.appendChild(tooltip);

        for (let layer = 0; layer < data.numLayers; layer++) {{
            g.append("text")
                .attr("x", padding.left / 2 + 25)
                .attr("y", height - (padding.bottom + layer * layerHeight))
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle")
                .text(layer.toString());
        }}

        for (let token = 0; token < data.numTokens; token++) {{
            g.append("text")
                .attr("x", padding.left + token * tokenWidth + tokenWidth / 2)
                .attr("y", height - padding.bottom / 2)
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle")
                .text(data.tokens?.[token] || `T${{token}}`);
        }}

        g.selectAll("circle")
            .data(nodes)
            .enter()
            .append("circle")
            .attr("class", "causal-node")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", d => selectedNodeLookup.has(`${{d.layer}}-${{d.token}}`) ? 9 : 5)
            .attr("fill", d => {{
                const nodeList = selectedNodeLookup.get(`${{d.layer}}-${{d.token}}`);
                const topNode = nodeList?.[0];
                return topNode
                    ? colorScale(Math.max(-1, Math.min(1, topNode.logit_diff_delta / (Math.abs(topNode.logit_diff_delta) || 1))))
                    : "#d6d3d1";
            }})
            .attr("stroke", d => selectedNodeLookup.has(`${{d.layer}}-${{d.token}}`) ? "#1f2937" : "none")
            .attr("stroke-width", d => selectedNodeLookup.has(`${{d.layer}}-${{d.token}}`) ? 1.5 : 0)
            .on("mouseover", function(event, d) {{
                const tooltipDiv = d3.select("#causal-graph-tooltip");
                const nodeList = selectedNodeLookup.get(`${{d.layer}}-${{d.token}}`);
                if (nodeList?.length) {{
                    const headLines = nodeList.map(node =>
                        `L${{node.sourceLayer}}H${{node.head}} (#${{(node.rankInLayer || 0) + 1}}): ${{node.logit_diff_delta.toFixed(4)}}`
                    ).join("<br>");
                    tooltipDiv.style("display", "block")
                        .html(
                            `Layer ${{d.layer}}, Token ${{d.token}}<br>` +
                            `Important heads:<br>${{headLines}}`
                        )
                        .style("left", (event.pageX + 12) + "px")
                        .style("top", (event.pageY - 12) + "px");
                }} else {{
                    tooltipDiv.style("display", "block")
                        .html(`Layer ${{d.layer}}, Token ${{d.token}}`)
                        .style("left", (event.pageX + 12) + "px")
                        .style("top", (event.pageY - 12) + "px");
                }}
            }})
            .on("mouseout", function() {{
                d3.select("#causal-graph-tooltip").style("display", "none");
            }});

        g.selectAll("path")
            .data(data.attentionPatterns)
            .enter()
            .append("path")
            .attr("class", "causal-line")
            .attr("d", edge => {{
                const source = nodes.find(node => node.id === `${{edge.sourceLayer}}-${{edge.sourceToken}}`);
                const target = nodes.find(node => node.id === `${{edge.sourceLayer + 1}}-${{edge.destToken}}`);
                const dx = target.x - source.x;
                const controlPoint1x = source.x + dx * 0.5;
                const controlPoint1y = source.y;
                const controlPoint2x = target.x - dx * 0.5;
                const controlPoint2y = target.y;
                return `M ${{source.x}} ${{source.y}} C ${{controlPoint1x}} ${{controlPoint1y}}, ${{controlPoint2x}} ${{controlPoint2y}}, ${{target.x}} ${{target.y}}`;
            }})
            .attr("stroke", edge => colorScale(Math.max(-1, Math.min(1, edge.logit_diff_delta / (Math.abs(edge.logit_diff_delta) || 1)))))
            .attr("stroke-width", edge => 2 + 8 * (Math.abs(edge.combined_score) / maxEdgeScore))
            .attr("stroke-opacity", edge => 0.28 + 0.72 * (Math.abs(edge.combined_score) / maxEdgeScore))
            .on("mouseover", function(event, edge) {{
                d3.select(this).attr("stroke-opacity", 1);
                d3.select("#causal-graph-tooltip")
                    .style("display", "block")
                    .html(
                        `L${{edge.sourceLayer}}H${{edge.head}}<br>` +
                        `source: ${{data.tokens?.[edge.sourceToken] || edge.sourceToken}} -> target: ${{data.tokens?.[edge.destToken] || edge.destToken}}<br>` +
                        `attention: ${{edge.weight.toFixed(4)}}<br>` +
                        `clean - corrupted: ${{edge.logit_diff_delta.toFixed(4)}}`
                    )
                    .style("left", (event.pageX + 12) + "px")
                    .style("top", (event.pageY - 12) + "px");
            }})
            .on("mouseout", function(event, edge) {{
                d3.select(this)
                    .attr("stroke-opacity", 0.28 + 0.72 * (Math.abs(edge.combined_score) / maxEdgeScore));
                d3.select("#causal-graph-tooltip").style("display", "none");
            }});

        const labelGroup = g.append("g");
        const highlightedNodes = selectedNodes.map(node => {{
            const graphNode = nodes.find(candidate => candidate.id === `${{node.layer}}-${{node.token}}`);
            return {{
                ...node,
                x: graphNode?.x ?? 0,
                y: graphNode?.y ?? 0
            }};
        }});

        const labelSelection = labelGroup.selectAll("g")
            .data(highlightedNodes)
            .enter()
            .append("g")
            .attr("transform", node => `translate(${{node.x + 12 + (node.rankInLayer || 0) * 52}}, ${{node.y - 10}})`);

        labelSelection.append("rect")
            .attr("class", "causal-head-halo")
            .attr("rx", 6)
            .attr("ry", 6)
            .attr("width", 66)
            .attr("height", 18);

        labelSelection.append("text")
            .attr("class", "causal-head-label")
            .attr("x", 6)
            .attr("y", 12)
            .text(node => `H${{node.head}}`);

        g.append("text")
            .attr("x", padding.left)
            .attr("y", padding.top - 12)
            .attr("font-size", "13px")
            .attr("font-weight", "600")
            .text(`Important heads for all layers at final token position ${{data.selectedPosition}}`);
    </script>
    """

    st.components.v1.html(html, height=800)

def convert_predefined_groups_to_head_groups(model: str) -> List[HeadGroup]:
    """Convert predefined groups from MODEL_SPECIFIC_GROUPS to HeadGroup objects."""
    groups = MODEL_SPECIFIC_GROUPS.get(model, [])
    return [
        HeadGroup(
            id=idx,
            name=group["name"],
            description=group.get("description"),
            heads=[HeadPair(layer=layer, head=head) for layer, head in group["vertices"]],
            color=None,
            group_type="predefined",
        )
        for idx, group in enumerate(groups)
    ]

def load_sample_data(model: str) -> Dict:
    """Load sample attention data from JSON file."""
    try:
        file_path = os.path.join("public", "data", f"sample-attention-{model}.json")
        if not os.path.exists(file_path):
            logger.error(f"Sample data file not found: {file_path}")
            st.error(f"Sample data file not found for model {model}. Please ensure the backend is running or contact support.")
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing sample data JSON: {str(e)}")
        st.error("Error parsing sample data. The file might be corrupted.")
        return None
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        st.error("An unexpected error occurred while loading sample data.")
        return None

def get_visible_head_pairs(
    selected_heads: List[HeadPair],
    head_groups: List[HeadGroup]
) -> List[HeadPair]:
    """Return unique visible heads from manual selection and groups."""
    visible = {}
    for head in selected_heads + [head for group in head_groups for head in group.heads]:
        visible[(head.layer, head.head)] = head
    return list(visible.values())

def get_filtered_attention_patterns(
    data: Optional[Dict],
    threshold: float,
    selected_heads: List[HeadPair],
    head_groups: List[HeadGroup]
) -> List[Dict]:
    """Return visible attention patterns above threshold."""
    if not data:
        return []

    visible_heads = {(head.layer, head.head) for head in get_visible_head_pairs(selected_heads, head_groups)}
    if not visible_heads:
        return []

    return [
        pattern for pattern in iter_attention_patterns(data)
        if pattern["weight"] >= threshold and (pattern["sourceLayer"], pattern["head"]) in visible_heads
    ]

def summarize_attention_metrics(
    data: Optional[Dict],
    threshold: float,
    selected_heads: List[HeadPair],
    head_groups: List[HeadGroup]
) -> Dict[str, Any]:
    """Build lightweight graph metrics for the workspace panels."""
    filtered_patterns = get_filtered_attention_patterns(data, threshold, selected_heads, head_groups)
    visible_heads = get_visible_head_pairs(selected_heads, head_groups)
    weights = [pattern["weight"] for pattern in filtered_patterns]

    return {
        "visible_heads": len(visible_heads),
        "visible_edges": len(filtered_patterns),
        "max_weight": max(weights) if weights else 0.0,
        "mean_weight": float(np.mean(weights)) if weights else 0.0,
        "top_edges": sorted(filtered_patterns, key=lambda pattern: pattern["weight"], reverse=True)[:8]
    }

def build_head_inspector_options(
    selected_heads: List[HeadPair],
    head_groups: List[HeadGroup]
) -> List[Tuple[str, Tuple[int, int]]]:
    """Create readable options for inspector selectbox."""
    options = []
    seen = set()

    for head in get_visible_head_pairs(selected_heads, head_groups):
        key = (head.layer, head.head)
        if key in seen:
            continue
        seen.add(key)
        options.append((f"L{head.layer}H{head.head}", key))

    return sorted(options, key=lambda item: item[1])

def summarize_head_detail(
    data: Optional[Dict],
    threshold: float,
    selected_heads: List[HeadPair],
    head_groups: List[HeadGroup],
    head_key: Optional[Tuple[int, int]]
) -> Optional[Dict[str, Any]]:
    """Build inspector details for a single visible head."""
    if not data or head_key is None:
        return None

    layer, head = head_key
    filtered_patterns = [
        pattern for pattern in get_filtered_attention_patterns(data, threshold, selected_heads, head_groups)
        if pattern["sourceLayer"] == layer and pattern["head"] == head
    ]

    if not filtered_patterns:
        return {
            "layer": layer,
            "head": head,
            "group_names": [
                group.name for group in head_groups
                if any(group_head.layer == layer and group_head.head == head for group_head in group.heads)
            ],
            "top_edges": [],
            "mean_weight": 0.0
        }

    group_names = [
        group.name for group in head_groups
        if any(group_head.layer == layer and group_head.head == head for group_head in group.heads)
    ]

    return {
        "layer": layer,
        "head": head,
        "group_names": group_names,
        "top_edges": sorted(filtered_patterns, key=lambda pattern: pattern["weight"], reverse=True)[:5],
        "mean_weight": float(np.mean([pattern["weight"] for pattern in filtered_patterns]))
    }

def render_signal_route_rows(edges: List[Dict], tokens: List[str]) -> str:
    """Render top routes as compact HTML rows without markdown indentation artifacts."""
    rows = []
    for edge in edges:
        source_token = tokens[edge["sourceToken"]] if edge["sourceToken"] < len(tokens) else f"T{edge['sourceToken']}"
        dest_token = tokens[edge["destToken"]] if edge["destToken"] < len(tokens) else f"T{edge['destToken']}"
        rows.append(
            (
                '<div class="signal-row">'
                f'<div class="signal-route">{html.escape(source_token)} -&gt; '
                f'L{edge["sourceLayer"]}H{edge["head"]} -&gt; {html.escape(dest_token)}</div>'
                f'<div class="signal-meta">Weight {edge["weight"]:.4f} • '
                f'source pos {edge["sourceToken"]} • destination pos {edge["destToken"]}</div>'
                '</div>'
            )
        )
    return '<div class="signal-list">' + "".join(rows) + '</div>'

def render_selected_heads_compact(selected_heads: List[HeadPair], max_visible: int = 48) -> str:
    """Render a compact summary for large head selections."""
    if not selected_heads:
        return ""

    layer_counts: Dict[int, int] = {}
    for head in selected_heads:
        layer_counts[head.layer] = layer_counts.get(head.layer, 0) + 1

    summary_html = "".join(
        f'<span class="selection-summary-chip">L{layer}: {count}</span>'
        for layer, count in sorted(layer_counts.items())
    )

    visible_heads = selected_heads[:max_visible]
    visible_html = "".join(
        f'<span class="head-button compact-head-chip" style="background-color: {head.color or "#3B82F6"}">{head.layer},{head.head}</span>'
        for head in visible_heads
    )

    overflow = len(selected_heads) - len(visible_heads)
    overflow_html = (
        f'<span class="selection-overflow-chip">+{overflow} more</span>'
        if overflow > 0 else ""
    )

    return (
        '<div class="compact-selection-card">'
        f'<div class="compact-selection-meta"><strong>{len(selected_heads)}</strong> selected heads</div>'
        f'<div class="selection-summary-row">{summary_html}</div>'
        f'<div class="compact-selection-grid">{visible_html}{overflow_html}</div>'
        '</div>'
    )

def get_unique_head_pairs(heads: List[HeadPair]) -> List[HeadPair]:
    """Deduplicate heads while preserving order."""
    seen = set()
    unique_heads = []
    for head in heads:
        key = (head.layer, head.head)
        if key in seen:
            continue
        seen.add(key)
        unique_heads.append(head)
    return unique_heads

def get_ablation_heads(
    mode: str,
    selected_heads: List[HeadPair],
    head_groups: List[HeadGroup],
    selected_group_name: Optional[str],
    selected_layer: Optional[int],
    attention_data: Optional[Dict]
) -> List[HeadPair]:
    """Resolve ablation target heads from the chosen UI mode."""
    if mode == "Selected heads":
        return get_unique_head_pairs(selected_heads)

    if mode == "Head group" and selected_group_name:
        for group in head_groups:
            if group.name == selected_group_name:
                return get_unique_head_pairs(group.heads)
        return []

    if mode == "All heads in layer" and selected_layer is not None:
        num_heads = (attention_data or {}).get("numHeads", 0)
        if num_heads:
            return [HeadPair(layer=selected_layer, head=head_index) for head_index in range(num_heads)]
        layer_heads = [
            head for head in get_unique_head_pairs(
                selected_heads + [candidate for group in head_groups for candidate in group.heads]
            )
            if head.layer == selected_layer
        ]
        return layer_heads

    return []

def format_metric_value(value: Optional[float], precision: int = 4) -> str:
    """Format scalar metric values for display."""
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"

def format_int_metric(value: Optional[int]) -> str:
    """Format integer metrics for display."""
    if value is None:
        return "n/a"
    return str(value)

def append_unique_heads(existing_heads: List[HeadPair], new_heads: List[HeadPair]) -> List[HeadPair]:
    """Append heads without duplicating layer/head pairs."""
    seen = {(head.layer, head.head) for head in existing_heads}
    merged_heads = list(existing_heads)
    for head in new_heads:
        key = (head.layer, head.head)
        if key in seen:
            continue
        seen.add(key)
        merged_heads.append(HeadPair(layer=head.layer, head=head.head, color=head.color))
    return merged_heads

def next_group_id(all_groups: List[HeadGroup]) -> int:
    """Return the next available group id."""
    return max((group.id for group in all_groups), default=-1) + 1

def refresh_dataset_catalog() -> None:
    """Refresh dataset catalog from backend and keep selection stable."""
    if not st.session_state.backend_available:
        st.session_state.dataset_catalog = []
        st.session_state.selected_dataset_detail = None
        return

    try:
        catalog = st.session_state.api_service.list_datasets()
        st.session_state.dataset_catalog = catalog
        dataset_names = [dataset["dataset_name"] for dataset in catalog]
        if dataset_names:
            if st.session_state.selected_dataset_name not in dataset_names:
                st.session_state.selected_dataset_name = dataset_names[0]
            st.session_state.selected_dataset_detail = st.session_state.api_service.get_dataset(
                st.session_state.selected_dataset_name
            )
        else:
            st.session_state.selected_dataset_name = None
            st.session_state.selected_dataset_detail = None
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")


def build_selected_head_payload(selected_heads: List[HeadPair], head_groups: List[HeadGroup]) -> List[Dict[str, int]]:
    """Serialize visible heads for backend filtering."""
    return [
        {"layer": head.layer, "head": head.head}
        for head in get_visible_head_pairs(selected_heads, head_groups)
    ]


def fetch_attention_data(force: bool = False) -> None:
    """Fetch sparse attention data from the backend for the current prompt."""
    if not st.session_state.backend_available or st.session_state.analysis_source != "Single prompt":
        return

    input_text = st.session_state.input_text.strip()
    if not input_text:
        return

    active_groups = get_active_head_groups(st.session_state.head_groups, st.session_state.active_group_ids)
    selected_head_payload = build_selected_head_payload(st.session_state.selected_heads, active_groups)
    request_signature = {
        "text": input_text,
        "model": st.session_state.current_model,
        "threshold": round(float(st.session_state.threshold), 4),
        "top_k": int(st.session_state.attention_top_k),
        "selected_heads": [(item["layer"], item["head"]) for item in selected_head_payload],
    }

    if not force:
        previous_signature = st.session_state.get("attention_request_signature")
        previous_text = st.session_state.get("last_processed_input_text")
        previous_model = st.session_state.get("last_processed_model")
        if (
            previous_signature == request_signature
            or previous_text != input_text
            or previous_model != st.session_state.current_model
        ):
            return

    st.session_state.loading = True
    try:
        st.session_state.attention_data = st.session_state.api_service.process_text(
            input_text,
            st.session_state.current_model,
            threshold=float(st.session_state.threshold),
            top_k=int(st.session_state.attention_top_k),
            selected_heads=selected_head_payload,
        )
        st.session_state.attention_request_signature = request_signature
        st.session_state.last_processed_input_text = input_text
        st.session_state.last_processed_model = st.session_state.current_model
    finally:
        st.session_state.loading = False

def fetch_max_logit_diff_graph(force: bool = False) -> None:
    """Fetch the causal graph payload for the current clean/corrupted prompt pair."""
    if not st.session_state.backend_available:
        return

    clean_text = st.session_state.input_text.strip()
    corrupted_text = st.session_state.corrupted_input_text.strip()
    if not clean_text or not corrupted_text:
        return

    active_groups = get_active_head_groups(st.session_state.head_groups, st.session_state.active_group_ids)
    selected_head_payload = build_selected_head_payload(st.session_state.selected_heads, active_groups)
    request_signature = {
        "text": clean_text,
        "corrupted_text": corrupted_text,
        "model": st.session_state.current_model,
        "target_token": st.session_state.target_token,
        "top_k": int(st.session_state.attention_top_k),
        "top_heads_per_layer": int(st.session_state.causal_top_heads_per_layer),
        "selected_heads": [(item["layer"], item["head"]) for item in selected_head_payload],
    }

    if not force and st.session_state.get("causal_graph_signature") == request_signature:
        return

    st.session_state.loading = True
    try:
        st.session_state.causal_graph_data = st.session_state.api_service.build_max_logit_diff_graph(
            text=clean_text,
            corrupted_text=corrupted_text,
            model=st.session_state.current_model,
            target_token=st.session_state.target_token,
            top_k=int(st.session_state.attention_top_k),
            top_heads_per_layer=int(st.session_state.causal_top_heads_per_layer),
            selected_heads=selected_head_payload,
        )
        st.session_state.causal_graph_signature = request_signature
    finally:
        st.session_state.loading = False

def causal_graph_matches_current_state() -> bool:
    """Check whether the cached causal graph matches the current controls."""
    signature = st.session_state.get("causal_graph_signature")
    if not signature:
        return False

    active_groups = get_active_head_groups(st.session_state.head_groups, st.session_state.active_group_ids)
    selected_head_payload = build_selected_head_payload(st.session_state.selected_heads, active_groups)
    current_signature = {
        "text": st.session_state.input_text.strip(),
        "corrupted_text": st.session_state.corrupted_input_text.strip(),
        "model": st.session_state.current_model,
        "target_token": st.session_state.target_token,
        "top_k": int(st.session_state.attention_top_k),
        "top_heads_per_layer": int(st.session_state.causal_top_heads_per_layer),
        "selected_heads": [(item["layer"], item["head"]) for item in selected_head_payload],
    }
    return signature == current_signature

def build_important_heads_frame(causal_graph_data: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Build a compact table of important heads across layers."""
    if not causal_graph_data:
        return pd.DataFrame()

    rows = [
        {
            "Layer": node["sourceLayer"],
            "Head": f"H{node['head']}",
            "Rank in layer": node.get("rankInLayer", 0) + 1,
            "Clean - corrupted": node["logit_diff_delta"],
            "Clean contribution": node["clean_logit_diff"],
            "Corrupted contribution": node["corrupted_logit_diff"],
        }
        for node in causal_graph_data.get("importantNodes", [])
    ]

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["Layer", "Rank in layer"]).reset_index(drop=True)

def main():
    st.set_page_config(layout="wide", page_title="Information Flow Visualization")

    # Initialize session state
    if 'api_service' not in st.session_state:
        st.session_state.api_service = APIService()
    if 'backend_available' not in st.session_state:
        st.session_state.backend_available = st.session_state.api_service.check_backend_health()
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "gpt2-small"
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 0.4
    if 'attention_top_k' not in st.session_state:
        st.session_state.attention_top_k = 12
    if 'causal_top_heads_per_layer' not in st.session_state:
        st.session_state.causal_top_heads_per_layer = 3
    if 'selected_heads' not in st.session_state:
        st.session_state.selected_heads = []
    if 'head_groups' not in st.session_state:
        st.session_state.head_groups = convert_predefined_groups_to_head_groups(st.session_state.current_model)
    if 'active_group_ids' not in st.session_state:
        st.session_state.active_group_ids = []
    if 'input_text' not in st.session_state:
        st.session_state.input_text = get_default_text_for_model(st.session_state.current_model)
    if 'attention_data' not in st.session_state:
        st.session_state.attention_data = None
    if 'attention_request_signature' not in st.session_state:
        st.session_state.attention_request_signature = None
    if 'last_processed_input_text' not in st.session_state:
        st.session_state.last_processed_input_text = ""
    if 'last_processed_model' not in st.session_state:
        st.session_state.last_processed_model = st.session_state.current_model
    if 'loading' not in st.session_state:
        st.session_state.loading = False
    if 'color_change_group' not in st.session_state:
        st.session_state.color_change_group = None
    if 'curve_type' not in st.session_state:
        st.session_state.curve_type = "cubic"
    if 'comparison_view' not in st.session_state:
        st.session_state.comparison_view = "Baseline"
    if 'inspector_head_key' not in st.session_state:
        st.session_state.inspector_head_key = None
    if 'target_token' not in st.session_state:
        st.session_state.target_token = ""
    if 'selected_task_preset' not in st.session_state:
        st.session_state.selected_task_preset = "Custom"
    if 'evaluation_result' not in st.session_state:
        st.session_state.evaluation_result = None
    if 'dataset_evaluation_result' not in st.session_state:
        st.session_state.dataset_evaluation_result = None
    if 'ablation_mode' not in st.session_state:
        st.session_state.ablation_mode = "Selected heads"
    if 'selected_group_for_ablation' not in st.session_state:
        st.session_state.selected_group_for_ablation = None
    if 'selected_layer_for_ablation' not in st.session_state:
        st.session_state.selected_layer_for_ablation = 0
    if 'new_candidate_group_name' not in st.session_state:
        st.session_state.new_candidate_group_name = ""
    if 'analysis_source' not in st.session_state:
        st.session_state.analysis_source = "Single prompt"
    if 'dataset_catalog' not in st.session_state:
        st.session_state.dataset_catalog = []
    if 'selected_dataset_name' not in st.session_state:
        st.session_state.selected_dataset_name = None
    if 'selected_dataset_detail' not in st.session_state:
        st.session_state.selected_dataset_detail = None
    if 'dataset_validation_result' not in st.session_state:
        st.session_state.dataset_validation_result = None
    if 'corrupted_input_text' not in st.session_state:
        st.session_state.corrupted_input_text = ""
    if 'graph_view_mode' not in st.session_state:
        st.session_state.graph_view_mode = "Attention weights"
    if 'causal_graph_data' not in st.session_state:
        st.session_state.causal_graph_data = None
    if 'causal_graph_signature' not in st.session_state:
        st.session_state.causal_graph_signature = None

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1.2rem 2rem;
            background: linear-gradient(180deg, #f6f1e8 0%, #fbfaf6 38%, #f4f6f1 100%);
        }
        .stTitle {
            font-size: 2.4rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.03em;
        }
        .stSelectbox > div > div {
            background-color: rgba(255, 252, 246, 0.92);
            border: 1px solid #d8d4cb;
        }
        .head-button {
            display: inline-block;
            padding: 2px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 0.8rem;
            color: white;
        }
        .compact-selection-card {
            background: rgba(243, 239, 228, 0.72);
            border: 1px solid rgba(95, 86, 66, 0.12);
            border-radius: 14px;
            padding: 0.8rem;
            margin-top: 0.65rem;
        }
        .compact-selection-meta {
            color: #1f2d2d;
            font-size: 0.9rem;
            margin-bottom: 0.55rem;
        }
        .selection-summary-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-bottom: 0.65rem;
        }
        .selection-summary-chip,
        .selection-overflow-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            background: rgba(255, 253, 249, 0.92);
            color: #6a6255;
            font-size: 0.76rem;
            border: 1px solid rgba(95, 86, 66, 0.1);
        }
        .compact-selection-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.25rem;
            max-height: 10rem;
            overflow-y: auto;
            padding-right: 0.2rem;
        }
        .compact-head-chip {
            margin: 0;
            font-size: 0.72rem;
            padding: 0.18rem 0.48rem;
        }
        .workspace-shell {
            background: rgba(255, 251, 245, 0.88);
            border: 1px solid rgba(95, 86, 66, 0.12);
            border-radius: 24px;
            padding: 1.2rem 1.2rem 0.6rem;
            box-shadow: 0 18px 50px rgba(89, 74, 45, 0.08);
            backdrop-filter: blur(10px);
        }
        .hero-band {
            padding: 1.4rem 1.5rem;
            border-radius: 22px;
            background:
                radial-gradient(circle at top right, rgba(188, 111, 60, 0.16), transparent 28%),
                linear-gradient(135deg, #1f2d2d 0%, #31493c 48%, #6d7f44 100%);
            color: #f9f4ec;
            margin-bottom: 1rem;
        }
        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            opacity: 0.72;
            margin-bottom: 0.65rem;
        }
        .hero-title {
            font-size: 2rem;
            line-height: 1;
            font-weight: 600;
            letter-spacing: -0.04em;
            margin-bottom: 0.55rem;
        }
        .hero-copy {
            max-width: 48rem;
            font-size: 0.98rem;
            line-height: 1.6;
            color: rgba(249, 244, 236, 0.86);
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.8rem 0 1rem;
        }
        .metric-cell {
            background: rgba(255, 253, 249, 0.92);
            border: 1px solid rgba(95, 86, 66, 0.12);
            border-radius: 18px;
            padding: 0.9rem 1rem;
        }
        .metric-label {
            color: #6a6255;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: #1f2d2d;
            font-size: 1.55rem;
            font-weight: 600;
            letter-spacing: -0.03em;
        }
        .panel {
            background: rgba(255, 253, 249, 0.9);
            border: 1px solid rgba(95, 86, 66, 0.12);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            min-height: 100%;
        }
        .panel-title {
            color: #1f2d2d;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            margin-bottom: 0.35rem;
        }
        .panel-copy {
            color: #6a6255;
            font-size: 0.9rem;
            line-height: 1.55;
            margin-bottom: 0.8rem;
        }
        .signal-list {
            display: grid;
            gap: 0.55rem;
        }
        .signal-row {
            background: rgba(243, 239, 228, 0.76);
            border-radius: 14px;
            padding: 0.7rem 0.8rem;
        }
        .signal-route {
            color: #1f2d2d;
            font-size: 0.92rem;
            font-weight: 600;
        }
        .signal-meta {
            color: #6a6255;
            font-size: 0.83rem;
            margin-top: 0.2rem;
        }
        .group-container {
            background-color: rgba(255, 253, 249, 0.95);
            padding: 1rem;
            border-radius: 14px;
            margin-bottom: 1rem;
            border: 1px solid rgba(95, 86, 66, 0.12);
        }
        .group-description {
            color: #6b7280;
            font-size: 0.875rem;
            margin: 0.5rem 0;
        }
        .graph-container {
            background: rgba(255, 253, 249, 0.94);
            border: 1px solid rgba(95, 86, 66, 0.12);
            border-radius: 24px;
            padding: 1rem 1rem 0.6rem;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(89, 74, 45, 0.05);
        }
        .controls-container {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 8px;
        }
        .controls-container::-webkit-scrollbar {
            width: 6px;
        }
        .controls-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        .controls-container::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 3px;
        }
        .controls-container::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        /* Style for the color picker button */
        div[data-testid="stHorizontalBlock"] > div:last-child button[data-testid="baseButton-primary"] {
            background: transparent;
            border: none;
            padding: 0;
            margin: 0;
            font-size: 1.2rem;
            cursor: pointer;
            color: #666;
            transition: color 0.2s;
        }
        div[data-testid="stHorizontalBlock"] > div:last-child button[data-testid="baseButton-primary"]:hover {
            color: #000;
        }
        /* Style for primary buttons (Add and Create) */
        button[data-testid="baseButton-primary"]:not(div[data-testid="stHorizontalBlock"] > div:last-child button) {
            background: #FF4B4B !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            margin: 0 !important;
            font-size: 0.875rem !important;
            cursor: pointer !important;
            color: white !important;
            border-radius: 0.25rem !important;
            transition: background-color 0.2s !important;
        }
        button[data-testid="baseButton-primary"]:not(div[data-testid="stHorizontalBlock"] > div:last-child button):hover {
            background: #FF3333 !important;
        }
        /* Additional style for the Add button */
        div[data-testid="stHorizontalBlock"] > div:first-child button[data-testid="baseButton-primary"] {
            background: #FF4B4B !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            margin: 0 !important;
            font-size: 0.875rem !important;
            cursor: pointer !important;
            color: white !important;
            border-radius: 0.25rem !important;
            transition: background-color 0.2s !important;
        }
        div[data-testid="stHorizontalBlock"] > div:first-child button[data-testid="baseButton-primary"]:hover {
            background: #FF3333 !important;
        }
        @media (max-width: 900px) {
            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Information Flow Visualization")
    st.markdown(
        """
        <div class="workspace-shell">
            <div class="hero-band">
                <div class="hero-kicker">Circuit Workbench</div>
                <div class="hero-title">Trace hypotheses, inspect head groups, and prepare causal experiments.</div>
            </div>
        """,
        unsafe_allow_html=True
    )

    # Model selection
    col1, _ = st.columns([1, 3])
    with col1:
        selected_model = st.selectbox(
            "Model",
            ["gpt2-small", "pythia-2.8b"],
            index=0 if st.session_state.current_model == "gpt2-small" else 1
        )
        
        # Handle model change
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            st.session_state.head_groups = convert_predefined_groups_to_head_groups(selected_model)
            st.session_state.active_group_ids = []
            st.session_state.input_text = get_default_text_for_model(selected_model)
            st.session_state.corrupted_input_text = ""
            st.session_state.attention_data = None
            st.session_state.evaluation_result = None
            st.session_state.dataset_evaluation_result = None
            st.session_state.causal_graph_data = None
            st.session_state.causal_graph_signature = None
            st.session_state.selected_task_preset = "Custom"
            presets = get_task_presets_for_model(selected_model)
            st.session_state.target_token = presets[0]["target_token"] if presets else ""


    # Backend status and sample data loading
    if not st.session_state.backend_available:
        st.warning("Backend is not available. Using sample data.")
        if st.session_state.attention_data is None:
            st.session_state.attention_data = load_sample_data(st.session_state.current_model)
            if st.session_state.attention_data is None:
                st.error("Failed to load sample data. Please try again later.")
                return

    if st.session_state.backend_available and not st.session_state.dataset_catalog:
        refresh_dataset_catalog()

    preset_definitions = get_task_presets_for_model(st.session_state.current_model)
    preset_names = ["Custom"] + [preset["name"] for preset in preset_definitions]
    if st.session_state.selected_task_preset not in preset_names:
        st.session_state.selected_task_preset = "Custom"
    if st.session_state.comparison_view not in ["Baseline", "Ablated", "Diff"]:
        st.session_state.comparison_view = "Baseline"

    task_col, compare_col = st.columns([1.65, 1])
    with task_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Task Context</div>', unsafe_allow_html=True)
        st.session_state.analysis_source = st.radio(
            "Analysis Source",
            ["Single prompt", "Dataset"],
            index=["Single prompt", "Dataset"].index(st.session_state.analysis_source),
            horizontal=True,
        )
        if st.session_state.analysis_source == "Single prompt":
            st.markdown(
                '<div class="panel-copy">Start from a benchmark-style prompt, then define the target token whose probability should rise or fall when you ablate candidate heads.</div>',
                unsafe_allow_html=True
            )
            selected_preset = st.selectbox(
                "Task Preset",
                options=preset_names,
                index=preset_names.index(st.session_state.selected_task_preset)
            )
            if selected_preset != st.session_state.selected_task_preset:
                st.session_state.selected_task_preset = selected_preset
                st.session_state.evaluation_result = None
                if selected_preset != "Custom":
                    preset = next(preset for preset in preset_definitions if preset["name"] == selected_preset)
                    st.session_state.input_text = preset["text"]
                    st.session_state.target_token = preset.get("target_token", "")

            if st.session_state.selected_task_preset != "Custom":
                active_preset = next(
                    preset for preset in preset_definitions
                    if preset["name"] == st.session_state.selected_task_preset
                )
                st.caption(active_preset.get("description", ""))

            st.session_state.target_token = st.text_input(
                "Target Token",
                value=st.session_state.target_token,
                placeholder="e.g. Mary, Paris, gate"
            )
            st.session_state.corrupted_input_text = st.text_area(
                "Corrupted Prompt (for clean-corrupted graph variant)",
                value=st.session_state.corrupted_input_text,
                height=120,
                placeholder="Optional, but required for the max clean-corrupted logit diff graph."
            )
        else:
            st.markdown(
                '<div class="panel-copy">Select a reusable benchmark set or upload a custom dataset for aggregate metrics and batch causal interventions.</div>',
                unsafe_allow_html=True
            )
            if st.session_state.backend_available:
                refresh_col1, refresh_col2 = st.columns([3, 1])
                with refresh_col1:
                    dataset_options = [
                        f"{dataset['name']} ({dataset['storage']})"
                        for dataset in st.session_state.dataset_catalog
                        if dataset.get("model") in (None, st.session_state.current_model)
                    ]
                    filtered_datasets = [
                        dataset for dataset in st.session_state.dataset_catalog
                        if dataset.get("model") in (None, st.session_state.current_model)
                    ]
                    if filtered_datasets:
                        selected_option = st.selectbox(
                            "Dataset",
                            options=dataset_options,
                            index=next(
                                (
                                    idx for idx, dataset in enumerate(filtered_datasets)
                                    if dataset["dataset_name"] == st.session_state.selected_dataset_name
                                ),
                                0,
                            ),
                        )
                        selected_dataset = filtered_datasets[dataset_options.index(selected_option)]
                        if selected_dataset["dataset_name"] != st.session_state.selected_dataset_name:
                            st.session_state.selected_dataset_name = selected_dataset["dataset_name"]
                            st.session_state.selected_dataset_detail = st.session_state.api_service.get_dataset(
                                st.session_state.selected_dataset_name
                            )
                            st.session_state.dataset_evaluation_result = None
                    else:
                        st.info("No datasets available for the current model yet.")
                with refresh_col2:
                    if st.button("Refresh", key="refresh_datasets"):
                        refresh_dataset_catalog()

                dataset_detail = st.session_state.selected_dataset_detail
                if dataset_detail:
                    st.caption(dataset_detail.get("description", ""))
                    st.caption(
                        f"{len(dataset_detail.get('examples', []))} examples • "
                        f"metric: {dataset_detail.get('metric', 'target_probability')} • "
                        f"causal: {'yes' if all(example.get('corrupted_text') for example in dataset_detail.get('examples', [])) else 'no'}"
                    )
                    preview_frame = pd.DataFrame(dataset_detail.get("examples", [])[:5])
                    if not preview_frame.empty:
                        st.dataframe(preview_frame, use_container_width=True, hide_index=True)
                    if st.button("Load first example into prompt workspace", key="load_dataset_example"):
                        first_example = dataset_detail["examples"][0]
                        st.session_state.input_text = first_example["text"]
                        st.session_state.target_token = first_example.get("target_token", "")
                        st.session_state.corrupted_input_text = first_example.get("corrupted_text", "")
            else:
                st.info("Dataset mode requires a live backend.")
        st.markdown('</div>', unsafe_allow_html=True)

    with compare_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Comparison View</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Switch between baseline and ablated analysis once you run a causal evaluation.</div>',
            unsafe_allow_html=True
        )
        st.session_state.comparison_view = st.radio(
            "View",
            ["Baseline", "Ablated", "Diff"],
            index=["Baseline", "Ablated", "Diff"].index(st.session_state.comparison_view),
            label_visibility="collapsed"
        )
        if st.session_state.evaluation_result is None:
            st.caption("Run Causal Compare to populate baseline vs ablated behavior metrics.")
        else:
            st.caption("Use Diff to inspect probability, loss, and top-prediction changes after ablation.")
        st.markdown('</div>', unsafe_allow_html=True)

    metrics = summarize_attention_metrics(
        st.session_state.attention_data,
        st.session_state.threshold,
        st.session_state.selected_heads,
        get_active_head_groups(st.session_state.head_groups, st.session_state.active_group_ids)
    )
    evaluation_result = st.session_state.evaluation_result
    baseline_metrics = evaluation_result.get("baseline") if evaluation_result else None
    ablated_metrics = evaluation_result.get("ablated") if evaluation_result else None
    diff_metrics = evaluation_result.get("delta") if evaluation_result else None

    if st.session_state.comparison_view == "Ablated" and ablated_metrics:
        metric_label = "Target Probability"
        metric_value = format_metric_value(ablated_metrics.get("target_probability"))
    elif st.session_state.comparison_view == "Diff" and diff_metrics:
        metric_label = "Target Prob Delta"
        metric_value = format_metric_value(diff_metrics.get("target_probability_delta"))
    else:
        metric_label = "Target Probability"
        metric_value = format_metric_value(baseline_metrics.get("target_probability")) if baseline_metrics else "n/a"

    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-cell">
                <div class="metric-label">Model</div>
                <div class="metric-value">{st.session_state.current_model}</div>
            </div>
            <div class="metric-cell">
                <div class="metric-label">Visible Heads</div>
                <div class="metric-value">{metrics["visible_heads"]}</div>
            </div>
            <div class="metric-cell">
                <div class="metric-label">Visible Edges</div>
                <div class="metric-value">{metrics["visible_edges"]}</div>
            </div>
            <div class="metric-cell">
                <div class="metric-label">{metric_label}</div>
                <div class="metric-value">{metric_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.analysis_source == "Dataset":
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Upload Custom Dataset</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Upload a JSON or CSV dataset, validate the schema, preview the examples, and save it into the local dataset catalog for repeated experiments.</div>',
            unsafe_allow_html=True
        )
        uploaded_dataset = st.file_uploader(
            "Dataset file",
            type=["json", "csv"],
            accept_multiple_files=False,
            key="dataset_uploader",
        )
        upload_col1, upload_col2 = st.columns([1, 1])
        if uploaded_dataset is not None:
            if upload_col1.button("Validate Dataset", key="validate_dataset_button"):
                try:
                    payload = parse_uploaded_dataset(uploaded_dataset)
                    st.session_state.dataset_validation_result = {
                        "payload": payload,
                        "result": st.session_state.api_service.validate_dataset(payload)
                        if st.session_state.backend_available else None,
                    }
                except Exception as e:
                    st.error(f"Error validating dataset: {str(e)}")

            validation_bundle = st.session_state.dataset_validation_result
            if validation_bundle and validation_bundle.get("payload"):
                payload = validation_bundle["payload"]
                validation = validation_bundle.get("result")
                if validation:
                    if validation["valid"]:
                        st.success(
                            f"Valid dataset: {validation['valid_examples']}/{validation['num_examples']} examples ready."
                        )
                    else:
                        st.error("Dataset validation failed.")
                    for error in validation.get("errors", []):
                        st.caption(f"Error: {error}")
                    for warning in validation.get("warnings", []):
                        st.caption(f"Warning: {warning}")

                preview_frame = pd.DataFrame(payload.get("examples", [])[:5])
                if not preview_frame.empty:
                    st.dataframe(preview_frame, use_container_width=True, hide_index=True)

                if upload_col2.button("Save Dataset", key="save_dataset_button", disabled=not validation or not validation["valid"]):
                    try:
                        save_result = st.session_state.api_service.save_dataset(payload)
                        st.success(f"Saved dataset '{save_result['dataset_name']}'")
                        refresh_dataset_catalog()
                        st.session_state.selected_dataset_name = save_result["dataset_name"]
                        st.session_state.selected_dataset_detail = save_result["dataset"]
                    except Exception as e:
                        st.error(f"Error saving dataset: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Controls section
    with st.expander("Controls", expanded=True):
        st.markdown('<div class="controls-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Head Groups")
            st.caption("Use the circuit library as hypotheses, and save promising current selections as candidate groups.")

            predefined_groups = [group for group in st.session_state.head_groups if group.group_type == "predefined"]
            custom_groups = [group for group in st.session_state.head_groups if group.group_type == "custom"]

            st.markdown("**Circuit Library**")
            for group in predefined_groups:
                is_active = group.id in st.session_state.active_group_ids
                group_container = st.container()
                with group_container:
                    group_col1, group_col2, group_col3, group_col4 = st.columns([4.7, 1.2, 1.2, 0.8])
                    with group_col1:
                        st.markdown(f"""
                            <div class="group-container">
                                <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                                    <div style="font-weight: 600;">{group.name}</div>
                                </div>
                                <div class="group-description">{group.description or ''}</div>
                                <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                                    {''.join([f'<span class="head-button" style="background-color: {group.color or COLOR_PALETTE[group.id % len(COLOR_PALETTE)]}">{head.layer},{head.head}</span>' for head in group.heads])}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    with group_col2:
                        button_label = "Hide" if is_active else "Show"
                        if st.button(button_label, key=f"toggle_group_{group.id}", type="primary" if not is_active else "secondary"):
                            if is_active:
                                st.session_state.active_group_ids = [active_id for active_id in st.session_state.active_group_ids if active_id != group.id]
                            else:
                                st.session_state.active_group_ids = st.session_state.active_group_ids + [group.id]
                    with group_col3:
                        if st.button("Use heads", key=f"use_group_heads_{group.id}"):
                            st.session_state.selected_heads = append_unique_heads(st.session_state.selected_heads, group.heads)
                    with group_col4:
                        if st.button("🎨", key=f"color_{group.id}", help="Change group color"):
                            group.color = get_random_color()

            st.markdown("**Candidate Groups**")
            st.session_state.new_candidate_group_name = st.text_input(
                "Save current selection as group",
                value=st.session_state.new_candidate_group_name,
                placeholder="e.g. IOI candidate circuit"
            )
            create_candidate_disabled = len(st.session_state.selected_heads) == 0
            if st.button("Save Selection", type="primary", disabled=create_candidate_disabled):
                if st.session_state.new_candidate_group_name.strip():
                    new_group_obj = HeadGroup(
                        id=next_group_id(st.session_state.head_groups),
                        name=st.session_state.new_candidate_group_name.strip(),
                        heads=get_unique_head_pairs(st.session_state.selected_heads),
                        color=get_random_color(),
                        description="Saved from current selected heads.",
                        group_type="custom",
                    )
                    st.session_state.head_groups.append(new_group_obj)
                    if new_group_obj.id not in st.session_state.active_group_ids:
                        st.session_state.active_group_ids = st.session_state.active_group_ids + [new_group_obj.id]
                    st.session_state.new_candidate_group_name = ""
                else:
                    st.error("Group name cannot be empty")

            if custom_groups:
                for group in custom_groups:
                    is_active = group.id in st.session_state.active_group_ids
                    group_container = st.container()
                    with group_container:
                        group_col1, group_col2, group_col3, group_col4, group_col5 = st.columns([4.4, 1.1, 1.1, 0.8, 0.8])
                        with group_col1:
                            st.markdown(f"""
                                <div class="group-container">
                                    <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                                        <div style="font-weight: 600;">{group.name}</div>
                                    </div>
                                    <div class="group-description">{group.description or ''}</div>
                                    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                                        {''.join([f'<span class="head-button" style="background-color: {group.color or COLOR_PALETTE[group.id % len(COLOR_PALETTE)]}">{head.layer},{head.head}</span>' for head in group.heads])}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        with group_col2:
                            button_label = "Hide" if is_active else "Show"
                            if st.button(button_label, key=f"toggle_custom_group_{group.id}", type="primary" if not is_active else "secondary"):
                                if is_active:
                                    st.session_state.active_group_ids = [active_id for active_id in st.session_state.active_group_ids if active_id != group.id]
                                else:
                                    st.session_state.active_group_ids = st.session_state.active_group_ids + [group.id]
                        with group_col3:
                            if st.button("Refresh", key=f"refresh_group_{group.id}"):
                                group.heads = get_unique_head_pairs(st.session_state.selected_heads)
                        with group_col4:
                            if st.button("🎨", key=f"color_custom_{group.id}", help="Change group color"):
                                group.color = get_random_color()
                        with group_col5:
                            if st.button("×", key=f"remove_group_{group.id}", help="Remove group"):
                                st.session_state.head_groups = [g for g in st.session_state.head_groups if g.id != group.id]
                                st.session_state.active_group_ids = [active_id for active_id in st.session_state.active_group_ids if active_id != group.id]
            else:
                st.info("Save selected heads as a candidate group once you find a promising route.")

        with col2:
            st.subheader("Individual Heads")            
            # Initialize the input state if not exists
            if 'head_input_value' not in st.session_state:
                st.session_state.head_input_value = ""
                
            head_input = st.text_input(
                "Add head",
                placeholder="layer,head or layer,: or :,head or :,:",
                help="Valid: Layer (0-32), Head (0-31). Press Enter to add.",
                key="head_input",
                value=st.session_state.head_input_value
            )
            
            def add_head(head_input: str) -> None:
                if not head_input.strip():
                    return
                    
                try:
                    num_layers, num_heads = get_model_dimensions(
                        st.session_state.current_model,
                        st.session_state.attention_data,
                    )
                    if num_layers <= 0 or num_heads <= 0:
                        st.error("Could not determine model dimensions for head selection.")
                        return

                    layer_str, head_str = head_input.split(',')
                    layer_str = layer_str.strip()
                    head_str = head_str.strip()
                    
                    if layer_str == ':' and head_str == ':':
                        for l in range(num_layers):
                            for h in range(num_heads):
                                # Generate a consistent grey color for wildcard heads
                                grey_value = 128 + ((l * num_heads + h) * 20) % 96
                                grey_hex = f"#{grey_value:02x}{grey_value:02x}{grey_value:02x}"
                                st.session_state.selected_heads.append(HeadPair(layer=l, head=h, color=grey_hex))
                    elif layer_str == ':':
                        head = int(head_str)
                        if not 0 <= head < num_heads:
                            st.error(f"Invalid head number. Must be between 0 and {num_heads-1}")
                            return
                        for l in range(num_layers):
                            st.session_state.selected_heads.append(HeadPair(layer=l, head=head, color=COLOR_PALETTE[head % len(COLOR_PALETTE)]))
                    elif head_str == ':':
                        layer = int(layer_str)
                        if not 0 <= layer < num_layers:
                            st.error(f"Invalid layer number. Must be between 0 and {num_layers-1}")
                            return
                        for h in range(num_heads):
                            st.session_state.selected_heads.append(HeadPair(layer=layer, head=h, color=COLOR_PALETTE[h % len(COLOR_PALETTE)]))
                    else:
                        layer = int(layer_str)
                        head = int(head_str)
                        if not 0 <= layer < num_layers:
                            st.error(f"Invalid layer number. Must be between 0 and {num_layers-1}")
                            return
                        if not 0 <= head < num_heads:
                            st.error(f"Invalid head number. Must be between 0 and {num_heads-1}")
                            return
                        st.session_state.selected_heads.append(HeadPair(layer=layer, head=head, color=COLOR_PALETTE[head % len(COLOR_PALETTE)]))
                    
                    # Clear the input field after successful addition
                    st.session_state.head_input_value = ""
                except ValueError as e:
                    st.error(f"Invalid input format. Please use numbers or ':' for wildcards (e.g., '1,2' or '1,:')")
                except Exception as e:
                    st.error(f"Error adding head: {str(e)}")

            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Add", type="primary"):
                    add_head(head_input)
            
            # Handle Enter key press
            if head_input and head_input.endswith('\n'):
                add_head(head_input)

            # Display selected heads
            if st.session_state.selected_heads:
                compact_mode = len(st.session_state.selected_heads) > 24
                if compact_mode:
                    action_col1, action_col2 = st.columns([1, 4])
                    with action_col1:
                        if st.button("Clear all", key="clear_all_heads"):
                            st.session_state.selected_heads = []
                    with action_col2:
                        st.caption("Compact mode is enabled automatically for large selections.")
                    st.markdown(
                        render_selected_heads_compact(st.session_state.selected_heads),
                        unsafe_allow_html=True
                    )
                else:
                    for head in st.session_state.selected_heads:
                        col1, col2, col3 = st.columns([5, 1, 1])
                        with col1:
                            st.markdown(f"""
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <span class="head-button" style="background-color: {head.color or '#3B82F6'}">
                                        {head.layer},{head.head}
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            if st.button("🎨", key=f"color_head_{head.layer}_{head.head}", help="Change head color"):
                                head.color = get_random_color()

                        with col3:
                            if st.button("×", key=f"remove_{head.layer}_{head.head}", help="Remove head"):
                                st.session_state.selected_heads = [h for h in st.session_state.selected_heads 
                                                                 if not (h.layer == head.layer and h.head == head.head)]
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close controls container

    # Threshold control
    st.markdown("<div style='margin: 1rem 0;'>", unsafe_allow_html=True)
    st.session_state.threshold = st.slider(
        "Edge Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.threshold,
        step=0.01
    )
    st.session_state.attention_top_k = st.number_input(
        "Max Edges per Head",
        min_value=1,
        max_value=100,
        value=st.session_state.attention_top_k,
        step=1,
        help="The backend keeps only the strongest edges per visible head after applying the weight threshold."
    )
    
    # Add curve type selector
    curve_types = {
        "Cubic Bezier": "cubic",
        "Quadratic Bezier": "quadratic",
        "Linear": "linear",
        "Spline": "spline"
    }
    
    # Create reverse mapping from internal values to display names
    curve_type_to_display = {v: k for k, v in curve_types.items()}
    
    # Get the current display name using the reverse mapping
    current_display = curve_type_to_display.get(st.session_state.curve_type, "Cubic Bezier")
    
    selected_display = st.selectbox(
        "Curve Type",
        options=list(curve_types.keys()),
        index=list(curve_types.keys()).index(current_display)
    )
    st.session_state.curve_type = curve_types[selected_display]

    graph_view_modes = ["Attention weights", "Max clean-corrupted logit diff"]
    st.session_state.graph_view_mode = st.selectbox(
        "Graph Variant",
        options=graph_view_modes,
        index=graph_view_modes.index(st.session_state.graph_view_mode)
        if st.session_state.graph_view_mode in graph_view_modes else 0,
        help="The causal variant picks the head with the largest clean-corrupted logit-diff contribution at one layer and token position."
    )

    if st.session_state.graph_view_mode == "Max clean-corrupted logit diff":
        build_disabled = (
            not st.session_state.backend_available
            or not st.session_state.input_text.strip()
            or not st.session_state.corrupted_input_text.strip()
        )
        target_position = max(((st.session_state.attention_data or {}).get("numTokens", 1)) - 1, 0)
        st.session_state.causal_top_heads_per_layer = st.number_input(
            "Important heads per layer",
            min_value=1,
            max_value=8,
            value=int(st.session_state.causal_top_heads_per_layer),
            step=1,
        )
        st.caption(
            f"This variant scans all layers at the final visible position ({target_position}) and keeps the strongest heads in each layer."
        )
        if st.button("Build Clean-Corrupted Graph", disabled=build_disabled):
            try:
                fetch_max_logit_diff_graph(force=True)
            except Exception as e:
                st.error(f"Error building clean-corrupted graph: {str(e)}")
        if not st.session_state.backend_available:
            st.caption("This graph variant requires a live backend.")
        elif not st.session_state.corrupted_input_text.strip():
            st.caption("Add a corrupted prompt to enable this graph variant.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Text input and processing
    if st.session_state.analysis_source == "Single prompt" and st.session_state.backend_available:
        st.text_input(
            "Input Text",
            key="input_text"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Process Text", type="primary"):
                try:
                    fetch_attention_data(force=True)
                    st.session_state.evaluation_result = None
                    st.session_state.causal_graph_data = None
                    st.session_state.causal_graph_signature = None
                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
    elif st.session_state.analysis_source == "Single prompt":
        st.text_input(
            "Input Text",
            key="input_text",
            disabled=True,
            help="Text input is disabled when using sample data. Please start the backend server to enable text processing."
        )

    if (
        st.session_state.analysis_source == "Single prompt"
        and st.session_state.backend_available
        and st.session_state.attention_data is not None
    ):
        try:
            fetch_attention_data(force=False)
        except Exception as e:
            st.error(f"Error refreshing attention data: {str(e)}")

    insight_col, intervention_col = st.columns([1.45, 1])
    with insight_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Signal Routes</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">These are the strongest currently visible edges after your threshold and head filters. They help you decide what subgraph deserves the next causal test.</div>',
            unsafe_allow_html=True
        )
        if metrics["top_edges"]:
            tokens = (st.session_state.attention_data or {}).get("tokens", [])
            st.markdown(render_signal_route_rows(metrics["top_edges"][:5], tokens), unsafe_allow_html=True)
        else:
            st.info("No visible edges yet. Lower the threshold or add a head/group to start tracing routes.")
        st.markdown('</div>', unsafe_allow_html=True)
    with intervention_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Intervention Workspace</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Choose which heads to ablate, then compare baseline and ablated behavior on the selected target token.</div>',
            unsafe_allow_html=True
        )
        intervention_target = st.selectbox(
            "Intervention Target",
            ["Selected heads", "Head group", "All heads in layer"],
            index=["Selected heads", "Head group", "All heads in layer"].index(st.session_state.ablation_mode)
        )
        st.session_state.ablation_mode = intervention_target

        ablation_group_name = None
        ablation_layer = None
        if intervention_target == "Head group":
            group_names = [group.name for group in st.session_state.head_groups]
            if group_names:
                if st.session_state.selected_group_for_ablation not in group_names:
                    st.session_state.selected_group_for_ablation = group_names[0]
                ablation_group_name = st.selectbox(
                    "Group",
                    options=group_names,
                    index=group_names.index(st.session_state.selected_group_for_ablation)
                )
                st.session_state.selected_group_for_ablation = ablation_group_name
            else:
                st.info("No head groups are available yet.")
        elif intervention_target == "All heads in layer":
            model_layers, _ = get_model_dimensions(
                st.session_state.current_model,
                st.session_state.attention_data,
            )
            layer_options = list(range(model_layers))
            if layer_options:
                if st.session_state.selected_layer_for_ablation not in layer_options:
                    st.session_state.selected_layer_for_ablation = layer_options[0]
                ablation_layer = st.selectbox(
                    "Layer",
                    options=layer_options,
                    index=layer_options.index(st.session_state.selected_layer_for_ablation)
                )
                st.session_state.selected_layer_for_ablation = ablation_layer

        ablation_heads = get_ablation_heads(
            intervention_target,
            st.session_state.selected_heads,
            get_active_head_groups(st.session_state.head_groups, st.session_state.active_group_ids),
            ablation_group_name,
            ablation_layer,
            st.session_state.attention_data,
        )
        st.caption(f"Ablation set size: {len(ablation_heads)} heads")

        compare_disabled = (
            (not st.session_state.backend_available)
            or (len(ablation_heads) == 0)
            or (st.session_state.analysis_source == "Dataset" and not st.session_state.selected_dataset_name)
        )
        if st.button("Run Causal Compare", type="primary", disabled=compare_disabled):
            try:
                if st.session_state.analysis_source == "Dataset":
                    st.session_state.dataset_evaluation_result = st.session_state.api_service.evaluate_dataset(
                        dataset_name=st.session_state.selected_dataset_name,
                        model=st.session_state.current_model,
                        ablated_heads=[{"layer": head.layer, "head": head.head} for head in ablation_heads],
                    )
                else:
                    st.session_state.evaluation_result = st.session_state.api_service.evaluate_text(
                        text=st.session_state.input_text,
                        model=st.session_state.current_model,
                        target_token=st.session_state.target_token,
                        ablated_heads=[{"layer": head.layer, "head": head.head} for head in ablation_heads],
                    )
                    if st.session_state.attention_data is None:
                        fetch_attention_data(force=True)
            except Exception as e:
                st.error(f"Error running causal compare: {str(e)}")

        if not st.session_state.backend_available:
            st.caption("Causal compare requires a live backend.")
        elif st.session_state.analysis_source == "Dataset" and not st.session_state.selected_dataset_name:
            st.caption("Choose a dataset before running batch causal compare.")
        elif len(ablation_heads) == 0:
            st.caption("Select at least one head, group, or layer before running ablation.")
        st.markdown('</div>', unsafe_allow_html=True)

    if evaluation_result:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Causal Comparison</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Baseline and ablated metrics for the current prompt. Use the delta values to decide whether the chosen heads are plausibly causal for the target behavior.</div>',
            unsafe_allow_html=True
        )
        comparison_rows = [
            {
                "Metric": "Top prediction",
                "Baseline": baseline_metrics.get("top_prediction") if baseline_metrics else "n/a",
                "Ablated": ablated_metrics.get("top_prediction") if ablated_metrics else "n/a",
                "Diff": "changed" if diff_metrics and diff_metrics.get("top_prediction_changed") else "unchanged",
            },
            {
                "Metric": "Target probability",
                "Baseline": format_metric_value(baseline_metrics.get("target_probability")) if baseline_metrics else "n/a",
                "Ablated": format_metric_value(ablated_metrics.get("target_probability")) if ablated_metrics else "n/a",
                "Diff": format_metric_value(diff_metrics.get("target_probability_delta")) if diff_metrics else "n/a",
            },
            {
                "Metric": "Logit diff",
                "Baseline": format_metric_value(baseline_metrics.get("logit_diff")) if baseline_metrics else "n/a",
                "Ablated": format_metric_value(ablated_metrics.get("logit_diff")) if ablated_metrics else "n/a",
                "Diff": format_metric_value(diff_metrics.get("logit_diff_delta")) if diff_metrics else "n/a",
            },
            {
                "Metric": "Correct token rank",
                "Baseline": format_int_metric(baseline_metrics.get("target_rank")) if baseline_metrics else "n/a",
                "Ablated": format_int_metric(ablated_metrics.get("target_rank")) if ablated_metrics else "n/a",
                "Diff": format_int_metric(diff_metrics.get("target_rank_delta")) if diff_metrics else "n/a",
            },
            {
                "Metric": "Loss",
                "Baseline": format_metric_value(baseline_metrics.get("loss")) if baseline_metrics else "n/a",
                "Ablated": format_metric_value(ablated_metrics.get("loss")) if ablated_metrics else "n/a",
                "Diff": format_metric_value(diff_metrics.get("loss_delta")) if diff_metrics else "n/a",
            },
        ]
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        top_tokens = (
            baseline_metrics.get("top_tokens", [])
            if st.session_state.comparison_view == "Baseline" or not ablated_metrics
            else ablated_metrics.get("top_tokens", [])
        )
        if top_tokens:
            st.caption("Top continuation candidates in the active comparison view")
            st.dataframe(pd.DataFrame(top_tokens), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.dataset_evaluation_result:
        dataset_result = st.session_state.dataset_evaluation_result
        aggregate = dataset_result.get("aggregate", {})
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Dataset Comparison</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Aggregate causal metrics across the selected dataset. This is the batch view for identifying heads that matter consistently rather than on a single prompt.</div>',
            unsafe_allow_html=True
        )
        aggregate_rows = [
            {"Metric": "Mean target probability delta", "Value": format_metric_value(aggregate.get("mean_target_probability_delta"))},
            {"Metric": "Mean logit diff delta", "Value": format_metric_value(aggregate.get("mean_logit_diff_delta"))},
            {"Metric": "Mean correct-token rank delta", "Value": format_metric_value(aggregate.get("mean_rank_delta"))},
            {"Metric": "Mean loss delta", "Value": format_metric_value(aggregate.get("mean_loss_delta"))},
            {"Metric": "Top prediction changed fraction", "Value": format_metric_value(aggregate.get("top_prediction_changed_fraction"))},
        ]
        st.dataframe(pd.DataFrame(aggregate_rows), use_container_width=True, hide_index=True)

        example_rows = []
        for example in dataset_result.get("examples", [])[:20]:
            delta = example.get("delta") or {}
            example_rows.append(
                {
                    "Example": example.get("example_id"),
                    "Target prob delta": delta.get("target_probability_delta"),
                    "Logit diff delta": delta.get("logit_diff_delta"),
                    "Rank delta": delta.get("target_rank_delta"),
                    "Loss delta": delta.get("loss_delta"),
                    "Prediction changed": delta.get("top_prediction_changed"),
                }
            )
        if example_rows:
            st.caption("Per-example results preview")
            st.dataframe(pd.DataFrame(example_rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    inspector_options = build_head_inspector_options(
        st.session_state.selected_heads,
        get_active_head_groups(st.session_state.head_groups, st.session_state.active_group_ids)
    )
    if inspector_options:
        valid_keys = [option[1] for option in inspector_options]
        if st.session_state.inspector_head_key not in valid_keys:
            st.session_state.inspector_head_key = valid_keys[0]

    inspector_col, hypothesis_col = st.columns([1.05, 1.2])
    with inspector_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Head Inspector</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Inspect one visible head at a time to connect graph structure with a concrete candidate mechanism.</div>',
            unsafe_allow_html=True
        )
        if inspector_options:
            option_map = {label: key for label, key in inspector_options}
            selected_label = st.selectbox(
                "Inspect Head",
                options=list(option_map.keys()),
                index=list(option_map.values()).index(st.session_state.inspector_head_key)
            )
            st.session_state.inspector_head_key = option_map[selected_label]
            head_summary = summarize_head_detail(
                st.session_state.attention_data,
                st.session_state.threshold,
                st.session_state.selected_heads,
                get_active_head_groups(st.session_state.head_groups, st.session_state.active_group_ids),
                st.session_state.inspector_head_key
            )
            if head_summary:
                group_label = ", ".join(head_summary["group_names"]) if head_summary["group_names"] else "Independent selection"
                st.markdown(f"**Group context:** {group_label}")
                st.markdown(f"**Mean visible weight:** {head_summary['mean_weight']:.4f}")
                if head_summary["top_edges"]:
                    for edge in head_summary["top_edges"]:
                        tokens = (st.session_state.attention_data or {}).get("tokens", [])
                        source_token = tokens[edge["sourceToken"]] if edge["sourceToken"] < len(tokens) else f"T{edge['sourceToken']}"
                        dest_token = tokens[edge["destToken"]] if edge["destToken"] < len(tokens) else f"T{edge['destToken']}"
                        st.caption(
                            f"{source_token} -> {dest_token} via L{edge['sourceLayer']}H{edge['head']} • {edge['weight']:.4f}"
                        )
                else:
                    st.caption("No visible edges for this head at the current threshold.")
        else:
            st.info("Add an individual head or keep a predefined group visible to activate the inspector.")
        st.markdown('</div>', unsafe_allow_html=True)
    with hypothesis_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Hypothesis Checklist</div>', unsafe_allow_html=True)
        checklist_items = [
            "Which source token appears to feed the answer-relevant position?",
            "Does the candidate route stay stable when you raise the threshold?",
            "Which selected head looks redundant or backup-like?",
            "What is the next ablation or patching test you would run?"
        ]
        for item in checklist_items:
            st.checkbox(item, value=False, key=f"checklist_{item}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Visualization
    if st.session_state.attention_data:
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        
        try:
            logger.info("Starting graph creation")
            if st.session_state.graph_view_mode == "Max clean-corrupted logit diff":
                if st.session_state.causal_graph_data and causal_graph_matches_current_state():
                    create_max_logit_diff_graph(st.session_state.causal_graph_data)
                    important_heads_frame = build_important_heads_frame(st.session_state.causal_graph_data)
                    if not important_heads_frame.empty:
                        st.caption("Important heads in each layer at the final position")
                        st.dataframe(important_heads_frame, use_container_width=True, hide_index=True)
                else:
                    st.info("Build the clean-corrupted graph to inspect important heads in each layer at the final token position.")
            else:
                logger.info("Attention data structure: " + json.dumps({
                    'numLayers': st.session_state.attention_data['numLayers'],
                    'numTokens': st.session_state.attention_data['numTokens'],
                    'numHeads': st.session_state.attention_data['numHeads'],
                    'numPatterns': get_attention_pattern_count(st.session_state.attention_data),
                    'hasTokens': 'tokens' in st.session_state.attention_data
                }, indent=2))
                create_attention_graph(
                    st.session_state.attention_data,
                    st.session_state.threshold,
                    st.session_state.selected_heads,
                    get_active_head_groups(st.session_state.head_groups, st.session_state.active_group_ids)
                )
        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}", exc_info=True)
            st.error(f"Error creating graph: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        logger.info("No attention data available yet")
        st.info("Enter text and click 'Process Text' to visualize attention patterns.")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 

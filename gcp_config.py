"""Shared GCP configuration for all DIGIT nodes.

Resolution order for project/region:
  1. Node input (if user typed something)
  2. DIGIT-specific env var (DIGIT_GCP_PROJECT, DIGIT_GCP_REGION, DIGIT_GCS_URI)
  3. Standard GCP env vars (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION)
  4. Legacy env vars (GCP_PROJECT_ID, GCP_REGION)
  5. GCP metadata service (auto-detect on Compute Engine / GKE)
  6. Error (project) or fallback default (region → "global")
"""

import os

import requests


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def get_gcp_metadata(path):
    """Fetch metadata from GCP metadata service (works on Compute Engine/GKE)."""
    try:
        response = requests.get(
            f"http://metadata.google.internal/computeMetadata/v1/{path}",
            headers={"Metadata-Flavor": "Google"},
            timeout=5,
        )
        response.raise_for_status()
        return response.text.strip()
    except requests.exceptions.RequestException:
        return None


def get_gcp_access_token():
    """Get an access token from Application Default Credentials."""
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default()
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def build_vertex_url(project, region, model, method="generateContent"):
    """Build the Vertex AI endpoint URL, handling the 'global' region correctly."""
    if region == "global":
        host = "aiplatform.googleapis.com"
    else:
        host = f"{region}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/{project}/locations/{region}"
        f"/publishers/google/models/{model}:{method}"
    )


# ---------------------------------------------------------------------------
# Centralized config resolution
# ---------------------------------------------------------------------------

def resolve_gcp_project(node_value=""):
    """Resolve GCP project ID from node input → env vars → metadata."""
    project = node_value.strip() if node_value else ""
    if project:
        return project

    project = (
        os.environ.get("DIGIT_GCP_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT_ID")
        or ""
    )
    if project:
        return project

    project = get_gcp_metadata("project/project-id")
    if project:
        return project

    raise ValueError(
        "GCP project ID is required. Set it in the node, or set the "
        "DIGIT_GCP_PROJECT environment variable, or run on a GCP instance."
    )


def resolve_gcp_region(node_value="", fallback="global"):
    """Resolve GCP region from node input → env vars → metadata → fallback."""
    region = node_value.strip() if node_value else ""
    if region:
        return region

    region = (
        os.environ.get("DIGIT_GCP_REGION")
        or os.environ.get("GOOGLE_CLOUD_REGION")
        or os.environ.get("GCP_REGION")
        or ""
    )
    if region:
        return region

    zone = get_gcp_metadata("instance/zone")
    if zone:
        zone_name = zone.split("/")[-1]
        return "-".join(zone_name.split("-")[:-1])

    return fallback


def resolve_gcp_config(project_input="", region_input="", region_fallback="global"):
    """Resolve both project and region. Convenience wrapper."""
    return resolve_gcp_project(project_input), resolve_gcp_region(region_input, region_fallback)


def resolve_gcs_uri(node_value="", fallback=""):
    """Resolve GCS bucket URI from node input → env var → fallback."""
    uri = node_value.strip() if node_value else ""
    if uri:
        return uri
    return os.environ.get("DIGIT_GCS_URI", fallback)

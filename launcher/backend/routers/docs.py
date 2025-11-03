from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys
import os
from typing import Optional

router = APIRouter()

# Map short project ids to folder names
PROJECT_MAP = {
    'ann': 'ANN_Project',
    'cnn': 'CNN_Project',
    'nlp': 'NLP',
    'rnn': 'rnn-text-generator'
}


def find_project_path(project_id: str) -> Optional[Path]:
    """Attempt to resolve the project directory on disk."""
    if project_id not in PROJECT_MAP:
        return None

    folder_name = PROJECT_MAP[project_id]
    # Try common locations relative to this file
    candidates = [
        Path(__file__).parent.parent.parent.parent / folder_name,  # repo root
        Path(__file__).parent.parent.parent / folder_name,         # launcher dir
    ]

    for p in candidates:
        try:
            exists = p.exists()
        except Exception:
            exists = False
        # debug output to help diagnose path resolution at runtime
        print(f"[docs.router] find_project_path: checking candidate {p} exists={exists}")
        if exists:
            resolved = p.resolve()
            print(f"[docs.router] find_project_path: resolved project path -> {resolved}")
            return resolved

    return None


@router.get("/{project_id}/technical")
async def get_technical_report(project_id: str):
    """Return the technical report markdown for the project if available."""
    proj_path = find_project_path(project_id)
    if proj_path is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Common doc locations
    possible_docs = [
        proj_path / 'docs' / 'report.md',
        proj_path / 'docs' / 'report.MD',
        proj_path / 'README.md',
        proj_path / 'docs' / 'report.txt'
    ]

    for doc in possible_docs:
        if doc.exists():
            return {"markdown": doc.read_text(encoding='utf-8')}

    raise HTTPException(status_code=404, detail="Technical report not found for project")


@router.get("/{project_id}/cost")
async def get_cost_report(project_id: str):
    """Generate a simple cost report markdown using cost_analysis.example_scenario or return a cost_report.md file if present."""
    proj_path = find_project_path(project_id)
    if proj_path is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # If project provides its own cost report file, prefer that
    possible_cost = [
        proj_path / 'docs' / 'deployment_cost_report.md',
        proj_path / 'docs' / 'deployment-cost-report.md',
        proj_path / 'docs' / 'cost_report.md',
        proj_path / 'docs' / 'cost-report.md'
    ]
    for f in possible_cost:
        if f.exists():
            return {"markdown": f.read_text(encoding='utf-8')}

    # Otherwise, try to generate a short summary using cost_analysis module if available
    try:
        # Ensure repo root is on path
        repo_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(repo_root))
        from cost_analysis import example_scenario

        tco = example_scenario()
        # Build a small markdown summary
        md_lines = [f"# Cost Report (Auto-generated)\n",
                    f"**Project:** {project_id}\n",
                    "## One-time & Training Costs\n",
                    f"- Training cost: ${tco.get('training_cost', 0):.2f}\n",
                    "## Monthly Costs\n",
                    f"- Monthly inference compute: ${tco.get('monthly_inference_cost', 0):.2f}\n",
                    f"- Monthly data transfer: ${tco.get('monthly_data_transfer_cost', 0):.2f}\n",
                    f"- Monthly misc: ${tco.get('monthly_misc', 0):.2f}\n",
                    "\n",
                    "## Total\n",
                    f"- Total over deployment: ${tco.get('total_over_deployment', 0):.2f}\n"
                   ]
        return {"markdown": "\n".join(md_lines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to generate cost report: {e}")


@router.get("/{project_id}/cost/json")
async def get_cost_json(project_id: str):
    """Return structured cost JSON for the project (uses cost_analysis.example_scenario() when available)."""
    proj_path = find_project_path(project_id)
    if proj_path is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # If a project provides its own JSON cost file, prefer that
    possible_json = [proj_path / 'docs' / 'cost_report.json', proj_path / 'docs' / 'cost-report.json']
    for f in possible_json:
        if f.exists():
            try:
                import json
                print(f"[docs.router] get_cost_json: returning JSON from file {f}")
                return json.loads(f.read_text(encoding='utf-8'))
            except Exception:
                print(f"[docs.router] get_cost_json: failed to load JSON from {f}, will try fallback")
                break

    try:
        # Ensure repo root is on path and import the cost module
        repo_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(repo_root))
        from cost_analysis import example_scenario

        print(f"[docs.router] get_cost_json: no project JSON found, falling back to cost_analysis.example_scenario()")
        tco = example_scenario()
        # example_scenario already returns a dict with top-level numeric fields
        return tco
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to generate cost JSON: {e}")

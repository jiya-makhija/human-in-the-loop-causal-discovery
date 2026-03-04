from typing import Dict, List, Set, Tuple

Edge = Tuple[str, str]


def load_toy() -> Dict:
    """
    Tiny dataset to sanity check the pipeline.
    """
    nodes = ["Smoking", "Tar", "Cancer", "Pollution"]
    descriptions = {
        "Smoking": "whether a person smokes cigarettes",
        "Tar": "tar exposure from smoking",
        "Cancer": "lung cancer diagnosis",
        "Pollution": "air pollution exposure",
    }
    # toy ground truth (not perfect, just for testing)
    true_edges: Set[Edge] = {
        ("Smoking", "Tar"),
        ("Tar", "Cancer"),
        ("Pollution", "Cancer"),
    }
    return {"name": "toy", "nodes": nodes, "descriptions": descriptions, "true_edges": true_edges}


def load_asia_placeholder() -> Dict:
    """
    Placeholder "Asia-like" dataset (NOT the real Asia BN).
    This exists so you can run experiments without extra deps.

    Later you can replace this with a real Asia loader from pgmpy.
    """
    nodes = ["VisitAsia", "Smoking", "Tuberculosis", "LungCancer", "Bronchitis", "XRay", "Dyspnea"]
    descriptions = {
        "VisitAsia": "recent travel to Asia",
        "Smoking": "smoker status",
        "Tuberculosis": "tuberculosis disease",
        "LungCancer": "lung cancer disease",
        "Bronchitis": "bronchitis disease",
        "XRay": "abnormal chest x-ray",
        "Dyspnea": "shortness of breath",
    }
    true_edges: Set[Edge] = {
        ("VisitAsia", "Tuberculosis"),
        ("Smoking", "LungCancer"),
        ("Smoking", "Bronchitis"),
        ("Tuberculosis", "XRay"),
        ("LungCancer", "XRay"),
        ("Tuberculosis", "Dyspnea"),
        ("LungCancer", "Dyspnea"),
        ("Bronchitis", "Dyspnea"),
    }
    return {"name": "asia_placeholder", "nodes": nodes, "descriptions": descriptions, "true_edges": true_edges}
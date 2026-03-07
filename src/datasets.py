from typing import Dict, Set, Tuple

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
    true_edges: Set[Edge] = {
        ("Smoking", "Tar"),
        ("Tar", "Cancer"),
        ("Pollution", "Cancer"),
    }
    return {
        "name": "toy",
        "nodes": nodes,
        "descriptions": descriptions,
        "true_edges": true_edges,
    }


def _asia_structure() -> Dict:
    """
    Returns the Asia network variable names, descriptions, and true edges.
    """
    nodes = [
        "VisitAsia",
        "Smoking",
        "Tuberculosis",
        "LungCancer",
        "Bronchitis",
        "XRay",
        "Dyspnea",
    ]

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

    return {
        "nodes": nodes,
        "descriptions": descriptions,
        "true_edges": true_edges,
    }


def load_asia_placeholder() -> Dict:
    """
    Asia structure without sampled data.
    """
    base = _asia_structure()
    return {
        "name": "asia_placeholder",
        "nodes": base["nodes"],
        "descriptions": base["descriptions"],
        "true_edges": base["true_edges"],
    }


def load_asia_generated(n_samples: int = 2000, seed: int = 42) -> Dict:
    """
    Generate samples from the Asia Bayesian network using pgmpy.

    Returns a dataset dict with:
    - name
    - nodes
    - descriptions
    - true_edges
    - data (pandas DataFrame)
    """
    base = _asia_structure()

    try:
        import numpy as np
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.sampling import BayesianModelSampling

        np.random.seed(seed)

        model = DiscreteBayesianNetwork(list(base["true_edges"]))

        cpd_visit_asia = TabularCPD(
            variable="VisitAsia",
            variable_card=2,
            values=[[0.99], [0.01]],
            state_names={"VisitAsia": [0, 1]},
        )

        cpd_smoking = TabularCPD(
            variable="Smoking",
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={"Smoking": [0, 1]},
        )

        cpd_tuberculosis = TabularCPD(
            variable="Tuberculosis",
            variable_card=2,
            values=[[0.99, 0.95], [0.01, 0.05]],
            evidence=["VisitAsia"],
            evidence_card=[2],
            state_names={"Tuberculosis": [0, 1], "VisitAsia": [0, 1]},
        )

        cpd_lung_cancer = TabularCPD(
            variable="LungCancer",
            variable_card=2,
            values=[[0.99, 0.90], [0.01, 0.10]],
            evidence=["Smoking"],
            evidence_card=[2],
            state_names={"LungCancer": [0, 1], "Smoking": [0, 1]},
        )

        cpd_bronchitis = TabularCPD(
            variable="Bronchitis",
            variable_card=2,
            values=[[0.70, 0.40], [0.30, 0.60]],
            evidence=["Smoking"],
            evidence_card=[2],
            state_names={"Bronchitis": [0, 1], "Smoking": [0, 1]},
        )

        cpd_xray = TabularCPD(
            variable="XRay",
            variable_card=2,
            values=[
                [0.95, 0.02, 0.10, 0.02],
                [0.05, 0.98, 0.90, 0.98],
            ],
            evidence=["Tuberculosis", "LungCancer"],
            evidence_card=[2, 2],
            state_names={
                "XRay": [0, 1],
                "Tuberculosis": [0, 1],
                "LungCancer": [0, 1],
            },
        )

        cpd_dyspnea = TabularCPD(
            variable="Dyspnea",
            variable_card=2,
            values=[
                [0.90, 0.30, 0.35, 0.20, 0.30, 0.20, 0.25, 0.10],
                [0.10, 0.70, 0.65, 0.80, 0.70, 0.80, 0.75, 0.90],
            ],
            evidence=["Tuberculosis", "LungCancer", "Bronchitis"],
            evidence_card=[2, 2, 2],
            state_names={
                "Dyspnea": [0, 1],
                "Tuberculosis": [0, 1],
                "LungCancer": [0, 1],
                "Bronchitis": [0, 1],
            },
        )

        model.add_cpds(
            cpd_visit_asia,
            cpd_smoking,
            cpd_tuberculosis,
            cpd_lung_cancer,
            cpd_bronchitis,
            cpd_xray,
            cpd_dyspnea,
        )

        if not model.check_model():
            raise ValueError("Asia Bayesian network is invalid")

        sampler = BayesianModelSampling(model)
        data = sampler.forward_sample(size=n_samples, seed=seed, show_progress=False)

        data = data[base["nodes"]].astype(int)

        return {
            "name": f"asia_generated_{n_samples}",
            "nodes": base["nodes"],
            "descriptions": base["descriptions"],
            "true_edges": base["true_edges"],
            "data": data,
        }

    except Exception as e:
        print("[datasets] failed to generate Asia samples, falling back:", e)
        return {
            "name": f"asia_generated_{n_samples}_fallback",
            "nodes": base["nodes"],
            "descriptions": base["descriptions"],
            "true_edges": base["true_edges"],
        }
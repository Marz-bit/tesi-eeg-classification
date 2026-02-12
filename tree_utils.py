import json
from pathlib import Path
from typing import Sequence, Dict

from bionic_apps.ai.tree_of_experts import NODE_FOLDERS

#costruisce mp composto dai 4 file json ognuno con dentro 8 weights file
def build_tree_weights_json(
    weights_root: str | Path,
    test_subjects: Sequence[int],
    *,
    prefix: str = "EEG_NET",
    out_dir: str | Path | None = None,
) -> Path:
    """
    Cerca per ogni nodo N1..N8 il file:
      EEG_NET_testsubj-1-2-3-4-5.weights.h5
    nella cartella:
      weights_root/<NODE_FOLDER>/models/

    e salva un JSON:
      {"N1": "...h5", ..., "N8":"...h5"}
    """
    #ogni file è legato alla LPO split perchè test_subjects è preso da init_tl_training dove fa LeavePSubjectGroupsOutSequentiall
    #weights_root è la root del run (es. CM_HIER_FALSEART_1-F)
    weights_root = Path(weights_root)
    test_subjects = tuple(sorted(int(s) for s in test_subjects)) #ordina id paziente
    tag = "testsubj-" + "-".join(map(str, test_subjects))

    #NODE_FOLDERS viene importato da tree_of_experts.py
    mapping: Dict[str, str] = {}
    for node, folder in NODE_FOLDERS.items():
        models_dir = weights_root / folder / "models"
        if not models_dir.exists():
            raise FileNotFoundError(f"Non trovo models_dir: {models_dir}")

        matches = sorted(models_dir.glob(f"{prefix}_{tag}*.weights.h5"))
        if len(matches) == 0:
            raise FileNotFoundError(f"Nessun peso per {node} in {models_dir} con tag {tag}")

        mapping[node] = str(matches[0]) #scegli il primo match in ordine alfabetico (perché sorted(...)).

    #Salva il JSON in una cartella dedicata, Quindi genera un file tipo:
    #<weights_root>/_tree_weight_json/tree_testsubj-1-2-3-4-5.json
    #e lo ritorna come Path
    if out_dir is None:
        out_dir = weights_root / "_tree_weight_json"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"tree_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=2)

    return json_path

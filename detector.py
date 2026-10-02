import os
import re

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False

try:
    import nodes
    HAS_NODES = True
except ImportError:
    HAS_NODES = False

# Known widget name -> folder type mapping
WIDGET_TO_FOLDER = {
    "ckpt_name": "checkpoints",
    "checkpoint": "checkpoints",
    "lora_name": "loras",
    "lora": "loras",
    "vae_name": "vae",
    "vae": "vae",
    "control_net_name": "controlnet",
    "controlnet_name": "controlnet",
    "unet_name": "unet",
    "clip_name": "clip",
    "clip_name1": "clip",
    "clip_name2": "clip",
    "clip_name3": "clip",
    "clip_vision_name": "clip_vision",
    "gligen_name": "gligen",
    "style_model_name": "style_models",
    "embedding_name": "embeddings",
    "upscale_model_name": "upscale_models",
    "tiny_vae": "vae_approx",
}

# Known node type prefix/keyword -> folder type mapping
NODE_TYPE_TO_FOLDER = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "clipvision": "clip_vision",
    "clip": "clip",
    "unet": "unet",
    "diffusion": "diffusion_models",
    "latentupscale": "latent_upscale_models",
    "upscale": "upscale_models",
    "gligen": "gligen",
    "animatediff": "animatediff_models",
    "ipadapter": "ipadapter",
    "moge": "geometry_estimation",
    "depth": "depth",
    "backgroundremoval": "rembg",
    "birefnet": "inpaint",
    "taesd": "vae_approx",
}

MODEL_EXTENSIONS = (".safetensors", ".gguf", ".ckpt", ".pt", ".bin", ".pth", ".onnx")

# Registered folder types that are not model directories and must never be download targets
EXCLUDED_FOLDERS = {"custom_nodes"}

# Custom (unregistered) folder names are created under models/, so keep them to a single safe path segment
CUSTOM_FOLDER_RE = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9_.\- ]*$")

def is_safe_folder_name(name: str) -> bool:
    """True for a single path segment that can't escape its parent (no separators, not just dots)."""
    return bool(name) and bool(CUSTOM_FOLDER_RE.match(name)) and name.strip(".").strip() != ""


def collect_workflow_model_urls(workflow: dict) -> dict:
    """
    ComfyUI templates embed download info as `models: [{name, url, directory, hash?, hash_type?}]`,
    both at the top level of the workflow and in each node's `properties`. Returns lowercase
    basename -> entry for every entry that has an http(s) URL.
    """
    if not isinstance(workflow, dict):
        return {}

    entry_lists = [workflow.get("models")]
    node_lists = [workflow.get("nodes")]
    for sg in (workflow.get("definitions") or {}).get("subgraphs") or []:
        if isinstance(sg, dict):
            node_lists.append(sg.get("nodes"))
    for nodes_list in node_lists:
        for node in nodes_list or []:
            if isinstance(node, dict) and isinstance(node.get("properties"), dict):
                entry_lists.append(node["properties"].get("models"))

    urls = {}
    for entries in entry_lists:
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name, url = entry.get("name"), entry.get("url")
            if not isinstance(name, str) or not isinstance(url, str):
                continue
            if not url.lower().startswith(("http://", "https://")):
                continue
            key = os.path.basename(name.replace("\\", "/")).lower()
            urls.setdefault(key, entry)
    return urls


class MissingModelDetector:
    def __init__(self):
        pass

    def get_registered_folders(self) -> dict:
        """Returns dictionary of folder_type -> list of directory paths."""
        if HAS_FOLDER_PATHS and hasattr(folder_paths, "folder_names_and_paths"):
            result = {}
            for folder_type, (dirs, exts) in folder_paths.folder_names_and_paths.items():
                if folder_type in EXCLUDED_FOLDERS:
                    continue
                result[folder_type] = {
                    "paths": dirs,
                    "extensions": list(exts) if isinstance(exts, set) else exts
                }
            return result
        # Fallback standard ComfyUI folder layout
        return {
            "checkpoints": {"paths": ["models/checkpoints"], "extensions": list(MODEL_EXTENSIONS)},
            "loras": {"paths": ["models/loras"], "extensions": list(MODEL_EXTENSIONS)},
            "vae": {"paths": ["models/vae"], "extensions": list(MODEL_EXTENSIONS)},
            "controlnet": {"paths": ["models/controlnet"], "extensions": list(MODEL_EXTENSIONS)},
            "unet": {"paths": ["models/unet", "models/diffusion_models"], "extensions": list(MODEL_EXTENSIONS)},
            "clip": {"paths": ["models/clip"], "extensions": list(MODEL_EXTENSIONS)},
            "clip_vision": {"paths": ["models/clip_vision"], "extensions": list(MODEL_EXTENSIONS)},
            "upscale_models": {"paths": ["models/upscale_models"], "extensions": list(MODEL_EXTENSIONS)},
            "embeddings": {"paths": ["models/embeddings"], "extensions": [".pt", ".bin", ".safetensors"]},
        }

    def get_existing_filenames(self, folder_type: str) -> list:
        """Returns list of existing model filenames known to ComfyUI for folder_type."""
        if HAS_FOLDER_PATHS:
            try:
                return folder_paths.get_filename_list(folder_type)
            except Exception:
                pass
        return []

    def get_target_directory(self, folder_type: str) -> str:
        """
        Returns the primary filesystem directory for a folder type. Unregistered names are
        treated as a custom folder under the models directory.
        """
        if folder_type in EXCLUDED_FOLDERS:
            raise ValueError(f"'{folder_type}' is not a model folder")
        if HAS_FOLDER_PATHS:
            try:
                paths = folder_paths.get_folder_paths(folder_type)
                if paths:
                    return paths[0]
            except Exception:
                pass

        if not is_safe_folder_name(folder_type):
            raise ValueError(f"Invalid folder name: '{folder_type}'")
        models_dir = getattr(folder_paths, "models_dir", None) if HAS_FOLDER_PATHS else None
        if not models_dir:
            models_dir = os.path.join(os.getcwd(), "models")
        return os.path.join(models_dir, folder_type.strip())

    def infer_folder_type(self, node_type: str, widget_name: str, value: str) -> str:
        """Infers the ComfyUI folder category from widget name, node class, or value."""
        widget_lower = widget_name.lower() if widget_name else ""
        node_lower = node_type.lower() if node_type else ""
        folder = None
        
        # 1. PRIORITY: Logical detection using ComfyUI node class definition
        if HAS_NODES and hasattr(nodes, "NODE_CLASS_MAPPINGS"):
            cls = nodes.NODE_CLASS_MAPPINGS.get(node_type)
            if cls and hasattr(cls, "INPUT_TYPES"):
                try:
                    inputs = cls.INPUT_TYPES()
                    all_inputs = {}
                    if "required" in inputs:
                        all_inputs.update(inputs["required"])
                    if "optional" in inputs:
                        all_inputs.update(inputs["optional"])

                    if widget_name in all_inputs:
                        spec = all_inputs[widget_name]
                        if isinstance(spec, tuple) and len(spec) > 0 and isinstance(spec[0], list):
                            choices = spec[0]
                            # Check which folder's filename list matches these choices exactly
                            # ONLY if choices is not empty. If it's empty, we can't uniquely match it.
                            if HAS_FOLDER_PATHS and len(choices) > 0:
                                for f_type in folder_paths.folder_names_and_paths.keys():
                                    try:
                                        f_list = folder_paths.get_filename_list(f_type)
                                        if f_list == choices:
                                            folder = f_type
                                            break
                                    except Exception:
                                        continue
                except Exception:
                    pass

        # 2. Fallback: Check known widget names
        if not folder and widget_lower in WIDGET_TO_FOLDER:
            folder = WIDGET_TO_FOLDER[widget_lower]

        # 3. Fallback: Check known node type substrings
        if not folder:
            for key, val in NODE_TYPE_TO_FOLDER.items():
                if key in node_lower:
                    folder = val
                    break

        # Default fallback
        if not folder:
            folder = "checkpoints"

        return folder

    def detect_missing_models(self, workflow_data: dict) -> list:
        """
        Analyzes workflow JSON (either graph format or prompt format or candidates list).
        Returns list of missing model objects.
        """
        registered_folders = self.get_registered_folders()
        folder_names = list(registered_folders.keys())
        # Collect raw candidate model references
        candidates = []
        prompt_data = workflow_data.get("prompt")
        frontend_candidates = workflow_data.get("candidates")

        # Format A: ComfyUI Prompt format (most accurate, expanded group nodes)
        if prompt_data and isinstance(prompt_data, dict):
            for node_id, node_spec in prompt_data.items():
                if not isinstance(node_spec, dict): continue
                class_type = node_spec.get("class_type", "")
                inputs = node_spec.get("inputs", {})
                if not isinstance(inputs, dict): continue

                for input_name, val in inputs.items():
                    if isinstance(val, str) and (
                        any(val.lower().endswith(ext) for ext in MODEL_EXTENSIONS) or
                        input_name.lower() in WIDGET_TO_FOLDER
                    ):
                        folder_type = self.infer_folder_type(class_type, input_name, val)
                        candidates.append({
                            "node_id": str(node_id),
                            "node_type": class_type,
                            "widget_name": input_name,
                            "value": val,
                            "folder_type": folder_type
                        })

        # Format B: Direct candidates provided by frontend (fallback)
        elif frontend_candidates and isinstance(frontend_candidates, list):
            candidates = frontend_candidates
            
            # Patch candidates with real class_type from workflow graph if available (useful for Group Nodes)
            workflow_graph = workflow_data.get("workflow")
            if workflow_graph and isinstance(workflow_graph, dict):
                class_map = {}
                # Check root nodes
                for n in workflow_graph.get("nodes", []):
                    if isinstance(n, dict) and "id" in n and "type" in n:
                        class_map[str(n["id"])] = n["type"]
                
                # Check subgraph nodes (this resolves the UUIDs for Group Nodes)
                subgraphs = workflow_graph.get("definitions", {}).get("subgraphs", [])
                for sg in subgraphs:
                    for n in sg.get("nodes", []):
                        if isinstance(n, dict) and "id" in n and "type" in n:
                            class_map[str(n["id"])] = n["type"]
                            
                for cand in candidates:
                    nid = str(cand.get("node_id", ""))
                    # If we found the real type, replace the frontend's guessed UUID
                    if nid in class_map and class_map[nid]:
                        cand["node_type"] = class_map[nid]
        elif "nodes" in workflow_data and isinstance(workflow_data["nodes"], list):
            for node in workflow_data["nodes"]:
                node_id = node.get("id")
                node_type = node.get("type", "")
                widgets_values = node.get("widgets_values", [])

                if not isinstance(widgets_values, list):
                    continue

                for idx, val in enumerate(widgets_values):
                    if isinstance(val, str) and any(val.lower().endswith(ext) for ext in MODEL_EXTENSIONS):
                        widget_name = f"widget_{idx}"
                        # Try to find widget name from node inputs/widgets if available
                        folder_type = self.infer_folder_type(node_type, widget_name, val)
                        candidates.append({
                            "node_id": node_id,
                            "node_type": node_type,
                            "widget_name": widget_name,
                            "value": val,
                            "folder_type": folder_type
                        })

        # Format 3: ComfyUI Prompt format (node_id -> {class_type, inputs})
        elif isinstance(workflow_data, dict):
            for node_id, node_spec in workflow_data.items():
                if not isinstance(node_spec, dict):
                    continue
                class_type = node_spec.get("class_type", "")
                inputs = node_spec.get("inputs", {})
                if not isinstance(inputs, dict):
                    continue

                for input_name, val in inputs.items():
                    if isinstance(val, str) and (
                        any(val.lower().endswith(ext) for ext in MODEL_EXTENSIONS) or
                        input_name.lower() in WIDGET_TO_FOLDER
                    ):
                        folder_type = self.infer_folder_type(class_type, input_name, val)
                        candidates.append({
                            "node_id": node_id,
                            "node_type": class_type,
                            "widget_name": input_name,
                            "value": val,
                            "folder_type": folder_type
                        })

        # Ensure all detected folder types are in the available folders list
        # so the frontend dropdown can actually select them, even if ComfyUI didn't register them.
        detected_folders = set(c.get("folder_type") for c in candidates if c.get("folder_type"))
        for f in detected_folders:
            if f not in folder_names:
                folder_names.append(f)

        workflow_urls = collect_workflow_model_urls(workflow_data.get("workflow") or workflow_data)

        # Deduplicate and verify against folder_paths
        missing_models = []
        seen = set()

        for cand in candidates:
            raw_val = cand.get("value")
            if not raw_val or not isinstance(raw_val, str):
                continue

            raw_val = raw_val.strip()
            # Normalize filename
            filename = os.path.basename(raw_val.replace("\\", "/"))
            folder_type = cand.get("folder_type") or self.infer_folder_type(cand.get("node_type", ""), cand.get("widget_name", ""), raw_val)

            dedup_key = (folder_type, filename.lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Check if file exists in ComfyUI
            existing_files = self.get_existing_filenames(folder_type)
            exists = False
            if filename in existing_files:
                exists = True
            else:
                # Check basename match (handles subfolders like "sub/model.safetensors")
                for ef in existing_files:
                    if os.path.basename(ef.replace("\\", "/")).lower() == filename.lower():
                        exists = True
                        break

            # Also verify via filesystem check in registered paths
            if not exists and HAS_FOLDER_PATHS:
                try:
                    full_path = folder_paths.get_full_path(folder_type, raw_val)
                    if full_path and os.path.exists(full_path):
                        exists = True
                except Exception:
                    pass

            # NEW: If still not found, search ALL registered ComfyUI folders to prevent false positives
            if not exists and HAS_FOLDER_PATHS:
                for f_type in folder_paths.folder_names_and_paths.keys():
                    if f_type == folder_type:
                        continue
                    try:
                        f_list = folder_paths.get_filename_list(f_type)
                        if filename in f_list or any(os.path.basename(ef.replace("\\", "/")).lower() == filename.lower() for ef in f_list):
                            exists = True
                            folder_type = f_type
                            break
                    except Exception:
                        pass

            if not exists:
                # Prefer the download URL and folder the workflow author recorded, over searching
                known = workflow_urls.get(filename.lower())
                directory = known.get("directory") if known else None
                if isinstance(directory, str) and is_safe_folder_name(directory):
                    folder_type = directory
                known_sha256 = ""
                if known and str(known.get("hash_type", "")).lower() == "sha256":
                    known_sha256 = str(known.get("hash") or "").lower()

                # Keep subfolders like "flux/model.safetensors" so the node's path resolves after download
                subfolder = os.path.dirname(raw_val.replace("\\", "/")).strip("/")
                if subfolder and not all(is_safe_folder_name(seg) for seg in subfolder.split("/")):
                    subfolder = ""

                try:
                    target_dir = self.get_target_directory(folder_type)
                except ValueError:
                    target_dir = ""
                missing_models.append({
                    "filename": filename,
                    "original_value": raw_val,
                    "subfolder": subfolder,
                    "known_url": known["url"] if known else "",
                    "known_sha256": known_sha256,
                    "folder_type": folder_type,
                    "target_dir": target_dir,
                    "available_folders": folder_names,
                    "node_id": cand.get("node_id"),
                    "node_type": cand.get("node_type", "UnknownNode"),
                    "widget_name": cand.get("widget_name", "")
                })

        return missing_models

detector = MissingModelDetector()

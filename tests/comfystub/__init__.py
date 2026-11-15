"""Enough of ComfyUI to exercise the node pack without a ComfyUI install.

Only the surfaces the pack actually touches: folder_paths, comfy.lora's key maps,
comfy.sd.load_lora_for_models, and the PromptServer route table the license routes register on.
Real ComfyUI is still the thing that proves a workflow runs; this proves the logic around it.
"""

import json
import os
import sys
import types
from pathlib import Path


class FakeRoutes:
    """aiohttp's RouteTableDef, reduced to recording which handler serves which method and path."""

    def __init__(self):
        self.handlers = {}

    def _add(self, method, path):
        def register(handler):
            self.handlers[(method, path)] = handler
            return handler

        return register

    def get(self, path):
        return self._add("GET", path)

    def post(self, path):
        return self._add("POST", path)

    def delete(self, path):
        return self._add("DELETE", path)


def _json_response(data, status=200):
    return types.SimpleNamespace(status=status, data=data, body=json.dumps(data).encode())


class FakeModel:
    def __init__(self, stems):
        self.model = types.SimpleNamespace(stems=stems)


class FakeVae:
    """Records what it was asked to encode, so a test can check the pixels that reached it."""

    def __init__(self):
        self.encoded = []

    def encode(self, pixels):
        self.encoded.append(tuple(pixels.shape))
        return f"latent{len(self.encoded)}"


class FakeClip:
    """Enough of a CLIP to prove the prompt reaches the encoder and conditioning comes back."""

    def __init__(self):
        self.cond_stage_model = types.SimpleNamespace()
        self.seen = None

    def tokenize(self, text):
        self.seen = text
        return {"tokens": text}

    def encode_from_tokens_scheduled(self, tokens):
        return [[f"cond:{tokens['tokens'][:40]}", {}]]


def install(models_dir: Path, unet_stems=(), clip_stems=(), user_dir: Path | None = None):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(models_dir)
    user = user_dir or models_dir.parent / "user"
    folder_paths.get_user_directory = lambda: str(user)
    folder_paths.folder_names_and_paths = {"loras": ([str(models_dir / "loras")], {".safetensors"})}

    def get_filename_list(category):
        paths, exts = folder_paths.folder_names_and_paths[category]
        out = []
        for base in paths:
            for path in Path(base).rglob("*"):
                if path.suffix in exts:
                    out.append(path.name)
        return out

    def get_folder_paths(category):
        return list(folder_paths.folder_names_and_paths[category][0])

    def get_full_path(category, name):
        # As ComfyUI does: rooted then made relative, so "../" and absolute names cannot escape.
        name = os.path.relpath(os.path.join("/", name), "/")
        for base in get_folder_paths(category):
            candidate = os.path.join(base, name)
            if os.path.isfile(candidate):
                return candidate
        return None

    folder_paths.get_filename_list = get_filename_list
    folder_paths.get_folder_paths = get_folder_paths
    folder_paths.get_full_path = get_full_path

    # node_helpers.conditioning_set_values, matching ComfyUI's append semantics.
    node_helpers = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values, append=False):
        out = []
        for entry in conditioning:
            extra = dict(entry[1])
            for key, value in values.items():
                if append:
                    extra[key] = list(extra.get(key, [])) + list(value)
                else:
                    extra[key] = value
            out.append([entry[0], extra])
        return out

    node_helpers.conditioning_set_values = conditioning_set_values

    comfy = types.ModuleType("comfy")
    lora = types.ModuleType("comfy.lora")
    sd = types.ModuleType("comfy.sd")

    def _offer(stems):
        """The shapes ComfyUI actually offers a module under: a wrapped dotted name and kohya's
        underscored one, both mapping to the model's own key. Neither carries an adapter suffix."""
        out = {}
        for stem in stems:
            out[f"diffusion_model.{stem}"] = stem
            out["lora_unet_" + stem.replace(".", "_")] = stem
        return out

    lora.model_lora_keys_unet = lambda model, key_map: {**key_map, **_offer(model.stems)}
    lora.model_lora_keys_clip = lambda model, key_map: {**key_map, **_offer(clip_stems)}

    def load_lora_for_models(model, clip, state, strength_model, strength_clip):
        return (("patched", model, strength_model), ("patched-clip", clip, strength_clip))

    sd.load_lora_for_models = load_lora_for_models
    comfy.lora = lora
    comfy.sd = sd

    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=FakeRoutes())
    )
    stubs = {}
    try:
        import aiohttp.web  # noqa: F401
    except ImportError:
        aiohttp = types.ModuleType("aiohttp")
        web = types.ModuleType("aiohttp.web")
        web.json_response = _json_response
        aiohttp.web = web
        stubs = {"aiohttp": aiohttp, "aiohttp.web": web}

    for name, module in {
        **stubs,
        "server": server,
        "folder_paths": folder_paths,
        "node_helpers": node_helpers,
        "comfy": comfy,
        "comfy.lora": lora,
        "comfy.sd": sd,
    }.items():
        sys.modules[name] = module
    return folder_paths

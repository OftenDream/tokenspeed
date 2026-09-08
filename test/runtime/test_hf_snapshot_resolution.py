"""Regression tests for single-snapshot Hugging Face metadata loading."""

import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.utils.hf_transformers_utils import (
    _CONFIG_REGISTRY,
    _find_deepseek_v4_encoding_file,
    _snapshot_commit_hash,
    get_config,
    get_generation_config,
    get_tokenizer,
)


class _SnapshotConfig(PretrainedConfig):
    model_type = "snapshot_test"

    def __init__(self, snapshot_value: str = "default", **kwargs) -> None:
        self.snapshot_value = snapshot_value
        super().__init__(**kwargs)


class _RecordingLock:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self):
        self.events.append("acquire")

    def __exit__(self, *exc_info):
        self.events.append("release")


class HFSnapshotResolutionTests(unittest.TestCase):
    def test_snapshot_commit_hash_requires_standard_hf_layout(self) -> None:
        commit = "c" * 40
        self.assertEqual(
            _snapshot_commit_hash(f"/cache/repo/snapshots/{commit}/"), commit
        )
        self.assertIsNone(_snapshot_commit_hash("/cache/repo/snapshots/main"))

    def _write_config(
        self,
        directory: str,
        model_type: str,
        *,
        snapshot_value: str | None = None,
    ) -> None:
        config = {
            "model_type": model_type,
            "architectures": ["SnapshotForCausalLM"],
        }
        if snapshot_value is not None:
            config["snapshot_value"] = snapshot_value
        with open(os.path.join(directory, "config.json"), "w") as file:
            json.dump(config, file)

    def test_registered_config_uses_revision_pinned_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as snapshot:
            self._write_config(snapshot, _SnapshotConfig.model_type)
            loaded = _SnapshotConfig(architectures=["SnapshotForCausalLM"])
            with (
                patch.dict(
                    _CONFIG_REGISTRY,
                    {_SnapshotConfig.model_type: _SnapshotConfig},
                ),
                patch(
                    "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                    return_value=nullcontext(),
                ),
                patch(
                    "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                    return_value=snapshot,
                ) as download,
                patch.object(
                    _SnapshotConfig,
                    "from_pretrained",
                    return_value=loaded,
                ) as from_pretrained,
            ):
                config = get_config(
                    "org/model",
                    trust_remote_code=False,
                    revision="model-revision",
                )

        download.assert_called_once_with(
            "org/model",
            revision="model-revision",
            ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
        )
        from_pretrained.assert_called_once_with(snapshot)
        self.assertIs(config, loaded)
        self.assertEqual(config._name_or_path, "org/model")

    def test_remote_code_config_uses_repo_and_commit_under_lock(self) -> None:
        commit = "a" * 40
        events: list[str] = []

        with tempfile.TemporaryDirectory() as root:
            snapshot = os.path.join(root, "models--org--model", "snapshots", commit)
            os.makedirs(snapshot)
            self._write_config(snapshot, _SnapshotConfig.model_type)
            loaded = _SnapshotConfig(architectures=["SnapshotForCausalLM"])

            def download_snapshot(*args, **kwargs):
                events.append("snapshot")
                return snapshot

            def load_config(*args, **kwargs):
                events.append("parse")
                return loaded

            with (
                patch.dict(_CONFIG_REGISTRY, {}, clear=True),
                patch(
                    "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                    return_value=_RecordingLock(events),
                ),
                patch(
                    "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                    side_effect=download_snapshot,
                ),
                patch(
                    "tokenspeed.runtime.utils.hf_transformers_utils.AutoConfig.from_pretrained",
                    side_effect=load_config,
                ) as from_pretrained,
            ):
                config = get_config("org/model", trust_remote_code=True)

        from_pretrained.assert_called_once_with(
            "org/model",
            trust_remote_code=True,
            revision=commit,
        )
        self.assertEqual(events, ["acquire", "snapshot", "parse", "release"])
        self.assertIs(config, loaded)
        self.assertEqual(config._name_or_path, "org/model")

    def test_remote_code_config_without_commit_falls_back_to_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as snapshot:
            self._write_config(snapshot, _SnapshotConfig.model_type)
            loaded = _SnapshotConfig(architectures=["SnapshotForCausalLM"])
            with (
                patch.dict(_CONFIG_REGISTRY, {}, clear=True),
                patch(
                    "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                    return_value=nullcontext(),
                ),
                patch(
                    "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                    return_value=snapshot,
                ),
                patch(
                    "tokenspeed.runtime.utils.hf_transformers_utils.AutoConfig.from_pretrained",
                    return_value=loaded,
                ) as from_pretrained,
                patch(
                    "tokenspeed.runtime.utils.hf_transformers_utils.logger.warning"
                ) as warn,
            ):
                config = get_config("org/model", trust_remote_code=True)

        from_pretrained.assert_called_once_with(snapshot, trust_remote_code=True)
        warn.assert_called_once()
        self.assertIs(config, loaded)
        self.assertEqual(config._name_or_path, "org/model")

    def test_ordinary_auto_config_uses_snapshot_after_releasing_lock(self) -> None:
        commit = "e" * 40
        events: list[str] = []

        with tempfile.TemporaryDirectory() as root:
            snapshot = os.path.join(root, "models--org--model", "snapshots", commit)
            os.makedirs(snapshot)
            self._write_config(snapshot, _SnapshotConfig.model_type)
            loaded = _SnapshotConfig(architectures=["SnapshotForCausalLM"])

            def download_snapshot(*args, **kwargs):
                events.append("snapshot")
                return snapshot

            def load_config(*args, **kwargs):
                events.append("parse")
                return loaded

            with (
                patch.dict(_CONFIG_REGISTRY, {}, clear=True),
                patch(
                    "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                    return_value=_RecordingLock(events),
                ),
                patch(
                    "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                    side_effect=download_snapshot,
                ),
                patch(
                    "tokenspeed.runtime.utils.hf_transformers_utils.AutoConfig.from_pretrained",
                    side_effect=load_config,
                ) as from_pretrained,
            ):
                config = get_config("org/model", trust_remote_code=False)

        from_pretrained.assert_called_once_with(
            snapshot,
            trust_remote_code=False,
        )
        self.assertEqual(events, ["acquire", "snapshot", "release", "parse"])
        self.assertIs(config, loaded)
        self.assertEqual(config._name_or_path, "org/model")

    def test_local_config_is_parsed_from_its_real_json(self) -> None:
        with tempfile.TemporaryDirectory() as snapshot:
            self._write_config(
                snapshot,
                _SnapshotConfig.model_type,
                snapshot_value="from-snapshot",
            )
            with (
                patch.dict(
                    _CONFIG_REGISTRY,
                    {_SnapshotConfig.model_type: _SnapshotConfig},
                ),
                patch(
                    "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                    side_effect=AssertionError("local configs must not download"),
                ) as download,
            ):
                config = get_config(snapshot, trust_remote_code=False)

        download.assert_not_called()
        self.assertEqual(config.snapshot_value, "from-snapshot")

    def test_longcat_architecture_loads_local_config_without_model_type(self) -> None:
        from tokenspeed.runtime.configs.longcat_config import LongcatConfig

        with tempfile.TemporaryDirectory() as snapshot:
            with open(os.path.join(snapshot, "config.json"), "w") as file:
                json.dump(
                    {
                        "architectures": ["LongcatCausalLM"],
                        "vocab_size": 163840,
                        "hidden_size": 8192,
                        "num_layers": 38,
                        "num_hidden_layers": 76,
                        "num_attention_heads": 64,
                        "n_routed_experts": 768,
                        "moe_topk": 12,
                        "qk_nope_head_dim": 128,
                        "qk_rope_head_dim": 64,
                    },
                    file,
                )

            config = get_config(snapshot, trust_remote_code=False)

        self.assertIsInstance(config, LongcatConfig)
        self.assertEqual(config.num_hidden_layers, 38)
        self.assertEqual(config.qk_head_dim, 192)
        self.assertEqual(config.num_experts_per_tok, 12)

    def test_legacy_longcat_flash_does_not_inherit_pro_oe_defaults(self) -> None:
        from tokenspeed.runtime.configs.longcat_config import LongcatConfig
        from tokenspeed.runtime.models.longcat_flash import _ensure_longcat_config

        with tempfile.TemporaryDirectory() as snapshot:
            with open(os.path.join(snapshot, "config.json"), "w") as file:
                json.dump(
                    {
                        "model_type": "longcat",
                        "architectures": ["LongcatFlashForCausalLM"],
                        "vocab_size": 131072,
                        "hidden_size": 6144,
                        "num_layers": 32,
                        "num_attention_heads": 48,
                    },
                    file,
                )

            config = get_config(snapshot, trust_remote_code=False)

        self.assertIsInstance(config, LongcatConfig)
        _ensure_longcat_config(config)
        self.assertIsNone(config.oe_vocab_size_ratio)
        self.assertFalse(config.use_over_embedding)

    def test_generation_config_uses_revision_pinned_snapshot(self) -> None:
        loaded = SimpleNamespace()
        with (
            patch(
                "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                return_value=nullcontext(),
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                return_value="/resolved/generation-snapshot",
            ) as download,
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.GenerationConfig.from_pretrained",
                return_value=loaded,
            ) as from_pretrained,
        ):
            config = get_generation_config(
                "org/generation-model",
                trust_remote_code=True,
                revision="generation-revision",
            )

        download.assert_called_once_with(
            "org/generation-model",
            revision="generation-revision",
            ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
        )
        from_pretrained.assert_called_once_with(
            "/resolved/generation-snapshot",
            trust_remote_code=True,
        )
        self.assertIs(config, loaded)

    def test_generation_config_missing_snapshot_metadata_returns_none(self) -> None:
        with (
            patch(
                "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                return_value=nullcontext(),
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                side_effect=OSError("generation metadata is unavailable"),
            ),
        ):
            config = get_generation_config(
                "org/generation-model-without-metadata",
                trust_remote_code=False,
                revision="missing-revision",
            )

        self.assertIsNone(config)

    def test_tokenizer_uses_revision_pinned_snapshot(self) -> None:
        events: list[str] = []

        tokenizer = SimpleNamespace(get_added_vocab=dict, init_kwargs={})

        def download_snapshot(*args, **kwargs):
            events.append("snapshot")
            return "/resolved/tokenizer-snapshot"

        def load_tokenizer(*args, **kwargs):
            events.append("parse")
            return tokenizer

        with (
            patch(
                "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                return_value=_RecordingLock(events),
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                side_effect=download_snapshot,
            ) as download,
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.AutoTokenizer.from_pretrained",
                side_effect=load_tokenizer,
            ) as from_pretrained,
            patch("tokenspeed.runtime.utils.hf_transformers_utils.warnings.warn"),
        ):
            result = get_tokenizer(
                "org/tokenizer",
                trust_remote_code=False,
                revision="tokenizer-revision",
            )

        download.assert_called_once_with(
            "org/tokenizer",
            revision="tokenizer-revision",
            ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
        )
        from_pretrained.assert_called_once_with(
            "/resolved/tokenizer-snapshot",
            trust_remote_code=False,
            clean_up_tokenization_spaces=False,
        )
        self.assertEqual(events, ["acquire", "snapshot", "release", "parse"])
        self.assertIs(result, tokenizer)
        self.assertEqual(result.name_or_path, "org/tokenizer")
        self.assertEqual(result.init_kwargs["name_or_path"], "org/tokenizer")

    def test_remote_code_tokenizer_uses_repo_and_commit_under_lock(self) -> None:
        commit = "b" * 40
        snapshot = f"/cache/models--org--tokenizer/snapshots/{commit}"
        events: list[str] = []

        tokenizer = SimpleNamespace(get_added_vocab=dict, init_kwargs={})

        def download_snapshot(*args, **kwargs):
            events.append("snapshot")
            return snapshot

        def load_tokenizer(*args, **kwargs):
            events.append("parse")
            return tokenizer

        with (
            patch(
                "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                return_value=_RecordingLock(events),
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                side_effect=download_snapshot,
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.AutoTokenizer.from_pretrained",
                side_effect=load_tokenizer,
            ) as from_pretrained,
            patch("tokenspeed.runtime.utils.hf_transformers_utils.warnings.warn"),
        ):
            result = get_tokenizer(
                "org/tokenizer",
                trust_remote_code=True,
                revision="tokenizer-revision",
            )

        from_pretrained.assert_called_once_with(
            "org/tokenizer",
            trust_remote_code=True,
            clean_up_tokenization_spaces=False,
            revision=commit,
        )
        self.assertEqual(events, ["acquire", "snapshot", "parse", "release"])
        self.assertIs(result, tokenizer)

    def test_remote_code_tokenizer_without_commit_falls_back_to_snapshot(self) -> None:
        events: list[str] = []

        tokenizer = SimpleNamespace(get_added_vocab=dict, init_kwargs={})

        def download_snapshot(*args, **kwargs):
            events.append("snapshot")
            return "/resolved/nonstandard-layout"

        def load_tokenizer(*args, **kwargs):
            events.append("parse")
            return tokenizer

        with (
            patch(
                "tokenspeed.runtime.model_loader.weight_utils.get_lock",
                return_value=_RecordingLock(events),
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                side_effect=download_snapshot,
            ),
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.AutoTokenizer.from_pretrained",
                side_effect=load_tokenizer,
            ) as from_pretrained,
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.logger.warning"
            ) as warn,
            patch("tokenspeed.runtime.utils.hf_transformers_utils.warnings.warn"),
        ):
            get_tokenizer(
                "org/tokenizer",
                trust_remote_code=True,
                revision="moving-branch",
            )

        from_pretrained.assert_called_once_with(
            "/resolved/nonstandard-layout",
            trust_remote_code=True,
            clean_up_tokenization_spaces=False,
        )
        self.assertEqual(events, ["acquire", "snapshot", "release", "parse"])
        warn.assert_called_once()

    def test_local_tokenizer_does_not_resolve_remote_snapshot(self) -> None:
        tokenizer = SimpleNamespace(get_added_vocab=dict, init_kwargs={})
        with (
            tempfile.TemporaryDirectory() as snapshot,
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                side_effect=AssertionError("local tokenizers must not download"),
            ) as download,
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as from_pretrained,
            patch("tokenspeed.runtime.utils.hf_transformers_utils.warnings.warn"),
        ):
            get_tokenizer(snapshot, trust_remote_code=True)

        download.assert_not_called()
        from_pretrained.assert_called_once_with(
            snapshot,
            trust_remote_code=True,
            clean_up_tokenization_spaces=False,
        )

    def test_longcat_tokenizers_enable_mistral_regex_fix(self) -> None:
        for architecture in (
            "FLASHLocalForCausalLM",
            "LongcatCausalLM",
        ):
            with self.subTest(architecture=architecture):
                tokenizer = SimpleNamespace(get_added_vocab=dict, init_kwargs={})
                with (
                    tempfile.TemporaryDirectory() as snapshot,
                    patch(
                        "tokenspeed.runtime.utils.hf_transformers_utils.AutoTokenizer.from_pretrained",
                        return_value=tokenizer,
                    ) as from_pretrained,
                    patch(
                        "tokenspeed.runtime.utils.hf_transformers_utils.warnings.warn"
                    ),
                ):
                    get_tokenizer(snapshot, architectures=[architecture])

                from_pretrained.assert_called_once_with(
                    snapshot,
                    trust_remote_code=False,
                    clean_up_tokenization_spaces=False,
                    fix_mistral_regex=True,
                )

    def test_tokenizer_rejects_conflicting_revisions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            get_tokenizer(
                "org/tokenizer",
                tokenizer_revision="tokenizer-revision",
                revision="model-revision",
            )

    def test_deepseek_encoding_missing_from_snapshot_fails_locally(self) -> None:
        with (
            tempfile.TemporaryDirectory() as snapshot,
            patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.cached_file",
                side_effect=AssertionError("must not re-resolve a local snapshot"),
            ) as cached,
            self.assertRaisesRegex(RuntimeError, "encoding/encoding_dsv4.py"),
        ):
            _find_deepseek_v4_encoding_file(snapshot, None)

        cached.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

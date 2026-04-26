"""
Unit tests for ``BaseVectorStorage._generate_collection_suffix`` length safety.

These tests verify that the table/collection suffix produced from
``embedding_func.model_name`` is bounded so the resulting PostgreSQL
identifiers (and any backend with similar limits) never get silently
truncated by ``NAMEDATALEN``.

Background: when the suffix combined with the longest base table name
``LIGHTRAG_VDB_RELATION`` and the explicit ``{table_name}_PK`` constraint
name embedded in the DDL exceeds 63 bytes, PostgreSQL silently truncates
the constraint name. The truncated name can then collide with the table
itself, producing a misleading ``relation already exists`` error on
``CREATE TABLE``.
"""

import pytest

from lightrag.base import BaseVectorStorage
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


def _make_storage(model_name: str | None, embedding_dim: int) -> BaseVectorStorage:
    """Build a minimal BaseVectorStorage subclass instance for suffix tests.

    BaseVectorStorage is abstract; we only need ``_generate_collection_suffix``
    which depends solely on ``embedding_func``. Provide stub implementations
    for the abstract members so the class can be instantiated.
    """

    async def _noop_embed(_texts):
        return []

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        func=_noop_embed,
        model_name=model_name,
    )

    class _Stub(BaseVectorStorage):
        async def query(self, query, top_k, query_embedding=None):
            return []

        async def upsert(self, data):
            return None

        async def delete_entity(self, entity_name):
            return None

        async def delete_entity_relation(self, entity_name):
            return None

        async def get_by_id(self, id):
            return None

        async def get_by_ids(self, ids):
            return []

        async def get_vectors_by_ids(self, ids):
            return {}

        async def delete(self, ids):
            return None

        async def drop(self):
            return {}

        async def index_done_callback(self):
            return None

    return _Stub(
        namespace="entities",
        workspace="",
        global_config={
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.2,
            },
        },
        embedding_func=embedding_func,
    )


# ---------------------------------------------------------------------------
# Behavior of ``_generate_collection_suffix``
# ---------------------------------------------------------------------------


class TestCollectionSuffixLength:
    """Suffix should always stay short enough to avoid PG truncation."""

    # The longest base table prefix used by the Postgres backend is
    # ``LIGHTRAG_VDB_RELATION`` (21 chars). With the underscore separator
    # plus the ``_PK`` constraint suffix, only 38 chars are available for
    # the model+dim suffix.
    LONGEST_BASE_PREFIX = "LIGHTRAG_VDB_RELATION_"
    PK_SUFFIX = "_PK"
    PG_NAMEDATALEN = 63

    def _full_constraint_name(self, suffix: str) -> str:
        return f"{self.LONGEST_BASE_PREFIX}{suffix}{self.PK_SUFFIX}"

    def test_returns_none_without_model_name(self):
        storage = _make_storage(model_name=None, embedding_dim=1024)
        assert storage._generate_collection_suffix() is None

    def test_short_model_name_unchanged(self):
        """Short model names produce the exact legacy suffix (backward compat)."""
        storage = _make_storage(model_name="bge-m3", embedding_dim=1024)
        assert storage._generate_collection_suffix() == "bge_m3_1024d"

    def test_existing_format_preserved_for_text_embedding_3_large(self):
        """Documented example from the existing docstring stays stable."""
        storage = _make_storage(
            model_name="text-embedding-3-large", embedding_dim=3072
        )
        assert (
            storage._generate_collection_suffix() == "text_embedding_3_large_3072d"
        )

    def test_long_model_name_is_capped(self):
        """The Qwen3 official ID triggered the original bug: must be capped."""
        storage = _make_storage(
            model_name="text-embedding-qwen3-embedding-0.6b",
            embedding_dim=1024,
        )
        suffix = storage._generate_collection_suffix()
        assert suffix is not None
        # Suffix must fit in the 38-char budget.
        assert len(suffix) <= 38

    def test_full_constraint_name_fits_in_namedatalen(self):
        """``{longest_base}_{suffix}_PK`` must stay within 63 bytes."""
        storage = _make_storage(
            model_name="text-embedding-qwen3-embedding-0.6b",
            embedding_dim=1024,
        )
        suffix = storage._generate_collection_suffix()
        full = self._full_constraint_name(suffix)
        assert len(full.encode("utf-8")) <= self.PG_NAMEDATALEN

    @pytest.mark.parametrize(
        "model_name,embedding_dim",
        [
            ("text-embedding-qwen3-embedding-0.6b", 1024),
            ("snowflake-arctic-embed-l-v2.0", 1024),
            ("nvidia/nv-embed-v2", 4096),
            ("a" * 200, 1536),  # Pathologically long
        ],
    )
    def test_real_world_long_model_names_fit(self, model_name, embedding_dim):
        storage = _make_storage(model_name=model_name, embedding_dim=embedding_dim)
        suffix = storage._generate_collection_suffix()
        full = self._full_constraint_name(suffix)
        assert (
            len(full.encode("utf-8")) <= self.PG_NAMEDATALEN
        ), f"{full!r} exceeds NAMEDATALEN-1"

    def test_deterministic_for_same_input(self):
        s1 = _make_storage(
            model_name="text-embedding-qwen3-embedding-0.6b", embedding_dim=1024
        )._generate_collection_suffix()
        s2 = _make_storage(
            model_name="text-embedding-qwen3-embedding-0.6b", embedding_dim=1024
        )._generate_collection_suffix()
        assert s1 == s2

    def test_different_models_produce_different_suffixes(self):
        s1 = _make_storage(
            model_name="text-embedding-qwen3-embedding-0.6b", embedding_dim=1024
        )._generate_collection_suffix()
        s2 = _make_storage(
            model_name="text-embedding-qwen3-embedding-0.7b", embedding_dim=1024
        )._generate_collection_suffix()
        assert s1 != s2

    def test_different_dims_produce_different_suffixes(self):
        s1 = _make_storage(
            model_name="text-embedding-qwen3-embedding-0.6b", embedding_dim=1024
        )._generate_collection_suffix()
        s2 = _make_storage(
            model_name="text-embedding-qwen3-embedding-0.6b", embedding_dim=768
        )._generate_collection_suffix()
        assert s1 != s2

    def test_suffix_keeps_dim_marker(self):
        suffix = _make_storage(
            model_name="text-embedding-qwen3-embedding-0.6b", embedding_dim=1024
        )._generate_collection_suffix()
        assert suffix is not None
        assert suffix.endswith("_1024d")

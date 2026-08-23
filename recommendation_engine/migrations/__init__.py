"""Numbered raw-SQL migrations for the `recommendations` schema.

Raw SQL with a small runner, matching `creditProxy/migrations/001_init.sql`,
rather than Alembic: there is no ORM here, so autogenerate has nothing to diff,
and hand-written vector DDL (HNSW build parameters, partial indexes, extension
guards) is exactly the kind of thing Alembic renders less readable.
"""

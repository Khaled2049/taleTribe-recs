"""Catalog ingest: read published stories, normalize, embed, load.

Every item is a TaleTribe story owned by story-data; `recommendations.items`
keys back to it by `story_id`. The CMU Book Summary Corpus that used to seed
this catalog for cold start is gone, along with its parser and genre crosswalk.
"""

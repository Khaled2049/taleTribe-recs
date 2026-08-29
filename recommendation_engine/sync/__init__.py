"""Reader signals, and everything derived from them.

Two producers, one table:

* **story-data** derives real signals from `story_likes`, `story_ratings` and
  `reading_progress` into `recommendations.interactions`
  (`story-data sync-recs`). It owns that path because `reading_progress` is
  private per-user data this service is not permitted to read — the grant in
  migration 000020 enforces it.
* **`synthetic.py`** generates `synth_`-prefixed readers, which `interactions.load()`
  writes. That prefix is what keeps the two producers from treading on each
  other: story-data's sync neither deletes nor derives over synthetic rows.

Everything downstream is this package's, and never leaves the schema:
`stats.py` recomputes `item_stats` (including `pop_score`), `user_taste` and
`item_cooccurrence` from `interactions` alone. Run it with
`seed.py --refresh-only` after a story-data sync.
"""

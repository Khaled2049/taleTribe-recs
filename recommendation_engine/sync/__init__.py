"""Getting reader behaviour into Postgres.

Deliberately split into a **source-agnostic loader** and pluggable sources:

    synthetic.py  ─┐
                   ├──→ InteractionRecord ──→ interactions.load()  ──→ interactions
    firestore.py  ─┘         (canonical)      stats.refresh()      ──→ item_stats
     (Phase 3)                                                     ──→ user_taste

The point of the split is that swapping synthetic seed data for real Firestore
signals changes *only the source*. Completion inference, engagement weighting,
popularity aggregation and taste-vector construction are written and tested once,
against whichever source is plugged in.

The canonical record is modelled on what Firestore actually offers — binary likes,
create-only 1-5 ratings, and reading-progress *state* — rather than on an idealized
event log the platform does not have. A synthetic source that emitted richer data
than production can supply would be building on sand.
"""

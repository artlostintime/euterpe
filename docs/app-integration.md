# App Integration Data Flow

How the local-first recommender embeds into a music streaming app:
what the app collects, what stays on-device, and how the catalog model
improves over time without any user's device ever training a model.

## 1. What the app collects

Every interaction becomes one `observe()` call on the LocalProfile:

| Event | Signal | Weight |
|---|---|---|
| Track played to >=90% | `completion` | 1.2 |
| Normal play | `play` | 1.0 |
| Replay within session | `replay` | 1.0 |
| Like / heart | `like` | 2.0 |
| Save to library | `save` | 1.5 |
| Add to playlist | `playlist_add` | 1.5 |
| Skim (<30s of a >2min track) | `short_play` | 0.3 |
| Skip | `skip` | 0.0 |

Each call stores `(item_id, timestamp, signal)`. That is the entire
collection surface. No audio, no location, no contacts, no device
fingerprint — the model needs nothing else.

Timestamps drive two more on-device structures, both learned for free:

- **ListenRhythm** (`src/context.py`): 24-hour and day-of-week
  listening histograms. Boosts repeat-leaning scores at the user's
  habitual listening times (x1.0-x1.1), neutral otherwise.
- **Exploration preference** (`profile.exploration`, 0-1): the
  discovery/repeat blend. App can expose this as a user setting
  ("Familiar / Mix / Discover more") or learn it from skip-vs-save
  ratios later; default 0.3.

## 2. What stays on-device

Everything in section 1. The profile JSON, rhythm histograms, and all
scoring happen locally:

```
[app event] ──> profile.observe(item, ts, signal)   # instant, free
                      │
                      ▼
   recommend() at refresh time:
     repeat score    = decayed weighted counts      (per-user, online)
     discovery score = cosine(taste_center, item)   (catalog PQ model)
     context weight  = rhythm histogram             (per-user, online)
     fused           = w_repeat*repeat + w_disc*discovery
```

Per-user learning is **online and instant**: one `like` changes the
next recommendation. No training, no kernel, no GPU. This half carries
most of measured quality (75% hitrate@20 on repeats comes from this
signal).

## 3. What cannot learn on-device — and the loop that replaces it

Collaborative embeddings (item2vec) are trained **across users**, so
a single device can never update them. The update path is the
contribution -> retrain -> ship loop:

```
USER DEVICE (opt-in only)                FLEET SERVER
─────────────────────────                ────────────
--contribute flag set by user
        │
        ▼
ContributionBuilder.build():
  consent gate (default off)
  min_count >= 5
  cap 200 items by count
  timestamps -> YYYY-MM buckets
  counts only, no user id
        │
        ▼  (one small JSON, user-initiated upload)
   contribution.v1 payload  ──────────>  aggregate with all
                                             other contributors
                                                 │
                                                 ▼
                                     periodic item2vec retrain
                                     (same pipeline as the
                                     MLHD+ kernels)
                                                 │
                                                 ▼
                                     PQ-quantize -> sonata_pq.npz
                                     (~43 MB, 16x compression)
                                                 │
                                                 ▼
   app update ships new model file  <───────────┘
   (device swaps the .npz; no on-device training ever)
```

Consent model: the flag IS the opt-in action. No contribution is
built or transmitted without it, and the payload is deliberately not
the profile format (counts + month buckets, nothing else).

## 4. Serving-time behavior

**Cold start (new user):** popularity ranking + default rhythm until
~10 plays, then personal decay scores take over. The eligibility
threshold in the eval protocol was chosen for exactly this regime.

**New releases (no embedding yet):** songs published after the last
retrain are invisible to discovery. The app needs a fresh-content
candidate source — new-release popularity tier merged into the
candidate set — until the next model update. (Not yet implemented;
see §6.)

**Feedback loops:** once recommendations are served, plays stop being
organic. The app should log exposure (which items were recommended
and shown) alongside plays, and the ranking should apply a fatigue
penalty to recently-recommended-but-ignored items. Without this the
repeat engine eats its own output. (Not yet implemented; see §6.)

**Latency budget (measured, i3-12100F):** cold start 0.24-0.41 s,
warm recommend 536-723 ms p50 at 512-535 MB RSS on the PQ tiers.
A refresh-triggered recommend fits comfortably in a background task.

## 5. Model artifacts shipped with the app

| Artifact | Size | Purpose |
|---|---|---|
| sonata_pq.npz | 42.8 MB | default discovery tier (16x, ANN recall 0.36) |
| etude_pq.npz | 20.9 MB | low-storage tier (32x, ANN recall 0.17) |
| mbid_index.bin | 53.5 MB | MBID -> item_id resolver (binary search) |
| top_items.json | 0.2 MB | cold-start popularity + name bridge |

Raw f32 (684.5 MB) is a desktop/server option, not a mobile one.

## 6. Honest gap list (product work, not research)

1. **Retrain on app data**: embeddings are MLHD+-catalog-specific.
   First fleet retrain should use the app's own listening logs —
   the pipeline is the product, not the .npz file.
2. **Fresh-content tier** for post-retrain releases (see §4).
3. **Exposure logging + fatigue penalty** (see §4).
4. **Signal availability**: like/save/skip weights exist, but
   ListenBrainz-style exports rarely carry them; in-app collection
   (§1) fixes this — one more reason the app is the right home.
5. **A/B evaluation**: offline protocol is frozen; serving quality
   needs online experiments (exposure-conditional metrics).

## 7. Privacy summary

- Raw event log: never leaves the device.
- Contribution payload: opt-in, minimized (top-200 items, counts,
  month buckets), no identifiers, no raw timestamps.
- Catalog model: contains no user data at all.
- The only network calls: contribution upload (user-initiated) and
  app updates shipping new model files.

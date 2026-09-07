# Validation benchmark report and internal replay boundary

The canonical benchmark workflow ID is `validation`; do not create or route to
a `validation-fixed` lane. Its adapter normalizes the exact preparation,
accepted plan, native operation results, assessment/review, terminal receipt,
template statuses, issue codes, dependency states, logs, and artifact links
into `bundle/run.json`.

Score the completed bundle through the shared benchmark boundary. The stable
artifact is `suite_result.json`; `report.html` is a portable presentation with
embedded, sanitized current/reference evidence and JSON artifact links:

```bash
python -m content_benchmark.cli score \
  --workflow validation \
  --bundle /path/to/run/bundle/run.json \
  --data-root "$PWD/.data/regression"
```

Only exact media re-derived from digest-bound operation evidence may populate
`renders`, `references`, scored thumbnails, or the report. Preserve the source
and image digests plus OVRTX metadata. Missing renderer, VLM, or evidence state
must remain explicit; a diagnostic, generic, unbound, or stale image is never a
visual pass.

Representative replay-video generation is internal-only. The internal
benchmark and `$content-workflow-cli-demo` implementations are excluded from
public staging, and public Validation must not encode video or require FFmpeg.
Use `report.html`, its digest-bound still images, and JSON artifact links as the
portable public presentation. In an internal checkout, any separately requested
replay must still fail closed without current accepted OVRTX evidence, preserve
digest bindings, sanitize command paths, and identify itself as a
representative replay rather than a live run.

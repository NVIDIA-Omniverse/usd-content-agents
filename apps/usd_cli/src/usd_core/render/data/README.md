# Studio environment map

`studio.exr` is the bundled Content Workbench-compatible environment map used
for lightless OVRTX scenes. It is [Studio Small 08](https://polyhaven.com/a/studio_small_08) by Sergej
Majboroda (Poly Haven, CC0 1.0), as re-encoded in the NVIDIA `ovrtx`
0.4.1.364340 test data (`ovrtx/tests/docs/data/studio.exr`); its SHA-256 is
`f0379ca1056f578b0081fc1d80b702d61e7a79d5c8000a030d50e9ada1cee539`. See
`ASSET_NOTICES.md`.

The bundled map defaults to intensity 600. An explicitly configured custom map
defaults to intensity 1 unless `OVRTX_DEFAULT_HDRI_INTENSITY` (or its legacy
World Understanding alias) is set.

# Studio environment map

`studio.exr` is the bundled Content Workbench-compatible environment map used
for lightless OVRTX scenes. It is a byte-for-byte copy of the NVIDIA-owned
OVRTX package fixture `ovrtx/tests/docs/data/studio.exr`; no modifications were
made. The NVIDIA-owned asset is provided under the repository's Apache-2.0
license. Its SHA-256 is
`f0379ca1056f578b0081fc1d80b702d61e7a79d5c8000a030d50e9ada1cee539`.

The bundled map defaults to intensity 600. An explicitly configured custom map
defaults to intensity 1 unless `OVRTX_DEFAULT_HDRI_INTENSITY` (or its legacy
World Understanding alias) is set.

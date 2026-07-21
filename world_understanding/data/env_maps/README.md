# Environment Maps

`studio.exr` is the active OVRTX default environment map for injected
`DomeLight` lighting. The renderer binds it at intensity `600` for lightless
scenes. It was copied from the NVIDIA-owned OVRTX package fixture
`ovrtx/tests/docs/data/studio.exr`; the vendored file checksum is
`f0379ca1056f578b0081fc1d80b702d61e7a79d5c8000a030d50e9ada1cee539`.

`SmartMaterials_Environment_with_Lights.exr` is retained as a historical
Kit-parity and debugging reference for comparing OVRTX behavior against the
older Kit rendering path. It is no longer the active OVRTX default.

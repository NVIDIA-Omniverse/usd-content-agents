# Content Agents Applications

The public applications are organized by agent and deployment surface:

| Agent | Local CLI and configuration | REST service | Release scope |
|---|---|---|---|
| Material | [`material_agent/`](material_agent/) | [`material_agent_service/`](material_agent_service/) | Supported |
| Physics | [`physics_agent/`](physics_agent/) | [`physics_agent_service/`](physics_agent_service/) | Supported |
| Joint | [`joint_agent/`](joint_agent/) | [`joint_agent_service/`](joint_agent_service/) | Opt-in 0.5 Research Preview |
| Texture | [`texture_agent/`](texture_agent/) | [`texture_agent_service/`](texture_agent_service/) | Supported |
| Validation | [`validation_agent/`](validation_agent/) | Not shipped | CLI/Python-contract 0.5 Research Preview |

Shared service components include the
[`ovrtx_rendering_api/`](ovrtx_rendering_api/) renderer and the public Texture
Variation adapters under `texture_gen_*_service/`. Start with the
[repository README](../README.md) for installation, quickstarts, and the
supported deployment topology.

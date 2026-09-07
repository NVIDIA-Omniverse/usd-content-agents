# Tire Bounce Guidance

Treat the behavior prompt and any supplied reference images as the behavior
contract. A valid result must show the tire leave the ground after first
contact, then tip or rotate naturally before coming to rest. Ground contact
combined with rotation or a rising rigid-body origin is not an airborne
rebound.

Inspect temporal evidence across the complete candidate motion, including the
drop, first contact, rebound apex, tipping, and final rest. Do not accept a
candidate only because it settles. Preserve the tire's existing rigid-body
hierarchy and `convexDecomposition` collider so the wheel opening remains
physically meaningful.

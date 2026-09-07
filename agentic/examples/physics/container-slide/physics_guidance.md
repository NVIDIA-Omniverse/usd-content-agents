# Container Slide Guidance

During tuning candidate review, judge the complete motion, not only the final
pose. The closed container must move in the +X direction after the initial
push, lose speed smoothly through friction, and come naturally to rest. Reject
candidates that remain stationary, accelerate, reverse direction, bounce, tip
over, or penetrate the floor.

The pre-tuning visual pass checks baseline authoring and runtime safety only.
Do not require the +X slide until a tuning sweep applies this scenario's
initial velocity.

Treat the lid and body as one rigid assembly throughout the recording. Preserve
the normalized rigid-body hierarchy and both existing `convexHull` colliders;
do not change collision topology to improve the score.

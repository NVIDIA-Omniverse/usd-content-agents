# Engine auxiliary-body and gravity-witness review

This is a separate conservative diagnostic. It changes no frozen evaluator, submission, source, v1 verdict, v1.1 verdict, or acceptance threshold.

The controlling assessment is `evaluation_adjudications/pilot-v1/06_engine/plain_astra/measurement_review/assessment.json`, SHA256 `0ac817cf65108c5c6e29433feb94db6e5797b19f22c65cf7f0eecc81310d0cc7`. It covers all six remaining v1.1 rejections and recommends **INCONCLUSIVE**, with no acceptance promotion or demonstrated false pass.

The published task does not unambiguously forbid an auxiliary moving body without original source visuals. It requires all bodies and joints to be declared, permits invisible collision approximations, and preserves all24 source instances. The additional universal source-backed-moving-body gate is present only in the frozen implementation. The original auxiliary body remains unchanged and its acceptance is not retroactively asserted.

All five independent loaded engine runs and the paired unloaded seed11 completed2880steps; engine motion, source, limits, closure, mass/inertia, speed, finite-state and paired-load gates passed. The other five failures concern the evaluator-owned freefall witness.

Synthetic native OvPhysX0.4.13 CPU diagnostics reproduce the rejected displacement exactly. With an unrelated fixed-joint body at128 TGS position iterations, witness `[1000,1000,1000]` falls only0.015625m at0.1s while its vertical speed correctly reaches-0.981000304m/s. Moving only that witness to `[1,1,1]` yields0.051094055m displacement, passing the original4mm displacement and0.03m/s velocity tolerances. Six unconstrained controls pass. The exact internal native cause is not established; the measured origin/constraint interaction is sufficient to invalidate this fixture result as evidence of a submitted-asset gravity failure.

`contract_review.json` is the read-only contract/native evidence capture written before the final synthetic contrast finished. `gravity_origin_probe_r2/results.json` contains six controls; `gravity_origin_probe_r3/results.json` contains the two constrained contrasts. The controlling assessment binds each script, scene, log, result, original acceptance, prior adjudication, and native evidence by SHA256, with exact parent/child commands.

The initial single-process multi-SDK fixture attempt failed during SDK reinitialization after its first control. Its script/log remain as unscored infrastructure evidence. Later probes use one SDK lifetime per child process. Its own7.8GB core dump was deleted; no author or frozen evidence was removed.

# Contact units erratum — acceptance unchanged

The frozen drawer01 solver and task remain byte-for-byte unchanged. Subsequent independent calibration of ovphysx0.4.13 on the same host established that `read_force_matrix` after `step_sync` returns **contact impulse in N·s**, despite the API docstring stating automatic conversion to force.

In drawer01 traces, the field `payload_drawer_contact_force_n` therefore contains raw contact impulse, **not newtons**. Divide by the frozen timestep, 1/240s, to obtain average contact force over that step. The frozen detector uses `norm(raw_matrix) > 0.01`; its effective threshold is **0.01N·s**, equivalent to **2.4N** at240Hz. The criterion still counts at least24 such contact samples. No predicate, threshold, solver, contract, or scored outcome has been changed.

Calibration is preserved in `general/case04_contact_units_evidence.json` and its executable probe `general/case04_contact_units_probe.py`. A supported1kg block plus1N downward force gave0.045041669N·s at1/240s and0.090083337N·s at1/120s, matching10.81N×dt. A2kg block plus1N gave0.085916672N·s at1/240s, matching20.62N×dt. The matrix matched raw contact-report impulses in all three trials, with relative conversion error below0.00001%.

This is a units-label correction and documentation of the existing detector. The calibration fixtures are nonscored and do not constitute a drawer or gripper source-asset trial.

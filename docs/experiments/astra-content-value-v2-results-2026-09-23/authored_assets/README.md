# Exact submitted drawer and conveyor assets

This companion contains both arms of the drawer and conveyor experiment:
11 submitted USD, bindings and texture files, **4,047,812 bytes**. The manifest
binds every file to its original retained archive member. It preserves failed or
successful task meaning separately: this package itself makes no acceptance claim.

Open `outputs/<case>/<arm>/final.usd`; its `bindings.json` is alongside it. The
plain drawer uses three relative texture dependencies retained in `textures/`.
The other three USD assets have no authored external asset dependencies. No asset
relocation or rewriting was performed. The files were inspected as uncomposed USD
layers with USD 25.5; runtime compatibility and physical acceptance are separate.

Run `python verify.py .` to check exact complete bundle membership and hashes.
Optional `--usd-review` additionally requires `usd-core==25.5` and rechecks the
full decoded-layer text hashes used by the private metadata review. No author
scripts, model/session data, credentials or raw logs are included.

Read [source notices](notices/SOURCE_NOTICES.md) and
[modifications](notices/MODIFICATIONS.md). The exact source/reference data and
frozen evaluator are published separately. This four-output subset is **not a
complete ten-case replay bundle** and does not contain cases05/06 geometry.

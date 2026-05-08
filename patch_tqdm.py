#!/usr/bin/env python3
"""
Patch notebook to add tqdm progress bars where relevant,
WITHOUT touching timing-sensitive measurement code.

Changes:
1. train_one_epoch: add use_tqdm/epoch params, wrap batch loop with tqdm
2. run_experiment: wrap epoch loop with tqdm, pass use_tqdm to train/eval,
   use tqdm.write() for prints inside the loop
3. log_layer_norm_gamma: use tqdm.write() instead of print()
"""

import json

filename = "notebook7b3ccadf89 (1).ipynb"

with open(filename, "r") as f:
    data = json.load(f)

patched = []

for cell in data["cells"]:
    if cell["cell_type"] != "code":
        continue

    source = "".join(cell["source"])

    # ── 1. train_one_epoch: add use_tqdm + epoch params, wrap loader ────────
    old_toe_sig = (
        'def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,\n'
        '                    device=DEVICE):\n'
    )
    new_toe_sig = (
        'def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,\n'
        '                    device=DEVICE, use_tqdm=False, epoch=None):\n'
    )
    if old_toe_sig in source:
        source = source.replace(old_toe_sig, new_toe_sig)
        patched.append("train_one_epoch signature")

    old_toe_loop = (
        '    for inputs, labels in loader:\n'
        '        inputs, labels = inputs.to(device), labels.to(device)\n'
        '        optimizer.zero_grad()\n'
    )
    new_toe_loop = (
        '    desc = f"Epoch {epoch}" if epoch is not None else "Train"\n'
        '    pbar = tqdm(loader, desc=desc, leave=False) if use_tqdm else loader\n'
        '    for inputs, labels in pbar:\n'
        '        inputs, labels = inputs.to(device), labels.to(device)\n'
        '        optimizer.zero_grad()\n'
    )
    if old_toe_loop in source and 'def train_one_epoch' in source:
        source = source.replace(old_toe_loop, new_toe_loop)
        patched.append("train_one_epoch batch loop")

    # ── 2. run_experiment: wrap epoch range with tqdm ───────────────────────
    old_epoch_loop = (
        '    for epoch in range(1, num_epochs + 1):\n'
        '        train_metrics = train_one_epoch(\n'
        '            model, train_loader, optimizer, scheduler, loss_fn, device\n'
        '        )\n'
        '        val_metrics = evaluate(model, test_loader, loss_fn, device=device)\n'
    )
    new_epoch_loop = (
        '    epoch_iter = tqdm(range(1, num_epochs + 1), desc=run_id, unit="ep") if verbose else range(1, num_epochs + 1)\n'
        '    for epoch in epoch_iter:\n'
        '        train_metrics = train_one_epoch(\n'
        '            model, train_loader, optimizer, scheduler, loss_fn, device,\n'
        '            use_tqdm=False, epoch=epoch\n'
        '        )\n'
        '        val_metrics = evaluate(model, test_loader, loss_fn, device=device, use_tqdm=False)\n'
    )
    if old_epoch_loop in source:
        source = source.replace(old_epoch_loop, new_epoch_loop)
        patched.append("run_experiment epoch loop + tqdm")

    # ── 3. Update epoch_pbar postfix after val ──────────────────────────────
    old_best_cra = (
        "        best_cra    = max(best_cra, val_metrics['cra'])\n"
        "\n"
        "        emp_lip = None\n"
    )
    new_best_cra = (
        "        best_cra    = max(best_cra, val_metrics['cra'])\n"
        "        if verbose and hasattr(epoch_iter, 'set_postfix'):\n"
        "            epoch_iter.set_postfix(\n"
        "                loss=f\"{train_metrics['loss']:.3f}\",\n"
        "                acc=f\"{train_metrics['accuracy']:.1f}%\",\n"
        "                cra=f\"{val_metrics['cra']:.1f}%\",\n"
        "            )\n"
        "\n"
        "        emp_lip = None\n"
    )
    if old_best_cra in source:
        source = source.replace(old_best_cra, new_best_cra)
        patched.append("run_experiment epoch postfix")

    # ── 4. log_layer_norm_gamma: print → tqdm.write ─────────────────────────
    old_log = 'print(f"  [Ep {epoch}] LayerNorm γ ‖γ‖₂ = {gamma_norm:.4f}  ({name})")'
    new_log = 'tqdm.write(f"  [Ep {epoch}] LayerNorm γ ‖γ‖₂ = {gamma_norm:.4f}  ({name})")'
    if old_log in source:
        source = source.replace(old_log, new_log)
        patched.append("log_layer_norm_gamma tqdm.write")

    # ── 5. run_experiment prints inside epoch loop → tqdm.write ─────────────
    old_lip_print = 'print(f"  [Ep {epoch}] Empirical Lipschitz ≈ {emp_lip:.4f}")'
    new_lip_print = 'tqdm.write(f"  [Ep {epoch}] Empirical Lipschitz ≈ {emp_lip:.4f}")'
    if old_lip_print in source:
        source = source.replace(old_lip_print, new_lip_print)
        patched.append("empirical lipschitz tqdm.write")

    old_ep_print = (
        '        if verbose and (epoch % 10 == 0 or epoch == 1):\n'
        '            print(f"  Ep {epoch:3d}/{num_epochs} | "\n'
    )
    new_ep_print = (
        '        if verbose and (epoch % 10 == 0 or epoch == 1):\n'
        '            tqdm.write(f"  Ep {epoch:3d}/{num_epochs} | "\n'
    )
    if old_ep_print in source:
        source = source.replace(old_ep_print, new_ep_print)
        patched.append("epoch summary tqdm.write")

    # ── 6. Final summary print → tqdm.write ─────────────────────────────────
    old_final = (
        '    if verbose:\n'
        '        print(f"\\n  → Best CRA: {best_cra:.2f}%  "\n'
        '              f"Final acc: {history[-1][\'val_accuracy\']:.2f}%")\n'
    )
    new_final = (
        '    if verbose:\n'
        '        tqdm.write(f"\\n  → Best CRA: {best_cra:.2f}%  "\n'
        '                   f"Final acc: {history[-1][\'val_accuracy\']:.2f}%")\n'
    )
    if old_final in source:
        source = source.replace(old_final, new_final)
        patched.append("final summary tqdm.write")

    # ── Write back as list of lines ─────────────────────────────────────────
    lines = [line + '\n' for line in source.split('\n')]
    # Remove trailing empty line artifact
    if lines[-1] == '\n':
        lines[-1] = ''
    elif lines[-1].endswith('\n'):
        lines[-1] = lines[-1][:-1]
    if lines[-1] == '':
        lines.pop()

    cell["source"] = lines

with open(filename, "w") as f:
    json.dump(data, f, indent=1)

print(f"Patched: {patched}")
if not patched:
    print("WARNING: No changes were applied! Check string matching.")
else:
    print(f"Total patches applied: {len(patched)}")

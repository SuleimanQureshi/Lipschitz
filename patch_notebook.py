import json

filename = "notebook7b3ccadf89 (1).ipynb"

with open(filename, "r") as f:
    data = json.load(f)

for cell in data["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # 1. Imports
        if 'from typing import Optional, Type, Callable\n' in source and 'from tqdm.auto import tqdm' not in source:
            source = source.replace(
                'from typing import Optional, Type, Callable\n',
                'from typing import Optional, Type, Callable\nfrom tqdm.auto import tqdm\n'
            )
            
        # 2. evaluate
        if '    device: torch.device = DEVICE,\n):\n    model.eval()\n' in source:
            source = source.replace(
                '    device: torch.device = DEVICE,\n):\n    model.eval()\n    total_loss, correct, certified, total = 0.0, 0, 0, 0\n\n    for inputs, labels in loader:\n',
                '    device: torch.device = DEVICE,\n    use_tqdm: bool = False,\n):\n    model.eval()\n    total_loss, correct, certified, total = 0.0, 0, 0, 0\n\n    pbar = tqdm(loader, desc="Evaluate", leave=False) if use_tqdm else loader\n    for inputs, labels in pbar:\n'
            )
            
        # 3. log_layer_norm_gamma
        if '            print(f"  [Ep {epoch}] LayerNorm γ ‖γ‖₂ = {gamma_norm:.4f}  ({name})")' in source:
            source = source.replace(
                '            print(f"  [Ep {epoch}] LayerNorm γ ‖γ‖₂ = {gamma_norm:.4f}  ({name})")',
                '            tqdm.write(f"  [Ep {epoch}] LayerNorm γ ‖γ‖₂ = {gamma_norm:.4f}  ({name})")'
            )
            
        # 4. train_one_epoch
        if 'def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,\n                    device=DEVICE):\n' in source:
            source = source.replace(
                'def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,\n                    device=DEVICE):\n    model.train()\n    total_loss, correct, total = 0.0, 0, 0\n    total_grad_norm = 0.0\n    n_batches       = 0\n\n    for inputs, labels in loader:\n',
                'def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,\n                    device=DEVICE, use_tqdm=False, epoch=None):\n    model.train()\n    total_loss, correct, total = 0.0, 0, 0\n    total_grad_norm = 0.0\n    n_batches       = 0\n\n    pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False) if use_tqdm else loader\n    for inputs, labels in pbar:\n'
            )
            
        # 5. run_experiment loop setup
        if '    for epoch in range(1, num_epochs + 1):\n        train_metrics = train_one_epoch(\n            model, train_loader, optimizer, scheduler, loss_fn, device\n        )\n        val_metrics = evaluate(model, test_loader, loss_fn, device=device)\n' in source:
            source = source.replace(
                '    for epoch in range(1, num_epochs + 1):\n        train_metrics = train_one_epoch(\n            model, train_loader, optimizer, scheduler, loss_fn, device\n        )\n        val_metrics = evaluate(model, test_loader, loss_fn, device=device)\n',
                '    epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Epochs", leave=True) if verbose else range(1, num_epochs + 1)\n    for epoch in epoch_pbar:\n        train_metrics = train_one_epoch(\n            model, train_loader, optimizer, scheduler, loss_fn, device, use_tqdm=verbose, epoch=epoch\n        )\n        val_metrics = evaluate(model, test_loader, loss_fn, device=device, use_tqdm=verbose)\n'
            )
            
        # 6. run_experiment lip print
        if '            if verbose:\n                print(f"  [Ep {epoch}] Empirical Lipschitz ≈ {emp_lip:.4f}")' in source:
            source = source.replace(
                '            if verbose:\n                print(f"  [Ep {epoch}] Empirical Lipschitz ≈ {emp_lip:.4f}")',
                '            if verbose:\n                tqdm.write(f"  [Ep {epoch}] Empirical Lipschitz ≈ {emp_lip:.4f}")'
            )
            
        # 7. run_experiment metric print
        if '        if verbose and (epoch % 10 == 0 or epoch == 1):\n            print(f"  Ep {epoch:3d}/{num_epochs} | "\\\n                  f"loss={train_metrics[\'loss\']:.4f} "\\\n                  f"acc={train_metrics[\'accuracy\']:.1f}% | "\\\n                  f"grad={train_metrics[\'grad_norm\']:.4f} | "\\\n                  f"val_acc={val_metrics[\'accuracy\']:.1f}% "\\\n                  f"CRA={val_metrics[\'cra\']:.1f}%")\n' in source:
            source = source.replace(
                '        if verbose and (epoch % 10 == 0 or epoch == 1):\n            print(f"  Ep {epoch:3d}/{num_epochs} | "\\\n                  f"loss={train_metrics[\'loss\']:.4f} "\\\n                  f"acc={train_metrics[\'accuracy\']:.1f}% | "\\\n                  f"grad={train_metrics[\'grad_norm\']:.4f} | "\\\n                  f"val_acc={val_metrics[\'accuracy\']:.1f}% "\\\n                  f"CRA={val_metrics[\'cra\']:.1f}%")\n',
                '        if verbose and (epoch % 10 == 0 or epoch == 1):\n            tqdm.write(f"  Ep {epoch:3d}/{num_epochs} | "\\\n                  f"loss={train_metrics[\'loss\']:.4f} "\\\n                  f"acc={train_metrics[\'accuracy\']:.1f}% | "\\\n                  f"grad={train_metrics[\'grad_norm\']:.4f} | "\\\n                  f"val_acc={val_metrics[\'accuracy\']:.1f}% "\\\n                  f"CRA={val_metrics[\'cra\']:.1f}%")\n'
            )
            
        # Update source
        # Jupyter stores source as a list of lines, so we can split it
        # Actually it's fine if we store it as a single string, jupyter lab handles both
        # But to be safe, let's keep it as list of lines with \n at the end
        lines = [line + '\n' for line in source.split('\n')]
        # Remove the extra newline at the end of the last line
        lines[-1] = lines[-1][:-1]
        if lines[-1] == "":
            lines.pop()
            
        cell["source"] = lines

with open(filename, "w") as f:
    json.dump(data, f, indent=1)

print("Notebook patched successfully!")

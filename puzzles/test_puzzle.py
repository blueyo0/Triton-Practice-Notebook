# Modified from https://github.com/srush/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb

import inspect
import triton
import torch

from interpreter import patch, collect_grid

def test(puzzle, puzzle_spec, nelem={}, B={"B0": 32}, print_log=False, device="cpu") -> bool:
    """Test a single puzzle."""

    B = dict(B)
    if "N1" in nelem and "B1" not in B:
        B["B1"] = 32
    if "N2" in nelem and "B2" not in B:
        B["B2"] = 32

    torch.manual_seed(0)
    signature = inspect.signature(puzzle_spec)
    args = {}
    for n, p in signature.parameters.items():
        # print(p)
        args[n + "_ptr"] = (p.annotation.dims, p)
    args["z_ptr"] = (signature.return_annotation.dims, None)

    tt_args = []
    for k, (v, t) in args.items():
        tt_args.append(torch.rand(*v, device=device) - 0.5)
        # tt_args.append(torch.ones(*v, device=device))
        if t is not None and t.annotation.dtype == "int32":
            tt_args[-1] = torch.randint(-100000, 100000, v, device=device)

    grid = lambda meta: (triton.cdiv(nelem["N0"], meta["B0"]),
                         triton.cdiv(nelem.get("N1", 1), meta.get("B1", 1)),
                         triton.cdiv(nelem.get("N2", 1), meta.get("B2", 1)))

    #for k, v in args.items():
    #    print(k, v)
    
    # triton_viz.trace(puzzle)[grid](*tt_args, **B, **nelem))
    with patch():
        puzzle[grid](*tt_args, **B, **nelem)
    
    z = tt_args[-1]
    tt_args = tt_args[:-1]
    z_ = puzzle_spec(*tt_args)
    match = torch.allclose(z, z_, rtol=1e-3, atol=1e-3)
    match_emoji = "✅" if match else "❌"
    print(match_emoji, "Results match:", match)

    if not match or print_log:
        print("Launch args: ", nelem, B)
        print("Inputs: ", tt_args)
        print("Yours:", z.dtype, z.shape, "\n", z)
        print("Spec:", z_.dtype, z_.shape, "\n", z_)
        print("Diff (True: correct, False: incorrect):", "\n", torch.isclose(z, z_))

    if device == "cuda":
        print("Memory access detection is not supported on GPU. Skip checking.")
        return match

    _, _, failures, access_offsets = collect_grid()
    mem_emoji = "✅" if not failures else "❌"

    if failures:
        print(mem_emoji, "Invalid access detected! ")
    else:
        print(mem_emoji, "No invalid access detected.")
    
    if failures or print_log:
        print("Launch args: ", nelem, B)
        print("Inputs: ", tt_args)
        for key, value in access_offsets.items():
            is_invalid = key  in failures
            valid = "✅ Valid" if not is_invalid else "❌ Invalid"
            print(f"{valid} access in block: ", key)
            print("Access offsets (in bytes. float32/int32=4 bytes per loc): \n", value)
            if is_invalid:
                print("Invalid access mask (True: valid access, False: invalid access): \n", failures[key])
    
    return match and not failures
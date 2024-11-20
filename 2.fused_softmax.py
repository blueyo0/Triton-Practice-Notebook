import torch
import triton
import triton.language as tl

# @torch.jit.script
def naive_softmax(x):
    """使用原生 pytorch 计算 X 的逐行 softmax

    我们减去最大元素以避免溢出。Softmax 对这种位移是不变的。
    因为 softmax = exp(x_i) / sum(exp(x_0), ..., exp(x_n))
    而 exp 对于指数的加减上下会消掉
    所以一个 safe softmax 的实现形式就是先减掉 max
    """
    # 读取 MN 个元素；写入 M 个元素
    x_max = x.max(dim=1)[0]
    # 读取 MN + M 个元素；写入 MN 个元素
    z = x - x_max[:, None]
    # 读取 MN 个元素；写入 MN 个元素
    numerator = torch.exp(z)
    # 读取 MN 个元素；写入 M 个元素
    denominator = numerator.sum(dim=1)
    # 读取 MN + M 个元素；写入 MN 个元素
    ret = numerator / denominator[:, None]
    # 总计：读取 5MN + 2M 个元素；写入 3MN + 2M 个元素
    return ret

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # softmax 的行是独立的，所以我们在这些行上并行化
    row_idx = tl.program_id(0)
    # 通过 row_stride & row_idx 控制 ptr 的位置
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # FYI, triton 要求 BLOCK_SIZE 一定要是 2**n
    # 因此，这里我们假设 BLOCK_SIZE 被设置成了 [M, N] 大小 softmax 中，大于N的最接近的 2**n
    # 所以在下面使用 mask=col_offsets < n_cols 来限制真实计算范围，防止超了
    # P.S. 感觉triton里的这些ptr可以当做c++的数组ptr来看待
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # 使用掩码将行加载到SRAM中，因为 BLOCK_SIZE 可能大于 n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # 减去最大值以保证数值稳定性
    row_minus_max = row - tl.max(row, axis=0)
    # 注意，在 Triton 中指数运算是快速但近似的（即，想象在 CUDA 中的 __expf）
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # 将输出写回到 DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

# 这里比官方教程简化了一些参数设置
def softmax(x):
    n_rows, n_cols = x.shape
    # 块大小是大于 `x` 中列数的最小2的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # 另一个我们可以使用的技巧是要求编译器通过
    # 增加每行分布的 warps 数量（`num_warps`）来使用更多线程。
    # 在下一个教程中，你将看到如何以更自然的方式自动调整这个值，
    # 这样你就不必自己提出手工启发式方法。
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # 分配输出
    y = torch.empty_like(x)
    # 排队内核。1D启动网格很简单：输入矩阵的每一行分配一个 kernel 实例
    softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

def checkAC():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = softmax(x)
    # y_torch = torch.softmax(x, axis=1)
    y_torch = naive_softmax(x)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    print(f'在torch和triton之间的最大差异是 '
        f'{torch.max(torch.abs(y_torch - y_triton))}')
    # triton 和 softmax 有很小的 AC 差别了,

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作图表x轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  # `x_name`的不同可能值
        line_arg='provider',  # 其值对应图表中不同线条的参数名
        line_vals=[
            'triton',
            'torch-native',
            'torch-jit',
        ],  # `line_arg`的可能值
        line_names=[
            "Triton",
            "Torch (native)",
            "Torch (jit)",
        ],  # 线条的标签名
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # 线条样式
        ylabel="GB/s",  # y轴的标签名
        plot_name="softmax-performance",  # 图表的名称。也用作保存图表的文件名。
        args={'M': 4096},  # 不在`x_names`和`y_name`中的函数参数值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__=="__main__":
    # checkAC()
    benchmark.run(show_plots=True, print_data=True, save_path='./output')
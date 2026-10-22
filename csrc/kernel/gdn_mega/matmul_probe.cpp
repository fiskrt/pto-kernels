// Diagnostic only: isolate arithmetic, operand movement, and accumulator
// feedback costs for the exact tile sizes used by the production kernel.
template <int N>
AICORE inline void mega_matmul_probe(__gm__ half* identity, __gm__ half* output,
                                     __gm__ uint64_t* counters)
{
#if defined(__DAV_CUBE__)
    constexpr int Repeats = 256;
    using Mat = Tile<TileType::Mat, half, N, N, BLayout::ColMajor, N, N, SLayout::RowMajor, 512>;
    Mat input, feedback;
    TileLeft<half, N, N> left;
    TileRight<half, N, N> right;
    TileAcc<float, N, N> first, second;
    TASSIGN(input, 0); TASSIGN(feedback, N * N * sizeof(half));
    TASSIGN(left, 0); TASSIGN(right, 0);
    TASSIGN(first, 0); TASSIGN(second, N * N * sizeof(float));
    mega_load_packed<N, N, 128>(input, input, identity, N, N);
    pipe_barrier(PIPE_ALL);
    TMOV(left, input); TMOV(right, input); pipe_barrier(PIPE_ALL);
    uint64_t start = get_sys_cnt();
    for (int i = 0; i < Repeats; ++i) {
        TMATMUL(first, left, right);
        TMATMUL(second, left, right);
    }
    pipe_barrier(PIPE_ALL);
    counters[0] = get_sys_cnt() - start;
    start = get_sys_cnt();
    for (int i = 0; i < Repeats; ++i) {
        TMOV(left, input); TMOV(right, input); pipe_barrier(PIPE_ALL);
        TMATMUL(first, left, right); pipe_barrier(PIPE_ALL);
        TMOV(left, input); TMOV(right, input); pipe_barrier(PIPE_ALL);
        TMATMUL(second, left, right); pipe_barrier(PIPE_ALL);
    }
    counters[1] = get_sys_cnt() - start;
    start = get_sys_cnt();
    for (int i = 0; i < Repeats; ++i) {
        TMOV(left, input); TMOV(right, input); pipe_barrier(PIPE_ALL);
        TMATMUL(first, left, right); pipe_barrier(PIPE_ALL);
        TMOV(feedback, first); pipe_barrier(PIPE_ALL);
        TMOV(left, feedback); TMOV(right, feedback); pipe_barrier(PIPE_ALL);
        TMATMUL(second, left, right); pipe_barrier(PIPE_ALL);
        TMOV(feedback, second); pipe_barrier(PIPE_ALL);
    }
    counters[2] = get_sys_cnt() - start;
    // The input is -I. Every result is +I, checked by the probe runner.
    mega_store_packed<N, N, 128>(second, second, output, N, N);
    pipe_barrier(PIPE_ALL);
#endif
}

template <int H>
AICORE inline void mega_store_probe(__gm__ half* identity, __gm__ half* scratch,
                                    __gm__ uint64_t* counters)
{
#if defined(__DAV_CUBE__)
    constexpr int N = 128, Repeats = 256;
    using Mat = Tile<TileType::Mat, half, N, N, BLayout::ColMajor, N, N, SLayout::RowMajor, 512>;
    Mat input, feedback;
    TileLeft<half, N, N> left;
    TileRight<half, N, N> right;
    TileAcc<float, N, N> result;
    TASSIGN(input, 0); TASSIGN(feedback, N * N * sizeof(half));
    TASSIGN(left, 0); TASSIGN(right, 0); TASSIGN(result, 0);
    mega_load_packed<N, N, N>(input, input, identity, N, N); pipe_barrier(PIPE_ALL);
    TMOV(left, input); TMOV(right, input); pipe_barrier(PIPE_ALL);
    TMATMUL(result, left, right); pipe_barrier(PIPE_ALL);
    uint64_t start = get_sys_cnt();
    for (int i = 0; i < Repeats; ++i) {
        mega_store_packed<N, N, N>(result, result, scratch + get_block_idx() * N * N, N, N);
        pipe_barrier(PIPE_ALL);
    }
    counters[0] = get_sys_cnt() - start;
    start = get_sys_cnt();
    for (int i = 0; i < Repeats; ++i) {
        mega_store_packed<N, N, H * N>(result, result, scratch + get_block_idx() * N, N, N);
        pipe_barrier(PIPE_ALL);
    }
    counters[1] = get_sys_cnt() - start;
    start = get_sys_cnt();
    for (int i = 0; i < Repeats; ++i) {
        TMOV(feedback, result); pipe_barrier(PIPE_ALL);
    }
    counters[2] = get_sys_cnt() - start;
#endif
}

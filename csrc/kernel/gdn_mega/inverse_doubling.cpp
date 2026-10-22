// Exact finite geometric series for a strictly triangular 128 x 128 matrix.
// X = I-A; P = A^2. Each step doubles the represented polynomial degree:
// X <- X + X P, P <- P^2. Since A^128=0, six updates finish the inverse.
// X stays in the FP32 accumulator across updates, eliminating X @ I copies.
AICORE inline void mega_inverse_doubling(__gm__ half* out, __gm__ half* in,
                                        __gm__ half* minus_identity, uint32_t matrices,
                                        uint32_t heads, __gm__ int32_t* cu)
{
#if defined(__DAV_CUBE__)
    constexpr int C = GDN_C;
    constexpr int Bytes = C * C * sizeof(half);
    using Mat = Tile<TileType::Mat, half, C, C, BLayout::ColMajor, C, C, SLayout::RowMajor, 512>;
    using MatTail = Tile<TileType::Mat, half, C, C, BLayout::ColMajor, DYNAMIC, DYNAMIC,
                         SLayout::RowMajor, 512, PadValue::Zero>;
    using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    Mat power, inverse, neg_i;
    TASSIGN(power, 0); TASSIGN(inverse, Bytes); TASSIGN(neg_i, 2 * Bytes);
    TileLeft<half, C, C> left;
    TileRight<half, C, C> right;
    TileAcc<float, C, C> x, square;
    TASSIGN(left, 0); TASSIGN(right, 0);
    TASSIGN(x, 0); TASSIGN(square, C * C * sizeof(float));
    GlobalTensor<half, Shape2, Stride2> identity_gm(minus_identity, {1, 1, 1, C, C}, {1, 1, 1, C, 1});
    TLOAD(neg_i, identity_gm);
    pipe_barrier(PIPE_ALL);
    for (uint32_t m = get_block_idx(); m < matrices; m += get_block_num()) {
        auto info = mk_solve::GetBSNDVarlenTileInfoFromCuSeqlens(m, heads, C, cu);
        int valid = info.valid_size;
        MatTail input(valid, valid);
        TASSIGN(input, 0);
        GlobalTensor<half, Shape2, Stride2> input_gm(in + info.bsnd_offset,
            {1, 1, 1, valid, valid}, {1, 1, 1, heads * C, 1});
        TLOAD(input, input_gm);
        pipe_barrier(PIPE_ALL);
        if (valid != C) TFILLPAD(input, input);
        TMOV(left, power);
        TMOV(right, power);
        pipe_barrier(PIPE_ALL);
        TMATMUL(square, left, right);
        pipe_barrier(PIPE_ALL);
        TMOV(power, square);
        TMOV(right, neg_i);
        pipe_barrier(PIPE_ALL);
        TMATMUL(x, left, right);
        pipe_barrier(PIPE_ALL);
        TMOV(left, neg_i);
        pipe_barrier(PIPE_ALL);
        TMATMUL_ACC(x, x, left, right);
        pipe_barrier(PIPE_ALL);
        TMOV(inverse, x);
        pipe_barrier(PIPE_ALL);
        for (int degree = 2; degree < C; degree *= 2) {
            TMOV(left, inverse);
            TMOV(right, power);
            pipe_barrier(PIPE_ALL);
            TMATMUL_ACC(x, x, left, right);
            pipe_barrier(PIPE_ALL);
            if (degree * 2 < C) {
                TMOV(inverse, x);
                TMOV(left, power);
                pipe_barrier(PIPE_ALL);
                TMATMUL(square, left, right);
                pipe_barrier(PIPE_ALL);
                TMOV(power, square);
                pipe_barrier(PIPE_ALL);
            }
        }
        TileAcc<float, C, C, DYNAMIC, DYNAMIC> result(valid, valid);
        TASSIGN(result, 0);
        GlobalTensor<half, Shape2, Stride2> output_gm(out + info.bsnd_offset,
            {1, 1, 1, valid, valid}, {1, 1, 1, heads * C, 1});
        TSTORE(output_gm, result);
        pipe_barrier(PIPE_ALL);
    }
#endif
}

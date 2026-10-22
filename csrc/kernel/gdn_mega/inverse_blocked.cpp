// Block inverse: invert four 32x32 diagonal blocks, then assemble only the
// nonzero lower off-diagonal blocks with -R^{-1} B L^{-1}.
template <int B, typename Mat>
AICORE inline void mega_inverse_diagonal(Mat input, Mat neg_i, Mat result, int offset)
{
#if defined(__DAV_CUBE__)
    using Small = Tile<TileType::Mat, half, B, B, BLayout::ColMajor, B, B, SLayout::RowMajor, 512>;
    Small power, inverse;
    TASSIGN(power, 5 * GDN_C * GDN_C * sizeof(half));
    TASSIGN(inverse, 5 * GDN_C * GDN_C * sizeof(half) + B * B * sizeof(half));
    TileLeft<half, B, B> a;
    TileRight<half, B, B> b;
    TileAcc<float, B, B> x, p;
    TASSIGN(a, 0); TASSIGN(b, 0); TASSIGN(x, 0); TASSIGN(p, B * B * sizeof(float));
    TEXTRACT(a, input, offset, offset); TEXTRACT(b, input, offset, offset);
    pipe_barrier(PIPE_ALL);
    TMATMUL(p, a, b); pipe_barrier(PIPE_ALL);
    TMOV(power, p); TEXTRACT(b, neg_i, 0, 0); pipe_barrier(PIPE_ALL);
    TMATMUL(x, a, b); pipe_barrier(PIPE_ALL);
    TEXTRACT(a, neg_i, 0, 0); pipe_barrier(PIPE_ALL);
    TMATMUL_ACC(x, x, a, b); pipe_barrier(PIPE_ALL);
    TMOV(inverse, x); pipe_barrier(PIPE_ALL);
    TileLeft<half, B, B> power_left;
    TASSIGN(power_left, B * B * sizeof(half));
    for (int degree = 2; degree < B; degree *= 2) {
        TMOV(a, inverse);
        TMOV(b, power);
        if (degree * 2 < B) TMOV(power_left, power);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(x, x, a, b);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        // Independent accumulator and L0A buffer: FIX can export X while
        // the cube computes the next power using the same resident L0B.
        if (degree * 2 < B) {
            TMATMUL(p, power_left, b);
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        }
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        if (degree * 2 < B) {
            TMOV(inverse, x);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
            TMOV(power, p);
        }
        set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    }
    TINSERT(result, x, offset, offset);
    pipe_barrier(PIPE_ALL);
#endif
}

template <int B, typename Mat>
AICORE inline void mega_inverse_join(Mat negative_input, Mat result, int offset)
{
#if defined(__DAV_CUBE__)
    using Small = Tile<TileType::Mat, half, B, B, BLayout::ColMajor, B, B, SLayout::RowMajor, 512>;
    Small tmp;
    TASSIGN(tmp, 5 * GDN_C * GDN_C * sizeof(half));
    TileLeft<half, B, B> a;
    TileRight<half, B, B> b;
    TileAcc<float, B, B> c;
    TASSIGN(a, 0); TASSIGN(b, 0); TASSIGN(c, 0);
    TEXTRACT(a, result, offset + B, offset + B);
    TEXTRACT(b, negative_input, offset + B, offset);
    pipe_barrier(PIPE_ALL);
    TMATMUL(c, a, b); pipe_barrier(PIPE_ALL);
    TMOV(tmp, c); pipe_barrier(PIPE_ALL);
    TMOV(a, tmp); TEXTRACT(b, result, offset, offset); pipe_barrier(PIPE_ALL);
    TMATMUL(c, a, b); pipe_barrier(PIPE_ALL);
    TINSERT(result, c, offset + B, offset); pipe_barrier(PIPE_ALL);
#endif
}

AICORE inline void mega_inverse_blocked(__gm__ half* out, __gm__ half* in,
                                       __gm__ half* minus_identity, uint32_t matrices,
                                       uint32_t heads, __gm__ int32_t* cu)
{
#if defined(__DAV_CUBE__)
    constexpr int C = GDN_C, Bytes = C * C * sizeof(half);
    using Mat = Tile<TileType::Mat, half, C, C, BLayout::ColMajor, C, C, SLayout::RowMajor, 512>;
    using Tail = Tile<TileType::Mat, half, C, C, BLayout::ColMajor, DYNAMIC, DYNAMIC,
                      SLayout::RowMajor, 512, PadValue::Zero>;
    using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    Mat input, negative, result, neg_i, identity;
    TASSIGN(input, 0); TASSIGN(negative, Bytes); TASSIGN(result, 2 * Bytes);
    TASSIGN(neg_i, 3 * Bytes); TASSIGN(identity, 4 * Bytes);
    TileLeft<half, C, C> a;
    TileRight<half, C, C> b;
    TileAcc<float, C, C> c;
    TASSIGN(a, 0); TASSIGN(b, 0); TASSIGN(c, 0);
    GlobalTensor<half, Shape2, Stride2> igm(minus_identity, {1, 1, 1, C, C}, {1, 1, 1, C, 1});
    TLOAD(neg_i, igm); pipe_barrier(PIPE_ALL);
    TMOV(a, neg_i); TMOV(b, neg_i); pipe_barrier(PIPE_ALL);
    TMATMUL(c, a, b); pipe_barrier(PIPE_ALL);
    TMOV(identity, c); pipe_barrier(PIPE_ALL);
    for (uint32_t m = get_block_idx(); m < matrices; m += get_block_num()) {
        auto info = mk_solve::GetBSNDVarlenTileInfoFromCuSeqlens(m, heads, C, cu);
        int valid = info.valid_size;
        Tail tail(valid, valid); TASSIGN(tail, 0);
        GlobalTensor<half, Shape2, Stride2> gm(in + info.bsnd_offset,
            {1, 1, 1, valid, valid}, {1, 1, 1, heads * C, 1});
        TLOAD(tail, gm); pipe_barrier(PIPE_ALL);
        if (valid != C) TFILLPAD(tail, tail);
        TMOV(a, neg_i); TMOV(b, input); pipe_barrier(PIPE_ALL);
        TMATMUL(c, a, b); pipe_barrier(PIPE_ALL);
        TMOV(negative, c); TMOV(result, c); pipe_barrier(PIPE_ALL);
#ifdef GDN_INVERSE_BASE64
        mega_inverse_diagonal<64>(input, neg_i, result, 0);
        mega_inverse_diagonal<64>(input, neg_i, result, 64);
#else
        for (int offset = 0; offset < C; offset += 32) {
            mega_inverse_diagonal<32>(input, neg_i, result, offset);
        }
        mega_inverse_join<32>(negative, result, 0);
        mega_inverse_join<32>(negative, result, 64);
#endif
        mega_inverse_join<64>(negative, result, 0);
        TMOV(a, result); TMOV(b, identity); pipe_barrier(PIPE_ALL);
        TMATMUL(c, a, b); pipe_barrier(PIPE_ALL);
        TileAcc<float, C, C, DYNAMIC, DYNAMIC> result_tail(valid, valid);
        TASSIGN(result_tail, 0);
        GlobalTensor<half, Shape2, Stride2> output(out + info.bsnd_offset,
            {1, 1, 1, valid, valid}, {1, 1, 1, heads * C, 1});
        TSTORE(output, result_tail); pipe_barrier(PIPE_ALL);
    }
#endif
}

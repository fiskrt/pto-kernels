// Compute a key Gram matrix once per key head, concurrently with vector gate
// preparation. Packed row storage also fits arbitrarily short sequence tails.
template <int H, int C, int D>
AICORE inline void mega_shared_grams(__gm__ half* k, __gm__ half* scratch,
                                    __gm__ int32_t* cu, uint32_t matrices, int key_heads,
                                    __gm__ half* q = nullptr, __gm__ half* qk_out = nullptr)
{
#if defined(__DAV_CUBE__)
    using Mat = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, C, D, SLayout::RowMajor, 512>;
    using MatTail = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, DYNAMIC, DYNAMIC,
                         SLayout::RowMajor, 512, PadValue::Zero>;
    using MatT = Tile<TileType::Mat, half, D, C, BLayout::RowMajor, D, C, SLayout::ColMajor, 512>;
    using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    Mat key;
    MatT kt;
    TileLeft<half, C, D> left;
    TileRight<half, D, C> right;
    TileAcc<float, C, C> acc;
    TASSIGN(key, 0); TASSIGN(left, 0); TASSIGN(right, 0); TASSIGN(acc, 0);
    for (uint32_t m = get_block_idx(); m < matrices / H * key_heads; m += get_block_num()) {
        auto info = mk_solve::GetBSNDVarlenTileInfoFromCuSeqlens(m, key_heads, C, cu);
        int valid = info.valid_size;
        int head = m % key_heads;
        int64_t pos = (info.bsnd_offset - head * C) / (key_heads * C);
        MatTail tail(valid, D);
        TASSIGN(tail, 0);
        GlobalTensor<half, Shape2, Stride2> input(k + (pos * key_heads + head) * D,
            {1, 1, 1, valid, D}, {1, 1, 1, key_heads * D, 1});
        TLOAD(tail, input);
        pipe_barrier(PIPE_ALL);
        if (valid != C) TFILLPAD(tail, tail);
        TRESHAPE(kt, key);
        TEXTRACT(left, key, 0, 0);
        TEXTRACT(right, kt, 0, 0);
        pipe_barrier(PIPE_ALL);
        TMATMUL(acc, left, right);
        pipe_barrier(PIPE_ALL);
        TileAcc<float, C, C, DYNAMIC, DYNAMIC> result(valid, C);
        TASSIGN(result, 0);
        GlobalTensor<half, Shape2, Stride2> output(scratch + info.bsnd_offset,
            {1, 1, 1, valid, C}, {1, 1, 1, key_heads * C, 1});
        TSTORE(output, result);
        pipe_barrier(PIPE_ALL);
#ifdef GDN_SHARED_OUTPUT
        GlobalTensor<half, Shape2, Stride2> query_gm(q + (pos * key_heads + head) * D,
            {1, 1, 1, valid, D}, {1, 1, 1, key_heads * D, 1});
        TLOAD(tail, query_gm); pipe_barrier(PIPE_ALL);
        if (valid != C) TFILLPAD(tail, tail);
        TEXTRACT(left, key, 0, 0); pipe_barrier(PIPE_ALL);
        // K^T remains in L0B while the Q tile replaces K in L1/L0A.
        TMATMUL(acc, left, right); pipe_barrier(PIPE_ALL);
        GlobalTensor<half, Shape2, Stride2> qk_gm(qk_out + info.bsnd_offset,
            {1, 1, 1, valid, C}, {1, 1, 1, key_heads * C, 1});
        TSTORE(qk_gm, result); pipe_barrier(PIPE_ALL);
#endif

    }
#endif
}

template <int H, int C, bool Output = false>
AICORE inline void mega_gate_grams(__gm__ half* raw, __gm__ float* g, __gm__ half* beta,
                                  __gm__ float* mask, __gm__ half* out, __gm__ int32_t* cu,
                                  uint32_t matrices, int key_heads, int64_t tokens,
                                  __gm__ half* q = nullptr, __gm__ half* q_scaled = nullptr, uint32_t first = UINT32_MAX, uint32_t last = UINT32_MAX)
{
#if defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
    constexpr int R = C / 2, F = R * C * sizeof(float);
    using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using Matrix = Tile<TileType::Vec, float, R, C, BLayout::RowMajor, R, C>;
    using Half = Tile<TileType::Vec, half, R, C, BLayout::RowMajor, R, C>;
    using HalfPart = Tile<TileType::Vec, half, R, C, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using Gates = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, C,
                       SLayout::NoneBox, 512, PadValue::Zero>;
    using GatesPart = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, DYNAMIC,
                           SLayout::NoneBox, 512, PadValue::Zero>;
    using Row = Tile<TileType::Vec, float, 1, R, BLayout::RowMajor, 1, R>;
    using Col = Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1>;
    Matrix values, coeff, rows, lower;
    Half values_half;
    Gates gates;
    Row row_gates, beta_f;
    TASSIGN(values, 0); TASSIGN(coeff, F); TASSIGN(rows, 2 * F); TASSIGN(lower, 3 * F);
    TASSIGN(values_half, 4 * F); TASSIGN(gates, 4 * F + F / 2);
    constexpr int BetaAddr = 4 * F + F / 2 + C * sizeof(float);
    constexpr int RowAddr = BetaAddr + R * sizeof(float);
    constexpr int HalfBetaAddr = RowAddr + R * sizeof(float);
    TASSIGN(beta_f, BetaAddr); TASSIGN(row_gates, RowAddr);
    int stripe = get_subblockid() * R;
    GlobalTensor<float, Shape2, Stride2> mask_gm(mask + stripe * C, {1, 1, 1, R, C}, {1, 1, 1, C, 1});
    TLOAD(lower, mask_gm);
    pipe_barrier(PIPE_ALL);
    for (uint32_t m = first == UINT32_MAX ? get_block_idx() : first;
         m < (last == UINT32_MAX ? matrices : last);
         m += first == UINT32_MAX ? get_block_num() : matrices) {
        auto info = mk_solve::GetBSNDVarlenTileInfoFromCuSeqlens(m, H, C, cu);
        int valid = info.valid_size;
        int count = min(R, valid - stripe);
        if (count <= 0) continue;
        int head = m % H, kh = head / (H / key_heads);
        int64_t pos = (info.bsnd_offset - head * C) / (H * C);
        GatesPart gpart(valid); TASSIGN(gpart, 4 * F + F / 2);
        GlobalTensor<float, Shape2, Stride2> ggm(g + head * tokens + pos,
            {1, 1, 1, 1, valid}, {1, 1, 1, valid, 1});
        TLOAD(gpart, ggm);
        using Beta = Tile<TileType::Vec, half, 1, R, BLayout::RowMajor, 1, R,
                          SLayout::NoneBox, 512, PadValue::Zero>;
        using BetaPart = Tile<TileType::Vec, half, 1, R, BLayout::RowMajor, 1, DYNAMIC,
                              SLayout::NoneBox, 512, PadValue::Zero>;
        Beta b; BetaPart bpart(count);
        TASSIGN(b, HalfBetaAddr); TASSIGN(bpart, HalfBetaAddr);
        GlobalTensor<half, Shape2, Stride2> bgm(beta + head * tokens + pos + stripe,
            {1, 1, 1, 1, count}, {1, 1, 1, count, 1});
        if constexpr (!Output) TLOAD(bpart, bgm);
        HalfPart raw_part(count, C); TASSIGN(raw_part, 4 * F);
        GlobalTensor<half, Shape2, Stride2> rawgm(raw + ((pos + stripe) * key_heads + kh) * C,
            {1, 1, 1, count, C}, {1, 1, 1, key_heads * C, 1});
        TLOAD(raw_part, rawgm);
        pipe_barrier(PIPE_ALL);
        if (valid != C) TFILLPAD_INPLACE(gates, gpart);
        Row grow; TASSIGN(grow, 4 * F + F / 2 + stripe * sizeof(float));
        if constexpr (Output) {
            TMOV(row_gates, grow); PipeBarrierVec();
        } else {
            if (count != R) TFILLPAD_INPLACE(b, bpart);
            TCVT(beta_f, b, RoundMode::CAST_NONE); PipeBarrierVec();
            TLOG(beta_f, beta_f); PipeBarrierVec();
            TADD(row_gates, grow, beta_f); PipeBarrierVec();
        }
        Col col; TASSIGN(col, RowAddr);
        TROWEXPAND(rows, col);
        TCOLEXPAND(coeff, gates); PipeBarrierVec();
        TSUB(coeff, rows, coeff); PipeBarrierVec();
        TMINS(coeff, coeff, 0.0f); PipeBarrierVec();
        TEXP(coeff, coeff);
        TCVT(values, values_half, RoundMode::CAST_NONE); PipeBarrierVec();
        TMUL(values, values, coeff); PipeBarrierVec();
        TMUL(values, values, lower); PipeBarrierVec();
        TCVT(values_half, values, RoundMode::CAST_NONE);
        pipe_barrier(PIPE_ALL);
        GlobalTensor<half, Shape2, Stride2> output(out + info.bsnd_offset + stripe * H * C,
            {1, 1, 1, count, C}, {1, 1, 1, H * C, 1});
        TSTORE(output, raw_part);
        pipe_barrier(PIPE_ALL);
        if constexpr (Output) {
            GlobalTensor<half, Shape2, Stride2> qgm(q + ((pos + stripe) * key_heads + kh) * C,
                {1, 1, 1, count, C}, {1, 1, 1, key_heads * C, 1});
            TLOAD(raw_part, qgm); pipe_barrier(PIPE_ALL);
            TEXP(row_gates, row_gates); PipeBarrierVec();
            TROWEXPAND(rows, col);
            TCVT(values, values_half, RoundMode::CAST_NONE); PipeBarrierVec();
            TMUL(values, values, rows); PipeBarrierVec();
            TCVT(values_half, values, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
            GlobalTensor<half, Shape2, Stride2> qout(q_scaled + ((pos + stripe) * H + head) * C,
                {1, 1, 1, count, C}, {1, 1, 1, H * C, 1});
            TSTORE(qout, raw_part); pipe_barrier(PIPE_ALL);
        }

    }
#endif
}

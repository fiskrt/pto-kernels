#pragma once

// Internal C=32 tiles let FP32 double-buffered operands fit the original
// caller-allocated per-core workspaces. Public tensor shapes remain unchanged.
template <int R, int K, typename T, typename Full>
AICORE inline void mega_vector_load(Full& full, __gm__ T* pointer, int rows, int cols, int stride)
{
#if defined(__DAV_VEC__)
    TEXPANDS(full, static_cast<T>(0)); PipeBarrierVec();
    if (rows > 0) {
        using Part = Tile<TileType::Vec, T, R, K, BLayout::RowMajor, DYNAMIC, DYNAMIC,
                          SLayout::NoneBox, 512, PadValue::Zero>;
        Part part(rows, cols); TASSIGN(part, (uint64_t)full.data());
        GlobalTensor<T, Shape<1, 1, 1, DYNAMIC, DYNAMIC>, pto::Stride<1, 1, 1, DYNAMIC, 1>> gm(
            pointer, {1, 1, 1, rows, cols}, {1, 1, 1, stride, 1});
        pipe_barrier(PIPE_ALL); TLOAD(part, gm); pipe_barrier(PIPE_ALL);
        Tile<TileType::Vec, T, R, K, BLayout::RowMajor, R, K,
             SLayout::NoneBox, 512, PadValue::Zero> padded;
        TASSIGN(padded, (uint64_t)full.data());
        TFILLPAD_INPLACE(padded, part); PipeBarrierVec();
    }
#endif
}

template <int R, int K, typename T, typename TileT>
AICORE inline void mega_full_store(__gm__ T* pointer, TileT& tile)
{
    GlobalTensor<T, Shape<1, 1, 1, R, K>, pto::Stride<1, 1, 1, K, 1>> gm(pointer);
    TSTORE(gm, tile); pipe_barrier(PIPE_ALL);
}

template <int R, int K, typename TileT>
AICORE inline void mega_float_load(TileT& tile, __gm__ float* pointer)
{
    GlobalTensor<float, Shape<1, 1, 1, R, K>, pto::Stride<1, 1, 1, K, 1>> gm(pointer);
    TLOAD(tile, gm); pipe_barrier(PIPE_ALL);
}

template <int D>
AICORE inline void mega_initialize_state(__gm__ half* initial, __gm__ half* final_state,
    __gm__ float* staging, bool has_initial, bool empty)
{
#if defined(__DAV_VEC__)
    constexpr int R = D / 2;
    Tile<TileType::Vec, half, R, D> source;
    Tile<TileType::Vec, float, R, D> value;
    TASSIGN(source, 0); TASSIGN(value, R * D * sizeof(half));
    if (has_initial) {
        GlobalTensor<half, Shape<1, 1, 1, R, D>, pto::Stride<1, 1, 1, D, 1>> gm(
            initial + get_subblockid() * R * D);
        TLOAD(source, gm); pipe_barrier(PIPE_ALL);
    } else { TEXPANDS(source, static_cast<half>(0)); PipeBarrierVec(); }
    if (empty) {
        mega_full_store<R, D>(final_state + get_subblockid() * R * D, source);
    } else {
        TCVT(value, source, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
        mega_full_store<R, D>(staging + get_subblockid() * R * D, value);
    }
#endif
}

template <int H, int C, int D>
AICORE inline void mega_prepare_chunk(
    __gm__ half* q, __gm__ half* k, __gm__ half* v, __gm__ float* g, __gm__ half* beta,
    __gm__ float* kk, __gm__ float* qk, __gm__ float* lower, __gm__ float* full, __gm__ half* neg_i,
    __gm__ float* inverse_gm, __gm__ float* w, __gm__ float* u, __gm__ float* scaled_k,
    __gm__ float* scaled_q, __gm__ float* attention, __gm__ float* decay,
    int64_t pos, int valid, int head, int kh, int64_t tokens)
{
#if defined(__DAV_VEC__)
    constexpr int R = C / 2, F = C * C * sizeof(float), Gate = 160 * 1024;
    using Vec = Tile<TileType::Vec, float, C, C>;
    using Row = Tile<TileType::Vec, float, 1, C>;
    using Col = Tile<TileType::Vec, float, C, 1, BLayout::ColMajor, C, 1>;
    Row gates, exponentials, betas;
    Tile<TileType::Vec, half, 1, C> beta_h;
    TASSIGN(gates, Gate); TASSIGN(exponentials, Gate + 512);
    TASSIGN(betas, Gate + 1024); TASSIGN(beta_h, Gate + 1536);
    mega_vector_load<1, C>(gates, g + head * tokens + pos, 1, valid, C);
    mega_vector_load<1, C>(beta_h, beta + head * tokens + pos, 1, valid, C);
    TEXP(exponentials, gates); TCVT(betas, beta_h, RoundMode::CAST_NONE); pipe_barrier(PIPE_ALL);
    float last = gates.GetValue(valid - 1), exp_last = exponentials.GetValue(valid - 1);
    int key_head = head / (H / kh);
    if (get_subblockid() == 0) {
        Vec a, raw, lower_mask, full_mask, coeff, inverse, transposed, temporary;
        TASSIGN(a, 0); TASSIGN(raw, F); TASSIGN(lower_mask, 2 * F);
        TASSIGN(full_mask, 3 * F); TASSIGN(coeff, 4 * F);
        TASSIGN(inverse, 5 * F); TASSIGN(transposed, 6 * F); TASSIGN(temporary, 7 * F);
        mega_vector_load<C, C>(lower_mask, lower, C, C, 128);
        mega_vector_load<C, C>(full_mask, full, C, C, 128);
        TSUB(inverse, full_mask, lower_mask); PipeBarrierVec();
        Col gcol, bcol; TASSIGN(gcol, Gate); TASSIGN(bcol, Gate + 1024);
        TCOLEXPAND(coeff, gates); PipeBarrierVec();
        TMULS(coeff, coeff, -1.0f); PipeBarrierVec();
        TROWEXPANDADD(coeff, coeff, gcol); PipeBarrierVec();
        TMINS(coeff, coeff, 0.0f); PipeBarrierVec();
        TEXP(coeff, coeff); PipeBarrierVec();
        TMUL(coeff, coeff, full_mask); PipeBarrierVec();
        mega_vector_load<C, C>(raw, qk + (pos * kh + key_head) * C, valid, C, kh * C);
        TMUL(raw, raw, coeff); pipe_barrier(PIPE_ALL);
        mega_full_store<C, C>(attention, raw);
        mega_vector_load<C, C>(raw, kk + (pos * kh + key_head) * C, valid, C, kh * C);
        TMUL(a, raw, coeff); PipeBarrierVec();
        TMUL(a, a, lower_mask); PipeBarrierVec();
        TROWEXPANDMUL(a, a, bcol); PipeBarrierVec();
        TMULS(a, a, -1.0f); PipeBarrierVec();
        mega_inverse_substitute<C>(inverse, a, transposed, temporary);
        pipe_barrier(PIPE_ALL); mega_full_store<C, C>(inverse_gm, inverse);
    }
    pipe_barrier(PIPE_ALL);
    // Both vector subblocks prepare half the operand rows.
    constexpr int Bytes = R * D * sizeof(float);
    Tile<TileType::Vec, float, R, D> values, keys;
    Tile<TileType::Vec, half, R, D> input;
    Tile<TileType::Vec, float, 1, R> row, grow, exp_grow, brow;
    Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1> column, beta_col;
    int stripe = get_subblockid() * R, count = max(0, min(R, valid - stripe));
    TASSIGN(values, 0); TASSIGN(keys, Bytes); TASSIGN(input, 2 * Bytes);
    TASSIGN(row, 3 * Bytes); TASSIGN(column, 3 * Bytes);
    TASSIGN(grow, Gate + stripe * sizeof(float));
    TASSIGN(exp_grow, Gate + 512 + stripe * sizeof(float));
    TASSIGN(brow, Gate + 1024 + stripe * sizeof(float));
    TASSIGN(beta_col, Gate + 1024 + stripe * sizeof(float));
    int64_t input_offset = ((pos + stripe) * kh + key_head) * D;
    mega_vector_load<R, D>(input, k + input_offset, count, D, kh * D);
    TCVT(keys, input, RoundMode::CAST_NONE); PipeBarrierVec();
    TMUL(row, exp_grow, brow); PipeBarrierVec();
    TMULS(row, row, -1.0f); PipeBarrierVec();
    TROWEXPANDMUL(values, keys, column); pipe_barrier(PIPE_ALL);
    mega_full_store<R, D>(w + stripe * D, values);
    TMULS(row, grow, -1.0f); PipeBarrierVec();
    TADDS(row, row, last); PipeBarrierVec(); TEXP(row, row); PipeBarrierVec();
    TROWEXPANDMUL(values, keys, column); pipe_barrier(PIPE_ALL);
    mega_full_store<R, D>(scaled_k + stripe * D, values);
    mega_vector_load<R, D>(input, q + input_offset, count, D, kh * D);
    TCVT(values, input, RoundMode::CAST_NONE); PipeBarrierVec();
    TMOV(row, exp_grow); PipeBarrierVec();
    TROWEXPANDMUL(values, values, column); pipe_barrier(PIPE_ALL);
    mega_full_store<R, D>(scaled_q + stripe * D, values);
    mega_vector_load<R, D>(input, v + ((pos + stripe) * H + head) * D, count, D, H * D);
    TCVT(values, input, RoundMode::CAST_NONE); PipeBarrierVec();
    TROWEXPANDMUL(values, values, beta_col); pipe_barrier(PIPE_ALL);
    mega_full_store<R, D>(u + stripe * D, values);
    // Decay remains FP32, including values that would underflow in FP16.
    constexpr int DR = D / 2;
    Tile<TileType::Vec, half, DR, D> identity;
    Tile<TileType::Vec, float, DR, D> diagonal;
    TASSIGN(identity, 0); TASSIGN(diagonal, DR * D * sizeof(half));
    GlobalTensor<half, Shape<1, 1, 1, DR, D>, pto::Stride<1, 1, 1, D, 1>> igm(
        neg_i + get_subblockid() * DR * D);
    TLOAD(identity, igm); pipe_barrier(PIPE_ALL);
    TCVT(diagonal, identity, RoundMode::CAST_NONE); PipeBarrierVec();
    TMULS(diagonal, diagonal, -exp_last); pipe_barrier(PIPE_ALL);
    mega_full_store<DR, D>(decay + get_subblockid() * DR * D, diagonal);
#endif
}

template <int H, int C, int D>
AICORE inline void mega_fused_pipeline(
    __gm__ half* q, __gm__ half* k, __gm__ half* v, __gm__ float* g, __gm__ half* beta,
    __gm__ float* kk, __gm__ float* qk, __gm__ float* lower, __gm__ float* full, __gm__ half* neg_i,
    __gm__ float* inverse_ws, __gm__ float* w_ws, __gm__ float* u_ws, __gm__ float* k_ws,
    __gm__ float* q_ws, __gm__ float* att_ws, __gm__ float* state_ws,
    __gm__ half* output, __gm__ half* final_state, __gm__ half* initial, bool has_initial,
    __gm__ int32_t* cu, int64_t sequences, int kh, int64_t tokens)
{
#if defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
#endif
#if defined(__DAV_CUBE__)
    using CD = Tile<TileType::Mat, float, C, D, BLayout::ColMajor, C, D, SLayout::RowMajor, 512>;
    using CC = Tile<TileType::Mat, float, C, C, BLayout::ColMajor, C, C, SLayout::RowMajor, 512>;
    using DD = Tile<TileType::Mat, float, D, D, BLayout::ColMajor, D, D, SLayout::RowMajor, 512>;
    CD w, u, kt, query, residual;
    CC inverse, attention;
    DD state, decay;
    TASSIGN(state, 0); TASSIGN(w, 64 * 1024); TASSIGN(u, 80 * 1024);
    TASSIGN(inverse, 96 * 1024); TASSIGN(kt, 112 * 1024); TASSIGN(query, 128 * 1024);
    TASSIGN(attention, 144 * 1024); TASSIGN(residual, 160 * 1024); TASSIGN(decay, 192 * 1024);
#endif
    auto block = get_block_idx();
    for (int64_t work = block; work < sequences * H; work += get_block_num()) {
        int seq = work / H, head = work % H;
        int64_t start = cu[seq], end = cu[seq + 1];
        auto stage = state_ws + block * 32768;
#if defined(__DAV_VEC__)
        mega_initialize_state<D>(has_initial ? initial + work * D * D : initial,
                                 final_state + work * D * D, stage, has_initial, start == end);
#endif
        if (start == end) continue;
#if defined(__DAV_VEC__)
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
#endif
#if defined(__DAV_CUBE__)
        wait_flag_dev(1);
        mega_float_load<D, D>(state, stage);
        ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | (7 << 8));
        ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | (8 << 8));
#endif
        for (int64_t pos = start, ci = 0; pos < end; pos += C, ++ci) {
            int valid = min(static_cast<int64_t>(C), end - pos), slot = ci & 1;
            int off = block * 8192 + slot * C * D;
            auto ip = inverse_ws + off, wp = w_ws + off, up = u_ws + off;
            auto kp = k_ws + off, qp = q_ws + block * 16384 + slot * C * D;
            auto ap = att_ws + off, dp = stage + slot * D * D;
#if defined(__DAV_VEC__)
            wait_flag_dev(7 + slot);
            mega_prepare_chunk<H, C, D>(q, k, v, g, beta, kk, qk, lower, full, neg_i,
                ip, wp, up, kp, qp, ap, dp, pos, valid, head, kh, tokens);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((5 + slot) << 8));
#endif
#if defined(__DAV_CUBE__)
            wait_flag_dev(5 + slot);
            mega_float_load<C, C>(inverse, ip); mega_float_load<C, C>(attention, ap);
            mega_float_load<C, D>(w, wp); mega_float_load<C, D>(u, up);
            mega_float_load<C, D>(kt, kp); mega_float_load<C, D>(query, qp);
            mega_float_load<D, D>(decay, dp);
            mega_recurrence_pipeline<H, C, D>(w, u, inverse, state, kt, query, attention, residual,
                decay, output + (pos * H + head) * D, valid, final_state + work * D * D, pos + C >= end,
                wp, up, dp);
            ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | ((7 + slot) << 8));
#endif
        }
#if defined(__DAV_VEC__)
        // Drain both slot permissions before the next sequence uses the same
        // staging memory for its initial state. No stale readiness may escape.
        wait_flag_dev(7); wait_flag_dev(8);
#endif
    }
    pipe_barrier(PIPE_ALL);
}

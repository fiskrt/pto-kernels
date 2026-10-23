#pragma once

// The A2 Acc-to-L1 instruction narrows to FP16/BF16. Use the already consumed
// producer slot for lossless FP32 export/reload; release it only afterwards.
template <int R, int K, typename Mat, typename Acc>
AICORE inline void mega_keep_fp32(Mat& dst, Acc& src, __gm__ float* scratch)
{
#if defined(__DAV_CUBE__)
    GlobalTensor<float, Shape<1, 1, 1, R, K>, pto::Stride<1, 1, 1, K, 1>> gm(scratch);
    TSTORE(gm, src); pipe_barrier(PIPE_ALL);
    TLOAD(dst, gm); pipe_barrier(PIPE_ALL);
#endif
}


// Preserve FP32 through every product and through recurrent state. Only the
// public output/final-state stores round to FP16. No inverse powers or skipped
// small state contributions are used.
template <int H, int C, int D, typename MatCD, typename MatCC, typename MatDD>
AICORE inline void mega_recurrence_pipeline(MatCD& w, MatCD& u, MatCC& inverse,
    MatDD& state, MatCD& kt, MatCD& query, MatCC& attention, MatCD& residual,
    MatDD& decay, __gm__ half* output, int valid, __gm__ half* final_state, bool last,
    __gm__ float* w_scratch, __gm__ float* r_scratch, __gm__ float* state_scratch)
{
#if defined(__DAV_CUBE__)
    TileLeft<float, C, C> inv_left, att_left;
    TileLeft<float, C, D> w_left, q_left;
    TileLeft<float, D, D> decay_left;
    TileLeft<float, D, C> kt_left;
    TileRight<float, C, D> values_right;
    TileRight<float, D, D> state_right;
    TileAcc<float, C, D> r_acc, out_acc;
    TileAcc<float, D, D> state_acc;
    TASSIGN(inv_left, 0); TASSIGN(att_left, 0); TASSIGN(w_left, 0);
    TASSIGN(q_left, 0); TASSIGN(decay_left, 0); TASSIGN(kt_left, 0);
    TASSIGN(values_right, 0); TASSIGN(state_right, 0);
    TASSIGN(r_acc, 0); TASSIGN(state_acc, 0); TASSIGN(out_acc, D * D * sizeof(float));
    TMOV(inv_left, inverse); TMOV(values_right, u); pipe_barrier(PIPE_ALL);
    TMATMUL(r_acc, inv_left, values_right); pipe_barrier(PIPE_ALL);
    TMOV(values_right, w); pipe_barrier(PIPE_ALL);
    TMATMUL(out_acc, inv_left, values_right); pipe_barrier(PIPE_ALL);
    mega_keep_fp32<C, D>(w, out_acc, w_scratch);
    TMOV(w_left, w); TMOV(state_right, state); pipe_barrier(PIPE_ALL);
    TMATMUL_ACC(r_acc, r_acc, w_left, state_right); pipe_barrier(PIPE_ALL);
    mega_keep_fp32<C, D>(residual, r_acc, r_scratch);
    TMOV(q_left, query); pipe_barrier(PIPE_ALL);
    TMATMUL(out_acc, q_left, state_right); pipe_barrier(PIPE_ALL);
    TMOV(decay_left, decay); pipe_barrier(PIPE_ALL);
    TMATMUL(state_acc, decay_left, state_right); pipe_barrier(PIPE_ALL);
    using Trans = Tile<TileType::Mat, float, D, C, BLayout::RowMajor, D, C, SLayout::ColMajor, 512>;
    Trans transpose; TRESHAPE(transpose, kt);
    TEXTRACT(kt_left, transpose, 0, 0); TMOV(values_right, residual); pipe_barrier(PIPE_ALL);
    TMATMUL_ACC(state_acc, state_acc, kt_left, values_right); pipe_barrier(PIPE_ALL);
    mega_keep_fp32<D, D>(state, state_acc, state_scratch);
    TMOV(att_left, attention); pipe_barrier(PIPE_ALL);
    TMATMUL_ACC(out_acc, out_acc, att_left, values_right); pipe_barrier(PIPE_ALL);
    TileAcc<float, C, D, DYNAMIC, D> tail(valid);
    TASSIGN(tail, D * D * sizeof(float));
    mega_store_packed<C, D, H * D>(out_acc, tail, output, valid, D);
    if (last) mega_store_packed<D, D, D>(state_acc, state_acc, final_state, D, D);
    pipe_barrier(PIPE_ALL);
#endif
}

#pragma once

// Forward substitution on (I-A) X = I. Updating the accumulated rows avoids
// large powers of A and their cancellation, even for identical keys.
// A is strictly lower triangular. All operands and results remain FP32.
template <int C, typename Matrix>
AICORE inline void mega_inverse_substitute(Matrix& inverse, Matrix& a,
                                           Matrix& transposed, Matrix& temporary)
{
#if defined(__DAV_VEC__)
    TTRANS(transposed, a, temporary); PipeBarrierVec();
    for (int i = 0; i + 1 < C; ++i) {
        using Row = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, C>;
        using Col = Tile<TileType::Vec, float, C, 1, BLayout::ColMajor, C, 1>;
        Row solved;
        Col coefficients;
        TASSIGN(solved, 5 * C * C * sizeof(float) + i * C * sizeof(float));
        TASSIGN(coefficients, 6 * C * C * sizeof(float) + i * C * sizeof(float));
        TCOLEXPAND(temporary, solved); PipeBarrierVec();
        TROWEXPANDMUL(temporary, temporary, coefficients); PipeBarrierVec();
        TADD(inverse, inverse, temporary); PipeBarrierVec();
    }
#endif
}

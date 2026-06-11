/*
 * monotonic_alignment/core.cu  (x方向並列版)
 *
 * 設計:
 *   Forward pass:
 *     grid  = (B,)          各ブロックが1バッチサンプルを担当
 *     block = (BLOCK_SIZE,) 各スレッドが1行内の複数x要素を担当
 *
 *     行間依存(y→y+1)は避けられないため行ループは逐次。
 *     行内(x方向)は互いに独立なので BLOCK_SIZE スレッドで並列処理。
 *     shared memory に前行の値をキャッシュすることで
 *     グローバルメモリへのランダムアクセスを削減する。
 *
 *   Traceback:
 *     index の更新が逐次依存なのでスレッド0のみで処理。
 *     value_buf の読み出しはcoalesced accessを意識したオフセット計算。
 *
 * 入力テンソル形状:
 *   neg_cent : [B, T_y, T_x]  float32, contiguous, GPU
 *   path     : [B, T_y, T_x]  int32,   zeros,      GPU
 *   t_y_max  : [B]             int32,   GPU
 *   t_x_max  : [B]             int32,   GPU
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// ブロックあたりのスレッド数。
// T_x の最大値(典型的に ~150)をカバーできる最小の2の冪。
// warpサイズ(32)の倍数にすることでwarp divergenceを抑える。
#define BLOCK_SIZE 256

// shared memory の最大 T_x サイズ。
// これを超える T_x には動的shared memoryが必要（今回は静的で十分）。
#define MAX_TX 512

/* ------------------------------------------------------------------ */
/*  Forward DP カーネル（x方向並列版）                                   */
/* ------------------------------------------------------------------ */
__global__ void maximum_path_forward_kernel(
    const float* __restrict__ neg_cent,   // [B, T_y, T_x]
    const int*   __restrict__ t_y_max,    // [B]
    const int*   __restrict__ t_x_max,    // [B]
    float*       __restrict__ value_buf,  // [B, T_y, T_x]
    const int T_y,
    const int T_x)
{
    const int b      = blockIdx.x;
    const int tid    = threadIdx.x;
    const int t_y    = t_y_max[b];
    const int t_x    = t_x_max[b];
    const int offset = b * T_y * T_x;

    // shared memory: 前の行(prev_row)と現在の行(curr_row)を保持
    // double-bufferingにより__syncthreads()の回数を最小化する
    __shared__ float prev_row[MAX_TX];
    __shared__ float curr_row[MAX_TX];

    const float MAX_NEG = -1e9f;

    // --- 行ループ（逐次: y方向の依存があるため並列化不可） ---
    for (int y = 0; y < t_y; ++y) {
        const int x_lo = max(0,   t_x + y - t_y);
        const int x_hi = min(t_x, y + 1);

        // --- x方向を BLOCK_SIZE スレッドで並列処理 ---
        for (int x = x_lo + tid; x < x_hi; x += BLOCK_SIZE) {
            float v_prev, v_cur;

            // v_cur  : 真上(y-1, x)から来る遷移。x==yは到達不可(モノトニック制約)
            // v_prev : 左上(y-1, x-1)から来る遷移。x==0のとき左上は存在しない
            if (y == 0) {
                // 初行(y=0): x_lo=0, x_hi=1 なので x==0 のみ処理される
                // x==0==y なので v_cur=MAX_NEG、x==0なのでv_prev=0(原点)
                v_cur  = MAX_NEG;
                v_prev = 0.0f;
            } else {
                // prev_row は前ループ末尾の __syncthreads() で確定済み
                v_cur  = (x == y) ? MAX_NEG : prev_row[x];
                v_prev = (x == 0) ? MAX_NEG : prev_row[x - 1];
            }

            float val = neg_cent[offset + y * T_x + x] + fmaxf(v_prev, v_cur);
            curr_row[x] = val;
            value_buf[offset + y * T_x + x] = val;
        }

        // curr_row の書き込み完了を全スレッドで待つ
        __syncthreads();

        // curr_row → prev_row へのコピー（次の行の計算に使用）
        for (int x = x_lo + tid; x < x_hi; x += BLOCK_SIZE) {
            prev_row[x] = curr_row[x];
        }

        // prev_row の更新完了を待つ
        __syncthreads();
    }
}

/* ------------------------------------------------------------------ */
/*  Traceback カーネル                                                   */
/*  index の更新が逐次依存なのでスレッド0のみ担当。                        */
/*  ただしvalue_bufのロードはshared memoryを介してcoalesced化する。        */
/* ------------------------------------------------------------------ */
__global__ void maximum_path_traceback_kernel(
    const float* __restrict__ value_buf,  // [B, T_y, T_x]
    const int*   __restrict__ t_y_max,    // [B]
    const int*   __restrict__ t_x_max,    // [B]
    int*         __restrict__ path,       // [B, T_y, T_x]
    const int T_y,
    const int T_x)
{
    const int b      = blockIdx.x;
    const int tid    = threadIdx.x;
    const int t_y    = t_y_max[b];
    const int t_x    = t_x_max[b];
    const int offset = b * T_y * T_x;

    // 1行分をshared memoryにロードしてからスレッド0が参照する
    __shared__ float row_cache[MAX_TX];

    int index = t_x - 1;

    for (int y = t_y - 1; y >= 0; --y) {
        // 全スレッドで (y-1) 行を shared memory へロード
        // y==0 のときは (y-1) = -1 行は存在しないのでロードしない
        if (y > 0) {
            for (int x = tid; x < t_x; x += BLOCK_SIZE) {
                row_cache[x] = value_buf[offset + (y - 1) * T_x + x];
            }
        }
        __syncthreads();

        // スレッド0のみでtraceback（逐次依存あり）
        if (tid == 0) {
            path[offset + y * T_x + index] = 1;
            // y==0 のとき index の更新は不要（ループがここで終わる）
            if (y > 0 && index != 0) {
                if (index == y ||
                    row_cache[index] < row_cache[index - 1])
                {
                    index -= 1;
                }
            }
        }

        __syncthreads();
    }
}

/* ------------------------------------------------------------------ */
/*  エントリポイント                                                      */
/* ------------------------------------------------------------------ */
void maximum_path_cuda(
    torch::Tensor& path,
    torch::Tensor& neg_cent,
    torch::Tensor& t_y_max,
    torch::Tensor& t_x_max)
{
    TORCH_CHECK(neg_cent.is_cuda(),  "neg_cent must be a CUDA tensor");
    TORCH_CHECK(path.is_cuda(),      "path must be a CUDA tensor");
    TORCH_CHECK(neg_cent.dtype() == torch::kFloat32, "neg_cent must be float32");
    TORCH_CHECK(path.dtype()     == torch::kInt32,   "path must be int32");

    const int B   = neg_cent.size(0);
    const int T_y = neg_cent.size(1);
    const int T_x = neg_cent.size(2);

    TORCH_CHECK(T_x <= MAX_TX,
        "T_x (", T_x, ") exceeds MAX_TX (", MAX_TX, "). "
        "Recompile with a larger MAX_TX.");

    auto value_buf = torch::empty_like(neg_cent);
    auto stream    = c10::cuda::getCurrentCUDAStream();

    maximum_path_forward_kernel<<<B, BLOCK_SIZE, 0, stream>>>(
        neg_cent.data_ptr<float>(),
        t_y_max.data_ptr<int>(),
        t_x_max.data_ptr<int>(),
        value_buf.data_ptr<float>(),
        T_y, T_x
    );

    maximum_path_traceback_kernel<<<B, BLOCK_SIZE, 0, stream>>>(
        value_buf.data_ptr<float>(),
        t_y_max.data_ptr<int>(),
        t_x_max.data_ptr<int>(),
        path.data_ptr<int>(),
        T_y, T_x
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maximum_path_cuda", &maximum_path_cuda,
          "Monotonic Alignment Search: x-parallel CUDA kernel",
          py::arg("path"),
          py::arg("neg_cent"),
          py::arg("t_y_max"),
          py::arg("t_x_max"));
}

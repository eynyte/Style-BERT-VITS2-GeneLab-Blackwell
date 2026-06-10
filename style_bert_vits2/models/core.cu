/*
 * monotonic_alignment/core.cu
 *
 * Numba JIT実装をCUDAカーネルに置き換えたもの。
 * バッチ内の各サンプルを1つのCUDAスレッドブロックに割り当て、
 * GPU↔CPU転送を完全に排除する。
 *
 * アルゴリズム概要:
 *   Forward pass : DP表 value[t_y, t_x] を埋める
 *   Traceback    : 最大スコアのパスを path[t_y, t_x] に書き込む
 *
 * 入力テンソル形状:
 *   neg_cent : [B, T_y, T_x]  float32, GPU
 *   path     : [B, T_y, T_x]  int32,   GPU (ゼロ初期化済み)
 *   t_y_max  : [B]             int32,   GPU (有効フレーム長)
 *   t_x_max  : [B]             int32,   GPU (有効音素長)
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

/* ------------------------------------------------------------------ */
/*  Forward DP カーネル                                                  */
/*  各ブロック = バッチの1サンプル                                        */
/*  スレッド0のみがDP計算を行う (逐次依存があるため並列化不可)              */
/* ------------------------------------------------------------------ */
__global__ void maximum_path_forward_kernel(
    const float* __restrict__ neg_cent,   // [B, T_y, T_x]
    const int*   __restrict__ t_y_max,    // [B]
    const int*   __restrict__ t_x_max,    // [B]
    float*       __restrict__ value_buf,  // [B, T_y, T_x] 作業バッファ
    const int T_y,
    const int T_x)
{
    const int b     = blockIdx.x;   // バッチインデックス
    const int t_y   = t_y_max[b];
    const int t_x   = t_x_max[b];
    const int offset = b * T_y * T_x;

    // value_buf を neg_cent でコピー初期化（スレッド0が担当）
    if (threadIdx.x == 0) {
        for (int i = 0; i < T_y * T_x; ++i)
            value_buf[offset + i] = neg_cent[offset + i];
    }
    __syncthreads();

    // スレッド0のみでDP（前向き計算）
    if (threadIdx.x == 0) {
        const float MAX_NEG = -1e9f;

        for (int y = 0; y < t_y; ++y) {
            int x_lo = max(0,   t_x + y - t_y);
            int x_hi = min(t_x, y + 1);
            for (int x = x_lo; x < x_hi; ++x) {
                float v_prev, v_cur;

                // v_cur: 同じ x から来る遷移（y-1 → y, x 固定）
                if (x == y) {
                    v_cur = MAX_NEG;
                } else {
                    v_cur = value_buf[offset + (y - 1) * T_x + x];
                }

                // v_prev: x-1 から来る遷移（y-1 → y, x-1 → x）
                if (x == 0) {
                    v_prev = (y == 0) ? 0.0f : MAX_NEG;
                } else {
                    v_prev = value_buf[offset + (y - 1) * T_x + (x - 1)];
                }

                value_buf[offset + y * T_x + x] += fmaxf(v_prev, v_cur);
            }
        }
    }
    __syncthreads();
}

/* ------------------------------------------------------------------ */
/*  Traceback カーネル                                                   */
/*  DP表から最大パスを逆方向に辿り path に 1 を書き込む                   */
/* ------------------------------------------------------------------ */
__global__ void maximum_path_traceback_kernel(
    const float* __restrict__ value_buf,  // [B, T_y, T_x] (forward後)
    const int*   __restrict__ t_y_max,    // [B]
    const int*   __restrict__ t_x_max,    // [B]
    int*         __restrict__ path,       // [B, T_y, T_x]
    const int T_y,
    const int T_x)
{
    const int b      = blockIdx.x;
    const int t_y    = t_y_max[b];
    const int t_x    = t_x_max[b];
    const int offset = b * T_y * T_x;

    if (threadIdx.x == 0) {
        int index = t_x - 1;
        for (int y = t_y - 1; y >= 0; --y) {
            path[offset + y * T_x + index] = 1;
            // index を左に動かすかどうかの判定
            if (index != 0 && (
                index == y ||
                value_buf[offset + (y - 1) * T_x + index] <
                value_buf[offset + (y - 1) * T_x + (index - 1)]))
            {
                index -= 1;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Python / PyTorch から呼ばれるエントリポイント                         */
/* ------------------------------------------------------------------ */
void maximum_path_cuda(
    torch::Tensor& path,       // [B, T_y, T_x] int32, zeros, GPU
    torch::Tensor& neg_cent,   // [B, T_y, T_x] float32, GPU
    torch::Tensor& t_y_max,    // [B] int32, GPU
    torch::Tensor& t_x_max)    // [B] int32, GPU
{
    TORCH_CHECK(neg_cent.is_cuda(),  "neg_cent must be a CUDA tensor");
    TORCH_CHECK(path.is_cuda(),      "path must be a CUDA tensor");
    TORCH_CHECK(neg_cent.dtype() == torch::kFloat32, "neg_cent must be float32");
    TORCH_CHECK(path.dtype()     == torch::kInt32,   "path must be int32");

    const int B   = neg_cent.size(0);
    const int T_y = neg_cent.size(1);
    const int T_x = neg_cent.size(2);

    // DP作業バッファ（neg_centをin-placeで書き換えないようにコピー）
    auto value_buf = neg_cent.clone();

    // Forward DP: バッチ数 = ブロック数, スレッド1本/ブロック
    // （DP自体が逐次依存なので並列化不可。ブロック間は独立）
    maximum_path_forward_kernel<<<B, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
        neg_cent.data_ptr<float>(),
        t_y_max.data_ptr<int>(),
        t_x_max.data_ptr<int>(),
        value_buf.data_ptr<float>(),
        T_y, T_x
    );

    // Traceback: 同じくバッチ数 = ブロック数
    maximum_path_traceback_kernel<<<B, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
        value_buf.data_ptr<float>(),
        t_y_max.data_ptr<int>(),
        t_x_max.data_ptr<int>(),
        path.data_ptr<int>(),
        T_y, T_x
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maximum_path_cuda", &maximum_path_cuda,
          "Monotonic Alignment Search (CUDA kernel implementation)",
          py::arg("path"),
          py::arg("neg_cent"),
          py::arg("t_y_max"),
          py::arg("t_x_max"));
}

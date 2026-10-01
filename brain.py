"""Мозг мухи: LIF-модель всего коннектома FlyWire (v783) на GPU.

Уравнения и параметры один-в-один из Shiu et al. 2024 (brain_model/model.py),
только вместо Brian2 — PyTorch, чтобы можно было крутить мозг шаг за шагом
в замкнутом цикле с виртуальным телом.

Два пути расчёта (математика одна и та же):
  * быстрый (fast=True, по умолчанию на CUDA): своё CUDA-ядро через NVRTC
    (компилируется при запуске, ничего ставить не надо). Один поток = один нейрон,
    он сам крутит до 18 шагов подряд; спайки не умножаются на всю матрицу,
    а «разбрасываются» только по исходящим синапсам спайкнувших нейронов
    (событийная доставка). Пуассоновский вход — счётчиковый ГСЧ Philox прямо в ядре.
  * старый (fast=False или brain.FAST = False): torch-операции + cuSPARSE.
"""
import ctypes
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent
PATH_COMP = ROOT / "brain_model" / "Completeness_783.csv"
PATH_CON = ROOT / "brain_model" / "Connectivity_783.parquet"
PATH_ANN = ROOT / "data" / "neuron_annotations.tsv"

FAST = True   # False -> все новые FlyBrain считаются старым путём (для сверки/отката)

# параметры из статьи (единицы: мс, мВ)
P = dict(
    v_0=-52.0, v_rst=-52.0, v_th=-45.0,
    t_mbr=20.0,   # мембранная постоянная времени
    tau=5.0,      # затухание синаптического тока
    t_rfc=2.2,    # рефрактерный период
    t_dly=1.8,    # синаптическая задержка
    w_syn=0.275,  # вес одного синапса
    f_poi=250,    # сила внешнего (сенсорного) пуассоновского входа
    # --- «противоэпилептика»: механизмы живого мозга, которых нет в исходной модели ---
    a_inc=0.0,    # мВ: насколько каждый спайк повышает адаптационный ток (0 = выкл)
    tau_a=200.0,  # мс: как быстро адаптация проходит
    U=0.0,        # доля синаптического ресурса, расходуемая на спайк (0 = без утомления)
    tau_d=300.0,  # мс: восстановление синапса
    inh_scale=1.0,  # множитель тормозных синапсов
)


def load_annotations():
    """root_id -> тип клетки, сторона, класс и т.п."""
    ann = pd.read_csv(PATH_ANN, sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_sub_class",
                               "cell_type", "hemibrain_type", "side", "top_nt"])
    return ann.set_index("root_id")


# ======================= быстрое ядро (CUDA C++, NVRTC) =======================
# Порядок операций и константы float32 повторяют FlyBrain._dyn побитно
# (компилируем с --fmad=false, чтобы a*b+c не сливалось в FMA).
_CUDA_SRC = r"""
#define PH_M0 0xD2511F53u
#define PH_M1 0xCD9E8D57u
#define PH_W0 0x9E3779B9u
#define PH_W1 0xBB67AE85u

// Philox4x32-10 (Salmon et al. 2011, как в Random123 / curand)
__device__ __forceinline__ uint4 philox4x32_10(uint4 c, uint2 k) {
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        unsigned int hi0 = __umulhi(PH_M0, c.x), lo0 = PH_M0 * c.x;
        unsigned int hi1 = __umulhi(PH_M1, c.z), lo1 = PH_M1 * c.z;
        c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k.x += PH_W0; k.y += PH_W1;
    }
    return c;
}

extern "C" __global__ void philox_test(const unsigned int* in, unsigned int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint4 r = philox4x32_10(make_uint4(in[6*i], in[6*i+1], in[6*i+2], in[6*i+3]),
                            make_uint2(in[6*i+4], in[6*i+5]));
    out[4*i] = r.x; out[4*i+1] = r.y; out[4*i+2] = r.z; out[4*i+3] = r.w;
}

// m (<= D) шагов подряд для каждого нейрона. RING[k] — входящий ток для позиции k
// кольца задержки (уже умноженный на веса); прочитанный слот обнуляем, а спайки
// этого запуска складываем в список -> scatter_spikes положит их в те же слоты
// (они прибудут ровно через D шагов).
extern "C" __global__ void lif_steps(
    float* __restrict__ V, float* __restrict__ G, float* __restrict__ A, float* __restrict__ X,
    int* __restrict__ RFC, const float* __restrict__ RATE, const unsigned char* __restrict__ SIL,
    float* __restrict__ RING, float* __restrict__ ACC,
    int* __restrict__ LIST, float* __restrict__ LVAL, int* __restrict__ NSPK,
    const long long* __restrict__ TDEV, const int* __restrict__ PDEV,
    const float* __restrict__ RND,
    int N, int D, int m, unsigned int key0, unsigned int key1,
    float v0, float vrst, float vth, float c_mbr, float dec_g, float dec_a, float c_d,
    float keep_x, float a_inc, float kick, float p_scale, int rfc_steps, int use_a, int use_u)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = V[i], g = G[i], a = A[i], x = X[i];
    int rfc = RFC[i];
    const float rate = RATE[i];
    const bool driven = rate > 0.f;
    const float thr = rate * p_scale;
    const bool sil = SIL[i] != 0;
    int p = *PDEV;
    const long long t0 = *TDEV;
    float cnt = 0.f;
    for (int s = 0; s < m; ++s) {
        float* slot = RING + (size_t)p * N + i;
        const float syn = *slot;
        if (syn != 0.f) *slot = 0.f;
        g = g + syn;
        // внешний пуассоновский вход бьёт прямо в мембрану
        if (driven) {
            float u;
            if (RND) {
                u = RND[(size_t)s * N + i];
            } else {
                unsigned long long t = (unsigned long long)(t0 + s);
                uint4 r = philox4x32_10(make_uint4((unsigned int)i, (unsigned int)t,
                                                   (unsigned int)(t >> 32), 0x5EEDu),
                                        make_uint2(key0, key1));
                u = (float)(r.x >> 8) * (1.0f / 16777216.0f);
            }
            if (u < thr) v = v + kick;
        }
        if (rfc <= 0) {
            v = v + (((v0 - v) + g) - a) * c_mbr;
            g = g * dec_g;
        }
        if (rfc > -1073741824) rfc -= 1;
        if (use_a) a = a * dec_a;
        if (use_u) x = x + (1.0f - x) * c_d;
        if (v > vth) {
            v = vrst; g = 0.f;
            if (!driven) rfc = rfc_steps;        // у нейронов с внешним входом нет рефрактерности
            cnt += 1.f;
            if (!sil) {
                const float val = use_u ? (driven ? 1.0f : x) : 1.0f;
                const int q = atomicAdd(NSPK, 1);
                LIST[q] = p * N + i;
                LVAL[q] = val;
            }
            if (use_u) x = x * keep_x;
            if (use_a) a = a + a_inc;
        }
        p = (p + 1 == D) ? 0 : p + 1;
    }
    V[i] = v; G[i] = g; A[i] = a; X[i] = x; RFC[i] = rfc;
    if (cnt != 0.f) ACC[i] += cnt;
}

// Событийная доставка: один варп = один спайк, идёт по исходящим синапсам (CSC).
// Последний завершившийся блок обнуляет счётчик спайков и сдвигает время/позицию кольца.
extern "C" __global__ void scatter_spikes(
    const int* __restrict__ CPTR, const int* __restrict__ POST, const float* __restrict__ WV,
    const int* __restrict__ LIST, const float* __restrict__ LVAL,
    int* NSPK, unsigned int* DONE, float* RING, long long* TDEV, int* PDEV,
    int N, int D, int m)
{
    const int n = *((volatile int*)NSPK);
    const int lane = threadIdx.x & 31;
    const int nw = (gridDim.x * blockDim.x) >> 5;
    for (int s = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; s < n; s += nw) {
        const int code = LIST[s];
        const float val = LVAL[s];
        const int k = code / N;
        const int j = code - k * N;
        float* dst = RING + (size_t)k * N;
        const int e1 = CPTR[j + 1];
        for (int e = CPTR[j] + lane; e < e1; e += 32)
            atomicAdd(dst + POST[e], WV[e] * val);
    }
    __shared__ int last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned int prev = atomicAdd(DONE, 1u);
        last = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (last && threadIdx.x == 0) {
        *NSPK = 0; *DONE = 0u;
        *TDEV += m;
        int p = *PDEV + m;
        if (p >= D) p -= D;
        *PDEV = p;
    }
}
"""

_KERNELS = {}      # (device index) -> dict name -> CUfunction
_LIBCUDA = None


def _nvrtc_ptx(src, arch):
    """CUDA C++ -> PTX через NVRTC из комплекта torch (без CUDA Toolkit и компилятора)."""
    major = int(torch.version.cuda.split(".")[0])
    lib = ctypes.CDLL(f"nvrtc64_{major}0_0.dll" if sys.platform == "win32" else f"libnvrtc.so.{major}")
    prog = ctypes.c_void_p()

    def ok(r, what):
        if r != 0:
            raise RuntimeError(f"NVRTC {what}: код {r}")
    ok(lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"brain.cu", 0, None, None), "create")
    opts = [f"--gpu-architecture=compute_{arch}".encode(), b"--fmad=false", b"-std=c++17"]
    res = lib.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
    if res != 0:
        n = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value)
        lib.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("NVRTC:\n" + log.value.decode(errors="replace"))
    n = ctypes.c_size_t()
    ok(lib.nvrtcGetPTXSize(prog, ctypes.byref(n)), "ptxsize")
    ptx = ctypes.create_string_buffer(n.value)
    ok(lib.nvrtcGetPTX(prog, ptx), "ptx")
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.value


def _get_kernels(dev):
    """Скомпилировать ядра NVRTC (один раз на процесс и устройство)."""
    global _LIBCUDA
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    if key in _KERNELS:
        return _KERNELS[key]
    from torch.cuda._utils import _get_gpu_runtime_library
    _LIBCUDA = _get_gpu_runtime_library()
    with torch.cuda.device(key):
        torch.zeros(1, device=f"cuda:{key}")               # контекст CUDA уже создан torch'ем
        pr = torch.cuda.get_device_properties(key)
        ptx = _nvrtc_ptx(_CUDA_SRC, f"{pr.major}{pr.minor}")
        mod = ctypes.c_void_p()
        r = _LIBCUDA.cuModuleLoadData(ctypes.byref(mod), ptx)
        if r != 0:
            raise RuntimeError(f"cuModuleLoadData: {r}")
        fn = {}
        for name in ("lif_steps", "scatter_spikes", "philox_test"):
            f = ctypes.c_void_p()
            r = _LIBCUDA.cuModuleGetFunction(ctypes.byref(f), mod, name.encode())
            if r != 0:
                raise RuntimeError(f"cuModuleGetFunction {name}: {r}")
            fn[name] = f
    _KERNELS[key] = fn
    return fn


def _pack(args):
    """Список аргументов ядра -> (массив void* для cuLaunchKernel, сами ctypes-объекты).
    torch-тензор -> указатель, None -> NULL, ctypes-значение как есть."""
    c = []
    for x in args:
        if isinstance(x, torch.Tensor):
            c.append(ctypes.c_void_p(x.data_ptr()))
        elif x is None:
            c.append(ctypes.c_void_p(0))
        else:
            c.append(x)
    return (ctypes.c_void_p * len(c))(*[ctypes.addressof(x) for x in c]), c


def _launch_raw(func, grid, block, arr, stream):
    res = _LIBCUDA.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, stream, arr, None)
    if res != 0:
        raise RuntimeError(f"cuLaunchKernel: ошибка CUDA {res}")


def _launch(func, grid, block, args):
    """cuLaunchKernel на текущем torch-потоке (работает и внутри захвата CUDA Graph).
    Параметры копируются драйвером в момент запуска/захвата."""
    arr, _keep = _pack(args)
    _launch_raw(func, grid, block, arr, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))


class FlyBrain:
    def __init__(self, dt=0.1, device=None, seed=0, drop_kc_kc=False, params=None, fast=None):
        self.P = dict(P, **(params or {}))
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dt = dt
        self.seed = int(seed)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

        comp = pd.read_csv(PATH_COMP, index_col=0)
        con = pd.read_parquet(PATH_CON)
        if drop_kc_kc:
            # Аксо-аксональные синапсы между клетками Кеньона в точечной модели
            # дают самоподдерживающуюся «эпилепсию» от любого запаха -> убираем.
            ann = load_annotations()
            kc = set(ann.index[ann.cell_class == "Kenyon_Cell"])
            con = con[~(con.Presynaptic_ID.isin(kc) & con.Postsynaptic_ID.isin(kc))]
        self.ids = comp.index.values                     # flywire root_id по индексу
        self.idx = {r: i for i, r in enumerate(self.ids)}
        self.N = n = len(self.ids)

        # W[post, pre] = знак * число синапсов * w_syn
        pre = torch.tensor(con["Presynaptic_Index"].values, dtype=torch.long)
        post = torch.tensor(con["Postsynaptic_Index"].values, dtype=torch.long)
        w = torch.tensor(con["Excitatory x Connectivity"].values * self.P["w_syn"], dtype=torch.float32)
        w[w < 0] *= self.P["inh_scale"]          # сила торможения (1 = как в оригинале)
        Wc = torch.sparse_coo_tensor(torch.stack([post, pre]), w, (n, n)).coalesce()
        self.W = Wc.to_sparse_csr().to(self.dev)
        self.n_syn = int(np.abs(con["Excitatory x Connectivity"].values).sum())

        self.delay_steps = int(round(P["t_dly"] / dt))
        self.rfc_steps = int(round(P["t_rfc"] / dt))
        self.silenced = torch.zeros(n, dtype=torch.bool, device=self.dev)

        self.fast = bool(FAST if fast is None else fast) and self.dev.type == "cuda"
        if self.fast:
            try:
                self._fast_init(Wc)
            except Exception as e:                # нет NVRTC/драйвера -> старый путь
                warnings.warn(f"быстрое ядро мозга не собралось, считаю старым путём: {e}")
                self.fast = False
        del Wc
        self.graph = None
        self.reset()

    # ---------- быстрый путь: подготовка ----------
    def _fast_init(self, Wc):
        n, d = self.N, self.dev
        self._k = _get_kernels(d)
        # исходящие синапсы по пресинаптическому нейрону (CSC): cptr[j]..cptr[j+1]
        post, pre = Wc.indices()
        order = torch.argsort(pre * n + post)
        pre_s = pre[order]
        self._post = post[order].to(torch.int32).to(d)
        self._wv = Wc.values()[order].contiguous().to(d)
        cnt = torch.bincount(pre_s, minlength=n)
        cptr = torch.zeros(n + 1, dtype=torch.long)
        cptr[1:] = torch.cumsum(cnt, 0)
        assert int(cptr[-1]) < 2**31
        self._cptr = cptr.to(torch.int32).to(d)
        D = self.delay_steps
        self._ring = torch.zeros((D, n), device=d)                  # входящий ток по слотам задержки
        self._acc = torch.zeros(n, device=d)                        # счётчик спайков для run()
        self._acc1 = torch.zeros(n, device=d)                       # ... и для step()
        self._list = torch.zeros(n * D, dtype=torch.int32, device=d)  # спайки запуска: k*N + j
        self._lval = torch.zeros(n * D, device=d)                   # и их «сила» (ресурс синапса x)
        self._nspk = torch.zeros(1, dtype=torch.int32, device=d)
        self._done = torch.zeros(1, dtype=torch.int32, device=d)
        self._tdev = torch.zeros(1, dtype=torch.int64, device=d)    # глобальный номер шага (для ГСЧ)
        self._pdev = torch.zeros(1, dtype=torch.int32, device=d)    # позиция в кольце задержки
        sms = torch.cuda.get_device_properties(d).multi_processor_count
        self._sc_grid = sms * 8
        self._set_consts()

    def _set_consts(self):
        """Константы float32 так же, как их видит torch в _dyn (скаляр -> float32)."""
        Pp, dt = self.P, self.dt
        f = lambda z: ctypes.c_float(float(np.float32(z)))
        self._cargs = [
            f(Pp["v_0"]), f(Pp["v_rst"]), f(Pp["v_th"]), f(dt / Pp["t_mbr"]),
            f(np.exp(-dt / Pp["tau"])), f(np.exp(-dt / Pp["tau_a"])), f(dt / Pp["tau_d"]),
            f(1 - Pp["U"]), f(Pp["a_inc"]), f(Pp["w_syn"] * Pp["f_poi"]), f(dt * 1e-3),
            ctypes.c_int(self.rfc_steps), ctypes.c_int(int(Pp["a_inc"] > 0)), ctypes.c_int(int(Pp["U"] > 0)),
        ]
        self._P_used = dict(Pp)
        self._args_key = None          # аргументы ядер пересоберутся
        self.graph = None              # граф запёк старые константы -> без графа до нового enable_graph()

    def _check_consts(self):
        """b.P поменяли после создания -> пересчитать константы ядра (как старый путь,
        который читает P на каждом шаге). w_syn/inh_scale в весах W не меняются — как раньше."""
        if self.P != self._P_used:
            self._set_consts()

    def _build_args(self):
        """Массивы аргументов ядер собираем один раз: тензоры состояния не пересоздаются
        (reset() всё обнуляет на месте); меняются только m, acc и rnd."""
        n, D = self.N, self.delay_steps
        self._m_c, self._acc_c, self._rnd_c = ctypes.c_int(D), ctypes.c_void_p(0), ctypes.c_void_p(0)
        self._lif_arr, self._lif_keep = _pack([
            self.v, self.g, self.a, self.x, self.rfc, self.rate_in, self.silenced,
            self._ring, self._acc_c, self._list, self._lval, self._nspk, self._tdev, self._pdev, self._rnd_c,
            ctypes.c_int(n), ctypes.c_int(D), self._m_c,
            ctypes.c_uint32(self.seed & 0xFFFFFFFF), ctypes.c_uint32((self.seed >> 32) & 0xFFFFFFFF),
            *self._cargs])
        self._sc_arr, self._sc_keep = _pack([
            self._cptr, self._post, self._wv, self._list, self._lval, self._nspk, self._done,
            self._ring, self._tdev, self._pdev, ctypes.c_int(n), ctypes.c_int(D), self._m_c])
        self._args_key = (self.v.data_ptr(), self.rate_in.data_ptr(), self.silenced.data_ptr())

    def _fast_launch(self, m, acc, rnd=None):
        """m (1..D) шагов: ядро нейронов + доставка спайков. Всё на GPU, без синхронизации;
        позиция кольца и номер шага живут на устройстве -> годится для CUDA Graph.
        rnd — готовые равномерные числа (m, N) вместо встроенного Philox (для сверки)."""
        n, D = self.N, self.delay_steps
        assert 1 <= m <= D
        if self._args_key != (self.v.data_ptr(), self.rate_in.data_ptr(), self.silenced.data_ptr()):
            self._build_args()
        if rnd is not None:
            assert rnd.is_contiguous() and rnd.dtype == torch.float32 and rnd.numel() >= m * n
        self._m_c.value = m
        self._acc_c.value = acc.data_ptr()
        self._rnd_c.value = None if rnd is None else rnd.data_ptr()
        st = ctypes.c_void_p(torch.cuda.current_stream(self.dev).cuda_stream)
        _launch_raw(self._k["lif_steps"], (n + 255) // 256, 256, self._lif_arr, st)
        _launch_raw(self._k["scatter_spikes"], self._sc_grid, 256, self._sc_arr, st)

    def _advance(self, m):
        self.ptr = (self.ptr + m) % self.delay_steps
        self.t += self.dt * m

    # ---------- состояние ----------
    def reset(self):
        n, d = self.N, self.dev
        if getattr(self, "v", None) is None:
            self.v = torch.empty(n, device=d)
            self.g = torch.empty(n, device=d)
            self.rfc = torch.empty(n, dtype=torch.int32, device=d)
            self.buf = torch.empty((self.delay_steps, n), device=d)    # спайки «в пути» по аксонам
            self.rate_in = torch.empty(n, device=d)                    # внешний вход, Гц
            self.a = torch.empty(n, device=d)                          # адаптация (мВ)
            self.x = torch.empty(n, device=d)                          # ресурс синапсов (0..1)
        # всё обнуляем НА МЕСТЕ — CUDA Graph держит адреса этих тензоров
        self.v.fill_(self.P["v_0"]); self.g.zero_(); self.rfc.zero_(); self.buf.zero_()
        self.rate_in.zero_(); self.a.zero_(); self.x.fill_(1.0)
        if getattr(self, "_ring", None) is not None:
            # номер шага _tdev не сбрасываем: ГСЧ идёт дальше, как torch.Generator
            self._ring.zero_(); self._nspk.zero_(); self._done.zero_(); self._pdev.zero_()
        self.ptr = 0
        self.t = 0.0

    def set_input(self, neuron_idx, rate_hz):
        """Задать частоту сенсорного входа (Гц) группе нейронов."""
        self.rate_in[neuron_idx] = float(rate_hz)

    def silence(self, neuron_idx):
        """«Выключить» нейроны: их спайки больше никуда не доходят."""
        self.silenced[neuron_idx] = True

    # ---------- динамика ----------
    @torch.no_grad()
    def step(self):
        """Один шаг 0.1 мс: спайки, отправленные delay мс назад, прибывают на синапсы."""
        if self.fast:
            self._check_consts()
            c = self._acc1
            c.zero_()
            self._fast_launch(1, c)
            self._advance(1)
            return c > 0
        return self._dyn(torch.mv(self.W, self.buf[self.ptr]))

    @torch.no_grad()
    def run_block(self):
        """Сразу delay_steps шагов (18 = 1.8 мс). Всё, что прилетит за эти шаги, уже
        отправлено раньше -> считаем одним умножением матрицы на 18 векторов.
        Математически то же самое, что 18 вызовов step(), но матрица читается 1 раз."""
        if self.fast:
            self._check_consts()
            c = torch.zeros(self.N, device=self.dev)
            self._fast_launch(self.delay_steps, c)
            self._advance(self.delay_steps)
            return c
        assert self.ptr == 0
        arr = torch.sparse.mm(self.W, self.buf.T.contiguous()).T.contiguous()   # (D, N)
        cnt = torch.zeros(self.N, device=self.dev)
        for k in range(self.delay_steps):
            cnt += self._dyn(arr[k])
        return cnt

    @torch.no_grad()
    def _dyn(self, syn_in, k=None, rnd=None):
        """Динамика одного шага (старый путь). Всё «на месте» (in-place), чтобы шаги можно было
        записать в CUDA Graph. k — позиция в буфере (для графа), иначе self.ptr.
        rnd — готовые равномерные числа (N,) вместо torch.rand (для побитной сверки)."""
        dt, P = self.dt, self.P
        v, g, a, x, rfc = self.v, self.g, self.a, self.x, self.rfc
        g.add_(syn_in)

        # внешний пуассоновский вход бьёт прямо в мембрану (как в оригинале)
        driven = self.rate_in > 0
        if rnd is None:
            rnd = torch.rand(self.N, generator=self.gen, device=self.dev)
        poi = (rnd < self.rate_in * (dt * 1e-3)) & driven
        v.add_(poi * (P["w_syn"] * P["f_poi"]))

        active = rfc <= 0
        v.copy_(torch.where(active, v + (P["v_0"] - v + g - a) * (dt / P["t_mbr"]), v))
        g.copy_(torch.where(active, g * np.exp(-dt / P["tau"]), g))
        rfc.sub_(1)
        if P["a_inc"] > 0:
            a.mul_(np.exp(-dt / P["tau_a"]))
        if P["U"] > 0:
            x.add_((1 - x) * (dt / P["tau_d"]))

        spk = v > P["v_th"]
        v.masked_fill_(spk, P["v_rst"])
        g.masked_fill_(spk, 0.0)
        # у нейронов с внешним входом нет рефрактерности (как в оригинале)
        rfc.copy_(torch.where(spk & ~driven, torch.full_like(rfc, self.rfc_steps), rfc))

        out = (spk & ~self.silenced).float()
        if P["U"] > 0:
            # у первичных сенсорных входов (driven) утомление не применяем:
            # их частота уже задаёт эффективную силу стимула
            out = out * torch.where(driven, torch.ones_like(x), x)
            x.copy_(torch.where(spk, x * (1 - P["U"]), x))
        if P["a_inc"] > 0:
            a.add_(spk * P["a_inc"])
        if k is None:
            self.buf[self.ptr].copy_(out)
            self.ptr = (self.ptr + 1) % self.delay_steps
            self.t += dt
        else:
            self.buf[k].copy_(out)
        return spk

    # ---------- CUDA Graph: блок из 18 шагов одним запуском ----------
    def enable_graph(self):
        """Записать блок (18 шагов) в CUDA Graph. Быстрый путь: 2 ядра на блок;
        старый: умножение матрицы + 18 шагов по ~25 мелких ядер."""
        if self.fast:
            self._check_consts()
            # запуск ядер при захвате ничего не считает -> состояние не трогаем
            torch.cuda.synchronize(self.dev)
            s = torch.cuda.Stream(self.dev)
            s.wait_stream(torch.cuda.current_stream(self.dev))
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=s):
                self._fast_launch(self.delay_steps, self._acc)
            torch.cuda.current_stream(self.dev).wait_stream(s)
            self._graph_fast = True
            return self

        D = self.delay_steps
        self._arr = torch.zeros((D, self.N), device=self.dev)
        self._cnt = torch.zeros(self.N, device=self.dev)

        def block():
            self._arr.copy_(torch.sparse.mm(self.W, self.buf.T.contiguous()).T)
            self._cnt.zero_()
            for k in range(D):
                self._cnt.add_(self._dyn(self._arr[k], k=k).float())

        saved = [t.clone() for t in (self.v, self.g, self.a, self.x, self.rfc, self.buf)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                block()                                   # прогрев
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        self.graph.register_generator_state(self.gen)
        with torch.cuda.graph(self.graph):
            block()
        for t, sv in zip((self.v, self.g, self.a, self.x, self.rfc, self.buf), saved):
            t.copy_(sv)                                   # вернуть состояние как было
        self._graph_fast = False
        return self

    def _graph_ok(self):
        return self.graph is not None and getattr(self, "_graph_fast", False) == self.fast

    @torch.no_grad()
    def run_block_graph(self):
        if self.fast:
            self._check_consts()
            self._acc.zero_()
            if self._graph_ok():
                self.graph.replay()
            else:
                self._fast_launch(self.delay_steps, self._acc)
            self._advance(self.delay_steps)
            return self._acc
        assert self.ptr == 0
        self.graph.replay()
        self.t += self.dt * self.delay_steps
        return self._cnt

    @torch.no_grad()
    def run(self, ms, record=None, tensor=False):
        """Прогнать мозг ms миллисекунд. Возвращает число спайков на нейрон
        (или только для индексов record; tensor=True -> torch-тензор на GPU)."""
        steps = max(0, int(round(ms / self.dt)))     # ms <= 0 -> ничего (divmod(-1, 18) дал бы 17 шагов)
        D = self.delay_steps
        if self.fast:
            self._check_consts()
            # блоки по D шагов с любой позиции кольца + хвост одним запуском
            acc = self._acc
            acc.zero_()
            nblk, rem = divmod(steps, D)
            g = self.graph if self._graph_ok() else None
            for _ in range(nblk):
                if g is not None:
                    g.replay()
                else:
                    self._fast_launch(D, acc)
            if rem:
                self._fast_launch(rem, acc)
            self._advance(steps)
            counts = acc.clone()
        else:
            counts = torch.zeros(self.N, device=self.dev)
            while steps > 0:
                if self.ptr == 0 and steps >= D:
                    counts += self.run_block_graph() if self._graph_ok() else self.run_block()
                    steps -= D
                else:
                    counts += self.step(); steps -= 1
        if tensor:
            return counts
        if record is not None:
            return counts[record].cpu().numpy()
        return counts.cpu().numpy()
